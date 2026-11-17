#!/usr/bin/env python3
"""Task-A -> Task-B grokking with FUNCTION-PRESERVING ReLU-MLP rescaling.

This is a separate training entry point. It imports the existing grokking/train.py
model, dataset construction and checkpoint helpers; it does NOT modify train.py.

Task A: (a+b) % 113, input [a,b,+].
Task B (default): (a+b+1) % 113, input [a,b,-] ('-' is a TASK ID).
At an A checkpoint, apply mlp_in.weight/bias *= c, mlp_out.weight /= c.
This leaves logits unchanged at the switch (up to floating-point precision).
Train only Task B, with freshly reset AdamW moments for ALL conditions.
Optionally rewarm learning rate from 1e-3 to 3e-3 over 200 B updates and
cosine-decay back to 1e-3 by B update 2000.

Example (from repo root):
  python grokking/rescale_transfer.py \\
      --checkpoint grokking/runs/a_nanda3_seed0/checkpoints/step_001000.pt \\
      --run-dir grokking/runs/rescale_shift1/control --mlp-scale 1 \\
      --steps 20000 --wandb-mode disabled --device cpu

For resumption, pass --resume and the same --run-dir. The original scaling is
NOT applied a second time. --steps is the TOTAL count of B updates.

Requires torch, existing grokking/train.py, and wandb unless mode=disabled.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# With `python grokking/rescale_transfer.py`, the script's directory is on
# sys.path, so this resolves to the existing grokking/train.py.
from train import (  # type: ignore
    OneLayerTransformer,
    load_checkpoint,
    make_data,
    optimizer_step_with_frozen_token,
    put_rng_state,
    save_checkpoint,
)


PHASE = "b_rescale"
B_OPERATIONS = ("sum_plus_one", "subtract", "two_a_plus_b")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Original Phase-A checkpoint (start new branch)")
    source.add_argument("--resume", type=Path, help="Resume checkpoint saved by THIS script")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--b-operation", choices=B_OPERATIONS, default=None,
                    help="Default: sum_plus_one; '+' and '-' are distinct task IDs")
    ap.add_argument("--mlp-scale", type=float, default=None,
                    help="Scale mlp_in weight+bias by c, mlp_out weight by 1/c (default 1)")
    ap.add_argument("--rewarm", action="store_true", help="Use temporary LR rewarming during B")
    ap.add_argument("--peak-lr", type=float, default=None, help="LR at warmup end (default 0.003)")
    ap.add_argument("--warmup-steps", type=int, default=None, help="Peak at B step 200 by default")
    ap.add_argument("--rewarm-end", type=int, default=None,
                    help="Cosine decay returns to original LR by B step 2000")
    ap.add_argument("--steps", type=int, default=20000, help="TOTAL Task-B optimizer updates")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--wandb-project", default="grokking-continual")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    ap.add_argument("--wandb-name", default=None)
    return ap.parse_args()


@torch.no_grad()
def rescale_mlp_preserving_function(model, c):
    """Valid for the current ReLU MLP: W2 ReLU(W1 x + b1) + b2."""
    if not math.isfinite(c) or c <= 0:
        raise ValueError("--mlp-scale must be finite and strictly positive")
    model.mlp_in.weight.mul_(c)
    model.mlp_in.bias.mul_(c)
    model.mlp_out.weight.div_(c)
    # Do NOT rescale mlp_out.bias: it is added after the linear map.


@torch.no_grad()
def norm_stats(model):
    return {
        "norm/mlp_in": model.mlp_in.weight.norm().item(),
        "norm/mlp_out": model.mlp_out.weight.norm().item(),
        "norm/mlp_in_bias": model.mlp_in.bias.norm().item(),
        "norm/token": model.token.weight.norm().item(),
        "norm/qkv": model.qkv.weight.norm().item(),
        "norm/unembed": model.unembed.weight.norm().item(),
    }


def scheduled_lr(step, cfg):
    base = cfg["base_lr"]
    if not cfg["rewarm"] or step > cfg["rewarm_end"]:
        return base
    peak = cfg["peak_lr"]
    warmup = cfg["warmup_steps"]
    if step <= warmup:
        return base + (peak - base) * step / warmup
    progress = (step - warmup) / (cfg["rewarm_end"] - warmup)
    return base + (peak - base) * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_both(model, data):
    """Report BOTH tasks even though Task-A labels never enter training."""
    model.eval()
    metrics = {}
    for split in ("train", "test"):
        a, b = data["A"], data["B"]
        xa, ya = a[f"{split}_x"], a[f"{split}_y"]
        xb, yb = b[f"{split}_x"], b[f"{split}_y"]
        logits_a, logits_b = model(xa), model(xb)
        pred_a, pred_b = logits_a.argmax(-1), logits_b.argmax(-1)
        for tag, logits, prediction, truth in (
            ("A", logits_a, pred_a, ya),
            ("B", logits_b, pred_b, yb),
        ):
            metrics[f"{tag}/{split}_loss"] = F.cross_entropy(logits, truth).item()
            metrics[f"{tag}/{split}_accuracy"] = (prediction == truth).float().mean().item()
        metrics[f"diagnostics/{split}_op_agreement"] = (pred_a == pred_b).float().mean().item()
        # Fraction of A-token predictions equal to *B's* correct answer.
        # For shifted addition, this tests whether the model outputs sum+1 even
        # on the '+' token rather than recovering the addition function.
        metrics[f"diagnostics/{split}_A_token_predicts_B_label"] = (
            pred_a == yb
        ).float().mean().item()
    model.train()
    return metrics


def check_function_preservation(model, c, data):
    """Compare pre/post-rescale logits on paired A,B inputs, with no A training."""
    model.eval()
    with torch.no_grad():
        sample = torch.cat((data["A"]["test_x"][:64], data["B"]["test_x"][:64]))
        before = model(sample).clone()
        rescale_mlp_preserving_function(model, c)
        after = model(sample)
        max_diff = (after - before).abs().max().item()
        is_close = torch.allclose(after, before, atol=1e-3, rtol=1e-4)
    model.train()
    if not is_close:
        raise RuntimeError(
            f"Function-preserving MLP rescaling failed numerical validation "
            f"(scale={c}, max |logit difference|={max_diff:.6g})."
        )
    return max_diff



@torch.no_grad()
def fixed_probe_preactivations(model, data):
    """MLP pre-ReLU activations on 64 fixed held-out pairs for each task.

    This is an EVALUATION-ONLY diagnostic; the probe inputs have no labels.
    Only the final (operator) position is used, where the prediction is made.
    """
    outputs = []

    def capture(_module, _inputs, value):
        outputs.append(value[:, -1, :].detach().clone())

    hook = model.mlp_in.register_forward_hook(capture)
    was_training = model.training
    model.eval()
    try:
        for task in ("A", "B"):
            model(data[task]["test_x"][:64])
    finally:
        hook.remove()
        model.train(was_training)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def feature_stats(model, data, initial_probe):
    current = fixed_probe_preactivations(model, data)
    baseline = initial_probe.to(device=current.device)
    return {
        "features/relative_preactivation_drift": (
            (current - baseline).norm() / baseline.norm().clamp_min(1e-30)
        ).item(),
        "features/relu_gate_agreement_initial": (
            (current > 0) == (baseline > 0)
        ).float().mean().item(),
        "features/relu_active_fraction": (current > 0).float().mean().item(),
    }


def make_config(args, parent):
    cfg = dict(parent["config"])
    if cfg.get("input_format") != "a,b,operator" or cfg.get("model_version") != 2:
        raise ValueError("Checkpoint is not from the 3-token model_version=2 baseline")
    cfg.update({
        "experiment": "task_b_function_preserving_mlp_rescale_and_lr_rewarm",
        "b_operation": args.b_operation or "sum_plus_one",
        "mlp_scale": 1.0 if args.mlp_scale is None else args.mlp_scale,
        "rewarm": bool(args.rewarm),
        "peak_lr": 0.003 if args.peak_lr is None else args.peak_lr,
        "warmup_steps": 200 if args.warmup_steps is None else args.warmup_steps,
        "rewarm_end": 2000 if args.rewarm_end is None else args.rewarm_end,
        "base_lr": float(cfg["lr"]),
        "optimizer_reset_at_switch": True,
        "training_task": "B_only_no_A_replay",
    })
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg.get("experiment") != "task_b_function_preserving_mlp_rescale_and_lr_rewarm":
        raise ValueError("Not a rescale-transfer experiment checkpoint")
    if cfg["b_operation"] not in B_OPERATIONS:
        raise ValueError("Unknown B operation")
    if not math.isfinite(cfg["mlp_scale"]) or cfg["mlp_scale"] <= 0:
        raise ValueError("mlp_scale must be finite and > 0")
    if cfg["base_lr"] <= 0 or cfg["peak_lr"] <= 0:
        raise ValueError("Learning rates must be strictly positive")
    if not 0 < cfg["warmup_steps"] < cfg["rewarm_end"]:
        raise ValueError("Require 0 < warmup-steps < rewarm-end")
    if cfg["rewarm"] and cfg["peak_lr"] < cfg["base_lr"]:
        raise ValueError("peak_lr must be >= base_lr when rewarming")


def append_csv(path, step, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"step": step, **metrics}
    # Ordering is fixed from run to run, including across resumption.
    fieldnames = ["step"] + sorted(metrics)
    first = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if first:
            writer.writeheader()
        writer.writerow(payload)


def main():
    args = parse_args()
    if min(args.steps, args.eval_every, args.checkpoint_every) <= 0:
        raise ValueError("steps/eval-every/checkpoint-every must be positive")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if args.resume:
        if not args.run_dir.is_dir():
            raise FileNotFoundError(f"Resume requires existing run-dir {args.run_dir}")
        ckpt = load_checkpoint(args.resume)
        if ckpt["phase"] != PHASE:
            raise ValueError("The resume checkpoint was not created by rescale_transfer.py")
        cfg = dict(ckpt["config"])
        validate_config(cfg)
        # Avoid inadvertently changing the scientific condition on resume.
        if args.b_operation is not None and args.b_operation != cfg["b_operation"]:
            raise ValueError("B operation mismatch with resume checkpoint")
        if args.mlp_scale is not None and args.mlp_scale != cfg["mlp_scale"]:
            raise ValueError("MLP scale mismatch with resume checkpoint")
        if args.rewarm and not cfg["rewarm"]:
            raise ValueError("Rewarming mismatch with resume checkpoint")
        if args.peak_lr is not None and args.peak_lr != cfg["peak_lr"]:
            raise ValueError("Peak LR mismatch with resume checkpoint")
        if args.warmup_steps is not None and args.warmup_steps != cfg["warmup_steps"]:
            raise ValueError("Warmup mismatch with resume checkpoint")
        if args.rewarm_end is not None and args.rewarm_end != cfg["rewarm_end"]:
            raise ValueError("Rewarm-end mismatch with resume checkpoint")
        start_step = int(ckpt["step"])
        origin = ckpt["origin"]
        wandb_id = ckpt.get("wandb_id")
    else:
        ckpt = load_checkpoint(args.checkpoint)
        if ckpt["phase"] != "a":
            raise ValueError("New branch must originate from Phase-A checkpoint")
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite nonempty run-dir {args.run_dir}")
        cfg = make_config(args, ckpt)
        start_step = 0
        origin = {"checkpoint": str(args.checkpoint.resolve()), "parent_step": int(ckpt["step"])}
        wandb_id = None

    if args.steps < start_step:
        raise ValueError(f"--steps={args.steps} precedes checkpoint step={start_step}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    model = OneLayerTransformer(cfg["p"], cfg["d_model"], cfg["heads"], cfg["d_mlp"]).to(device)
    model.load_state_dict(ckpt["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["base_lr"],
                                  betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    if args.resume:
        optimizer.load_state_dict(ckpt["optimizer"])
        put_rng_state(ckpt["rng"])
    # On a new branch we deliberately DO NOT load the A optimizer state.

    data, n_train, n_test = make_data(
        cfg["p"], cfg["fraction"], cfg["split_seed"], device,
        b_operation=cfg["b_operation"],
    )
    if not args.resume:
        max_logit_diff = check_function_preservation(model, cfg["mlp_scale"], data)
        cfg["initial_rescale_max_logit_diff"] = max_logit_diff
        (args.run_dir / "config.json").write_text(
            json.dumps({**cfg, "phase": PHASE, "origin": origin}, indent=2) + "\n"
        )
    else:
        max_logit_diff = cfg.get("initial_rescale_max_logit_diff", 0.0)

    probe_path = args.run_dir / "initial_mlp_probe.pt"
    if args.resume:
        if not probe_path.is_file():
            raise FileNotFoundError(f"Missing initial representation probe {probe_path}")
        initial_probe = torch.load(probe_path, map_location=device, weights_only=True)
    else:
        initial_probe = fixed_probe_preactivations(model, data)
        torch.save(initial_probe.cpu(), probe_path)

    print(f"Rescale B-only: {cfg['b_operation']}, c={cfg['mlp_scale']}, "
          f"rewarm={cfg['rewarm']}, optimizer moments RESET at A->B switch", flush=True)
    print(f"A checkpoint: {origin}; samples/task train={n_train}, test={n_test}; "
          f"resume at B step {start_step}, end={args.steps}; device={device}", flush=True)
    print(f"Max |logit difference| immediately after rescale: {max_logit_diff:.3g}", flush=True)

    wandb = None
    if args.wandb_mode != "disabled":
        import wandb as wandb_module
        wandb = wandb_module
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
            name=args.wandb_name or args.run_dir.name,
            id=wandb_id if args.resume else None,
            resume="allow" if args.resume else None,
            dir=str(args.run_dir),
            config={**cfg, "phase": PHASE, "origin": origin,
                    "train_size": n_train, "test_size": n_test,
                    "max_steps_requested": args.steps,
                    "eval_every": args.eval_every, "checkpoint_every": args.checkpoint_every},
        )
        wandb_id = run.id

    metrics = evaluate_both(model, data)
    if not args.resume:
        initial_metrics = {
            **metrics, **norm_stats(model), **feature_stats(model, data, initial_probe),
            "training/lr": cfg["base_lr"],
            "training/batch_loss": float("nan"),
            "update/relative_mlp_in": float("nan"),
            "update/relative_mlp_out": float("nan"),
            "diagnostics/prepost_rescale_max_logit_diff": max_logit_diff,
        }
        if wandb:
            wandb.log(initial_metrics, step=0)
        append_csv(args.run_dir / "metrics.csv", 0, initial_metrics)
        save_checkpoint(args.run_dir, 0, model, optimizer, cfg, PHASE, origin,
                        wandb_id, initial_metrics)
        print(f"Initial B train/test accuracy: {metrics['B/train_accuracy']:.4f}/"
              f"{metrics['B/test_accuracy']:.4f}; A train/test: "
              f"{metrics['A/train_accuracy']:.4f}/{metrics['A/test_accuracy']:.4f}", flush=True)

    try:
        last_log_time = time.monotonic()
        frozen_A_token = cfg["p"]
        for step in range(start_step + 1, args.steps + 1):
            lr = scheduled_lr(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            report = step == 1 or step % args.eval_every == 0 or step == args.steps
            if report:
                before_in = model.mlp_in.weight.detach().clone()
                before_out = model.mlp_out.weight.detach().clone()

            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = data["B"]
            logits = model(batch["train_x"])
            loss = F.cross_entropy(logits, batch["train_y"])
            loss.backward()
            optimizer_step_with_frozen_token(model, optimizer, frozen_A_token)

            should_log = step % args.eval_every == 0 or step == args.steps
            should_save = step % args.checkpoint_every == 0 or step == args.steps
            if should_log or should_save:
                metrics = evaluate_both(model, data)
                if report:
                    relative_in = ((model.mlp_in.weight - before_in).norm() /
                                   before_in.norm().clamp_min(1e-30)).item()
                    relative_out = ((model.mlp_out.weight - before_out).norm() /
                                    before_out.norm().clamp_min(1e-30)).item()
                if should_log:
                    all_metrics = {
                        **metrics, **norm_stats(model), **feature_stats(model, data, initial_probe),
                        "training/lr": lr,
                        "training/batch_loss": float(loss.item()),
                        "update/relative_mlp_in": relative_in,
                        "update/relative_mlp_out": relative_out,
                        "diagnostics/prepost_rescale_max_logit_diff": max_logit_diff,
                    }
                    if wandb:
                        wandb.log(all_metrics, step=step)
                    append_csv(args.run_dir / "metrics.csv", step, all_metrics)
                    print(f"B step {step:6d} LR={lr:.5g} "
                          f"A train/test={metrics['A/train_accuracy']:.3f}/{metrics['A/test_accuracy']:.3f} "
                          f"B train/test={metrics['B/train_accuracy']:.3f}/{metrics['B/test_accuracy']:.3f} "
                          f"agreement={metrics['diagnostics/test_op_agreement']:.3f} "
                          f"elapsed={time.monotonic()-last_log_time:.1f}s", flush=True)
                    last_log_time = time.monotonic()
                if should_save:
                    # Use a stable metric dict even when the checkpoint step is not a log step.
                    checkpoint_metrics = all_metrics if should_log else metrics
                    path = save_checkpoint(args.run_dir, step, model, optimizer, cfg,
                                           PHASE, origin, wandb_id, checkpoint_metrics)
                    print(f"Checkpoint: {path}", flush=True)
    finally:
        if wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
