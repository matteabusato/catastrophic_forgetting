#!/usr/bin/env python3
"""One-layer modular-arithmetic grokking; manual, checkpointed phases.

From the catastrophic_forgetting repository root:
  python grokking/train.py --phase a --run-dir grokking/runs/a_seed0
  python grokking/train.py --phase joint --run-dir grokking/runs/joint_seed0
  python grokking/train.py --phase b --checkpoint grokking/runs/a_seed0/checkpoints/step_005000.pt --run-dir grokking/runs/b_from_5000
  python grokking/train.py --phase b --b-operation sum_plus_one --checkpoint grokking/runs/a_seed0/checkpoints/step_001000.pt --run-dir grokking/runs/shift_from_1000
  python grokking/train.py --phase a_control --checkpoint grokking/runs/a_seed0/checkpoints/step_005000.pt --run-dir grokking/runs/a_control_from_5000
  python grokking/train.py --phase b --resume grokking/runs/b_from_5000/checkpoints/step_005000.pt --run-dir grokking/runs/b_from_5000 --steps 40000

The ``--steps`` argument is the total updates *within this phase*, not the
number of extra updates for resumed runs.  Every phase is started manually.
Dependencies: torch, wandb (unless --wandb-mode disabled). Python >=3.9.

Nanda-style baseline: addition mod 113, 30% fixed split, one layer,
4 attention heads, 128 residual width, 512 ReLU MLP, zero-initialized MLP
biases, no LayerNorm/dropout, full-batch AdamW (lr 1e-3, betas .9/.98,
weight_decay 1). The model uses Nanda et al.'s initialization distributions.
Deliberate deviation: the input is [a, b, operator], with '+' or '-' in the
last position, rather than Nanda's [a, b, '=']. Output logits cover numbers
0..112 (not operation tokens), so this remains an *adaptation*, not a bit-for-
bit reproduction of the paper. The dataset split and optimizer stay unchanged. The unused `-` input token
can also serve as a B-task identifier for shifted addition, allowing existing
addition checkpoints to be reused without changing the architecture.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class OneLayerTransformer(nn.Module):
    """Decoder-only, one causal attention block + ReLU MLP, no LayerNorm."""

    def __init__(self, p=113, d_model=128, heads=4, d_mlp=512):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.p, self.heads = p, heads
        # Numbers 0..p-1, '+' at p and '-' at p+1. Three positions: [a, b, +/-].
        self.token = nn.Embedding(p + 2, d_model)
        self.position = nn.Parameter(torch.empty(3, d_model))
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.mlp_in = nn.Linear(d_model, d_mlp, bias=True)
        self.mlp_out = nn.Linear(d_mlp, d_model, bias=True)
        self.unembed = nn.Linear(d_model, p, bias=False)

        # Nanda et al., progress-measures-paper/transformers.py:
        # W_E, W_pos, W_{Q,K,V,O}, W_in and W_out ~ N(0, 1/d_model),
        # W_U ~ N(0, 1/d_vocab); MLP biases start at zero.
        # Our extended input vocabulary has d_vocab=p+2 (113 numbers + +/-).
        # The qkv rows are equivalent to separate Q, K and V matrices.
        with torch.no_grad():
            for tensor in (self.token.weight, self.position,
                           self.qkv.weight, self.attn_out.weight,
                           self.mlp_in.weight, self.mlp_out.weight):
                nn.init.normal_(tensor, mean=0.0, std=1.0 / math.sqrt(d_model))
            nn.init.normal_(self.unembed.weight, mean=0.0,
                            std=1.0 / math.sqrt(p + 2))
            nn.init.zeros_(self.mlp_in.bias)
            nn.init.zeros_(self.mlp_out.bias)

    def forward(self, tokens):
        batch, length = tokens.shape
        if length != 3:
            raise ValueError("Expected [a, b, operator] (3 tokens)")
        x = self.token(tokens) + self.position[None, :, :]
        n_heads = self.heads
        d_head = x.shape[-1] // n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split_heads(t):
            return t.view(batch, length, n_heads, d_head).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            split_heads(q), split_heads(k), split_heads(v),
            dropout_p=0.0, is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, length, -1)
        x = x + self.attn_out(attn)
        x = x + self.mlp_out(F.relu(self.mlp_in(x)))
        return self.unembed(x[:, -1, :])  # Answer predicted at the final +/- token.


def make_data(p, fraction, split_seed, device, b_operation="subtract"):
    """Use the SAME pair split for A and B, so B train never reveals A test pairs."""
    if b_operation not in ("subtract", "sum_plus_one"):
        raise ValueError(f"Unrecognized B operation: {b_operation}")
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    n = p * p
    generator = torch.Generator(device="cpu").manual_seed(split_seed)
    order = torch.randperm(n, generator=generator)
    n_train = int(n * fraction)
    train_ids, test_ids = order[:n_train], order[n_train:]
    data = {}
    for task, operator in (("A", p), ("B", p + 1)):
        if task == "A":
            answers = (a + b) % p
        elif b_operation == "subtract":
            answers = (a - b) % p
        else:
            answers = (a + b + 1) % p
        tokens = torch.stack((a, b, torch.full_like(a, operator)), dim=1)
        data[task] = {
            "train_x": tokens[train_ids].to(device),
            "train_y": answers[train_ids].to(device),
            "test_x": tokens[test_ids].to(device),
            "test_y": answers[test_ids].to(device),
        }
    return data, n_train, n - n_train


@torch.no_grad()
def evaluate(model, data, tasks):
    model.eval()
    metrics = {}
    for task in tasks:
        d = data[task]
        for split in ("train", "test"):
            logits = model(d[f"{split}_x"])
            truth = d[f"{split}_y"]
            metrics[f"{task}/{split}_loss"] = F.cross_entropy(logits, truth).item()
            metrics[f"{task}/{split}_accuracy"] = (
                (logits.argmax(dim=-1) == truth).float().mean().item()
            )
    model.train()
    return metrics


def get_rng_state():
    result = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def put_rng_state(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(directory, step, model, optimizer, config, phase,
                    origin, wandb_id, metrics):
    path = directory / "checkpoints" / f"step_{step:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step, "phase": phase, "config": config,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "origin": origin, "wandb_id": wandb_id, "rng": get_rng_state(),
        "metrics": metrics,
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(path):
    # Treat checkpoint files as trusted: model and optimizer state are pickled.
    return torch.load(path, map_location="cpu", weights_only=False)


def optimizer_step_with_frozen_token(model, optimizer, op_token):
    """Protect unused operator row from BOTH Adam moments and AdamW decay.

    Reusing one embedding matrix otherwise silently decays the old '+' row
    during Task B, even though its task gradients are zero.
    """
    weights = model.token.weight
    with torch.no_grad():
        previous = weights[op_token].clone()
        state = optimizer.state.get(weights, {})
        previous_moments = {
            k: state[k][op_token].clone()
            for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
            if k in state
        }
    optimizer.step()
    with torch.no_grad():
        weights[op_token].copy_(previous)
        state = optimizer.state.get(weights, {})
        for key, value in previous_moments.items():
            state[key][op_token].copy_(value)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("a", "b", "a_control", "joint"), required=True,
                        help="a: addition baseline; b: subtraction only; a_control: continue A; joint: train A+B equally from scratch")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--b-operation", choices=("subtract", "sum_plus_one"), default=None,
                        help="B labels: (a-b) mod p (default) or (a+b+1) mod p. "
                             "Reuses the existing '-' input token as B task identifier.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="Branch from a manually selected A checkpoint")
    source.add_argument("--resume", type=Path, help="Resume interrupted SAME phase, preserving optimizer/RNG")
    parser.add_argument("--steps", type=int, default=40000,
                        help="Total SGD updates for THIS phase (also when resuming)")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--train-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--reset-optimizer", action="store_true",
                        help="On NEW branch only: reset AdamW moments instead of inheriting A moments")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--wandb-project", default="grokking-continual")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-name", default=None)
    return parser.parse_args()


def main():
    args = arguments()
    if args.phase in ("a", "a_control") and args.b_operation is not None:
        raise ValueError("--b-operation is only meaningful for Phase b or joint")
    if args.steps <= 0 or args.eval_every <= 0 or args.checkpoint_every <= 0:
        raise ValueError("--steps, --eval-every and --checkpoint-every must be positive")
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be in (0,1)")
    if args.phase in ("a", "joint") and args.checkpoint:
        raise ValueError("Phase a/joint starts from scratch; use --resume to continue an existing run")
    if args.phase in ("b", "a_control") and not (args.checkpoint or args.resume):
        raise ValueError("Phase b or a_control requires --checkpoint (or --resume)")
    if args.reset_optimizer and (args.phase in ("a", "joint") or args.resume):
        raise ValueError("--reset-optimizer is only supported for branching from --checkpoint")
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if args.device == "auto" else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA, but torch.cuda.is_available() is false")

    if args.resume:
        ckpt = load_checkpoint(args.resume)
        if ckpt["phase"] != args.phase:
            raise ValueError("--resume phase does not match checkpoint phase")
        if not args.run_dir.exists():
            raise ValueError("For --resume, --run-dir must be the existing original run directory")
        start_step = int(ckpt["step"])
        origin = ckpt["origin"]
        wandb_id = ckpt.get("wandb_id")
        config = dict(ckpt["config"])
        previous_b_op = config.get("b_operation", "subtract")
        if args.b_operation is not None and args.b_operation != previous_b_op:
            raise ValueError("Cannot change B operation when resuming an existing run")
    elif args.checkpoint:
        ckpt = load_checkpoint(args.checkpoint)
        if ckpt["phase"] != "a":
            raise ValueError("New branches must originate from Phase A checkpoints")
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing run: {args.run_dir}")
        start_step = 0
        origin = {"checkpoint": str(args.checkpoint.resolve()), "parent_step": ckpt["step"]}
        wandb_id = None
        config = dict(ckpt["config"])
        config["b_operation"] = args.b_operation or "subtract"
    else:
        ckpt = None
        if args.run_dir.exists() and any(args.run_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing run: {args.run_dir}")
        start_step = 0
        origin = None
        wandb_id = None
        config = {"p": args.p, "d_model": 128, "heads": 4, "d_mlp": 512,
                  "fraction": args.train_fraction, "seed": args.seed,
                  "split_seed": args.split_seed, "lr": args.lr, "weight_decay": args.weight_decay,
                  "betas": (0.9, 0.98), "input_format": "a,b,operator",
                  "init_scheme": "nanda-2023", "model_version": 2,
                  "b_operation": args.b_operation or "subtract"}
    # Prevent silently mixing old 4-token checkpoints with this 3-token model.
    if config.get("input_format") != "a,b,operator" or config.get("model_version") != 2:
        raise ValueError(
            "Checkpoint uses the old 4-token/different-initialization model. "
            "Please train Phase A again in a NEW run directory with this version. "
            "Existing grokking/runs/a_seed0 checkpoints must not be reused."
        )
    if args.steps < start_step:
        raise ValueError(f"Requested --steps={args.steps} is before resume step {start_step}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = OneLayerTransformer(config["p"], config["d_model"],
                                config["heads"], config["d_mlp"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  betas=tuple(config["betas"]), weight_decay=config["weight_decay"])
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        if not args.reset_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
        if args.resume:
            put_rng_state(ckpt["rng"])

    data, n_train, n_test = make_data(config["p"], config["fraction"], config["split_seed"], device,
                                     b_operation=config.get("b_operation", "subtract"))
    train_task = "B" if args.phase == "b" else ("joint" if args.phase == "joint" else "A")
    frozen_op = config["p"] if train_task == "B" else config["p"] + 1
    eval_tasks = ("A",) if args.phase == "a" else ("A", "B")
    print(f"Phase={args.phase}, device={device}, model parameters={sum(t.numel() for t in model.parameters()):,}", flush=True)
    print(f"p={config['p']}, train={n_train}, test={n_test}, initial_step={start_step}, target_step={args.steps}", flush=True)
    if args.phase in ("b", "joint"):
        print(f"B task operation={config.get('b_operation', 'subtract')} (input token '-' marks B)", flush=True)
    if args.phase == "joint":
        print("Joint A+B: 50/50 mean train CE; both operator embeddings are trainable", flush=True)
    else:
        print(f"Frozen operator embedding id={frozen_op}, optimizer={'reset' if args.reset_optimizer else 'inherited / fresh'}", flush=True)

    wandb = None
    if args.wandb_mode != "disabled":
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise RuntimeError("Install wandb in the cluster environment: pip install wandb") from exc
        wandb = wandb_module
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
            name=args.wandb_name or args.run_dir.name,
            id=wandb_id if args.resume else None,
            resume="allow" if args.resume else None,
            dir=str(args.run_dir),
            config={**config, "phase": args.phase, "origin": origin,
                    "steps": args.steps, "train_size": n_train, "test_size": n_test,
                    "eval_every": args.eval_every, "checkpoint_every": args.checkpoint_every},
        )
        wandb_id = run.id

    metrics = evaluate(model, data, eval_tasks)
    if start_step == 0:
        if wandb:
            wandb.log(metrics, step=0)
        location = save_checkpoint(args.run_dir, 0, model, optimizer, config, args.phase,
                                   origin, wandb_id, metrics)
        print(f"Initial checkpoint: {location}", flush=True)
    last_log = time.monotonic()
    try:
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if args.phase == "joint":
                d_a, d_b = data["A"], data["B"]
                # One full-batch update over both task datasets, weighted equally.
                logits = model(torch.cat((d_a["train_x"], d_b["train_x"]), dim=0))
                labels = torch.cat((d_a["train_y"], d_b["train_y"]), dim=0)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()  # Both +/- embeddings actively trained.
            else:
                d = data[train_task]
                logits = model(d["train_x"])
                loss = F.cross_entropy(logits, d["train_y"])
                loss.backward()
                optimizer_step_with_frozen_token(model, optimizer, frozen_op)
            should_log = step % args.eval_every == 0 or step == args.steps
            should_save = step % args.checkpoint_every == 0 or step == args.steps
            if should_log or should_save:
                metrics = evaluate(model, data, eval_tasks)
                if should_log:
                    if wandb:
                        wandb.log(metrics, step=step)
                    elapsed = time.monotonic() - last_log
                    if args.phase == "joint":
                        message = (f"step {step:6d}  "
                                   f"A train/test CE={metrics['A/train_loss']:.4f}/{metrics['A/test_loss']:.4f}"
                                   f" acc={metrics['A/train_accuracy']:.3f}/{metrics['A/test_accuracy']:.3f}  "
                                   f"B train/test CE={metrics['B/train_loss']:.4f}/{metrics['B/test_loss']:.4f}"
                                   f" acc={metrics['B/train_accuracy']:.3f}/{metrics['B/test_accuracy']:.3f}")
                    else:
                        message = f"step {step:6d}  {train_task} train/test CE="
                        message += f"{metrics[f'{train_task}/train_loss']:.4f}/{metrics[f'{train_task}/test_loss']:.4f}"
                        message += f"  acc={metrics[f'{train_task}/train_accuracy']:.3f}/{metrics[f'{train_task}/test_accuracy']:.3f}"
                        if args.phase != "a":
                            message += f"  A test acc={metrics['A/test_accuracy']:.3f}"
                    print(message + f"  ({elapsed:.1f}s since log)", flush=True)
                    last_log = time.monotonic()
                if should_save:
                    saved = save_checkpoint(args.run_dir, step, model, optimizer, config,
                                            args.phase, origin, wandb_id, metrics)
                    print(f"Checkpoint: {saved}", flush=True)
    finally:
        if wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
