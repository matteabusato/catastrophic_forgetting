from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROBE_LABELS = {"medium_sized_mammals": "Related: medium-sized mammals",
                "vehicles_2": "Unrelated: vehicles 2"}


def mean_std(ax, frame, x, y, label, **kwargs):
    grouped = frame.groupby(x)[y].agg(["mean", "std"]).reset_index()
    ax.plot(grouped[x], grouped["mean"], marker="o", label=label, **kwargs)
    if grouped["std"].notna().any():
        ax.fill_between(grouped[x], grouped["mean"] - grouped["std"],
                        grouped["mean"] + grouped["std"], alpha=0.18)


def main():
    parser = argparse.ArgumentParser(description="Plot sequential transfer results")
    parser.add_argument("--results-dir", default="./results")
    args = parser.parse_args()
    root = Path(args.results_dir)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    final_path = root / "probes" / "probe_final.csv"
    trajectory_path = root / "probes" / "trajectories" / "probe_trajectory.csv"
    if not final_path.exists() or not trajectory_path.exists():
        raise FileNotFoundError("Probe result files not found; run evaluate_probes.py first")
    final = pd.read_csv(final_path)
    trajectory = pd.read_csv(trajectory_path)
    final = final.drop_duplicates(["seed", "checkpoint", "probe", "method"], keep="last")
    sequential = final[final.checkpoint.isin(["init", "after_task_1", "after_task_2", "after_task_3", "after_task_4"])]
    colors = {"medium_sized_mammals": "tab:blue", "vehicles_2": "tab:orange"}
    # Figure 1: final accuracy by learned task, separate panel per probe method.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, method in zip(axes, ["linear", "finetune"]):
        subset = sequential[sequential.method == method]
        for probe in subset.probe.unique():
            mean_std(ax, subset[subset.probe == probe], "tasks_learned", "test_accuracy",
                     PROBE_LABELS.get(probe, probe), color=colors.get(probe))
        ax.set(title=f"{method.replace('_', ' ').title()} evaluation", xlabel="Sequential tasks learned",
               xticks=range(5), ylim=(0, 1))
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Probe test accuracy")
    axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "figure1_transfer_accuracy.png", dpi=180); plt.close(fig)

    # Figure 2: gains over the corresponding random-initialization checkpoint.
    random_rows = sequential[sequential.tasks_learned == 0][["seed", "probe", "method", "test_accuracy"]].rename(
        columns={"test_accuracy": "random_accuracy"})
    gains = sequential.merge(random_rows, on=["seed", "probe", "method"], how="left")
    gains["gain"] = gains.test_accuracy - gains.random_accuracy
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, method in zip(axes, ["linear", "finetune"]):
        subset = gains[gains.method == method]
        for probe in subset.probe.unique():
            mean_std(ax, subset[subset.probe == probe], "tasks_learned", "gain",
                     PROBE_LABELS.get(probe, probe), color=colors.get(probe))
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=f"{method.replace('_', ' ').title()}", xlabel="Sequential tasks learned", xticks=range(5))
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Accuracy gain over random initialization")
    axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "figure2_transfer_gain.png", dpi=180); plt.close(fig)

    # Figure 3: probe learning trajectories for checkpoints 0, 2, and 4.
    subset = trajectory[trajectory.tasks_learned.isin([0, 2, 4]) & trajectory.checkpoint.isin(
        ["init", "after_task_2", "after_task_4"])]
    for probe in subset.probe.unique():
        for metric, ylabel, filename in [("train_loss", "Probe training loss", "loss"),
                                         ("test_accuracy", "Probe test accuracy", "accuracy")]:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
            for ax, method in zip(axes, ["linear", "finetune"]):
                method_frame = subset[(subset.probe == probe) & (subset.method == method)]
                for tasks, group in method_frame.groupby("tasks_learned"):
                    label = f"{tasks} tasks"
                    mean_std(ax, group, "epoch", metric, label)
                ax.set(title=method.replace("_", " ").title(), xlabel="Probe training epoch")
                ax.grid(alpha=.25)
            axes[0].set_ylabel(ylabel)
            axes[1].legend(frameon=False)
            fig.suptitle(PROBE_LABELS.get(probe, probe))
            fig.tight_layout(); fig.savefig(out / f"figure3_{probe}_{filename}.png", dpi=180); plt.close(fig)

    diagnostics_path = root / "representation_diagnostics.csv"
    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        fig, ax = plt.subplots(figsize=(6, 4))
        mean_std(ax, diagnostics, "tasks_learned", "participation_ratio", "Participation ratio")
        ax.set(xlabel="Sequential tasks learned", ylabel="Feature participation ratio", xticks=range(5))
        ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(out / "figure4_representation_diagnostics.png", dpi=180); plt.close(fig)

    baseline_rows = final[final.checkpoint.isin(["last_task_only"])]
    if not baseline_rows.empty:
        compare = pd.concat([sequential[sequential.checkpoint.isin(["after_task_4"])], baseline_rows], ignore_index=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, method in zip(axes, ["linear", "finetune"]):
            frame = compare[compare.method == method].copy()
            frame["variant"] = frame.checkpoint.map({"after_task_4": "Sequential (4 tasks)", "last_task_only": "Task 4 only"})
            grouped = frame.groupby(["variant", "probe"]).test_accuracy.agg(["mean", "std"]).reset_index()
            for variant in grouped.variant.unique():
                rows = grouped[grouped.variant == variant]
                ax.bar([f"{p}\n{variant}" for p in rows.probe], rows["mean"],
                       yerr=rows["std"].fillna(0), capsize=3, label=variant if ax is axes[0] else None)
            ax.set(title=method.title(), ylabel="Probe test accuracy", ylim=(0, 1))
            ax.grid(axis="y", alpha=.25)
        axes[0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(out / "figure5_baseline_comparison.png", dpi=180); plt.close(fig)
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
