"""Plot attention CE and output MSE from saved Experiment-1 runs."""

import csv
import json
from pathlib import Path


def plot_run_metrics(run_dirs, output_path=None):
    """One row per run, with CE and MSE on separate axes and shared task colors.

    Evaluation MSE averages all output entries (the legacy generalization_error),
    whereas the original training MSE sums positions and divides by d.
    """
    import matplotlib.pyplot as plt

    run_dirs = [Path(path) for path in run_dirs]
    if not run_dirs:
        raise ValueError("Select at least one run directory containing metrics.csv.")
    fig, axes = plt.subplots(len(run_dirs), 2, figsize=(13, 3.8 * len(run_dirs)),
                             squeeze=False, constrained_layout=True)
    colors = {"pretraining": "C0", "task1": "C1", "task2": "C2"}
    for row_index, run_dir in enumerate(run_dirs):
        config = json.loads((run_dir / "config.json").read_text())
        with (run_dir / "metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"No metrics in {run_dir}")
        # Plot the pre-insertion state separately, rather than joining it to the
        # initialized adapter at the same time coordinate.
        baseline = next((r for r in rows if r.get("stage") == "before_adapter_init"), None)
        trajectory = [r for r in rows if r.get("stage") != "before_adapter_init"]
        if "task2_step_over_d" in rows[0]:
            x_key, x_label = "task2_step_over_d", "Task-2 updates / d"
        elif "finetune_step_over_d" in rows[0]:
            x_key, x_label = "finetune_step_over_d", "Task-1 updates / d"
        else:
            x_key, x_label = "step_over_d2", "Pre-training updates / d²"
        pretraining_only = x_key == "step_over_d2"
        for task in (["pretraining"] if pretraining_only else colors):
            ce_key = "attention_ce" if pretraining_only else f"{task}_attention_ce"
            mse_key = "output_mse" if pretraining_only else f"{task}_output_mse"
            legacy_mse = "generalization_error" if pretraining_only else f"{task}_error"
            for column, key in enumerate([ce_key, mse_key]):
                ax = axes[row_index, column]
                actual_key = legacy_mse if column == 1 and key not in rows[0] else key
                points = [(float(r[x_key]), float(r[actual_key])) for r in trajectory
                          if r.get(actual_key, "") != ""]
                if points:
                    ax.plot(*zip(*points), color=colors[task], label=task)
                if baseline and baseline.get(actual_key, ""):
                    ax.scatter([0], [float(baseline[actual_key])], color=colors[task],
                               marker="x", s=45)
        for column, ax in enumerate(axes[row_index]):
            ax.set_xlabel(x_label)
            ax.set_ylabel("Attention CE (nats per query)" if column == 0 else "Output MSE (mean per entry)")
            ax.set_title(f"{config['experiment']} · {config.get('loss_type', 'output_mse')} · d={config['d']}")
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend()
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    return fig, axes
