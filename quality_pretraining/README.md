# Experiment 1: attention cross-entropy

The model architecture and teacher tasks are unchanged. Select the optimization
objective with `--loss attention_ce` or `--loss output_mse` (the legacy default).
Teacher attention is the soft target. CE sums over key positions and averages
over examples and query positions, using student `log_softmax` for stability.
All attention scores retain the original trace centering and scaling.

Every run records **both attention CE and continuous-output MSE** on fixed test
inputs. CE runs optimize only CE. Teacher attention entropy and KL = CE − entropy
are also saved. Matching teacher attention gives zero KL, not zero CE.
The evaluation MSE averages all output entries; the legacy MSE training objective
sums sequence positions and divides by d, so its value is T times evaluation MSE.

## Run the three stages

Run the following from `quality_pretraining`, using your PyTorch environment.
These are short pilot budgets, not convergence settings. Retune learning rates
and horizons for CE before running the dimension/checkpoint/seed sweep.

```bash
python -m scripts.run_pretraining_quality \
    --loss attention_ce --run-name pilot_d100 \
    --d 100 --T 5 --target-steps 10000 \
    --checkpoint-steps 0 1000 10000 --n-test 256 --device cpu
```

CE fresh-run names receive an `_attention_ce` suffix automatically. Select a
checkpoint and train Task 1:

```bash
python -m scripts.run_task1_finetuning \
    --pretrain-run-dir results/quality_pretraining/pilot_d100_attention_ce \
    --pretrain-step 10000 --target-steps 10000 --device cpu
```

Fine-tuning inherits the source objective if `--loss` is omitted. Its final output
prints the run directory; supply that exact path to Task 2:

```bash
python -m scripts.run_task2_finetuning \
    --pretrain-run-dir results/quality_pretraining/pilot_d100_attention_ce \
    --pretrain-step 10000 \
    --task1-run-dir /absolute/path/to/the/task1/run \
    --task1-step 10000 --target-steps 10000 --device cpu
```

Task 2 loads the saved Task-1 vector without resetting it. Both fine-tuning stages
freeze W. An objective mismatch between the requested loss and source
configurations/checkpoints is rejected. Existing configurations without objective
metadata are interpreted as `output_mse`. Use fresh CE pre-training checkpoints
for the CE experiment rather than silently switching an old MSE run.

To resume CE pre-training, provide its full saved run name; the loss is inherited:

```bash
python -m scripts.run_pretraining_quality \
    --run-name pilot_d100_attention_ce --resume --target-steps 20000 --device cpu
```

Existing result directories require `--overwrite` before a fresh run can replace
them. Legacy metric CSVs are extended on resume, leaving previously unavailable
CE values blank. `--results-root` can relocate saved runs for each stage.

## Plot CE and MSE

The last section of `notebooks/analyse_pretraining_quality.ipynb` discovers saved
CE runs and plots the two metrics side by side. Run that section independently
of the original MSE analysis and edit `SELECTED_INDICES` to select runs.

For a standalone PNG or PDF:

```bash
python -m scripts.plot_attention_experiment \
    --run-dirs /absolute/path/to/pretraining/run \
               /absolute/path/to/task1/run /absolute/path/to/task2/run \
    --output results/attention_ce_and_mse.png
```

Each run occupies one row with CE on the left and MSE on the right. Task colors
are shared across panels. Task-1 `x` markers show the base-only state before
inserting the random adapter; the trajectory starts with the initialized adapter.
Task-2 CE forgetting is measured from the selected Task-1 endpoint. Normalized
CE forgetting uses acquisition from the base-only state, and is undefined (NaN)
when acquired improvement is nonpositive or below 1e-12.

## Cluster launchers

- Fresh scaling runs: `LOSS=attention_ce sbatch scripts/run_pretraining_diffscales.sh`.
- `run_pretraining_quality.sh` resumes the objective saved in the run configuration.
- Task-1 and Task-2 shell launchers inherit their source objectives. Their optional
  final arguments (6 and 7, respectively) can explicitly select the objective.

The original cluster paths and GPU resource settings in these launchers remain
specific to the existing cluster environment.

## Verification

```bash
python -m pytest tests -q
```

Tests cover attention CE gradients and normalization, fixed teacher targets,
pre-training resume, frozen-base sequential adaptation, CLI checkpoint handoffs
for both objectives, CSV compatibility, and export of both plotted metrics.
