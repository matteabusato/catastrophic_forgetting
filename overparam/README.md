# Overparameterization and sequential forgetting

The recommended interface is two independent jobs: train Task 1, inspect its
learning curves, then submit Task 2 from a selected Task-1 checkpoint. Requires
Python >=3.9 and PyTorch. Run commands from the repository root.

## Code plan

- `scripts/run_task1.py`: CLI for one student width trained only on Task 1.
- `scripts/run_task2.py`: independent CLI loading a selected Task-1 checkpoint.
- `scripts/run_task1.sh`: Slurm array launcher, one job per kappa.
- `scripts/run_task2.sh`: independent Slurm launcher for one selected source run.
- `src/phases.py`: shared configuration, exact continuation, checkpoint schedule,
  metric logging, source handoff and single-phase training implementation.
- `src/model.py`, `data.py`, `losses.py`, `observables.py`: attention architecture,
  teacher/input generation, MSE and diagnostics shared by both phases.
- `tests/test_phases.py`: schedule, pairing, exact resume and task-switch checks.

The earlier combined `src/experiment.py` interface remains available for small
programmatic experiments. Use the separate scripts for this checkpointed protocol.

## Agreed experiment

| Setting | Default |
|---|---|
| d, tokens T | 100, 5 |
| kappa in Task-1 array | 1, 1.2, 1.5, 2, 3, 4, 5, 10, 20 |
| Student width P | round(kappa*d) |
| Teacher widths | P_star=d (both fixed across student widths) |
| Teachers | independent standard Gaussian factors, rho=0 |
| Student initialization | independent standard Gaussian factor W of shape (d,P) |
| Replicate seed | 0 |
| Trainable parameters | all of W, no adapter |
| Optimizer | plain SGD, batch size 1, fixed lr=0.01 |
| Loss | MSE averaged over examples, tokens and features |
| Task-1 horizon | 40*d^2 = 400000 updates |
| Metric interval | d = 100 updates |
| Test / independent probe size | 2048 / 256 sequences |
| Checkpoints, phase-local alpha | 5, 10, 20, 30, 40, then every 10 |
| Precision | float64 |

For every sequence, S=WW.T/sqrt(P), logits=(X S X.T-tr(S) I_T)/d,
attention=row_softmax(logits), output=attention @ X. Centering is differentiable
and uses the exact Gaussian expectation conditional on the current weights.
MSE normalization differs by a factor T from the legacy sum-over-tokens objective
in quality_pretraining; lr=0.01 here applies to the normalized MSE.

Within a replicate, role seeds give identical teachers, evaluation inputs and
phase-specific online input streams across all widths, including independent
Slurm array jobs. Student seeds depend on P. CPU-local generators are transferred
to the requested device. Exact cross-device/version reproducibility is not
promised. Save initial matrix diagnostics: Gaussian factor initialization changes
the distribution of S with P, so width effects include initialization/history.

## Submit Task 1

From the repository root on the cluster, create logs BEFORE sbatch (Slurm opens
these files before the script executes), then submit:

```bash
mkdir -p logs
sbatch overparam/scripts/run_task1.sh \
  --d 100 --tokens 5 --teacher-rank 100 --seed 0 \
  --target-alpha 40 --lr 0.01 --batch-size 1 \
  --eval-every 100 \
  --checkpoint-alphas 5 10 20 30 40 --checkpoint-every-alpha 10
```

This creates nine independent GPU jobs (array indices 0-8). No Task-2 job is
created. The launcher defaults to partition gpu, one GPU, four CPU cores, 16G
memory and six hours per array member, following existing repository launchers.
These are resource requests, not a runtime estimate; override with sbatch flags:

```bash
sbatch --time=12:00:00 --mem=32G --array=0-8%3 \
  overparam/scripts/run_task1.sh --target-alpha 40
```

`%3` limits concurrent widths to three. Use `--array=0` for a single kappa=1
pilot. Array mapping: 0=1, 1=1.2, 2=1.5, 3=2, 4=3, 5=4, 6=5, 7=10, 8=20.
To change the grid, set `OVERPARAM_KAPPAS="1 2 4"` and submit with `--array=0-2`.

Python is selected from OVERPARAM_PYTHON, the active environment, repo .venv,
parent .venv, then python3. For another environment, activate it before submission
or export OVERPARAM_PYTHON=/absolute/path/to/python. It must already have PyTorch;
the scripts never install or activate external dependencies. Set
OVERPARAM_REPO_ROOT if submitting from outside the repository root. Set
OVERPARAM_DEVICE=cpu for a local shell smoke test; Slurm launchers default to cuda.

Task-1 outputs: `overparam/results/task1/d100_seed0_P100/`, ...,
`overparam/results/task1/d100_seed0_P2000/`. Override --results-root or --run-dir.
Existing directories are refused unless --resume is explicit. The default grid
performs 3.6 million total Task-1 updates. Exact fixed-set evaluation every 100
updates adds substantial compute; wall time is workload- and device-dependent.

## Inspect and extend Task 1

Inspect metrics.csv and checkpoints before deciding to proceed. No stopping
criterion, plateau test or learning-rate schedule is applied. A completed horizon
is not labelled convergence. Metrics are saved at step zero, every eval_every
updates, every checkpoint, and the exact final update. Minibatch MSE is recorded
before its optimizer update; held-out diagnostics describe the resulting state.

To extend just the P=100 run from alpha=40 to alpha=60:

```bash
sbatch --array=0 overparam/scripts/run_task1.sh \
  --resume --run-dir overparam/results/task1/d100_seed0_P100 \
  --target-alpha 60
```

Continuation restores W, optimizer and the training input generator and appends
metrics. Checkpoints at alpha=50 and 60 are added. --target-steps and --target-alpha
specify TOTAL phase updates, not additional updates. --target-steps takes
precedence. The default Task-1 horizon is always 40d^2; explicit shorter horizons
are allowed for smoke tests and produce a warning.

## Submit Task 2 independently

Select an actual Task-1 checkpoint, then submit the second script. Example:

```bash
sbatch overparam/scripts/run_task2.sh \
  --task1-run-dir overparam/results/task1/d100_seed0_P100 \
  --task1-step 400000 --target-alpha 40
```

Omitting --task1-step chooses the latest complete checkpoint, including a
checkpoint from an interrupted job. Task 2 preserves W exactly, creates a fresh
plain-SGD optimizer, and uses its independent reserved input stream. It inherits
architecture, teachers, dtype, lr, batch size and metric/checkpoint settings;
optimization, logging and device options can be explicitly overridden. Task-2
horizon defaults to 40d^2 and can be set independently. No source files are changed.

The new run copies both teachers, evaluation inputs, original initialization and
the selected switch checkpoint. After setup, Task-2 continuation is independent
of the source folder. Default output example:
`overparam/results/task2/d100_seed0_P100_from_400000/`.

To extend that Task-2 run:

```bash
sbatch overparam/scripts/run_task2.sh --resume \
  --run-dir overparam/results/task2/d100_seed0_P100_from_400000 \
  --target-alpha 60
```

Absolute forgetting is test_task1_mse minus its value at the selected Task-1
checkpoint. Relative forgetting divides by Task-1 improvement from the original
random student; it is null/blank when improvement <=1e-12. Neither measure is
clipped. Task-2 phase-local updates/alpha start at zero, while total_step includes
the selected Task-1 update count. The switch baseline is preserved on resume.

## Exposed variables

Run either Python script with --help for the full interface.

- Model/data (Task 1): --d, --tokens (alias --T), --kappa, --teacher-rank,
  --init-std, --teacher-std, --n-test, --n-probe, --dtype.
- RNG (Task 1): --seed plus --teacher1-seed, --teacher2-seed, --student-seed,
  --train-seed, --task2-train-seed, --test-seed, --probe-seed.
- Both phases: --target-alpha, --target-steps, --lr, --batch-size, --eval-every,
  --eval-batch-size, --checkpoint-alphas, --checkpoint-every-alpha, --device,
  --threads, --print-every, --run-dir, --results-root, --resume, --train-seed.
- Task-2 source: --task1-run-dir and --task1-step. Architecture and diagnostic
  sets are inherited to keep the tasks and forgetting baselines consistent.
- On resume, scientific settings are inherited and conflicting values are
  rejected. Only horizon, device, threads, evaluation batch size and console
  interval can change. To alter scientific settings, start a fresh run.

Each run saves config.json, status.json, run_history.jsonl, teachers.pt,
evaluation_inputs.pt, initial_W.pt, metrics.csv and summary.json. Atomic
checkpoints in checkpoints/step_XXXXXXXXXXXX.pt contain W, optimizer state, input
RNG state, current metrics, original Task-1 error, configuration and software
version. Checkpoint zero is saved as well as the requested milestones and final
step. Kernel file locking prevents two jobs from writing one run simultaneously.

On interrupted-run resume, metric rows after the last complete checkpoint are
rolled back (an audit backup is retained), and those updates are replayed.
This avoids duplicate or inconsistent histories. The interruption status may
remain running if Slurm killed the process; --resume still works from complete
checkpoints. Probe metrics are independent distribution diagnostics, not empirical
training-set errors. Test/probe values never control the training horizon.

## Verification

```bash
python3 -m unittest discover -s overparam/tests -v
bash -n overparam/scripts/run_task1.sh overparam/scripts/run_task2.sh
```
