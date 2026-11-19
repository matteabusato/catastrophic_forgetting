# Sequential feature learning on CIFAR-100

Reproducible ResNet-18 pilot for forward transfer and adaptation after sequential task learning. This is an empirical experiment; the code does not implement replica theory. CIFAR-100 labels are read from the official archive's fine and coarse label fields.

## Setup

Use a PyTorch/torchvision installation compatible with the compute environment. On Izar, verify the available CUDA module/environment before installing; the scripts do not assume a specific GPU model or cluster path.

```bash
cd sequential_feature_learning
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Edit `configs/pilot.yaml` for local storage, data location, seed count, and resource budgets. Defaults use one seed, 40 epochs per task, 2,500 training images per task (500 per fine class), 50 probe examples per class, and 30 probe epochs for each method. Set `experiment.data_dir`, `output_dir`, and `checkpoint_dir` to paths available on the target machine. CIFAR-100 is downloaded automatically when absent.

## Local commands

From this directory:

```bash
# One seed on the available device; checkpoints/seed_17/init.pt and after_task_*.pt are written.
python -m src.train_sequential --config configs/pilot.yaml --seed 17

# Resume an interrupted seed from its epoch-level last.pt checkpoint.
python -m src.train_sequential --config configs/pilot.yaml --seed 17 --resume

# Train the prioritized task-4-only baseline. Random initialization uses init.pt from sequential training.
python -m src.train_baselines --config configs/pilot.yaml --seed 17

# Evaluate all five sequential checkpoints for each configured seed.
python -m src.evaluate_probes --config configs/pilot.yaml

# Add task-4-only checkpoint to the probe evaluation.
python -m src.evaluate_probes --config configs/pilot.yaml \
  --checkpoint checkpoints/seed_17/baselines/last_task_only.pt

# Optional representation diagnostics and plots.
python -m src.diagnostics --config configs/pilot.yaml
python -m src.plot_results --results-dir results
```

Each checkpoint contains the backbone, learned heads, task order, seed, config, optimizer step count, examples processed, and metadata. `last.pt` also stores optimizer, scheduler, and AMP scaler state during a task for epoch-level resumption. The optimizer and cosine schedule are recreated at every new sequential task.

## Probe protocol

The four sequential tasks are aquatic mammals, fish, large carnivores, and small mammals. Probes are medium-sized mammals and vehicles 2. The latter are excluded from sequential training by construction. Task labels are filtered using CIFAR-100's official coarse labels; fine labels are remapped to local labels 0–4 in sorted fine-label order.

For each seed, stratified training subsets are fixed across checkpoints and methods. Each probe/method starts with the same seeded five-way head initialization at every checkpoint. Linear probing and full fine-tuning use separate model instances loaded from the saved backbone. Linear probing freezes all backbone parameters and keeps BatchNorm in evaluation mode. Probe test metrics are diagnostic and are not used to select checkpoints or hyperparameters. Probe trajectories report train/test loss and accuracy, best test accuracy, final test accuracy, and probe optimizer steps. Sequential training logs loss, accuracy, learning rate, cumulative optimizer steps, and examples processed.

Training uses random crop with padding, horizontal flips, and CIFAR-100 normalization. Evaluation uses fixed normalization only. AMP is enabled only on CUDA and can be disabled in the YAML config. `--device cpu` is available for small smoke checks, but full training is intended for GPU execution.

## Izar / SLURM

Transfer this directory to a location available on Izar, activate the verified PyTorch environment, edit the YAML paths, and run from this directory. The batch scripts are generic SLURM examples; adjust time, memory, GPU request, and any site-specific account or partition directives to local policy.

```bash
mkdir -p logs
sbatch slurm/train.sh
# After training completes:
sbatch slurm/evaluate.sh
```

For three independent runs, set `experiment.seeds: [17, 29, 41]` in YAML. Training processes seeds serially per job. Evaluation is a separate job and refuses to run if one of the expected checkpoints is missing. To execute the baseline control:

```bash
python -m src.train_baselines --config configs/pilot.yaml
python -m src.evaluate_probes --config configs/pilot.yaml \
  --checkpoint checkpoints/seed_17/baselines/last_task_only.pt
```

Set corresponding baseline checkpoint paths for all seeds when using multiple seeds. Results are appended under `results/` as CSV and JSONL trajectories. Run plotting after evaluations complete; figures are written to `results/figures/` (transfer accuracy, gain over random initialization, learning trajectories, optional representation diagnostics, and optional task-4-only comparison).

No GPU job is started by these scripts automatically.
