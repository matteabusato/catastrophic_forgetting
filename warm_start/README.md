# Ash & Adams warm-start reproduction

`replicate_figure1.py` runs the two-stage CIFAR-10 ResNet-18 experiment from
Figure 1 of Ash and Adams, *On Warm-Starting Neural Network Training*.

The fixed protocol is encoded in the script: a random one-third holdout from
the CIFAR-10 training set, equal halves of the remaining fit split, 350 epochs
on the first half followed by 350 epochs on all fit data, and a cold-started
full-data comparison for 700 epochs. It uses batch size 128, Adam at 0.001,
zero weight decay, cross-entropy, channel normalization to mean/std 0.5, and no
augmentation or learning-rate schedule. Adam state is reset between the two
warm-start stages. Five seeds (0–4) are run with paired data splits.

The model, preprocessing, and defaults follow the authors' released PyTorch
implementation. The paper's Figure 1 caption does not identify its optimizer;
Adam is used here because it is the released implementation's default. The
paper also does not publish random seeds. Those choices are recorded to make
this reproduction repeatable.

Run from the repository root after installing compatible PyTorch and
torchvision builds:

```bash
python warm_start/replicate_figure1.py
```

The CIFAR-10 training archive is downloaded under `warm_start/data` by default.
Set `CIFAR10_ROOT` to use another data location. The script writes per-seed
curves and checkpoints, `summary_by_seed.csv`, and (when matplotlib is
installed) `figure1_replication.png` under `warm_start/outputs`.

## Seed-0 ELR re-warming continuations

The two continuation jobs start from
`outputs/seed_00/warm_after_half.pt`, the saved model after the first 350
half-data epochs. Each then trains on the same full-data fit split for 700
epochs with a fresh Adam state. The LR jumps from the baseline `1e-3` to
`1e-2` at the transition and follows a cosine decay to `1e-3` over those 700
epochs. This one-shot schedule adapts the ELR re-warming idea to our longer
Ash–Adams run; it is not the exact cyclic schedule or 70+70 duration used in
the later paper.

`lr_rewarm` changes the LR only. `lr_rewarm_project` uses the same schedule
and, after every optimizer update, projects each convolutional and linear
weight tensor to its norm at seed-0 random initialization. The latter tests
whether controlling norm growth helps the re-warmed LR continue to produce
feature changes. BatchNorm parameters and buffers are not projected.

Submit the jobs independently from the repository root:

```bash
sbatch warm_start/run_seed0_lr_rewarm_izar.sh
sbatch warm_start/run_seed0_lr_rewarm_project_izar.sh
```

They write separate `curves.csv` files and checkpoints under
`outputs/seed_00/lr_rewarm/` and `outputs/seed_00/lr_rewarm_project/`.
Intermediate model checkpoints are saved every 50 full-data epochs.

After both jobs finish, make separate seed-0 training-accuracy and test-
accuracy plots with:

```bash
python warm_start/plot_seed0_rewarming.py
```

The plots are saved in `warm_start/outputs/seed_00/`. Add
`--exclude-projection` to omit the projection run and save separate
`*_no_projection.png` files.

Submit to Izar from the repository root with:

```bash
sbatch warm_start/run_figure1_izar.sh
```

The job uses the repository-root `.venv` by default. Set `VENV` if the cluster
environment lives elsewhere.
