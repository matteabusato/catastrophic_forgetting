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
Regenerate the plot from existing curves without rerunning training with
`python warm_start/replicate_figure1.py --plot-only`. In this plot, the cold
700-epoch trajectory is shifted to start at epoch 350 and therefore ends at
epoch 1,050.

## ELR re-warming continuations

Each variant starts from `outputs/seed_XX/warm_after_half.pt`, the saved
model after the first 350 half-data epochs. It then trains on that seed's
full-data fit split for 700 epochs with a fresh Adam state. The LR jumps from
the baseline `1e-3` to
`1e-2` at the transition and follows a cosine decay to `1e-3` over those 700
epochs. This one-shot schedule adapts the ELR re-warming idea to our longer
Ash–Adams run; it is not the exact cyclic schedule or 70+70 duration used in
the later paper.

`lr_rewarm` changes the LR only. `lr_rewarm_project` uses the same schedule
and, after every optimizer update, projects each convolutional and linear
weight tensor to its norm at that seed's random initialization. The latter tests
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
The Python runner also accepts `--seed`; use a seed-specific output directory
to run the same variant for other seeds. The five-seed array job is:

```bash
sbatch warm_start/run_lr_rewarm_array_izar.sh
```

After both jobs finish, make the combined seed-0 training/test accuracy figure
with:

```bash
python warm_start/plot_seed0_rewarming.py
```

The figure is saved as `warm_start/outputs/seed_00/seed0_warm_vs_warm_lr_vs_cold.png`.
It shows the shared 50%-data pretraining segment, then the warm baseline and
LR-scheduled continuations alongside the cold-start curve through epoch 700.

Submit to Izar from the repository root with:

```bash
sbatch warm_start/run_figure1_izar.sh
```

The job uses the repository-root `.venv` by default. Set `VENV` if the cluster
environment lives elsewhere.

## Long-run convergence and loss tracking

After the warm, cold, and LR-rewarmed endpoints exist for each seed, continue
the arms to a shared 3,500 full-data epochs. The warm baseline has completed
350 full-data epochs; cold and LR-rewarmed have completed 700:

```bash
python warm_start/continue_long_run.py \
    --seeds 0 1 2 3 4 \
    --target-full-data-epochs 3500
```

On Izar, submit the equivalent resumable job with:

```bash
sbatch warm_start/run_long_run_izar.sh
```

The runner records per-epoch training and held-out cross-entropy and accuracy,
plus the learning rate. It writes per-seed curves, endpoint summaries, and a
mean/std trajectory summary across the selected seeds. It saves resumable
checkpoints every 50 epochs under
`warm_start/outputs/long_run/`, including optimizer state. The LR-rewarmed arm
continues at its final `1e-3` learning rate; warm and cold arms also continue at
`1e-3`. See `EXPERIMENT_LOG.md` for study status and artifact prerequisites.

## Reduced-depth ResNet experiment

The self-contained `run_small_resnet_experiment.py` compares only warm and
cold starts for a CIFAR ResNet with one BasicBlock per stage (`[1,1,1,1]`),
keeping the existing channels and training recipe. Warm uses 350 epochs on
half the fit data and 350 on all fit data; cold uses 350 epochs on all fit
data. The script logs per-epoch train/held-out loss and accuracy, run
configuration, and endpoint summaries under `warm_start/outputs/small_resnet/`.

Submit seed 0 on Izar with:

```bash
sbatch warm_start/run_small_resnet_izar.sh
```

Run subsequent seeds one at a time by setting `SEED`:

```bash
SEED=1 sbatch warm_start/run_small_resnet_izar.sh
```

After the job completes, plot the training and held-out accuracy curves with:

```bash
python warm_start/plot_small_resnet.py
```

The plot is saved as
`warm_start/outputs/small_resnet/seed_00/accuracy_curves.png` by default.
The cold curve is shifted to begin at epoch 350 for phase alignment with the
warm arm's full-data phase; this is a plot alignment, not extra cold training.
