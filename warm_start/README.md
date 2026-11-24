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

Submit to Izar from the repository root with:

```bash
sbatch warm_start/run_figure1_izar.sh
```

The job uses the repository-root `.venv` by default. Set `VENV` if the cluster
environment lives elsewhere.
