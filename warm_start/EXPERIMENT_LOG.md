# Warm-start experiment log

This is the running record for experiments related to Figure 1 of Ash and
Adams, [*On Warm-Starting Neural Network Training*](https://arxiv.org/abs/1910.08475).
The paper compares training on half the CIFAR-10 data and then continuing on
all of it against training on all data from a fresh initialization. Its Figure
1 reports similar training performance but worse test performance for the
warm-started model.

## Current status

### Large-data warm-versus-fresh run — completed

The research update records a seed-0 run in our model setting:

- Dimensions: `d = P = P* = 100`; `T = 5`.
- Dataset size: `N = 800,000`; held-out test size: `10,000`.
- Warm-up: `400,000` updates, then `800,000` updates per final-training arm.
- Optimizer: SGD, batch size 1, learning rate `0.01`.
- Reported endpoint MSE: about `0.023` for warm initialization and `0.050`
  for fresh initialization.

The warm-initialized arm has lower training and test loss in this run. The
training errors differ, so this is not a matched-fit comparison. The plot and
slide are in `../research_update_beamer/Figures/rq3_warm_start_losses.png` and
`../research_update_beamer/main.tex`.

### CIFAR-10 Figure 1 reproduction — protocol implemented; completion unverified

`replicate_figure1.py` implements a five-seed reproduction with a held-out
third of CIFAR-10's training set. The warm arm trains for 350 epochs on half
the remaining fit data, then 350 epochs on all fit data. The cold arm trains
on all fit data for 700 epochs. It uses batch size 128, cross-entropy, no
augmentation or learning-rate schedule, and Adam at `0.001` (the paper's
Figure 1 caption does not state the optimizer; Adam follows the authors'
released implementation). Adam state is reset at the warm transition.

The code and run script are present, but `warm_start/outputs/` is absent from
the available workspace. Treat the reproduction as **not verified complete**
until its curves, summary, or checkpoints are available.

### Seed-0 re-warming continuations — protocols implemented; completion unverified

Two planned continuations start from the seed-0 half-data checkpoint and train
on the full fit split for 700 epochs with a cosine learning-rate schedule from
`1e-2` to `1e-3`:

1. `lr_rewarm`: changes the learning-rate schedule only.
2. `lr_rewarm_project`: uses the same schedule and projects convolutional and
   linear weight tensors to their seed-0 initialization norms after each
   optimizer update. BatchNorm parameters and buffers are not projected.

The README describes these runs as adapting ELR re-warming; they are not the
paper's exact cyclic schedule. Their local output directories and checkpoint
prerequisite are absent, so completion and results are not yet verified here.

## Next experiment

The research update specifies a repeated-pass experiment with `N = 10,000`
(5,000 in the first-stage subset), 400,000 warm-up updates, and 800,000 final
updates per arm. It is marked as pending in the slide deck. Record results
here when runs and metrics are available.

## Planned next phase: long-run convergence and loss tracking

Requested on 2026-11-25. The goal is to test whether warm initialization,
cold initialization, and warm initialization with the LR re-warming schedule
approach the same generalization performance after substantially longer
training, across seeds, while tracking their loss trajectories.

### Proposed comparison protocol

- Run seeds `0–4`, keeping each seed's train/held-out split paired across all
  three arms.
- Continue each arm to a common target of 3,500 full-data epochs, and extend
  further if held-out metrics are still changing at that point. The warm
  baseline endpoint has 350 full-data epochs (plus 350 half-data epochs); the
  cold endpoint has 700 full-data epochs; the LR-rewarmed endpoint has 700
  full-data epochs after the same 350-epoch half-data warm-up. Thus the
  continuation adds 3,150 epochs to warm and 2,800 to cold and LR-rewarmed.
- For the LR-decay arm, retain its existing cosine schedule during its first
  700 full-data epochs, ending at `1e-3`; continue the added phase at `1e-3`.
  Continue the standard warm and cold arms at their existing `1e-3` rate.
- Reset Adam state at the start of the continuation, as the existing saved
  model checkpoints do not include optimizer state. If resumable continuation
  checkpoints are produced, save optimizer state in those checkpoints.
- Log each epoch's mean training cross-entropy, held-out cross-entropy,
  training accuracy, held-out accuracy, and learning rate. Save periodic model
  checkpoints so a long run can resume without losing its trajectory.
- Compare per-seed held-out accuracy and loss at common training horizons,
  plus the mean and standard deviation across seeds. Accuracy convergence
  should be judged alongside held-out loss, since similar accuracy can conceal
  different confidence/calibration behavior. Loss curves describe optimization
  trajectories; by themselves they do not map the full loss landscape.

### Status and prerequisites

This is a protocol, not a completed run. A resumable runner is now available
as `continue_long_run.py`; it writes per-epoch loss/accuracy curves and
periodic checkpoints with optimizer state. The re-warming script now accepts
`--seed`, so the LR-decay endpoint can be generated for seeds 0–4 from their
`warm_after_half.pt` checkpoints. Example for seed 1:

```bash
python warm_start/continue_seed0_rewarming.py \
  --seed 1 --variant lr_rewarm \
  --output-dir warm_start/outputs/seed_01/lr_rewarm
```

On Izar, `run_lr_rewarm_array_izar.sh` submits the LR-decay run for seeds 0–4
as an array. Once the endpoints exist, `run_long_run_izar.sh` submits the
resumable long-run comparison.

Once the warm and cold Figure 1 endpoints and the LR-rewarmed endpoint exist
for each seed, continue them to the proposed horizon with:

```bash
python warm_start/continue_long_run.py \
  --seeds 0 1 2 3 4 --target-full-data-epochs 3500
```

The CIFAR-10 outputs/checkpoints are not present in the current workspace, so
the runner has not been launched here. Retrieve the warm and cold endpoint
checkpoints for all five seeds and produce the matching LR-decay endpoints
first. If those artifacts cannot be retrieved, rerun the three arms as a fresh
matched experiment. Add run IDs, checkpoint paths, and actual metrics here as
they become available.

## Deferred: architecture comparisons

After the long-run and loss-trajectory comparison, test whether the observed
behavior persists with a smaller transformer, a linear network, or a reduced
ResNet. Keep the CIFAR-10 data split and training/evaluation protocol fixed
where the architecture permits, and compare warm, cold, and LR-decay arms.
Architecture runs have not started; specify each model and parameter count
before adding results.

## Update convention

For each new run, add its date or run identifier, question, protocol changes,
completion state, key metrics, and paths to the raw outputs or plots. Keep
planned or code-only experiments distinct from runs with inspectable results.
