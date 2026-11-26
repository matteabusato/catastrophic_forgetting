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

The long-run protocol is now being executed. A resumable runner is available
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

The outputs/checkpoints are on Izar and are not present in this local
workspace. The user confirmed the warm, cold, and LR-rewarm endpoint files for
seeds 0–4 on Izar; the LR-rewarm array job was `3265795`. The long-run job
`3265892` was submitted on 2026-11-25 and observed running on node `i51` at
21:40 Zurich time. It targets 3,500 full-data epochs for all five seeds. No
long-run metrics have been reported yet. Its Slurm logs are
`warm-start-long-3265892.out` and `warm-start-long-3265892.err` in the Izar
repository root. Add completion state, output paths, and metrics when they are
available.

### Remaining actions for this phase

1. Monitor job `3265892`; if it reaches the time limit, resubmit
   `sbatch warm_start/run_long_run_izar.sh` to resume from the latest checkpoint.
2. Confirm all 15 seed/method runs reach full-data epoch 3,500. Inspect
   `outputs/long_run/trajectory_summary.csv`, `summary_by_seed.csv`, and
   `summary_across_seeds.csv` after completion.
3. Plot held-out accuracy and train/held-out cross-entropy against full-data
   epochs, with per-seed traces and mean ± standard deviation. Compare paired
   per-seed endpoint differences for warm vs cold and LR-rewarmed vs cold.
4. Before calling the arms converged, check that the held-out accuracy gap is
   small across seeds and that held-out accuracy/loss have plateaued near the
   final horizon. If they have not plateaued, extend the same runs to a longer
   common horizon using the saved checkpoints.
5. Add the plots, endpoint values, across-seed spread, and conclusion here.

## Proposed architecture experiment: reduced-depth ResNet

The first architecture experiment will test only the original warm-start and
cold-start arms, one seed at a time. The LR-decay arm and other architectures
are out of scope for this first pass.

### Model and question

Use the same CIFAR ResNet stem, BasicBlock definition, stage widths
`16/32/64/128`, downsampling shortcuts, BatchNorm, and classifier as the
current CIFAR ResNet-18 reproduction. Reduce depth from two BasicBlocks per
stage (`[2,2,2,2]`) to one per stage (`[1,1,1,1]`). This keeps the residual and
convolutional structure while reducing depth and capacity. Record the exact
parameter count from the instantiated model in the run metadata.

Question: does the warm-versus-cold generalization gap remain with a smaller,
shallower residual network?

### Matched protocol

- Start with seed `0`, using the same seed-0 split as the ResNet-18 runs: a
  random one-third held out, with half of the remaining fit set used for
  warm-start training.
- Warm arm: 350 epochs on half the fit set, then 350 epochs on the full fit
  set, resetting Adam state at the transition.
- Cold arm: fresh initialization and **350 epochs** on the full fit set.
- Keep the current settings: batch size 128, Adam at `1e-3`, cross-entropy,
  weight decay 0, channel normalization mean/std `0.5`, no augmentation, and
  no learning-rate schedule.
- Use paired train/held-out examples and matching data-loader seeds across
  warm and cold arms. Use the same initialization seed offsets as the current
  protocol where tensor shapes allow.
- Log per-epoch training and held-out cross-entropy and accuracy, epoch and
  wall-clock time. Save the warm transition checkpoint and both final
  checkpoints. Outputs go under `warm_start/outputs/small_resnet/seed_XX/`.

This intentionally uses 350 cold full-data epochs, as requested. The warm arm
also has its 350-epoch half-data phase, so the arms do not have equal total
training examples or optimizer updates; report those counts and wall-clock
times alongside generalization results.

### Decision after seed 0

Compare endpoint held-out accuracy and loss, the full learning curves, and
training accuracy/loss to check whether both arms optimized adequately. If the
warm-versus-cold behavior appears and the model fits the training set, repeat
with the remaining seeds one at a time before making an across-seed claim. If
both arms underfit or fail to optimize, label any architecture-specific tuning
as a separate follow-up rather than changing this matched first pass.

The self-contained runner `run_small_resnet_experiment.py` and Izar job script
`run_small_resnet_izar.sh` are implemented. Seed-0 job `3265916` completed
successfully on 2026-11-26 after 2:44:52. Its logs are
`warm-start-small-resnet-3265916.out` and
`warm-start-small-resnet-3265916.err` in the Izar repository root. Results are
under `warm_start/outputs/small_resnet/seed_00/`:

- Parameter count: `308,826`.
- Warm held-out accuracy/loss: `73.69%` / `2.220`.
- Cold held-out accuracy/loss: `75.23%` / `1.899`.
- Warm minus cold: `-1.54` percentage points accuracy and `+0.321` held-out
  cross-entropy.
- Both arms reached 100% training accuracy with near-zero final training
  cross-entropy. The cold held-out accuracy was still fluctuating slightly in
  the final epochs.

The recorded held-out metric uses the reserved third of CIFAR-10's training
set; this run did not evaluate on the official CIFAR-10 test split. The plotter
`plot_small_resnet.py` produces `accuracy_curves.png` from `curves.csv`.
Submit other seeds one at a time with `SEED=1 sbatch
warm_start/run_small_resnet_izar.sh`, changing the seed as needed. Later
candidates remain a linear
softmax/factorized-linear control, a small tanh MLP, and a small Vision
Transformer; do not include them in this first experiment.

## Update convention

For each new run, add its date or run identifier, question, protocol changes,
completion state, key metrics, and paths to the raw outputs or plots. Keep
planned or code-only experiments distinct from runs with inspectable results.
