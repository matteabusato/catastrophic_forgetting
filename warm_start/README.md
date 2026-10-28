# Warm starts in centered self-attention

This experiment asks whether training history can increase held-out prediction
error even when models fit the same final training dataset comparably well.
It implements a synthetic analogue of Figure 1 in Ash and Adams,
[On Warm-Starting Neural Network Training (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/file/288cd2567953f06e460a33951f55daaf-Paper.pdf).
The supplied defaults are a pilot configuration, not a known regime exhibiting
the effect and not a reproduction of the paper's ResNet/CIFAR experiment.

## Model and expectation

For one sequence, X has shape (T,d) and W has shape (P,d). We implement

```
S(W) = W.T @ W / sqrt(P)
L(X,W) = X @ S(W) @ X.T / d - trace(S(W)) * I_T / d
A(X,W) = row_softmax(L(X,W))
y(X,W) = A(X,W) @ X
```

Each token is independently N(0,I_d). Conditional on the current W, the
expectation over X of the uncentered logits is exactly trace(S(W))*I_T/d:
off-diagonal expectations vanish, and diagonal expectations equal trace(S)/d.
We subtract this analytical expectation **before** softmax. This is neither
an empirical batch mean nor an expectation of the softmax. The trace is computed
from the current weights and remains differentiable. Correlated or nonzero-mean
tokens would require a different centering formula.

The final multiplication is by X under the stated (T,d) convention, producing
a (T,d) output. W is transposed relative to `quality_pretraining`, where the
factor has shape (d,P); the resulting function class is the same. Only W is
trained. There are no adapters, biases, separate query/key factors or value
projections. T must be at least two because one-token attention is constant.

The teacher uses the same centered architecture with a frozen, independently
sampled W_star of shape (P_teacher,d). All labels are noiseless. Matching ranks
and scales gives a realizable starting setting; changing P_teacher permits
capacity-mismatch studies. Teacher and student factors start as scaled standard
Gaussians, with scales configurable separately.

## Implemented components and why they are needed

| File in `src/` | Responsibility | Purpose in this experiment |
| --- | --- | --- |
| `model.py` | Student, frozen teacher, centered logits | Implements precisely one shared attention architecture and the analytical expectation correction. |
| `data.py` | Finite Gaussian datasets, teacher outputs and attention, disjoint A/B split | Keeps the task fixed and retains all previously seen examples after dataset expansion. |
| `losses.py` | Normalized output MSE and diagnostic attention KL | Uses one primary objective consistently; KL measures attention mismatch independently of output error. |
| `observables.py` | Batched MSE, KL, student attention entropy, matrix overlap/error, weight norm and displacement | Separates prediction quality from representation and saturation diagnostics without retaining autograd graphs. |
| `training.py` | Shuffled finite-data epochs, SGD/Adam, train-only stopping, histories and timing | Makes optimization budgets and stopping explicit, resets optimizer state per phase and prevents test-driven stopping. |
| `experiment.py` | Reproducible paired protocol, final comparisons and multi-seed aggregation | Isolates initialization history while sharing data, W0 and final-phase minibatch order within each pair. |
| `io_utils.py` | JSON/CSV reports and CPU tensor snapshots | Preserves the actual datasets, split, initial/intermediate/final weights, seeds and settings for auditing. |

`__main__.py` supplies a runnable command-line entry point. Imports are package
relative, so these modules do not import the unrelated `src` packages in the
other experiments. `configs/pilot.json` provides an explicit starting setup.
`tests/test_experiment.py` checks scientific invariants and the paired protocol.

## Controlled comparison

One replicate constructs one teacher, one finite training dataset D and one
independent test dataset. D is randomly partitioned into A and B (50/50 by
default). Two independent copies of the same sampled W0 are used:

1. Warm: train all entries of W on A, then train on the **whole D = A union B**.
2. Fresh: train all entries of W directly on D from W0.

The final phases have identical configurations and use the same shuffled
minibatch order until either run stops. Every `train` call constructs a new
optimizer; Adam moments from A are therefore discarded at the transition.
The pretraining configuration can differ from the shared final configuration.
Examples are fixed: the loop does not generate fresh Gaussian training inputs.

Role-specific RNG seeds are derived deterministically from the replicate seed
using SHA-256 and saved. Model/data generation and shuffling use local generators.
Across replicate seeds the teacher, datasets, initialization and shuffling all
change; within each replicate the comparison is paired. Exact cross-device or
cross-version numerical reproducibility is not guaranteed.

## Objective, stopping and interpretation

The primary objective is

```
output_mse = sum((prediction - target)**2) / (N*T*d)
delta_gen = warm_test_mse - fresh_test_mse
```

Training and evaluation use exactly this normalization. In particular, this
differs from the older experiment's training loss that sums over tokens.
KL is KL(teacher attention || student attention), averaged over examples and
query positions; it is a diagnostic, not the training objective. Attention
entropy is that of the student in nats. The matrix error is
||S-S_star||_F^2 / ||S_star||_F^2, and matrix overlap is Frobenius cosine overlap.
Factor displacement is relative to the original W0, including in the warm
final phase; it is parameterization-dependent. Matrix diagnostics are not
substitutes for held-out prediction error.

In epoch mode, full training loss is checked every `eval_every` epochs and at the final epoch.
Training stops at the configured loss tolerance, an optional plateau condition,
or `max_epochs`. Plateau detection is disabled by default. Its patience counts
evaluation points. A plateau or epoch cap is **not** labelled convergence, and
no validation/test quantity controls stopping or checkpoint selection.

The single-seed scripts below instead use exact optimizer-update budgets with
early stopping disabled. `TrainingConfig.max_steps` overrides the epoch budget,
including a final partial epoch; `log_every_steps` selects update-based logging.
In these histories `epoch` counts completed passes through the phase's dataset.

Each summary reports both final training errors, their absolute difference,
and a configurable comparability flag:

```
abs(warm_train_mse - fresh_train_mse)
    <= training_match_atol
       + training_match_rtol * max(warm_train_mse, fresh_train_mse)
```

It separately reports whether both reached the final loss tolerance. Positive
delta_gen with comparable, sufficiently small training errors is the intended
signal. A positive gap accompanied by substantially different training errors
may reflect incomplete optimization. The code does not automatically declare
the hypothesis confirmed. Ten independent paired replicates are a reasonable
first comparison; increase budgets and check multiple learning rates/sample
sizes before interpreting a persistent gap.

Metrics include initialization (epoch zero), training histories on A, B, D and
test, updates and examples processed. Phase-local epoch/update counters restart
at the warm transition; concatenate pretraining and warm histories with an
offset when plotting the entire trajectory. The summary distinguishes warm
final-phase compute from warm total compute including pretraining. Runtime
excludes diagnostic evaluation and includes synchronization on CUDA/MPS.
It remains sensitive to run order and machine load. A longer fresh run matched
to total compute is a follow-up control, not an additional implemented arm.

Aggregate output gives mean paired delta_gen, sample standard deviation and
standard error, plus counts of positive gaps, comparable training fits and
both-fitted pairs. For one replicate SD and SE are null. These are descriptive
statistics, not a significance test; all pairs are retained in the aggregate.

## Run

Requires Python >=3.9 and PyTorch (tested locally with PyTorch 2.8.0). No new
third-party packages are required. Run from the repository root:

### Single-seed scripts (recommended starting point)

The Python runner and shell launcher live in `warm_start/scripts/`. The launcher
works from any working directory and passes all options through to Python. It
uses `WARM_START_PYTHON` if supplied, then an active virtual environment, then the
repository `.venv`, then `python3`. It does not install dependencies or activate
a cluster-specific environment.

From the repository root:

```bash
bash warm_start/scripts/run_warm_start.sh \
  --seed 0 --pretrain-steps 2000 --train-steps 5000 --log-every 100
```

Equivalent Python entry point:

```bash
python3 warm_start/scripts/run_warm_start.py \
  --seed 0 --pretrain-steps 2000 --train-steps 5000 --log-every 100
```

These defaults train one fixed teacher-labelled task with 64 examples in A,
64 in B and 2048 held-out examples. The same W0 initializes the pretraining-A
and fresh-full trajectories. The warm-full trajectory starts from the post-A
weights. The teacher is frozen throughout all three phases. Each full-data arm
uses exactly 5000 optimizer updates, and the warm arm additionally uses 2000
updates on A. Both full-data arms use all of A union B and identical minibatch
orders. Adjust `--n-train` and `--initial-fraction` to change the split.

Other options include `--d`, `--tokens`, `--rank`, `--teacher-rank`, `--n-test`,
`--init-scale`, `--teacher-scale`, `--batch-size`, `--eval-batch-size`,
`--optimizer`, `--lr`, `--pretrain-lr`, `--dtype`, `--device`, `--threads`,
`--fit-tolerance` and `--output`. Use `--help` for details. For CUDA, use
`--device cuda` in a suitable environment. The generic launcher can be submitted
with site-specific `sbatch` resource flags; it does not hard-code a partition,
account, remote checkout path or virtual environment.

Unless `--output` is supplied, a new timestamped directory under
`warm_start/outputs/` is created. Explicit relative output paths are resolved
against the caller's working directory. The folder contains:

- `config.json`: the requested model, data and training settings, saved before training.
- `metrics.csv`: live, flushed CSV logging for all phases, including step zero,
  every `--log-every` updates and each phase's last update even if off-interval.
- `status.json`: running, complete or failed, with an exception description on failure.
- `summary.json`: the final paired generalization comparison.
- `seed_<seed>/`: the full reports, histories, weights and datasets listed below.

Each live row includes the regime, phase, `phase_step`, `regime_step` (including
the warm arm's A pretraining), and metrics on the active training dataset,
the whole training set, A, B and test. `train_loss` and `train_error` both equal
`train_mse`; `test_loss` and `test_error` both equal `test_mse`. This duplication
is explicit because prediction error and the optimized loss are the same
normalized MSE here. Attention KL, entropy and matrix diagnostics remain separate.
During A pretraining, `train_*` means A; use `full_*` to compare on A union B.
Console output includes current train and held-out loss/error at each log point.

The runner processes exactly one seed. Future repetitions can invoke it with
different `--seed` values and distinct output directories. It does not start a
multi-seed sweep automatically. Fixed update budgets do not guarantee a fit;
check final training errors and extend budgets before interpreting a test gap.

### d=100, kappa=1 single-seed preset

```bash
bash warm_start/scripts/run_warm_start_d100.sh
```

This launcher sets d=100, T=5, P=P_teacher=100, N=20*d^2=200000,
|A|=|B|=100000, n_test=d^2=10000, both initialization standard deviations=1,
SGD with batch size one, lr=0.01 in both phases, log_every=50 and seed=0.
The warm arm takes 100000 updates on A and 200000 on A union B; the fresh
arm takes 200000 updates on A union B from the same W0. This is one shuffled
pass through each phase's fixed dataset, totaling 500000 optimizer updates
across the two histories. This is stochastic gradient descent, not full-batch
gradient descent. Dtype remains float64 and device remains CPU unless overridden.
Trailing options override the preset, for example `--device cuda`.

N=20*d^2 is a proposed high-data starting point, not a proved sample-complexity
threshold. The student factor has P*d=10000 entries, and its symmetric matrix S
has at most d*(d+1)/2=5050 independent entries. Each training example is an
entire five-token sequence with a structured matrix output, so counting samples
against factor entries alone does not determine identifiability/generalization.
The previously observed optimization timescale of order d^2 does not establish
that order d^2 distinct examples are necessary. Treat this run as a high-data
control, with smaller N/d^2 values (e.g. 2, 5, 10) for a later data-size sweep.
Changing N while tying update counts to N changes both data and compute;
separate these factors in a controlled follow-up sweep.

One pass is an explicit compute budget, not evidence of convergence. Inspect
post-A and final full-data training errors before interpreting a test gap.
If A is not fitted, increase warm-up updates independently of |A|. If final
training errors remain high or differ substantially, extend both final budgets.

At these settings the existing exact logging evaluates the entire relevant
datasets every 50 updates, not just the current minibatch. There are 10003
log points across the three phases, and overlapping train/full/A/B diagnostics
are currently evaluated separately. This amounts to roughly 5.9 billion
sequence evaluations, so diagnostic computation will dominate the 500000
training updates. A future fixed diagnostic subset can reduce this cost, but
would need to be labelled explicitly as a subset estimate. Full final metrics
should still be evaluated on the complete datasets. No subset substitution is
made by this preset. Stored full data plus A/B copies and test data occupy about
3.4 GB in float64, before temporary tensors and other runtime allocations.

### Existing configuration-based entry point

The earlier entry point remains available for epoch-based configurations or
multi-seed execution:

```bash
python3 -m warm_start --config warm_start/configs/pilot.json \
  --output warm_start/outputs/pilot --seeds 0 1 2 3 4 5 6 7 8 9
```

The output directory must be new; existing results are not overwritten. CPU
threads default to one for these small matrix workloads; use `--threads` to
change this. To change dimensions, budgets, optimizer, loss tolerance or device,
copy/edit the JSON config. Nested training dictionaries inherit unspecified
`TrainingConfig` defaults. Without `--seeds`, the JSON `seed` (default 0) is used.

For Python use:

```python
from warm_start.src import ExperimentConfig, TrainingConfig, run_experiment

config = ExperimentConfig(
    pretraining=TrainingConfig(max_epochs=500),
    final_training=TrainingConfig(max_epochs=1000),
)
result = run_experiment(config)
print(result.summary)
```

Each `seed_<seed>/` directory contains:

- `config.json`, `metadata.json`, `summary.json`: settings, normalization,
  role-specific seeds, software version, stop reasons and final comparisons.
- `pretraining_history.csv`, `warm_history.csv`, `fresh_history.csv`: learning
  curves and diagnostics, suitable for Figure 1-style training/test plots.
- `weights.pt`: teacher, W0, post-A weights and both final student weights.
- `datasets.pt`: actual fixed inputs, outputs, target attention and A/B indices.

The top-level `aggregate.json` summarizes the paired gaps. Tensor snapshots
are audit/evaluation artifacts; they do not include optimizer state for exact
mid-training resumption. Large datasets will require memory/storage budgeting.

## Verification and next experiments

```bash
python3 -m unittest discover -s warm_start/tests -v
```

Tests cover Monte Carlo agreement with the analytical centering, autograd
through the trace subtraction, exact teacher matching, MSE normalization,
disjoint A/B coverage, evaluation weighting for partial batches, training
independence from test labels, paired reproducibility, optimizer resets,
stopping, output persistence and paired aggregation.

After checking that the fresh baseline can fit D, sweep initial training duration,
N, P/d and learning rate. Keep the test set out of model/hyperparameter selection;
use a separate validation set if you tune rather than report a prespecified sweep.
Plot both final training errors alongside delta_gen. Shrink-and-perturb, label
noise, distribution shifts and adapter-only training are intentionally outside
this initial controlled experiment.
