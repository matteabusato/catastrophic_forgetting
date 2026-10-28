import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch

from src.data_model import generate_dataset, sample_teachers, teacher_attention
from src.io_utils import append_csv_metric
from src.losses import attention_cross_entropy, resolve_loss_type, teacher_student_loss
from src.model import Student
from src.observables import attention_metrics, generalization_error
from src.training import finetune_sequential, pretrain

DTYPE = torch.float64
PROJECT = Path(__file__).resolve().parents[1]


def test_attention_ce_matches_entropy_and_detaches_teacher():
    logits = torch.randn(2, 3, 3, dtype=DTYPE, requires_grad=True)
    teacher_logits = torch.randn(2, 3, 3, dtype=DTYPE, requires_grad=True)
    target = teacher_logits.softmax(-1)
    loss = attention_cross_entropy(logits, target)
    loss.backward()
    assert teacher_logits.grad is None
    assert torch.allclose(logits.grad, (logits.detach().softmax(-1) - target.detach()) / 6)
    matched = attention_cross_entropy(teacher_logits, target)
    entropy = -torch.special.xlogy(target, target).sum(-1).mean()
    assert torch.allclose(matched, entropy)
    # Duplicating examples or query rows must not change the normalization.
    assert torch.allclose(loss, attention_cross_entropy(logits.repeat(2, 1, 1), target.repeat(2, 1, 1)))
    assert torch.allclose(loss, attention_cross_entropy(logits.repeat(1, 2, 1), target.repeat(1, 2, 1)))


def test_attention_ce_is_stable_for_extreme_logits():
    logits = torch.tensor([[[1000., -1000.], [-1000., 1000.]]], requires_grad=True)
    loss = attention_cross_entropy(logits, torch.full_like(logits, 0.5))
    loss.backward()
    assert loss.item() == pytest.approx(1000.)
    assert torch.isfinite(logits.grad).all()


def test_matching_attention_and_output_diagnostics():
    teachers = sample_teachers(6, 1., dtype=DTYPE)
    data = generate_dataset(5, 3, 6, teachers.S_star, 7)
    model = Student(6, 1., dtype=DTYPE)
    model.load_large_rank(teachers.W_star)
    assert torch.allclose(model.attention(data.X).sum(-1), torch.ones(5, 3, dtype=DTYPE))
    assert torch.allclose(model(data.X), data.y)
    values = attention_metrics(model, data.X, data.attention_target)
    assert values['attention_kl'] == pytest.approx(0., abs=1e-12)
    assert values['attention_ce'] == pytest.approx(values['attention_entropy'])
    assert generalization_error(model, data.X, data.y).item() == pytest.approx(0.)


def test_ce_pretraining_uses_ce_gradient_and_resume_is_exact():
    torch.manual_seed(10)
    teachers = sample_teachers(6, 1., dtype=DTYPE)
    data = generate_dataset(4, 3, 6, teachers.S_star, 7)
    model = Student(6, 1., dtype=DTYPE)
    initial = model.W.detach().clone()
    generator = torch.Generator().manual_seed(11)
    X = torch.randn(1, 3, 6, generator=generator, dtype=DTYPE)
    expected_loss = teacher_student_loss(model, X, teachers.S_star, 'attention_ce')
    expected_grad = torch.autograd.grad(expected_loss, model.W)[0]
    first = pretrain(model, data, teachers.S_star, 3, 1, .05, [1], 1,
                     seed=11, loss_type='attention_ce')
    assert torch.allclose(model.W, initial - .05 * expected_grad)
    assert model.w.grad is None
    checkpoint = first.checkpoints[-1]
    uninterrupted = Student(6, 1., dtype=DTYPE)
    uninterrupted.load_large_rank(initial)
    full = pretrain(uninterrupted, data, teachers.S_star, 3, 5, .05, [5], 1,
                    seed=11, loss_type='attention_ce')
    resumed = pretrain(model, data, teachers.S_star, 3, 5, .05, [5], 1,
                       start_step=1, optimizer_state=checkpoint.optimizer_state,
                       generator_state=checkpoint.generator_state, loss_type='attention_ce')
    assert torch.equal(full.checkpoints[-1].W, resumed.checkpoints[-1].W)
    assert math.isfinite(resumed.metrics[-1].attention_ce)
    assert math.isfinite(resumed.metrics[-1].output_mse)


def test_sequential_ce_freezes_base_and_records_both_metrics():
    teachers = sample_teachers(6, 1., dtype=DTYPE)
    data = [generate_dataset(8, 3, 6, S, seed) for seed, S in
            enumerate([teachers.S_star, teachers.S1_star, teachers.S2_star])]
    model = Student(6, 1., dtype=DTYPE)
    W = model.W.detach().clone()
    result = finetune_sequential(model, W, data[0], data[1], data[1], data[2], data[2],
                                 teachers, 3, 3, .05, eval_every=1, loss_type='attention_ce')
    assert torch.equal(model.W, W)
    assert model.W.grad is None
    assert model.w.grad is not None and model.w.grad.norm() > 0
    assert result.history[1].stage == 'after_adapter_init'
    for row in result.history:
        assert math.isfinite(row.task1_attention_ce)
        assert math.isfinite(row.task1_error)
    expected = result.attention_metrics_after_task2['task1_attention_ce'] - result.attention_metrics_after_task1['task1_attention_ce']
    assert result.forgetting_task1_attention_ce == pytest.approx(expected)


def test_objective_metadata_and_legacy_csv(tmp_path):
    assert resolve_loss_type(None, {}) == 'output_mse'
    assert resolve_loss_type(None, {'loss_type': 'attention_ce'}) == 'attention_ce'
    with pytest.raises(ValueError, match='Loss mismatch'):
        resolve_loss_type('attention_ce', {})
    with pytest.raises(ValueError, match='reduction'):
        resolve_loss_type(None, {'attention_reduction': 'sum'})
    path = tmp_path / 'metrics.csv'
    path.write_text('step,generalization_error\n0,0.5\n')
    append_csv_metric(path, {'step': 1, 'generalization_error': .4, 'attention_ce': .7},
                       ['step', 'generalization_error', 'attention_ce'])
    rows = list(csv.DictReader(path.open()))
    assert rows[0]['generalization_error'] == '0.5' and rows[0]['attention_ce'] == ''
    assert rows[1]['attention_ce'] == '0.7'


def run_script(script, *args, check=True):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PROJECT), os.environ.get("PYTHONPATH", "")]), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / 'attention-ce-matplotlib'))
    result = subprocess.run([sys.executable, '-m', f'scripts.{script}', *map(str, args)],
                            cwd=PROJECT, env=env, capture_output=True, text=True)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def read_rows(run_dir):
    with (run_dir / 'metrics.csv').open() as stream:
        return list(csv.DictReader(stream))


@pytest.mark.parametrize('objective', ['attention_ce', 'output_mse'])
def test_cli_pipeline_records_ce_and_mse_and_preserves_handoffs(tmp_path, objective):
    root = tmp_path / 'results'
    run_script('run_pretraining_quality', '--results-root', root, '--run-name', 'pilot',
               '--loss', objective, '--d', 6, '--T', 3, '--n-test', 6,
               '--target-steps', 4, '--checkpoint-steps', 0, 4, '--device', 'cpu')
    pre_dir = next(root.glob('pilot*'))
    # Resume inherits the objective; explicitly changing it is rejected.
    run_script('run_pretraining_quality', '--results-root', root, '--run-name', pre_dir.name,
               '--resume', '--target-steps', 6, '--device', 'cpu')
    wrong = 'output_mse' if objective == 'attention_ce' else 'attention_ce'
    rejected = run_script('run_pretraining_quality', '--results-root', root, '--run-name', pre_dir.name,
                           '--resume', '--target-steps', 7, '--loss', wrong, check=False)
    assert rejected.returncode != 0 and 'Loss mismatch' in rejected.stderr
    run_script('run_task1_finetuning', '--pretrain-run-dir', pre_dir, '--pretrain-step', 6,
               '--target-steps', 4, '--n-test', 6, '--eval-every', 1,
               '--results-root', root, '--device', 'cpu')
    task1_dir = next((root / 'finetunetask1_d6_T3').iterdir())
    run_script('run_task2_finetuning', '--pretrain-run-dir', pre_dir, '--pretrain-step', 6,
               '--task1-run-dir', task1_dir, '--task1-step', 4,
               '--target-steps', 4, '--n-test', 6, '--eval-every', 1,
               '--results-root', root, '--device', 'cpu')
    task2_dir = next((root / 'finetunetask2_d6_T3').iterdir())
    for run_dir in [pre_dir, task1_dir, task2_dir]:
        assert json.loads((run_dir / 'config.json').read_text())['loss_type'] == objective
        for row in read_rows(run_dir):
            ce_keys = [k for k in row if k.endswith('attention_ce')]
            mse_keys = [k for k in row if k.endswith('output_mse')]
            assert ce_keys and mse_keys
            assert all(math.isfinite(float(row[k])) for k in ce_keys + mse_keys)
    t1_rows, t2_rows = read_rows(task1_dir), read_rows(task2_dir)
    assert [r['stage'] for r in t1_rows[:2]] == ['before_adapter_init', 'after_adapter_init']
    assert float(t2_rows[0]['task1_attention_ce']) == pytest.approx(float(t1_rows[-1]['task1_attention_ce']))
    assert float(t2_rows[0]['task1_output_mse']) == pytest.approx(float(t1_rows[-1]['task1_output_mse']))
    assert float(t2_rows[0]['forgetting_task1_attention_ce']) == pytest.approx(0.)
    for metric in ['attention_ce', 'output_mse']:
        final = t2_rows[-1]
        assert math.isfinite(float(final[f'task2_{metric}']))
    state = torch.load(task2_dir / 'checkpoints/step_000000004/training_state.pt', weights_only=False)
    assert state['loss_type'] == objective
    # Reload the final model and independently verify the saved metrics.
    from src.io_utils import load_teachers, load_pretraining_checkpoint
    teachers = load_teachers(pre_dir, 'cpu', DTYPE)
    model = Student(6, 1., dtype=DTYPE)
    model.load_large_rank(load_pretraining_checkpoint(pre_dir)['W'])
    model.set_finetuning_mode()
    with torch.no_grad():
        model.w.copy_(torch.load(task2_dir / 'checkpoints/step_000000004/w.pt', weights_only=True))
    config = json.loads((task2_dir / 'config.json').read_text())
    data = generate_dataset(6, 3, 6, teachers.S2_star, config['task2_test_seed'])
    expected = attention_metrics(model, data.X, data.attention_target)
    assert float(t2_rows[-1]['task2_attention_ce']) == pytest.approx(expected['attention_ce'])
    assert float(t2_rows[-1]['task2_output_mse']) == pytest.approx(generalization_error(model, data.X, data.y).item())


def test_plotting_cli_exports_ce_and_mse(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    # Exercise the pre-training and both adaptation CSV formats together.
    runs = []
    for stage, x_key in [('quality_pretraining', 'step_over_d2'),
                          ('finetune_task1', 'finetune_step_over_d'),
                          ('finetune_task2', 'task2_step_over_d')]:
        run = tmp_path / stage
        run.mkdir()
        (run / 'config.json').write_text(json.dumps({'experiment': stage, 'd': 6,
                                                    'loss_type': 'attention_ce'}))
        if stage == 'quality_pretraining':
            keys = ['attention_ce', 'output_mse']
        else:
            keys = [f'{task}_{metric}' for task in ['pretraining', 'task1', 'task2']
                    for metric in ['attention_ce', 'output_mse']]
        with (run / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=[x_key] + keys)
            writer.writeheader()
            for step in range(3):
                writer.writerow({x_key: step, **{k: (0.25 if k.endswith('output_mse') else 1.) / (step + 1) for k in keys}})
        runs.append(run)
    image = tmp_path / 'ce_mse.png'
    run_script('plot_attention_experiment', '--run-dirs', *runs, '--output', image)
    assert image.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    from src.plotting import plot_run_metrics
    import matplotlib.pyplot as plt
    fig, axes = plot_run_metrics(runs)
    for row in axes:
        assert list(row[0].lines[0].get_ydata()) == pytest.approx([1., .5, 1. / 3])
        assert list(row[1].lines[0].get_ydata()) == pytest.approx([.25, .125, 1. / 12])
    plt.close(fig)


@pytest.mark.parametrize('objective', ['attention_ce', 'output_mse'])
def test_task1_resume_matches_uninterrupted_training(tmp_path, objective):
    root = tmp_path / 'results'
    run_script('run_pretraining_quality', '--results-root', root, '--run-name', 'pre',
               '--loss', objective, '--d', 6, '--T', 3, '--n-test', 6,
               '--target-steps', 2, '--device', 'cpu')
    pre = next(root.glob('pre*'))
    for output, endpoint in [(root / 'split', 4), (root / 'whole', 8)]:
        run_script('run_task1_finetuning', '--pretrain-run-dir', pre, '--pretrain-step', 2,
                   '--target-steps', endpoint, '--results-root', output,
                   '--eval-every', 1, '--device', 'cpu')
    split = next((root / 'split/finetunetask1_d6_T3').iterdir())
    whole = next((root / 'whole/finetunetask1_d6_T3').iterdir())
    original_rows = read_rows(split)
    run_script('resume_task1_finetuning', '--run-dir', split,
               '--additional-steps', 4, '--checkpoint-every', 2, '--device', 'cpu')
    resumed_rows = read_rows(split)
    assert resumed_rows[:len(original_rows)] == original_rows
    assert [int(r['finetune_step']) for r in resumed_rows[len(original_rows):]] == [5, 6, 7, 8]
    for filename in ['w.pt', 'training_state.pt', 'metrics.json']:
        assert (split / 'checkpoints/step_000000006' / filename).is_file()
    expected_w = torch.load(whole / 'checkpoints/step_000000008/w.pt', weights_only=True)
    actual_w = torch.load(split / 'checkpoints/step_000000008/w.pt', weights_only=True)
    assert torch.equal(actual_w, expected_w)
    actual_state = torch.load(split / 'checkpoints/step_000000008/training_state.pt', weights_only=False)
    expected_state = torch.load(whole / 'checkpoints/step_000000008/training_state.pt', weights_only=False)
    assert torch.equal(actual_state['generator_state'], expected_state['generator_state'])
    assert actual_state['optimizer_state'] == expected_state['optimizer_state']
    assert resumed_rows[-1] == read_rows(whole)[-1]
    assert json.loads((split / 'config.json').read_text())['target_steps'] == 8
    rejected = run_script('resume_task1_finetuning', '--run-dir', split,
                           '--additional-steps', 0, '--checkpoint-every', 2, check=False)
    assert rejected.returncode != 0
    # Unsaved metric tails must never silently overlap appended continuation rows.
    with (split / 'metrics.csv').open('a', newline='') as stream:
        extra = dict(resumed_rows[-1], finetune_step='9')
        csv.DictWriter(stream, fieldnames=list(resumed_rows[-1])).writerow(extra)
    rejected = run_script('resume_task1_finetuning', '--run-dir', split,
                           '--additional-steps', 2, '--checkpoint-every', 2, check=False)
    assert rejected.returncode != 0 and 'metrics endpoint' in rejected.stderr


@pytest.mark.parametrize('objective', ['attention_ce', 'output_mse'])
def test_task2_resume_preserves_dynamics_and_forgetting_baselines(tmp_path, objective):
    root = tmp_path / 'results'
    run_script('run_pretraining_quality', '--results-root', root, '--run-name', 'pre',
               '--loss', objective, '--d', 6, '--T', 3, '--n-test', 6,
               '--target-steps', 2, '--device', 'cpu')
    pre = next(root.glob('pre*'))
    run_script('run_task1_finetuning', '--pretrain-run-dir', pre, '--pretrain-step', 2,
               '--target-steps', 4, '--results-root', root,
               '--eval-every', 1, '--device', 'cpu')
    task1 = next((root / 'finetunetask1_d6_T3').iterdir())
    for output, endpoint in [(root / 'split', 4), (root / 'whole', 8)]:
        run_script('run_task2_finetuning', '--pretrain-run-dir', pre, '--pretrain-step', 2,
                   '--task1-run-dir', task1, '--task1-step', 4,
                   '--target-steps', endpoint, '--results-root', output,
                   '--eval-every', 1, '--device', 'cpu')
    split = next((root / 'split/finetunetask2_d6_T3').iterdir())
    whole = next((root / 'whole/finetunetask2_d6_T3').iterdir())
    original_rows = read_rows(split)
    original_config = json.loads((split / 'config.json').read_text())
    run_script('resume_task2_finetuning', '--run-dir', split,
               '--additional-steps', 4, '--checkpoint-every', 2, '--device', 'cpu')
    resumed_rows = read_rows(split)
    assert resumed_rows[:len(original_rows)] == original_rows
    assert [int(r['task2_step']) for r in resumed_rows[len(original_rows):]] == [5, 6, 7, 8]
    assert (split / 'checkpoints/step_000000006/w.pt').is_file()
    expected_w = torch.load(whole / 'checkpoints/step_000000008/w.pt', weights_only=True)
    actual_w = torch.load(split / 'checkpoints/step_000000008/w.pt', weights_only=True)
    assert torch.equal(actual_w, expected_w)
    actual_state = torch.load(split / 'checkpoints/step_000000008/training_state.pt', weights_only=False)
    expected_state = torch.load(whole / 'checkpoints/step_000000008/training_state.pt', weights_only=False)
    assert torch.equal(actual_state['generator_state'], expected_state['generator_state'])
    assert actual_state['optimizer_state'] == expected_state['optimizer_state']
    # Every final metric must match, including all absolute/relative forgetting fields.
    assert resumed_rows[-1] == read_rows(whole)[-1]
    new_config = json.loads((split / 'config.json').read_text())
    for key in ['attention_baselines', 'pretraining_error_before_task1',
                'pretraining_error_before_task2', 'task1_error_before_task2']:
        assert new_config[key] == original_config[key]
    assert new_config['target_steps'] == 8
    assert abs(float(resumed_rows[-1]['forgetting_task1_attention_ce'])) > 1e-10
    # No baseline resets: the final forgetting is still relative to the original switch.
    assert float(resumed_rows[-1]['forgetting_task1_attention_ce']) == pytest.approx(
        float(resumed_rows[-1]['task1_attention_ce']) - float(original_rows[0]['task1_attention_ce']))
    with (split / 'metrics.csv').open('a', newline='') as stream:
        extra = dict(resumed_rows[-1], task2_step='9')
        csv.DictWriter(stream, fieldnames=list(resumed_rows[-1])).writerow(extra)
    rejected = run_script('resume_task2_finetuning', '--run-dir', split,
                           '--additional-steps', 2, '--checkpoint-every', 2, check=False)
    assert rejected.returncode != 0 and 'metrics endpoint' in rejected.stderr
