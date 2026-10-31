"""Render the published measurements; this script does not train or score models."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
read = lambda p: json.loads((ROOT / p).read_text())
pct = lambda x: f'{100*x:.2f}%'


def main():
    final = read('research/final-evaluation.json')
    frozen = final['frozen']
    runs = {p.stem: json.loads(p.read_text()) for p in (ROOT / 'research/runs').glob('*.json')}
    plan = read('research/experiments.json')
    winner = runs[frozen['selected_experiment']]
    artifact = read('models/origin-ranker.json')
    latency = read('results/latency-node.json')
    export = frozen['selected_quantization']
    text = f'''# Measured results: click-chess

Run: 2026-10-31. All accuracies concern destinations with **multiple legal origins**.
Promotion-piece selection remains a separate UI choice.

## Deployment model

147 candidate features → {' → '.join(map(str, artifact['hidden']))} ReLU units → 1 shared score.
**{artifact['parameter_count']:,} parameters**, including biases. The previous two plies are included.
The selected experiment is `{frozen['selected_experiment']}`; selection used the fixed
36,024-example novel-position February validation set, not either fresh test set.

The deployment model is **{export['gzip_bytes']:,} bytes gzipped** ({export['json_bytes']:,} bytes
as JSON), using {export['quantization']}. Weights are dequantized once at load;
inference uses floating-point arithmetic. No engine search or inference server.
Float32 and INT8 exports are both supplied. Validation accuracy is
{pct(frozen['float32_validation']['accuracy'])} for float32 and {pct(export['validation']['accuracy'])} for INT8.

## Fresh held-out comparison

Both versions are measured on identical examples. Baseline means the original
6,977-parameter INT8 model, preserved in `research/baseline/`. Models and export
choices were frozen before either fresh test was scored.

| Test | Games | Ambiguous examples | Baseline top-1 | Improved top-1 | Improved top-2 | Error reduction |
|---|---:|---:|---:|---:|---:|---:|
'''
    labels = {'fresh_april_2023': 'Lichess Elite, April 2023', 'fresh_classical_2025': 'Tata Steel classical, 2025'}
    for name, r in final['test_sets'].items():
        old, new = r['models']['baseline_int8'], r['models']['improved_int8']
        text += f"| {labels[name]} | {new['games']:,} | {new['examples']:,} | {pct(old['accuracy'])} | **{pct(new['accuracy'])}** | {pct(new['top2'])} | {pct(r['relative_error_reduction'])} |\n"
    text += '\nPaired accuracy gains, with 95% intervals from bootstrapping whole games:\n\n'
    for name, r in final['test_sets'].items():
        lo, hi = r['paired_gain_95pct_game_bootstrap']
        text += f"- {labels[name]}: +{100*r['paired_accuracy_gain']:.2f} percentage points (95% CI {100*lo:.2f}–{100*hi:.2f}).\n"
    text += '''
Top-2 means the played origin appears among the first two choices. It does not
mean the first click was correct. Repeated clicks cycle the cached ranking.

| Test | Novel examples | Baseline novel top-1 | Improved novel top-1 | Float32 full top-1 | Mean choices, INT8 |
|---|---:|---:|---:|---:|---:|
'''
    for name, r in final['test_sets'].items():
        old, new = r['novel']['baseline_int8'], r['novel']['improved_int8']
        text += f"| {labels[name]} | {new['examples']:,} | {pct(old['accuracy'])} | {pct(new['accuracy'])} | {pct(r['models']['improved_float32']['accuracy'])} | {r['models']['improved_int8']['mean_clicks_if_cycling']:.3f} |\n"
    text += '''
Novel means position+destination absent from **both** February development data
and January extra training. Identical full move-sequence games are excluded
across development and tests. Players and opening families can still overlap.

The classical sample contains the 84 Masters and 84 Challengers records available
in the January/February 2025 broadcast archives. Three tiebreak games were
excluded. It is not a claim to contain every game of the tournaments. Both-player
rating floor 2100 applies; Lichess and FIDE ratings are not interchangeable.

## Research trials

The original baseline was reproduced exactly. The immutable novel validation set
contains 36,024 examples from 1,132 games. February training has 376,508 examples;
combined January/February training has 826,387. January excludes all February
validation games and position+destination keys before training. No classical
games enter training.

| Trial | Parameters | Validation top-1 | Validation NLL | Best / total epochs | Seconds |
|---|---:|---:|---:|---:|---:|
'''
    for c in plan:
        r = runs.get(c['name'])
        if not r:
            continue
        if 'validation' not in r:
            text += f"| {c['name']} | — | {r['status']} | — | — | — |\n"
            continue
        text += f"| {c['name']} | {r['parameter_count']:,} | {pct(r['validation']['accuracy'])} | {r['validation']['nll']:.5f} | {r['best_epoch']} / {len(r['curve'])} | {r['seconds']:.1f} |\n"
    total = sum(r.get('seconds', 0) for r in runs.values())
    text += f'''
{len(runs)} trials consumed {total/60:.1f} summed training minutes on CPU, excluding
data preparation/startup. Some ran concurrently, so this is not elapsed wall time.
The ordinary 120-second runs are not equal-update comparisons. Feature ablations
use the same seed, optimizer, architecture, data and 24 epochs. Their parameter
counts include zeroed input rows. The second combined-data run uses a different
seed from scratch; the other combined-data runs use warm starts. This is limited
seed evidence, not a multi-seed significance study.

The controlled comparisons show that board-wide material-risk features provide
most of the gain. Competing-origin context gives a smaller contribution. Width
alone was less useful. The trial log retains unsuccessful alternatives.

![Validation learning curves](research/learning-curves.png)

The best checkpoint is restored by validation NLL. Training loss can continue to
fall after validation stops improving; the plot is evidence about this bounded
search, not proof of a global optimum. Further progress may require better
features, more varied games or distillation rather than simply training longer.

## Latency

Warm {latency['runtime']}, {latency['platform']} {latency['architecture']},
{latency['fixture_count']} real/edge-case fixtures. CPU: {latency['cpu']}.

| Operation | Median | p95 | p99 |
|---|---:|---:|---:|
'''
    for name in ['rank_with_features_legal_moves_supplied', 'full_chessjs_legal_generation_and_ranking', 'cached_second_click']:
        r = latency[name]
        text += f"| {name} | {r['median_us']/1000:.6f} ms | {r['p95_us']/1000:.6f} ms | {r['p99_us']/1000:.6f} ms |\n"
    text += '''
Feature extraction is included in ranking. Legal-move enumeration is measured
separately; reuse the UI's legal list when possible. Cached repeat clicks do not
run the network. Excludes UI rendering, cold model load and history reconstruction.
These are Node measurements, not browser/mobile latency guarantees.

## Verification and limits

- 14 Python tests cover legality, castling, en passant, promotions, history,
  canonical reconstruction, loss gradient, normalization folding and quantization.
- 305 Python/JavaScript fixtures check both feature versions and float32/INT8
  ranking/probability parity, including 200 additional real-game positions. Probability
  tolerance is 1e-4; feature tolerance is 1e-6.
- Quantization is selected using validation alone, allowing at most 0.2 percentage
  points of validation accuracy loss. Frozen hashes are checked before testing.
- Source hashes, trial settings, curves and data manifests are published. The
  source-commit fields identify local experiment commits; exact runner snapshots
  are also retained under `research/source-snapshots/` for hash verification.
- This predicts choices in strong-player games, not an individual's guaranteed
  intent. No measured novice, Chess960, other-variant or promotion-piece accuracy.
- Softmax probabilities are not separately calibrated. Keep corrections easy.
- The old no-history model remains an optional fallback; these new results concern
  the history-enabled model. The original results are archived under
  `research/baseline/RESULTS.md`; they use different tests and must not be compared
  directly with the new test columns.

See [research/README.md](research/README.md) for reproduction and [program.md](program.md)
for the experiment contract. The fresh tests are now public and should serve as
regression sets in subsequent research, with another unseen final test reserved.
'''
    (ROOT / 'RESULTS.md').write_text(text)
    plot(runs, frozen['selected_experiment'])


def plot(runs, selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout='constrained')
    for name, label in [('baseline-recheck', 'Original 107 → 64'), ('rich-deep-96-32', '147 → 96 → 32')]:
        curve = runs[name]['curve']
        axes[0].plot([r['epoch'] for r in curve], [r['nll'] for r in curve], label=label)
    axes[0].set_title('Same February validation, improved features')
    curve = runs[selected]['curve']
    for key, label in [('training_nll', 'Training (epoch average)'), ('nll', 'Validation (checkpoint)')]:
        axes[1].plot([r['epoch'] for r in curve], [r[key] for r in curve], marker='.', label=label)
    axes[1].axvline(runs[selected]['best_epoch'], color='#555555', linestyle=':', label='Selected checkpoint')
    axes[1].set_title('Selected run: ' + selected)
    for ax in axes:
        ax.set_xlabel('Epoch within this run')
        ax.set_ylabel('Cross entropy (lower is better)')
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(ROOT / 'research/learning-curves.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
