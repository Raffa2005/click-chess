"""Render completed measurements; never train or select against test results."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'research/round2'
read=lambda p:json.loads(p.read_text())
pct=lambda value:f'{100*value:.2f}%'

def main():
    report=read(STUDY/'final-evaluation.json');adoption=read(STUDY/'adoption.json');selection=read(STUDY/'selection.json')
    frozen=report['frozen'];latency=read(ROOT/'results/latency-node.json');candidate_latency=read(STUDY/'latency-candidate.json')
    runs={p.stem:read(p) for p in (STUDY/'runs').glob('*.json')};plan=read(STUDY/'experiments.json')
    labels={'fresh_may_2023':'May 2023 online (fresh)','fresh_classical_2025':'Prague/Cyprus 2025 classical (fresh)','april_2023':'April 2023 online (regression)','tata_steel_2025':'Tata Steel 2025 classical (regression)'}
    text='''# Round 2 results: a modest online gain, mixed classical results

The existing **17,345-parameter model remains the default**. Eight new CPU training
runs and six weight averages produced a candidate with better online accuracy,
but it missed the predeclared requirement to improve on both fresh test sets.
The classical point estimate fell slightly; its confidence interval includes zero.
No model was retrained or reselected after these test results.

The useful deployment improvement ships independently: a lossless binary export
of the existing quantized model. Predictions are identical to its JSON export.

## Fresh tests and regression checks

Both columns use identical examples and INT8 deployment models. Fresh tests were
scored only after checkpoint and quantization freeze. Regression sets were
previously visible and were not selection targets.

| Test | Games | Decisions | Current default | Candidate | Gain (percentage points), 95% game-bootstrap CI |
|---|---:|---:|---:|---:|---:|
'''
    for group in ['test_sets','regression_sets']:
        for name,r in report[group].items():
            a,b=r['models']['incumbent_int8'],r['models']['candidate_int8'];lo,hi=r['paired_gain_95pct_game_bootstrap']
            text+=f"| {labels[name]} | {a['games']:,} | {a['examples']:,} | {pct(a['accuracy'])} | {pct(b['accuracy'])} | {100*r['paired_accuracy_gain']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |\n"
    text+='''
Intervals resample whole games (2,000 paired bootstrap replicates). The online
gain is supported by the fresh-test interval; the classical difference is
uncertain. Regression improvements do not override the fresh-test adoption rule.

| Fresh test | Novel decisions | Default novel top-1 | Candidate novel top-1 | Candidate top-2 | Candidate float32 top-1 |
|---|---:|---:|---:|---:|---:|
'''
    for name,r in report['test_sets'].items():
        a,b=r['novel']['incumbent_int8'],r['novel']['candidate_int8']
        text+=f"| {labels[name]} | {a['examples']:,} | {pct(a['accuracy'])} | {pct(b['accuracy'])} | {pct(r['models']['candidate_int8']['top2'])} | {pct(r['models']['candidate_float32']['accuracy'])} |\n"
    text+='''
Novel means position+destination absent from all December/January training and
February development records. Complete game-sequence duplicates are excluded
across development and evaluation. Players/opening families may overlap.
Top-2 is coverage within the first two choices, not first-click accuracy.

The new classical sample comprises 45 Prague Masters, 45 Prague Challengers and
45 Cyprus Women's Grand Prix games; two Prague tiebreaks were excluded. Both-player
rating floor 2100 applies. Online Lichess ratings and over-the-board FIDE ratings
are not interchangeable. Promotion-piece selection and variants remain separate.

## Validation-guided search

Training grew from 826,387 to **1,227,149 ambiguous decisions**. All 400,762 added
December examples exclude every February validation game and position+destination.
The original 36,024-example novel validation contract stayed fixed.

| Trial | Parameters | Validation top-1 | Validation NLL | Best / completed epochs | Seconds |
|---|---:|---:|---:|---:|---:|
'''
    for config in plan:
        r=runs[config['name']];v=r['validation'];epochs=f"{r['best_epoch']} / {len(r['curve'])}" if r['curve'] else 'weight average'
        text+=f"| {config['name']} | {r['parameter_count']:,} | {pct(v['accuracy'])} | {v['nll']:.6f} | {epochs} | {r['seconds']:.1f} |\n"
    total=sum(r['seconds'] for r in runs.values());ci=selection['paired_nll_difference_95pct_game_bootstrap']
    text+=f'''
These runs used {total/60:.1f} summed optimization/scoring minutes, excluding
preparation/startup; some ran concurrently. The December runs allowed 16 epochs
and 420 seconds, and some hit the time cap first. Architectural comparisons are
therefore exploratory, not strict equal-update causal studies. Input pooling,
learned mean/max pooling and weight averages did not win at these budgets.

The deepest candidate achieved the lowest validation NLL (0.353864), versus
0.354136 for the unchanged architecture. Their paired validation loss difference
was {selection['incumbent_minus_deep_validation_nll']:.6f}, with a game-bootstrap
interval [{ci[0]:.6f}, {ci[1]:.6f}]. Applying the existing preference for a cheaper
model when results are effectively tied selected **december-incumbent**, preserving
17,345 parameters. This interval is descriptive after adaptive selection, not a
multiplicity-adjusted hypothesis test. The judgment preceded all fresh tests.

The chosen INT8 candidate had {pct(frozen['selected_quantization']['validation']['accuracy'])}
validation accuracy, versus 83.74% for the previous deployed model. Both per-tensor
and per-output quantization were compared on validation alone. Lowest NLL selected
per-output scaling; test feedback did not change that decision.

![Validation and paired test gains](research-figure.png)

## Deployment size and latency

The default weights remain byte-for-byte unchanged. The new CCQ1 binary format
stores the same INT8 weights, float32 biases and scales as JSON.

| Current default artifact | Bytes |
|---|---:|
| Binary | {adoption['deployment_sizes']['origin-ranker.bin']:,} |
| Binary, gzipped | **{adoption['deployment_sizes']['origin-ranker.bin.gz']:,}** |
| Previous JSON, gzipped | {adoption['deployment_sizes']['origin-ranker.int8.json.gz']:,} |

Binary packing reduces the compressed download by
{100*(1-adoption['deployment_sizes']['origin-ranker.bin.gz']/adoption['deployment_sizes']['origin-ranker.int8.json.gz']):.1f}%
without changing predictions. The candidate is optional under `candidate/`; it is
not installed as the default. Its binary is {frozen['selected_quantization']['binary_bytes']:,}
bytes, or {frozen['selected_quantization']['binary_gzip_bytes']:,} gzipped.

Warm Node {latency['runtime']}, Linux x64, 305 real/edge-case fixtures:

| Operation, current default | Median | p95 |
|---|---:|---:|
'''
    for name in ['rank_with_features_legal_moves_supplied','full_chessjs_legal_generation_and_ranking','cached_second_click']:
        r=latency[name];text+=f"| {name} | {r['median_us']/1000:.6f} ms | {r['p95_us']/1000:.6f} ms |\n"
    text+=f'''
Candidate feature-plus-ranking p95 was {candidate_latency['rank_with_features_legal_moves_supplied']['p95_us']/1000:.3f} ms.
Feature extraction is included; legal-move enumeration is measured separately.
Cached repeat clicks do not rerun inference. Excludes rendering, cold load, network
and history reconstruction. These are Node results, not mobile/browser guarantees.

## Verification and reproducibility

- 18 Python tests, including masked pooled-network gradients, candidate-order
  invariance, quantization and exact binary round trips.
- 305 Python/JavaScript feature and probability fixtures; exact binary/JSON
  prediction parity on all 305. Additional fixtures cover all three experimental
  architectures and both scalar/per-output quantization.
- Frozen hashes and data manifests retained; current default model hash matches
  the archived incumbent. No test-set retuning.
- Two concurrent preparation attempts exited 137 under heavy memory use, and
  another was interrupted before producing epoch results. The completed cache
  hashes were verified, construction was made exclusive/atomic, and configurations
  were restarted. See `infrastructure-attempts.json`.

See [README.md](README.md), [protocol.md](protocol.md), [selection.json](selection.json),
[final-evaluation.json](final-evaluation.json), and [adoption.json](adoption.json).
The prior study is preserved in [baseline/RESULTS.md](baseline/RESULTS.md).
Fresh tests are now public and must become regression sets for later research.
'''
    (STUDY/'RESULTS.md').write_text(text)
    root=text.replace('(research-figure.png)','(research/round2/research-figure.png)')
    for name in ['README.md','protocol.md','selection.json','final-evaluation.json','adoption.json','baseline/RESULTS.md']:
        root=root.replace(f']({name})',f'](research/round2/{name})')
    (ROOT/'RESULTS.md').write_text(root)
    plot(runs,report,labels)

def plot(runs,report,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name,label in [('incumbent-refine','More epochs, same data'),('december-incumbent','More data, same model'),('december-deeper-head','More data, deeper model')]:
        curve=runs[name]['curve'];axes[0].plot([r['seconds'] for r in curve],[r['nll'] for r in curve],label=label)
    axes[0].axhline(.36005795,color='#777777',linestyle=':',label='Previous best')
    axes[0].set(xlabel='Optimization seconds',ylabel='Validation cross entropy',title='Validation improves with more data')
    axes[0].legend(fontsize=8,frameon=False)
    y=0;names=[]
    for group in ['test_sets','regression_sets']:
        for name,r in report[group].items():
            gain=100*r['paired_accuracy_gain'];lo,hi=np_array(r['paired_gain_95pct_game_bootstrap'])*100
            axes[1].errorbar(gain,y,xerr=[[gain-lo],[hi-gain]],fmt='o',capsize=4,color='#206ca4' if group=='test_sets' else '#999999')
            names.append(labels[name]);y+=1
    axes[1].set_yticks(range(y),names,fontsize=8);axes[1].invert_yaxis();axes[1].axvline(0,color='#555555',linestyle=':')
    axes[1].set(xlabel='Accuracy gain, percentage points (95% CI)',title='Fresh classical result remains uncertain')
    for ax in axes:ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.savefig(STUDY/'research-figure.png',dpi=160);plt.close(fig)

def np_array(value):
    import numpy as np
    return np.asarray(value)

if __name__=='__main__':main()
