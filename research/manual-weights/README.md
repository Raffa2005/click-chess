# Hand-written weights: measured results

The manually authored ReLU network scored **74.73% online** and **72.93% classical**, compared with **85.13%** and **82.39%** for the trained INT8 model. It is much smaller, but this single hand-designed attempt did not match the trained model’s accuracy. No weights were revised after scoring.

## Accuracy

Both players are rated at least 2100. Only destinations with more than one legal origin count; promotion-piece choice stays separate. All models score exactly the same examples.

| Benchmark | Decisions / games with ambiguity | Manual ReLU | Manual linear ablation | Trained INT8 | Random legal origin (expected) |
|---|---:|---:|---:|---:|---:|
| Lichess Elite, May 2023 | 125,578 / 3,264 | 74.73% | 71.92% | 85.13% | 44.46% |
| Prague + Cyprus, 2025 | 5,888 / 134 | 72.93% | 70.31% | 82.39% | 44.19% |

| Benchmark | Manual top two | Trained top two | Manual mean clicks | Trained mean clicks | Manual NLL | Trained NLL |
|---|---:|---:|---:|---:|---:|---:|
| Online | 97.24% | 98.87% | 1.283 | 1.161 | 0.585 | 0.334 |
| Classical | 96.38% | 98.23% | 1.310 | 1.195 | 0.620 | 0.387 |

Game-cluster paired bootstrap (2,000 resamples), manual minus trained:

- Online: **-10.40 percentage points** (95% interval -10.67 to -10.14).
- Classical: **-9.46 percentage points** (95% interval -10.78 to -8.07).

The linear ablation scored 71.92% / 70.31%; the eleven nonlinear chess preferences added about 2.8 / 2.6 percentage points in this fixed comparison. The simplest least-valuable-piece rule scored 52.00% / 48.54%. Manual softmax probabilities have no fitted calibration; NLL is descriptive.

## Exactly what was handwritten

The primary network has **147 inputs, 13 ReLU hidden units, and one score per legal origin**. All weights and biases are explicitly specified in [manual_weights.py](../../src/tinychess/manual_weights.py). There were zero training steps, coefficient searches or copied trained coefficients. It uses the same existing chess features and two-ply history as the trained model.

Begin with the linear chess score `z = sum(a[j] * x[j])`. We encode that signed score with two neurons using `ReLU(z) - ReLU(-z) = z`. Eleven further ReLU units contribute manually chosen opening development, king-activity and pawn-advance preferences. Other explicit coefficients reward captures, castling and checks, and penalize exposed valuable pieces. Candidate softmax converts scores to an origin ranking. This is a hand-coded chess heuristic expressed as a neural network.

| Model | Dense parameters | Nonzero parameters | Raw download | Gzipped download |
|---|---:|---:|---:|---:|
| Manual ReLU (float JSON) | 1,938 | 88 | 9,049 B | 632 B |
| Manual linear (float JSON) | 148 | 18 | 1,349 B | 357 B |
| Trained INT8 (binary) | 17,345 | — | 18,619 B | 16,755 B |

The primary manual model contains 88 nonzero weights/biases and 1,850 explicit zeros. Its dense float32 parameter storage is 7,752 bytes. The small gzip download reflects this sparsity; the benchmark uses the ordinary dense runtime, not a specialized sparse scorer.

## Latency

Warm Node.js v24.22.1 on Linux x64, 305 shared positions, 3,000 iterations per model/path, with model order rotated each iteration. CPU model was unavailable in the sandbox. All figures below are p95 milliseconds.

| Path | Manual ReLU | Manual linear | Trained INT8 |
|---|---:|---:|---:|
| Network only; features supplied | 0.01934 | 0.00490 | 0.25716 |
| Features + ranking; legal moves supplied | 0.24317 | 0.16457 | 0.56268 |
| chess.js legal generation + ranking | 6.73454 | 6.50440 | 6.81190 |
| Cached repeat click (10,000 iterations) | 0.00012 | 0.00012 | 0.00012 |

Ranking was faster for the small manual network. Including chess.js legal generation, the measured paths were both about 6.8 ms; the small difference is not a demonstrated system-level speedup. Cached repeats are equally cheap. These exclude UI rendering, cold start and Android storage; they are not phone latency measurements.

## Reproducibility and limits

Weights were frozen and [published in commit 240eaf6](https://github.com/Raffa2005/click-chess/commit/240eaf64f3881f01f8bce8d0a4e964e25e4e9d85) before this experiment scored labels. [frozen.json](frozen.json) records the coefficients, rationale and file hashes. [protocol.md](protocol.md) specifies the comparison.

The archive and selected-classical PGN hashes, example counts and trained-model accuracies exactly reproduce the previous round-2 report. These were previously held-out games, but their aggregate results are now public, so this is a regression benchmark, not a newly unseen generalization test. It evaluates one handwritten design and its predeclared linear ablation; it does not establish a limit on what another manual design could achieve.

Validation: 20 Python tests; both manual models agree with Python reference scores/rankings on all 305 JavaScript fixtures (maximum score discrepancy 0.00000191); existing feature, binary, experimental-architecture and deployed-model checks also pass. Special moves, black orientation, legality and cycling are covered.

```sh
pip install -e .
npm ci --ignore-scripts
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m tinychess.manual_prepare
python -m tinychess.manual_evaluate --output research/manual-weights/reproduced-evaluation.json
python -m unittest discover -s tests -v
npm test
npm run benchmark:manual
```

The published artifacts are already frozen; do not rerun `manual_weights freeze` over them. The evaluator verifies the frozen hashes and refuses to overwrite its output. Preparation obtains only the three pinned evaluation archives, not training data. Online data derives from CC0 Lichess games; classical broadcasts are CC BY-SA 4.0, with URLs and hashes in [manual_prepare.py](../../src/tinychess/manual_prepare.py). Raw games/caches remain excluded from git.

Full metrics: [evaluation.json](evaluation.json), [latency.json](latency.json). The deployed model and Android app were not changed.
