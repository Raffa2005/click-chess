# Measured results

Run: 2026-10-31. All accuracies below concern destinations with **multiple legal origins**. Promotion-piece selection is separate.

## Selected model

107 candidate features → 64 ReLU units → 1 score, shared across legal origins. 6,977 parameters. Last two plies included. Masked categorical cross entropy; Adam; NumPy on CPU. Selected by novel-validation NLL at epoch 28. No test-set model selection.

INT8 weights with float32 biases: 7,172 bytes, excluding scalar quantization metadata. Deployable JSON: 20,162 bytes, or **7,212 bytes gzipped**. Float32 parameters: 27,908 bytes. INT8 is dequantized once at load; inference uses floating-point arithmetic.

## Held-out accuracy

| Test | Games | Ambiguous origins | INT8 top-1 | INT8 top-2 | Float32 top-1 | Uniform baseline | Least-value baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal_lichess | 3,216 | 123,378 | 82.06% | 98.35% | 82.20% | 44.49% | 51.52% |
| classical_otb | 323 | 14,066 | 80.07% | 97.87% | 80.17% | 44.26% | 50.02% |

INT8 top-1 95% game-bootstrap intervals: temporal_lichess: 81.81–82.31%; classical_otb: 79.29–80.84%.

Top-2 means the observed origin appears in the first two ranked choices, not that the first prediction is correct. Repeated clicks cycle a cached ranking. Mean choices required: temporal_lichess: 1.197; classical_otb: 1.222.

## Novel positions and difficult slices

| Test (float32) | Unseen position+destination | Same piece type | Different piece types | ≥3 legal origins |
|---|---:|---:|---:|---:|
| temporal_lichess | 80.36% (105,505) | 74.04% | 82.52% | 79.53% |
| classical_otb | 77.35% (11,738) | 68.68% | 80.54% | 77.13% |

The lower novel-position scores show how repeated opening positions affect the full-test numbers. No engine evaluation or search is used. Human intent remains uncertain, especially in similar-piece choices. See `results/evaluation.json` for phase slices and NLL.

## Validation comparisons

Training: 376,508 examples from 9,894 games. Model-selection validation: 36,024 novel-position examples from 1,132 games.

| Representation | Hidden units | Parameters | Validation accuracy | Validation NLL | Best epoch | Training seconds |
|---|---:|---:|---:|---:|---:|---:|
| history | 0 | 108 | 67.87% | 0.6651 | 16 | 7.1 |
| no_history | 32 | 3,041 | 80.23% | 0.4354 | 17 | 51.1 |
| history | 16 | 1,745 | 79.41% | 0.4482 | 16 | 20.2 |
| history | 32 | 3,489 | 79.97% | 0.4372 | 17 | 22.6 |
| planes | 32 | 28,961 | 72.03% | 0.5886 | 7 | 62.3 |
| no_history | 64 | 6,081 | 80.50% | 0.4256 | 22 | 47.9 |
| history | 64 | 6,977 | 80.90% | 0.4204 | 28 | 59.9 |

All seven training experiments took 271.2 measured seconds in total (data extraction and I/O excluded). The initial five runs allowed 18 epochs; the 64-unit runs allowed 45 with early stopping. These comparisons combine capacity and training-budget differences, so they are not strict equal-budget ablations.

History helped the wider model modestly. Omitting history from the history-trained model reduces novel-validation accuracy from 80.90% to 76.22%. Use the separately trained `origin-ranker.no-history.json` when only FEN is available. Its reported validation accuracy is 80.50%; the main held-out table evaluates the selected history model.

## Latency

Node v24.22.1, Linux x64, warm process; 105 real/edge-case positions. CPU model unavailable in this environment. No GPU or server call.

| Operation | Median | p95 | p99 |
|---|---:|---:|---:|
| rank_with_features_legal_moves_supplied | 0.091396 ms | 0.295181 ms | 0.645268 ms |
| full_chessjs_legal_generation_and_ranking | 4.302440 ms | 8.721843 ms | 12.737341 ms |
| cached_second_click | 0.000072 ms | 0.000125 ms | 0.000325 ms |

Includes feature extraction where applicable; cached click is just an array lookup. Excludes UI rendering, history reconstruction, network and cold model loading. This is a Node benchmark, not measured browser/mobile latency. chess.js legal-move enumeration dominates total time; reusing the UI’s legal moves avoids that cost.

## Data accounting

| Source | Archive games scanned | Games selected | Plies | Origin ambiguities | Promotion-only ambiguities |
|---|---:|---:|---:|---:|---:|
| feb | 267,507 | 11,034 | 934,714 | 418,765 | 1,438 |
| march | 307,510 | 3,218 | 275,493 | 123,378 | 440 |
| classical | 323 | 323 | 29,324 | 14,066 | 19 |

Some selected games contain no ambiguous-origin examples, so evaluated game counts can be lower. Dates, exact archive SHA-256 hashes and deterministic sampling rules are in the dataset JSON reports. The February and March samples cover their full archives, not prefixes.

Classical evaluation uses these records available in the January/April 2024 broadcast archives (not a claim of a complete tournament collection):

- Tata Steel Masters 2024: 91 games.
- Tata Steel Challengers: 91 games.
- Tata Steel Toptienkamp 2024: 45 games.
- FIDE Candidates 2024: 48 games.
- FIDE Women's Candidates 2024: 48 games.

7 Tata Steel playoff games were explicitly excluded. Both-player rating floor 2100 applies to every corpus.

## Verification and limits

- 10 Python tests: legal destinations, pinned pieces, both-origin en passant, castling, promotions, board/history preservation, color reflection, mask/loss derivative and click cycling.
- 105 Python/JavaScript feature-parity fixtures, plus float32 and INT8 ranking/probability parity. Maximum feature difference: 2.56e-8; probability tolerance: 1e-4.
- Model weights were frozen before temporal and classical test evaluation. Uniform and least-value baselines are deterministic/analytic, not tuned on test data.
- INT8 quantization changes approximately 2–3% of argmax choices but changes aggregate accuracy by only about 0.1 percentage point. Both versions are reported; no quantization tuning was performed on these test results.
- Ratings are Lichess ratings for online games and FIDE ratings for the classical sample; they are not treated as interchangeable.
- Players and opening families can repeat across partitions; game-separated and exact-position-novel checks do not establish generalization to new players or new openings.
- The sample is much smaller than the full Lichess database. It validates a practical small model, not a claim of optimal achievable accuracy.
- No measured accuracy for promotion-piece prediction, Chess960 or other variants. Probabilities have not been separately calibrated.
