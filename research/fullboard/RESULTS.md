# Full-board cross-entropy results

The old input model used compressed, engineered candidate features. A new
optional native backend instead reads complete board histories using externally
pretrained Maia3. Cross-entropy is the primary metric; no accuracy gate is used.
This is a substantial improvement in prediction loss, not proof of reaching
human entropy or a completed Android deployment.

## Final primary test: one position per forward pass

Loss is **nats per ambiguous origin decision**, lower is better. All models are
compared on exactly the same examples within each row. Calibration temperatures
were fitted on separate September 2025 development data and frozen before test
scoring. No model weights were trained or adjusted in this study.

| Held-out games | Decisions / games | Old MLP, calibrated | Full-board 5M INT8 | Larger 23M INT8 reference |
|---|---:|---:|---:|---:|
| October 2025 online | 11,027 / 281 | 0.348612 | 0.185536 | 0.168060 |
| January 2026 Tata Steel classical | 7,605 / 168 | 0.379506 | 0.211387 | 0.190431 |

Online primary verification uses the predeclared modulus-4 game-hash reference
subset of the larger 42,896-decision test sample. Classical includes all eligible
January games (the event continues beyond that monthly archive). Both player
ratings must be at least 2100; Lichess Elite's source filter is stricter. Entire
games are disjoint from development. Published Maia3 training ends in July 2025,
so these dates avoid known training-game overlap. Opening-position overlap with
pretraining remains unknown. See the pinned sources and game manifests.

The 5M improvement over the calibrated old model is:

- Online: -0.163076 nats (95% game-bootstrap CI -0.174081 to -0.151693).
- Classical: -0.168119 nats (95% game-bootstrap CI -0.181476 to -0.154216).

Full confidence intervals and examples absent from development positions are in
[final-crossentropy.json](final-crossentropy.json). This is decision-weighted mean
loss with whole-game bootstrap resampling, not independent-move standard errors.
The smaller model is **0.01748 nats online / 0.02096 classical** above the larger
reference. That is the measured gap to the best model tested here, **not** the gap
to irreducible human uncertainty.

## Entropy floor: unknown

Starting from `L(q) = E[-log q(O|X)]`, add and subtract `log p(O|X)` to obtain
`L(q) = H(O|X) + E KL(p(.|X)||q(.|X))`. Therefore the minimum population loss is
`H(O|X)`. Neither that quantity nor the remaining excess loss is identified by
these experiments. More player/context information would define a different
conditioning variable and potentially a different entropy floor.

An unrestricted lookup table can achieve empirical sample losses of **0.002340
online / 0.001546 classical** at the new full-board inputs. But 94.6% / 92.1% of
those inputs appear only once, automatically contributing zero lookup loss. These
numbers are in-sample memorization minima, not credible estimates of human
entropy. The earlier 147-feature sample audit and its 0.005203 / 0.001355 lookup
minima are preserved in [previous-model-crossentropy.json](previous-model-crossentropy.json).

## What the models see

- **Current app/default:** 147 engineered numbers per candidate origin; shared
  `147 -> 96 -> 32 -> 1` MLP, 17,345 parameters, softmax across legal origins.
  Features summarize piece/move geometry, material, attacks, risk, checks,
  castling and two prior plies. Complete piece placement is absent.
- **New optional backend:** 64 square tokens, each containing 12 piece/color
  indicators for each of eight snapshots (96 values per square), plus rating
  embeddings fixed at 2400 in this study. Each snapshot uses upstream's
  side-to-move orientation; short histories repeat the earliest position.
  Castling/en-passant legality is imposed externally, not directly encoded in
  the pretrained board tokens. Global attention can use full-board relationships;
  this does not guarantee engine-level tactical correctness.

One forward pass predicts all moves. Conditioning on the clicked destination
sums legal promotion variants by origin and normalizes over origins. It performs
no search. The policy can be precomputed on a board change; repeated clicks cache
the ranking. The pretrained models/checkpoints are external work, not hand-written
weights or a network trained from scratch by this project.

## Runtime and deployment limits

Native ONNX Runtime 1.31.1, warm model, one thread, this Linux x86-64 host:

| Model | Parameters in source network | INT8 ONNX bytes | Network p50 / p95, 256 recorded positions |
|---|---:|---:|---:|
| 5M | 5,230,084 | 6,800,689 | 12.7 / 20.1 ms |
| 23M | 22,936,580 | 25,209,757 | 31.2 / 71.9 ms |

The initial 16.7 ms frame aim was **not met** at p95. Two/four threads did not
materially improve it. Full Python ranking on 303 FEN fixtures, including board
encoding and legal generation, had p95 90.1 ms in one run and 40.2 ms in a repeat;
the host exhibited substantial outliers. With policy precomputed, p95 was about
2 ms; a cached repeat click was about 0.0002 ms. Both complete runs and a CPU-time
instrumented repeat are published. Do not substitute network-only timing for
complete input latency or claim physical Android performance from these numbers.

Dynamic INT8 quantization can depend on other positions in the inference batch.
Consequently the final primary table uses **batch size 1**. Four independent
single-position calls run concurrently only to accelerate evaluation; serial
output equivalence is checked. Temperatures and weights remain the original
frozen choices. Earlier batch-size-32/64 reports are retained as diagnostics.

The JavaScript encoder/head/cache tests pass, but ONNX Runtime Web INT8 numerical
parity **failed**: up to 0.0355 absolute probability error versus native on eight
fixtures, with two near-tie promotion rankings changed. No browser loss or Android
latency claim is made. **The current APK/default is unchanged.** The native
Python/ONNX backend is usable; the browser adapter remains experimental.

## Validation and reproduction

- 27 Python unit tests; all existing JavaScript inference/model fixtures pass.
- 34 new JS/Python full-board input/head fixtures, including history, black
  orientation, castling, promotions, cache reuse and repeat clicks.
- Upstream tokenization checked on a legal trajectory; float ONNX/native PyTorch
  origin log-probability parity maximum error 0.0000133 on 128 development inputs.
- INT8 is approximate: its raw development NLL change versus float was +0.000146
  on the matched subset. Quantization is never described as lossless.
- Frozen source, checkpoint and calibration hashes are in `frozen.json`, published
  before final tests in commit `610453543f8f2a2257cdeee3d6723dfcf2866801`.
- The later batch-one rerun corrects execution-path measurement; it did not change
  the selected weights or temperatures using test outcomes.

See [README.md](README.md) for pinned downloads/export and use. Final metrics:

```sh
python -m tinychess.fullboard_single --data research/fullboard/cache/online-test.npz --game-modulus 4 --size 5m --output research/fullboard/online-single-5m.json
python -m tinychess.fullboard_single --data research/fullboard/cache/classical-test.npz --size 5m --output research/fullboard/classical-single-5m.json
# Repeat both commands with --size 23m and corresponding output names.
python -m tinychess.fullboard_report
```

Reports refuse accidental overwriting. The upstream AGPL code/weights are not
bundled or relabeled as this repository's MIT-licensed model training output.
