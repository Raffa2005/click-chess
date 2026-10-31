# Round 2: more data, candidate comparisons, and compact export

The [protocol](protocol.md) preserves the February validation contract. This
study contains eight CPU training runs and six fixed weight-average candidates.
The previous deployed model and results are preserved under [baseline/](baseline/).
See [RESULTS.md](RESULTS.md) for the measurements.

Experiments include pooled input features, learned mean/max candidate-set
context, continued training, and wider/deeper heads initialized from the existing
model. December adds 400,762 examples after validation exclusions, giving
1,227,149 training decisions. Test data never enters training.

`experiments.json`, `results.tsv`, `runs/`, and the logs retain successful and
unsuccessful trials. The new classical allowlist is Prague Masters/Challengers
and the Cyprus Women's Grand Prix, with tiebreaks excluded. May 2023 online games
form the fresh temporal test. Previous April/Tata Steel tests are regression
checks only, evaluated after the new freeze.

The lowest-NLL model and the cheaper existing architecture were compared on
validation before testing. `selection.json` records the paired game-bootstrap
comparison and the decision to prefer the smaller model when the interval
includes zero. This implements the original protocol's preference for cheaper
models when results are effectively tied. `frozen.json` records all export
options and hashes. `adoption.json` applies the predeclared deployment rule.

## Reproduce

Install with `pip install -e .`; use one BLAS thread. First reproduce the earlier
February/January caches following [../README.md](../README.md), then:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m tinychess.round2_prepare
python -m tinychess.round2_bigdata --prepare-only
```

Wait for preparation to finish before starting parallel jobs. The larger feature
array uses a shared memory-mapped cache; preparation is locked and published
atomically. Allow several GB of RAM and disk. A stale `.busy`/`.building` marker
from a forcibly killed process must be removed only after confirming that its
preparation job has ended. `infrastructure-attempts.json` records an initial
preparation concurrency failure and its resolution; no trained result was hidden.

Use a disposable checkout to rerun the published plan. Preserve its results first:

```sh
mkdir -p research/round2/published
mv research/round2/runs research/round2/results.tsv research/round2/frozen.json research/round2/final-evaluation.json research/round2/adoption.json research/round2/published/
mkdir research/round2/runs
python -m tinychess.round2_loop --max-runs 8 --max-minutes 90 --blends
```

The initial checkpoints are supplied under `baseline/`. The wider initialization
duplicates 16 second-layer units, divides their outgoing weights by duplication
counts, and adds small paired perturbations. The deeper initialization inserts an
exact 32×32 identity layer. Trials allow 240–420 optimization seconds; the three
December comparisons cap epochs at 16. Some hit the time cap first, so these are
not strict equal-update architectural comparisons. Two set-context runs have
their own tested masked gradient. Weight averages interpolate already aligned
neurons; they do not run an ensemble at inference.

To add a new trial, give it a unique name in `experiments.json` and run:

```sh
python -m tinychess.round2_loop --experiments YOUR_NAME --max-runs 1
```

The controller skips completed results and stops on an execution failure. It
does not schedule future work. Source snapshots support the recorded hashes.

Select using validation, freeze, then score the tests exactly once:

```sh
python -m tinychess.round2_evaluate freeze --experiment SELECTED_NAME
python -m tinychess.round2_evaluate evaluate
node js/benchmark.mjs research/round2/candidate/origin-ranker.int8.json research/round2/latency-candidate.json
python -m tinychess.round2_evaluate adopt
python -m tinychess.export_predictions
python -m unittest discover -s tests -v
npm test
```

Promotion requires a top-1 gain on both fresh test sets and p95 ranking below
1 ms. No retraining follows test feedback. Future research needs another unseen
test source; these tests are now public regression sets.

## Binary model format

`CCQ1`, a little-endian uint32 JSON-header length, then the header. Each layer
stores row-major signed INT8 weights, little-endian float32 biases, and one or
one-per-output float32 scale. The header specifies layer shapes and architecture.
Python and JavaScript decoders reject truncation/trailing data. Binary and JSON
exports produce identical predictions; this is lossless packing of the same
quantized model, not extra model compression or retraining. JSON remains available.

```python
model = Ranker('models/origin-ranker.bin')
```

```js
const bytes = await (await fetch('./models/origin-ranker.bin')).arrayBuffer();
const model = OriginRanker.fromBinary(bytes);
```

The `.bin` file works without HTTP compression. The `.bin.gz` file is an optional
compressed artifact; servers must send the appropriate content-encoding header
or clients must decompress it before decoding.
