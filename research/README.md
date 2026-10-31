# Bounded CPU research

For the completed follow-up study and current deployment decision, see
[round2/README.md](round2/README.md).

This is an implemented measure/keep/discard project, inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch). It uses an
explicit experiment plan and a small NumPy trainer, rather than an unattended
service. The objective and test-isolation rules are in [program.md](../program.md).

## Inspect the completed study

- [experiments.json](experiments.json): hypotheses, budgets, seeds and settings.
- [results.tsv](results.tsv): append-only keep/discard ledger.
- [runs/](runs/): every completed trial, including training/validation curves.
- [contract.json](contract.json): frozen February data and validation identities.
- [frozen.json](frozen.json): selection and validation-only quantization comparison.
- [final-evaluation.json](final-evaluation.json): paired fresh-test comparison.
- [baseline/](baseline/): original models and results for comparison.
- [../RESULTS.md](../RESULTS.md): interpretation, deployment size and latency.

`keep` means the trial lowered the best validation NLL available at completion.
`discard` models are still recorded; some improve over the original baseline but
lose to a stronger contemporary trial. Deployment size, quantization tolerance,
inference parity and latency are checked before publishing the final winner.

## Reproduce the data

Install the project with `pip install -e .`. Use one BLAS thread:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m tinychess.download
python -m tinychess.data data/raw/lichess_elite_2023-02.zip --output data/processed/feb.npz --modulus 24
python -m tinychess.research_prepare
```

The first downloader also downloads the original test sources. The research
preparer obtains January/April 2023 online archives and January/February 2025
broadcast archives, extracts an explicit classical event allowlist, and builds
features. It does **not** score models on the tests. Source hashes and sampling
counts are retained in the manifests embedded in the run/final reports. The
February-rich cache is checked against the frozen source checksum and validation
index checksum.

Preparation is the expensive part: allow several GB of RAM and a few GB of disk.
Inference has much smaller requirements. The original archive extractor stores
board planes for feature experiments; the extra-data extractor stores only
candidate features. Raw archives and caches are intentionally ignored by git.

## Add an experiment

Add a unique name and hypothesis to `experiments.json`, copying a suitable
configuration. The controller dispatches feature version 1 or 2 appropriately:

```sh
python -m tinychess.autoresearch --experiments YOUR_NEW_NAME --max-runs 1 --max-minutes 10
```

The runner refuses to overwrite completed trials. Checkpoints go into the ignored
`research/checkpoints/` directory. `init` can reference a previous compatible
checkpoint. `extra_training: true` adds January examples while excluding all
February validation game IDs and exact position+destination keys. It also removes
games duplicated across the training sources.

The controller resumes an explicit plan with `--max-runs` and `--max-minutes`.
The outer time limit prevents launching another trial after the limit; a running
trial can finish within its own timeout. It is not a hard process-wide deadline.
Ordinary exploratory trials use 120 seconds. The controlled feature ablations use
24 epochs with a 180-second safety cap; extra-data trials use 240 seconds, followed by a 180-second refinement trial. Budgets
exclude process startup/data loading and include in-training validation. Final
validation/export adds a small overhead to recorded runtime.

Several exploratory jobs ran concurrently, each with one BLAS thread. Thus their
wall-time budgets are not equal numbers of optimization updates. The three
controlled feature runs use identical seeds, architecture, data, optimizer and
epoch counts; inspect those for feature attribution. The second combined-data
run uses a fresh seed and initialization, whereas the warm-start run continues
the best February model. These are not a large multi-seed statistical study.

## Reproduce the published plan

Use a disposable checkout. Back up the published results before rerunning:

```sh
mkdir -p research/published
mv research/runs research/results.tsv research/frozen.json research/final-evaluation.json research/published/
mkdir research/runs
python -m tinychess.autoresearch --max-runs 20 --max-minutes 60
```

The fixed data contract stays in place. Wall-time stopping and BLAS/platform
differences can alter the exact winner. Select the lowest validation NLL among
eligible models, then explicitly freeze its checkpoint:

```sh
python -m tinychess.freeze --experiment CHOSEN_NAME
python -m tinychess.research_evaluate
python -m tinychess.export_predictions
python -m unittest discover -s tests -v
npm ci --ignore-scripts
npm test
npm run benchmark
```

The freezer compares per-tensor and per-output INT8 quantization using validation
only, permits at most 0.2 percentage points of validation accuracy loss, and
enforces parameter/size caps. It refuses to replace an existing freeze. The test
evaluator verifies the frozen model hashes and refuses to overwrite an existing
test report. Committed prediction fixtures are regenerated only after an
intentional model change, never in CI.

The April/2025 test results are now public. Future research must reserve another
unseen source for a genuinely fresh final test; these can remain regression sets.
Keep promotions as a separate UI choice and preserve all legality/parity checks.
