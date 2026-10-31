# Round 2, frozen before new tests

The February novel validation contract remains unchanged (36,024 examples).
Prior study tests are now regression sets, never model-selection targets.
The incumbent is the 17,345-parameter model in `baseline/`.

New final tests: May 2023 Lichess Elite, deterministic modulus-96 game sample;
Prague 2025 Masters/Challengers and Cyprus Women's Grand Prix classical games
from February/March broadcasts. Cyprus was added based only on archive headers,
before test extraction/evaluation, to increase classical sample coverage.
Check event names and exclude playoffs before extracting any test features.
Do not score these tests until architecture, checkpoint and quantization freeze.

Continue selecting lowest validation NLL, with 20,000 parameters, 24 KiB gzipped
and warm Node feature-plus-ranking p95 below 1 ms. Compare against the incumbent
on identical fresh examples, with game-bootstrap confidence intervals. Report
negative and uncertain results. Publish completed trials and preserve prior work.

The size cap applies to a supported deployment artifact. Compare lossless binary
packing with JSON; quantized weights and predictions must match exactly between
the formats. This choice is made before fresh tests are scored.

Adoption rule declared before final scoring: promote the frozen candidate only
if full-set top-1 accuracy improves on both fresh tests and the latency target
passes. Confidence intervals determine how strong a claim is justified. If the
rule fails, preserve the incumbent deployment and publish the candidate/report.
Do not train another candidate in response to test outcomes.

Hypotheses: competing-origin context; architecture; longer/lower-rate training;
another disjoint month of training data. December 2022 extra training must exclude
all February validation game IDs and position+destination keys. No previous or
new test games may be used as training data.

Individual exploratory budgets: 240 seconds, with explicitly recorded longer
confirmation/extra-data trials when justified. The run list is bounded and does
not schedule recurring work. Log every trial including failures. Keep data,
feature, training and evaluation sources separate.

Before final scoring, also test a fixed grid of weight averages: 25%, 50%, 75%
between the December incumbent continuation and the earlier refinement, and
between that December model (with an exact identity layer inserted) and the
deepened continuation. These six validation-only candidates add no deployed
parameters compared with their respective architecture. All parents share the
same initialization, so neuron indices are aligned. Record every blend.
