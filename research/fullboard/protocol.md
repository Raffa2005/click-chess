# Full-board cross-entropy study (2026-11-05)

The user's new objective supersedes the earlier 20,000-parameter cap: minimize
held-out origin cross-entropy with a seamless, local, single forward pass.

## Metric and limits

The primary metric is mean negative log probability of the observed origin,
in nats per ambiguous destination. Normalize over legal origins only; sum all
legal promotion probabilities belonging to the same origin. Report game-cluster
bootstrap confidence intervals and secondary latency measurements. Do not select
models on accuracy. Unambiguous clicks are excluded because their loss is zero.

For population cross-entropy, L(q) = H(O|X) + E KL(p(.|X)||q(.|X)). The entropy and
therefore excess loss are unknown. The best evaluated model is an achievable
reference, not an estimate of the irreducible floor. Also report the unrestricted
empirical minimum (frequency table per exact input) and repeat coverage, explicitly
as a memorization diagnostic, not a population entropy estimate.

## Data and experimental choices fixed before evaluation

- Optional external reference: CSSLab Maia3 5M and 23M. Published model cards say
  January 2023--July 2025 training. Do not use the earlier click-chess tests as
  held-out evidence for these models.
- Development: September 2025 Lichess Elite, deterministic PGN-hash sample
  (modulus 256), both ratings at least 2100, no variants/FEN starts/parse errors.
- Fresh online test: October 2025 Lichess Elite, same sampling rule. Exclude
  complete-game duplicates across development and test. Report all and inputs
  absent from development separately; pretraining position overlap is unknown.
- Fresh classical test: January 2026 Tata Steel Masters and Challengers, regular
  classical games only, both ratings at least 2100. Selection is by PGN headers,
  never prediction losses. No test fitting or repeated test-guided selection.
- Compare shipped 147-feature MLP, full-board 5M, and 23M reference. Inspect only
  development losses before freezing selection and calibration. Evaluate raw and
  temperature-calibrated models; temperature is fitted on development only.
- Default rating input: 2400 for both players, so this version works without user
  metadata. No future move, result, clock-after-move, or engine evaluation input.
- Full-board input: the upstream model's eight position snapshots, 64 square
  tokens, 12 piece/color indicators per snapshot. Follow upstream orientation
  and early-game padding exactly. Legal move mask carries castling/en-passant
  legality. The external pretrained model does not directly encode those rights.

## Runtime and reproducibility

Compute-budget amendment before reading any Maia3 development cross-entropy:
the larger 23M reference and the float32 5M quantization ablation use the
deterministic `int(game_hash,16) % 4 == 0` subset of online development/test games.
Report paired comparisons on identical subsets and their decision/game counts.
The selected 5M INT8 candidate and baseline retain the full sampled sets. Classical
test evaluation retains all selected games for both model sizes. This limits CPU
reference evaluation cost; it does not select games by loss.

Measure single-example CPU forward passes and full rank calls with model loaded,
including p50/p95, input construction and legal generation separately. Cached
repeat clicks reuse the ranking. Aim for warm model + input construction under
one 16.7 ms frame on this host; host timing is not an Android device guarantee.
A larger reference may exceed this target. Investigate INT8 if necessary, checking
output errors and development loss before final test. No inference network or
search. Record dependencies, source/checkpoint hashes, sampling manifests, and
all attempted variants. Keep the existing app default until a deployable runtime
passes its own tests. External Maia3 code is AGPL-3.0; keep it an optional external
research dependency, do not mislabel it as this repository's MIT code.
