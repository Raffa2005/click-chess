# click-chess autonomous research protocol

Improve the probability of selecting a human's intended origin from a clicked
destination while preserving instantaneous local inference. This adapts the
measure/keep/discard approach of https://github.com/karpathy/autoresearch to CPU
chess move input. It is not a fork of its GPU language-model training code.

The completed second study, fresh tests, and adoption rule are documented in
[research/round2/protocol.md](research/round2/protocol.md). Earlier test sets below
are now regression sets for subsequent work.

The active full-board study and cross-entropy reporting contract are in
[research/fullboard/protocol.md](research/fullboard/protocol.md). Its revised
latency-based constraints supersede the parameter cap below for that study.

## Immutable evaluation contract

- Freeze the original February 2023 game split and the 36,024 novel validation
  examples before experimenting. Do not change labels, masks or sample weights
  for validation. New training data must exclude validation game IDs and exact
  position+destination keys. Report training-data changes explicitly.
- Primary selection metric: validation negative log likelihood. Also record
  top-1/top-2, parameter count, training time and artifact size. Keep an improvement
  only if it satisfies the deployment constraints. Retain a Pareto frontier for
  users who prefer a smaller model.
- Never use March 2023 or the previously inspected 2024 classical results to guide
  this new search. They are regression reports only after the winner is frozen.
- Fresh final tests: a deterministic sample of April 2023 Lichess Elite and
  previously unused classical games. Do not evaluate them until all architecture,
  feature, checkpoint and export choices are frozen. No training on these games.
- Legal moves, promotions and standard-chess semantics remain authoritative.
  Test positions must never enter training via distillation or feature fitting.

## Deployment constraints

- At most 20,000 trainable parameters and 24 KiB gzipped model data.
- CPU only; no engine search, runtime network call or large inference framework.
- Target warm Node p95 <1 ms for feature extraction + ranking with legal moves
  supplied. Separately measure legal generation and cached repeat clicks.
- Retain working Python and JavaScript implementations with feature and output
  parity, including castling, pins, en passant, black orientation and promotions.

## Experiment loop

1. State a falsifiable hypothesis and one configuration in `research/experiments.json`.
2. Run `python -m tinychess.autoresearch --experiments NAME` with single-thread BLAS.
   Each normal trial has a 120-second optimization budget (data load and final
   verification excluded), with a maximum-epoch guard. Record the actual budget.
   To resume the whole bounded plan, use `python -m tinychess.autoresearch
   --max-runs 20 --max-minutes 60`. Completed trials are skipped, never overwritten.
3. Save config, source hash, data hashes, learning curve, best checkpoint, metrics,
   elapsed time, error information and keep/discard decision. Append to
   `research/results.tsv`; never erase an unsuccessful trial.
4. Prefer a cheaper change when results are effectively tied. Check promising
   changes using another seed and an equal-budget ablation before claiming cause.
5. Change the next hypothesis based on validation evidence. Do not schedule or
   claim perpetual unattended work; a run performs its explicit experiment list.
6. Freeze the winner and quantization using validation only. Run the fresh tests
   once, report failures as well as improvements, verify inference and publish
   code, results, protocol and deployment model to the repository.

## Useful directions

Learning-rate decay; longer patience; feature standardization folded into the
first layer; shallow versus two-layer networks; stronger but inexpensive tactical
features; larger disjoint training sample; teacher distillation if worthwhile.
Keep data and evaluation code separate from experimental configurations.
