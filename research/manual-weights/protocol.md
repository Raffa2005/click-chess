# Manually authored weights

Question: how accurately can explicitly hand-chosen network weights predict the
origin of a human move, given its destination, compared with the trained model?

The primary model is a 147 → 13 → 1 ReLU origin scorer, with the input dimension
checked against the existing version-2 feature names. Its signed linear score is
represented by ReLU(z) − ReLU(−z); eleven other units encode manually chosen
development, king-activity and pawn-advance preferences. A single linear layer
using the same base coefficients is a predeclared ablation. All values and their
reasoning are retained in `src/tinychess/manual_weights.py` and `frozen.json`.

No optimizer, random weight search, distillation, copied trained coefficients,
validation fitting or tuning after test feedback is permitted. Both artifacts
and the builder are hashed before evaluation. There is no promotion to the app
or replacement of the deployed model in this experiment.

Use exactly the previous May 2023 online sample and the Prague/Cyprus 2025
classical allowlist. Require both players' Elo ≥2100. Rebuild caches from pinned
archives and match their source hashes, example counts and incumbent accuracies
to the prior round-2 report. The original report excluded zero development-game
examples on these two sets. Both models score identical legal origin sets, with
promotion piece choice separate. These are public regression benchmarks, not
new unseen test data. Reuse is disclosed; no fresh-generalization claim is made.

Report first-choice accuracy, top-two accuracy, mean clicks when cycling, NLL,
uniform and least-valuable-piece baselines, game-cluster bootstrap intervals,
model size and same-machine warm JavaScript latency. NLL is descriptive: the
manual logits have no fitted calibration. Benchmark identical positions for
both models, separating inference with legal moves supplied, full legal move
generation plus ranking, and cached repeated clicks. Tests must verify the
generic Python/JS rankers agree and preserve standard-chess legality.
