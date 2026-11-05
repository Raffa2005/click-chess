# Full-board, cross-entropy-first input prediction

This study replaces the old model-selection accuracy gate with held-out
cross-entropy, in **nats per ambiguous click**. Zero-loss unambiguous clicks and
promotion-only choices are excluded. Candidate promotions are marginalized to
origins, while the UI still chooses the promotion piece separately.

## What the minimum means

For the true origin distribution `p` and predictor `q`, start with
`L(q) = E[-log q(O|X)]`. Adding and subtracting `log p(O|X)` gives
`L(q) = H(O|X) + E_X KL(p(.|X) || q(.|X))`. Thus the minimum population loss is
conditional entropy; the excess is the expected KL divergence. We do not know
that entropy and do not claim a percentage of the optimum achieved. A stronger
model's test loss is an achievable reference, not an entropy-floor estimate.

On a finite recorded sample, an unrestricted lookup table has a different
optimum: group identical inputs, assign each origin its observed frequency, and
average `-log(frequency)`. Singletons have zero loss by construction. This is a
memorization diagnostic, not a generalization target.

The prior shipped model's independently recomputed results are:

| Existing regression set | Cross-entropy | 95% game-bootstrap CI | Uniform legal origins | Empirical lookup minimum at exact 147-feature input |
|---|---:|---:|---:|---:|
| May 2023 online, 125,578 decisions | 0.333818 | 0.329901–0.337577 | 0.833313 | 0.005203 |
| Prague/Cyprus 2025 classical, 5,888 decisions | 0.386949 | 0.369069–0.405818 | 0.840233 | 0.001355 |

89% and 91% of those exact feature inputs occur only once. This is why subtracting
the empirical lookup minimum does **not** measure our distance from human entropy.
Reproduce using `python -m tinychess.crossentropy_audit --data CACHE.npz --output REPORT.json`.

## Representation and new optional backend

The shipped Android/JavaScript model remains a shared `147 -> 96 -> 32 -> 1` MLP
with 17,345 parameters. Each legal origin gets one feature row: piece type,
from/to coordinates, move geometry, piece counts, attack/defence and material-risk
summaries before/after the candidate, checks, castling rights, mobility, pawn
structure, and relative encodings of the previous two plies. It does not receive
the complete piece placement. Black ranks are reflected. Softmax is over legal
candidate origins, not over 64 unconstrained squares.

The new optional backend uses **externally pretrained Maia3**, not a network we
trained from scratch. Its eight position snapshots become 64 square tokens with
12 piece/color indicators per snapshot (96 values per square). History orientation
and early padding match upstream exactly. Ratings are fixed at 2400 for both sides
in this experiment. Castling and en-passant rights are enforced by the legal mask;
the pretrained network does not directly encode these rights. No game outcome,
future clock, or observed next move is an input.

A single forward pass predicts the full move policy. The adapter gathers all
legal moves to the clicked destination, sums promotion probabilities by origin,
and normalizes. `prepare(board)` can compute the policy when the position changes;
subsequent destination choices reuse it. `ClickSession` also caches repeat taps.
No engine search or network request is made at inference.

## Reproduce and use

Run from the repository root. Keep the optional environment separate from the
tiny NumPy/JavaScript runtime. The upstream code is AGPL-3.0, and upstream terms
apply to its checkpoints and exports; no third-party weights/code are bundled or
represented as MIT-licensed click-chess training output.

```sh
python -m venv .venv-fullboard
. .venv-fullboard/bin/activate
pip install -e .
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install onnx==1.23.2 onnxruntime==1.31.1
pip install --no-deps 'git+https://github.com/CSSLab/maia3.git@1e13597c42d4858b7cfd7cfdae01e297263364b2'
python -m tinychess.fullboard_fetch --size all --data
python -m tinychess.fullboard_export --size 5m
python -m tinychess.fullboard_export --size 23m
```

Exact checkpoint revisions and archive checksums: [sources.json](sources.json).
Only already downloaded local files are used by `OnnxFullBoardRanker`.

```python
import chess
from tinychess.fullboard import OnnxFullBoardRanker
from tinychess.model import ClickSession

model = OnnxFullBoardRanker('research/fullboard/checkpoints/maia3-5m.int8.onnx')
board = chess.Board()
model.prepare(board)  # optional: once when the position changes
print(model.rank(board, 'f3'))
session = ClickSession(model, board, 'f3')
first, alternative = session.click(), session.click()
```

Use the temperature in the frozen model configuration for calibrated probabilities.
Read [protocol.md](protocol.md) before scoring tests. The existing app deployment is
unchanged; this is a locally usable Python/ONNX research backend, not an Android APK.

```sh
python -m tinychess.fullboard_prepare data/raw/lichess_elite_2025-09.zip --output research/fullboard/cache/development.npz
python -m tinychess.fullboard_prepare data/raw/lichess_elite_2025-10.zip --output research/fullboard/cache/online-test.npz --exclude research/fullboard/cache/development.npz
python -m tinychess.fullboard_prepare data/raw/broadcast-2026-01.pgn.zst --output research/fullboard/cache/classical-test.npz --classical --modulus 1
python -m tinychess.fullboard_study --data research/fullboard/cache/development.npz --size 5m --onnx research/fullboard/checkpoints/maia3-5m.int8.onnx --fit --output research/fullboard/development-5m-int8.json
# Final tests: omit --fit and pass --calibration research/fullboard/development-5m-int8.json.
# The costly 23M online reference and 5M float ablation use --game-modulus 4.
```

The scripts refuse to overwrite a completed score report or prepared dataset.
Raw and calibrated losses, sample sizes, empirical repeat diagnostics, and
cluster-bootstrap confidence intervals are saved as JSON. INT8 export errors are
checked separately; quantization is not assumed lossless.
