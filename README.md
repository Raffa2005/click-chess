# click-chess

A CPU-trained neural network that predicts the intended **origin square**, given
a standard-chess position and a clicked destination. It ranks only legal origins.
It does not play chess, search variations, or infer an illegal move.

## Definition

A move is ambiguous if its destination does not uniquely identify the legal move.
This includes different piece types, castling, and promotion choices, not just SAN
ambiguity between identical pieces. The network solves the **origin-choice** part.
Promotion alternatives from one origin remain an explicit UI choice. A queen
promotion represents each origin while scoring; this is not an automatic promotion
decision. Positions with only promotion-choice ambiguity have no origin loss and
are counted separately.

## Model and results

The deployed network has **17,345 parameters**, a **16.8 KB gzipped binary**
(**18.6 KB uncompressed**), and **0.31 ms p95** feature extraction + ranking in the
warm Node benchmark with legal moves supplied. Its round-one test accuracy was
**85.11%** on April 2023 online games and **82.98%** on Tata Steel 2025 classical
games. Second-choice coverage was 98.77% and 98.53%.

[Round two](research/round2/RESULTS.md) tested eight additional training runs and
six weight averages. Its candidate improved fresh online accuracy but missed the
classical adoption requirement, so the existing weights remain the default.
The new binary export preserves predictions exactly; repeat clicks remain cached.

The latest candidate comparison is in [RESULTS.md](RESULTS.md); the previous
model study is preserved in [research/round2/baseline/RESULTS.md](research/round2/baseline/RESULTS.md).
The bounded [research protocol](program.md), [trial log](research/results.tsv), and
[reproduction guide](research/README.md) preserve successful and unsuccessful experiments.
Original baseline reports remain in [results/](results/) and [research/baseline/](research/baseline/). The model is a shared candidate scorer: the same tiny MLP
produces one logit for each legal origin. A softmax over these logits is trained
with cross entropy against the origin used in the actual game. A 64-entry output
is available, with zero probability at illegal origins.

Candidate features include piece type, origin/destination geometry, material,
castling rights, check, attack/defense information before and after the candidate,
mobility, king distances and pawn-file counts. Version 2 adds competing-origin
context and board-wide material-at-risk summaries before and after each candidate.
These are cheap attack-map heuristics, including pinned attackers, not an engine
evaluation. All 147 features are computed solely from the
pre-move position; examining each legal candidate is not looking at the observed
answer. Black positions are rank-reflected. The selected model includes the last
two plies. Initial experiments also compared no history and raw board planes. See
[`features.py`](src/tinychess/features.py) and [`tactical.py`](src/tinychess/tactical.py).

## Use in Python

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

```python
import chess
from tinychess.model import Ranker, ClickSession

model = Ranker('models/origin-ranker.bin')  # load once
board = chess.Board()
ranking = model.rank(board, 'f3')  # f2 or g1 can move to f3
print(ranking)
probabilities = model.probabilities64(board, 'f3')  # a1=0 ... h8=63

session = ClickSession(model, board, 'f3')
first = session.click()
alternative = session.click()  # cycles cached candidates, no inference
```

The caller commits the selected move separately. Do not advance the board between
clicks that cycle a choice; create a new session when the position or destination
changes. An empty ranking means the destination has no legal move. A unique
origin bypasses the network. The `moves` field exposes all legal promotion choices
for that origin. Standard chess only; Python rejects Chess960 and invalid boards.

## Use in JavaScript / browsers

[`js/ranker.mjs`](js/ranker.mjs) has no runtime dependencies. Supply legal moves
from your existing chess library. This example uses chess.js:

```js
import { Chess } from 'chess.js';
import { OriginRanker } from './js/ranker.mjs';

const bytes = await (await fetch('./models/origin-ranker.bin')).arrayBuffer();
const model = OriginRanker.fromBinary(bytes); // load once, outside click handler
const chess = new Chess();
const legal = chess.moves({ verbose: true });
const session = model.clickSession(chess.fen(), legal, 'f3');
const choice = session.click();
// Render/confirm choice; chess.move(...) is a separate application decision.
```

For a game already in progress, pass its previous two plies as the fourth argument
to `rank` or `clickSession` (after the destination). The selected model uses this
history. A FEN alone does not contain it. In Python, retain the `chess.Board` move
stack. If history is unavailable, load `models/origin-ranker.no-history.json`
instead; that separately trained baseline uses only the current position and has
not received the new research improvements.

Use the current position's complete legal move list, including underpromotions.
Do not pass pseudo-legal moves: JavaScript delegates legality and board validity to
the caller's chess library. Legal moves need `from`, `to`, and optional `promotion`
fields; chess.js verbose moves work directly. History-enabled experimental models
also take an array of prior `{from,to}` moves, oldest first, retaining at least the
last two. Avoid reconstructing full game history on each click.

## Reproduce and improve

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest discover -s tests -v
npm ci --ignore-scripts
npm test
npm run benchmark
```

See [research/round2/README.md](research/round2/README.md) and
[research/README.md](research/README.md) for downloading sources, preparing
features, reproducing the study, and adding bounded experiments. Training uses
NumPy on CPU; no GPU or PyTorch is needed. Data preparation uses substantially
more memory than inference. Archives and caches are not committed.

The old `tinychess.train` / `tinychess.evaluate` commands reproduce the original
baseline and can overwrite deployment files. Use the new research runner for
version-2 experiments. Every completed trial remains in the log. A resumed plan
skips those trials; it does not silently train forever.

## Evaluation design

- February 2023 Lichess Elite: deterministic hash sampling over the entire archive;
  90/10 game-sequence-hash train/validation split. Both players must pass 2100; the
  upstream collection is stricter: 2500+ versus 2300+, excluding bullet.
- All played destinations with multiple legal origins are extracted from sampled
  games. Training samples are not limited to SAN-disambiguated moves.
- Validation examples matching a training position+destination are removed before
  model selection. Choose lowest validation negative log likelihood.
- January 2023 adds training examples after excluding every February validation
  game and position+destination. The February validation set remains fixed.
- The original March 2023 and classical 2024 results were already visible before
  this research. New final tests use April 2023 online games and 2025 Tata Steel
  Masters/Challengers classical games. The new tests are scored only after
  architecture, weights and quantization have been selected on validation.
- Complete move-sequence duplicates are removed within archives and across
  development/test sets. Report both full test and novel-position results.
- Accuracy is per ambiguous origin decision; confidence intervals bootstrap whole
  games. Players may appear in multiple partitions. No claim of player-disjoint
  evaluation or performance on novices, variants or arbitrary compositions.

These are human-choice predictions, not guarantees of the move a particular user
intended. Keep corrections easy. Probabilities are softmax scores and have not
been separately calibrated. Promotion-piece accuracy is not measured.

## Sources and licenses

- [Lichess open database](https://database.lichess.org/): online game data CC0.
- [Lichess Elite Database, Nikonoel](https://database.nikonoel.fr/): filtered online
  games; the source and filtering criteria are preserved in the reports.
- [Lichess broadcasts](https://database.lichess.org/#broadcasts): CC BY-SA 4.0;
  used only for evaluation. Classical events are allowlisted, with playoff games
  excluded. No broadcast PGN is redistributed here.
- Original code: MIT. Dependencies retain their own licenses: python-chess
  GPL-3.0-or-later, NumPy BSD-3-Clause, chess.js BSD-2-Clause, zstandard BSD-3-Clause.
  The JavaScript inference module is independent of python-chess.

The exported model was trained only on the CC0 online game data.
