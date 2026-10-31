# tiny-chess-input

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

Measured results are in [RESULTS.md](RESULTS.md), with machine-readable reports in
[results/](results/). The model is a shared candidate scorer: the same tiny MLP
produces one logit for each legal origin. A softmax over these logits is trained
with cross entropy against the origin used in the actual game. A 64-entry output
is available, with zero probability at illegal origins.

Candidate features include piece type, origin/destination geometry, material,
castling rights, check, attack/defense information before and after the candidate,
mobility, king distances and pawn-file counts. They are computed solely from the
pre-move position; examining each legal candidate is not looking at the observed
answer. Black positions are rank-reflected. An ablation adds the last two plies;
another uses the raw 12-plane board. See the versioned feature names in
[`features.py`](src/tinychess/features.py).

## Use in Python

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

```python
import chess
from tinychess.model import Ranker, ClickSession

model = Ranker('models/origin-ranker.int8.json')  # load once
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

const artifact = await (await fetch('./models/origin-ranker.int8.json')).json();
const model = new OriginRanker(artifact); // load once, outside click handler
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
instead; that separately trained model uses only the current position.

Use the current position's complete legal move list, including underpromotions.
Do not pass pseudo-legal moves: JavaScript delegates legality and board validity to
the caller's chess library. Legal moves need `from`, `to`, and optional `promotion`
fields; chess.js verbose moves work directly. History-enabled experimental models
also take an array of prior `{from,to}` moves, oldest first, retaining at least the
last two. Avoid reconstructing full game history on each click.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m tinychess.download
python -m tinychess.data data/raw/lichess_elite_2023-02.zip --output data/processed/feb.npz --modulus 24
python -m tinychess.data data/raw/lichess_elite_2023-03.zip --output data/processed/march.npz --modulus 96
python -m tinychess.data data/raw/classical-2024.pgn --output data/processed/classical.npz --corpus classical
python -m tinychess.train --data data/processed/feb.npz --epochs 18
python -m tinychess.train --data data/processed/feb.npz --epochs 45 --extended
python -m tinychess.evaluate --train data/processed/feb.npz --test data/processed/march.npz --classical data/processed/classical.npz
python -m unittest discover -s tests -v
npm ci --ignore-scripts
npm test
npm run benchmark
```

Downloads total approximately 154 MB compressed. Training needs NumPy, not a GPU
or PyTorch. Extracting features uses considerably more memory than inference.
Archives and processed arrays are deliberately not committed. Source hashes,
sampling parameters, counts, seeds, epoch logs and selection criteria are recorded.
Exact numerical results may vary slightly with NumPy/BLAS/platform versions.

## Evaluation design

- February 2023 Lichess Elite: deterministic hash sampling over the entire archive;
  90/10 game-sequence-hash train/validation split. Both players must pass 2100; the
  upstream collection is stricter: 2500+ versus 2300+, excluding bullet.
- All played destinations with multiple legal origins are extracted from sampled
  games. Training samples are not limited to SAN-disambiguated moves.
- Validation examples matching a training position+destination are removed before
  model selection. Choose lowest validation negative log likelihood.
- March 2023 is a separate temporal test; 2024 classical broadcast games are an
  external test. Neither test set is used for model/hyperparameter selection.
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
