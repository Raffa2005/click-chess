"""Add diverse real-game inference fixtures from the February training partition."""
import io
import json
from pathlib import Path
import chess
import chess.pgn
import numpy as np
from .data import open_pgn, pgn_records, digest
from .features import encode, origins
from .tactical import augment


def main():
    path = Path('tests/fixtures.json')
    fixtures = json.loads(path.read_text())
    seen = {(f['fen'], f['destination']) for f in fixtures}
    rng = np.random.default_rng(32941)
    added = 0
    with open_pgn('data/raw/lichess_elite_2023-02.zip') as stream:
        for raw in pgn_records(stream):
            if int(digest(raw), 16) % 24:
                continue
            game = chess.pgn.read_game(io.StringIO(raw))
            if game is None or game.errors or game.headers.get('Variant', 'Standard') != 'Standard' or 'FEN' in game.headers:
                continue
            if min(int(game.headers.get('WhiteElo', 0)), int(game.headers.get('BlackElo', 0))) < 2100:
                continue
            moves = list(game.mainline_moves())
            if int(digest(' '.join(m.uci() for m in moves)), 16) % 10 == 0:
                continue
            candidates = []
            board = game.board()
            for actual in moves:
                legal_origins = origins(board, actual.to_square)
                destination = chess.square_name(actual.to_square)
                if 20 <= board.ply() <= 100 and len(legal_origins) > 1 and (board.fen(), destination) not in seen:
                    candidates.append((board.copy(), legal_origins, actual))
                board.push(actual)
            if not candidates:
                continue
            board, legal_origins, actual = candidates[int(rng.integers(len(candidates)))]
            base = encode(board, legal_origins)
            destination = chess.square_name(actual.to_square)
            fixtures.append({'fen': board.fen(), 'history': [m.uci() for m in board.move_stack[-2:]],
                             'destination': destination, 'actual': actual.uci(),
                             'origins': [m.from_square for m in legal_origins],
                             'features': base.tolist(), 'rich_features': augment(board, legal_origins, base).tolist()})
            seen.add((board.fen(), destination))
            added += 1
            if added == 200:
                break
    if added != 200:
        raise ValueError('Insufficient training fixtures')
    path.write_text(json.dumps(fixtures) + '\n')
    print(json.dumps({'added': added, 'total': len(fixtures)}))


if __name__ == '__main__':
    main()
