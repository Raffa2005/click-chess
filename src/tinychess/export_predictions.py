"""Regenerate cross-language expected outputs after deliberately changing a model.

Do not run this in CI: committed expectations are checked independently there.
"""
import json
from pathlib import Path
import chess
from .model import Ranker


def main():
    fixtures = json.loads(Path('tests/fixtures.json').read_text())
    for suffix in ['', '.int8']:
        model = Ranker(f'models/origin-ranker{suffix}.json')
        outputs = []
        for fixture in fixtures:
            board = chess.Board(fixture['fen'])
            # Features only inspect the previous move endpoints. Push/pop of
            # hypothetical candidates preserves this supplied endpoint history.
            board.move_stack = [chess.Move.from_uci(m) for m in fixture['history']]
            outputs.append(model.rank(board, fixture['destination']))
        Path(f'tests/predictions{suffix}.json').write_text(json.dumps(outputs) + '\n')


if __name__ == '__main__':
    main()
