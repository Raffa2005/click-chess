import json
import unittest
from pathlib import Path

import chess
import numpy as np
from tinychess.manual_evaluate import verify
from tinychess.manual_weights import BASE, GATES, ROOT, build, vector
from tinychess.model import Ranker, ClickSession


class ManualWeightsTests(unittest.TestCase):
    def test_frozen_compilation_and_signed_score(self):
        verify()
        fixtures=json.loads(Path('tests/fixtures.json').read_text())
        for name,nonlinear in [('manual-relu',True),('manual-linear',False)]:
            self.assertEqual(build(nonlinear),json.loads((ROOT/(name+'.json')).read_text()))
            model=Ranker(ROOT/(name+'.json'))
            # The deployed weight matrices must preserve the authored signed
            # score, including negative values and nonlinear opening gates.
            for f in fixtures:
                x=np.array(f['rich_features'],np.float32)
                expected=x@vector(BASE)
                if nonlinear:
                    for _,coeff,bias,weight in GATES:
                        expected+=np.maximum(x@vector(coeff)+bias,0)*weight
                np.testing.assert_allclose(model.scores(x),expected,atol=1e-5,rtol=1e-5)

    def test_legality_special_moves_and_cycle_preserve_position(self):
        for name in ['manual-relu','manual-linear']:
            model=Ranker(ROOT/(name+'.json'))
            for fen,destination,expected in [
                (chess.STARTING_FEN,'e5',set()),
                (chess.STARTING_FEN,'c3',{'b1','c2'}),
                ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1','g1',{'e1','h1'}),
                ('4k3/8/8/2PpP3/8/8/8/4K3 w - d6 0 1','d6',{'c5','e5'}),
                ('4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1','d6',set()),
                ('7k/P7/8/8/8/8/8/7K w - - 0 1','a8',{'a7'}),
                (chess.STARTING_FEN.replace(' w ',' b '),'c6',{'b8','c7'}),
            ]:
                board=chess.Board(fen);before=board.fen();legal={m.uci() for m in board.legal_moves}
                rows=model.rank(board,destination)
                self.assertEqual({r['from'] for r in rows},expected)
                for row in rows:self.assertTrue(set(row['moves'])<=legal)
                if destination=='a8':self.assertEqual(set(rows[0]['moves']),{'a7a8q','a7a8r','a7a8b','a7a8n'})
                session=ClickSession(model,board,destination)
                if rows:
                    first=session.click()
                    for _ in range(len(rows)-1):session.click()
                    self.assertEqual(first,session.click())
                self.assertEqual(board.fen(),before)


if __name__=='__main__':unittest.main()
