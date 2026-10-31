import json,unittest
from pathlib import Path
import chess
import numpy as np
from tinychess.features import encode,origins,planes
from tinychess.tactical import augment,reconstruct,risk_summary,NAMES

class TacticalTests(unittest.TestCase):
    def test_canonical_reconstruction_and_no_mutation(self):
        for f in json.loads(Path('tests/fixtures.json').read_text()):
            b=chess.Board(f['fen']);b.move_stack=[chess.Move.from_uci(m) for m in f['history']]
            ms=origins(b,chess.parse_square(f['destination']));base=encode(b,ms)
            fen=b.fen();history=b.move_stack[:];rich=augment(b,ms,base)
            self.assertEqual(rich.shape,(len(ms),147));self.assertEqual(b.fen(),fen);self.assertEqual(b.move_stack,history)
            np.testing.assert_array_equal(rich[:,:107],base)
            flip=0 if b.turn else 56;p=planes(b,ms)[0,:775]
            canonical,cm=reconstruct(p,[(m.from_square^flip,m.to_square^flip) for m in ms])
            np.testing.assert_allclose(rich,augment(canonical,cm,base),atol=1e-7)
    def test_hanging_queen(self):
        b=chess.Board('7k/8/8/8/8/3r4/3Q4/K7 w - - 0 1')
        self.assertAlmostEqual(risk_summary(b,chess.WHITE)[1],9/12)

if __name__=='__main__':unittest.main()
