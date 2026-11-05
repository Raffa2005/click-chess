import unittest
from collections import deque
import chess
import numpy as np
from tinychess.fullboard import FullBoardRanker, packed_board, packed_history, candidate_indices, conditional_logits, move_index


class CountingPolicy(FullBoardRanker):
    def __init__(self):self.rating=2400.;self.temperature=1.;self.calls=0
    def policy_logits(self,packed):
        self.calls+=1
        return np.zeros((len(packed),4352),np.float32)


class FullBoardTests(unittest.TestCase):
    def test_cache_reuses_policy_across_destinations_and_invalidates(self):
        b=chess.Board();m=CountingPolicy();fen=b.fen()
        m.prepare(b);m.rank(b,'f3');m.rank(b,'h3')
        self.assertEqual(m.calls,1)
        self.assertEqual(b.fen(),fen)
        b.push_san('e4');m.prepare(b)
        self.assertEqual(m.calls,2)
        m.rating=2500.;m.prepare(b)
        self.assertEqual(m.calls,3)
        m.clear_cache();m.prepare(b)
        self.assertEqual(m.calls,4)

    def test_canonical_black_and_history(self):
        b=chess.Board()
        np.testing.assert_array_equal(packed_board(b),packed_board(b.mirror()))
        expected=[]
        for san in ['e4','e5','Nf3','Nc6','Bb5','a6','Ba4','Nf6','O-O']:
            expected.append(packed_board(b));b.push_san(san)
        expected.append(packed_board(b))
        np.testing.assert_array_equal(packed_history(b),expected[-8:])
        np.testing.assert_array_equal(packed_history(chess.Board()),np.tile(packed_board(chess.Board()),(8,1)))

    def test_underpromotions_summed_and_illegal_ignored(self):
        b=chess.Board('1r5k/P1P5/8/8/8/8/8/7K w - - 0 1')
        ms,ix=candidate_indices(b,chess.B8)
        self.assertEqual(len(ms),2)
        self.assertTrue(np.all(ix>=4096))
        logits=np.full(4352,100.)  # illegal logits must not affect the result
        logits[ix[0]]=[0.,0.,0.,0.]
        logits[ix[1]]=[0.,-100.,-100.,-100.]
        scores=conditional_logits(logits,ix)
        np.testing.assert_allclose(np.exp(scores-np.logaddexp.reduce(scores)),[.8,.2])
        for m in b.legal_moves:
            self.assertEqual(move_index(b,m),move_index(b.mirror(),chess.Move(m.from_square^56,m.to_square^56,promotion=m.promotion)))

    def test_castling_and_en_passant_legal_mask(self):
        b=chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
        ms,_=candidate_indices(b,chess.G1)
        self.assertIn(chess.Move.from_uci('e1g1'),ms)
        b=chess.Board('7k/8/8/3pP3/8/8/8/7K w - d6 0 1')
        ms,_=candidate_indices(b,chess.D6)
        self.assertIn(chess.Move.from_uci('e5d6'),ms)


if __name__=='__main__':unittest.main()
