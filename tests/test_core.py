import json,unittest
import chess
import numpy as np
from tinychess.features import origins,encode,planes,destination_moves,HISTORY_DIM
from tinychess.model import Ranker,ClickSession

class RulesTests(unittest.TestCase):
    def test_cross_piece_ambiguity(self):
        b=chess.Board();ms=origins(b,chess.F3)
        self.assertEqual({m.uci() for m in ms},{'f2f3','g1f3'})
    def test_pin_removes_illegal_candidate(self):
        b=chess.Board('4r1k1/8/8/8/8/8/R3R3/4K3 w - - 0 1')
        self.assertEqual([m.uci() for m in origins(b,chess.C2)],['a2c2'])
    def test_castling_uses_king_destination(self):
        b=chess.Board('r2q2k1/1b3rbp/p1n3p1/1p2pP2/1Pp1P1PP/P2pQNN1/1BP5/R3K2R w KQ - 2 22')
        self.assertIn('e1c1',[m.uci() for m in origins(b,chess.C1)])
    def test_promotion_is_not_four_origins(self):
        b=chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1')
        self.assertEqual(len(destination_moves(b,chess.A8)),4)
        self.assertEqual(len(origins(b,chess.A8)),1)
    def test_two_pawn_origins_and_promotions(self):
        b=chess.Board('1r5k/P1P5/8/8/8/8/8/7K w - - 0 1')
        self.assertEqual(len(destination_moves(b,chess.B8)),8)
        self.assertEqual(len(origins(b,chess.B8)),2)
    def test_en_passant(self):
        b=chess.Board('7k/8/8/2PpP3/8/8/8/7K w - d6 0 1')
        self.assertEqual({m.uci() for m in origins(b,chess.D6)},{'c5d6','e5d6'})
    def test_features_do_not_change_board_or_history(self):
        b=chess.Board();b.push_san('e4');b.push_san('e5');before=b.fen();stack=b.move_stack[:]
        ms=origins(b,chess.F3);x=encode(b,ms)
        self.assertEqual(x.shape,(len(ms),HISTORY_DIM));self.assertTrue(np.isfinite(x).all())
        self.assertEqual(before,b.fen());self.assertEqual(stack,b.move_stack)
    def test_color_reflection(self):
        b=chess.Board();ms=origins(b,chess.F3);x=encode(b,ms,False)
        c=b.mirror();c.fullmove_number=1
        ns=[chess.Move(m.from_square^56,m.to_square^56) for m in ms];y=encode(c,ns,False)
        # Black-to-move changes absolute ply by one, deliberately represented.
        from tinychess.features import feature_names
        col=feature_names(False).index('ply');x[:,col]=y[:,col]
        np.testing.assert_allclose(x,y)
    def test_rank_and_cached_clicks(self):
        a={'feature_version':1,'representation':'history','weights':[np.zeros((HISTORY_DIM,1)).tolist()],'biases':[[0]]}
        r=Ranker(a);b=chess.Board();c=ClickSession(r,b,'f3')
        self.assertNotEqual(c.click()['from'],c.click()['from']);self.assertEqual(b.fen(),chess.STARTING_FEN)
        self.assertEqual(r.rank(b,'e5'),[])
        self.assertEqual(r.rank(b,'e4')[0]['probability'],1)
    def test_gradient(self):
        from tinychess.train import forward,probabilities
        rng=np.random.default_rng(1);x=rng.normal(size=(6,4)).astype(np.float32);w=rng.normal(size=(4,1)).astype(np.float32)
        valid=np.array([[1,1,0],[1,1,1]],bool);y=np.array([1,0]);p=probabilities((x@w).reshape(2,3),valid);p[np.arange(2),y]-=1
        analytic=x.T@(p/2).reshape(-1,1)
        def loss(w):
            p=probabilities((x@w).reshape(2,3),valid);return -np.log(p[np.arange(2),y]).mean()
        for j in range(4):
            plus=w.copy();minus=w.copy();plus[j,0]+=.001;minus[j,0]-=.001
            self.assertAlmostEqual(float(analytic[j,0]),float((loss(plus)-loss(minus))/.002),places=3)

if __name__=='__main__':unittest.main()
