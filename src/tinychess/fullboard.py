"""Optional full-board human-policy adapter. Requires external AGPL Maia3.

No external source or weights are bundled. One model forward pass per initial
ambiguous click; promotion probabilities are marginalized to the origin.
"""
import numpy as np
import chess
from .features import destination_moves, origins
from .crossentropy import logsumexp


def packed_board(board):
    """Exactly the external tokenizer's side-to-move canonical orientation."""
    flip = 0 if board.turn else 56
    out = np.zeros(64, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        out[square ^ flip] = piece.piece_type + (0 if piece.color == board.turn else 6)
    return out


def packed_history(board):
    b = board.copy(stack=7)
    history = [packed_board(b)]
    while b.move_stack and len(history) < 8:
        b.pop(); history.append(packed_board(b))
    history.reverse()
    return np.array([history[0]]*(8-len(history)) + history)


def token_tensor(packed):
    import torch
    return torch.from_numpy(token_array(packed))


def token_array(packed):
    x = np.eye(13, dtype=np.float32)[np.asarray(packed)][..., 1:]
    return x.transpose(0,2,1,3).reshape(len(x),64,96).copy()


def move_index(board, move):
    flip = 0 if board.turn else 56
    f, t = move.from_square ^ flip, move.to_square ^ flip
    if move.promotion:
        return 4096 + ((f % 8)*8 + t % 8)*4 + [chess.QUEEN,chess.ROOK,chess.BISHOP,chess.KNIGHT].index(move.promotion)
    return f*64+t


def candidate_indices(board, destination):
    ms = origins(board,destination)
    ix = np.full((len(ms),4), -1, dtype=np.int16)
    legal = destination_moves(board,destination)
    for i, rep in enumerate(ms):
        entries = [move_index(board,m) for m in legal if m.from_square == rep.from_square]
        ix[i,:len(entries)] = entries
    return ms, ix


def conditional_logits(policy_logits, indices):
    """Each row of indices is all legal promotion variants of one origin."""
    values = np.asarray(policy_logits,dtype=np.float64)[np.maximum(indices,0)]
    return logsumexp(np.where(indices >= 0, values, -np.inf))


class FullBoardRanker:
    def __init__(self, checkpoint, size='5m', temperature=1.0, rating=2400):
        import torch
        from maia3.uci import parse_args
        from maia3.models import MAIA3Model
        self.cfg = parse_args(['--model',size,'--checkpoint-path',str(checkpoint),'--device','cpu'])
        self.model = MAIA3Model(self.cfg)
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = ckpt.get('model_state_dict',ckpt)
        self.model.load_state_dict({k.replace('smolgen','gab'):v for k,v in state.items()},strict=True)
        self.model.eval()
        self.temperature, self.rating = float(temperature), float(rating)

    def policy_logits(self, packed):
        import torch
        x = token_tensor(packed)
        ratings = torch.full((len(x),), self.rating)
        with torch.inference_mode():
            return self.model(x,ratings,ratings)[0].numpy()

    def prepare(self, board):
        """Optional precomputation when the board changes, before any square tap.

        The full move policy does not depend on the destination. Caching is keyed
        by the exact history tensor and rating; legal masks are always recomputed.
        """
        packed=packed_history(board)
        key=(packed.tobytes(),self.rating)
        cached=getattr(self,'_policy_cache',None)
        if cached is not None and cached[0]==key:return cached[1]
        policy=self.policy_logits(packed[None])[0]
        policy.setflags(write=False)
        self._policy_cache=(key,policy)
        return policy

    def clear_cache(self):
        self._policy_cache=None

    def rank(self, board, destination):
        if board.chess960 or not board.is_valid():
            raise ValueError('Valid standard-chess position required')
        if isinstance(destination,str): destination=chess.parse_square(destination)
        ms, ix = candidate_indices(board,destination)
        if not ms: return []
        logits = np.zeros(1) if len(ms)==1 else conditional_logits(self.prepare(board),ix)/self.temperature
        p = np.exp(logits-logsumexp(logits))
        legal = destination_moves(board,destination)
        return [{'from':chess.square_name(ms[i].from_square),'probability':float(p[i]),
                 'moves':[m.uci() for m in legal if m.from_square==ms[i].from_square]}
                for i in np.argsort(-p,kind='stable')]


class OnnxFullBoardRanker(FullBoardRanker):
    """Same origin API, optional local ONNX Runtime, no PyTorch at inference."""
    def __init__(self, path, temperature=1.0, rating=2400, threads=1):
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        self.temperature,self.rating=float(temperature),float(rating)

    def policy_logits(self, packed):
        x=token_array(packed);rating=np.full((len(x),),self.rating,np.float32)
        return self.session.run(None,{'tokens':x,'self_elo':rating,'opponent_elo':rating})[0]
