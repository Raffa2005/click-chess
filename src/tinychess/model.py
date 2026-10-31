"""Small NumPy MLP; no torch, engine, accelerator or inference server."""
import json,os
import numpy as np
from .features import encode, origins, destination_moves, planes, VERSION

class Ranker:
    def __init__(self, artifact):
        if isinstance(artifact,(str,bytes,os.PathLike)):
            with open(artifact) as f: artifact=json.load(f)
        if artifact['feature_version']!=VERSION:raise ValueError('Unsupported feature version')
        self.artifact=artifact
        if 'weights' in artifact:self.weights=[np.array(w,dtype=np.float32) for w in artifact['weights']]
        else:self.weights=[np.array(w,dtype=np.float32)*s for w,s in zip(artifact['weights_int8'],artifact['weight_scales'])]
        self.biases=[np.array(b,dtype=np.float32) for b in artifact['biases']]

    def scores(self,x):
        x=np.asarray(x,dtype=np.float32)
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            x=x@w+b
            if i<len(self.weights)-1:x=np.maximum(x,0)
        return x.ravel()

    def rank(self,board,destination):
        import chess
        if board.chess960:raise ValueError('Standard chess only')
        if not board.is_valid():raise ValueError('Invalid board')
        if isinstance(destination,str):destination=chess.parse_square(destination)
        ms=origins(board,destination)
        if not ms:return []
        if len(ms)==1:p=np.ones(1)
        else:
            rep=self.artifact['representation']
            x=planes(board,ms) if rep=='planes' else encode(board,ms,history=rep=='history')
            s=self.scores(x);p=np.exp(s-s.max());p/=p.sum()
        legal=destination_moves(board,destination)
        # Stable tie ordering by a1=0,...,h8=63; same as training and JavaScript.
        order=np.argsort(-p,kind='stable')
        return [{'from':chess.square_name(ms[i].from_square),'probability':float(p[i]),
                 'moves':[m.uci() for m in legal if m.from_square==ms[i].from_square]} for i in order]

    def probabilities64(self,board,destination):
        import chess
        p=np.zeros(64,dtype=np.float32)
        for row in self.rank(board,destination):p[chess.parse_square(row['from'])]=row['probability']
        return p

class ClickSession:
    """Rank once, cycle on repeated clicks, without advancing the chess board.

    The UI commits a chosen legal move separately. Promotion choices remain
    visible to the user. Creating a session never mutates its input board.
    """
    def __init__(self,ranker,board,destination):
        self.ranking=ranker.rank(board,destination);self.index=0
    def click(self):
        if not self.ranking:return None
        choice=self.ranking[self.index%len(self.ranking)];self.index+=1
        return choice
