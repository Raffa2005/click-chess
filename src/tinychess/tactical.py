"""Version-2 features: legal-candidate context and inexpensive board-wide risk.

These are static attack-map heuristics, not engine evaluations or game labels.
They deliberately use pseudo-attacks, including pinned attackers, just like v1.
"""
import chess
import numpy as np
from .features import VALUES,feature_names

RISK_NAMES=['sum_risk','max_risk','second_risk','loose_count','attacked_material']
EXTRA_NAMES=[f'candidate_fraction_{p}' for p in range(1,7)]+['candidate_count','candidate_min_value','candidate_max_attacks','candidate_attacked_max_value']
EXTRA_NAMES += [f'{when}_{side}_{stat}' for when in ('before','after','delta') for side in ('us','them') for stat in RISK_NAMES]
NAMES=feature_names(True)+EXTRA_NAMES

def risk_summary(board,color):
    risks=[];loose=0;attacked_material=0.
    for s in chess.scan_forward(board.occupied_co[color]&~board.kings):
        enemies=board.attackers_mask(not color,s)
        if not enemies:continue
        value=float(VALUES[board.piece_type_at(s)]);attacked_material+=value
        friends=board.attackers_mask(color,s)
        if not friends:risk=value;loose+=1
        else:
            cheapest=min(float(VALUES[board.piece_type_at(a)]) for a in chess.scan_forward(enemies))
            risk=max(0,value-cheapest)
        risks.append(risk)
    risks.sort(reverse=True)
    return [sum(risks)/40,(risks[0] if risks else 0)/12,(risks[1] if len(risks)>1 else 0)/12,loose/16,attacked_material/40]

def context(base):
    idx=feature_names(True);v=base[:,idx.index('value')];att=base[:,idx.index('from_enemy_count')]
    return list(base[:,:6].mean(0))+[len(base)/8,float(v.min()),float(att.max()),float(np.max(np.where(att>0,v,0)))]

def augment(board,moves,base):
    if not moves:return np.empty((0,len(NAMES)),np.float32)
    color=board.turn;before=np.array(risk_summary(board,color)+risk_summary(board,not color),np.float32);ctx=context(base)
    rows=[]
    for m,x in zip(moves,base):
        board.push(m)
        try:after=np.array(risk_summary(board,color)+risk_summary(board,not color),np.float32)
        finally:board.pop()
        rows.append(np.r_[x,ctx,before,after,after-before])
    return np.array(rows,np.float32)

def reconstruct(plane,squares):
    """Reconstruct canonical side-to-move-as-White board from stored input planes."""
    b=chess.Board(None)
    for channel,s in np.argwhere(plane[:768].reshape(12,64)>0.5):
        b.set_piece_at(int(s),chess.Piece(int(channel%6)+1,bool(channel<6)))
    b.turn=chess.WHITE;b.castling_rights=0
    for present,s in zip(plane[768:772],[chess.H1,chess.A1,chess.H8,chess.A8]):
        if present>.5:b.castling_rights|=chess.BB_SQUARES[s]
    b.ep_square=int(round(float(plane[772])*63)) if plane[772]>=0 else None
    moves=[]
    for f,t in squares:
        f,t=int(f),int(t);promotion=chess.QUEEN if b.piece_type_at(f)==chess.PAWN and t//8==7 else None
        moves.append(chess.Move(f,t,promotion=promotion))
    return b,moves
