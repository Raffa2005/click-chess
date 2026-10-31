"""Versioned candidate features. Everything is computed BEFORE the observed move.

Ranks are reflected for Black; files are unchanged. Attack maps include pinned
pieces (as in python-chess); candidate moves themselves are strictly legal.
"""
import chess
import numpy as np

VERSION = 1
VALUES = np.array([0, 1, 3, 3.25, 5, 9, 12], dtype=np.float32)

def destination_moves(board, destination):
    # python-chess applies castling to_mask to the rook square internally.
    # Include castling rooks, then filter the public standard-chess destination.
    mask=chess.BB_SQUARES[destination] | board.castling_rights
    return [m for m in board.generate_legal_moves(to_mask=mask) if m.to_square==destination]

def origins(board, destination):
    """One representative per legal origin; promotion is a separate UI choice."""
    found = {}
    for m in destination_moves(board, destination):
        if m.from_square not in found or m.promotion == chess.QUEEN:
            found[m.from_square] = m
    return [found[s] for s in sorted(found)]

def attack_info(board, color, square):
    ss = board.attackers(color, square)
    return [min(len(ss), 8) / 8, min((float(VALUES[board.piece_type_at(s)]) for s in ss), default=0) / 12]

def activity(board, square, color):
    attacks = board.attacks(square)
    enemy = attacks & chess.SquareSet(board.occupied_co[not color])
    friend = attacks & chess.SquareSet(board.occupied_co[color])
    return [len(attacks) / 32, len(enemy) / 8,
            max((float(VALUES[board.piece_type_at(s)]) for s in enemy), default=0) / 12,
            len(friend) / 8]

def feature_names(history=True):
    names = [f'piece_{p}' for p in range(1, 7)]
    names += [f'{loc}_{axis}_{i}' for loc in ('from','to') for axis in ('file','rank') for i in range(8)]
    names += ['dx','dy','distance','manhattan','from_center','to_center']
    names += [f'{side}_count_{p}' for side in ('us','them') for p in range(1,7)]
    names += ['ply','halfmove','check','our_oo','our_ooo','their_oo','their_ooo',
              'capture','captured_value','promotion','castle','en_passant','value']
    names += [f'{loc}_{side}_{stat}' for loc in ('from','to_after') for side in ('enemy','friend') for stat in ('count','min_value')]
    names += [f'{when}_{stat}' for when in ('before','after') for stat in ('mobility','targets','max_target','defended')]
    names += ['gives_check','from_own_king','to_own_king','from_enemy_king','to_enemy_king','from_pinned','pawn_same_file','pawn_adjacent_files']
    if history:
        names += [f'past_{i}_{s}' for i in (1,2) for s in ('exists','from_dx','from_dy','to_dx','to_dy','same_piece','same_destination')]
    return names

BASE_DIM = len(feature_names(False))
HISTORY_DIM = len(feature_names(True))

def encode(board, moves, history=True):
    turn = board.turn
    flip = 0 if turn else 56
    counts = [len(board.pieces(p, c)) / 8 for c in (turn, not turn) for p in range(1, 7)]
    globals_ = [min(board.ply(), 200)/200, min(board.halfmove_clock, 100)/100, float(board.is_check()),
        float(board.has_kingside_castling_rights(turn)),float(board.has_queenside_castling_rights(turn)),
        float(board.has_kingside_castling_rights(not turn)),float(board.has_queenside_castling_rights(not turn))]
    ownking, enemyking = board.king(turn), board.king(not turn)
    rows = []
    for m in moves:
        f,t = m.from_square,m.to_square
        cf,ct = f ^ flip,t ^ flip
        ff,fr,tf,tr = cf%8,cf//8,ct%8,ct//8
        pt = board.piece_type_at(f)
        row = [float(pt==p) for p in range(1,7)]
        row += [float(j==v) for v in (ff,fr,tf,tr) for j in range(8)]
        row += [(tf-ff)/7,(tr-fr)/7,max(abs(tf-ff),abs(tr-fr))/7,(abs(tf-ff)+abs(tr-fr))/14,
                (abs(ff-3.5)+abs(fr-3.5))/7,(abs(tf-3.5)+abs(tr-3.5))/7]
        row += counts + globals_
        captured = board.piece_type_at(t) or (chess.PAWN if board.is_en_passant(m) else 0)
        row += [float(board.is_capture(m)),float(VALUES[captured])/12,float(bool(m.promotion)),
                float(board.is_castling(m)),float(board.is_en_passant(m)),float(VALUES[pt])/12]
        row += attack_info(board,not turn,f)+attack_info(board,turn,f)
        before = activity(board,f,turn)
        pinned = float(board.is_pinned(turn,f))
        same = len(board.pieces(chess.PAWN,turn) & chess.SquareSet(chess.BB_FILES[tf]))/8
        adjacent = sum(len(board.pieces(chess.PAWN,turn) & chess.SquareSet(chess.BB_FILES[x])) for x in (tf-1,tf+1) if 0<=x<8)/16
        past=[]
        if history:
            for i in (1,2):
                if len(board.move_stack)>=i:
                    prev=board.move_stack[-i];pf,pt2=prev.from_square^flip,prev.to_square^flip
                    past += [1,(pf%8-ff)/7,(pf//8-fr)/7,(pt2%8-tf)/7,(pt2//8-tr)/7,float(prev.to_square==f),float(prev.to_square==t)]
                else: past += [0]*7
        board.push(m)
        try:
            row += attack_info(board,not turn,t)+attack_info(board,turn,t)
            row += before+activity(board,t,turn)
            row += [float(board.is_check()),chess.square_distance(f,ownking)/7,chess.square_distance(t,ownking)/7,
                chess.square_distance(f,enemyking)/7,chess.square_distance(t,enemyking)/7,pinned,same,adjacent]
        finally: board.pop()
        row += past
        rows.append(row)
    return np.asarray(rows,dtype=np.float32).reshape(len(moves),HISTORY_DIM if history else BASE_DIM)

def planes(board, moves):
    """Alternative raw board representation + origin/destination one-hots."""
    flip=0 if board.turn else 56
    base=np.zeros(768+7,dtype=np.float32)
    for s,p in board.piece_map().items():
        base[((0 if p.color==board.turn else 6)+p.piece_type-1)*64+(s^flip)]=1
    base[768:]=[board.has_kingside_castling_rights(board.turn),board.has_queenside_castling_rights(board.turn),
        board.has_kingside_castling_rights(not board.turn),board.has_queenside_castling_rights(not board.turn),
        (board.ep_square^flip)/63 if board.ep_square is not None else -1,min(board.ply(),200)/200,board.is_check()]
    rows=np.tile(np.r_[base,np.zeros(128,dtype=np.float32)],(len(moves),1))
    for i,m in enumerate(moves): rows[i,775+(m.from_square^flip)]=1;rows[i,839+(m.to_square^flip)]=1
    return rows
