"""Prepare later, game-disjoint evaluation data without observing predictions."""
import argparse
from collections import Counter, deque
import hashlib, io, json, pathlib, re, time
import chess.pgn
import numpy as np
from .data import open_pgn, pgn_records, digest
from .features import encode
from .tactical import augment
from .fullboard import packed_board, candidate_indices


def prepare(path, output, modulus=256, classical=False, exclude=None):
    out=pathlib.Path(output); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise FileExistsError(out)
    excluded=set(np.load(exclude)['game']) if exclude else set()
    seen=set(); counts=Counter(); events=Counter(); dates=Counter()
    x=[]; tokens=[]; indices=[]; offsets=[0]; labels=[]; games=[]; keys=[]; fullkeys=[]; plies=[]; destinations=[]
    start=time.perf_counter()
    with open_pgn(path) as stream:
        for raw in pgn_records(stream):
            counts['archive_games']+=1
            h=dict(re.findall(r'^\[(\w+) "(.*)"\]',raw,re.M))
            if h.get('Variant','Standard')!='Standard' or 'FEN' in h:continue
            if classical:
                event=h.get('Event','').lower()
                if 'tata steel' not in event or not any(s in event for s in ('masters','challengers')):continue
                if any(s in raw.split('\n\n',1)[0].lower() for s in ('tiebreak','tie-break','blitz','rapid')):continue
            try:
                if min(int(h.get('WhiteElo','0')),int(h.get('BlackElo','0')))<2100:continue
            except ValueError:continue
            if int(digest(raw),16)%modulus:continue
            game=chess.pgn.read_game(io.StringIO(raw))
            if game is None or game.errors:counts['parse_errors']+=1;continue
            moves=list(game.mainline_moves()); gid=digest(' '.join(m.uci() for m in moves))
            if gid in seen or gid in excluded:counts['duplicate_or_development_games']+=1;continue
            seen.add(gid);counts['selected_games']+=1
            events[h.get('Event','')]+=1;dates[h.get('Date','')]+=1
            b=game.board();history=deque(maxlen=8)
            for actual in moves:
                history.append(packed_board(b));ms,ix=candidate_indices(b,actual.to_square)
                if len(ms)>1:
                    packed=np.array([history[0]]*(8-len(history))+list(history))
                    base=encode(b,ms);x.append(augment(b,ms,base));tokens.append(packed)
                    indices.append(ix);offsets.append(offsets[-1]+len(ms))
                    labels.append([m.from_square for m in ms].index(actual.from_square))
                    games.append(gid);plies.append(b.ply());destinations.append(actual.to_square)
                    key=' '.join(b.fen(en_passant='legal').split()[:4])+':'+str(actual.to_square)
                    keys.append(digest(key))
                    # Include absolute origin order and legal indices so identical
                    # inputs imply identical label meanings even across colors.
                    fullkeys.append(hashlib.sha256(packed.tobytes()+ix.tobytes()+bytes(m.from_square for m in ms)).hexdigest())
                b.push(actual)
            if counts['selected_games']%250==0:
                print(json.dumps({'games':counts['selected_games'],'decisions':len(labels),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    if not labels:raise ValueError('No eligible ambiguous decisions')
    np.savez_compressed(out,x=np.concatenate(x),packed=np.asarray(tokens),indices=np.concatenate(indices),offsets=offsets,
                        y=np.asarray(labels,np.int16),game=games,key=keys,fullkey=fullkeys,ply=np.asarray(plies,np.int16),destination=np.asarray(destinations,np.uint8))
    manifest={'source':str(path),'sha256':hashlib.file_digest(open(path,'rb'),'sha256').hexdigest(),
              'modulus':modulus,'both_elo_min':2100,'exclude_games_from':exclude,'counts':dict(counts),
              'decisions':len(labels),'events':dict(events),'dates':dict(dates),'seconds':time.perf_counter()-start}
    out.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('events','dates')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path');p.add_argument('--output',required=True)
    p.add_argument('--modulus',type=int,default=256);p.add_argument('--classical',action='store_true');p.add_argument('--exclude')
    a=p.parse_args();prepare(a.path,a.output,a.modulus,a.classical,a.exclude)
