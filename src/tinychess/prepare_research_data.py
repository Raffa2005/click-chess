"""Memory-conscious feature-only extraction for extra training and sealed tests."""
import argparse,collections,hashlib,io,json,pathlib,re,time
import chess,chess.pgn
import numpy as np
from .data import open_pgn,pgn_records,digest
from .features import origins,encode,destination_moves
from .tactical import augment

def main():
    ap=argparse.ArgumentParser();ap.add_argument('paths',nargs='+');ap.add_argument('--output',required=True);ap.add_argument('--modulus',type=int,default=1);ap.add_argument('--min-elo',type=int,default=2100);ap.add_argument('--corpus',default='lichess');ap.add_argument('--extra-training',action='store_true');a=ap.parse_args()
    blocked_games=set();blocked_keys=set()
    if a.extra_training:
        with np.load('data/processed/feb.npz') as z:
            # Protect every February validation game/position, not only novel subset.
            blocked_games=set(z['game'][z['validation']]);blocked_keys=set(z['key'][z['validation']])
    counts=collections.Counter();dates=collections.Counter();events=collections.Counter();parts=[];offsets=[0];y=[];games=[];keys=[];plies=[];kinds=[];seen=set();start=time.perf_counter()
    for path in a.paths:
        with open_pgn(path) as stream:
            for raw in pgn_records(stream):
                counts['archive_games']+=1
                h=dict(re.findall(r'^\[(\w+) "(.*)"\]',raw,re.M))
                if h.get('Variant','Standard')!='Standard' or 'FEN' in h:continue
                try:elo=min(int(h.get('WhiteElo','0')),int(h.get('BlackElo','0')))
                except ValueError:continue
                if elo<a.min_elo or int(digest(raw),16)%a.modulus:continue
                g=chess.pgn.read_game(io.StringIO(raw))
                if g is None or g.errors:counts['parse_errors']+=1;continue
                seq=list(g.mainline_moves());gid=digest(' '.join(m.uci() for m in seq))
                if gid in seen or gid in blocked_games:counts['duplicate_or_validation_games_skipped']+=1;continue
                seen.add(gid);counts['selected_games']+=1;dates[h.get('Date','?')]+=1;events[h.get('Event','?')]+=1
                event=h.get('Event','').lower()
                kind=3 if a.corpus=='classical' or 'classical' in event else (2 if 'rapid' in event else (1 if 'blitz' in event else 0))
                b=g.board()
                for actual in seq:
                    counts['plies']+=1;ms=origins(b,actual.to_square)
                    if len(ms)>1:
                        key=digest(' '.join(b.fen(en_passant='legal').split()[:4])+':'+str(actual.to_square))
                        if key in blocked_keys:counts['validation_positions_skipped']+=1
                        else:
                            base=encode(b,ms);parts.append(augment(b,ms,base));offsets.append(offsets[-1]+len(ms))
                            y.append([m.from_square for m in ms].index(actual.from_square));games.append(gid);keys.append(key);plies.append(b.ply());kinds.append(kind)
                            counts['examples']+=1
                    elif len(destination_moves(b,actual.to_square))>1:counts['promotion_only']+=1
                    b.push(actual)
                if counts['selected_games']%500==0:print(json.dumps({'games':counts['selected_games'],'examples':len(y),'seconds':time.perf_counter()-start}),flush=True)
    out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,x=np.concatenate(parts),offsets=np.array(offsets,dtype=np.int64),y=np.array(y,dtype=np.int16),game=np.array(games),key=np.array(keys),ply=np.array(plies,dtype=np.int16),kind=np.array(kinds,dtype=np.uint8))
    sha={}
    for p in a.paths:
        with open(p,'rb') as f:sha[p]=hashlib.file_digest(f,'sha256').hexdigest()
    meta={'source_sha256':sha,'modulus':a.modulus,'min_both_elo':a.min_elo,'extra_training':a.extra_training,'counts':dict(counts),'dates':dict(dates),'events':dict(events),'feature_version':2,'seconds':time.perf_counter()-start}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
