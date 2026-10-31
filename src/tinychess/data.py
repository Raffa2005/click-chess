"""Extract all ambiguous played destinations from a deterministic game sample."""
import argparse, collections, contextlib, hashlib, io, json, pathlib, re, time, zipfile
import chess, chess.pgn
import numpy as np
import zstandard
from .features import encode, planes, origins, destination_moves

def digest(s): return hashlib.sha256(s.encode()).hexdigest()[:24]

@contextlib.contextmanager
def open_pgn(path):
    path=pathlib.Path(path)
    with contextlib.ExitStack() as stack:
        if path.suffix=='.zip':
            z=stack.enter_context(zipfile.ZipFile(path))
            members=[n for n in z.namelist() if n.endswith('.pgn')]
            if len(members)!=1: raise ValueError('Expected one PGN in ZIP')
            stream=stack.enter_context(z.open(members[0]))
        elif path.suffix=='.zst':
            raw=stack.enter_context(path.open('rb'))
            stream=stack.enter_context(zstandard.ZstdDecompressor().stream_reader(raw))
        else: stream=stack.enter_context(path.open('rb'))
        yield stack.enter_context(io.TextIOWrapper(stream,encoding='utf-8-sig',errors='strict'))

def pgn_records(stream):
    """Public archives use Event as the first tag of each game."""
    lines=[]
    for line in stream:
        if line.startswith('[Event ') and lines:
            yield ''.join(lines); lines=[]
        lines.append(line)
    if lines: yield ''.join(lines)

def extract(paths, output, modulus=1, min_elo=2100, corpus='lichess', max_games=None):
    counts=collections.Counter(); types=collections.Counter(); dates=collections.Counter()
    feats=[]; boards=[]; squares=[]; offsets=[0]; labels=[]; games=[]; keys=[]; splits=[]; kinds=[]; plies=[]
    seen=set(); fixtures=[]; start=time.time()
    for path in paths:
        with open_pgn(path) as stream:
            for raw in pgn_records(stream):
                counts['archive_games']+=1
                headers=dict(re.findall(r'^\[(\w+) "(.*)"\]',raw,re.M))
                if headers.get('Variant','Standard')!='Standard' or 'FEN' in headers:counts['nonstandard_skipped']+=1;continue
                try: elo=min(int(headers.get('WhiteElo','0')),int(headers.get('BlackElo','0')))
                except ValueError: elo=0
                if elo<min_elo: counts['rating_filtered']+=1;continue
                # Hash complete PGN for sampling across the whole month, not a prefix.
                if int(digest(raw),16)%modulus:continue
                game=chess.pgn.read_game(io.StringIO(raw))
                if game is None or game.errors:counts['parse_errors']+=1;continue
                sequence=list(game.mainline_moves())
                gid=digest(' '.join(m.uci() for m in sequence))
                if gid in seen:counts['duplicate_games']+=1;continue
                seen.add(gid)
                if max_games is not None and counts['selected_games']>=max_games:break
                counts['selected_games']+=1
                dates[headers.get('Date','?')]+=1
                event=headers.get('Event','').lower()
                kind=3 if corpus=='classical' or 'classical' in event else (2 if 'rapid' in event else (1 if 'blitz' in event else 0))
                types[str(kind)]+=1
                split=int(gid,16)%10==0
                board=game.board()
                for actual in sequence:
                    counts['plies']+=1
                    legal=destination_moves(board,actual.to_square)
                    ms=origins(board,actual.to_square)
                    if len(legal)>1:counts['ambiguous_moves_including_promotions']+=1
                    if len(ms)==1 and len(legal)>1:counts['promotion_only_ambiguities']+=1
                    if len(ms)>1:
                        if actual.from_square not in [m.from_square for m in ms]:raise ValueError('Missing played origin')
                        feats.append(encode(board,ms))
                        bp=planes(board,ms)
                        boards.append(bp[0,:775])
                        flip=0 if board.turn else 56
                        squares.extend([(m.from_square^flip,m.to_square^flip) for m in ms])
                        offsets.append(offsets[-1]+len(ms));labels.append([m.from_square for m in ms].index(actual.from_square))
                        games.append(gid);keys.append(digest(' '.join(board.fen(en_passant='legal').split()[:4])+':'+str(actual.to_square)))
                        splits.append(split);kinds.append(kind);plies.append(board.ply())
                        counts['origin_ambiguities']+=1
                        counts['same_piece_ambiguities' if len({board.piece_type_at(m.from_square) for m in ms})==1 else 'different_piece_ambiguities']+=1
                        if len(fixtures)<100 and counts['selected_games']%max(1,modulus)==0:
                            fixtures.append({'fen':board.fen(),'history':[m.uci() for m in board.move_stack[-2:]],'destination':chess.square_name(actual.to_square),'actual':actual.uci(),'features':feats[-1].tolist(),'origins':[m.from_square for m in ms]})
                    board.push(actual)
                if counts['selected_games']%500==0:print(json.dumps({'games':counts['selected_games'],'examples':len(labels),'seconds':round(time.time()-start,1)}),flush=True)
    out=pathlib.Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    if not labels:raise ValueError('No eligible examples')
    np.savez_compressed(out,x=np.concatenate(feats),board=np.asarray(boards,dtype=np.float32),squares=np.array(squares,dtype=np.uint8),
        offsets=np.array(offsets,dtype=np.int64),y=np.array(labels,dtype=np.int16),game=np.array(games),key=np.array(keys),
        validation=np.array(splits),kind=np.array(kinds,dtype=np.uint8),ply=np.array(plies,dtype=np.int16))
    manifest={'paths':[str(p) for p in paths],'source_sha256':{str(p):hashlib.file_digest(open(p,'rb'),'sha256').hexdigest() for p in paths},
        'sampling_modulus':modulus,'min_both_elo':min_elo,'corpus':corpus,'counts':dict(counts),'game_time_controls':dict(types),
        'dates':dict(dates),'seconds':time.time()-start,'feature_dim':feats[0].shape[1]}
    out.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    out.with_suffix('.fixtures.json').write_text(json.dumps(fixtures)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('paths',nargs='+');p.add_argument('--output',required=True);p.add_argument('--modulus',type=int,default=1);p.add_argument('--min-elo',type=int,default=2100);p.add_argument('--corpus',default='lichess');p.add_argument('--max-games',type=int)
    a=p.parse_args();extract(a.paths,a.output,a.modulus,a.min_elo,a.corpus,a.max_games)
