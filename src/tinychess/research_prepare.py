"""Download research sources and reproduce feature caches; never score test data."""
import collections,json,os,pathlib,re,subprocess,sys,urllib.request
from .data import open_pgn,pgn_records

SOURCES={
 'lichess_elite_2023-01.zip':'https://database.nikonoel.fr/lichess_elite_2023-01.zip',
 'lichess_elite_2023-04.zip':'https://database.nikonoel.fr/lichess_elite_2023-04.zip',
 'broadcast-2025-01.pgn.zst':'https://database.lichess.org/broadcast/lichess_db_broadcast_2025-01.pgn.zst',
 'broadcast-2025-02.pgn.zst':'https://database.lichess.org/broadcast/lichess_db_broadcast_2025-02.pgn.zst'}

def classical_subset():
    allowed={'Tata Steel Chess Tournament 2025 Masters','Tata Steel Chess Tournament 2025 Challengers'};counts=collections.Counter();excluded=0
    with open('data/raw/classical-2025-sealed.pgn','w') as out:
        for name in ['broadcast-2025-01.pgn.zst','broadcast-2025-02.pgn.zst']:
            with open_pgn(pathlib.Path('data/raw')/name) as stream:
                for raw in pgn_records(stream):
                    h=dict(re.findall(r'^\[(\w+) "(.*)"\]',raw,re.M))
                    if h.get('Event') not in allowed:continue
                    if re.search(r'tie.?break|rapid|blitz',h.get('StudyName',''),re.I) or int(h.get('Round','0').split('.')[0])>13:excluded+=1;continue
                    counts[h['Event']]+=1;out.write(raw.rstrip()+'\n\n')
    pathlib.Path('research/classical-sources.json').write_text(json.dumps({'events':dict(counts),'tiebreaks_excluded':excluded,'license':'CC BY-SA 4.0, Lichess broadcasts','source_urls':[v for k,v in SOURCES.items() if k.startswith('broadcast')]},indent=2)+'\n')

def main():
    root=pathlib.Path('data/raw');root.mkdir(exist_ok=True,parents=True);pathlib.Path('research/cache').mkdir(exist_ok=True,parents=True)
    for name,url in SOURCES.items():
        p=root/name
        if p.exists():continue
        temp=p.with_suffix(p.suffix+'.part')
        print('Downloading',name,flush=True)
        with urllib.request.urlopen(url,timeout=60) as r,temp.open('wb') as f:
            while chunk:=r.read(1048576):f.write(chunk)
        temp.replace(p)
    classical_subset()
    jobs=[
      ('research/cache/feb-rich.npy',['tinychess.augment_data','--data','data/processed/feb.npz','--output','research/cache/feb-rich.npy','--workers','2']),
      ('research/cache/january-extra.npz',['tinychess.prepare_research_data','data/raw/lichess_elite_2023-01.zip','--output','research/cache/january-extra.npz','--modulus','24','--extra-training']),
      ('research/cache/april-sealed.npz',['tinychess.prepare_research_data','data/raw/lichess_elite_2023-04.zip','--output','research/cache/april-sealed.npz','--modulus','96']),
      ('research/cache/classical-sealed.npz',['tinychess.prepare_research_data','data/raw/classical-2025-sealed.pgn','--output','research/cache/classical-sealed.npz','--corpus','classical'])]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for output,args in jobs:
        if pathlib.Path(output).exists() and pathlib.Path(output).with_suffix('.json').exists():continue
        subprocess.run([sys.executable,'-m',*args],env=env,check=True)

if __name__=='__main__':main()
