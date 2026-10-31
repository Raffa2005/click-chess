"""Download pinned historical sources and select known classical events."""
import collections,json,pathlib,re,urllib.request
from .data import open_pgn,pgn_records

SOURCES={
 'lichess_elite_2023-02.zip':'https://database.nikonoel.fr/lichess_elite_2023-02.zip',
 'lichess_elite_2023-03.zip':'https://database.nikonoel.fr/lichess_elite_2023-03.zip',
 'broadcast-2024-01.pgn.zst':'https://database.lichess.org/broadcast/lichess_db_broadcast_2024-01.pgn.zst',
 'broadcast-2024-04.pgn.zst':'https://database.lichess.org/broadcast/lichess_db_broadcast_2024-04.pgn.zst',
}
CLASSICAL_EVENTS={'Tata Steel Masters 2024','Tata Steel Challengers','Tata Steel Toptienkamp 2024','FIDE Candidates 2024',"FIDE Women's Candidates 2024"}

def select_classical(root):
    counts=collections.Counter();excluded=0
    with (root/'classical-2024.pgn').open('w',encoding='utf-8') as out:
        for name in ['broadcast-2024-01.pgn.zst','broadcast-2024-04.pgn.zst']:
            with open_pgn(root/name) as stream:
                for raw in pgn_records(stream):
                    h=dict(re.findall(r'^\[(\w+) "(.*)"\]',raw,re.M))
                    if h.get('Event') not in CLASSICAL_EVENTS:continue
                    if re.search(r'tie.?break|rapid|blitz',h.get('StudyName',''),re.I):excluded+=1;continue
                    # Extra guard against Masters playoff rounds with the same Event tag.
                    if h['Event']=='Tata Steel Masters 2024' and int(h['Round'].split('.')[0])>13:excluded+=1;continue
                    out.write(raw.rstrip()+'\n\n');counts[h['Event']]+=1
    report={'source_urls':SOURCES,'classical_events':dict(counts),'tiebreaks_excluded':excluded,
        'licenses':{'elite_online':'CC0 (Lichess original games)','broadcasts':'CC BY-SA 4.0; Lichess broadcasts, https://database.lichess.org/#broadcasts'}}
    pathlib.Path('results/sources.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

def main():
    root=pathlib.Path('data/raw');root.mkdir(parents=True,exist_ok=True)
    for name,url in SOURCES.items():
        p=root/name
        if p.exists():continue
        temp=p.with_suffix(p.suffix+'.part')
        print('Downloading',url,flush=True)
        with urllib.request.urlopen(url,timeout=60) as r,temp.open('wb') as f:
            while chunk:=r.read(1024*1024):f.write(chunk)
        temp.replace(p)
    pathlib.Path('results').mkdir(exist_ok=True);select_classical(root)

if __name__=='__main__':main()
