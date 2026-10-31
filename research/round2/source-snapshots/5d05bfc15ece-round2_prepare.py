"""Prepare a new training month and sealed tests without scoring any model."""
import collections
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request
from .data import open_pgn, pgn_records

ROOT = pathlib.Path('research/round2')
SOURCES = {
    'lichess_elite_2022-12.zip':'https://database.nikonoel.fr/lichess_elite_2022-12.zip',
    'lichess_elite_2023-05.zip':'https://database.nikonoel.fr/lichess_elite_2023-05.zip',
    'broadcast-2025-02.pgn.zst':'https://database.lichess.org/broadcast/lichess_db_broadcast_2025-02.pgn.zst',
    'broadcast-2025-03.pgn.zst':'https://database.lichess.org/broadcast/lichess_db_broadcast_2025-03.pgn.zst',
}
ALLOWED = {'PICF 2025 - Masters', 'PICF 2025 - Challengers', "2025 Cyprus FIDE Women's Grand Prix"}


def classical_subset():
    counts, excluded = collections.Counter(), collections.Counter()
    output = pathlib.Path('data/raw/classical-round2-sealed.pgn')
    with output.open('w') as out:
        for name in ['broadcast-2025-02.pgn.zst', 'broadcast-2025-03.pgn.zst']:
            with open_pgn(pathlib.Path('data/raw')/name) as stream:
                for raw in pgn_records(stream):
                    h = dict(re.findall(r'^\[(\w+) "(.*)"\]', raw, re.M))
                    event = h.get('Event', '')
                    if event == 'PICF 2025 - Challengers - Tiebreak':
                        excluded[event] += 1
                    if event not in ALLOWED:
                        continue
                    if re.search('tie.?break|rapid|blitz', h.get('StudyName', ''), re.I):
                        excluded[event] += 1
                        continue
                    counts[event] += 1
                    out.write(raw.rstrip()+'\n\n')
    (ROOT/'classical-sources.json').write_text(json.dumps(dict(events=dict(counts), excluded=dict(excluded),
        selection='Exact event allowlist chosen from headers before model evaluation.',
        license='Lichess broadcasts, CC BY-SA 4.0'), indent=2)+'\n')


def main():
    (ROOT/'cache').mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = pathlib.Path('data/raw')/name
        if path.exists():
            continue
        temporary = path.with_suffix(path.suffix+'.part')
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
            while chunk := response.read(1048576):
                out.write(chunk)
        temporary.replace(path)
    (ROOT/'source-urls.json').write_text(json.dumps(SOURCES, indent=2)+'\n')
    classical_subset()
    jobs = [
        ('may-sealed', ['data/raw/lichess_elite_2023-05.zip', '--modulus', '96']),
        ('classical-sealed', ['data/raw/classical-round2-sealed.pgn', '--corpus', 'classical']),
        ('december-extra', ['data/raw/lichess_elite_2022-12.zip', '--modulus', '24', '--extra-training'])]
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for name, args in jobs:
        output = ROOT/'cache'/(name+'.npz')
        if output.exists() and output.with_suffix('.json').exists():
            continue
        subprocess.run([sys.executable, '-m', 'tinychess.prepare_research_data', *args, '--output', str(output)], check=True, env=env)


if __name__ == '__main__':
    main()
