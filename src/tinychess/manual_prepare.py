"""Recreate only the two published regression benchmarks; no training data."""
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from .manual_weights import checksum
from .round2_prepare import classical_subset

SOURCES = {
    'lichess_elite_2023-05.zip':('https://database.nikonoel.fr/lichess_elite_2023-05.zip','5386aa71165a631c483ac8f0d66158d60afd91e1fe9efc2c262dbe66498a759a'),
    'broadcast-2025-02.pgn.zst':('https://database.lichess.org/broadcast/lichess_db_broadcast_2025-02.pgn.zst','f92d0fea95e6c733898a920c0790a8c02f3f8a1e1c9ca0c42f15330cd5430434'),
    'broadcast-2025-03.pgn.zst':('https://database.lichess.org/broadcast/lichess_db_broadcast_2025-03.pgn.zst','a0a6bda3a00ddc34cfc15f8297c2c63081f43de0c2dd2ba16f6183cd84b34923'),
}


def main():
    root=Path('data/raw');root.mkdir(parents=True,exist_ok=True)
    for name,(url,expected) in SOURCES.items():
        path=root/name
        if not path.exists():
            temporary=path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as output:
                while chunk:=response.read(1024*1024):output.write(chunk)
            assert checksum(temporary)==expected, 'Downloaded source checksum mismatch'
            temporary.replace(path)
        assert checksum(path)==expected, 'Source checksum mismatch'
    classical_subset()
    assert checksum(root/'classical-round2-sealed.pgn')=='bd24fbdbd28aca580f2f8262e0098f0c0fcd0f3f3e9cfe1e3890f57591cebf60'
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for name,args in [('may-sealed',[str(root/'lichess_elite_2023-05.zip'),'--modulus','96']),
                      ('classical-sealed',[str(root/'classical-round2-sealed.pgn'),'--corpus','classical'])]:
        output=Path('research/round2/cache')/(name+'.npz')
        if not (output.exists() and output.with_suffix('.json').exists()):
            subprocess.run([sys.executable,'-m','tinychess.prepare_research_data',*args,'--output',str(output)],check=True,env=env)
    print('Three archive hashes and classical selection verified; benchmark caches available.')


if __name__=='__main__':main()
