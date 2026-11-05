"""Download pinned study inputs with SHA-256 verification (run from repo root)."""
import argparse,hashlib,json,pathlib,urllib.request


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',action='store_true');p.add_argument('--size',choices=['5m','23m','all'],default='5m');a=p.parse_args()
    sources=json.loads(pathlib.Path('research/fullboard/sources.json').read_text())
    for item in sources['downloads']:
        path=pathlib.Path(item['path'])
        if path.suffix!='.pt' and not a.data:continue
        if path.suffix=='.pt' and a.size!='all' and not path.name.endswith('-'+a.size+'.pt'):continue
        path.parent.mkdir(parents=True,exist_ok=True)
        candidate=path
        if not path.exists():
            candidate=path.with_suffix(path.suffix+'.part')
            urllib.request.urlretrieve(item['url'],candidate)
        with candidate.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=item['sha256']:raise ValueError(f'Checksum mismatch: {candidate}')
        if candidate!=path:candidate.replace(path)
        print(f'Verified {path}',flush=True)


if __name__=='__main__':main()
