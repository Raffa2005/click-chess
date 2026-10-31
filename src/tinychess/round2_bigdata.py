"""Share the larger training feature array through a memory-mapped cache."""
import json
import os
import sys
import numpy as np
from . import round2
from .research import fingerprint

CACHE = round2.ROOT/'cache'
PREPARE = round2.data


def build():
    if (CACHE/'combined.json').exists():
        return
    lock = CACHE/'combined.building'
    try:
        descriptor = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError('Wait for the existing --prepare-only job before launching training') from error
    os.close(descriptor)
    try:
        d, tr, val = PREPARE(True, 2)
        np.save(CACHE/'combined-x.part.npy', d['x'])
        np.savez_compressed(CACHE/'combined-meta.part.npz', **{k:v for k,v in d.items() if k not in ('x', 'validation')}, train_indices=tr, validation_indices=val)
        metadata = dict(shape=list(d['x'].shape), train_examples=len(tr), validation_examples=len(val),
                        x_sha256=fingerprint(CACHE/'combined-x.part.npy'), metadata_sha256=fingerprint(CACHE/'combined-meta.part.npz'))
        (CACHE/'combined-x.part.npy').replace(CACHE/'combined-x.npy')
        (CACHE/'combined-meta.part.npz').replace(CACHE/'combined-meta.npz')
        (CACHE/'combined.part.json').write_text(json.dumps(metadata, indent=2)+'\n')
        (CACHE/'combined.part.json').replace(CACHE/'combined.json')
        print(json.dumps(metadata), flush=True)
    finally:
        lock.unlink(missing_ok=True)


def data(extra=False, version=2):
    if not extra:
        return PREPARE(False, version)
    if version != 2:
        raise ValueError('The shared cache is version 2')
    with np.load(CACHE/'combined-meta.npz') as archive:
        d = {k:archive[k] for k in archive.files}
    tr, val = d.pop('train_indices'), d.pop('validation_indices')
    d['x'] = np.load(CACHE/'combined-x.npy', mmap_mode='r')
    assert len(val) == 36024
    return d, tr, val


def main():
    build()
    if '--prepare-only' in sys.argv:
        return
    source_hash = fingerprint(__file__)
    round2.data = data
    round2.main()
    name = sys.argv[sys.argv.index('--experiment')+1]
    path = round2.ROOT/'runs'/(name+'.json')
    report = json.loads(path.read_text())
    report['source_hashes'][__file__] = source_hash
    report['shared_cache'] = json.loads((CACHE/'combined.json').read_text())
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
