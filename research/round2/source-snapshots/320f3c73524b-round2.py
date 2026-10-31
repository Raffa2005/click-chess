"""Second bounded study, preserving all earlier models and evaluation reports."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import traceback
import numpy as np
from .research import run, fingerprint
from .research_rich import prepare, lightweight
from .comparative import transform_dataset

ROOT = pathlib.Path('research/round2')


def data(extra=False, version=2):
    d, tr, val = prepare(True)
    if extra:
        e = lightweight('research/round2/cache/december-extra.npz')
        known = set(d['game'])
        assert not set(e['key']) & set(d['key'][val])
        ids = np.flatnonzero(np.array([g not in known for g in e['game']]))
        rows = np.concatenate([np.arange(e['offsets'][i], e['offsets'][i+1]) for i in ids])
        count, row_count = len(d['y']), len(d['x'])
        d['x'] = np.concatenate([d['x'], e['x'][rows]])
        d['offsets'] = np.r_[d['offsets'], row_count + np.cumsum(e['n'][ids])]
        for key in ['y', 'game', 'key', 'ply', 'kind', 'n']:
            d[key] = np.concatenate([d[key], e[key][ids]])
        tr = np.r_[tr, np.arange(count, len(d['y']))]
    if version == 3:
        d['x'] = transform_dataset(d['x'], d['offsets'])
    return d, tr, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment', required=True)
    args = ap.parse_args()
    config = next(c for c in json.loads((ROOT/'experiments.json').read_text()) if c['name'] == args.experiment)
    output = ROOT/'runs'/(config['name']+'.json')
    if output.exists():
        raise ValueError('Experiment already recorded')
    try:
        d, tr, val = data(config.get('december', False), config.get('feature_version', 2))
        artifact, report = run(d, tr, val, config)
        report['source_hashes'] = {p: fingerprint(p) for p in [__file__, 'src/tinychess/research.py', 'src/tinychess/comparative.py']}
        report['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        report['validation_indices_sha256'] = hashlib.sha256(val.tobytes()).hexdigest()
        report['development_sources'] = ['data/processed/feb.npz', 'research/cache/january-extra.npz']
        if config.get('december'):
            report['development_sources'].append('research/round2/cache/december-extra.npz')
        report['data_manifests'] = {p: json.loads(pathlib.Path(p).with_suffix('.json').read_text()) for p in report['development_sources']}
        prior = [json.loads(p.read_text()) for p in (ROOT/'runs').glob('*.json')]
        baseline = json.loads((ROOT/'baseline/frozen.json').read_text())['float32_validation']['nll']
        best = min([r['validation']['nll'] for r in prior if 'validation' in r] + [baseline])
        report['status'] = 'keep' if report['validation']['nll'] < best else 'discard'
        (ROOT/'checkpoints'/(config['name']+'.json')).write_text(json.dumps(artifact, separators=(',', ':'))+'\n')
        output.write_text(json.dumps(report, indent=2)+'\n')
        with (ROOT/'results.tsv').open('a') as ledger:
            ledger.write(f"{config['name']}\t{report['status']}\t{report['validation']['nll']:.7f}\t{report['validation']['accuracy']:.7f}\t{report['parameter_count']}\t{report['seconds']:.1f}\n")
        print('RESULT', json.dumps({k:v for k,v in report.items() if k not in ('curve', 'data_manifests')}), flush=True)
    except Exception as error:
        output.write_text(json.dumps({'config':config, 'status':'crash', 'error':str(error), 'traceback':traceback.format_exc()}, indent=2)+'\n')
        with (ROOT/'results.tsv').open('a') as ledger:
            ledger.write(f"{config['name']}\tcrash\t\t\t\t\t{error}\n")
        raise


if __name__ == '__main__':
    main()
