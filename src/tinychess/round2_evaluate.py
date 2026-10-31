"""Freeze on validation, evaluate once, and apply a predeclared adoption rule."""
import argparse
import gzip
import hashlib
import json
import pathlib
import shutil
import numpy as np
from .binary import encode, decode
from .comparative import INDICES
from .evaluate import summarize
from .freeze import quantize
from .model import Ranker
from .research import fingerprint
from .research_rich import prepare, lightweight
from .research_evaluate import paired_interval
from .round2 import ROOT
from .round2_bigdata import data
from .train import batch, forward, probabilities, metric


def predict(d, idx, artifact):
    model = Ranker(artifact)
    params = [p for pair in zip(model.weights, model.biases) for p in pair]
    out = np.zeros((len(idx), int(d['n'][idx].max())), np.float32)
    for start in range(0, len(idx), 1024):
        ids = idx[start:start+1024]
        x, valid = batch(d, ids, 'history')
        if artifact['feature_version'] == 1:
            x = x[:, :, :107]
        elif artifact['feature_version'] == 3:
            mean = (x[:, :, INDICES]*valid[:, :, None]).sum(1)/valid.sum(1)[:, None]
            x = np.concatenate([x, np.broadcast_to(mean[:, None, :], (*x.shape[:2], len(INDICES)))], axis=2).astype(np.float32)
        if artifact.get('architecture') == 'set-context':
            from .setnet import forward as set_forward
            scores, _ = set_forward(x, valid, params, artifact.get('pooling', 'mean'))
        else:
            scores, _ = forward(x.reshape(-1, x.shape[-1]), params)
            scores = scores.reshape(valid.shape)
        out[start:start+len(ids), :valid.shape[1]] = probabilities(scores, valid)
    return out


def freeze(name):
    target = ROOT/'frozen.json'
    if target.exists():
        raise ValueError('Already frozen; do not revise using test feedback')
    path = ROOT/'checkpoints'/(name+'.json')
    artifact = json.loads(path.read_text())
    d, _, val = prepare(False)
    float_metrics = metric(predict(d, val, artifact), d['y'][val])
    options = []
    for per_output in [False, True]:
        packed, dequantized = quantize(artifact, per_output)
        binary = encode(packed)
        decoded = decode(binary)
        p = predict(d, val, decoded)
        reference = predict(d, val, packed)
        np.testing.assert_array_equal(p, reference)
        met = metric(p, d['y'][val])
        text = json.dumps(packed, separators=(',', ':'))+'\n'
        report = dict(quantization=packed['quantization'], validation=met, binary_bytes=len(binary),
                      binary_gzip_bytes=len(gzip.compress(binary, mtime=0)), json_bytes=len(text),
                      json_gzip_bytes=len(gzip.compress(text.encode(), mtime=0)))
        report['eligible'] = artifact['parameter_count'] <= 20000 and report['binary_gzip_bytes'] <= 24576 and met['accuracy'] >= float_metrics['accuracy']-.002
        options.append((report, text, binary))
    choices = [x for x in options if x[0]['eligible']]
    if not choices:
        raise ValueError('No deployment export passes the validation and size constraints')
    chosen, text, binary = min(choices, key=lambda x:x[0]['validation']['nll'])
    directory = ROOT/'candidate'
    directory.mkdir(exist_ok=True)
    (directory/'origin-ranker.json').write_bytes(path.read_bytes())
    (directory/'origin-ranker.int8.json').write_text(text)
    (directory/'origin-ranker.int8.json.gz').write_bytes(gzip.compress(text.encode(), mtime=0))
    (directory/'origin-ranker.bin').write_bytes(binary)
    (directory/'origin-ranker.bin.gz').write_bytes(gzip.compress(binary, mtime=0))
    info = dict(selected_experiment=name, parameter_count=artifact['parameter_count'], float32_validation=float_metrics,
                quantization_options=[x[0] for x in options], selected_quantization=chosen,
                hashes={p.name:fingerprint(p) for p in directory.iterdir()},
                note='Checkpoint and export frozen using the original validation split before any round-two test scoring.')
    target.write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))


def evaluate():
    output = ROOT/'final-evaluation.json'
    if output.exists():
        raise ValueError('Test results already recorded')
    frozen = json.loads((ROOT/'frozen.json').read_text())
    for name, sha in frozen['hashes'].items():
        assert fingerprint(ROOT/'candidate'/name) == sha
    development, _, _ = data(True, 2)
    games, keys = set(development['game']), set(development['key'])
    specs = {
        'incumbent_int8':json.loads((ROOT/'baseline/origin-ranker.int8.json').read_text()),
        'candidate_float32':json.loads((ROOT/'candidate/origin-ranker.json').read_text()),
        'candidate_int8':decode((ROOT/'candidate/origin-ranker.bin').read_bytes())}
    report = dict(frozen=frozen, test_sets={}, regression_sets={}, novel_definition='Absent from all December/January training and February development position+destination keys.')
    tests = [
        ('test_sets', 'fresh_may_2023', ROOT/'cache/may-sealed.npz'),
        ('test_sets', 'fresh_classical_2025', ROOT/'cache/classical-sealed.npz'),
        ('regression_sets', 'april_2023', pathlib.Path('research/cache/april-sealed.npz')),
        ('regression_sets', 'tata_steel_2025', pathlib.Path('research/cache/classical-sealed.npz'))]
    for group, name, path in tests:
        d = lightweight(path)
        ids = np.array([i for i,g in enumerate(d['game']) if g not in games])
        novel = np.array([k not in keys for k in d['key'][ids]])
        result = dict(removed_development_game_examples=len(d['y'])-len(ids), models={}, novel={})
        correct = {}
        for label, artifact in specs.items():
            p = predict(d, ids, artifact)
            result['models'][label] = summarize(d, ids, p)
            result['novel'][label] = summarize(d, ids[novel], p[novel])
            correct[label] = p.argmax(1) == d['y'][ids]
        old, new = correct['incumbent_int8'], correct['candidate_int8']
        result['paired_accuracy_gain'] = float(new.mean()-old.mean())
        result['paired_gain_95pct_game_bootstrap'] = paired_interval(old, new, d['game'][ids])
        result['source_manifest'] = json.loads(path.with_suffix('.json').read_text())
        report[group][name] = result
        print(group, name, json.dumps({k:v['accuracy'] for k,v in result['models'].items()}), flush=True)
    output.write_text(json.dumps(report, indent=2)+'\n')


def adopt():
    report = json.loads((ROOT/'final-evaluation.json').read_text())
    latency = json.loads((ROOT/'latency-candidate.json').read_text())
    improved = all(r['paired_accuracy_gain'] > 0 for r in report['test_sets'].values())
    fast = latency['rank_with_features_legal_moves_supplied']['p95_us'] < 1000
    decision = dict(adopted=improved and fast, accuracy_improved_on_both_fresh_tests=improved, latency_passed=fast,
                    rule='Promote only if top-1 accuracy improves on both fresh tests and ranking p95 remains below 1 ms; no retraining using these results.')
    if decision['adopted']:
        for path in (ROOT/'candidate').iterdir():
            shutil.copyfile(path, pathlib.Path('models')/path.name)
        shutil.copyfile(ROOT/'latency-candidate.json', 'results/latency-node.json')
    else:
        # The useful lossless packing change can ship independently of a model.
        incumbent=json.loads(pathlib.Path('models/origin-ranker.int8.json').read_text())
        binary=encode(incumbent)
        pathlib.Path('models/origin-ranker.bin').write_bytes(binary)
        pathlib.Path('models/origin-ranker.bin.gz').write_bytes(gzip.compress(binary,mtime=0))
        assert fingerprint('models/origin-ranker.int8.json')==fingerprint(ROOT/'baseline/origin-ranker.int8.json')
    decision['lossless_binary_export']=True
    decision['deployment_sizes']={name:pathlib.Path('models',name).stat().st_size for name in ['origin-ranker.bin','origin-ranker.bin.gz','origin-ranker.int8.json.gz']}
    (ROOT/'adoption.json').write_text(json.dumps(decision, indent=2)+'\n')
    print(json.dumps(decision))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'evaluate', 'adopt'])
    parser.add_argument('--experiment')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.experiment)
    elif args.command == 'evaluate':
        evaluate()
    else:
        adopt()


if __name__ == '__main__':
    main()
