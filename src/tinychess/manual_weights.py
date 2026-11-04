"""Compile explicitly chosen chess coefficients into ordinary MLP weight arrays.

No optimizer, game data, labels, trained weights, or random initialization is used.
All numbers below are authored before benchmark scoring. This is a heuristic
expressed as a ReLU neural network, not evidence of learning without training.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from .tactical import NAMES

ROOT = Path('research/manual-weights')

# Features are normalized by the existing v2 extractor. Factors 12, 32 and 40
# below undo those normalizations; the other factors are manually chosen scores.
BASE = {
    'captured_value': 12 * 1.00,
    'value': -12 * 0.08,
    'piece_6': 12 * 0.08,  # Undo the arbitrary king "material value" of 12.
    'promotion': 8.0,
    'castle': 2.5,
    'gives_check': 0.60,
    'before_mobility': -32 * 0.035,
    'after_mobility': 32 * 0.035,
    'before_targets': -8 * 0.10,
    'after_targets': 8 * 0.10,
    'after_max_target': 12 * 0.05,
    'from_center': 0.20,
    'to_center': -0.20,
    'delta_us_sum_risk': -40 * 0.60,
    'after_us_max_risk': -12 * 1.25,
    'after_us_second_risk': -12 * 0.30,
    'after_them_max_risk': 12 * 0.65,
    'after_them_second_risk': 12 * 0.25,
}

# Each row is (explanation, affine input coefficients, bias, output coefficient).
# ReLU gates introduce early-development and piece-specific nonlinear preferences.
GATES = [
    ('Develop a back-rank knight early', {'piece_2':1,'from_rank_0':1,'ply':-4}, -1, 1.40),
    ('Develop a back-rank bishop early', {'piece_3':1,'from_rank_0':1,'ply':-4}, -1, 1.20),
    ('Avoid early queen moves', {'piece_5':1,'ply':-8}, 0, -0.90),
    ('Keep a corner rook while both castling rights remain',
     {'piece_4':1,'our_oo':1,'our_ooo':1,'ply':-4}, -2, -0.60),
    ('Avoid moving the same piece twice in the opening', {'past_2_same_piece':1,'ply':-5}, 0, -0.55),
    ('Encourage the d-pawn center push', {'piece_1':1,'to_file_3':1,'to_rank_3':1,'ply':-4}, -2, 0.55),
    ('Encourage the e-pawn center push', {'piece_1':1,'to_file_4':1,'to_rank_3':1,'ply':-4}, -2, 0.55),
    ('Activate the king when opposing major pieces are absent',
     {'piece_6':1,'them_count_5':-8,'them_count_4':-4}, 0, 0.65),
    ('Discourage king moves with an opposing queen',
     {'piece_6':4,'them_count_5':8}, -4, -0.90),
    ('Pawn advance to the sixth rank', {'piece_1':1,'to_rank_5':1}, -1, 0.20),
    ('Pawn advance to the seventh rank', {'piece_1':1,'to_rank_6':1}, -1, 0.45),
]


def vector(coefficients):
    out = np.zeros(len(NAMES), np.float32)
    for name, value in coefficients.items():
        out[NAMES.index(name)] = value
    return out


def build(nonlinear=True):
    base = vector(BASE)
    if nonlinear:
        # ReLU(z)-ReLU(-z)=z preserves the signed base score exactly.
        columns = [base, -base] + [vector(g[1]) for g in GATES]
        weights = [np.stack(columns, axis=1), np.array([1,-1]+[g[3] for g in GATES],np.float32)[:,None]]
        biases = [np.array([0,0]+[g[2] for g in GATES],np.float32), np.zeros(1,np.float32)]
    else:
        weights, biases = [base[:,None]], [np.zeros(1,np.float32)]
    return {
        'feature_version':2, 'representation':'history', 'architecture':'mlp',
        'hidden':[len(GATES)+2] if nonlinear else [],
        'parameter_count':sum(w.size+b.size for w,b in zip(weights,biases)),
        'nonzero_parameter_count':int(sum(np.count_nonzero(w)+np.count_nonzero(b) for w,b in zip(weights,biases))),
        'weights':[w.tolist() for w in weights], 'biases':[b.tolist() for b in biases],
        'provenance':'Manually authored chess rules; zero training steps; no coefficient search or test-guided revision.',
    }


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    target=ROOT/'frozen.json'
    if target.exists():
        raise ValueError('Weights already frozen; do not revise them after seeing test results')
    ROOT.mkdir(parents=True,exist_ok=True)
    files={}
    for name,nonlinear in [('manual-relu',True),('manual-linear',False)]:
        artifact=build(nonlinear)
        # Same generic Python/JS ranker supports this JSON; no custom scorer needed.
        data=(json.dumps(artifact,separators=(',',':'))+'\n').encode()
        path=ROOT/(name+'.json');path.write_bytes(data)
        compressed=gzip.compress(data,mtime=0);path.with_suffix('.json.gz').write_bytes(compressed)
        files[name]={'file':str(path),'sha256':checksum(path),
                     'parameter_count':artifact['parameter_count'],
                     'nonzero_parameter_count':int(artifact['nonzero_parameter_count']),
                     'float32_parameter_bytes':artifact['parameter_count']*4,
                     'json_bytes':len(data),'json_gzip_bytes':len(compressed)}
    contract={'primary':'manual-relu','ablation':'manual-linear','training_steps':0,
        'selection':'One predeclared nonlinear model and its linear ablation; no model selection or test-feedback revisions.',
        'builder_sha256':checksum(__file__),
        'incumbent_file':'models/origin-ranker.bin','incumbent_sha256':checksum('models/origin-ranker.bin'),
        'feature_names':NAMES,'base':BASE,
        'gates':[dict(reason=g[0],coefficients=g[1],bias=g[2],output=g[3]) for g in GATES],
        'artifacts':files,
        'benchmarks':{'online':'May 2023 Lichess Elite, raw-PGN SHA-256 modulo 96, both Elo >=2100',
                      'classical':'The existing 2025 Prague Masters/Challengers and Cyprus Women Grand Prix allowlist; tiebreaks excluded'},
        'status':'Frozen before this experiment scored any benchmark labels. These are previously published regression benchmarks, not newly unseen test sets.'}
    target.write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps(files,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze']);parser.parse_args();freeze()
