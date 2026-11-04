"""One frozen, paired comparison of hand-written and trained weights."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from .evaluate import summarize
from .features import feature_names
from .manual_weights import ROOT, checksum
from .model import Ranker
from .research_evaluate import paired_interval
from .research_rich import lightweight
from .train import predict, metric


def verify():
    frozen=json.loads((ROOT/'frozen.json').read_text())
    assert checksum(Path(__file__).with_name('manual_weights.py'))==frozen['builder_sha256']
    assert checksum(frozen['incumbent_file'])==frozen['incumbent_sha256']
    for artifact in frozen['artifacts'].values():
        assert checksum(artifact['file'])==artifact['sha256']
    return frozen


def predictions(data, path):
    model=Ranker(path)
    params=[p for pair in zip(model.weights,model.biases) for p in pair]
    return predict(data,np.arange(len(data['y'])),'history',params)


def evaluate(output=ROOT/'evaluation.json'):
    output=Path(output)
    if output.exists():
        raise ValueError('Results already recorded; this experiment forbids test-feedback tuning')
    frozen=verify();started=time.perf_counter()
    previous=json.loads(Path('research/round2/final-evaluation.json').read_text())['test_sets']
    models={'manual_relu':frozen['artifacts']['manual-relu']['file'],
            'manual_linear':frozen['artifacts']['manual-linear']['file'],
            'trained_int8':frozen['incumbent_file']}
    report={'frozen_sha256':checksum(ROOT/'frozen.json'),'freeze_commit':'240eaf64f3881f01f8bce8d0a4e964e25e4e9d85',
            'training_steps':0,'weights_changed_after_scoring':False,
            'test_status':'Previously published held-out regression sets, exactly reconstructed; not new unseen tests.',
            'runtime':{'python':platform.python_version(),'numpy':np.__version__},'sets':{}}
    for name,cache,old_name in [
        ('online_2100plus','may-sealed','fresh_may_2023'),
        ('classical_2100plus','classical-sealed','fresh_classical_2025')]:
        path=Path('research/round2/cache')/(cache+'.npz')
        data=lightweight(path);meta=json.loads(path.with_suffix('.json').read_text());old=previous[old_name]
        assert meta['source_sha256']==old['source_manifest']['source_sha256']
        assert meta['counts']==old['source_manifest']['counts']
        assert old['removed_development_game_examples']==0
        assert len(data['y'])==old['models']['incumbent_int8']['examples']
        assert len(set(data['game']))==old['models']['incumbent_int8']['games']
        ids=np.arange(len(data['y']));n=data['n'];y=data['y']
        first=data['offsets'][:-1]
        least=np.array([data['x'][a:b,feature_names().index('value')].argmin() for a,b in zip(first,data['offsets'][1:])])
        same=np.array([len(set(data['x'][a:b,:6].argmax(1)))==1 for a,b in zip(first,data['offsets'][1:])])
        slices={'same_piece_type':same,'different_piece_types':~same,'two_origins':n==2,'three_plus_origins':n>=3,
                'opening_under20ply':data['ply']<20,'middle_20to79ply':(data['ply']>=20)&(data['ply']<80),'late_80plusply':data['ply']>=80}
        result={'source_manifest':meta,'models':{},'slices':{},'baselines':{
            'uniform_expected_accuracy':float((1/n).mean()),'first_origin_accuracy':float((y==0).mean()),
            'least_value_accuracy':float((least==y).mean())}}
        correctness={}
        for label,model_path in models.items():
            probs=predictions(data,model_path)
            assert np.isfinite(probs).all() and np.allclose(probs.sum(1),1)
            assert all(np.all(probs[i,count:]==0) for i,count in enumerate(n))
            result['models'][label]=summarize(data,ids,probs)
            result['slices'][label]={key:{'examples':int(mask.sum()),**metric(probs[mask],y[mask])} for key,mask in slices.items() if mask.any()}
            correctness[label]=probs.argmax(1)==y
        assert result['models']['trained_int8']['accuracy']==old['models']['incumbent_int8']['accuracy'], 'Incumbent accuracy must reproduce the original benchmark exactly'
        manual,trained=correctness['manual_relu'],correctness['trained_int8']
        result['paired_manual_minus_trained_accuracy']=float(manual.mean()-trained.mean())
        result['paired_manual_minus_trained_95pct_game_bootstrap']=paired_interval(trained,manual,data['game'])
        result['disagreements']={'manual_only_correct':int((manual&~trained).sum()),'trained_only_correct':int((trained&~manual).sum())}
        result['incumbent_accuracy_exactly_reproduced']=True
        report['sets'][name]=result
        print(name,json.dumps({key:{'accuracy':v['accuracy'],'top2':v['top2'],'nll':v['nll']} for key,v in result['models'].items()}),flush=True)
    verify();report['evaluation_seconds']=time.perf_counter()-started
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'evaluation.json'))
    evaluate(parser.parse_args().output)
