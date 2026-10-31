"""Final paired comparison on sealed test sets. No fitting or model selection."""
import hashlib,json,pathlib
import numpy as np
from .research_rich import lightweight
from .evaluate import params,summarize
from .train import predict
from .model import Ranker

def model_params(path):
    r=Ranker(path);out=[]
    for w,b in zip(r.weights,r.biases):out.extend([w,b])
    return r.artifact,out

def paired_interval(before,after,groups):
    _,ids=np.unique(groups,return_inverse=True);n=np.bincount(ids);s=np.bincount(ids,weights=after.astype(float)-before.astype(float));rng=np.random.default_rng(9384);samples=[]
    for _ in range(2000):
        chosen=rng.integers(len(n),size=len(n));samples.append(s[chosen].sum()/n[chosen].sum())
    return np.quantile(samples,[.025,.975]).tolist()

def main():
    frozen=json.loads(pathlib.Path('research/frozen.json').read_text())
    assert hashlib.sha256(pathlib.Path('models/origin-ranker.int8.json').read_bytes()).hexdigest()==frozen['int8_sha256']
    assert hashlib.sha256(pathlib.Path('models/origin-ranker.json').read_bytes()).hexdigest()==frozen['float32_sha256']
    output=pathlib.Path('research/final-evaluation.json')
    if output.exists():raise ValueError('Final results already recorded; do not repeatedly tune against this test')
    development=lightweight('data/processed/feb.npz');extra=lightweight('research/cache/january-extra.npz')
    games=set(development['game'])|set(extra['game']);keys=set(development['key'])|set(extra['key'])
    model_specs={
      'baseline_int8':model_params('research/baseline/origin-ranker.int8.json'),
      'improved_float32':model_params('models/origin-ranker.json'),
      'improved_int8':model_params('models/origin-ranker.int8.json')}
    report={'frozen':frozen,'test_sets':{},'novel_definition':'Position + destination absent from all February development and January extra-training records.'}
    for name,path in [('fresh_april_2023','research/cache/april-sealed.npz'),('fresh_classical_2025','research/cache/classical-sealed.npz')]:
        d=lightweight(path);idx=np.array([i for i,g in enumerate(d['game']) if g not in games]);novel=np.array([k not in keys for k in d['key'][idx]])
        result={'removed_development_game_examples':len(d['y'])-len(idx),'models':{},'novel':{}};predictions={}
        for label,(art,pars) in model_specs.items():
            dd=dict(d)
            if art['feature_version']==1:dd['x']=d['x'][:,:107]
            p=predict(dd,idx,'history',pars);predictions[label]=p.argmax(1)
            result['models'][label]=summarize(d,idx,p);result['novel'][label]=summarize(d,idx[novel],p[novel])
        old=predictions['baseline_int8']==d['y'][idx];new=predictions['improved_int8']==d['y'][idx]
        result['paired_accuracy_gain']=float(new.mean()-old.mean());result['paired_gain_95pct_game_bootstrap']=paired_interval(old,new,d['game'][idx])
        result['relative_error_reduction']=float((new.mean()-old.mean())/(1-old.mean()))
        result['source_manifest']=json.loads(pathlib.Path(path).with_suffix('.json').read_text());report['test_sets'][name]=result
        print(name,json.dumps({k:v['accuracy'] for k,v in result['models'].items()}),flush=True)
    output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
