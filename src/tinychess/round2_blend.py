"""Validate a fixed grid of aligned weight averages, without larger inference."""
import copy,json,time
import numpy as np
from .round2 import ROOT
from .round2_evaluate import predict
from .research_rich import prepare
from .research import fingerprint
from .train import metric

def main():
    d,_,val=prepare(False);configs=json.loads((ROOT/'experiments.json').read_text())
    for family,first,second in [('refine','december-incumbent','incumbent-refine'),('deep','december-incumbent','december-deeper-head')]:
        paths=[ROOT/'checkpoints'/(name+'.json') for name in [first,second]]
        a,b=[json.loads(path.read_text()) for path in paths]
        if family=='deep':
            a=copy.deepcopy(a);a['weights'].insert(2,np.eye(32).tolist());a['biases'].insert(2,[0.]*32)
            a['hidden']=[96,32,32];a['parameter_count']=18401
        for fraction in [.25,.5,.75]:
            name=f'blend-{family}-{int(fraction*100)}';output=ROOT/'runs'/(name+'.json')
            if output.exists():continue
            start=time.perf_counter();candidate=copy.deepcopy(a)
            for key in ['weights','biases']:
                candidate[key]=[((1-fraction)*np.asarray(x,np.float32)+fraction*np.asarray(y,np.float32)).tolist() for x,y in zip(a[key],b[key])]
            candidate['research_experiment']=name;p=predict(d,val,candidate);metrics=metric(p,d['y'][val])
            metrics['top2']=float(np.any(np.argsort(-p,axis=1,kind='stable')[:,:2]==d['y'][val,None],axis=1).mean())
            config=dict(name=name,kind='weight_blend',parents=[first,second],second_fraction=fraction,hidden=candidate['hidden'],feature_version=2,hypothesis='Average aligned weights in nearby fine-tuning solutions without adding deployed parameters.')
            if not any(c['name']==name for c in configs):configs.append(config)
            prior=[json.loads(path.read_text()) for path in (ROOT/'runs').glob('*.json')];best=min(r['validation']['nll'] for r in prior if 'validation' in r)
            report=dict(config=config,validation=metrics,parameter_count=candidate['parameter_count'],best_epoch=0,curve=[],seconds=time.perf_counter()-start,validation_examples=len(val),train_examples=1227149,status='keep' if metrics['nll']<best else 'discard',source_hashes={__file__:fingerprint(__file__)},parent_sha256={str(path):fingerprint(path) for path in paths})
            output.write_text(json.dumps(report,indent=2)+'\n');(ROOT/'checkpoints'/(name+'.json')).write_text(json.dumps(candidate,separators=(',',':'))+'\n')
            with (ROOT/'results.tsv').open('a') as ledger:ledger.write(f"{name}\t{report['status']}\t{metrics['nll']:.7f}\t{metrics['accuracy']:.7f}\t{candidate['parameter_count']}\t{report['seconds']:.1f}\n")
            print(name,json.dumps(metrics),flush=True)
    (ROOT/'experiments.json').write_text(json.dumps(configs,indent=2)+'\n')

if __name__=='__main__':main()
