"""Bounded, auditable CPU experiments; validation and test data stay separate."""
import argparse,hashlib,json,math,pathlib,subprocess,time,traceback
import numpy as np
from .train import load,batch,forward,probabilities,predict,metric
from .features import feature_names

ROOT=pathlib.Path('research')

def split(d):
    tr=np.flatnonzero(~d['validation']);keys=set(d['key'][tr])
    val=np.array([i for i in np.flatnonzero(d['validation']) if d['key'][i] not in keys],dtype=np.int64)
    assert len(val)==36024 and len(tr)==376508, 'The frozen baseline split changed'
    assert not set(d['game'][tr])&set(d['game'][val])
    return tr,val

def fingerprint(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fold_normalization(params,mean,std):
    out=[p.copy() for p in params]
    out[0]=params[0]/std[:,None]
    out[1]=params[1]-(mean/std)@params[0]
    return out

def artifact(params,c,dim):
    return {'format':'tiny-chess-input-v1','feature_version':c.get('feature_version',1),'representation':'history',
        'hidden':c['hidden'],'seed':c['seed'],'parameter_count':sum(p.size for p in params),
        'weights':[p.tolist() for p in params[::2]],'biases':[p.tolist() for p in params[1::2]],
        'research_experiment':c['name'],'feature_names':feature_names(True) if dim==107 else None}

def run(d,tr,val,c):
    rng=np.random.default_rng(c['seed']);dim=d['x'].shape[1];dims=[dim]+c['hidden']+[1]
    params=[]
    for a,b in zip(dims,dims[1:]):params += [(rng.normal(size=(a,b))*np.sqrt(2/a)).astype(np.float32),np.zeros(b,np.float32)]
    assert sum(p.size for p in params)<=20000
    mean=np.zeros(dim,np.float32);std=np.ones(dim,np.float32)
    if c.get('normalize'):
        # Only training rows contribute; never fit a transform on validation/test.
        rows=np.concatenate([np.arange(d['offsets'][i],d['offsets'][i+1]) for i in tr[::4]])
        mean=d['x'][rows].mean(0);std=np.maximum(d['x'][rows].std(0),.05)
    if c.get('init'):
        a=json.loads(pathlib.Path(c['init']).read_text());params=[]
        for w,b in zip(a['weights'],a['biases']):params.extend([np.array(w,np.float32),np.array(b,np.float32)])
        # The serialized initial model already has normalization folded in.
        original=params[0].copy();params[0]=std[:,None]*original;params[1]=params[1]+mean@original
    m=[np.zeros_like(p) for p in params];v=[np.zeros_like(p) for p in params]
    step=0;best_loss=1e9;best=None;stale=0;curve=[];start=time.perf_counter();bs=c.get('batch_size',512)
    for epoch in range(c['epochs']):
        fraction=min(1,max(epoch/max(1,c['epochs']-1),(time.perf_counter()-start)/c['seconds']))
        lr=c['lr']*(.05+.95*.5*(1+math.cos(math.pi*fraction))) if c.get('schedule')=='cosine' else c['lr']
        train_loss=0.;seen=0;order=rng.permutation(tr)
        for off in range(0,len(order),bs):
            ids=order[off:off+bs];x,valid=batch(d,ids,'history');flat=(x.reshape(-1,dim)-mean)/std
            scores,acts=forward(flat,params);p=probabilities(scores.reshape(valid.shape),valid)
            train_loss+=float(-np.log(np.maximum(p[np.arange(len(ids)),d['y'][ids]],1e-12)).sum());seen+=len(ids)
            p[np.arange(len(ids)),d['y'][ids]]-=1;grad=(p/len(ids)).reshape(-1,1);grads=[None]*len(params)
            for layer in reversed(range(len(dims)-1)):
                j=layer*2;grads[j]=acts[layer].T@grad+c.get('l2',1e-5)*params[j];grads[j+1]=grad.sum(0)
                if layer:grad=(grad@params[j].T)*(acts[layer]>0)
            step+=1
            for j,g in enumerate(grads):
                m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*g*g
                params[j]-=lr*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
            if time.perf_counter()-start>=c['seconds']:break
        folded=fold_normalization(params,mean,std)
        pred=predict(d,val,'history',folded);met=metric(pred,d['y'][val])
        row={'epoch':epoch+1,'lr':lr,'training_nll':train_loss/seen,'seconds':time.perf_counter()-start,**met};curve.append(row)
        print(json.dumps({'experiment':c['name'],**row}),flush=True)
        if met['nll']<best_loss-1e-5:best_loss=met['nll'];best=folded;best_epoch=epoch+1;stale=0
        else:stale+=1
        if stale>=c['patience'] or time.perf_counter()-start>=c['seconds']:break
    p=predict(d,val,'history',best);order=np.argsort(-p,axis=1,kind='stable')
    met=metric(p,d['y'][val]);met['top2']=float(np.any(order[:,:2]==d['y'][val,None],axis=1).mean())
    return artifact(best,c,dim),{'config':c,'best_epoch':best_epoch,'validation':met,'parameter_count':sum(p.size for p in best),
        'seconds':time.perf_counter()-start,'train_examples':len(tr),'validation_examples':len(val),'curve':curve}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',default='all');ap.add_argument('--data',default='data/processed/feb.npz');a=ap.parse_args()
    configs=json.loads((ROOT/'experiments.json').read_text());configs=[c for c in configs if a.experiment=='all' or c['name']==a.experiment]
    if not configs:raise ValueError('Unknown experiment')
    for p in ['runs','checkpoints','cache']:(ROOT/p).mkdir(exist_ok=True)
    d=load(a.data);tr,val=split(d)
    contract={'data_sha256':fingerprint(a.data),'validation_indices_sha256':hashlib.sha256(val.tobytes()).hexdigest(),'validation_examples':len(val),'train_examples':len(tr)}
    cp=ROOT/'contract.json'
    if cp.exists():assert json.loads(cp.read_text())==contract,'Immutable experiment data changed'
    else:cp.write_text(json.dumps(contract,indent=2)+'\n')
    ts=ROOT/'results.tsv'
    if not ts.exists():ts.write_text('experiment\tstatus\tval_nll\taccuracy\tparameters\tseconds\thypothesis\n')
    for c in configs:
        out=ROOT/'runs'/(c['name']+'.json')
        if out.exists():print('Already recorded:',c['name'],flush=True);continue
        try:
            art,report=run(d,tr,val,c)
            report['source_sha256']=fingerprint(__file__);report['contract']=contract
            report['source_commit']=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()
            prior=[json.loads(p.read_text()) for p in (ROOT/'runs').glob('*.json')]
            best=min([r['validation']['nll'] for r in prior if 'validation' in r]+[float('inf')])
            report['status']='keep' if report['validation']['nll']<best else 'discard'
            (ROOT/'checkpoints'/(c['name']+'.json')).write_text(json.dumps(art,separators=(',',':'))+'\n')
            out.write_text(json.dumps(report,indent=2)+'\n')
            with ts.open('a') as f:f.write(f"{c['name']}\t{report['status']}\t{report['validation']['nll']:.7f}\t{report['validation']['accuracy']:.7f}\t{report['parameter_count']}\t{report['seconds']:.1f}\t{c['hypothesis']}\n")
            print('RESULT',json.dumps({k:v for k,v in report.items() if k not in ('curve','contract')}),flush=True)
        except Exception as e:
            out.write_text(json.dumps({'config':c,'status':'crash','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n')
            with ts.open('a') as f:f.write(f"{c['name']}\tcrash\t\t\t\t\t{str(e)}\n")
            raise

if __name__=='__main__':main()
