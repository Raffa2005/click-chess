"""Masked categorical cross entropy over legal origins, trained on CPU."""
import argparse, json, pathlib, time, shutil
import numpy as np
from .features import BASE_DIM, VERSION, feature_names

def load(path):
    with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    d['n']=np.diff(d['offsets']);return d

def batch(d,idx,rep):
    n=d['n'][idx];k=int(n.max());valid=np.arange(k)[None,:]<n[:,None]
    rows=np.minimum(d['offsets'][idx,None]+np.arange(k)[None,:],len(d['x'])-1)
    if rep=='planes':
        x=np.zeros((len(idx),k,903),dtype=np.float32);x[:,:,:775]=d['board'][idx,None,:]
        sq=d['squares'][rows];a=np.arange(len(idx))[:,None];b=np.arange(k)[None,:]
        x[a,b,775+sq[:,:,0].astype(int)]=1;x[a,b,839+sq[:,:,1].astype(int)]=1
    else:x=d['x'][rows,:BASE_DIM if rep=='no_history' else d['x'].shape[1]]
    return x,valid

def forward(x,params):
    acts=[x]
    for i in range(0,len(params),2):
        x=x@params[i]+params[i+1]
        if i<len(params)-2:x=np.maximum(x,0)
        acts.append(x)
    return x,acts

def probabilities(s,valid):
    s=np.where(valid,s,-1e9);s-=s.max(axis=1,keepdims=True)
    e=np.exp(s)*valid;return e/e.sum(axis=1,keepdims=True)

def predict(d,idx,rep,params):
    k=int(d['n'][idx].max());out=np.zeros((len(idx),k),dtype=np.float32)
    for start in range(0,len(idx),2048):
        ids=idx[start:start+2048];x,valid=batch(d,ids,rep)
        s,_=forward(x.reshape(-1,x.shape[-1]),params)
        out[start:start+len(ids),:valid.shape[1]]=probabilities(s.reshape(valid.shape),valid)
    return out

def metric(p,y):return {'accuracy':float((p.argmax(1)==y).mean()),'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())}

def train_one(d,train_idx,val_idx,rep,hidden,seed=20261031,epochs=18):
    rng=np.random.default_rng(seed);dim=903 if rep=='planes' else (BASE_DIM if rep=='no_history' else d['x'].shape[1])
    dims=[dim]+([hidden] if hidden else [])+[1]
    params=[]
    for a,b in zip(dims,dims[1:]):params += [(rng.normal(size=(a,b))*np.sqrt(2/a)).astype(np.float32),np.zeros(b,dtype=np.float32)]
    m=[np.zeros_like(p) for p in params];v=[np.zeros_like(p) for p in params]
    best=None;best_loss=float('inf');step=0;stale=0;log=[];start=time.time()
    for epoch in range(epochs):
        order=rng.permutation(train_idx)
        for off in range(0,len(order),512):
            ids=order[off:off+512];x,valid=batch(d,ids,rep);flat=x.reshape(-1,dim)
            scores,acts=forward(flat,params);p=probabilities(scores.reshape(valid.shape),valid)
            p[np.arange(len(ids)),d['y'][ids]]-=1
            grad=(p/len(ids)).reshape(-1,1);grads=[None]*len(params)
            for layer in reversed(range(len(dims)-1)):
                j=layer*2;grads[j]=acts[layer].T@grad+1e-5*params[j];grads[j+1]=grad.sum(0)
                if layer:grad=(grad@params[j].T)*(acts[layer]>0)
            step+=1
            for j,g in enumerate(grads):
                m[j]=.9*m[j]+.1*g;v[j]=.999*v[j]+.001*g*g
                params[j]-=.002*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
        met=metric(predict(d,val_idx,rep,params),d['y'][val_idx]);met.update(epoch=epoch+1,seconds=time.time()-start);log.append(met)
        print(json.dumps({'representation':rep,'hidden':hidden,**met}),flush=True)
        if met['nll']<best_loss-1e-5:best_loss=met['nll'];best=[p.copy() for p in params];best_epoch=epoch+1;stale=0
        else:stale+=1
        if stale>=4:break
    artifact={'format':'tiny-chess-input-v1','feature_version':VERSION,'representation':rep,'hidden':hidden,'seed':seed,
        'parameter_count':sum(p.size for p in best),'weights':[p.tolist() for p in best[::2]],'biases':[p.tolist() for p in best[1::2]],
        'feature_names':feature_names(rep=='history') if rep!='planes' else None}
    return artifact,{'representation':rep,'hidden':hidden,'best_epoch':best_epoch,'parameter_count':artifact['parameter_count'],
        'validation':metric(predict(d,val_idx,rep,best),d['y'][val_idx]),'training_seconds':time.time()-start,'epochs':log}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--epochs',type=int,default=18);p.add_argument('--extended',action='store_true');a=p.parse_args()
    d=load(a.data);tr=np.where(~d['validation'])[0];val=np.where(d['validation'])[0]
    # Model selection uses novel position+destination examples only.
    trainkeys=set(d['key'][tr]);val=val[np.array([k not in trainkeys for k in d['key'][val]])]
    assert not set(d['game'][tr])&set(d['game'][val])
    report={'train_examples':len(tr),'validation_novel_examples':len(val),'train_games':len(set(d['game'][tr])),
        'validation_games':len(set(d['game'][val])),'selection':'lowest novel-validation NLL; no test data loaded','experiments':[]}
    pathlib.Path('models/checkpoints').mkdir(parents=True,exist_ok=True)
    best_score=float('inf');winner=None
    if a.extended:
        report=json.loads(pathlib.Path('results/training.json').read_text())
        winner=json.loads(pathlib.Path('models/origin-ranker.json').read_text())
        best_score=min(r['validation']['nll'] for r in report['experiments'])
    configs=[('no_history',64),('history',64)] if a.extended else [('history',0),('no_history',32),('history',16),('history',32),('planes',32)]
    for rep,hidden in configs:
        art,result=train_one(d,tr,val,rep,hidden,epochs=a.epochs)
        name=f'{rep}-{hidden}';path=pathlib.Path('models/checkpoints')/(name+'.json');path.write_text(json.dumps(art,separators=(',',':'))+'\n')
        result['name']=name;report['experiments'].append(result)
        if result['validation']['nll']<best_score:best_score=result['validation']['nll'];winner=art;report['selected']=name
        pathlib.Path('results/training.json').write_text(json.dumps(report,indent=2)+'\n')
    pathlib.Path('models/origin-ranker.json').write_text(json.dumps(winner,separators=(',',':'))+'\n')
    fallback=pathlib.Path('models/checkpoints/no_history-64.json')
    if a.extended and fallback.exists():shutil.copyfile(fallback,'models/origin-ranker.no-history.json')
    print('SELECTED',report['selected'],flush=True)

if __name__=='__main__':main()
