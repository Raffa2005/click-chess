"""One final test pass after validation-only model selection."""
import argparse, gzip, json, pathlib, platform, time
import numpy as np
from .train import load,batch,predict,metric
from .features import feature_names

def bootstrap(correct,groups):
    _,g=np.unique(groups,return_inverse=True);n=np.bincount(g);s=np.bincount(g,weights=correct)
    rng=np.random.default_rng(73491);values=[]
    for _ in range(1000):
        chosen=rng.integers(len(n),size=len(n));values.append(s[chosen].sum()/n[chosen].sum())
    return np.quantile(values,[.025,.975]).tolist()

def summarize(d,idx,p):
    y=d['y'][idx];order=np.argsort(-p,axis=1,kind='stable');ranks=(order==y[:,None]).argmax(1)+1
    correct=ranks==1
    return {'examples':len(idx),'games':len(set(d['game'][idx])),**metric(p,y),
        'accuracy_95pct_game_bootstrap':bootstrap(correct,d['game'][idx]),
        'top2':float((ranks<=2).mean()),'mean_reciprocal_rank':float((1/ranks).mean()),'mean_clicks_if_cycling':float(ranks.mean()),
        'uniform_legal_origin_accuracy':float((1/d['n'][idx]).mean())}

def params(art):
    out=[]
    for w,b in zip(art['weights'],art['biases']):out.extend([np.array(w,dtype=np.float32),np.array(b,dtype=np.float32)])
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--test',required=True);ap.add_argument('--classical',required=True);a=ap.parse_args()
    art=json.loads(pathlib.Path('models/origin-ranker.json').read_text());pars=params(art)
    if art['feature_version']!=1:raise ValueError('This is the original baseline evaluator. Use tinychess.research_evaluate for v2 models.')
    train=load(a.train);trainkeys=set(train['key'][~train['validation']]);developmentgames=set(train['game'])
    # Quantize weights only; bias stays float32. Test metrics include this deployable variant.
    quant=dict(art);quant['weights']=[];scales=[];ints=[]
    for w in pars[::2]:
        scale=float(np.max(np.abs(w))/127) or 1.;q=np.clip(np.rint(w/scale),-127,127).astype(np.int8)
        scales.append(scale);ints.append(q.tolist());quant['weights'].append((q.astype(np.float32)*scale).tolist())
    qp=params(quant)
    deployment={k:v for k,v in art.items() if k not in ('weights','feature_names')};deployment.update(quantization='symmetric-int8-per-tensor',weight_scales=scales,weights_int8=ints)
    deployment_text=json.dumps(deployment,separators=(',',':'))+'\n'
    pathlib.Path('models/origin-ranker.int8.json').write_text(deployment_text)
    pathlib.Path('models/origin-ranker.int8.json.gz').write_bytes(gzip.compress(deployment_text.encode(),mtime=0))
    report={'selected_representation':art['representation'],'parameters':art['parameter_count'],'float32_weight_bytes':art['parameter_count']*4,
        'int8_weight_and_float32_bias_bytes':sum(w.size for w in pars[::2])+sum(b.nbytes for b in pars[1::2]),
        'int8_json_bytes':len(deployment_text.encode()),'int8_json_gzip_bytes':len(gzip.compress(deployment_text.encode(),mtime=0)),
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},'sets':{}}
    valuecol=feature_names().index('value')
    for name,path in [('temporal_lichess',a.test),('classical_otb',a.classical)]:
        d=load(path);idx=np.array([i for i,g in enumerate(d['game']) if g not in developmentgames],dtype=int)
        if not len(idx):raise ValueError('No test games after deduplication')
        p=predict(d,idx,art['representation'],pars);pq=predict(d,idx,art['representation'],qp)
        result={'float32':summarize(d,idx,p),'int8':summarize(d,idx,pq),'duplicate_development_examples_removed':len(d['y'])-len(idx)}
        novel=np.array([k not in trainkeys for k in d['key'][idx]])
        result['novel_position_destination']=summarize(d,idx[novel],p[novel])
        first=d['y'][idx]==0;least=[];same=[]
        for i in idx:
            x=d['x'][d['offsets'][i]:d['offsets'][i+1]]
            least.append(x[:,valuecol].argmin()==d['y'][i]);same.append(len(set(x[:,:6].argmax(1)))==1)
        result['baselines']={'first_origin':float(first.mean()),'least_valuable_piece':float(np.mean(least))}
        groups={'same_piece_type':np.array(same),'different_piece_types':~np.array(same),'2_candidates':d['n'][idx]==2,'3plus_candidates':d['n'][idx]>=3,
            'opening_ply_lt20':d['ply'][idx]<20,'middlegame_ply20_79':(d['ply'][idx]>=20)&(d['ply'][idx]<80),'late_ply_ge80':d['ply'][idx]>=80}
        result['slices']={n:{'examples':int(mask.sum()),**metric(p[mask],d['y'][idx][mask])} for n,mask in groups.items() if mask.any()}
        # Counterpart disagreement and top-two are important for repeat-click UI.
        result['quantization_argmax_disagreement']=float((p.argmax(1)!=pq.argmax(1)).mean())
        report['sets'][name]=result
    pathlib.Path('results/evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
