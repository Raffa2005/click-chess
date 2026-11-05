"""Cross-entropy-first optional pretrained full-board evaluation."""
import argparse, hashlib, json, pathlib, time
import numpy as np
from .crossentropy import losses, fit_temperature, clustered_mean, empirical_floor
from .model import Ranker
from .fullboard import FullBoardRanker, OnnxFullBoardRanker, conditional_logits


def subset(data,modulus):
    if modulus==1:return data
    ids=np.array([i for i,g in enumerate(data['game']) if int(g,16)%modulus==0])
    off=data['offsets'];rows=np.concatenate([np.arange(off[i],off[i+1]) for i in ids])
    out={k:v[rows if k in ('x','indices') else ids] for k,v in data.items() if k!='offsets'}
    out['offsets']=np.r_[0,np.cumsum(np.diff(off)[ids])]
    return out


def score(data, model, batch_size=32):
    off=data['offsets'];n=np.diff(off);out=np.full((len(n),int(n.max())),-np.inf)
    if isinstance(model,Ranker):
        flat=model.scores(data['x'])
        for i in range(len(n)):out[i,:n[i]]=flat[off[i]:off[i+1]]
    else:
        start=time.perf_counter()
        for lo in range(0,len(n),batch_size):
            policies=model.policy_logits(data['packed'][lo:lo+batch_size])
            for j,pol in enumerate(policies):
                i=lo+j;out[i,:n[i]]=conditional_logits(pol,data['indices'][off[i]:off[i+1]])
            if lo%2048==0:print(json.dumps({'scored':lo+len(policies),'total':len(n),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--size',choices=['baseline','5m','23m'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--calibration');p.add_argument('--fit',action='store_true')
    p.add_argument('--threads',type=int,default=1);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--onnx',help='Use exported local ONNX model instead of PyTorch')
    p.add_argument('--game-modulus',type=int,default=1,help='Deterministic game-hash subset for costly references')
    a=p.parse_args();out=pathlib.Path(a.output)
    if out.exists():raise FileExistsError(out)
    if a.fit and a.calibration:raise ValueError('Fitting and testing must be separate')
    d=subset(dict(np.load(a.data)),a.game_modulus);n=np.diff(d['offsets'])
    if a.size=='baseline':model=Ranker('models/origin-ranker.bin');checkpoint='models/origin-ranker.bin'
    elif a.onnx:
        checkpoint=a.onnx;model=OnnxFullBoardRanker(checkpoint,threads=a.threads)
    else:
        import torch
        torch.set_num_threads(a.threads)
        checkpoint=f'research/fullboard/checkpoints/maia3-{a.size}.pt'
        model=FullBoardRanker(checkpoint,a.size)
    logits=score(d,model,a.batch_size)
    t=fit_temperature(logits,d['y']) if a.fit else (json.loads(pathlib.Path(a.calibration).read_text())['temperature'] if a.calibration else 1.)
    result={'size':a.size,'data':a.data,'game_modulus':a.game_modulus,'checkpoint_sha256':hashlib.file_digest(open(checkpoint,'rb'),'sha256').hexdigest(),
            'temperature':t,'temperature_fit_on':a.data if a.fit else a.calibration,'primary_metric':'cross_entropy_nats_per_ambiguous_click',
            'raw':clustered_mean(losses(logits,d['y']),d['game']),
            'calibrated':clustered_mean(losses(logits,d['y'],t),d['game']),
            'uniform_ce_nats':float(np.log(n).mean()),
            'empirical_floor_board_destination':empirical_floor(d['key'],d['y']),
            'empirical_floor_fullboard_inputs':empirical_floor(d['fullkey'],d['y'])}
    out.parent.mkdir(parents=True,exist_ok=True)
    cache=pathlib.Path('research/fullboard/cache')/(out.stem+'-logits.npz')
    np.savez_compressed(cache,logits=logits,loss=losses(logits,d['y'],t),raw_loss=losses(logits,d['y']),game=d['game'],key=d['key'])
    result['cached_predictions']=str(cache)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k.startswith(('size','temperature','raw','calibrated','uniform'))}),flush=True)


if __name__=='__main__':main()
