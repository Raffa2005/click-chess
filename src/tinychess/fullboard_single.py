"""Verify deployment cross-entropy with batch size ONE, independent requests.

Dynamic activation quantization can depend on the other examples in a batch.
This is a correctness verification of frozen weights/temperatures, not a new
model selection round. Parallel workers run independent single-position calls.
"""
import argparse,hashlib,json,pathlib,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .crossentropy import losses,clustered_mean,empirical_floor
from .fullboard import OnnxFullBoardRanker,conditional_logits
from .fullboard_study import subset


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--size',choices=['5m','23m'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--game-modulus',type=int,default=1);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();out=pathlib.Path(a.output)
    if out.exists():raise FileExistsError(out)
    root=pathlib.Path('research/fullboard');frozen=json.loads((root/'frozen.json').read_text());cfg=frozen['choices'][a.size+'-int8']
    checkpoint=root/f'checkpoints/maia3-{a.size}.int8.onnx'
    assert hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()==cfg['checkpoint_sha256']
    m=OnnxFullBoardRanker(checkpoint);d=subset(dict(np.load(a.data)),a.game_modulus);off=d['offsets'];n=np.diff(off)
    scores=np.full((len(n),int(n.max())),-np.inf);start=time.perf_counter()
    def one(i):
        policy=m.policy_logits(d['packed'][i:i+1])[0]
        return conditional_logits(policy,d['indices'][off[i]:off[i+1]])
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for lo in range(0,len(n),512):
            for i,s in zip(range(lo,min(lo+512,len(n))),pool.map(one,range(lo,min(lo+512,len(n))))):scores[i,:n[i]]=s
            if lo%2048==0:print(json.dumps({'scored':min(lo+512,len(n)),'total':len(n),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    # Thread concurrency must not change inference compared with serial calls.
    for i in range(min(8,len(n))):np.testing.assert_allclose(scores[i,:n[i]],one(i),atol=1e-6,rtol=1e-6)
    loss=losses(scores,d['y'],cfg['temperature'])
    result={'size':a.size,'data':a.data,'game_modulus':a.game_modulus,'batch_size':1,'independent_workers':a.workers,
            'checkpoint_sha256':cfg['checkpoint_sha256'],'temperature':cfg['temperature'],
            'primary_metric':'cross_entropy_nats_per_ambiguous_click','raw':clustered_mean(losses(scores,d['y']),d['game']),
            'calibrated':clustered_mean(loss,d['game']),'uniform_ce_nats':float(np.log(n).mean()),
            'empirical_floor_fullboard_inputs':empirical_floor(d['fullkey'],d['y']),
            'seconds':time.perf_counter()-start,'selection':'Frozen weights/temperature; deployment-path verification only.'}
    cache=root/('cache/'+out.stem+'-logits.npz');np.savez_compressed(cache,logits=scores,loss=loss,game=d['game'],key=d['key'])
    result['cached_predictions']=str(cache);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
