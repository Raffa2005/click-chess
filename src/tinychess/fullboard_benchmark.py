"""Single-position latency on diverse recorded inputs, not batch throughput."""
import argparse,json,pathlib,time,platform
import numpy as np
from .fullboard import OnnxFullBoardRanker,token_array,conditional_logits


def summary(ts):
    return dict(zip(['p50_ms','p95_ms','max_ms'],map(float,np.quantile(ts,[.5,.95,1.]))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=['5m','23m'],default='5m');p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=1);a=p.parse_args()
    import onnxruntime
    m=OnnxFullBoardRanker(f'research/fullboard/checkpoints/maia3-{a.size}.int8.onnx',threads=a.threads)
    d=np.load('research/fullboard/cache/development.npz');packed=d['packed'];off=d['offsets'];ix=d['indices']
    ids=np.random.default_rng(20261105).choice(len(packed),256,replace=False)
    # Warm up independently of recorded timings.
    for i in ids[:20]:m.policy_logits(packed[i:i+1])
    encode=[];forward=[];combined=[];condition=[];raw_policy=None
    for i in ids:
        start=time.perf_counter_ns();x=token_array(packed[i:i+1]);end=time.perf_counter_ns()
        encode.append((end-start)/1e6)
        rating=np.array([2400.],np.float32)
        start=time.perf_counter_ns();raw_policy=m.session.run(None,{'tokens':x,'self_elo':rating,'opponent_elo':rating})[0][0];end=time.perf_counter_ns()
        forward.append((end-start)/1e6)
        start=time.perf_counter_ns();scores=conditional_logits(raw_policy,ix[off[i]:off[i+1]]);np.argsort(-scores);end=time.perf_counter_ns()
        condition.append((end-start)/1e6);combined.append(encode[-1]+forward[-1]+condition[-1])
    result={'model':a.size,'runtime':onnxruntime.__version__,'python':platform.python_version(),'platform':platform.platform(),
            'threads':a.threads,'warmup':20,'measured_positions':256,'cpu_identity':'Unavailable: sandbox blocks /proc/cpuinfo',
            'input_onehot':summary(encode),'network_forward':summary(forward),'cached_policy_conditioning':summary(condition),
            'input_plus_forward_plus_conditioning':summary(combined),
            'scope':'Host CPU timing; excludes chess legal generation, reconstructing history from PGN, model load, UI rendering and Android device performance.'}
    pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
