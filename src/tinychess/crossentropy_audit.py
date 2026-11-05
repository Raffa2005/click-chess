"""Recompute shipped-model loss and finite-sample lookup floors."""
import argparse,hashlib,json,pathlib
import numpy as np
from .crossentropy import empirical_floor,losses,clustered_mean
from .model import Ranker


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--model',default='models/origin-ranker.bin');p.add_argument('--output',required=True);a=p.parse_args()
    d=dict(np.load(a.data));off=d['offsets'];n=np.diff(off)
    flat=Ranker(a.model).scores(d['x']);scores=np.full((len(n),int(n.max())),-np.inf)
    for i in range(len(n)):scores[i,:n[i]]=flat[off[i]:off[i+1]]
    keys=[hashlib.sha256(d['x'][lo:hi].tobytes()).hexdigest() for lo,hi in zip(off[:-1],off[1:])]
    result={'data':a.data,'model':a.model,'cross_entropy_nats':clustered_mean(losses(scores,d['y']),d['game']),
            'uniform_ce_nats':float(np.log(n).mean()),'board_destination_empirical':empirical_floor(d['key'],d['y']),
            'exact_model_input_empirical':empirical_floor(keys,d['y'])}
    pathlib.Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
