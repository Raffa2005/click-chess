"""Resume an explicit, bounded round-two training plan; never overwrite results."""
import argparse,json,os,subprocess,sys,time
from .round2 import ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiments',nargs='*');ap.add_argument('--max-runs',type=int,default=10);ap.add_argument('--max-minutes',type=float,default=90);ap.add_argument('--blends',action='store_true');args=ap.parse_args()
    configs=json.loads((ROOT/'experiments.json').read_text());known={c['name'] for c in configs}
    if args.experiments and set(args.experiments)-known:raise ValueError('Unknown experiment name')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');start=time.monotonic();count=0
    for c in configs:
        if c.get('kind')=='weight_blend' or (args.experiments and c['name'] not in args.experiments):continue
        if (ROOT/'runs'/(c['name']+'.json')).exists():continue
        if count>=args.max_runs or time.monotonic()-start>=args.max_minutes*60:break
        if c.get('december'):
            subprocess.run([sys.executable,'-m','tinychess.round2_bigdata','--prepare-only'],env=env,check=True)
            module='tinychess.round2_bigdata'
        elif c.get('architecture')=='set-context':module='tinychess.round2_sets'
        else:module='tinychess.round2'
        print('RUN',c['name'],flush=True)
        with (ROOT/(c['name']+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-m',module,'--experiment',c['name']],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=c['seconds']+180)
        count+=1
    if args.blends:subprocess.run([sys.executable,'-m','tinychess.round2_blend'],env=env,check=True)
    print(json.dumps(dict(completed_runs=count,elapsed_seconds=time.monotonic()-start)))

if __name__=='__main__':main()
