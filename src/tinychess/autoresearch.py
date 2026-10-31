"""Resume an explicit bounded experiment plan, preserving every trial result."""
import argparse,json,os,pathlib,subprocess,sys,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiments',nargs='*');ap.add_argument('--max-runs',type=int,default=20);ap.add_argument('--max-minutes',type=float,default=60);a=ap.parse_args()
    configs=json.loads(pathlib.Path('research/experiments.json').read_text())
    if a.experiments:
        names={c['name'] for c in configs};unknown=set(a.experiments)-names
        if unknown:raise ValueError(f'Unknown experiments: {unknown}')
        configs=[c for c in configs if c['name'] in a.experiments]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');start=time.monotonic();count=0
    for c in configs:
        if pathlib.Path('research/runs',c['name']+'.json').exists():continue
        if count>=a.max_runs or time.monotonic()-start>=a.max_minutes*60:break
        module='tinychess.research_rich' if c.get('feature_version',1)==2 else 'tinychess.research'
        print('RUN',c['name'],flush=True)
        with pathlib.Path('research',c['name']+'.log').open('w') as f:
            # Hard outer timeout includes startup; the trainer enforces its own budget.
            try:subprocess.run([sys.executable,'-m',module,'--experiment',c['name']],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=c['seconds']+180)
            except (subprocess.SubprocessError,OSError) as error:
                print('FAILED',c['name'],str(error),flush=True)
                result=pathlib.Path('research/runs',c['name']+'.json')
                if not result.exists():
                    result.parent.mkdir(parents=True,exist_ok=True)
                    result.write_text(json.dumps({'config':c,'status':'crash','error':str(error)},indent=2)+'\n')
                    with pathlib.Path('research/results.tsv').open('a') as ledger:
                        ledger.write(c['name']+'\tcrash\t\t\t\t\t'+str(error).replace('\n',' ').replace('\t',' ')+'\n')
        count+=1
    print(json.dumps({'completed_attempts':count,'elapsed_seconds':time.monotonic()-start,'note':'No background or recurring task was scheduled.'}),flush=True)

if __name__=='__main__':main()
