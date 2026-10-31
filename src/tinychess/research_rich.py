"""Run the same frozen-validation protocol on rich features and optional extra data."""
import argparse,json,pathlib,time,hashlib,subprocess,traceback
import numpy as np
from .research import run,split,fingerprint,ROOT

def lightweight(path):
    with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files if k not in ('board','squares')}
    d['n']=np.diff(d['offsets']);return d

def prepare(extra=False):
    d=lightweight('data/processed/feb.npz');tr,val=split(d)
    contract=json.loads((ROOT/'contract.json').read_text())
    metadata=json.loads(pathlib.Path('research/cache/feb-rich.json').read_text())
    assert fingerprint('data/processed/feb.npz')==contract['data_sha256']
    assert metadata['source_data_sha256']==contract['data_sha256']
    assert hashlib.sha256(val.tobytes()).hexdigest()==contract['validation_indices_sha256']
    rich=np.load('research/cache/feb-rich.npy',mmap_mode='r')
    assert len(rich)==len(d['x']) and np.array_equal(rich[::1000,:107],d['x'][::1000])
    d['x']=rich
    if extra:
        e=lightweight('research/cache/january-extra.npz')
        # Extra extraction already excludes all February validation games/keys.
        assert not set(e['game'])&set(d['game'][d['validation']])
        assert not set(e['key'])&set(d['key'][d['validation']])
        # Avoid weighting exact game duplicates across training sources twice.
        known=set(d['game']);ids=np.array([i for i,g in enumerate(e['game']) if g not in known])
        rows=np.concatenate([np.arange(e['offsets'][i],e['offsets'][i+1]) for i in ids])
        oldn=len(d['y']);oldrows=len(d['x'])
        d['x']=np.concatenate([d['x'],e['x'][rows]])
        d['offsets']=np.r_[d['offsets'],oldrows+np.cumsum(e['n'][ids])]
        for k in ['y','game','key','ply','kind','n']:d[k]=np.concatenate([d[k],e[k][ids]])
        tr=np.r_[tr,np.arange(oldn,len(d['y']))]
    assert len(val)==36024
    return d,tr,val

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',required=True);a=ap.parse_args()
    configs=json.loads((ROOT/'experiments.json').read_text());c=next(c for c in configs if c['name']==a.experiment)
    out=ROOT/'runs'/(c['name']+'.json')
    if out.exists():raise ValueError('Experiment already recorded; choose a new name')
    source_sha=fingerprint(__file__);d,tr,val=prepare(c.get('extra_training',False))
    if c.get('ablate_context'):d['x']=d['x'].copy();d['x'][:,107:117]=0
    if c.get('ablate_risk'):d['x']=d['x'].copy();d['x'][:,117:]=0
    try:
        art,report=run(d,tr,val,c)
        # Ablated columns must also be disabled in exported inference weights.
        for lo,hi,enabled in [(107,117,c.get('ablate_context')),(117,147,c.get('ablate_risk'))]:
            if enabled:
                for j in range(lo,hi):art['weights'][0][j]=[0.]*len(art['weights'][0][j])
        report['source_sha256']=source_sha;report['tactical_source_sha256']=fingerprint('src/tinychess/tactical.py')
        report['source_commit']=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip();report['feature_version']=2
        report['contract']=json.loads((ROOT/'contract.json').read_text());report['rich_features']=json.loads(pathlib.Path('research/cache/feb-rich.json').read_text())
        if c.get('extra_training'):report['extra_data']=json.loads(pathlib.Path('research/cache/january-extra.json').read_text())
        prior=[json.loads(p.read_text()) for p in (ROOT/'runs').glob('*.json')]
        best=min([r['validation']['nll'] for r in prior if 'validation' in r]+[float('inf')]);report['status']='keep' if report['validation']['nll']<best else 'discard'
        (ROOT/'checkpoints'/(c['name']+'.json')).write_text(json.dumps(art,separators=(',',':'))+'\n');out.write_text(json.dumps(report,indent=2)+'\n')
        with (ROOT/'results.tsv').open('a') as f:f.write(f"{c['name']}\t{report['status']}\t{report['validation']['nll']:.7f}\t{report['validation']['accuracy']:.7f}\t{report['parameter_count']}\t{report['seconds']:.1f}\t{c['hypothesis']}\n")
        print('RESULT',json.dumps({k:v for k,v in report.items() if k not in ('curve','contract','rich_features','extra_data')}),flush=True)
    except Exception as e:
        out.write_text(json.dumps({'config':c,'status':'crash','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n')
        with (ROOT/'results.tsv').open('a') as f:f.write(f"{c['name']}\tcrash\t\t\t\t\t{str(e)}\n")
        raise

if __name__=='__main__':main()
