"""Choose compression on frozen validation, then record final model hashes."""
import argparse,gzip,hashlib,json,pathlib
import numpy as np
from .research_rich import prepare
from .train import predict,metric
from .evaluate import params

def quantize(art,per_channel):
    packed={k:v for k,v in art.items() if k not in ('weights','feature_names')}
    packed.update(weights_int8=[],weight_scales=[],quantization='symmetric-int8-per-output' if per_channel else 'symmetric-int8-per-tensor')
    dequant=dict(art);dequant['weights']=[]
    for w in params(art)[::2]:
        scale=np.maximum(np.max(np.abs(w),axis=0) if per_channel else np.max(np.abs(w)),1e-9)/127
        q=np.clip(np.rint(w/scale),-127,127).astype(np.int8)
        packed['weights_int8'].append(q.tolist());packed['weight_scales'].append(scale.tolist())
        dequant['weights'].append((q.astype(np.float32)*scale).tolist())
    return packed,dequant

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',required=True);a=ap.parse_args()
    if pathlib.Path('research/frozen.json').exists():raise ValueError('A model is already frozen; use a fresh research workspace for a new study')
    checkpoint=pathlib.Path('research/checkpoints')/(a.experiment+'.json');art=json.loads(checkpoint.read_text())
    if art['feature_version']!=2:raise ValueError('This export command expects rich feature models')
    d,tr,val=prepare(False);floatmet=metric(predict(d,val,'history',params(art)),d['y'][val]);options=[]
    for per_channel in (False,True):
        packed,dq=quantize(art,per_channel);txt=json.dumps(packed,separators=(',',':'))+'\n';gz=gzip.compress(txt.encode(),mtime=0)
        met=metric(predict(d,val,'history',params(dq)),d['y'][val]);entry={'quantization':packed['quantization'],'gzip_bytes':len(gz),'json_bytes':len(txt.encode()),'validation':met}
        entry['eligible']=len(gz)<=24576 and art['parameter_count']<=20000 and met['accuracy']>=floatmet['accuracy']-.002
        options.append((entry,packed,txt,gz))
    eligible=[v for v in options if v[0]['eligible']]
    if not eligible:raise ValueError('No compact export passes the fixed validation accuracy tolerance')
    entry,packed,txt,gz=min(eligible,key=lambda v:v[0]['validation']['nll'])
    pathlib.Path('models/origin-ranker.json').write_text(json.dumps(art,separators=(',',':'))+'\n')
    pathlib.Path('models/origin-ranker.int8.json').write_text(txt);pathlib.Path('models/origin-ranker.int8.json.gz').write_bytes(gz)
    info={'selected_experiment':a.experiment,'float32_validation':floatmet,'quantization_options':[x[0] for x in options],
        'selected_quantization':entry,'float32_sha256':hashlib.sha256(pathlib.Path('models/origin-ranker.json').read_bytes()).hexdigest(),
        'int8_sha256':hashlib.sha256(txt.encode()).hexdigest(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'note':'Selected and frozen using validation only, before fresh test evaluation.'}
    pathlib.Path('research/frozen.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))

if __name__=='__main__':main()
