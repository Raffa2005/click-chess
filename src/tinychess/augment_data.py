"""Cache rich features without replaying PGNs. Labels are never consulted."""
import argparse,hashlib,json,multiprocessing as mp,pathlib,time
import numpy as np
from .train import load
from .tactical import augment,reconstruct,NAMES

DATA=None
def work(bounds):
    lo,hi=bounds;d=DATA;parts=[]
    for i in range(lo,hi):
        start,end=d['offsets'][i:i+2];b,moves=reconstruct(d['board'][i],d['squares'][start:end])
        parts.append(augment(b,moves,d['x'][start:end]))
    return lo,np.concatenate(parts)

def main():
    global DATA
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    DATA=load(a.data);d=DATA;out=pathlib.Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    result=np.lib.format.open_memmap(out,mode='w+',dtype=np.float32,shape=(len(d['x']),len(NAMES)))
    start=time.perf_counter();bounds=[(i,min(i+500,len(d['y']))) for i in range(0,len(d['y']),500)]
    with mp.get_context('fork').Pool(a.workers) as pool:
        for k,(i,rows) in enumerate(pool.imap_unordered(work,bounds,chunksize=1)):
            offset=d['offsets'][i];result[offset:offset+len(rows)]=rows
            if k%40==0:print(json.dumps({'batches':k+1,'total':len(bounds),'seconds':time.perf_counter()-start}),flush=True)
    result.flush()
    with open(a.data,'rb') as f:source=hashlib.file_digest(f,'sha256').hexdigest()
    with out.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    meta={'source_data_sha256':source,'output_sha256':sha,'shape':list(result.shape),'feature_version':2,'feature_names':NAMES,'seconds':time.perf_counter()-start}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
