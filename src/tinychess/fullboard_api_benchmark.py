"""Public rank API latency, including board encoding and legal generation."""
import argparse,json,pathlib,time
import chess
import numpy as np
from .fullboard import OnnxFullBoardRanker
from .model import Ranker,ClickSession
from .features import origins
from .fullboard_benchmark import summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='research/fullboard/api-benchmark.json');args=parser.parse_args()
    # Existing fixtures are used for API timing only, never loss/model selection.
    fixtures=json.loads(pathlib.Path('tests/fixtures.json').read_text())
    cases=[(chess.Board(f['fen']),chess.parse_square(f['destination'])) for f in fixtures]
    cases=[(b,d) for b,d in cases if len(origins(b,d))>1]
    model=OnnxFullBoardRanker('research/fullboard/checkpoints/maia3-5m.int8.onnx')
    base=Ranker('models/origin-ranker.bin')
    times={name:[] for name in ['uncached_full_rank','cached_full_rank','baseline_full_rank','cached_repeat_click']}
    cpu={name:[] for name in times}
    for b,d in cases[:10]:model.rank(b,d)
    for b,d in cases:
        model.clear_cache()
        for name,fn in [('uncached_full_rank',lambda:model.rank(b,d)),
                        ('cached_full_rank',lambda:model.rank(b,d)),
                        ('baseline_full_rank',lambda:base.rank(b,d))]:
            start=time.perf_counter_ns();cstart=time.thread_time_ns();fn();cpu[name].append((time.thread_time_ns()-cstart)/1e6);times[name].append((time.perf_counter_ns()-start)/1e6)
        session=ClickSession(model,b,d)
        start=time.perf_counter_ns();cstart=time.thread_time_ns()
        for _ in range(100):session.click()
        times['cached_repeat_click'].append((time.perf_counter_ns()-start)/1e8)
        cpu['cached_repeat_click'].append((time.thread_time_ns()-cstart)/1e8)
    result={'cases':len(cases),'source':'tests/fixtures.json','history':'FEN-only fixtures; earliest board repeated as documented',
            'threads':1,'warm_model':True,'scope':'Python host CPU API, includes legal generation and history encoding; excludes UI, model load, physical Android device.',
            'timings':{k:summary(v) for k,v in times.items()},'thread_cpu_times':{k:summary(v) for k,v in cpu.items()}}
    pathlib.Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
