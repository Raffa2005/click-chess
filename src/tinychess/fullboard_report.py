"""Paired, batch-one test comparisons; no model fitting or selection."""
import json,pathlib
import numpy as np
from .crossentropy import clustered_mean


def main():
    root=pathlib.Path('research/fullboard');report={}
    development=np.load(root/'cache/development.npz');devkeys=set(development['key'])
    for corpus in ['online','classical']:
        d=np.load(root/f'cache/{corpus}-test.npz')
        mask=np.array([int(g,16)%4==0 for g in d['game']]) if corpus=='online' else np.ones(len(d['game']),bool)
        games=d['game'][mask];keys=d['key'][mask]
        baseline=np.load(root/f'cache/{corpus}-test-baseline-logits.npz')['loss'][mask]
        values={'old_model_calibrated':baseline}
        for size in ['5m','23m']:
            pred=np.load(root/f'cache/{corpus}-single-{size}-logits.npz')
            np.testing.assert_array_equal(pred['game'],games);np.testing.assert_array_equal(pred['key'],keys)
            values[size]=pred['loss']
        novel=np.array([key not in devkeys for key in keys])
        r={'scope':'One position per forward pass, frozen native ONNX INT8 models and temperatures.',
           'online_game_hash_modulus':4 if corpus=='online' else 1,
           'models':{name:clustered_mean(v,games) for name,v in values.items()},
           'paired_delta':{'5m_minus_old':clustered_mean(values['5m']-baseline,games),
                           '23m_minus_5m':clustered_mean(values['23m']-values['5m'],games)},
           'absent_from_development':{'note':'Only development-key exclusion; pretrained training-position overlap is unknown.',
               'models':{name:clustered_mean(v[novel],games[novel]) for name,v in values.items()}},
           'human_entropy_floor':None,'gap_to_human_entropy':None}
        report[corpus]=r
    (root/'final-crossentropy.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{'models':v['models'],'delta':v['paired_delta']} for k,v in report.items()}),flush=True)


if __name__=='__main__':main()
