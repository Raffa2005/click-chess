"""Generate cross-language fixtures for experimental architectures and packing."""
import base64
import json
from pathlib import Path
import numpy as np
from .binary import encode
from .comparative import augment
from .freeze import quantize
from .model import Ranker


def main():
    fixtures = json.loads(Path('tests/fixtures.json').read_text())
    indices = sorted(set(np.linspace(0, len(fixtures)-1, 20, dtype=int).tolist()+list(range(100, 105))))
    rng = np.random.default_rng(44071)
    models = []
    for version, pooling in [(2, 'mean'), (2, 'mean_max'), (3, None)]:
        shapes = [(147, 4), ((3 if pooling == 'mean_max' else 2)*4, 3), (3, 1)] if pooling else [(239, 5), (5, 1)]
        artifact = dict(feature_version=version, representation='history', weights=[], biases=[])
        if pooling:
            artifact.update(architecture='set-context', pooling=pooling)
        for left, right in shapes:
            artifact['weights'].append(rng.normal(0, .2, (left, right)).astype(np.float32).tolist())
            artifact['biases'].append(rng.normal(0, .1, right).astype(np.float32).tolist())
        packed, _ = quantize(artifact, pooling != 'mean')
        model = Ranker(packed)
        cases = []
        for index in indices:
            x = np.array(fixtures[index]['rich_features'], np.float32)
            if version == 3:
                x = augment(x)
            cases.append(dict(index=index, scores=model.scores(x).tolist(), **({'features':x.tolist()} if version == 3 else {})))
        models.append(dict(binary_base64=base64.b64encode(encode(packed)).decode(), cases=cases))
    Path('tests/round2-fixtures.json').write_text(json.dumps(models, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    main()
