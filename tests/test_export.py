import unittest
import numpy as np
from tinychess.research import fold_normalization
from tinychess.train import forward
from tinychess.freeze import quantize
from tinychess.model import Ranker


class ExportTests(unittest.TestCase):
    def test_normalization_folding_preserves_deep_network(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(31, 7)).astype(np.float32)
        mean = rng.normal(size=7).astype(np.float32)
        std = rng.uniform(.05, 2, size=7).astype(np.float32)
        parameters = []
        for a, b in [(7, 9), (9, 4), (4, 1)]:
            parameters.extend([rng.normal(size=(a, b)).astype(np.float32),
                               rng.normal(size=b).astype(np.float32)])
        expected, _ = forward((x - mean) / std, parameters)
        actual, _ = forward(x, fold_normalization(parameters, mean, std))
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=1e-5)

    def test_both_quantization_formats_load_correctly(self):
        rng = np.random.default_rng(43)
        first = rng.normal(size=(7, 4))
        first[:, 3] = 0  # Include a zero-range output channel.
        art = {'feature_version': 2, 'representation': 'history',
               'weights': [first.tolist(), rng.normal(size=(4, 1)).tolist()],
               'biases': [[.1, -.2, .3, 0.], [.4]]}
        x = rng.normal(size=(9, 7)).astype(np.float32)
        for per_output in [False, True]:
            packed, reference = quantize(art, per_output)
            np.testing.assert_allclose(Ranker(packed).scores(x), Ranker(reference).scores(x), atol=1e-6)
            self.assertTrue(np.isfinite(Ranker(packed).scores(x)).all())


if __name__ == '__main__':
    unittest.main()
