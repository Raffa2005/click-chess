import unittest
import numpy as np
from tinychess.setnet import forward, backward
from tinychess.train import probabilities
from tinychess.comparative import augment, transform_dataset


class SetContextTests(unittest.TestCase):
    def test_masked_gradient_and_permutation(self):
        rng = np.random.default_rng(219)
        x = rng.normal(size=(2, 3, 3))
        valid = np.array([[1, 1, 0], [1, 1, 1]], bool)
        y = np.array([1, 2])
        for pooling, factor in [('mean', 2), ('mean_max', 3)]:
            params = []
            for left, right in [(3, 4), (4*factor, 3), (3, 1)]:
                params += [rng.normal(size=(left, right)), rng.normal(size=right)]
            scores, cache = forward(x, valid, params, pooling)
            p = probabilities(scores, valid)
            p[np.arange(2), y] -= 1
            gradients = backward(cache, params, p/2)
            def loss():
                s, _ = forward(x, valid, params, pooling)
                return -np.log(probabilities(s, valid)[np.arange(2), y]).mean()
            for parameter, gradient in zip(params, gradients):
                for index in np.ndindex(parameter.shape):
                    value = parameter[index]
                    parameter[index] = value+1e-5
                    plus = loss()
                    parameter[index] = value-1e-5
                    minus = loss()
                    parameter[index] = value
                    self.assertAlmostEqual(float(gradient[index]), float((plus-minus)/2e-5), places=6)
            perm = [2, 0, 1]
            permuted, _ = forward(x[:, perm], valid[:, perm], params, pooling)
            np.testing.assert_allclose(permuted, scores[:, perm], atol=1e-12)
            changed = x.copy()
            changed[0, 2] = 10000
            masked, _ = forward(changed, valid, params, pooling)
            np.testing.assert_allclose(masked[valid], scores[valid], atol=1e-12)

    def test_dataset_pooling_matches_single_decisions(self):
        rng = np.random.default_rng(33)
        x = rng.normal(size=(11, 147)).astype(np.float32)
        offsets = np.array([0, 2, 6, 8, 11])
        expected = np.concatenate([augment(x[a:b]) for a, b in zip(offsets[:-1], offsets[1:])])
        np.testing.assert_allclose(transform_dataset(x, offsets), expected, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
