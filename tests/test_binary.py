import json
import unittest
from pathlib import Path
import numpy as np
from tinychess.binary import encode, decode
from tinychess.model import Ranker


class BinaryTests(unittest.TestCase):
    def test_exact_prediction_round_trip(self):
        artifact = json.loads(Path('models/origin-ranker.int8.json').read_text())
        data = encode(artifact)
        decoded = decode(data)
        rng = np.random.default_rng(992)
        x = rng.normal(size=(7, len(artifact['weights_int8'][0]))).astype(np.float32)
        np.testing.assert_array_equal(Ranker(artifact).scores(x), Ranker(decoded).scores(x))
        self.assertLess(len(data), 24576)

    def test_truncation_and_trailing_data_rejected(self):
        artifact = json.loads(Path('models/origin-ranker.int8.json').read_text())
        data = encode(artifact)
        for broken in [b'', data[:7], data[:20], data[:-1], data+b'!']:
            with self.assertRaises(ValueError):
                decode(broken)


if __name__ == '__main__':
    unittest.main()
