"""Version 3: candidate features plus pooled competing-origin context.

Only legal inputs contribute. There are no observed-move labels in this transform.
"""
import numpy as np
from .tactical import NAMES as BASE_NAMES

INDICES = list(range(22)) + list(range(38, 44)) + list(range(63, 107)) + list(range(127, 147))
NAMES = BASE_NAMES + ['candidate_mean_' + BASE_NAMES[i] for i in INDICES]


def augment(base):
    if not len(base):
        return np.empty((0, len(NAMES)), np.float32)
    context = np.asarray(base, np.float32)[:, INDICES].mean(0)
    return np.concatenate([base, np.broadcast_to(context, (len(base), len(context)))], axis=1).astype(np.float32)


def transform_dataset(x, offsets):
    """Vectorized means over ragged candidate groups, including all legal origins."""
    out = np.empty((len(x), len(NAMES)), np.float32)
    out[:, :len(BASE_NAMES)] = x
    for start in range(0, len(offsets)-1, 8192):
        stop = min(start+8192, len(offsets)-1)
        lo, hi = offsets[start], offsets[stop]
        counts = np.diff(offsets[start:stop+1])
        pooled = np.add.reduceat(x[lo:hi, INDICES], offsets[start:stop]-lo, axis=0) / counts[:, None]
        out[lo:hi, len(BASE_NAMES):] = np.repeat(pooled.astype(np.float32), counts, axis=0)
    return out
