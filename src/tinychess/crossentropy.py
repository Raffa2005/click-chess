"""Origin log loss, calibration, and explicitly empirical entropy diagnostics."""
from collections import Counter, defaultdict
import math
import numpy as np


def logsumexp(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    top = x.max(axis=axis, keepdims=True)
    return np.squeeze(top + np.log(np.exp(x - top).sum(axis=axis, keepdims=True)), axis)


def losses(logits, labels, temperature=1.0):
    """Padded invalid origins must have -inf logits. No probability clipping."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    logits = np.asarray(logits, dtype=np.float64) / temperature
    return logsumexp(logits) - logits[np.arange(len(logits)), labels]


def fit_temperature(logits, labels):
    """Golden-section minimization on validation data only; T in [0.1, 10]."""
    a, b = math.log(.1), math.log(10.)
    r = (math.sqrt(5) - 1) / 2
    c, d = b-r*(b-a), a+r*(b-a)
    f = lambda t: float(losses(logits, labels, math.exp(t)).mean())
    fc, fd = f(c), f(d)
    for _ in range(64):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b-r*(b-a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a+r*(b-a); fd = f(d)
    t = math.exp((a+b)/2)
    return t if losses(logits, labels, t).mean() <= losses(logits, labels).mean() else 1.0


def clustered_mean(values, games, draws=2000, seed=20261105):
    """Decision-weighted mean, resampling whole games; also supports paired deltas."""
    values = np.asarray(values, dtype=np.float64)
    _, inv = np.unique(games, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        ix = rng.integers(len(sums), size=len(sums))
        means.append(sums[ix].sum()/counts[ix].sum())
    return {'mean': float(values.mean()), 'ci95': np.quantile(means, [.025,.975]).tolist(),
            'decisions': len(values), 'games': len(sums)}


def empirical_floor(keys, labels):
    """Unrestricted lookup-table optimum ON THIS SAMPLE, never population entropy.

    A key must describe identical model inputs and a common origin-label ordering.
    Singletons contribute zero by construction. This optimistic plug-in quantity
    is provided with coverage, not presented as a generalization lower bound.
    """
    groups = defaultdict(Counter)
    for key, label in zip(keys, labels):
        groups[key][int(label)] += 1
    total = len(labels)
    repeated = conflicts = singleton = 0
    ent = 0.0
    for counter in groups.values():
        n = sum(counter.values())
        singleton += n == 1
        repeated += n if n > 1 else 0
        conflicts += n if len(counter) > 1 else 0
        ent -= sum(c * math.log(c/n) for c in counter.values())
    return {'empirical_lookup_minimum_nats': ent/total,
            'decisions': total, 'distinct_inputs': len(groups),
            'singleton_decisions': singleton, 'repeated_input_decisions': repeated,
            'conflicting_label_decisions': conflicts,
            'interpretation': 'In-sample memorization floor; NOT irreducible population loss.'}
