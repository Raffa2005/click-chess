"""Tiny permutation-equivariant scorer with learned candidate-set context."""
import json
import math
import pathlib
import time
import numpy as np
from .train import batch, probabilities, metric
from .research import fold_normalization


def forward(x, valid, params, pooling='mean'):
    w1, b1, w2, b2, w3, b3 = params
    h = np.maximum(x @ w1 + b1, 0)
    count = valid.sum(1, keepdims=True)
    mean = (h * valid[:, :, None]).sum(1) / count
    context = [h, np.broadcast_to(mean[:, None, :], h.shape)]
    maximum = None
    if pooling == 'mean_max':
        maximum = np.where(valid[:, :, None], h, -1e9).max(1)
        context.append(np.broadcast_to(maximum[:, None, :], h.shape))
    joined = np.concatenate(context, axis=2).astype(x.dtype)
    hidden = np.maximum(joined @ w2 + b2, 0)
    scores = (hidden @ w3 + b3).squeeze(-1)
    return scores, (x, valid, h, count, maximum, joined, hidden)


def backward(cache, params, score_gradient):
    x, valid, h, count, maximum, joined, hidden = cache
    w1, b1, w2, b2, w3, b3 = params
    grad = score_gradient[:, :, None]
    gw3 = hidden.reshape(-1, hidden.shape[-1]).T @ grad.reshape(-1, 1)
    gb3 = grad.sum((0, 1))
    dhidden = (grad @ w3.T) * (hidden > 0)
    gw2 = joined.reshape(-1, joined.shape[-1]).T @ dhidden.reshape(-1, hidden.shape[-1])
    gb2 = dhidden.sum((0, 1))
    djoined = dhidden @ w2.T
    width = h.shape[-1]
    dh = djoined[:, :, :width].copy()
    dmean = djoined[:, :, width:2*width].sum(1) / count
    dh += dmean[:, None, :] * valid[:, :, None]
    if maximum is not None:
        winners = (h == maximum[:, None, :]) & valid[:, :, None]
        dmax = djoined[:, :, 2*width:].sum(1)
        dh += winners * (dmax / winners.sum(1))[:, None, :]
    dh *= (h > 0)
    gw1 = x.reshape(-1, x.shape[-1]).T @ dh.reshape(-1, width)
    gb1 = dh.sum((0, 1))
    return [gw1, gb1, gw2, gb2, gw3, gb3]


def predict(d, idx, params, pooling='mean'):
    out = np.zeros((len(idx), int(d['n'][idx].max())), np.float32)
    for start in range(0, len(idx), 1024):
        ids = idx[start:start+1024]
        x, valid = batch(d, ids, 'history')
        scores, _ = forward(x, valid, params, pooling)
        out[start:start+len(ids), :valid.shape[1]] = probabilities(scores, valid)
    return out


def run(d, tr, val, config):
    rng = np.random.default_rng(config['seed'])
    dim = d['x'].shape[1]
    width, head = config['hidden']
    pooling = config.get('pooling', 'mean')
    shapes = [(dim, width), ((3 if pooling == 'mean_max' else 2)*width, head), (head, 1)]
    params = []
    for left, right in shapes:
        params += [(rng.normal(size=(left, right))*np.sqrt(2/left)).astype(np.float32), np.zeros(right, np.float32)]
    assert sum(p.size for p in params) <= 20000
    if config.get('init'):
        artifact = json.loads(pathlib.Path(config['init']).read_text())
        if artifact['feature_version'] != config['feature_version']:
            raise ValueError('Warm-start features differ')
        source = []
        for w, b in zip(artifact['weights'], artifact['biases']):
            source.extend([np.array(w, np.float32), np.array(b, np.float32)])
        for i, (target, value) in enumerate(zip(params, source)):
            if i == 2 and value.shape == (width, head):
                target[:] = 0
                target[:width] = value
            elif value.shape == target.shape:
                target[:] = value
            else:
                raise ValueError('Warm-start architecture differs')
    m, v = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
    best, best_loss, best_epoch, stale, step = None, float('inf'), 0, 0, 0
    start = time.perf_counter()
    curve = []
    for epoch in range(config['epochs']):
        fraction = min(1, max(epoch/max(1, config['epochs']-1), (time.perf_counter()-start)/config['seconds']))
        lr = config['lr']*(.05+.95*.5*(1+math.cos(math.pi*fraction)))
        total_loss, seen = 0., 0
        order = rng.permutation(tr)
        for offset in range(0, len(order), config.get('batch_size', 512)):
            ids = order[offset:offset+config.get('batch_size', 512)]
            x, valid = batch(d, ids, 'history')
            scores, cache = forward(x, valid, params, pooling)
            p = probabilities(scores, valid)
            total_loss += float(-np.log(np.maximum(p[np.arange(len(ids)), d['y'][ids]], 1e-12)).sum())
            seen += len(ids)
            p[np.arange(len(ids)), d['y'][ids]] -= 1
            grads = backward(cache, params, p/len(ids))
            step += 1
            for j, gradient in enumerate(grads):
                if j % 2 == 0:
                    gradient += config.get('l2', 1e-5)*params[j]
                m[j] = .9*m[j] + .1*gradient
                v[j] = .999*v[j] + .001*gradient*gradient
                params[j] -= lr*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
            if time.perf_counter()-start >= config['seconds']:
                break
        met = metric(predict(d, val, params, pooling), d['y'][val])
        row = dict(epoch=epoch+1, lr=lr, training_nll=total_loss/seen, seconds=time.perf_counter()-start, **met)
        curve.append(row)
        print(json.dumps(dict(experiment=config['name'], **row)), flush=True)
        if met['nll'] < best_loss-1e-5:
            best_loss, best_epoch, stale = met['nll'], epoch+1, 0
            best = [p.copy() for p in params]
        else:
            stale += 1
        if stale >= config['patience'] or time.perf_counter()-start >= config['seconds']:
            break
    p = predict(d, val, best, pooling)
    met = metric(p, d['y'][val])
    met['top2'] = float(np.any(np.argsort(-p, axis=1, kind='stable')[:, :2] == d['y'][val, None], axis=1).mean())
    artifact = dict(format='tiny-chess-input-v1', feature_version=config['feature_version'], architecture='set-context',
                    pooling=pooling, representation='history', hidden=config['hidden'], seed=config['seed'],
                    parameter_count=sum(p.size for p in best), weights=[p.tolist() for p in best[::2]],
                    biases=[p.tolist() for p in best[1::2]], research_experiment=config['name'])
    return artifact, dict(config=config, best_epoch=best_epoch, validation=met, parameter_count=artifact['parameter_count'],
                          seconds=time.perf_counter()-start, train_examples=len(tr), validation_examples=len(val), curve=curve)
