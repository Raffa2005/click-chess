"""Portable little-endian deployment format for quantized click-chess models."""
import json
import struct
import numpy as np

MAGIC = b'CCQ1'


def encode(artifact):
    if 'weights_int8' not in artifact:
        raise ValueError('Binary deployment requires quantized weights')
    metadata = {k:v for k,v in artifact.items() if k not in ('weights_int8', 'weight_scales', 'biases', 'feature_names')}
    layers, payload = [], []
    for weights, biases, scales in zip(artifact['weights_int8'], artifact['biases'], artifact['weight_scales']):
        w = np.asarray(weights, dtype=np.int8)
        b = np.asarray(biases, dtype='<f4')
        s = np.atleast_1d(np.asarray(scales, dtype='<f4'))
        if w.ndim != 2 or len(b) != w.shape[1] or len(s) not in (1, w.shape[1]):
            raise ValueError('Invalid quantized layer dimensions')
        layers.append(dict(inputs=w.shape[0], outputs=w.shape[1], scales=len(s)))
        payload.extend([w.tobytes(), b.tobytes(), s.tobytes()])
    header = json.dumps(dict(artifact=metadata, layers=layers), separators=(',', ':')).encode()
    return MAGIC + struct.pack('<I', len(header)) + header + b''.join(payload)


def decode(data):
    if len(data) < 8 or data[:4] != MAGIC:
        raise ValueError('Unsupported binary model')
    length = struct.unpack_from('<I', data, 4)[0]
    if length > len(data)-8:
        raise ValueError('Truncated binary header')
    header = json.loads(data[8:8+length])
    artifact = dict(header['artifact'], weights_int8=[], biases=[], weight_scales=[])
    offset = 8+length
    for layer in header['layers']:
        a, b, n = layer['inputs'], layer['outputs'], layer['scales']
        if a <= 0 or b <= 0 or n not in (1, b) or offset+a*b+4*(b+n) > len(data):
            raise ValueError('Invalid or truncated binary layer')
        w = np.frombuffer(data, dtype=np.int8, count=a*b, offset=offset).reshape(a, b)
        offset += a*b
        bias = np.frombuffer(data, dtype='<f4', count=b, offset=offset)
        offset += 4*b
        scale = np.frombuffer(data, dtype='<f4', count=n, offset=offset)
        offset += 4*n
        artifact['weights_int8'].append(w.tolist())
        artifact['biases'].append(bias.tolist())
        artifact['weight_scales'].append(float(scale[0]) if n == 1 else scale.tolist())
    if offset != len(data):
        raise ValueError('Unexpected trailing binary data')
    return artifact
