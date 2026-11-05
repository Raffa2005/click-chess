# Implementation and export attempts

- External pretrained 5M and 23M checkpoints loaded with strict parameter matching.
- Initial CPU PyTorch p95 was 67/116 ms on a short, concurrent-work benchmark;
  these measurements motivated the ONNX runtime, not a quality claim.
- Initial opset-18 export failed because RMSNorm was unsupported by the legacy
  exporter. Replaced that operator with its equivalent elementary arithmetic
  during export; checked output equivalence. Network weights were not trained.
- ONNX float32 numerical parity passed on 128 development positions; INT8 is
  approximate and evaluated explicitly. Its paired raw validation NLL change was
  +0.000146 nats (95% game-bootstrap CI -0.000354 to +0.000636).
- Independently timed 1/2/4-thread 5M runs had approximately 20 ms forward p95.
  More threads did not meet the initial 16.7 ms aim. Do not claim one-frame or
  physical-Android latency. Policy precomputation and repeat-click caches work.
- 23M is a larger, slower quality reference; 5M is the faster optional prototype.
- All calibration and choices are frozen before fresh-test scoring. Existing
  Android/JavaScript deployment is retained; no app release is produced here.
