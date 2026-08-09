# Scheduler token-service calibration — A100 witness (2026-08-09)

Issue: #5778

This ledger is a scrubbed public summary read back from a sanctioned lab GPU
node. The raw 27 controlled observations and six held-out observations remain in
the remote artifact named by the JSON provenance; private control identifiers do
not cross into this repository.

- Accelerator: A100-class accelerator
- Model/engine: Qwen 2.5 72B Instruct through an OpenAI-compatible SGLang endpoint
- Held-out weighted-model error: MAPE 11.04%, p95 APE 36.70%
- Held-out scalar-total baseline error: MAPE 171.94%, p95 APE 483.28%

These are **observed calibration results**, not a net scheduling gain claim:
queueing effects and scheduler overhead were not measured, so the net-true
boundary remains open.

