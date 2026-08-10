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


## Held-workload scheduling value (2026-08-10)

The versioned `fak schedule-held` witness reorders the six hardware-measured
held requests without using their measured service time at admission. Under the
declared simultaneous-arrival, single-server, non-preemptive scope, weighted
shortest-predicted-job-first reduced mean completion from 5607.28 ms to
3575.13 ms versus the tuned scalar-total order: **2032.15 ms/request (36.24%)**.
The same ordering and negligible decision overhead were independently reproduced
through all three sanctioned lab nodes. Full scrubbed ledger:
[`held-schedule-fleet-a100-2026-08-10.json`](held-schedule-fleet-a100-2026-08-10.json).

This is an observed net gain for the declared held queue, not a claim about a
live multi-tenant production scheduler. Makespan and p95 were unchanged because
all six jobs arrive together and the longest completion is total service time.
