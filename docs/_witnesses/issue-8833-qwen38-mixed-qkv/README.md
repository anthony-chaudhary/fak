# Issue 8833 mixed-QKV immutable witness packet

Status: **REJECT / selector remains default-off** until a real immutable packet is supplied.

This directory intentionally contains no benchmark receipts. Do not synthesize them. A reviewable
packet must add immutable control and candidate receipts plus their SHA-256 manifest. KEEP is valid
only when every quality, lifecycle, and identity gate passes; candidate median net decode is greater
than control; candidate emits fewer command-buffer events; and no individual run has a catastrophic
regression. REJECT is always valid and leaves `FAK_QWEN35_MIXED_QKV` off.

Required receipt facts:

- exact model/artifact and executable identities;
- q/k/v oracle errors within the existing tolerances;
- per-call unique call IDs and scoped execution events;
- control: exactly two committed command buffers and two completed waits;
- candidate: exactly one committed command buffer and one completed wait;
- no encode/commit/wait/readback hidden inside candidate sub-calls;
- complete lifecycle and cleanup/zero-ownership observations;
- repeated control/candidate net-decode measurements and individual-regression gate.
