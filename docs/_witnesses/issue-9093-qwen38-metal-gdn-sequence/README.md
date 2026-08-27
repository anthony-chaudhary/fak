# Issue #9093 — resident Metal Qwen GDN sequence closure

Verdict: **REJECT for production promotion; implementation retained opt-in**.

The original work was decomposed into three landed leaves:

- #9095 added the backend-neutral preprojected-sequence capability and
  session-owned auxiliary-state lifecycle.
- #9096 added the fak-owned Metal sequence primitive with persistent,
  identity-stable convolution and recurrent buffers.
- #9097 wired that primitive into resident-Q4_K production prefill under
  `FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE=1`.

Focused portable and Darwin tests bind fail-before-mutation admission, distinct
session/layer state, unchanged handles, CPU-oracle output/final-state parity,
zero state H2D/D2H transfers and zero host recurrence steps inside an accepted
operation, no post-submit fallback, exact greedy-token handoff, and exact-once
cleanup.

The exact 36 GiB M3 Pro P=32,800/T=8 production canary remains the acceptance
authority. It reached
`metal/qwen35-gdn-preprojected-sequence-v1` with
`engine=inkernel`, `backend=metal`, Q4_K enabled, and fallback disabled, but
three consecutive samples crossed the 12 GiB swap-growth guard before response
headers. Actual memorystatus percentage was also unbound. The immutable raw
packet remains under
`../issue-9097-qwen38-metal-gdn-production/`; `packet.json` pins its hashes.
The selector therefore remains default-off.

The next measured boundary is #9230: graph-scoped activation/output ownership
and terminal synchronization across the whole Qwen forward, not another
subsystem-local resident wrapper.

## Readback

```console
go test ./docs/_witnesses/issue-9093-qwen38-metal-gdn-sequence -run '^TestQwen38MetalGDNSequenceWitness$' -count=1 -v
```
