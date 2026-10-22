# Experimental packed-row runtime source

SOURCE ONLY. UNWIRED, UNCOMPILED, UNRUN. Published for independent review and
verification, not as a usable backend or a performance/numerical claim.

These are integration sketches for `internal/compute`, not a standalone Go
package. The underscore directory excludes recursive Go package discovery;
`ignore` constraints additionally exclude its Go files from ordinary builds.
No shim includes, default dispatch, model calls, capabilities, registry entries,
or compiled shader modules are added. Do not remove these boundaries to infer
admission. `v41PackedAttentionKernel` remains unconditionally `nullptr`.

## Contents and provenance

- Private synchronous storage owner: whole 288-byte records, generation/live
  owner checks, checked copy/upload/zero, poisoned allocations and accounting.
- Private host-result attention bridge: ordered selection, separate positive
  finite scale eligibility, five-buffer/32-byte ABI, checked gather/dispatch,
  retained uncertain work, no output or fallback after selected failure.
- Unwired batch-lifetime bookkeeping and API proposal; see `SEAM.md`.
- Two authored, device-free Go regressions; neither is compiled or executed here.

Storage and binding prototypes were composed without changing their behavior.
Source dependencies were inspected in reviewed merge tree
`e2ea4b86a2909f57e64731fe56dfa6f6806dc694` (frozen source/public3b790 context).
This is source inspection, not proof of integration or ABI compatibility.
Original input SHA-256 values:

- Storage patch: `ac64e20e888044b440ab1a0010a74680e5188eb5273202ec067bfc5a532efdd9`
- Binding patch: `d5e108bbd4dc91f5d6e7412fc6aca185d995d4bbb10022743a1fce1ecd0fdcae`
- Batch bookkeeping header: `c97a29c96bb631c2a7e43cc5eaf82d149976258af1c72f29dd4abaa5b61dd512`

Shader source is a separate deliverable. No registry successor is reserved here;
V32 is already allocated. Any later registration must inspect the current registry
and use a separately reviewed append-only identity, not reuse an existing slot.

## Remaining validity gaps

Storage refuses open step/batch, inflight and restore activity, unhealthy context
and stale generation. These checks must remain intact. Native owner teardown,
Session lifetime and accounting integration are not implemented. Host-result
staging is not device-resident output, and owner-live charges are not physical
release or whole-process totals.

Attention admission still needs exact compiled module/ABI and binary32 float
controls, pipeline creation/retirement, generation reset and partial-failure
cleanup. Pipeline shape alone is insufficient. Existing F32 attention is a
different ABI and must not be substituted. There is no model publication path.

Batch use requires native-issued identity, checked-wait receipts and independent
native destruction/reuse guards; the bookkeeping class alone supplies none of
these proofs. Public initialized rows must not expose pending writes.

Before any enablement, independently compile/link the reconciled source and
execute authored regressions, checked end/reset/submit/wait fault tests, actual
visibility/quarantine tests, generation/alias/release cases and numerical device
witnesses. All are UNRUN for this package. Retain failures as evidence rather
than treating source publication as successful verification.
