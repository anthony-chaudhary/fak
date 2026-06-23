// Package residency is the multi-model weight-residency leaf: it hosts many
// prefill-warm *model.Model under one resident weight-byte budget with LRU page-out,
// reusing internal/polymodel.Pool as the budget + eviction POLICY and binding each
// admitted residency descriptor to the real in-kernel weights it governs.
//
// Tier: mechanism (2) — see internal/architest. It composes two foundation leaves
// (internal/model for the *model.Model weight handle, internal/polymodel for the proven
// residency + LRU policy) under the root ABI. It is rung #531 of epic #529
// (docs/serving/polymodel-prefill-share-plan.md §3, §7): the "host many models on one
// backend" axis, expressed as the layer that lifts the single-*model.Model assumption
// (internal/modelengine.Default is one *model.Model).
//
// # What it wires (the policy reused, the binding added)
//
// internal/polymodel.Pool already proves the residency DECISIONS — a weight-byte
// budget, LRU eviction of the coldest UNPINNED model, all-or-nothing admit, the
// pinned-exemption — but it "owns no weights and no KV (the model leaf does)". This
// leaf is the binding that turns those decisions into a live pool of *model.Model:
//
//   - Admit binds one residency descriptor (id, weightBytes, family, prefixDigest,
//     pinned) to a real *model.Model. The budget test and the LRU victim choice are
//     polymodel.Pool's; this layer only binds the descriptor to the weights and, on
//     eviction, hands the evicted *model.Model back so the caller can page it out /
//     free its resident memory (the page-out signal polymodel.Pool cannot give).
//   - Get / Touch / Evict / Resident / Used / Budget all delegate to polymodel.Pool,
//     so every invariant the polymodel witness suite asserts (used<=budget,
//     pinned-never-evicted, admit-unchanged-on-error) holds here by construction.
//
// # What it deliberately does NOT do (the honest boundary)
//
// residency moves no weight bytes and touches no GPU. It is the policy + binding layer:
// the real per-backend weight residency — actually loading/evicting weights on a device
// under the compute HAL's per-weight budget (internal/compute) and the process-wide
// gpulease — is the deeper GAP this rung's anchor names, sequenced for the backend
// wiring (a future rung consumes Admit/Evict to drive Upload/Release). WeightBytes is
// caller-supplied at Admit (consistent with polymodel.Model.WeightBytes); the real
// resident footprint a quantized backend reports is what the caller measures. No
// tokens/sec number is produced (rung #535).
//
// # Off by default until ready (two safety layers)
//
// This leaf is NOT blank-imported in the defconfig (internal/registrations), so the fak
// binary never even links it — the strongest gate, byte-unchanged kernel. It registers
// nothing with the ABI from init() (it is a library type a caller constructs, exactly
// like internal/polymodel), so it reaches a live request path only when a future rung
// constructs a Manager behind polymodel.Enabled() (FAK_POLYMODEL, default off). The
// pure helpers here are deterministic and safe to call directly.
package residency
