// Package polymodel is the deterministic core for hosting many models on one
// kernel and serializing decode to one lane — the "host 10s of models, share the
// prefill, decode one" design, expressed as proven arithmetic with no GPU, model,
// or network dependency.
//
// Tier: foundation (1) — see internal/architest. This package imports only the
// standard library (it is below even the root ABI), so it is testable on any host
// and never drags weights or a backend onto the request path.
//
// # The asymmetry it is built on
//
// A transformer turn has two cost regimes with opposite bottlenecks, both visible
// in internal/model:
//
//   - PREFILL is compute-bound. One weight stream is amortized over every prompt
//     token in a batched GEMM (model.prefillBatched / parallel.matMulBatch), so a
//     model's prefill is parallelizable and interleavable — the cheap, shareable
//     half. Keeping many models warm costs capacity (HBM residency), not bandwidth.
//   - DECODE is HBM-bandwidth-bound. Each token streams the model's WHOLE weight
//     set from memory to emit one token (model.Session.Step; the Q4/Q4K flags exist
//     precisely to cut decode bytes/token). Two models decoding at once need 2×
//     weight bandwidth against a fixed memory bus, so the decode lane must be
//     SERIAL: exactly one model decodes at a time. That is the "decode one" half.
//
// So you host many because residency is cheap, and you decode one because
// bandwidth is scarce. polymodel is the scheduling + accounting + speculative-accept
// math that makes that contract explicit and checkable.
//
// # What it provides (the proven layer)
//
//   - Pool: a prefill-warm multi-model residency set under one weight-byte budget,
//     with LRU eviction of the coldest UNPINNED model. The budget — not an
//     architectural limit — bounds how many models stay warm.
//   - Schedule / NextDecoder: a single serial decode lane time-shared across
//     models, with round-robin fairness and the load-bearing invariant that at most
//     one model decodes per step. DecodeBandwidthBytes quantifies why.
//   - AcceptGreedy / AcceptTree: the cache-led "next-gen multi-token prediction"
//     core. AcceptGreedy is the linear greedy speculative-decoding accept rule;
//     AcceptTree generalizes it to a token TREE (Medusa / EAGLE-2 / SpecInfer style),
//     where many candidate continuations share a KV prefix, are verified in one pass,
//     and only the accepted path is kept. Their KEEP/EVICT counts map 1:1 onto the
//     model leaf's bit-exact KV primitives (model.KVCache.Clone as the branch fork,
//     model.KVCache.Evict as the bit-exact rollback of rejected drafts/branches).
//   - PickDrafter / EffectiveTokensPerVerify: the ensemble-speculation policy (the
//     idle co-resident models double as drafters for the active decoder) and the
//     honest geometric-series model of the throughput it buys.
//   - CanShare: the cross-model prefill-share gate (the "especially the prefill" half)
//     — model B may reuse model A's already-computed prefix KV iff they share a
//     tokenizer (Family) and the prefill-relevant weights are byte-identical
//     (PrefixDigest), so the reuse is lossless. It is the verdict-layer unlock that
//     lifts the cache's exact-ModelID barrier; the KV splice itself is KVCache.Clone
//     at the engine layer (a sequenced GAP).
//
// # What it deliberately does NOT do (the honest boundary)
//
// polymodel runs no model and moves no KV bytes. It is the deterministic core that
// a future engine wiring drives: the GPU verify pass, the actual KVCache.Evict
// rollback, the multi-model residency on a real backend, and the cross-model
// prefill share (which the model-agnostic radix tree in internal/radixkv makes
// structurally possible but the cachemeta verdict layer still gates) are the
// integration rungs tracked in docs/serving/polymodel-prefill-share-plan.md. The
// ABI already carries the frozen seam for the speculative half (abi.SpeculationContext,
// abi.TxnID, abi.Outcome, abi.ProvisionalSink); polymodel is the policy/accounting
// brain that a ProvisionalSink implementation would consult.
//
// # Off by default until ready (two safety layers)
//
// This lane must not affect mainline until it is production-ready, so it is gated
// twice: (1) the leaf is deliberately NOT blank-imported in the defconfig
// (internal/registrations), so the kernel never even links it — the strongest gate;
// (2) when the integration rungs DO wire it onto a request path, that wiring must
// guard on Enabled() (the FAK_POLYMODEL env flag, default off). The pure helpers here
// are always safe to call (deterministic library functions touching no global state);
// only the live-path integration consults the flag.
package polymodel
