// Package spec is the speculative-execution leaf: the first implementation of the
// frozen abi.ProvisionalSink seam, and the registrant of the reserved OpsSpec ops
// (OpSpecCommit / OpSpecSquash). It turns the poly-model lane's accept DECISION
// (internal/polymodel: AcceptGreedy / AcceptTree) into the actual, bit-exact KV
// rollback — model.KVCache.Evict — so a speculative turn's KV is either promoted
// (the accepted branch becomes durable) or squashed (the rejected branch is removed
// byte-for-byte, as if it had never been drafted).
//
// Tier: mechanism (2) — see internal/architest. It composes two foundation leaves
// (internal/model for the bit-exact KVCache primitives, internal/polymodel for the
// accept arithmetic) under the root ABI; it is the speculation sibling of the
// context-MMU (the other ProvisionalSink). It is the rung #532 of epic #529
// (docs/serving/polymodel-prefill-share-plan.md §5 + §7).
//
// # What it wires (the seam made live)
//
// The ABI froze the speculation envelope and the retract contract but shipped no
// implementation: abi.SpeculationContext / abi.TxnID / abi.Outcome ride every
// ToolCall, abi.ProvisionalSink{Promote,Rollback} is the retract interface, the
// OpsSpec range [64,96) is reserved, and abi.RegisterProvisionalSink takes the
// registrant — but until this leaf there were ZERO ProvisionalSink registrants and
// no OpsSpec op. internal/spec supplies them:
//
//   - Sink implements abi.ProvisionalSink. A speculative turn records its
//     provisional KV span with Open; the kernel later resolves it. Promote drops the
//     bookkeeping (the accepted KV already lives in the cache). Rollback removes the
//     provisional span with model.KVCache.Evict — the bit-exact re-RoPE compaction
//     that makes the cache byte-identical to never having drafted the span (the
//     rollback a page-shared engine cannot do exactly).
//   - OpSpecCommit / OpSpecSquash are the reserved OpsSpec ops: invoking one fans
//     Promote / Rollback across every registered ProvisionalSink for the call's
//     (Txn, Spec.Epoch), so the speculation lifecycle is reachable through the
//     frozen kernel op table, not a private back door.
//   - SpeculativeGreedy is the lossless decode round-driver: each round it appends a
//     draft, lets polymodel.AcceptGreedy pick the accepted prefix, promotes that
//     prefix and squashes the rejected suffix THROUGH the Sink, and emits the
//     committed tokens. Greedy speculation is provably lossless, so its output is
//     token-identical to plain greedy decode — which holds ONLY if the squash is
//     bit-exact. That is the native witness the plan calls for (the cmd/polymodelbench
//     selfcheck made an in-tree test that goes through the ProvisionalSink seam).
//
// # The verify EXECUTION (rung #533, shipped)
//
// The verify step is the single-pass batched forward (model.VerifyForward), not kk
// sequential Steps: a kk-token draft is verified in ONE pass, bit-identical to the
// sequential verify (TestVerifyForwardChainMatchesSerial), so SpeculativeGreedy stays
// token-identical to plain greedy. The TREE twin — model.VerifyForward with depth-based
// positions (siblings share base+depth-1) and an ancestor attention mask (tree-attention
// masks, Medusa/EAGLE-2/SpecInfer) — is driven by VerifyTree/SpeculativeTree through
// AcceptTree; the accepted path is token-identical to greedy.
//
// # What it deliberately does NOT do (the honest boundary)
//
// internal/spec moves real KV bytes but on a CPU synthetic model (PreNorm regime); the
// single-pass verify is that regime only — no GPU, so no tokens/sec (the speedup is
// EffectiveTokensPerVerify arithmetic; a measured number needs the bench harness #535),
// and the tree recomputes the accepted path's KV (a tree-aware KV-compaction primitive is
// the sequenced cost, not a correctness gap). Multi-model residency on a backend is rung
// #531.
//
// # Off by default until ready (two safety layers)
//
// This leaf is NOT blank-imported in the defconfig (internal/registrations), so the
// fak binary never even links it — the strongest gate, byte-unchanged kernel. The
// second layer is Install(): it registers the sink + ops ONLY when the poly-model
// lane is enabled (polymodel.Enabled(); FAK_POLYMODEL, default off), and is a no-op
// returning nil otherwise — so even an accidental import never mutates the global
// ABI registries while the lane is off. The pure helpers (Sink, SpeculativeGreedy)
// are deterministic and safe to call directly; only Install touches global state.
package spec
