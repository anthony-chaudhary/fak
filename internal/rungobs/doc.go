// Package rungobs is the passive rung-decision distribution counter — the
// aggregate observability dual of `fak preflight --explain`.
//
// fak's pitch is a graduated capability floor: a call is decided by the cheapest
// adjudication rung that can establish the property, and only climbs to a costlier
// one when a cheaper one cannot conclude. `kernel.FoldExplain` already records
// WHICH rung won for a single hand-fed call; what was missing is the aggregate
// distribution over the LIVE stream — "is the deny path dominated by the name-level
// deny-map, by the witness gate, or by fail-closed default-deny? is the vDSO
// carrying real load or hitting 0%?". rungobs answers that without adding a single
// rung to adjudication.
//
// It is a pure abi.Emitter (the same seam harvest uses): register it with
// abi.RegisterEmitter and every decided call is folded into a labeled histogram
// keyed (rung, kind, reason). It subscribes — via EventSubscriber — to ONLY
// EvDecide/EvDeny/EvVDSOHit, so it is never invoked on the EvSubmit/EvDispatch/
// EvComplete path every syscall walks (EmittersFor is O(interested) and
// allocation-free). On a decision event it RE-FOLDS the call's chain with
// kernel.FoldExplain (off the critical section — the verdict is already resolved
// and emit is O(interested)) to recover the winning rung's concrete type, reads
// the verdict kind + reason, and bumps its bucket. A vDSO-served call ran no
// adjudication, so it is counted under a single rung="vdso" bucket.
//
// The same de-duplicated event fold also records the fused-turn KPI: classified
// calls (fusedturn.MetaClassKey) are grouped by TraceID, classical/weight op totals
// are counted by family, and a turn becomes fused exactly once when both families
// appear. Unknown ops are counted as unknown but do not enter the rate denominator.
//
// PASSIVE BY CONSTRUCTION. The observer never mutates the call, the verdict, or
// the kernel's Counters — it only reads. The hot path (kernel.Submit / Fold) is
// byte-for-byte unchanged whether or not rungobs is registered; the observer
// attaches entirely through the existing emit() fan-out.
//
// Honesty boundary — RE-DERIVATION, NOT GROUND TRUTH. rungobs re-derives the
// winning rung by re-folding the process-global registry chain
// (abi.AdjudicatorsFor) rather than reading it off the event. Today that re-fold
// is EXACT, because the hot path folds that same global chain. It would diverge
// only for a kernel constructed with an injected WithAdjudicators chain the
// global registry cannot reproduce — that is the documented Rung-B escalation
// trigger (carry the winning-rung identity on the Event), deliberately out of
// scope here. The counter is in-memory and process-local, like the existing
// fak_kernel_*_total counters; it resets on restart and is not persisted.
//
// Tier: mechanism (2) — see internal/architest. This package may import only
// packages whose tier is <= 2 (abi, fusedturn, kernel); an upward import fails the architest gate.
// See AGENTS.md and internal/architest for the layering contract.
package rungobs
