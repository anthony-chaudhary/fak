package rungobs

import (
	"context"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// subs is the STABLE subscription list returned by Subscriptions. It is a
// package-level value (not allocated per call) so Subscriptions is allocation-free
// and its identity is constant — buildEmitterIndex reads it exactly once, at
// registration. The observer scopes itself to the decision + vDSO events only, so
// EmittersFor never hands it an EvSubmit/EvDispatch/EvComplete: adding this observer
// adds zero work (and zero allocations — see TestSubscriptionsScopedAndZeroAllocOnUnsubscribed)
// to the every-syscall event path.
var subs = []abi.EventKind{abi.EvDecide, abi.EvDeny, abi.EvVDSOHit}

// dedupCap bounds the per-call dedup window. emit() is synchronous, so a call's
// paired decision events (the require-witness re-emit, or the EvDecide-then-EvDeny
// of a deny) arrive back-to-back with no room for more than a handful of other
// in-flight calls to interleave; 256 is far past any realistic concurrency in that
// microsecond window while keeping membership O(1) and memory bounded.
const dedupCap = 256

// decKey is one histogram bucket.
type decKey struct {
	rung   string
	kind   string
	reason string
}

// bucketCost is the per-bucket COST fold (#1149): the summed EvSubmit->EvDecide
// adjudication tax (elapsed-ns) and the signed token-delta (tokens ADDED by a
// transform vs SAVED by a vDSO hit) the kernel stamped on abi.Event.Fields. It rides
// beside the verdict count so the offline read-out folds cost, not just verdict.
type bucketCost struct {
	adjNanos   int64
	tokenDelta int64
}

// Observer is the passive rung-decision distribution counter. Construct it with
// New and register it with abi.RegisterEmitter; read the histogram with Snapshot.
//
// It is safe for concurrent use: every mutation goes through mu. The dedup window
// (seen + ring), the counts map, and the per-bucket cost are all guarded by it.
type Observer struct {
	mu     sync.Mutex
	counts map[decKey]int64
	cost   map[decKey]bucketCost // #1149: per-bucket adjudication-ns + token-delta
	seen   map[uint64]struct{}   // calls already counted (SeqNo > 0 only)
	ring   []uint64              // FIFO eviction order for `seen`
	rhead  int                   // next slot to overwrite in `ring`
}

// New returns an empty, ready-to-register Observer.
func New() *Observer {
	return &Observer{
		counts: map[decKey]int64{},
		cost:   map[decKey]bucketCost{},
		seen:   map[uint64]struct{}{},
		ring:   make([]uint64, dedupCap),
	}
}

// Subscriptions scopes the observer to EvDecide/EvDeny/EvVDSOHit (EventSubscriber).
// Returning a stable package-level slice keeps it allocation-free; the registry
// reads it once at registration.
func (o *Observer) Subscriptions() []abi.EventKind { return subs }

// Emit folds one lifecycle event into the histogram. It counts EXACTLY ONE bucket
// increment per call:
//
//   - A vDSO-served call (EvVDSOHit) ran no adjudication, so it is bucketed once
//     under rung="vdso" and never re-folded.
//   - A deny is carried by BOTH EvDecide and EvDeny on the Submit path; the EvDecide
//     is skipped (the deny Kind short-circuits) so the EvDeny counts it once. The
//     require-witness path emits EvDecide twice (the gate verdict, then the resolved
//     allow); the second is dropped by SeqNo dedup so the call counts once.
//
// EvDecide for a non-deny verdict is the single counting point for allow/transform.
// The observer never mutates the event, the call, the verdict, or any kernel state.
func (o *Observer) Emit(ev abi.Event) {
	adjNs, tokDelta := costOf(ev)
	if ev.Kind == abi.EvVDSOHit {
		// No adjudication ran; the verdict is always an allow by the vDSO. Bucket it
		// distinctly so a vDSO hit can never be misattributed to a structural rung.
		kind, reason := "ALLOW", ""
		if ev.Verdict != nil {
			kind = kindOf(ev.Verdict.Kind)
			reason = reasonOf(ev.Verdict.Reason)
		}
		o.bump("vdso", kind, reason, adjNs, tokDelta)
		return
	}
	if ev.Verdict == nil || ev.Call == nil {
		return
	}
	switch ev.Kind {
	case abi.EvDecide:
		// The deny verdict is also emitted as EvDeny below; defer to that event so a
		// denied call counts once. (The Decide() pure-adjudication path and the Submit
		// path both emit this EvDecide; only Submit's follow-up EvDeny carries the
		// authoritative deny, but the verdict object is identical either way.)
		if ev.Verdict.Kind == abi.VerdictDeny {
			return
		}
		// RequireWitness is an intermediate verdict on Submit: the kernel resolves it
		// to a final Allow or Deny and emits that second event. Count the final event
		// so the histogram reconciles with kernel.Counters(). A direct Decide() call
		// has SeqNo==0 and no follow-up, so it is still counted here.
		if ev.Verdict.Kind == abi.VerdictRequireWitness && ev.Call.SeqNo != 0 {
			return
		}
		if !o.claim(ev.Call.SeqNo) {
			return
		}
		o.attribute(ev.Call, ev.Verdict, adjNs, tokDelta)
	case abi.EvDeny:
		if !o.claim(ev.Call.SeqNo) {
			return
		}
		o.attribute(ev.Call, ev.Verdict, adjNs, tokDelta)
	}
}

// costOf reads the per-span COST the kernel stamped on the OPEN Event.Fields channel
// (#1149): the EvSubmit->EvDecide adjudication-ns and the signed token-delta. A missing
// or wrong-typed field reads as 0, so an event from a producer that does not stamp cost
// (or a pre-#1149 stream) folds as a zero-cost decision, never a panic.
func costOf(ev abi.Event) (adjNs, tokDelta int64) {
	if ev.Fields == nil {
		return 0, 0
	}
	if v, ok := ev.Fields[kernel.FieldElapsedNanos].(int64); ok {
		adjNs = v
	}
	if v, ok := ev.Fields[kernel.FieldTokenDelta].(int64); ok {
		tokDelta = v
	}
	return adjNs, tokDelta
}

// claim records that SeqNo's call as counted and reports whether this is the FIRST
// claim for it (so a re-emitted decision for the same call — the require-witness
// second EvDecide — is dropped). SeqNo == 0 is the Decide()/pure-adjudication path,
// which never re-emits a decision for one call, so it is always claimed with no
// bookkeeping.
func (o *Observer) claim(seq uint64) bool {
	if seq == 0 {
		return true
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if _, ok := o.seen[seq]; ok {
		return false
	}
	// Bound the window with FIFO eviction once full. rhead always points at the
	// oldest recorded SeqNo; the ring starts zeroed but 0 is never a real SeqNo in
	// `seen` (the seq==0 path returns above), so a stale zero victim is never deleted.
	if len(o.seen) >= dedupCap {
		delete(o.seen, o.ring[o.rhead])
	}
	o.seen[seq] = struct{}{}
	o.ring[o.rhead] = seq
	o.rhead = (o.rhead + 1) % dedupCap
	return true
}

// bump increments one (rung, kind, reason) bucket's count and folds its span cost
// (#1149): the adjudication-ns and signed token-delta sum into the bucket beside the
// verdict tally.
func (o *Observer) bump(rung, kind, reason string, adjNs, tokDelta int64) {
	k := decKey{rung, kind, reason}
	o.mu.Lock()
	o.counts[k]++
	c := o.cost[k]
	c.adjNanos += adjNs
	c.tokenDelta += tokDelta
	o.cost[k] = c
	o.mu.Unlock()
}

// attribute re-folds the call's chain off the hot path to recover the winning rung,
// then bumps its bucket. The re-fold is exact today because the hot path folds this
// same process-global registry chain (abi.AdjudicatorsFor); see the package doc for
// the honesty boundary. kind/reason are read from the canonical Decision so the
// bucket labels agree with `fak preflight --explain` for the same call.
func (o *Observer) attribute(call *abi.ToolCall, verdict *abi.Verdict, adjNs, tokDelta int64) {
	_, d := kernel.FoldExplain(context.Background(), abi.AdjudicatorsFor(call), call)
	kind, reason := d.Verdict, d.Reason
	if verdict != nil {
		kind = kindOf(verdict.Kind)
		reason = reasonOf(verdict.Reason)
	}
	o.bump(winnerRung(d), kind, reason, adjNs, tokDelta)
}

// winnerRung is the winning rung's concrete adjudicator type (the answer to "which
// rung decided"), falling back to the verdict's synthesized decider when no
// structural rung won — the empty-policy / all-defer fail-closed paths, which carry
// no winning RungVerdict.
func winnerRung(d kernel.Decision) string {
	for _, r := range d.Rungs {
		if r.Winner {
			return r.Rung
		}
	}
	if d.By != "" {
		return d.By
	}
	return "unknown"
}

// DecisionRow is one (rung, kind, reason) bucket and its count — the labeled
// prometheus counter row renderMetrics emits as fak_kernel_decisions_total. AdjNanos
// and TokenDelta carry the folded span COST (#1149): the summed EvSubmit->EvDecide
// adjudication tax (ns) and the signed token-delta (+added by transform/quarantine,
// -saved by vDSO/radix) for the bucket. AdjNanos is wall-clock and so NOT
// byte-stable — it is the offline twin of the live
// fak_gateway_operation_duration_seconds{adjudicator-rung}; TokenDelta is
// deterministic.
type DecisionRow struct {
	Rung       string
	Kind       string
	Reason     string
	Count      int64
	AdjNanos   int64
	TokenDelta int64
}

// Snapshot returns the per-(rung,kind,reason) counts and folded cost, sorted by
// (rung, kind, reason) for stable /metrics output. The slice is a fresh copy; callers
// may mutate it.
func (o *Observer) Snapshot() []DecisionRow {
	o.mu.Lock()
	rows := make([]DecisionRow, 0, len(o.counts))
	for k, c := range o.counts {
		bc := o.cost[k]
		rows = append(rows, DecisionRow{Rung: k.rung, Kind: k.kind, Reason: k.reason, Count: c, AdjNanos: bc.adjNanos, TokenDelta: bc.tokenDelta})
	}
	o.mu.Unlock()
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Rung != rows[j].Rung {
			return rows[i].Rung < rows[j].Rung
		}
		if rows[i].Kind != rows[j].Kind {
			return rows[i].Kind < rows[j].Kind
		}
		return rows[i].Reason < rows[j].Reason
	})
	return rows
}

// Total returns the sum of every bucket — the full decision count including the
// rung="vdso" bucket. For an adjudicated-only view, subtract the vDSO row count.
func (o *Observer) Total() int64 {
	o.mu.Lock()
	defer o.mu.Unlock()
	var t int64
	for _, c := range o.counts {
		t += c
	}
	return t
}

// kindOf mirrors kernel's verdict-kind names so the vDSO bucket label matches the
// structural Decision labels (which FoldExplain already stringifies).
func kindOf(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
	}
	return "KIND_UNKNOWN"
}

func reasonOf(r abi.ReasonCode) string {
	if r == abi.ReasonNone {
		return ""
	}
	return abi.ReasonName(r)
}
