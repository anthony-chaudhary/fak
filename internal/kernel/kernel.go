// Package kernel is the fak core: the one concrete implementation of abi.Kernel.
// It is a driver-blind integrator — it never imports a leaf package; it only
// WALKS the abi registries (Adjudicators, FastPaths, ResultAdmitters, engines,
// emitters). That is what lets the leaf subsystems be built in disjoint trees and
// linked in by a single blank-import line in internal/registrations.
//
// The dispatch chain, in order:
//
//	Submit:  vDSO FastPath lookup  ->  (miss) fold Adjudicator chain  ->  route verdict
//	Reap:    engine.Complete       ->  fold ResultAdmitter chain (context-MMU)
//
// Adjudication happens entirely at Submit and touches neither the engine nor the
// network, so Submit's latency IS the tool-call adjudication latency the A/B
// benchmark reports. Reap carries the (slow, mockable) engine round-trip.
package kernel

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Counters are the kernel's call-path tallies (read by the metrics tap + tests).
type Counters struct {
	Submits      int64
	VDSOHits     int64
	EngineCalls  int64
	Denies       int64
	Transforms   int64
	Quarantines  int64
	ResultDenies int64
	Admitted     int64
}

// Kernel is the concrete abi.Kernel. Construct with New.
type Kernel struct {
	engineID string
	resolver abi.Resolver
	vdsoOff  bool // the --vdso=off ablation (unit 5): skip the fast path entirely

	// adjChain is an OPT-IN explicit adjudicator chain (see WithAdjudicators). When
	// non-nil, decide() folds THIS chain (scoped per call) instead of walking the
	// process-global abi.AdjudicatorsFor registry — letting independent kernels carry
	// independent policies and run CONCURRENTLY without colliding on the global
	// monitor's mutable policy. When nil (the default for every existing caller),
	// decide() reads the global registry EXACTLY as before, so back-compat is total.
	adjChain []abi.Adjudicator

	mu      sync.Mutex
	seq     uint64
	pending map[uint64]*pendingCall

	ctr Counters
}

// Option configures a Kernel at construction (a functional option, additive: New
// with no options builds the v0.1 kernel unchanged).
type Option func(*Kernel)

// WithAdjudicators injects an EXPLICIT adjudicator chain the kernel folds INSTEAD of
// the process-global abi.AdjudicatorsFor registry. It is the per-kernel adjudicator
// injection (issue #500): a replay driver can hand each of K policy arms its own
// monitor (e.g. []abi.Adjudicator{adjudicator.New(arm.Policy)}) and run them in
// parallel goroutines, with NO arm mutating the shared adjudicator.Default. The chain
// is folded per call through the same restrictiveness lattice and the same CallScope
// tool-scoping as the global path (via abi.ScopedFor), so a kernel given the global
// chain is verdict-identical to one reading the registry. Passing an empty/nil chain
// is a NO-OP — the kernel falls back to the global registry — so it can never silently
// install an empty (default-deny-everything) policy.
func WithAdjudicators(chain []abi.Adjudicator) Option {
	return func(k *Kernel) {
		if len(chain) > 0 {
			k.adjChain = chain
		}
	}
}

// SetVDSO toggles the vDSO fast path (the --vdso on/off ablation, unit 5). When
// off, Submit never consults a FastPath, so every call falls through to
// adjudication + engine — the "off" arm of the A/B benchmark.
func (k *Kernel) SetVDSO(enabled bool) { k.vdsoOff = !enabled }

// VDSOEnabled reports whether the fast path is active.
func (k *Kernel) VDSOEnabled() bool { return !k.vdsoOff }

type pendingCall struct {
	call    *abi.ToolCall
	verdict abi.Verdict
	ready   *abi.Result // set when the fast path already produced a result
	denied  bool
}

// New builds a kernel bound to a registered engine id ("" => first/any engine,
// or a no-op engine if none registered). The Resolver comes from the registered
// RegionBackend (blob store in v0.1). Options (e.g. WithAdjudicators) are applied
// after construction; New with no options builds the v0.1 kernel unchanged.
func New(engineID string, opts ...Option) *Kernel {
	k := &Kernel{
		engineID: engineID,
		resolver: abi.ActiveResolver(),
		pending:  map[uint64]*pendingCall{},
	}
	for _, opt := range opts {
		opt(k)
	}
	return k
}

// Counters returns a snapshot of the kernel's call-path tallies.
func (k *Kernel) Counters() Counters {
	return Counters{
		Submits:      atomic.LoadInt64(&k.ctr.Submits),
		VDSOHits:     atomic.LoadInt64(&k.ctr.VDSOHits),
		EngineCalls:  atomic.LoadInt64(&k.ctr.EngineCalls),
		Denies:       atomic.LoadInt64(&k.ctr.Denies),
		Transforms:   atomic.LoadInt64(&k.ctr.Transforms),
		Quarantines:  atomic.LoadInt64(&k.ctr.Quarantines),
		ResultDenies: atomic.LoadInt64(&k.ctr.ResultDenies),
		Admitted:     atomic.LoadInt64(&k.ctr.Admitted),
	}
}

// Decide folds ONLY the Adjudicator chain and returns the resolved Verdict. It
// touches no fast path, engine, or network — it is the pure in-process
// adjudication path the benchmark times against a spawned hook. Exported so the
// `fak hook` spawned-baseline mode and BenchmarkDecide share one code path.
func (k *Kernel) Decide(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	v := k.decide(ctx, c)
	emit(abi.Event{Kind: abi.EvDecide, Call: c, Verdict: &v})
	if v.Kind == abi.VerdictDeny {
		emit(abi.Event{Kind: abi.EvDeny, Call: c, Verdict: &v})
	}
	return v
}

func (k *Kernel) decide(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	return Fold(ctx, k.adjudicatorsFor(c), c)
}

// adjudicatorsFor returns the rung chain this call folds. With an explicit chain
// injected (WithAdjudicators) it folds THAT chain, tool-scoped exactly as the global
// path is (abi.ScopedFor mirrors abi.AdjudicatorsFor's CallScope filtering); with no
// injection it walks the process-global registry via abi.AdjudicatorsFor — bit-for-bit
// the v0.1 behavior. The injected path touches nothing global, so two kernels with
// different chains adjudicate concurrently without colliding.
func (k *Kernel) adjudicatorsFor(c *abi.ToolCall) []abi.Adjudicator {
	if k.adjChain != nil {
		return abi.ScopedFor(k.adjChain, c)
	}
	return abi.AdjudicatorsFor(c)
}

// BatchDecide vectorizes adjudication over a list of calls in one pass (unit 75,
// the "set" batch shape — dos-plan-price generalized). The result is, by
// construction, identical to deciding each call serially: it simply folds the
// same chain per call without a per-call kernel round-trip. The dual of
// speculative decoding — the expensive model proposes a plan, the cheap kernel
// prunes it in one pass.
func (k *Kernel) BatchDecide(ctx context.Context, calls []*abi.ToolCall) []abi.Verdict {
	out := make([]abi.Verdict, len(calls))
	for i, c := range calls {
		// Per-call chain selection: a batch may mix tools, and AdjudicatorsFor is
		// O(1), so each call folds only the rungs that can refuse its tool. With no
		// tool-scoped rung registered this returns the same full chain for every
		// call, so the batch result is unchanged. An injected explicit chain (see
		// WithAdjudicators) is folded the same way, scoped per call.
		out[i] = Fold(ctx, k.adjudicatorsFor(c), c)
	}
	return out
}

// Fold runs an Adjudicator chain and resolves it by the restrictiveness lattice:
// the most-restrictive conclusive verdict wins (fail-closed). An empty chain
// yields Deny (default-deny on no policy — unit 15). A Defer from every link also
// yields Deny (nothing affirmatively allowed it). An Indeterminate is non-
// committable: a later conclusive rung resolves it; a residual Indeterminate
// fails closed.
func Fold(ctx context.Context, chain []abi.Adjudicator, c *abi.ToolCall) abi.Verdict {
	if len(chain) == 0 {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "empty-policy"}
	}
	best := abi.Verdict{Kind: abi.VerdictDefer, By: "no-link"}
	bestRank := -1
	sawConclusive := false
	sawIndeterminate := false
	indeterminateBy := ""
	for _, a := range chain {
		v := a.Adjudicate(ctx, c)
		switch v.Kind {
		case abi.VerdictDefer:
			continue
		case abi.VerdictIndeterminate:
			sawIndeterminate = true
			if indeterminateBy == "" {
				indeterminateBy = v.By
			}
			continue
		}
		if r := abi.FoldRank(v.Kind); r > bestRank {
			bestRank, best = r, v
			sawConclusive = true
			if isMaxFoldRank(r) {
				break
			}
		}
	}
	if sawConclusive {
		return best
	}
	if sawIndeterminate {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: indeterminateBy,
			Meta: map[string]string{"fold": "indeterminate"}}
	}
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "all-defer"}
}

func isMaxFoldRank(rank int) bool {
	return rank >= abi.FoldRank(abi.VerdictDeny)
}

// resolveWitness drives the require-witness gate. It asks every registered
// WitnessResolver to corroborate the verdict's claimed EFFECT from evidence the
// agent did not author (git ancestry, object existence, a tracked path...). The
// first CONFIRMED opens the gate (Allow); any REFUTED is a provable trust
// violation; otherwise (every resolver abstains, or none is registered) the gate
// stays closed with ReasonUnwitnessed — the fail-closed default that matches v0.1
// exactly when no resolver exists.
func (k *Kernel) resolveWitness(ctx context.Context, c *abi.ToolCall, v abi.Verdict) abi.Verdict {
	claim := ""
	if wp, ok := v.Payload.(abi.WitnessPayload); ok {
		claim = wp.Claim
	}
	refuted := false
	for _, w := range abi.Witnesses() {
		switch w.Resolve(ctx, c, claim) {
		case abi.WitnessConfirmed:
			return abi.Verdict{Kind: abi.VerdictAllow, By: "witness",
				Meta: map[string]string{"witness": "confirmed", "claim": claim}}
		case abi.WitnessRefuted:
			refuted = true
		}
	}
	reason := abi.ReasonUnwitnessed
	outcome := "unwitnessed"
	if refuted {
		reason = abi.ReasonTrustViolation
		outcome = "refuted"
	}
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: reason, By: "witness",
		Meta: map[string]string{"witness": outcome, "claim": claim}}
}

// AdmitResult runs a produced result through the ResultAdmitter chain (the
// result-side IFC containment + quarantine + per-trace taint ledger) and returns
// the resolved admission verdict. It is the EXPORTED dual of Decide (which folds
// only the pre-call chain): the gateway's served path calls it to arm the
// result-side stack on a result a CLIENT produced and handed back over the wire,
// so the exfil floor is no longer inert on the proxy/adjudicate topology. It has
// exactly the same effect on r as the in-process Reap path's admitResult.
func (k *Kernel) AdmitResult(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	return k.admitResult(ctx, c, r)
}

// admitResult folds the ResultAdmitter (context-MMU) chain over a produced
// result, applying the most-restrictive admission verdict, and returns it.
func (k *Kernel) admitResult(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	chain := abi.ResultAdmittersFor(c)
	if len(chain) == 0 || r == nil {
		atomic.AddInt64(&k.ctr.Admitted, 1)
		return abi.Verdict{Kind: abi.VerdictAllow, By: "default-admit"}
	}
	best := abi.Verdict{Kind: abi.VerdictAllow, By: "default-admit"}
	bestRank := abi.FoldRank(abi.VerdictAllow)
	for _, ra := range chain {
		v := ra.Admit(ctx, c, r)
		if rk := abi.FoldRank(v.Kind); rk > bestRank {
			bestRank, best = rk, v
		}
	}
	switch best.Kind {
	case abi.VerdictQuarantine:
		atomic.AddInt64(&k.ctr.Quarantines, 1)
		// The admitter is responsible for having paged the bytes out and
		// rewritten r.Payload to a pointer; record the resolution.
		r.Outcome = abi.OutcomeCommitted
		if r.Meta == nil {
			r.Meta = map[string]string{}
		}
		r.Meta["admit"] = "quarantined"
		emit(abi.Event{Kind: abi.EvQuarantine, Call: c, Verdict: &best, Result: r})
	case abi.VerdictTransform:
		if tp, ok := best.Payload.(abi.TransformPayload); ok {
			r.Payload = tp.NewArgs
		}
		if r.Meta == nil {
			r.Meta = map[string]string{}
		}
		r.Meta["admit"] = "transformed"
		atomic.AddInt64(&k.ctr.Admitted, 1)
	case abi.VerdictRequireWitness:
		resolved := k.resolveWitness(ctx, c, best)
		if resolved.Kind == abi.VerdictAllow {
			atomic.AddInt64(&k.ctr.Admitted, 1)
			return resolved
		}
		return k.denyResultAdmission(c, r, resolved)
	case abi.VerdictDeny:
		return k.denyResultAdmission(c, r, best)
	default:
		atomic.AddInt64(&k.ctr.Admitted, 1)
	}
	return best
}

func (k *Kernel) denyResultAdmission(c *abi.ToolCall, r *abi.Result, v abi.Verdict) abi.Verdict {
	atomic.AddInt64(&k.ctr.ResultDenies, 1)
	denied := DenyResult(c, v)
	denied.Meta["admit"] = "denied"
	if r != nil {
		*r = *denied
	}
	emit(abi.Event{Kind: abi.EvResultDeny, Call: c, Verdict: &v, Result: r})
	return v
}

// Submit adjudicates and admits the call. The vDSO is consulted FIRST (unit 30);
// a hit returns immediately with no adjudication and no engine call. On a miss
// the Adjudicator chain is folded; a denied call is never enqueued.
func (k *Kernel) Submit(ctx context.Context, c *abi.ToolCall) (abi.SubmissionHandle, abi.Verdict) {
	atomic.AddInt64(&k.ctr.Submits, 1)
	k.mu.Lock()
	k.seq++
	if c.SeqNo == 0 {
		c.SeqNo = k.seq
	}
	seq := c.SeqNo
	k.mu.Unlock()
	h := abi.SubmissionHandle{Seq: seq}

	emit(abi.Event{Kind: abi.EvSubmit, Call: c})

	// vDSO fast path first (skipped entirely in the --vdso=off ablation arm).
	for _, fp := range abi.FastPaths() {
		if k.vdsoOff {
			break
		}
		if r, ok := fp.Lookup(ctx, c); ok {
			atomic.AddInt64(&k.ctr.VDSOHits, 1)
			v := abi.Verdict{Kind: abi.VerdictAllow, By: "vdso"}
			k.store(seq, &pendingCall{call: c, verdict: v, ready: r})
			emit(abi.Event{Kind: abi.EvVDSOHit, Call: c, Verdict: &v, Result: r})
			return h, v
		}
	}

	// Adjudicate.
	v := k.decide(ctx, c)
	emit(abi.Event{Kind: abi.EvDecide, Call: c, Verdict: &v})
	switch v.Kind {
	case abi.VerdictRequireWitness:
		// Resolve the require-witness gate against independent evidence (the witness
		// rung): a CONFIRMED claim opens the gate (allow); a refuted/uncorroborated
		// one stays closed (fail-closed). With no resolver registered every claim
		// abstains, so this preserves v0.1's deny-on-require-witness behavior.
		v = k.resolveWitness(ctx, c, v)
		if v.Kind == abi.VerdictAllow {
			k.store(seq, &pendingCall{call: c, verdict: v})
			emit(abi.Event{Kind: abi.EvDecide, Call: c, Verdict: &v})
			return h, v
		}
		atomic.AddInt64(&k.ctr.Denies, 1)
		k.store(seq, &pendingCall{call: c, verdict: v, denied: true})
		emit(abi.Event{Kind: abi.EvDeny, Call: c, Verdict: &v})
		return h, v
	case abi.VerdictDeny:
		atomic.AddInt64(&k.ctr.Denies, 1)
		k.store(seq, &pendingCall{call: c, verdict: v, denied: true})
		emit(abi.Event{Kind: abi.EvDeny, Call: c, Verdict: &v})
		return h, v
	case abi.VerdictTransform:
		atomic.AddInt64(&k.ctr.Transforms, 1)
		if tp, ok := v.Payload.(abi.TransformPayload); ok {
			c.Args = tp.NewArgs
		}
		k.store(seq, &pendingCall{call: c, verdict: v})
		return h, v
	case abi.VerdictAllow, abi.VerdictDefer:
		// The only dispatching outcomes: an affirmative allow (or a defer that the
		// fold left unrefused). Everything else is held.
		k.store(seq, &pendingCall{call: c, verdict: v})
		return h, v
	default:
		// A non-allow verdict the core does not special-case — a REGISTERED
		// escalation/restrictive kind (e.g. plancfi's RequireApproval). Fail-closed:
		// the call is HELD, not dispatched, and surfaced as a deny-as-value carrying
		// the verdict kind so a host harness can route the escalation (human
		// approval). Nothing in v0.1 produces such a kind, so this only ever engages
		// for an additive driver and never changes the Allow/Deny/Transform paths.
		atomic.AddInt64(&k.ctr.Denies, 1)
		k.store(seq, &pendingCall{call: c, verdict: v, denied: true})
		emit(abi.Event{Kind: abi.EvDeny, Call: c, Verdict: &v})
		return h, v
	}
}

func (k *Kernel) store(seq uint64, p *pendingCall) {
	k.mu.Lock()
	k.pending[seq] = p
	k.mu.Unlock()
}

// TestHandle is the concrete-kernel non-consuming poll for a submitted handle.
// It borrows MPI_Test's request-poll shape only: polling drives no progress engine
// and does not dispatch a tool. StatusPending means this process still has an
// admitted call that must run its engine round-trip via Reap. Already-ready fast
// path hits, denied calls, and unknown/reaped handles are locally complete from
// the poller's point of view and report StatusOK; Reap remains the consuming path.
func (k *Kernel) TestHandle(h abi.SubmissionHandle) abi.Status {
	k.mu.Lock()
	p := k.pending[h.Seq]
	k.mu.Unlock()
	if p == nil || p.ready != nil || p.denied {
		return abi.StatusOK
	}
	return abi.StatusPending
}

// ErrDenied is returned by Reap for a call that adjudication refused.
var ErrDenied = errors.New("kernel: call denied by adjudicator")

// Reap completes a submission. A fast-path result is returned directly. A denied
// call yields a structured deny Result (Status=Error, Meta carries reason +
// disposition) so the loop can consume it (deny-as-value). An allowed call is
// dispatched to the engine and the result run through the ResultAdmitter chain.
func (k *Kernel) Reap(ctx context.Context, h abi.SubmissionHandle) (*abi.Result, error) {
	k.mu.Lock()
	p := k.pending[h.Seq]
	delete(k.pending, h.Seq)
	k.mu.Unlock()
	if p == nil {
		return nil, fmt.Errorf("kernel: no pending submission %d", h.Seq)
	}
	if p.ready != nil { // vDSO hit
		return p.ready, nil
	}
	if p.denied {
		return DenyResult(p.call, p.verdict), nil
	}
	// Dispatch to the selected engine. A call-level Engine route overrides the
	// kernel default, preserving the old process-wide binding when unset.
	route := k.routeFor(p.call)
	eng := abi.Engine(route)
	if eng == nil {
		return nil, fmt.Errorf("kernel: no engine registered for route %q", route)
	}
	atomic.AddInt64(&k.ctr.EngineCalls, 1)
	emit(abi.Event{Kind: abi.EvDispatch, Call: p.call, Verdict: &p.verdict})
	r, err := eng.Complete(ctx, p.call)
	if err != nil {
		return nil, err
	}
	if r.Call == nil {
		r.Call = p.call
	}
	k.admitResult(ctx, p.call, r)
	emit(abi.Event{Kind: abi.EvComplete, Call: p.call, Result: r})
	return r, nil
}

func (k *Kernel) routeFor(c *abi.ToolCall) string {
	if c != nil && c.Engine != "" {
		return c.Engine
	}
	return k.engineID
}

// emit fans an event to the observers SUBSCRIBED to its kind (KPI taps, the vDSO
// cache-fill, stewards, the label harvester). EmittersFor returns only the
// interested observers (an observer that didn't scope itself via EventSubscriber
// is universal and receives every kind), so emit cost is O(interested), not
// O(all observers) — adding an observer that only watches EvDeny adds nothing to
// the EvSubmit/EvDispatch/EvComplete path every syscall walks. A nil/empty
// registry is a no-op.
func emit(ev abi.Event) {
	for _, e := range abi.EmittersFor(ev.Kind) {
		e.Emit(ev)
	}
}

// Syscall is Submit then Reap (the synchronous convenience every caller uses).
func (k *Kernel) Syscall(ctx context.Context, c *abi.ToolCall) (*abi.Result, abi.Verdict) {
	h, v := k.Submit(ctx, c)
	r, err := k.Reap(ctx, h)
	if err != nil {
		return &abi.Result{Call: c, Status: abi.StatusError,
			Meta: map[string]string{"error": err.Error()}}, v
	}
	return r, v
}

// Resolver is the active Ref backend.
func (k *Kernel) Resolver() abi.Resolver { return k.resolver }

// Negotiate intersects advertised caps with registered ones.
func (k *Kernel) Negotiate(advertised []abi.Capability) []abi.Capability {
	var out []abi.Capability
	for _, c := range advertised {
		if abi.Supported(c) {
			out = append(out, c)
		}
	}
	return out
}

// DenyResult builds the structured deny-as-value (unit 74): a Result carrying the
// reason token + derived disposition the next model turn consumes, with bounded
// witness disclosure (only the offending set, sourced from the verdict).
func DenyResult(c *abi.ToolCall, v abi.Verdict) *abi.Result {
	meta := map[string]string{
		"verdict":     "deny",
		"reason":      abi.ReasonName(v.Reason),
		"disposition": Disposition(v.Reason),
		"by":          v.By,
	}
	if wp, ok := v.Payload.(abi.WitnessPayload); ok && wp.Claim != "" {
		meta["witness"] = wp.Claim // bounded disclosure: the offending set only
	}
	// Issue #699: surface the advisory retry-after on a WAIT disposition — the
	// recoverable back-off the loop pairs with WAIT the way errno pairs EAGAIN with
	// a retry window. Only WAIT denies carry one (a RATE_LIMITED over-cap), and only
	// when the verdict supplies it; a non-WAIT deny or a WAIT without a hint degrades
	// to today's bare-token behavior.
	if meta["disposition"] == "WAIT" && v.Meta != nil {
		if ra := v.Meta["retry_after"]; ra != "" {
			meta["retry_after"] = ra
		}
		if ra := v.Meta["retry_after_ms"]; ra != "" {
			meta["retry_after_ms"] = ra
		}
	}
	return &abi.Result{Call: c, Status: abi.StatusError, Outcome: abi.OutcomeCommitted, Meta: meta}
}

// Disposition derives the deny-loopback disposition (RETRYABLE / WAIT / ESCALATE /
// TERMINAL) from the reason's category. Only MISROUTE is model-fixable
// (RETRYABLE); the rest steer the loop without another wasted model turn.
func Disposition(r abi.ReasonCode) string {
	switch r {
	case abi.ReasonMisroute, abi.ReasonMalformed:
		return "RETRYABLE"
	case abi.ReasonRateLimited, abi.ReasonLeaseHeld:
		return "WAIT"
	case abi.ReasonSelfModify, abi.ReasonTrustViolation:
		return "ESCALATE"
	default:
		return "TERMINAL"
	}
}

var _ abi.Kernel = (*Kernel)(nil)
