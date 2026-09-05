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
	"reflect"
	"sync"
	"sync/atomic"
	"time"

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
	t0 := time.Now()
	v := k.decide(ctx, c)
	// L0 self-tax (#1149): stamp the EvSubmit->EvDecide adjudication tax (elapsed-ns)
	// and any transform token-delta on the events rungobs folds — EvDecide for an
	// allow/transform, EvDeny for a deny.
	fields := costFields(time.Since(t0).Nanoseconds(), transformTokenDelta(v, c.Args))
	emit(abi.Event{Kind: abi.EvDecide, Call: c, Verdict: &v, Fields: fields})
	if v.Kind == abi.VerdictDeny {
		emit(abi.Event{Kind: abi.EvDeny, Call: c, Verdict: &v, Fields: fields})
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
	if pc, ok := abi.PolicyFromContext(ctx); ok && pc.Posture == abi.PostureDefaultOpen {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "all-defer(default-open)"}
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
		reason = abi.ReasonIntegrityRefuted
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
			// A vDSO hit ran no adjudication (0 ns of tax) and SAVED the engine
			// round-trip: the token-delta is negative, the local result's tokens the
			// kernel did not have to generate (#1149).
			emit(abi.Event{Kind: abi.EvVDSOHit, Call: c, Verdict: &v, Result: r, Fields: costFields(0, -refLen(r.Payload))})
			return h, v
		}
	}

	// Adjudicate. Time the fold so the EvSubmit->EvDecide adjudication tax (#1149)
	// rides every decision event rungobs folds; the transform token-delta is read
	// off the ORIGINAL args, before the Transform arm rewrites c.Args below.
	t0 := time.Now()
	v := k.decide(ctx, c)
	adjNs := time.Since(t0).Nanoseconds()
	emit(abi.Event{Kind: abi.EvDecide, Call: c, Verdict: &v, Fields: costFields(adjNs, transformTokenDelta(v, c.Args))})
	switch v.Kind {
	case abi.VerdictRequireWitness:
		// Resolve the require-witness gate against independent evidence (the witness
		// rung): a CONFIRMED claim opens the gate (allow); a refuted/uncorroborated
		// one stays closed (fail-closed). With no resolver registered every claim
		// abstains, so this preserves v0.1's deny-on-require-witness behavior.
		v = k.resolveWitness(ctx, c, v)
		if v.Kind == abi.VerdictAllow {
			k.store(seq, &pendingCall{call: c, verdict: v})
			emit(abi.Event{Kind: abi.EvDecide, Call: c, Verdict: &v, Fields: costFields(adjNs, 0)})
			return h, v
		}
		atomic.AddInt64(&k.ctr.Denies, 1)
		k.store(seq, &pendingCall{call: c, verdict: v, denied: true})
		emit(abi.Event{Kind: abi.EvDeny, Call: c, Verdict: &v, Fields: costFields(adjNs, 0)})
		return h, v
	case abi.VerdictDeny:
		atomic.AddInt64(&k.ctr.Denies, 1)
		k.store(seq, &pendingCall{call: c, verdict: v, denied: true})
		emit(abi.Event{Kind: abi.EvDeny, Call: c, Verdict: &v, Fields: costFields(adjNs, 0)})
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
		emit(abi.Event{Kind: abi.EvDeny, Call: c, Verdict: &v, Fields: costFields(adjNs, 0)})
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
	status, _ := k.testHandle(h)
	return status
}

func (k *Kernel) testHandle(h abi.SubmissionHandle) (abi.Status, bool) {
	k.mu.Lock()
	p := k.pending[h.Seq]
	k.mu.Unlock()
	if p == nil || p.ready != nil || p.denied {
		return abi.StatusOK, p != nil
	}
	return abi.StatusPending, true
}

// ErrNoHandles is returned by ReapAny for an empty request set.
var ErrNoHandles = errors.New("kernel: no submission handles")

// ErrDenied is returned by Reap for a call that adjudication refused.
var ErrDenied = errors.New("kernel: call denied by adjudicator")

// ReapAny completes one handle from a request set. It first consumes an already
// locally-complete handle (fast-path hit or denied call) if one is present. With no
// progress engine to poll, an all-pending set is driven by reaping the first handle
// in the caller's set. The method consumes exactly the returned handle; other
// handles remain pending for later Reap/ReapAll calls.
//
// It matches a completion to its submission by SubmissionHandle.Seq; the Queue/Opaque
// fields are reserved, inert routing/correlation slots no scheduler reads here (there
// is one global engine fold, no multi-queue tag-matcher). The async-addressing seam —
// what Seq/Queue/Opaque and the Ext/ExtKey sidecar are shaped for, and the MPI-analogue
// caveats — is documented in docs/proofs/async-addressing.md.
func (k *Kernel) ReapAny(ctx context.Context, handles []abi.SubmissionHandle) (abi.SubmissionHandle, *abi.Result, error) {
	if len(handles) == 0 {
		return abi.SubmissionHandle{}, nil, ErrNoHandles
	}
	for _, h := range handles {
		if status, ok := k.testHandle(h); ok && status == abi.StatusOK {
			r, err := k.Reap(ctx, h)
			return h, r, err
		}
	}
	h := handles[0]
	r, err := k.Reap(ctx, h)
	return h, r, err
}

// ReapAll completes every handle in the caller's order and returns results in the
// same index order. It borrows MPI_Waitall's request-set shape only: each handle
// was already adjudicated at Submit, and this fan-in performs no new admission or
// transport progress beyond the underlying local Reap calls.
func (k *Kernel) ReapAll(ctx context.Context, handles []abi.SubmissionHandle) ([]*abi.Result, error) {
	results := make([]*abi.Result, len(handles))
	if len(handles) == 0 {
		return results, nil
	}

	var wg sync.WaitGroup
	errs := make(chan error, len(handles))
	for i, h := range handles {
		wg.Add(1)
		go func(i int, h abi.SubmissionHandle) {
			defer wg.Done()
			r, err := k.Reap(ctx, h)
			if err != nil {
				errs <- fmt.Errorf("handle %d/%d seq %d: %w", i, len(handles), h.Seq, err)
				return
			}
			results[i] = r
		}(i, h)
	}
	wg.Wait()
	close(errs)
	if err, ok := <-errs; ok {
		return results, err
	}
	return results, nil
}

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
	// Time the EvDispatch->EvComplete engine span (#1149): the engine cost, stamped
	// on the completion event so a cost observer can separate engine time from the
	// adjudication tax EvDecide already carries.
	t0 := time.Now()
	r, err := eng.Complete(ctx, p.call)
	engNs := time.Since(t0).Nanoseconds()
	if err != nil {
		return nil, err
	}
	if r.Call == nil {
		r.Call = p.call
	}
	k.admitResult(ctx, p.call, r)
	k.observeCompletedResult(ctx, p.call, r)
	emit(abi.Event{Kind: abi.EvComplete, Call: p.call, Result: r, Fields: costFields(engNs, 0)})
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
//
// The fan-out is FAIL-OPEN (#4266): every observer runs under its OWN recover, so a
// tap that panics can neither kill the syscall it was only watching nor starve the
// observers behind it in the walk. Observers are INSTRUMENTATION — and
// instrumentation is optional in a way the syscall is not; a faulty tap degrades to
// "no telemetry from that tap". The isolation is not silent: each recovered panic is
// counted and its offender recorded (see [ObserverPanics] / [LastObserverPanic]),
// because trading a crash for a mystery is not a fix.
//
// The fan-out is also SELF-HEALING (#5091): an observer that keeps panicking is
// DISARMED after [ObserverDisarmThreshold] recovered panics and never re-entered,
// so a reliably-broken tap stops taxing every subsequent syscall with a recover +
// record-store forever. The disarm is kernel-local — the abi registry snapshot is
// never mutated, so its lock-free read stays uncontended — and it is witnessed
// (see [DisarmedObservers]): a silently-dropped observer would be its own mystery.
// Until the first disarm ever happens the skip check is a single atomic bool load,
// so the happy path stays allocation-free (TestEmitFanoutHappyPathZeroAlloc).
func emit(ev abi.Event) {
	skipDisarmed := observerDisarmedAny.Load()
	for _, e := range abi.EmittersFor(ev.Kind) {
		if skipDisarmed && observerIsDisarmed(e) {
			continue
		}
		emitOne(e, ev)
	}
}

// emitOne delivers one event to one observer, containing any panic to that observer.
// The recover is free when nothing panics — an open-coded defer, no allocation, pinned
// by TestEmitFanoutHappyPathZeroAlloc — so the fail-open guarantee costs the syscall
// path nothing. Recovering HERE (below the kernel's own frames) is also what keeps the
// call-path invariants intact: the kernel's callers never unwind, so a lock they hold
// across an emit is released by their own normal return, not skipped by a panic.
func emitOne(e abi.Emitter, ev abi.Event) {
	defer func() {
		if r := recover(); r != nil {
			recordObserverPanic(e, ev.Kind, r)
		}
	}()
	e.Emit(ev)
}

// ObserverPanic is one isolated observer failure: the offending observer's concrete
// type, the event kind it was observing, and its recovered panic value. It is the
// witness that a fail-open fan-out is not a fail-SILENT one — the record an operator
// reads to name the broken tap.
type ObserverPanic struct {
	Observer string        // the observer's concrete Go type, e.g. "*kpi.Tap"
	Kind     abi.EventKind // the event kind it panicked observing
	Value    string        // the recovered panic value, rendered
}

var (
	observerPanicCount atomic.Int64
	observerPanicLast  atomic.Pointer[ObserverPanic]
)

// recordObserverPanic tallies an isolated observer panic and records its offender. It
// runs ONLY on the panic path, so the strings it renders never cost the happy path.
func recordObserverPanic(e abi.Emitter, kind abi.EventKind, r any) {
	observerPanicCount.Add(1)
	observerPanicLast.Store(&ObserverPanic{
		Observer: fmt.Sprintf("%T", e),
		Kind:     kind,
		Value:    fmt.Sprint(r),
	})
	trackObserverPanicForDisarm(e, kind)
}

// ObserverDisarmThreshold is how many recovered panics one observer may cost before
// the fan-out disarms it (#5091). The count is CUMULATIVE per observer, not
// consecutive: resetting a streak on success would require touching per-observer
// state on the happy path, which the zero-alloc pin forbids — and an
// instrumentation tap that panics this many times is broken either way.
const ObserverDisarmThreshold = 3

// DisarmedObserver is the witness for one disarmed tap (#5091): which observer was
// disarmed, after how many recovered panics, and on which event kind it tripped the
// threshold. The record an operator reads to learn WHY a tap went quiet.
type DisarmedObserver struct {
	Observer string        // the observer's concrete Go type, e.g. "*kpi.Tap"
	Kind     abi.EventKind // the event kind of the panic that tripped the threshold
	Panics   int64         // how many recovered panics it cost before disarm
}

// observerDisarmState is the copy-on-write disarmed set: emitters is what the
// fan-out's skip check compares against (by interface identity), records is the
// parallel witness DisarmedObservers returns. Replaced whole under
// observerDisarmMu, read lock-free.
type observerDisarmState struct {
	emitters []abi.Emitter
	records  []DisarmedObserver
}

var (
	// observerDisarmedAny is the hot-path gate: false until the FIRST observer is
	// ever disarmed, so a fan-out where nothing has ever panicked pays one atomic
	// bool load and no per-observer lookup (the zero-alloc guarantee's condition).
	observerDisarmedAny atomic.Bool
	observerDisarmed    atomic.Pointer[observerDisarmState]

	// observerPanicStreaks counts panics per LIVE (not yet disarmed) observer. It is
	// touched only on the panic path, under observerDisarmMu.
	observerDisarmMu     sync.Mutex
	observerPanicStreaks map[abi.Emitter]int64
)

// trackObserverPanicForDisarm advances an observer's panic count and disarms it at
// the threshold. Panic path only. An observer whose dynamic type is not comparable
// cannot be identity-tracked in a map; it stays armed (counted and recorded by
// recordObserverPanic exactly as before #5091) rather than risking a panic inside
// the recover that is supposed to be containing one.
func trackObserverPanicForDisarm(e abi.Emitter, kind abi.EventKind) {
	if t := reflect.TypeOf(e); t == nil || !t.Comparable() {
		return
	}
	observerDisarmMu.Lock()
	defer observerDisarmMu.Unlock()
	if observerPanicStreaks == nil {
		observerPanicStreaks = map[abi.Emitter]int64{}
	}
	observerPanicStreaks[e]++
	n := observerPanicStreaks[e]
	if n < ObserverDisarmThreshold {
		return
	}
	delete(observerPanicStreaks, e) // disarmed observers are skipped; drop the streak
	next := &observerDisarmState{}
	if cur := observerDisarmed.Load(); cur != nil {
		for _, d := range cur.emitters {
			if d == e {
				return // already disarmed (a racing panic in flight before the skip took)
			}
		}
		next.emitters = append(append([]abi.Emitter{}, cur.emitters...), e)
		next.records = append(append([]DisarmedObserver{}, cur.records...), DisarmedObserver{
			Observer: fmt.Sprintf("%T", e), Kind: kind, Panics: n,
		})
	} else {
		next.emitters = []abi.Emitter{e}
		next.records = []DisarmedObserver{{Observer: fmt.Sprintf("%T", e), Kind: kind, Panics: n}}
	}
	observerDisarmed.Store(next)
	observerDisarmedAny.Store(true)
}

// observerIsDisarmed reports whether the fan-out has disarmed e. Lock-free and
// allocation-free: a linear identity scan of the (tiny — one entry per broken tap
// in the process's lifetime) disarmed set. Every stored emitter has a comparable
// dynamic type, so the interface comparison cannot panic.
func observerIsDisarmed(e abi.Emitter) bool {
	s := observerDisarmed.Load()
	if s == nil {
		return false
	}
	for _, d := range s.emitters {
		if d == e {
			return true
		}
	}
	return false
}

// DisarmedObservers returns the witness records for every observer the fan-out has
// disarmed since process start (#5091), in disarm order. Empty means every
// registered tap is still being entered. The companion to [ObserverPanics]: the
// counter says panics happened, this says which taps the kernel gave up on.
func DisarmedObservers() []DisarmedObserver {
	s := observerDisarmed.Load()
	if s == nil {
		return nil
	}
	out := make([]DisarmedObserver, len(s.records))
	copy(out, s.records)
	return out
}

// ObserverPanics returns how many observer panics the fan-out has isolated since
// process start — the metric that makes a broken tap a visible number rather than
// either a crashed syscall or a silence.
func ObserverPanics() int64 { return observerPanicCount.Load() }

// LastObserverPanic returns the most recently isolated observer panic and true, or a
// zero value and false when no observer has panicked.
func LastObserverPanic() (ObserverPanic, bool) {
	if p := observerPanicLast.Load(); p != nil {
		return *p, true
	}
	return ObserverPanic{}, false
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
// TERMINAL) from the reason's category. MISROUTE, MALFORMED, SHELL_DIALECT, and
// POLICY_BLOCK are call-shape refusals: the refused call stays denied, while the
// next turn may use the sanctioned alternative or continue with unrelated work.
// A policy refusal must not become a session-wide stop merely because one call
// crossed the floor (#5197).
func Disposition(r abi.ReasonCode) string {
	switch r {
	case abi.ReasonMisroute, abi.ReasonMalformed, abi.ReasonShellDialect, abi.ReasonPolicyBlock, abi.ReasonTaintEgress:
		return "RETRYABLE"
	case abi.ReasonRateLimited, abi.ReasonLeaseHeld:
		return "WAIT"
	case abi.ReasonSelfModify, abi.ReasonTrustViolation, abi.ReasonIntegrityRefuted:
		return "ESCALATE"
	default:
		return "TERMINAL"
	}
}

var _ abi.Kernel = (*Kernel)(nil)

// completedResultObserver is an optional post-engine seam implemented by policy
// rungs that need independently observed effect receipts. It is deliberately
// local to kernel: the frozen ABI remains additive-only and old adjudicators are
// unaffected.
type completedResultObserver interface {
	ObserveResult(context.Context, *abi.ToolCall, *abi.Result)
}

func (k *Kernel) observeCompletedResult(ctx context.Context, c *abi.ToolCall, r *abi.Result) {
	if r == nil || r.Status != abi.StatusOK || r.Outcome != abi.OutcomeCommitted {
		return
	}
	for _, adj := range k.adjudicatorsFor(c) {
		if observer, ok := adj.(completedResultObserver); ok {
			observer.ObserveResult(ctx, c, r)
		}
	}
}
