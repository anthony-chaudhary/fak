package agent

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_moe_residency.go — the SERVE half of R6 (#5617, epic #5606): carry activated-expert
// residency out of a request and into something an operator can read.
//
// The gap this closes is structural, not cosmetic. model.Session.MoEResidency reports one session's
// ring, and on the device path generateReusedContextWithBias builds a session per request
// (p.m.NewBackendSession) and `defer s.Close()`s it. So the entire ledger the ladder spent seven
// rungs building — page-ins, hits, evictions, page-in bytes, coverage, placement drift — is
// constructed, filled, and destroyed inside one request, and no serve surface ever sees a byte of
// it. An operator sizing --n-cpu-moe against a live workload had to infer the ring's behavior from
// tokens per second.
//
// noteMoEResidency folds each finished request's report into a planner-scoped ledger just before
// that session's teardown, following the same shape RequestMemoryStats already uses for per-request
// memory: a small mutex, cumulative counters, and the last full report kept whole so a reader can
// see one request in detail rather than only aggregates.
//
// What the aggregate deliberately does NOT do is average rates. Averaging per-request hit rates
// would weight a two-token request equally with a two-thousand-token one; summing the raw counters
// and dividing once at read time is the number an operator actually tunes against. Every derived
// rate here is therefore computed from the sums, and each guards its denominator so a serve that
// never engaged a ring reports 0 rather than NaN in its telemetry JSON.
//
// Reading PageInBytes/Tokens against a per-request session scope. Because each request gets a fresh
// ring, an expert paged in for request N is gone by request N+1, and the bytes-per-token this
// ledger reports is therefore the COLD-start cost repeated every turn — not the steady-state cost a
// long-lived ring would show. That is a real property of the current serve wiring and the number is
// the honest measure of it, which is precisely why it is worth surfacing: it is the evidence for
// giving the planner a model-scoped ring (the SharedExpertRing R7/#5618 already built at
// (model, device) scope) instead of a guess that one would help.

// MoEResidencyLedger is a serve's activated-expert residency across every request that engaged a
// routed-expert ring. Requests==0 means no request ever engaged one — either no operator declared a
// budget (the default) or the model has no routed experts — and every other field is then 0.
type MoEResidencyLedger struct {
	// Requests is how many completed requests contributed, and Tokens how many tokens those
	// requests actually forwarded through the model (prompt tokens not served from the prefix cache,
	// plus generated ones). Tokens is the denominator of the byte rates, so it counts FORWARDED
	// tokens rather than prompt length: a token served from the radix cache activated no expert and
	// would make the ring look cheaper than it is.
	Requests int64 `json:"requests"`
	Tokens   int64 `json:"tokens"`
	// The ring ledger summed across requests. Lookups is every staging request the rings received;
	// Hits resident reuses; PageIns cold uploads; Evictions page-outs; Refusals stagings no budget
	// could admit, which fall back to permanent unbounded residency and are the first sign a budget
	// is being abandoned rather than enforced.
	Lookups   int64 `json:"lookups"`
	Hits      int64 `json:"hits"`
	PageIns   int64 `json:"page_ins"`
	Evictions int64 `json:"evictions"`
	Refusals  int64 `json:"refusals"`
	// PageInBytes is the device bytes cold uploads actually moved — the numerator of the
	// bytes-per-token an operator sizes a budget against, and not recoverable from PageIns because
	// routed projections differ in size and quantization.
	PageInBytes int64 `json:"page_in_bytes"`
	// BudgetBytes is the declared ceiling most recently observed and PeakBytes the high-water
	// footprint across every request. PeakBytes <= BudgetBytes is the boundedness claim, held here
	// across requests rather than only within one.
	BudgetBytes int64 `json:"budget_bytes"`
	PeakBytes   int64 `json:"peak_bytes"`
	// ReconciliationFailures counts requests whose own report failed its internal identity checks
	// (hits+page-ins+refusals == lookups, resident within budget, and the cross-agent pairs under a
	// shared ring). It should be 0 forever. A non-zero value means the ring's accounting disagreed
	// with itself, so every number above it is suspect — which is worth an alarm rather than a
	// silently wrong dashboard, and is the reason the per-request report computes those checks from
	// independent increments instead of restating one.
	ReconciliationFailures int64 `json:"reconciliation_failures"`
	// Last is the most recent request's full report, kept whole. The aggregate answers "what is this
	// serve costing"; Last answers "what did one request actually do", including the placement basis
	// and drift, which do not sum across requests in any meaningful way.
	Last model.MoEResidencyReport `json:"last,omitempty"`
}

// HitRate is Hits/(Hits+PageIns) over the whole serve — the activated-set hit rate, weighted by
// staging volume rather than by request count. It answers 0 when no staging ever happened, which
// reads as "not measured" and not as "everything missed".
func (l MoEResidencyLedger) HitRate() float64 {
	staged := l.Hits + l.PageIns
	if staged <= 0 {
		return 0
	}
	return float64(l.Hits) / float64(staged)
}

// RefusalRate is Refusals/Lookups. Any non-zero value is a sizing bug, not a tuning dial: a refused
// staging did not fail the forward, it silently promoted that weight to permanent residency, so the
// budget the operator declared stopped being the bound.
func (l MoEResidencyLedger) RefusalRate() float64 {
	if l.Lookups <= 0 {
		return 0
	}
	return float64(l.Refusals) / float64(l.Lookups)
}

// ExpertBytesPerToken is the device bytes each forwarded token cost in expert page-ins — the number
// --n-cpu-moe is actually sized against, and the one that falls when residency is working.
func (l MoEResidencyLedger) ExpertBytesPerToken() float64 {
	if l.Tokens <= 0 {
		return 0
	}
	return float64(l.PageInBytes) / float64(l.Tokens)
}

// PeakBudgetUsed is PeakBytes/BudgetBytes. Well under 1 across a real workload means the budget is
// larger than the activated working set needs and the difference could be given back to KV.
func (l MoEResidencyLedger) PeakBudgetUsed() float64 {
	if l.BudgetBytes <= 0 {
		return 0
	}
	return float64(l.PeakBytes) / float64(l.BudgetBytes)
}

// noteMoEResidency folds one finished request's residency into the planner ledger. It must be
// called while the session is still alive — on the device path the caller's `defer s.Close()` takes
// the ring with it — and it is a no-op for a session that never engaged one, which is the default
// and every serve whose operator declared no expert budget. So a serve that is not using the ladder
// pays one nil check and one struct read per request.
//
// tokens is the count actually FORWARDED, which the caller computes; a request that served its
// whole prompt from the prefix cache and generated nothing contributes a ring read and no tokens,
// and must not be allowed to divide by zero downstream.
func (p *InKernelPlanner) noteMoEResidency(s *model.Session, tokens int64) {
	if p == nil || s == nil {
		return
	}
	p.foldMoEResidency(s.MoEResidency(model.MoEResidencyOptions{Tokens: tokens}), tokens)
}

// foldMoEResidency is the accumulation itself, split from the session read so the arithmetic is
// witnessable against hand-built reports — a ledger that summed wrong would otherwise only be
// catchable by driving a real ring and reasoning backwards from the total.
func (p *InKernelPlanner) foldMoEResidency(rep model.MoEResidencyReport, tokens int64) {
	if !rep.Ring.Enabled {
		return // no ring on this session: nothing was bounded, so there is nothing to report
	}
	if tokens < 0 {
		tokens = 0
	}
	p.moeMu.Lock()
	defer p.moeMu.Unlock()
	l := &p.moeResidency
	l.Requests++
	l.Tokens += tokens
	l.Lookups += int64(rep.Ring.Lookups)
	l.Hits += int64(rep.Ring.Hits)
	l.PageIns += int64(rep.Ring.PageIns)
	l.Evictions += int64(rep.Ring.Evictions)
	l.Refusals += int64(rep.Ring.Refusals)
	l.PageInBytes += rep.Ring.PageInBytes
	l.BudgetBytes = rep.Ring.BudgetBytes
	if rep.Ring.PeakBytes > l.PeakBytes {
		l.PeakBytes = rep.Ring.PeakBytes
	}
	if !rep.Reconciliation.OK {
		l.ReconciliationFailures++
	}
	l.Last = rep
}

// MoEResidencyStats returns the serve's activated-expert residency. It is the accessor a telemetry
// surface or a `fak` verb reads; the zero value (Requests==0) is the honest answer for a serve that
// never engaged a ring, and callers should render that as "not engaged" rather than as all-zero
// metrics that look like a ring doing nothing.
func (p *InKernelPlanner) MoEResidencyStats() MoEResidencyLedger {
	if p == nil {
		return MoEResidencyLedger{}
	}
	p.moeMu.Lock()
	defer p.moeMu.Unlock()
	return p.moeResidency
}

// moeResidencyState is the planner-side storage, embedded rather than declared inline in the
// InKernelPlanner struct so this rung adds one field to that already-long definition.
type moeResidencyState struct {
	moeMu        sync.Mutex
	moeResidency MoEResidencyLedger
}
