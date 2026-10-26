package gateway

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/kvbudget"
	"github.com/anthony-chaudhary/fak/pkg/turncost"
)

// bindWarmupStreams uses the live admission bound, never a reporter-supplied
// stream count. A missing or unbounded controller cannot reserve fixed state.
func bindWarmupStreams(cap kvbudget.WarmupCapacity, ctl *AdmissionController) kvbudget.WarmupCapacity {
	if cap.FixedBytesPerStream > 0 {
		cap.MaxConcurrentStreams = 0
		if ctl != nil {
			cap.MaxConcurrentStreams = int64(ctl.Policy().MaxNumSeqs)
		}
	}
	return cap
}

func kvStatsWarmupCapacity(st agent.KVMemoryStats, ctl *AdmissionController) (kvbudget.WarmupCapacity, bool) {
	usable := st.FitBudgetBytes
	if usable <= 0 {
		usable = st.CapacityFreeBytes
	}
	if usable <= 0 {
		usable = st.CapacityTotalBytes
	}
	cap := bindWarmupStreams(kvbudget.WarmupCapacity{
		UsableBytes: usable, BytesPerToken: st.BytesPerToken,
		FixedBytesPerStream: st.FixedBytesPerStream,
	}, ctl)
	// Preserve legacy missing-measurement behavior for uniform layouts. Known
	// fixed state must instead reach the derive, including its refusal reasons.
	return cap, st.FixedBytesPerStream > 0 || (st.BytesPerToken > 0 && usable > 0)
}

type fixedWarmupCapacityError struct{ reason kvbudget.Reason }

func (e *fixedWarmupCapacityError) Error() string {
	return "compressed KV warmup capacity unavailable: " + string(e.reason)
}

func (s *Server) fixedWarmupAdmissionError() error {
	s.admissionMu.RLock()
	cap, ctl, fraction := s.warmupCapacity, s.admissionCtl, s.warmupReserveFraction
	s.admissionMu.RUnlock()
	return fixedWarmupCapacityAdmissionError(cap, ctl, fraction)
}

func fixedWarmupCapacityAdmissionError(cap *kvbudget.WarmupCapacity, ctl *AdmissionController, fraction float64) error {
	if cap != nil && cap.FixedBytesPerStream > 0 {
		if fraction <= 0 {
			fraction = kvbudget.DefaultReserveFraction
		}
		if derived := bindWarmupStreams(*cap, ctl).DeriveTokenBudget(fraction); !derived.Derived() {
			return &fixedWarmupCapacityError{reason: derived.Reason}
		}
	}
	return nil
}

// SetWarmupCapacity records a byte-measured warmup probe.
func (s *Server) SetWarmupCapacity(cap kvbudget.WarmupCapacity) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	s.warmupCapacity = &cap
	ctl := s.admissionCtl
	fraction := s.warmupReserveFraction
	s.admissionMu.Unlock()

	if ctl != nil && ctl.TokenBudgetProvenance() != "explicit" {
		if fraction <= 0 {
			fraction = kvbudget.DefaultReserveFraction
		}
		derived := bindWarmupStreams(cap, ctl).DeriveTokenBudget(fraction)
		if derived.Derived() {
			ctl.SetTokenBudgetWithProvenance(int(derived.TokenBudget), "measured")
		}
	}
}

// SetWarmupBlockCapacity records a block-measured warmup probe.
func (s *Server) SetWarmupBlockCapacity(cap kvbudget.WarmupBlockCapacity) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	s.warmupBlockCapacity = &cap
	ctl := s.admissionCtl
	fraction := s.warmupReserveFraction
	fixedCap := s.warmupCapacity
	s.admissionMu.Unlock()

	// A block-only measurement cannot replace a known compressed fixed charge.
	if fixedCap != nil && fixedCap.FixedBytesPerStream > 0 {
		return
	}
	if ctl != nil && ctl.TokenBudgetProvenance() != "explicit" {
		if fraction <= 0 {
			fraction = kvbudget.DefaultReserveFraction
		}
		derived := cap.DeriveTokenBudget(fraction)
		if derived.Derived() {
			ctl.SetTokenBudgetWithProvenance(int(derived.TokenBudget), "measured")
		}
	}
}

// SetWarmupReserveFraction configures the reserve fraction withheld from measured capacity.
func (s *Server) SetWarmupReserveFraction(fraction float64) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	s.warmupReserveFraction = fraction
}

// WarmupCapacity returns the measured warmup capacity and true if available.
func (s *Server) WarmupCapacity() (kvbudget.WarmupCapacity, bool) {
	if s == nil {
		return kvbudget.WarmupCapacity{}, false
	}
	s.admissionMu.RLock()
	defer s.admissionMu.RUnlock()
	if s.warmupCapacity != nil {
		return bindWarmupStreams(*s.warmupCapacity, s.admissionCtl), true
	}
	if s.planner != nil {
		if r, ok := s.planner.(WarmupCapacityReporter); ok {
			cap, found := r.WarmupCapacity()
			return bindWarmupStreams(cap, s.admissionCtl), found
		}
		if r, ok := s.planner.(interface {
			WarmupCapacity() kvbudget.WarmupCapacity
		}); ok {
			return bindWarmupStreams(r.WarmupCapacity(), s.admissionCtl), true
		}
		if r, ok := s.planner.(agent.KVMemoryReporter); ok {
			return kvStatsWarmupCapacity(r.KVMemoryStats(), s.admissionCtl)
		}
	}
	return kvbudget.WarmupCapacity{}, false
}

// WarmupDerivedBudget calculates the token budget derived from measured warmup capacity.
func (s *Server) WarmupDerivedBudget() (kvbudget.DerivedBudget, bool) {
	if s == nil {
		return kvbudget.DerivedBudget{}, false
	}
	s.admissionMu.RLock()
	cap := s.warmupCapacity
	blockCap := s.warmupBlockCapacity
	fraction := s.warmupReserveFraction
	p := s.planner
	ctl := s.admissionCtl
	s.admissionMu.RUnlock()

	if fraction <= 0 {
		fraction = kvbudget.DefaultReserveFraction
	}

	if cap != nil {
		derived := bindWarmupStreams(*cap, ctl).DeriveTokenBudget(fraction)
		if derived.Derived() || cap.FixedBytesPerStream > 0 {
			return derived, derived.Derived()
		}
	}
	if blockCap != nil {
		derived := blockCap.DeriveTokenBudget(fraction)
		if derived.Derived() {
			return derived, true
		}
	}
	if p != nil {
		if r, ok := p.(WarmupCapacityReporter); ok {
			if c, okCap := r.WarmupCapacity(); okCap {
				derived := bindWarmupStreams(c, ctl).DeriveTokenBudget(fraction)
				if derived.Derived() || c.FixedBytesPerStream > 0 {
					return derived, derived.Derived()
				}
			}
		} else if r, ok := p.(interface {
			WarmupCapacity() kvbudget.WarmupCapacity
		}); ok {
			c := r.WarmupCapacity()
			derived := bindWarmupStreams(c, ctl).DeriveTokenBudget(fraction)
			if derived.Derived() || c.FixedBytesPerStream > 0 {
				return derived, derived.Derived()
			}
		}
		if r, ok := p.(interface {
			WarmupBlockCapacity() (kvbudget.WarmupBlockCapacity, bool)
		}); ok {
			if bc, okBC := r.WarmupBlockCapacity(); okBC {
				derived := bc.DeriveTokenBudget(fraction)
				if derived.Derived() {
					return derived, true
				}
			}
		} else if r, ok := p.(interface {
			WarmupBlockCapacity() kvbudget.WarmupBlockCapacity
		}); ok {
			bc := r.WarmupBlockCapacity()
			derived := bc.DeriveTokenBudget(fraction)
			if derived.Derived() {
				return derived, true
			}
		}
		if r, ok := p.(agent.KVMemoryReporter); ok {
			if c, found := kvStatsWarmupCapacity(r.KVMemoryStats(), ctl); found {
				derived := c.DeriveTokenBudget(fraction)
				if derived.Derived() || c.FixedBytesPerStream > 0 {
					return derived, derived.Derived()
				}
			}
		}
	}
	return kvbudget.DerivedBudget{}, false
}

// SetMaxTotalTokens sets the configured upper bound on total tokens.
func (s *Server) SetMaxTotalTokens(n int) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()
	s.maxTotalTokens = n
}

// CheckMaxTotalTokens validates that n does not exceed the admission controller's token budget.
func (s *Server) CheckMaxTotalTokens(n int) error {
	if s == nil {
		return nil
	}
	s.admissionMu.RLock()
	ctl := s.admissionCtl
	s.admissionMu.RUnlock()
	if n > 0 && ctl != nil {
		budget := ctl.Policy().TokenBudget
		if n > budget {
			return fmt.Errorf("max_total_tokens (%d) exceeds admission token budget (%d); decrease --max-batch-prefill-tokens or --max-total-tokens", n, budget)
		}
	}
	return nil
}

// SetAdmissionController wires the native serving admission gate (#35) onto the Server so its
// L2 serving-metrics fragment (fak_sched_*) renders into the live /metrics surface. The host
// calls this once the native iteration scheduler (modelengine.NativeScheduler) is on the serve
// loop; passing nil detaches it (the surface goes inert — no fak_sched_* series). Settable
// after New, mirroring SetFleetMembership / SetKVResidencyReclaimer. A nil receiver is a no-op.
func (s *Server) SetAdmissionController(c *AdmissionController) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	s.admissionCtl = c
	s.admissionMu.Unlock()
	if c != nil {
		// Retain declared fixed state even before RunWarmup. Both initialization
		// orders must enforce a refused derive on the served path, including when
		// an explicit token budget would otherwise skip measured derivation.
		if cap, found := s.WarmupCapacity(); found && cap.FixedBytesPerStream > 0 {
			s.SetWarmupCapacity(cap)
		}
		if s.table != nil {
			c.SetTable(s.table)
		}
		if s.scheduler != nil {
			c.SetSequencer(s.scheduler)
		}
		if s.pool != nil {
			c.SetFleet(s.pool)
		}
		if c.Policy().TokenBudgetProvenance != "explicit" {
			if derived, ok := s.WarmupDerivedBudget(); ok && derived.Derived() {
				c.SetTokenBudgetWithProvenance(int(derived.TokenBudget), "measured")
			}
		}
		s.admissionMu.RLock()
		maxTokens := s.maxTotalTokens
		s.admissionMu.RUnlock()
		if maxTokens > 0 {
			if err := s.CheckMaxTotalTokens(maxTokens); err != nil && s.logf != nil {
				s.logf("gateway: %v", err)
			}
		}
	}
}

// writeAdmissionMetrics folds the wired admission gate's L2 serving-metrics fragment
// (fak_sched_running/waiting/admitted/...) onto the gateway /metrics surface (#35). A Server
// with no controller attached emits nothing — no phantom zero series — the same host-injected,
// inert-by-default posture as writeFleetMembershipMetrics and the KV-residency seams.
func (s *Server) writeAdmissionMetrics(b *strings.Builder) {
	if s == nil || b == nil {
		return
	}
	s.admissionMu.RLock()
	c := s.admissionCtl
	s.admissionMu.RUnlock()
	if c == nil {
		return
	}
	c.WriteMetrics(b)
}

// admitStreamedTurn is beginServedAdmission plus the refusal both streaming surfaces own
// identically: a refused admission is logged with the surface's own `lane` word, answered on w as
// an upstream error, and reported ok=false — the caller's cue to stop and return "response
// owned". On ok=true the caller MUST defer the returned lease's Release, exactly as it would with
// beginServedAdmission directly; the lease is nil-safe when no admission control is attached.
func (s *Server) admitStreamedTurn(ctx context.Context, w http.ResponseWriter, lane string, turn servedSessionTurn, messages []agent.Message, tools []agent.ToolDef, maxTokens int) (*AdmissionLease, bool) {
	lease, err := s.beginServedAdmission(ctx, turn, messages, tools, maxTokens)
	if err != nil {
		s.logf("gateway: scheduler admission refused (%s): %v", lane, err)
		s.writeUpstreamErr(w, err)
		return nil, false
	}
	return lease, true
}

// beginServedAdmission is the served request's admission boundary, narrowest budget last:
// the scheduler slot (admissionCtl, #35) first — a queued request must not hold token
// budget while it waits — then the caller's OWN per-principal allotment
// (principalTokenRates, #5379), then the shared provider token window (tokenRateGate,
// #2019). A shed at any stage frees everything the earlier stages granted. With no gate
// attached it is inert (nil lease, nil error) and the request path is byte-for-byte
// historical.
//
// The per-principal allotment is checked BEFORE the shared provider window on purpose: a
// tenant already over its own cap must be shed without first reserving — and briefly
// holding — a slice of the budget every other tenant is competing for. The reverse order
// would let a tenant that is going to be refused anyway still push the shared window
// toward saturation.
func (s *Server) beginServedAdmission(ctx context.Context, turn servedSessionTurn, messages []agent.Message, tools []agent.ToolDef, maxTokens int, queuedObserver ...func()) (*AdmissionLease, error) {
	if s == nil {
		return nil, nil
	}
	var onQueued func()
	if len(queuedObserver) > 0 {
		onQueued = queuedObserver[0]
	}
	s.admissionMu.RLock()
	c := s.admissionCtl
	g := s.tokenRateGate
	b := s.principalTokenRates
	cap, fraction := s.warmupCapacity, s.warmupReserveFraction
	s.admissionMu.RUnlock()
	// Zero disables the controller's token limit, so a refused fixed-state
	// derive must stop here rather than install zero or retain a default budget.
	// Validate the SAME captured controller used for Acquire: a concurrent
	// replacement must not check one stream bound then acquire under another.
	if err := fixedWarmupCapacityAdmissionError(cap, c, fraction); err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, err
	}
	if c == nil && g == nil && b == nil {
		return nil, nil
	}
	preallocCeiling := c.preallocCeiling()
	var lease *AdmissionLease
	if c != nil {
		var err error
		timePhase(turn.turnCost, turncost.PhaseAdmission, func() {
			req := SeqRequest{
				TraceID:  turn.traceID,
				Priority: turn.state.Priority,
				Tokens:   estimateServedAdmissionTokensWithCap(messages, tools, maxTokens, preallocCeiling),
			}
			lease, err = acquireObserved(ctx, c, req, onQueued)
		})
		if err != nil {
			return nil, err
		}
	}
	estimate := estimateServedTokenUsageWithCap(messages, tools, maxTokens, preallocCeiling)
	var res *TokenReservation
	if b != nil {
		var err error
		// principalFromContext is "" for an unidentified caller; the book charges those to
		// its single shared allotment rather than to a fresh (bypassable) one.
		res, err = b.Admit(principalFromContext(ctx), estimate)
		if err != nil {
			lease.Release() // nil-safe: free the scheduler slot this tenant's cap refused to feed
			return nil, err
		}
	}
	if g != nil {
		shared, err := g.Admit(estimate)
		if err != nil {
			// The provider window refused, so the provider is never called: cancel (charge
			// nothing) rather than Release (charge the estimate) the allotment this tenant
			// held for one instant — a neighbour's saturation must not spend its budget.
			res.cancel()
			lease.Release()
			return nil, err
		}
		shared.linked = res // settle/release the allotment in lockstep with the shared window
		res = shared
	}
	if res != nil {
		if lease == nil {
			lease = &AdmissionLease{}
		}
		lease.tokenRes = res
	}
	return lease, nil
}

type admissionAcquireResult struct {
	lease *AdmissionLease
	err   error
}

func acquireObserved(ctx context.Context, c *AdmissionController, req SeqRequest, onQueued func()) (*AdmissionLease, error) {
	if onQueued == nil {
		return c.Acquire(ctx, req)
	}
	result := make(chan admissionAcquireResult, 1)
	go func() {
		lease, err := c.Acquire(ctx, req)
		result <- admissionAcquireResult{lease: lease, err: err}
	}()
	ticker := time.NewTicker(5 * time.Millisecond)
	defer ticker.Stop()
	queued := false
	for {
		select {
		case got := <-result:
			return got.lease, got.err
		case <-ticker.C:
			if queued {
				continue
			}
			receipt, ok := c.LastAdmissionDecision(req.TraceID)
			if ok && receipt.Verdict == VerdictQueued.String() {
				queued = true
				onQueued()
			}
		case <-ctx.Done():
			got := <-result
			return got.lease, got.err
		}
	}
}

// servedPromptChars is the raw character volume of one served turn's prompt side —
// messages plus tool schemas — shared by the scheduler gate's single-axis estimate and
// the token-rate gate's input/output split.
func servedPromptChars(messages []agent.Message, tools []agent.ToolDef) int {
	chars := 0
	for _, m := range messages {
		chars += len(m.Role) + len(m.Content) + len(m.ToolCallID) + len(m.Name)
		if m.FunctionCall != nil {
			chars += len(m.FunctionCall.Name) + len(m.FunctionCall.Arguments)
		}
		for _, tc := range m.ToolCalls {
			chars += len(tc.ID) + len(tc.Type) + len(tc.Function.Name) + len(tc.Function.Arguments)
		}
	}
	for _, t := range tools {
		chars += len(t.Type) + len(t.Function.Name) + len(t.Function.Description) + len(t.Function.Parameters)
	}
	return chars
}

func estimateServedAdmissionTokens(messages []agent.Message, tools []agent.ToolDef, maxTokens int) int {
	return estimateServedAdmissionTokensWithCap(messages, tools, maxTokens, DefaultPreallocCeiling)
}

func estimateServedAdmissionTokensWithCap(messages []agent.Message, tools []agent.ToolDef, maxTokens int, preallocCeiling int) int {
	chars := servedPromptChars(messages, tools)
	tokens := chars / 4
	if chars > 0 && tokens == 0 {
		tokens = 1
	}
	if preallocCeiling <= 0 {
		preallocCeiling = DefaultPreallocCeiling
	}
	if maxTokens > 0 {
		gen := maxTokens
		if gen > preallocCeiling {
			gen = preallocCeiling
		}
		tokens += gen
	} else {
		tokens++
	}
	if tokens <= 0 {
		return 1
	}
	return tokens
}

func sampleMaxTokens(opts []agent.SampleOpt) int {
	var sp agent.SampleParams
	for _, opt := range opts {
		if opt != nil {
			opt(&sp)
		}
	}
	if sp.MaxTokens == nil {
		return 0
	}
	return *sp.MaxTokens
}

func admissionErrorStatus(err error) (status int, code, msg string, ok bool) {
	var capacityErr *fixedWarmupCapacityError
	if errors.As(err, &capacityErr) {
		return http.StatusServiceUnavailable, "warmup_capacity_unavailable", capacityErr.Error(), true
	}
	var ae *AdmissionError
	if !errors.As(err, &ae) {
		return 0, "", "", false
	}
	switch ae.Verdict {
	case VerdictRefused:
		reason := strings.TrimSpace(ae.Reason)
		if reason == "" {
			reason = "request envelope exceeds capacity"
		}
		return http.StatusBadRequest, "context_length_exceeded", reason, true
	case VerdictShed:
		msg := "scheduler overloaded — back off and retry"
		// A token-rate shed (#2019) names the provider cap that fired so the client's
		// backoff can be informed; the historical slot shed carries no reason.
		if reason := strings.TrimSpace(ae.Reason); reason != "" {
			msg += ": " + reason
		}
		return http.StatusTooManyRequests, "scheduler_overloaded", msg, true
	case VerdictDenied:
		reason := strings.TrimSpace(ae.Reason)
		if reason == "" {
			reason = "trust verdict denied admission"
		}
		return http.StatusForbidden, "scheduler_admission_denied",
			"scheduler admission denied: " + reason, true
	default:
		return http.StatusServiceUnavailable, "scheduler_unavailable",
			"scheduler admission refused", true
	}
}
