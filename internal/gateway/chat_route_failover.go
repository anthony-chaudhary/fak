package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

const defaultRouteCircuitCooldown = 30 * time.Second

type accountCircuitState struct {
	failures       int
	cooldownUntil  time.Time
	lastErr        string
	totalAttempts  int64
	totalFailovers int64
	totalSuccesses int64
}

type accountCircuitSnapshot struct {
	accountCircuitState
	cooldownRemaining time.Duration
}

type routeCircuitSnapshot struct {
	accounts       map[string]accountCircuitSnapshot
	totalAttempts  int64
	totalSuccesses int64
	totalFailovers int64
}

// routeCircuitTracker tracks per-account health, cooldown windows, and failover
// counters across routed chat requests.
type routeCircuitTracker struct {
	mu       sync.Mutex
	cooldown time.Duration
	now      func() time.Time
	accounts map[string]*accountCircuitState
}

func newRouteCircuitTracker(cooldown time.Duration, now func() time.Time) *routeCircuitTracker {
	if cooldown <= 0 {
		cooldown = defaultRouteCircuitCooldown
	}
	if now == nil {
		now = time.Now
	}
	return &routeCircuitTracker{
		cooldown: cooldown,
		now:      now,
		accounts: make(map[string]*accountCircuitState),
	}
}

func (s *Server) circuitTracker() *routeCircuitTracker {
	if s == nil {
		return nil
	}
	s.routeCircuitOnce.Do(func() {
		if s.routeCircuit == nil {
			s.routeCircuit = newRouteCircuitTracker(defaultRouteCircuitCooldown, nil)
		}
	})
	return s.routeCircuit
}

func (t *routeCircuitTracker) currentTime() time.Time {
	if t != nil && t.now != nil {
		return t.now()
	}
	return time.Now()
}

func (t *routeCircuitTracker) ensureAccountLocked(accountID string) *accountCircuitState {
	st, ok := t.accounts[accountID]
	if !ok {
		st = &accountCircuitState{}
		t.accounts[accountID] = st
	}
	return st
}

func (t *routeCircuitTracker) isCoolingDown(accountID string) (bool, time.Duration) {
	if t == nil {
		return false, 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st, ok := t.accounts[accountID]
	if !ok || st.cooldownUntil.IsZero() {
		return false, 0
	}
	rem := st.cooldownUntil.Sub(t.currentTime())
	if rem <= 0 {
		return false, 0
	}
	return true, rem
}

func (t *routeCircuitTracker) recordSuccess(accountID string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st := t.ensureAccountLocked(accountID)
	st.totalAttempts++
	st.totalSuccesses++
	st.failures = 0
	st.cooldownUntil = time.Time{}
	st.lastErr = ""
}

func (t *routeCircuitTracker) recordFailure(accountID string, err error) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st := t.ensureAccountLocked(accountID)
	st.totalAttempts++
	st.failures++
	st.cooldownUntil = t.currentTime().Add(t.cooldown)
	if err != nil {
		st.lastErr = err.Error()
	}
}

func (t *routeCircuitTracker) recordFailover(fromAccount, toAccount string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	st := t.ensureAccountLocked(fromAccount)
	st.totalFailovers++
}

func (t *routeCircuitTracker) snapshot() routeCircuitSnapshot {
	out := routeCircuitSnapshot{
		accounts: make(map[string]accountCircuitSnapshot),
	}
	if t == nil {
		return out
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	now := t.currentTime()
	for id, st := range t.accounts {
		var rem time.Duration
		if !st.cooldownUntil.IsZero() {
			if d := st.cooldownUntil.Sub(now); d > 0 {
				rem = d
			}
		}
		out.accounts[id] = accountCircuitSnapshot{
			accountCircuitState: *st,
			cooldownRemaining:   rem,
		}
		out.totalAttempts += st.totalAttempts
		out.totalSuccesses += st.totalSuccesses
		out.totalFailovers += st.totalFailovers
	}
	return out
}

// isFailoverEligibleError reports whether an upstream error is transient and
// eligible for account failover. Client errors (400, 401, 403, 404, 422) and
// caller-canceled contexts never fail over.
func isFailoverEligibleError(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var se *agent.UpstreamStatusError
	if errors.As(err, &se) {
		switch se.Status {
		case 400, 401, 403, 404, 422:
			return false
		case 408, 429, 500, 502, 503, 504, 529:
			return true
		default:
			return se.Status >= 500
		}
	}
	var ue *agent.UpstreamUnreachableError
	if errors.As(err, &ue) {
		return true
	}
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
		return true
	}
	if errors.Is(err, syscall.ECONNREFUSED) || errors.Is(err, syscall.ECONNRESET) {
		return true
	}
	var ne net.Error
	return errors.As(err, &ne)
}

// isContextOverflowError reports whether err indicates the upstream rejected
// the request because the prompt exceeded its context window, returning any
// upstream-reported prompt token count when available.
func isContextOverflowError(err error) (bool, int) {
	if err == nil {
		return false, 0
	}
	if errors.Is(err, agent.ErrContextOverflow) {
		return true, 0
	}
	var ikce *agent.InKernelContextLengthError
	if errors.As(err, &ikce) {
		return true, ikce.PromptTokens
	}
	var coe *agent.ContextOverflowError
	if errors.As(err, &coe) {
		return true, 0
	}
	var use *agent.UpstreamStreamError
	if errors.As(err, &use) {
		if upstreamContextOverflowCodes[strings.ToLower(strings.TrimSpace(use.Code))] ||
			isUpstreamContextOverflow(use.Message) {
			return true, 0
		}
	}
	var se *agent.UpstreamStatusError
	if errors.As(err, &se) {
		d := parseUpstream400(se.Body)
		if d.ContextOverflow {
			return true, d.PromptTokens
		}
	}
	return false, 0
}

func contextRequiredTokens(currentMin, reportedPrompt, failedWindow int) int {
	req := currentMin
	if reportedPrompt > req {
		req = reportedPrompt
	}
	if failedWindow > 0 && failedWindow+1 > req {
		req = failedWindow + 1
	}
	return req
}

func estimatePromptTokens(messages []agent.Message, tools []agent.ToolDef) int {
	return int(estimateServedTokenUsage(messages, tools, 0).InputTokens)
}

type failoverCandidate struct {
	target   modelroute.Target
	planner  agent.Planner
	locality servingLocality
}

// failoverPlanner wraps an ordered list of admitted account planners for one
// routed model, skipping cooling-down accounts first and enforcing residency
// floors fail-closed before every dispatch.
type failoverPlanner struct {
	requestedModel string
	residencyFloor string
	candidates     []failoverCandidate
	circuit        *routeCircuitTracker
}

var (
	_ agent.Planner          = (*failoverPlanner)(nil)
	_ agent.StreamingPlanner = (*failoverPlanner)(nil)
)

func (p *failoverPlanner) Model() string {
	if p.requestedModel != "" {
		return p.requestedModel
	}
	if len(p.candidates) > 0 && p.candidates[0].planner != nil {
		return p.candidates[0].planner.Model()
	}
	return ""
}

func (p *failoverPlanner) orderedCandidates() []failoverCandidate {
	if len(p.candidates) <= 1 || p.circuit == nil {
		return p.candidates
	}
	healthy := make([]failoverCandidate, 0, len(p.candidates))
	cooling := make([]failoverCandidate, 0, len(p.candidates))
	for _, cand := range p.candidates {
		if down, _ := p.circuit.isCoolingDown(cand.target.Account); down {
			cooling = append(cooling, cand)
		} else {
			healthy = append(healthy, cand)
		}
	}
	return append(healthy, cooling...)
}

func (p *failoverPlanner) checkResidency(cand failoverCandidate) error {
	if p.residencyFloor == "local_only" && !cand.target.Zone().SelfHosted() {
		return fmt.Errorf("RESIDENCY_FAILOVER_DENIED: model %q has residency_floor=local_only; refusing failover to non-self-hosted account %q (kind=%s)",
			p.requestedModel, cand.target.Account, cand.target.Kind)
	}
	return nil
}

func candidateSampleOpts(cand failoverCandidate, opts []agent.SampleOpt, singleAttempt bool) []agent.SampleOpt {
	out := make([]agent.SampleOpt, 0, len(opts)+1)
	out = append(out, opts...)
	out = append(out, func(params *agent.SampleParams) {
		params.Model = cand.target.UpstreamModel
		params.UpstreamAPIKey = ""
		params.RawRequestBody = nil
		params.UpstreamBeta = ""
		if singleAttempt {
			params.MaxAttempts = 1
		}
	})
	return out
}

func (p *failoverPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	ordered := p.orderedCandidates()
	if len(ordered) == 0 {
		return nil, fmt.Errorf("gateway: chat route: no admitted candidates for model %q", p.requestedModel)
	}
	minContextTokens := estimatePromptTokens(messages, tools)
	var lastErr error
	var overflowFrom string
	for i, cand := range ordered {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		if minContextTokens > 0 && !cand.target.FitsContext(minContextTokens) {
			if overflowFrom == "" {
				overflowFrom = cand.target.Account
			}
			continue
		}
		if err := p.checkResidency(cand); err != nil {
			if lastErr != nil {
				return nil, fmt.Errorf("%w (after upstream error: %v)", err, lastErr)
			}
			return nil, err
		}
		if overflowFrom != "" {
			if p.circuit != nil {
				p.circuit.recordFailover(overflowFrom, cand.target.Account)
			}
			overflowFrom = ""
		}
		comp, err := cand.planner.Complete(ctx, messages, tools, candidateSampleOpts(cand, opts, i+1 < len(ordered))...)
		if err == nil {
			if p.circuit != nil {
				p.circuit.recordSuccess(cand.target.Account)
			}
			return comp, nil
		}
		lastErr = err
		if overflow, reportedPrompt := isContextOverflowError(err); overflow {
			minContextTokens = contextRequiredTokens(minContextTokens, reportedPrompt, cand.target.ContextTokens)
			overflowFrom = cand.target.Account
			continue
		}
		if !isFailoverEligibleError(err) {
			return nil, err
		}
		if p.circuit != nil {
			p.circuit.recordFailure(cand.target.Account, err)
		}
		if i+1 < len(ordered) {
			next := ordered[i+1]
			if resErr := p.checkResidency(next); resErr != nil {
				return nil, fmt.Errorf("%w (after upstream error: %v)", resErr, err)
			}
			if p.circuit != nil {
				p.circuit.recordFailover(cand.target.Account, next.target.Account)
			}
		}
	}
	if lastErr != nil {
		return nil, lastErr
	}
	if overflowFrom != "" {
		return nil, fmt.Errorf("%w: prompt (~%d tokens) exceeds context_tokens on all admitted candidates for model %q",
			agent.ErrContextOverflow, minContextTokens, p.requestedModel)
	}
	return nil, lastErr
}

func (p *failoverPlanner) StreamingSupported() bool {
	for _, cand := range p.candidates {
		if sp, ok := cand.planner.(agent.StreamingPlanner); ok && sp.StreamingSupported() {
			return true
		}
	}
	return false
}

func (p *failoverPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	ordered := p.orderedCandidates()
	if len(ordered) == 0 {
		return nil, fmt.Errorf("gateway: chat route: no admitted candidates for model %q", p.requestedModel)
	}
	minContextTokens := estimatePromptTokens(messages, tools)
	var lastErr error
	var overflowFrom string
	for i, cand := range ordered {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		if minContextTokens > 0 && !cand.target.FitsContext(minContextTokens) {
			if overflowFrom == "" {
				overflowFrom = cand.target.Account
			}
			continue
		}
		if err := p.checkResidency(cand); err != nil {
			if lastErr != nil {
				return nil, fmt.Errorf("%w (after upstream error: %v)", err, lastErr)
			}
			return nil, err
		}
		sp, ok := cand.planner.(agent.StreamingPlanner)
		if !ok || !sp.StreamingSupported() {
			lastErr = agent.ErrStreamingUnsupported
			continue
		}
		if overflowFrom != "" {
			if p.circuit != nil {
				p.circuit.recordFailover(overflowFrom, cand.target.Account)
			}
			overflowFrom = ""
		}
		var emitted atomic.Bool
		var wrappedSink agent.StreamSink
		if sink != nil {
			wrappedSink = func(part string) error {
				if part != "" {
					emitted.Store(true)
				}
				return sink(part)
			}
		}
		callCtx := agent.WithSerialStreamBoundary(ctx, func() { emitted.Store(true) })
		comp, err := sp.CompleteStream(callCtx, wrappedSink, messages, tools, candidateSampleOpts(cand, opts, i+1 < len(ordered))...)
		if err == nil {
			if p.circuit != nil {
				p.circuit.recordSuccess(cand.target.Account)
			}
			return comp, nil
		}
		lastErr = err
		if emitted.Load() {
			return nil, err
		}
		if overflow, reportedPrompt := isContextOverflowError(err); overflow {
			minContextTokens = contextRequiredTokens(minContextTokens, reportedPrompt, cand.target.ContextTokens)
			overflowFrom = cand.target.Account
			continue
		}
		if !isFailoverEligibleError(err) {
			return nil, err
		}
		if p.circuit != nil {
			p.circuit.recordFailure(cand.target.Account, err)
		}
		if i+1 < len(ordered) {
			next := ordered[i+1]
			if resErr := p.checkResidency(next); resErr != nil {
				return nil, fmt.Errorf("%w (after upstream error: %v)", resErr, err)
			}
			if p.circuit != nil {
				p.circuit.recordFailover(cand.target.Account, next.target.Account)
			}
		}
	}
	if lastErr == nil && overflowFrom != "" {
		return nil, fmt.Errorf("%w: prompt (~%d tokens) exceeds context_tokens on all admitted candidates for model %q",
			agent.ErrContextOverflow, minContextTokens, p.requestedModel)
	}
	if lastErr == nil {
		return nil, agent.ErrStreamingUnsupported
	}
	return nil, lastErr
}
