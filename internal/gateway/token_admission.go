package gateway

// token_admission.go — HOST-LEVEL PROVIDER-TOKEN admission for the served path (#2019,
// M19 under the microagent epic #2000). The binding scale constraint for a fleet of
// microagents is the PROVIDER's rate limits — TPM (Anthropic: ITPM vs OTPM) plus a
// separate concurrency cap enforced simultaneously — not local CPU/RAM. The scheduler
// gate in admission.go bounds what runs on THIS node per instant; nothing bounded what
// the node spends against the provider per MINUTE. This file is that gate: a rolling-
// window token/concurrency admission built on the pure caps arithmetic in
// internal/ratelimit (TokenCaps.Decide — input/cached/uncached/output/total token caps
// + concurrency + the TargetUtilization headroom knob, ~0.9 in the issue's framing).
//
// SHAPE — reserve on estimate, settle on truth. A request is admitted against an
// ESTIMATE (the same chars/4 + max_tokens footprint the scheduler gate charges, split
// into input vs output); when the provider answers, the reservation is SETTLED with the
// provider's REAL usage, normalized across providers by agent.Usage's
// CachedPromptTokens()/UncachedPromptTokens() pair (the OpenAI/Gemini nested-details
// shape and the Anthropic separate-field shape both fold to the same TokenUsage — never
// the Anthropic-only field). Over-estimates return their headroom the moment the truth
// arrives, so the window tracks what the provider actually charged. A Release WITHOUT a
// settlement (planner error, an abandoned stream) settles the ESTIMATE conservatively:
// the provider may have consumed the input, and over-counting only under-admits — the
// safe direction for the no-429-storm acceptance.
//
// HONEST FENCE — what this is NOT (yet). One gate = ONE provider budget (one
// account/seat). The per-seat pool (M20) and the slot scheduler (M6) compose ABOVE
// this: a host wires one gate per seat and picks the seat first. Inert by default — no
// gate attached (SetTokenRateGate) leaves the request path byte-for-byte historical,
// the same inject-after-New posture as SetAdmissionController.

import (
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ratelimit"
)

// defaultTokenRateWindow is the accounting window when TokenRatePolicy.Window is unset —
// one minute, the "M" every provider's TPM/ITPM/OTPM ceilings are quoted in.
const defaultTokenRateWindow = time.Minute

// TokenRatePolicy configures one provider budget's admission window.
type TokenRatePolicy struct {
	// Caps are the per-window provider ceilings (ratelimit.TokenCaps): concurrency plus
	// input/cached-input/uncached-input/output/total token caps, each 0 = unlimited, all
	// scaled by TargetUtilization to preserve provider headroom (0.9 keeps ~10% free).
	Caps ratelimit.TokenCaps
	// Window is the rolling accounting window the token caps cover. <=0 defaults to one
	// minute (defaultTokenRateWindow).
	Window time.Duration
}

// TokenRateGate is one provider budget's rolling-window token/concurrency admission
// gate. Build with NewTokenRateGate; the zero value is not usable. Safe for concurrent
// use. Settled usage expires when the window rolls; in-flight reservations carry over
// (their calls are still outstanding against the provider).
type TokenRateGate struct {
	mu          sync.Mutex
	policy      TokenRatePolicy
	now         func() time.Time // injectable clock (tests); never nil after New
	windowStart time.Time
	settled     ratelimit.TokenUsage            // provider-confirmed usage this window
	reserved    map[uint64]ratelimit.TokenUsage // in-flight admission estimates
	seq         uint64
}

// NewTokenRateGate builds a gate under the given policy.
func NewTokenRateGate(p TokenRatePolicy) *TokenRateGate {
	if p.Window <= 0 {
		p.Window = defaultTokenRateWindow
	}
	g := &TokenRateGate{policy: p, now: time.Now, reserved: map[uint64]ratelimit.TokenUsage{}}
	g.windowStart = g.now()
	return g
}

// TokenReservation is one admitted call's hold on the window's token budget. Exactly one
// of Settle (the provider answered — record the real usage) or Release (no truth arrived —
// keep the conservative estimate) takes effect; both are idempotent and nil-safe.
type TokenReservation struct {
	gate     *TokenRateGate
	id       uint64
	estimate ratelimit.TokenUsage
	once     sync.Once
}

// Settle replaces the reservation's admission-time estimate with the provider's real
// normalized usage — the honest-accounting feedback edge (#2019 scope item 2).
func (r *TokenReservation) Settle(actual ratelimit.TokenUsage) {
	if r == nil || r.gate == nil {
		return
	}
	r.once.Do(func() { r.gate.settle(r.id, actual) })
}

// Release settles the original ESTIMATE if no real usage was ever reported (planner
// error, abandoned stream). Conservative by design: the provider may have consumed the
// input, and over-counting only under-admits. A no-op after Settle.
func (r *TokenReservation) Release() {
	if r == nil || r.gate == nil {
		return
	}
	r.once.Do(func() { r.gate.settle(r.id, r.estimate) })
}

// Admit checks one call's estimated footprint against the window's remaining budget and,
// on admission, reserves it. A refusal is the typed served-path *AdmissionError
// (VerdictShed → HTTP 429 via admissionErrorStatus — shed HERE instead of a 429 storm at
// the provider), naming the cap that fired and its arithmetic. A nil gate admits freely.
func (g *TokenRateGate) Admit(estimate ratelimit.TokenUsage) (*TokenReservation, error) {
	if g == nil {
		return nil, nil
	}
	estimate = estimate.Normalize()
	g.mu.Lock()
	defer g.mu.Unlock()
	g.rollWindowLocked()
	if d := g.policy.Caps.Decide(g.loadLocked(), estimate); !d.Admit {
		return nil, &AdmissionError{Verdict: VerdictShed, Reason: tokenShedReason(d, g.policy.Window)}
	}
	g.seq++
	g.reserved[g.seq] = estimate
	return &TokenReservation{gate: g, id: g.seq, estimate: estimate}, nil
}

// TokenRateSnapshot is the gate's current window state — the observability surface a
// host (or test) reads: what the provider confirmed, what is still reserved in flight.
type TokenRateSnapshot struct {
	WindowStart time.Time
	InFlight    int                  // live reservations (the concurrency dimension)
	Settled     ratelimit.TokenUsage // provider-confirmed usage this window
	Reserved    ratelimit.TokenUsage // in-flight admission estimates
}

// Snapshot returns the current window's accounting.
func (g *TokenRateGate) Snapshot() TokenRateSnapshot {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.rollWindowLocked()
	var reserved ratelimit.TokenUsage
	for _, r := range g.reserved {
		reserved = reserved.Add(r)
	}
	return TokenRateSnapshot{
		WindowStart: g.windowStart,
		InFlight:    len(g.reserved),
		Settled:     g.settled,
		Reserved:    reserved,
	}
}

func (g *TokenRateGate) settle(id uint64, usage ratelimit.TokenUsage) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.rollWindowLocked()
	delete(g.reserved, id)
	g.settled = g.settled.Add(usage)
}

// rollWindowLocked expires the settled usage once the window has elapsed. In-flight
// reservations survive the roll — their calls are still outstanding. Caller holds g.mu.
func (g *TokenRateGate) rollWindowLocked() {
	now := g.now()
	if now.Sub(g.windowStart) >= g.policy.Window {
		g.settled = ratelimit.TokenUsage{}
		g.windowStart = now
	}
}

// loadLocked is the window's current load: settled truth plus in-flight estimates, with
// the live reservation count as the concurrency dimension. Caller holds g.mu.
func (g *TokenRateGate) loadLocked() ratelimit.TokenLoad {
	tokens := g.settled
	for _, r := range g.reserved {
		tokens = tokens.Add(r)
	}
	return ratelimit.TokenLoad{Concurrent: len(g.reserved), Tokens: tokens}
}

// tokenShedReason renders the refusing cap's arithmetic so a client/operator learns
// which provider ceiling fired and by how much — the backoff hint on the 429.
func tokenShedReason(d ratelimit.TokenAdmission, window time.Duration) string {
	return fmt.Sprintf("provider token budget: cap %s used %d + requested %d exceeds limit %d per %s",
		d.Cap, d.Used, d.Requested, d.Limit, window)
}

// tokenUsageFromAgent folds a completion's provider-reported usage into the normalized
// ratelimit.TokenUsage the gate settles with. It goes through agent.Usage's
// CachedPromptTokens()/UncachedPromptTokens() pair — the provider-neutral split whose
// sum is the full resident prompt on every provider (OpenAI/Gemini fold the cache hit
// INTO prompt_tokens; Anthropic carries it in a separate field) — never the
// Anthropic-only counter (#2019 scope item 2).
func tokenUsageFromAgent(u agent.Usage) ratelimit.TokenUsage {
	cached := int64(u.CachedPromptTokens())
	uncached := int64(u.UncachedPromptTokens())
	return ratelimit.NewTokenUsage(uncached+cached, cached, int64(u.CompletionTokens))
}

// estimateServedTokenUsage is the admission-time estimate of one served turn's provider
// token footprint, split input vs output for the per-dimension caps: the same chars/4
// prompt heuristic the scheduler gate charges (estimateServedAdmissionTokens), with the
// request's max_tokens as the planned output ceiling (floor 1, matching that gate's
// unset-max posture). The whole prompt estimate is charged as UNCACHED input —
// conservative: an estimate can never borrow headroom from an unproven cache hit.
func estimateServedTokenUsage(messages []agent.Message, tools []agent.ToolDef, maxTokens int) ratelimit.TokenUsage {
	chars := servedPromptChars(messages, tools)
	input := int64(chars / 4)
	if chars > 0 && input == 0 {
		input = 1
	}
	output := int64(maxTokens)
	if output <= 0 {
		output = 1
	}
	return ratelimit.NewTokenUsage(input, 0, output)
}

// SetTokenRateGate wires the host-level provider-token admission gate (#2019) onto the
// Server: served requests reserve against it before the planner runs and settle it with
// the provider's real normalized usage after. nil detaches it (the request path goes
// byte-for-byte historical). Settable after New, guarded by admissionMu alongside
// SetAdmissionController. A nil receiver is a no-op.
func (s *Server) SetTokenRateGate(g *TokenRateGate) {
	if s == nil {
		return
	}
	s.admissionMu.Lock()
	s.tokenRateGate = g
	s.admissionMu.Unlock()
}
