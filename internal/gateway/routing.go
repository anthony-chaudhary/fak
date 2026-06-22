package gateway

import (
	"errors"
	"fmt"
	"sort"
	"sync"
)

// routing.go — intelligent request routing and tiered serving (#398).
//
// BASELINE (the routing behavior this replaces). Today the gateway is single-tier:
// Config.EngineID/Model name ONE engine and every request — a one-token "hi" and a
// 100k-token refactor alike — is served by it. There is no notion of "send the cheap
// request to the cheap model." Cost-per-token is therefore flat: the floor is set by
// the single configured tier, so small/best-effort traffic pays the large-tier price.
//
// WHAT THIS ADDS. A pure, deterministic routing policy that classifies an inbound
// request and selects the appropriate serving Tier, with health-aware fallback. It is
// ADDITIVE: it touches no existing request path and breaks no API, so it can be wired
// into dispatch incrementally (the host picks the EngineID from Route().Tier.Model).
// The whole policy is data-in / decision-out — no I/O, no goroutines on the hot path —
// so it is trivially A/B-testable: run two RouterConfigs over the same RequestClass and
// diff the selected tiers.
//
// THE ALGORITHM (Router.Route).
//  1. CANDIDATES — keep every tier that is (a) healthy, (b) has capacity for the
//     prompt (Tier.MaxPromptTokens == 0 means unbounded, else >= PromptTokens), (c) is
//     Interactive-capable when the request demands interactive latency, and (d) meets
//     the complexity floor (a High-complexity request is never served by the smallest
//     tier — complexity demands at least the Nth tier by capability order).
//  2. SELECT — among candidates, the Strategy decides the winner:
//       - SizeBased / Hybrid: the SMALLEST adequate tier (minimize resource use — the
//         small request goes to the small model).
//       - CostBased: the lowest CostPerMTok candidate (cheapest that still fits).
//       - LatencyBased: interactive => the smallest/fastest candidate; batch => the
//         largest-capacity candidate (best throughput for high-volume work).
//  3. FALLBACK — the remaining healthy candidates, ascending by capacity, are the
//     ordered fallback chain: if the selected tier fails health mid-flight the caller
//     walks to the next. If NO tier qualifies, Route returns ErrNoTier (a structured
//     refusal the caller routes to a 503 / replan, never a silent mis-route).
//
// Tiers are CONFIGURABLE (RouterConfig.Tiers) and the strategy is selectable, so a
// deployment expresses its own size/latency/cost trade-off without a code change.

// RoutingStrategy selects how Route picks among the tiers that satisfy a request.
type RoutingStrategy string

const (
	// StrategySizeBased routes by prompt size: the smallest tier that fits. Minimizes
	// resource use — the default.
	StrategySizeBased RoutingStrategy = "size"
	// StrategyLatencyBased routes interactive traffic to the fastest tier and batch
	// traffic to the highest-throughput (largest) tier.
	StrategyLatencyBased RoutingStrategy = "latency"
	// StrategyCostBased routes to the cheapest tier that still satisfies the request.
	StrategyCostBased RoutingStrategy = "cost"
	// StrategyHybrid combines every signal (capacity + latency + complexity), then
	// minimizes resource use among the survivors. The recommended general policy.
	StrategyHybrid RoutingStrategy = "hybrid"
)

// LatencyClass is the latency requirement carried by a request.
type LatencyClass string

const (
	// LatencyUnknown leaves latency unconstrained (no interactive-only filtering).
	LatencyUnknown LatencyClass = ""
	// LatencyInteractive is a low-latency, user-facing turn: never route it to a
	// batch-only tier.
	LatencyInteractive LatencyClass = "interactive"
	// LatencyBatch is high-throughput, best-effort work: prefer the cheapest/biggest
	// tier.
	LatencyBatch LatencyClass = "batch"
)

// Complexity is the coarse difficulty of a request. It sets a floor on tier capability:
// a harder request must land on at least the Nth tier by capability order.
type Complexity int

const (
	// ComplexityLow may be served by any tier (floor index 0).
	ComplexityLow Complexity = iota
	// ComplexityMedium requires at least the 2nd tier by capability (floor index 1).
	ComplexityMedium
	// ComplexityHigh requires at least the 3rd tier by capability (floor index 2).
	ComplexityHigh
)

// floorIndex maps a Complexity to the minimum tier ordinal (capped at the last tier)
// that may serve it.
func (c Complexity) floorIndex(nTiers int) int {
	idx := int(c)
	if idx >= nTiers {
		idx = nTiers - 1
	}
	if idx < 0 {
		idx = 0
	}
	return idx
}

// Tier is one serving tier: a (model, hardware) class with a capacity ceiling, a
// relative cost, and whether it is suitable for interactive latency. Tiers are ordered
// in RouterConfig.Tiers ascending by capability (smallest/cheapest first).
type Tier struct {
	// Name is the tier label used in decisions and metrics (e.g. "small").
	Name string
	// Model is the engine/model id the host dispatches to when this tier is chosen.
	Model string
	// MaxPromptTokens is the largest prompt this tier serves. 0 means unbounded.
	MaxPromptTokens int
	// CostPerMTok is the relative cost per million tokens, used by StrategyCostBased.
	CostPerMTok float64
	// Interactive marks a tier fast enough for low-latency interactive turns. A
	// batch-only (Interactive == false) tier is never chosen for LatencyInteractive.
	Interactive bool
}

// RequestClass is the classified shape of an inbound request — the input to Route.
// Build it from the wire with Classify, or populate it directly.
type RequestClass struct {
	// PromptTokens is the estimated prompt length in tokens.
	PromptTokens int
	// Latency is the request's latency requirement.
	Latency LatencyClass
	// Complexity is the request's coarse difficulty.
	Complexity Complexity
}

// RouterConfig is the configurable routing policy: the strategy and the ordered tiers.
type RouterConfig struct {
	Strategy RoutingStrategy
	Tiers    []Tier
}

// DefaultRouterConfig returns a sensible three-tier hybrid policy (small / medium /
// large) for out-of-the-box use. Capacities and costs are relative defaults a
// deployment is expected to override.
func DefaultRouterConfig() RouterConfig {
	return RouterConfig{
		Strategy: StrategyHybrid,
		Tiers: []Tier{
			{Name: "small", Model: "small", MaxPromptTokens: 4096, CostPerMTok: 1, Interactive: true},
			{Name: "medium", Model: "medium", MaxPromptTokens: 32768, CostPerMTok: 5, Interactive: true},
			{Name: "large", Model: "large", MaxPromptTokens: 0, CostPerMTok: 20, Interactive: false},
		},
	}
}

// Validate checks a RouterConfig is well-formed: at least one tier, unique non-empty
// names, non-negative capacities, and a known strategy ("" defaults to size-based).
func (c RouterConfig) Validate() error {
	if len(c.Tiers) == 0 {
		return errors.New("routing: config has no tiers")
	}
	switch c.Strategy {
	case "", StrategySizeBased, StrategyLatencyBased, StrategyCostBased, StrategyHybrid:
	default:
		return fmt.Errorf("routing: unknown strategy %q", c.Strategy)
	}
	seen := make(map[string]bool, len(c.Tiers))
	for i, t := range c.Tiers {
		if t.Name == "" {
			return fmt.Errorf("routing: tier %d has an empty name", i)
		}
		if seen[t.Name] {
			return fmt.Errorf("routing: duplicate tier name %q", t.Name)
		}
		seen[t.Name] = true
		if t.MaxPromptTokens < 0 {
			return fmt.Errorf("routing: tier %q has negative MaxPromptTokens", t.Name)
		}
	}
	return nil
}

// ErrNoTier is returned by Route when no healthy tier can satisfy the request (the
// prompt exceeds every capacity, every adequate tier is unhealthy, or an interactive
// request finds no interactive-capable tier). It is a structured refusal: the caller
// maps it to a 503 / replan, never a silent mis-route.
var ErrNoTier = errors.New("routing: no healthy tier satisfies the request")

// Decision is the outcome of Route: the selected tier, the ordered health fallback
// chain, the strategy that decided, and a human-readable reason.
type Decision struct {
	Tier      Tier
	Fallbacks []Tier
	Strategy  RoutingStrategy
	Reason    string
}

// Router selects a serving tier for a classified request under a configured policy,
// tracking per-tier health for fallback. It is safe for concurrent use.
type Router struct {
	cfg      RouterConfig
	strategy RoutingStrategy

	mu        sync.RWMutex
	unhealthy map[string]bool // tier name -> down
}

// NewRouter validates cfg and returns a Router with every tier healthy.
func NewRouter(cfg RouterConfig) (*Router, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	strategy := cfg.Strategy
	if strategy == "" {
		strategy = StrategySizeBased
	}
	return &Router{cfg: cfg, strategy: strategy, unhealthy: make(map[string]bool)}, nil
}

// SetHealth marks a tier up (healthy=true) or down (healthy=false) for fallback. An
// unknown tier name is a no-op. Health checking is the caller's job; this is where the
// result lands so Route can route around a failed tier.
func (r *Router) SetHealth(name string, healthy bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if healthy {
		delete(r.unhealthy, name)
	} else {
		r.unhealthy[name] = true
	}
}

// Healthy reports whether the named tier is currently considered up.
func (r *Router) Healthy(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return !r.unhealthy[name]
}

// Route classifies and selects. It returns the chosen Decision or ErrNoTier when no
// healthy tier qualifies. Pure given the current health snapshot — no I/O.
func (r *Router) Route(req RequestClass) (Decision, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	floor := req.Complexity.floorIndex(len(r.cfg.Tiers))

	// 1. CANDIDATES: index-tagged so the complexity floor and ascending order survive.
	type cand struct {
		idx  int
		tier Tier
	}
	var cands []cand
	for i, t := range r.cfg.Tiers {
		if r.unhealthy[t.Name] {
			continue
		}
		if i < floor {
			continue // below the complexity floor
		}
		if t.MaxPromptTokens != 0 && req.PromptTokens > t.MaxPromptTokens {
			continue // not enough capacity for the prompt
		}
		if req.Latency == LatencyInteractive && !t.Interactive {
			continue // interactive turn cannot use a batch-only tier
		}
		cands = append(cands, cand{idx: i, tier: t})
	}
	if len(cands) == 0 {
		return Decision{}, ErrNoTier
	}

	// 2. SELECT by strategy.
	winner := 0 // index into cands
	switch r.strategy {
	case StrategyCostBased:
		for i := range cands {
			if cands[i].tier.CostPerMTok < cands[winner].tier.CostPerMTok {
				winner = i
			}
		}
	case StrategyLatencyBased:
		if req.Latency == LatencyBatch {
			winner = len(cands) - 1 // largest capacity == best throughput
		} else {
			winner = 0 // smallest/fastest adequate tier
		}
	default: // StrategySizeBased, StrategyHybrid: smallest adequate tier.
		winner = 0
	}

	// 3. FALLBACK: every OTHER healthy candidate, ascending by capacity (unbounded last).
	fallbacks := make([]Tier, 0, len(cands)-1)
	for i := range cands {
		if i == winner {
			continue
		}
		fallbacks = append(fallbacks, cands[i].tier)
	}
	sort.SliceStable(fallbacks, func(a, b int) bool {
		return tierCapacityLess(fallbacks[a], fallbacks[b])
	})

	return Decision{
		Tier:      cands[winner].tier,
		Fallbacks: fallbacks,
		Strategy:  r.strategy,
		Reason: fmt.Sprintf("strategy=%s prompt_tokens=%d latency=%s complexity=%d -> tier=%s",
			r.strategy, req.PromptTokens, req.Latency, req.Complexity, cands[winner].tier.Name),
	}, nil
}

// tierCapacityLess orders tiers ascending by capacity; an unbounded tier (0) sorts last.
func tierCapacityLess(a, b Tier) bool {
	if a.MaxPromptTokens == 0 {
		return false
	}
	if b.MaxPromptTokens == 0 {
		return true
	}
	return a.MaxPromptTokens < b.MaxPromptTokens
}

// Classify derives a RequestClass from raw request signals. PromptTokens is the
// estimated prompt length; latency and complexity are the caller's declared hints
// ("" / zero are valid and route conservatively). It is the wire-facing entry point
// the gateway uses before Route.
func Classify(promptTokens int, latency LatencyClass, complexity Complexity) RequestClass {
	if promptTokens < 0 {
		promptTokens = 0
	}
	return RequestClass{PromptTokens: promptTokens, Latency: latency, Complexity: complexity}
}
