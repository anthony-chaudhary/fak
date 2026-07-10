package metrics

import (
	"fmt"
	"sort"
)

// cache_break.go — the per-session cache-break witness counter (#2916), the
// observability half of the cache-integrity invariant. Hermes keeps cache
// integrity by CONVENTION (a sibling `--now`-style discipline) and by
// "behavior contracts over snapshots" tests, but has no RUNTIME counter of how
// often the prefix actually broke or what each break cost. Without a witnessed
// counter a cache regression is invisible until the invoice, and a code-review
// rule cannot catch a mutation a downstream harness or a provider quirk slips
// in. fak already reports cache-value figures with a witnessed basis
// (internal/gateway metrics_render, internal/metrics/provider_cache.go), so a
// first-class `cache_break_events` counter — per-event token cost AND a closed
// cause label — belongs on the same guard-exit-summary / Prometheus surface, so
// a rise is caught by a gate rather than an eyeball.
//
//   - CacheBreakCause is the CLOSED cause vocabulary: a break is one of
//     toolset_change / altered_turn / rebuilt_prompt / provider_quirk, or the
//     unknown fallback. A closed set (not free text) is what lets the gate
//     reason about a regression by cause instead of eyeballing a string.
//   - WitnessCacheBreak records one break: its cause and the cold-rebuild token
//     cost it caused (the warm prompt prefix that must now be re-prefilled).
//     The cost is caller-supplied from cache accounting, not invented here.
//   - FoldCacheBreaks folds a session's break witnesses into the operator
//     readout the issue asks for: total events, total token cost, and the
//     per-cause tally — the labeled, per-session metric.
//   - OpenMetricFamilies lowers that readout onto the Prometheus surface as the
//     `fak_cache_break_events_total` / `fak_cache_break_cost_tokens_total`
//     counter families, one sample per cause.
//   - CheckBudget / RegressesAgainst are the gate: a break count or token cost
//     above a DEFINED threshold (or a recorded baseline) fails — not "any
//     nonzero count fails", which the contract's confusion note rules out.
//
// This package stays pure — no engine, no kernel import — so the seam is
// unit-testable and any guard-exit or serve path can fold its own detected
// breaks through it. The SOURCE of the events (the mid-conversation
// prefix-mutation detector) is sibling issue #2915; this file is the counter
// that consumes those events, not the detector. Until #2915 wires a live
// producer, this primitive lands inert (mirroring the deferred-vs-now
// primitive in cache_invalidation.go, #2895).
//
// Generation intent: gen/next foundation (Hermes-evidence epic #2908). A
// near-term cache observability foundation that is runnable once its live
// producer (#2915) and a gate exist — the cache/context program map classifies
// live provider/cache metrics as gen/next, promoted to gen/now only by a live
// caller and net-true corroboration.
//   - Promotion evidence (toward "now"): sibling #2915's detector feeds real
//     WitnessCacheBreak events into a FoldCacheBreaks readout on the guard exit
//     summary, and the per-event CostTokens is corroborated against measured
//     provider cache-write tokens (internal/metrics/provider_cache.go); a gate
//     wired on CheckBudget then fails a deliberate prefix-break regression in a
//     live guard/serve session, not just a unit fixture.
//   - Demotion / retirement evidence: if #2915 never lands a detector, no path
//     produces CacheBreakEvents and this counter stays a zero-valued fixture —
//     an unwitnessed counter earns nothing and should be retired with it.
//   - Invalidating assumption: that a break's cost is a single re-prefill token
//     figure. A provider quirk can partially invalidate and repay across turns,
//     so CostTokens is the cold-rebuild UPPER bound the caller supplies, not a
//     settled invoice; the gate thresholds a witnessed proxy, not the bill.

// CacheBreakCause is the closed vocabulary for why a cached prompt prefix broke.
// A closed set (rather than a free-text field) is what lets the regression gate
// reason about a break by cause. Any value outside the set folds to
// CacheBreakUnknown.
type CacheBreakCause string

const (
	// CacheBreakToolsetChange is a break caused by the available tool set
	// changing mid-conversation (a skill/tool install or removal).
	CacheBreakToolsetChange CacheBreakCause = "toolset_change"
	// CacheBreakAlteredTurn is a break caused by an already-sent turn being
	// mutated after it was cached (a downstream harness rewriting history).
	CacheBreakAlteredTurn CacheBreakCause = "altered_turn"
	// CacheBreakRebuiltPrompt is a break caused by the system prompt / prefix
	// being rebuilt, invalidating the warm prefix.
	CacheBreakRebuiltPrompt CacheBreakCause = "rebuilt_prompt"
	// CacheBreakProviderQuirk is a break attributed to a provider-side cache
	// eviction or vary-axis quirk rather than a fak-authored mutation.
	CacheBreakProviderQuirk CacheBreakCause = "provider_quirk"
	// CacheBreakUnknown is the fallback for an unattributed or out-of-vocabulary
	// cause. It keeps the label closed without dropping the event.
	CacheBreakUnknown CacheBreakCause = "unknown"
)

// cacheBreakCauseOrder is the canonical, deterministic ordering used when a
// report enumerates causes (report ByCause and the Prometheus samples), so the
// witnessed output is stable across runs.
var cacheBreakCauseOrder = map[CacheBreakCause]int{
	CacheBreakToolsetChange: 0,
	CacheBreakAlteredTurn:   1,
	CacheBreakRebuiltPrompt: 2,
	CacheBreakProviderQuirk: 3,
	CacheBreakUnknown:       4,
}

// Valid reports whether c is a known member of the closed vocabulary.
func (c CacheBreakCause) Valid() bool {
	_, ok := cacheBreakCauseOrder[c]
	return ok
}

// NormalizeCacheBreakCause maps any cause to the closed set: a known member
// passes through, anything else (including empty) folds to CacheBreakUnknown.
func NormalizeCacheBreakCause(c CacheBreakCause) CacheBreakCause {
	if c.Valid() {
		return c
	}
	return CacheBreakUnknown
}

// CacheBreakEvent is one witnessed cache-break: the closed cause and the
// cold-rebuild token cost the break caused (the warm prompt prefix that must
// now be re-prefilled). CostTokens is caller-supplied from cache accounting.
type CacheBreakEvent struct {
	Cause      CacheBreakCause `json:"cause"`
	CostTokens int64           `json:"cost_tokens"`
}

// WitnessCacheBreak records one cache-break event under the closed cause
// vocabulary. A negative cost is clamped to zero; an out-of-vocabulary cause
// folds to CacheBreakUnknown so the label stays closed.
func WitnessCacheBreak(cause CacheBreakCause, costTokens int64) CacheBreakEvent {
	if costTokens < 0 {
		costTokens = 0
	}
	return CacheBreakEvent{Cause: NormalizeCacheBreakCause(cause), CostTokens: costTokens}
}

// CacheBreakCauseTally is the per-cause fold: how many breaks that cause
// produced and their total token cost.
type CacheBreakCauseTally struct {
	Cause      CacheBreakCause `json:"cause"`
	Events     int             `json:"events"`
	CostTokens int64           `json:"cost_tokens"`
}

// CacheBreakReport is the per-session operator readout the issue asks for: the
// total break count, the total token cost, and the per-cause breakdown. It is
// the labeled, per-session metric — the report, not per-command convention, is
// what a gate reads.
type CacheBreakReport struct {
	Events     int                    `json:"events"`
	CostTokens int64                  `json:"cost_tokens"`
	ByCause    []CacheBreakCauseTally `json:"by_cause"`
}

// FoldCacheBreaks folds a session's break witnesses into the report. Causes are
// normalized to the closed set and the per-cause tally is emitted in canonical
// order, so the witnessed output is deterministic.
func FoldCacheBreaks(events []CacheBreakEvent) CacheBreakReport {
	tally := make(map[CacheBreakCause]*CacheBreakCauseTally)
	var r CacheBreakReport
	for _, e := range events {
		cause := NormalizeCacheBreakCause(e.Cause)
		cost := e.CostTokens
		if cost < 0 {
			cost = 0
		}
		r.Events++
		r.CostTokens += cost
		t, ok := tally[cause]
		if !ok {
			t = &CacheBreakCauseTally{Cause: cause}
			tally[cause] = t
		}
		t.Events++
		t.CostTokens += cost
	}
	r.ByCause = make([]CacheBreakCauseTally, 0, len(tally))
	for _, t := range tally {
		r.ByCause = append(r.ByCause, *t)
	}
	sort.SliceStable(r.ByCause, func(i, j int) bool {
		return cacheBreakCauseOrder[r.ByCause[i].Cause] < cacheBreakCauseOrder[r.ByCause[j].Cause]
	})
	return r
}

// The Prometheus counter family names for the cache-break surface. Counter
// families carry the conventional `_total` suffix (see fak_cache_* in
// internal/gateway/metrics_render.go).
const (
	// CacheBreakEventsMetric counts cache-break events, labeled by cause.
	CacheBreakEventsMetric = "fak_cache_break_events_total"
	// CacheBreakCostMetric sums the cold-rebuild token cost of cache breaks,
	// labeled by cause.
	CacheBreakCostMetric = "fak_cache_break_cost_tokens_total"
)

// OpenMetricFamilies lowers the report onto the Prometheus surface as two
// counter families — event count and token cost — each with one sample per
// cause. A session with no breaks renders the families with no samples (a clean
// zero), which is exactly what a regression gate wants to see.
func (r CacheBreakReport) OpenMetricFamilies() []OpenMetricFamily {
	events := make([]OpenMetricSample, 0, len(r.ByCause))
	cost := make([]OpenMetricSample, 0, len(r.ByCause))
	for _, t := range r.ByCause {
		labels := []OpenMetricLabel{{Name: "cause", Value: string(t.Cause)}}
		events = append(events, OpenMetricSample{Labels: labels, Value: float64(t.Events)})
		cost = append(cost, OpenMetricSample{Labels: labels, Value: float64(t.CostTokens)})
	}
	return []OpenMetricFamily{
		{
			Name:    CacheBreakEventsMetric,
			Help:    "Cache-break events this session, labeled by closed cause (toolset_change/altered_turn/rebuilt_prompt/provider_quirk/unknown). A rise means the warm prompt prefix broke more often.",
			Type:    OpenMetricCounter,
			Samples: events,
		},
		{
			Name:    CacheBreakCostMetric,
			Help:    "Cold-rebuild token cost of cache breaks this session, labeled by closed cause. Each break's cost is the warm prompt prefix that had to be re-prefilled.",
			Type:    OpenMetricCounter,
			Samples: cost,
		},
	}
}

// CacheBreakBudget is the defined regression threshold the gate reads: the most
// break events and the most token cost a session may incur before the gate
// fails. It is a defined threshold, not "any nonzero count fails".
type CacheBreakBudget struct {
	MaxEvents     int   `json:"max_events"`
	MaxCostTokens int64 `json:"max_cost_tokens"`
}

// CheckBudget returns a non-nil error naming what exceeded the budget when the
// session's break count or token cost is over the defined threshold, so a gate
// can fail a regression by returning this error. A report within budget returns
// nil.
func (r CacheBreakReport) CheckBudget(b CacheBreakBudget) error {
	if r.Events > b.MaxEvents {
		return fmt.Errorf("metrics: cache_break_events regression: %d events exceeds budget of %d", r.Events, b.MaxEvents)
	}
	if r.CostTokens > b.MaxCostTokens {
		return fmt.Errorf("metrics: cache_break cost regression: %d tokens exceeds budget of %d", r.CostTokens, b.MaxCostTokens)
	}
	return nil
}

// RegressesAgainst reports whether this session's cache-break count or token
// cost rose above a recorded baseline — the baseline form of the gate, for a
// caller that compares against a prior witnessed session rather than a fixed
// budget.
func (r CacheBreakReport) RegressesAgainst(baseline CacheBreakReport) bool {
	return r.Events > baseline.Events || r.CostTokens > baseline.CostTokens
}
