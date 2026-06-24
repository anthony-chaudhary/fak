// Package cacheobs is the process-global observability tap for in-kernel KV-prefix
// reuse — the LIVE measurement of the "frozen-trajectory cache cliff"
// (docs/explainers/frozen-trajectory-cache-cliff.md).
//
// The in-kernel planner already computes, on every served turn, how many prompt tokens
// it prefilled (promptTokens) and how many of them it served from the cached KV prefix
// (the RadixAttention match). That ratio IS the realized cache-hit. Until now it reached
// only a log line; this tap accumulates it so the gateway can scrape it onto /metrics
// (the fak_gateway_kv_prefix_* family), making the cliff observable in any fak kernel run
// rather than a number you can only model offline.
//
// It mirrors the established process-global stats pattern (blob.Default, vdso.Default):
// a low-tier (foundation) leaf with no imports beyond the stdlib, fed by the hot path and
// read by the metrics renderer. The cliff is legible from two derived signals:
//
//   - reuse ratio = reusedTokens / promptTokens — the realized cache-hit. A single,
//     linear, append-only agent climbs toward ~1 (the frozen ceiling); flexibility, cold
//     fan-out, or a divergent prefix drives it down.
//   - the per-regime turn buckets — frozen (reuse >= FrozenFloor), partial, cold
//     (reuse < ColdCeil) — show WHEN turns leave the frozen regime, which a single
//     cumulative ratio hides.
package cacheobs

import "sync"

// FrozenFloor / ColdCeil bucket a turn's reuse ratio into the three cliff regimes. A turn
// at or above FrozenFloor reused almost its whole prefix (the append-only ceiling); a turn
// below ColdCeil reused almost nothing (a cold first prefill, or a head-mutated / fanned-out
// turn that left the frozen regime). Between them is partial reuse.
const (
	FrozenFloor = 0.90
	ColdCeil    = 0.10
)

// Default is the process-global observer the in-kernel planner feeds and the gateway
// scrapes. One per process, like blob.Default / vdso.Default.
var Default = New()

// Observer accumulates in-kernel KV-prefix reuse across served turns. Safe for concurrent
// use — the gateway may drive Complete (which feeds Observe) from many goroutines.
type Observer struct {
	mu           sync.Mutex
	turns        uint64
	promptTokens uint64
	reusedTokens uint64
	frozen       uint64 // turns with reuse ratio >= FrozenFloor (the append-only ceiling)
	partial      uint64 // turns between ColdCeil and FrozenFloor
	cold         uint64 // turns with reuse ratio < ColdCeil (cold / head-mutated / fanned-out)
}

// New returns a fresh observer (tests use it for isolation; production uses Default).
func New() *Observer { return &Observer{} }

// Observe records one served in-kernel turn: promptTokens prefilled, of which
// reusedPrefixTokens were served from the cached KV prefix (the planner's `matched`).
// A non-positive promptTokens is ignored (no turn to attribute); reusedPrefixTokens is
// clamped into [0, promptTokens] so a miscount can never push the ratio outside [0,1] or
// the reused total above the prompt total.
func (o *Observer) Observe(promptTokens, reusedPrefixTokens int) {
	if o == nil || promptTokens <= 0 {
		return
	}
	if reusedPrefixTokens < 0 {
		reusedPrefixTokens = 0
	}
	if reusedPrefixTokens > promptTokens {
		reusedPrefixTokens = promptTokens
	}
	ratio := float64(reusedPrefixTokens) / float64(promptTokens)
	o.mu.Lock()
	o.turns++
	o.promptTokens += uint64(promptTokens)
	o.reusedTokens += uint64(reusedPrefixTokens)
	switch {
	case ratio >= FrozenFloor:
		o.frozen++
	case ratio < ColdCeil:
		o.cold++
	default:
		o.partial++
	}
	o.mu.Unlock()
}

// Stats is a point-in-time snapshot of the accumulated reuse.
type Stats struct {
	Turns        uint64
	PromptTokens uint64
	ReusedTokens uint64
	FrozenTurns  uint64
	PartialTurns uint64
	ColdTurns    uint64
	// ReuseRatio is reusedTokens/promptTokens — the realized cache-hit across all observed
	// turns. 0 when no turns have prompt tokens yet (an idle process never reports a
	// phantom ratio).
	ReuseRatio float64
}

// Snapshot returns the current accumulated stats. The ratio is derived under the lock so
// it is always consistent with the totals it is computed from.
func (o *Observer) Snapshot() Stats {
	if o == nil {
		return Stats{}
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	s := Stats{
		Turns:        o.turns,
		PromptTokens: o.promptTokens,
		ReusedTokens: o.reusedTokens,
		FrozenTurns:  o.frozen,
		PartialTurns: o.partial,
		ColdTurns:    o.cold,
	}
	if o.promptTokens > 0 {
		s.ReuseRatio = float64(o.reusedTokens) / float64(o.promptTokens)
	}
	return s
}
