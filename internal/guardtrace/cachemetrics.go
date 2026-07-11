package guardtrace

// cachemetrics.go extends the guardtrace Fixture family with the frozen shed/reuse
// FOLD that the golden regression (#3639, epic #3569 cache-verify) pins. A canonical
// transcript records, per turn, the provider's token accounting (fresh input, cache
// read, cache creation); this fold projects that recorded usage into the two numbers
// the managed-cache transforms (anchor / upgrade / prune / defer) move — the KV-prefix
// REUSE ratio and the window SHED count — so a silent drift in how those numbers are
// derived reds a golden instead of nothing.
//
// It DEFINES NO NEW METRIC (out of scope for #3639): the reuse ratio is computed by the
// canonical process-global observer internal/cacheobs (the same reusedTokens/promptTokens
// realized cache-hit and the same frozen/partial/cold regime buckets the gateway scrapes
// onto /metrics), and the shed count is the internal/ablate ShedTokens sense — "how many
// resident tokens the fire removed" — summed over the transcript's turn-over-turn drop in
// resident prompt. guardtrace is tier 4 and cacheobs is tier 1, so this import is a legal
// downward composition (no hot-path code, replay harness only).

import (
	"math"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// CacheMetrics is the frozen shed/reuse projection of ONE canonical transcript. It is the
// row the golden regression pins: every field is a deterministic fold of the fixture's
// recorded per-turn usage, so two folds of the same fixture are byte-identical and any
// change to the derivation (or to the recorded transcript) shifts a field and reds the
// golden until it is regenerated with -update.
type CacheMetrics struct {
	SliceID string `json:"slice_id"`
	// Turns is the number of turns that carried a positive prompt (the turns cacheobs
	// actually attributed) — normally every turn in the fixture.
	Turns int `json:"turns"`
	// PromptTokens is the summed resident prompt across the transcript (fresh input +
	// cache_read + cache_creation per turn) — the denominator of the reuse ratio.
	PromptTokens int `json:"prompt_tokens"`
	// ReusedTokens is the summed cache_read across the transcript — the prefix tokens
	// the provider served warm, i.e. the numerator of the reuse ratio.
	ReusedTokens int `json:"reused_tokens"`
	// CacheCreationTokens is the summed cache_creation across the transcript (prompt
	// tokens freshly written to the cache), surfaced so a golden diff shows which axis moved.
	CacheCreationTokens int `json:"cache_creation_tokens"`
	// ShedTokens is the total resident prompt REMOVED from the window across the
	// transcript: the sum of each turn's positive drop in resident prompt versus the turn
	// before it (the internal/ablate ShedTokens sense). An append-only trajectory never
	// sheds (0); a compaction/prune fire that collapses the window sheds the collapse.
	ShedTokens int `json:"shed_tokens"`
	// ReuseRatio is the aggregate realized cache-hit ReusedTokens/PromptTokens as
	// computed by cacheobs, rounded to 6 decimals so the golden is float-stable across
	// platforms and Go versions.
	ReuseRatio float64 `json:"reuse_ratio"`
	// FrozenTurns / PartialTurns / ColdTurns bucket the turns by per-turn reuse ratio
	// (cacheobs FrozenFloor / ColdCeil) — the regime shape a single aggregate ratio hides.
	FrozenTurns  int `json:"frozen_turns"`
	PartialTurns int `json:"partial_turns"`
	ColdTurns    int `json:"cold_turns"`
}

// residentPromptTokens is the total prompt the provider prefilled for a turn: fresh input
// plus the cache_read served from the prefix plus the cache_creation freshly written. It
// is the per-turn resident size whose turn-over-turn drop is the shed, and the reuse-ratio
// denominator cacheobs is fed.
func residentPromptTokens(u Usage) int {
	return u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
}

// CacheMetrics folds the fixture's recorded per-turn usage into the frozen shed/reuse row.
// The reuse ratio + regime buckets come from a fresh cacheobs.Observer (the canonical
// metric, per-fixture isolated so it never touches the process-global Default); the shed
// count is the summed positive resident-prompt drop turn over turn.
func (f *Fixture) CacheMetrics() CacheMetrics {
	obs := cacheobs.New()
	m := CacheMetrics{SliceID: f.SliceID}
	prevResident := 0
	for i, t := range f.Turns {
		resident := residentPromptTokens(t.Usage)
		reused := t.Usage.CacheReadInputTokens
		obs.Observe(resident, reused)
		m.PromptTokens += resident
		m.ReusedTokens += reused
		m.CacheCreationTokens += t.Usage.CacheCreationInputTokens
		if i > 0 && resident < prevResident {
			m.ShedTokens += prevResident - resident
		}
		prevResident = resident
	}
	s := obs.Snapshot()
	m.Turns = int(s.Turns)
	m.ReuseRatio = round6(s.ReuseRatio)
	m.FrozenTurns = int(s.FrozenTurns)
	m.PartialTurns = int(s.PartialTurns)
	m.ColdTurns = int(s.ColdTurns)
	return m
}

// round6 rounds a ratio to 6 decimal places so a golden float compares exactly rather
// than carrying a platform-dependent last ULP.
func round6(x float64) float64 {
	return math.Round(x*1e6) / 1e6
}
