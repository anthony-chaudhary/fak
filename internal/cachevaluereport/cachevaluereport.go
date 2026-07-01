// Package cachevaluereport rolls up the durable kernel cache-value ledger
// (internal/cachevalueledger, docs/nightrun/cache-value.jsonl) into a TREND over
// time — the by-week / by-session_type view that cachevalueledger.ScoreLedger
// deliberately does not produce (it collapses every row into a single all-time
// aggregate gate number).
//
// This is rung A (the keystone) of epic #1301 — the cache-effectiveness P&L
// roll-up. It builds TRACK 1 only: the WITNESSED pure-kernel cache value (the
// in-kernel KV-prefix reuse fak authors byte-for-byte). The OBSERVED provider-$
// track (Track 2) is rung B/C and joins here later via a sibling fold.
//
// The fold is PURE and deterministic: rows + a caller-supplied `now` in, a Report
// out — no clock, no I/O, no network — mirroring the internal/cadencereport and
// internal/execrollup pattern (pure Fold + impure collect shell elsewhere).
//
// HONESTY FENCE (#1066, the DeepSWE cache-value fence): the honest single-session
// kernel value is the marginal over a tuned warm-KV server (~1.0x); this package
// publishes the WITNESSED realized reuse ratio and the marginal-over-warm-KV family
// and NEVER computes the vs-naive 1/(1-reuse) re-prefill multiple. The omission is
// the fence — mirroring cachevalueledger.ScoreLedgerResult and
// cachewitness.CacheValue self-labeling.
package cachevaluereport

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
)

// Schema versions the roll-up envelope so a downstream reader can pin it.
const Schema = "fak-cache-value-rollup/1"

// PublishableValueFamily mirrors cachevalueledger.PublishableValueFamily — the only
// cache-value framing the #1066 fence permits for a published number.
const PublishableValueFamily = cachevalueledger.PublishableValueFamily

// MinBucketTurns is the multi-turn-turn floor below which a bucket's realized reuse
// is reported but flagged Thin (not trend-significant) — the same posture
// cachevalueledger.MinGateTurns takes for the all-time gate.
const MinBucketTurns = cachevalueledger.MinGateTurns

// reuseEpsilon is the dead-band below which a bucket-over-bucket reuse change reads
// flat rather than improved/regressed, so float noise does not manufacture a trend.
const reuseEpsilon = 0.005

// Trend is one bucket's direction versus the previous (chronological) bucket.
type Trend string

const (
	// TrendNew marks the first bucket in the series (no prior to compare).
	TrendNew Trend = "new"
	// TrendImproved marks a realized-reuse gain over the prior bucket beyond the epsilon.
	TrendImproved Trend = "improved"
	// TrendRegressed marks a realized-reuse drop below the prior bucket beyond the epsilon.
	TrendRegressed Trend = "regressed"
	// TrendFlat marks a change within the epsilon dead-band.
	TrendFlat Trend = "flat"
)

// Bucket is one period's WITNESSED kernel-cache roll-up (Track 1). RealizedReuseRatio
// is GateReusedTokens/GatePromptTokens over MULTI-TURN sessions (turns >= 2) only —
// a single-turn cold run has no previous turn to reuse from, so folding it in would
// manufacture a false reuse number, exactly as ScoreLedger excludes it.
type Bucket struct {
	Period            string `json:"period"`   // ISO week, e.g. "2026-W26"
	Start             string `json:"start"`    // earliest row date in the bucket (YYYY-MM-DD)
	Sessions          int    `json:"sessions"` // rows with turns > 0
	MultiTurnSessions int    `json:"multi_turn_sessions"`

	Turns          uint64 `json:"turns"`
	MultiTurnTurns uint64 `json:"multi_turn_turns"`
	FrozenTurns    uint64 `json:"frozen_turns"`
	PartialTurns   uint64 `json:"partial_turns"`
	ColdTurns      uint64 `json:"cold_turns"`

	PromptTokens     uint64 `json:"prompt_tokens"`
	ReusedTokens     uint64 `json:"reused_tokens"`
	GatePromptTokens uint64 `json:"gate_prompt_tokens"` // multi-turn only
	GateReusedTokens uint64 `json:"gate_reused_tokens"`

	RealizedReuseRatio float64 `json:"realized_reuse_ratio"`
	Thin               bool    `json:"thin"` // MultiTurnTurns < MinBucketTurns

	Trend           Trend   `json:"trend"`
	DeltaReuseRatio float64 `json:"delta_reuse_ratio"`

	// BySessionType counts sessions per session_type ("guard"|"serve"|"run") so the
	// roll-up can attribute reuse to the front door it came through.
	BySessionType map[string]int `json:"by_session_type"`
}

// Report is the rolled-up envelope (schema/ok/verdict/finding/reason/next_action),
// matching the repo's canonical report contract. Buckets are chronological.
type Report struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	Granularity string `json:"granularity"` // "week"

	TotalRows         int `json:"total_rows"`     // rows with turns > 0
	TotalSessions     int `json:"total_sessions"` // == TotalRows (alias for the human report)
	MultiTurnSessions int `json:"multi_turn_sessions"`

	// LatestReuseRatio is the most recent bucket's realized reuse; LatestTrend is its
	// direction. These are the headline a card shows.
	LatestReuseRatio float64 `json:"latest_reuse_ratio"`
	LatestTrend      Trend   `json:"latest_trend"`

	Buckets []Bucket `json:"buckets"`

	// #1066 fence self-labels — a downstream reader can never mistake the realized
	// reuse for the forbidden vs-naive multiple.
	PublishableValueFamily  string `json:"publishable_value_family"`
	VsNaiveMultipleExcluded bool   `json:"vs_naive_multiple_excluded"`

	OK         bool   `json:"ok"`
	Verdict    string `json:"verdict"` // MEASURED | INSUFFICIENT
	Finding    string `json:"finding"`
	Reason     string `json:"reason"`
	NextAction string `json:"next_action"`
}

// isoWeek formats a date as its ISO-8601 year-week key ("2026-W26").
func isoWeek(t time.Time) string {
	y, w := t.ISOWeek()
	return fmt.Sprintf("%04d-W%02d", y, w)
}

// sortedPeriodKeys returns the string keys of a period-indexed map in chronological
// order. ISO week keys (and the "\x00"-joined provider/mechanism variants) sort
// chronologically as plain strings, so a lexical sort is the chronological sort.
func sortedPeriodKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Fold rolls a slice of ledger rows up into a weekly trend Report. It is pure: the
// only time input is `now`, used solely to stamp GeneratedAt — bucketing comes from
// each row's own Date. Rows with zero turns (no session activity) are skipped, the
// same way ScoreLedger skips them.
func Fold(rows []cachevalueledger.Row, now time.Time) Report {
	r := Report{
		Schema:                  Schema,
		GeneratedAt:             now.UTC().Format(time.RFC3339),
		Granularity:             "week",
		PublishableValueFamily:  PublishableValueFamily,
		VsNaiveMultipleExcluded: true,
		Verdict:                 "INSUFFICIENT",
		OK:                      true,
		LatestTrend:             TrendNew,
	}

	type agg struct {
		b     Bucket
		start time.Time
	}
	byPeriod := map[string]*agg{}

	for _, row := range rows {
		if row.Turns == 0 {
			continue
		}
		d, err := time.Parse("2006-01-02", row.Date)
		if err != nil {
			continue
		}
		r.TotalRows++
		key := isoWeek(d)
		a := byPeriod[key]
		if a == nil {
			a = &agg{b: Bucket{Period: key, BySessionType: map[string]int{}}, start: d}
			byPeriod[key] = a
		}
		if d.Before(a.start) {
			a.start = d
		}
		b := &a.b
		b.Sessions++
		b.Turns += row.Turns
		b.FrozenTurns += row.FrozenTurns
		b.PartialTurns += row.PartialTurns
		b.ColdTurns += row.ColdTurns
		b.PromptTokens += row.PromptTokens
		b.ReusedTokens += row.ReusedTokens
		st := row.SessionType
		if st == "" {
			st = "unknown"
		}
		b.BySessionType[st]++
		if row.Turns >= 2 {
			b.MultiTurnSessions++
			b.MultiTurnTurns += row.Turns
			b.GatePromptTokens += row.PromptTokens
			b.GateReusedTokens += row.ReusedTokens
		}
	}

	r.TotalSessions = r.TotalRows

	keys := sortedPeriodKeys(byPeriod)

	var prev *Bucket
	for _, k := range keys {
		a := byPeriod[k]
		b := a.b
		b.Start = a.start.Format("2006-01-02")
		if b.GatePromptTokens > 0 {
			b.RealizedReuseRatio = float64(b.GateReusedTokens) / float64(b.GatePromptTokens)
		}
		b.Thin = b.MultiTurnTurns < MinBucketTurns
		if prev == nil {
			b.Trend = TrendNew
		} else {
			b.DeltaReuseRatio = b.RealizedReuseRatio - prev.RealizedReuseRatio
			switch {
			case b.DeltaReuseRatio > reuseEpsilon:
				b.Trend = TrendImproved
			case b.DeltaReuseRatio < -reuseEpsilon:
				b.Trend = TrendRegressed
			default:
				b.Trend = TrendFlat
			}
		}
		r.MultiTurnSessions += b.MultiTurnSessions
		r.Buckets = append(r.Buckets, b)
		prev = &r.Buckets[len(r.Buckets)-1]
	}

	if n := len(r.Buckets); n > 0 {
		last := r.Buckets[n-1]
		r.LatestReuseRatio = last.RealizedReuseRatio
		r.LatestTrend = last.Trend
	}

	r.fillVerdict()
	return r
}

// fillVerdict sets the report-contract fields. This is a REPORT, not a gate: OK stays
// true; Verdict is INSUFFICIENT only when no bucket carries multi-turn evidence to
// trend on (mirroring ScoreLedger falling open on a thin corpus).
func (r *Report) fillVerdict() {
	hasMulti := false
	for _, b := range r.Buckets {
		if b.MultiTurnTurns > 0 {
			hasMulti = true
			break
		}
	}
	if !hasMulti {
		r.Verdict = "INSUFFICIENT"
		r.Finding = fmt.Sprintf("%d session(s) across %d week(s); no multi-turn reuse to trend yet", r.TotalSessions, len(r.Buckets))
		r.Reason = "realized KV-prefix reuse needs sessions with >= 2 turns; single-turn cold runs have no prior turn to reuse from"
		r.NextAction = "accumulate multi-turn guard/serve sessions into docs/nightrun/cache-value.jsonl, then re-roll"
		return
	}
	r.Verdict = "MEASURED"
	r.Finding = fmt.Sprintf("latest week realized reuse %.3f (%s) over %d week(s), %d session(s)",
		r.LatestReuseRatio, r.LatestTrend, len(r.Buckets), r.TotalSessions)
	r.Reason = "WITNESSED in-kernel KV-prefix reuse, multi-turn sessions only; " + PublishableValueFamily
	r.NextAction = "join Track 2 (OBSERVED provider-$ savings, epic #1301 rungs B/C) to complete the P&L"
}

// Render produces a compact, deterministic terminal table of the weekly trend. Rich
// visuals (sparklines, mermaid xychart) are epic #1301 rung D; this is the plain
// fallback so the package is usable on its own.
func Render(r Report) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "cache-value roll-up (Track 1, WITNESSED kernel reuse) — %s\n", r.Verdict)
	fmt.Fprintf(&sb, "  %s\n", r.Finding)
	fmt.Fprintf(&sb, "  fence: %s\n", PublishableValueFamily)
	if len(r.Buckets) == 0 {
		return sb.String()
	}
	fmt.Fprintf(&sb, "\n  %-9s  %8s  %7s  %6s  %-10s  %s\n", "week", "sessions", "m-turns", "reuse", "trend", "regime f/p/c")
	for _, b := range r.Buckets {
		thin := ""
		if b.Thin {
			thin = " (thin)"
		}
		fmt.Fprintf(&sb, "  %-9s  %8d  %7d  %5.1f%%  %-10s  %d/%d/%d%s\n",
			b.Period, b.Sessions, b.MultiTurnTurns, 100*b.RealizedReuseRatio, b.Trend,
			b.FrozenTurns, b.PartialTurns, b.ColdTurns, thin)
	}
	return sb.String()
}
