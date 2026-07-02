// hooklat.go — the guard-hook latency rollup (issue #1993): fold the DOS
// hook-observation stream's per-observation latency_ms into percentiles and judge
// the tail against a declared budget.
//
// Guard adjudication fires a pretool + posttool hook on every tool call, and each
// observation row records its own latency_ms — but nothing rolled the stream up: no
// p50/p99, no budget, no alarm. At fleet scale the hook tax is the dominant
// guard-imposed wall-clock cost per tool call (~85ms mean / ~175ms p99 measured on
// the 2026-07-01 guard-audit run) and it was invisible in aggregate. This file is
// the missing fold: parse → percentile rollup → budget verdict.
//
// Three honesty rules, mirroring the rest of the self-tax plane (epic #1147):
//
//   - The breach names ONE closed-vocabulary token — GateLatencyRegression
//     ("GATE_LATENCY_REGRESSION"), already declared in dos.toml — never free-text
//     prose, so an alarm is emittable, verifiable, and refusable like any other
//     structured refusal.
//
//   - The alarm abstains on a thin sample (issue #1993's own caveat: its evidence
//     was n=13). Fewer than MinHookAlarmSamples observations report as Thin — the
//     percentiles still print, the alarm does not fire. A single slow spike must
//     accumulate into a trustworthy tail before it may red the plane.
//
//   - The parser is tolerant of a shared stream: rows of a foreign schema family,
//     rows without a measured latency, and non-JSON lines are COUNTED as skipped
//     and never fatal, so one malformed row cannot hide the whole rollup.
//
// The budget here is a v0.1 DECLARED calibration ceiling (generous, per the
// overheadbudget.go doctrine), not a measured p99: it exists so a GROSS regression
// reads back as a structured breach while normal jitter stays OK. Tightening it
// toward a measured fleet p99 is the #2073 follow-on, gated on this fold existing.
package turntaxmeter

import (
	"bufio"
	"encoding/json"
	"io"
	"sort"
	"time"
)

// GateLatencyRegression is the closed-vocabulary alarm token a hook-latency budget
// breach names. It MUST stay byte-identical to the dos.toml
// [reasons.GATE_LATENCY_REGRESSION] declaration so the token this fold emits is the
// same one `dos check-reason` verifies and the loop routes to a replan.
const GateLatencyRegression = "GATE_LATENCY_REGRESSION"

// DefaultHookP99BudgetMS is the v0.1 declared per-observation p99 ceiling in
// milliseconds. Calibration: the #1993 evidence measured p99≈175ms on a live
// guard-audit run, so 250ms is a generous "gross regression only" envelope — a
// breach means the hook path got materially slower than the day it was measured,
// not that one call jittered.
const DefaultHookP99BudgetMS = 250

// MinHookAlarmSamples is the smallest sample count the p99 alarm may fire on.
// Below it the verdict reports Thin and abstains: a p99 over a handful of rows is
// a spike detector, not a tail, and #1993 explicitly asks for an accumulated
// trustworthy rollup rather than a single-spike alarm.
const MinHookAlarmSamples = 8

// HookObservation is one measured hook firing from the hook-observation stream:
// which hook (pretool/posttool), what the guard decided (outcome), and how long the
// observation took end to end.
type HookObservation struct {
	// Verb is the hook that fired — "pretool" | "posttool" in the v1 stream.
	Verb string
	// Outcome is the hook's decision label (e.g. "passthrough"). Carried for
	// per-outcome splits by callers; the latency fold does not branch on it.
	Outcome string
	// LatencyMS is the row's own measured wall-clock in milliseconds.
	LatencyMS float64
	// At is the row's timestamp; the zero time when the row carried none. A zero
	// At row still folds into an all-time rollup but is excluded by any
	// since-window filter (it cannot prove it is inside the window).
	At time.Time
}

// hookObservationRow is the wire shape of one hook-observation v1 JSONL row —
// only the fields this fold consumes. LatencyMS is a pointer so "no latency
// recorded" is distinguishable from a measured 0ms and skipped rather than folded
// as a free zero that would drag the percentiles down.
type hookObservationRow struct {
	Schema struct {
		Family  string `json:"family"`
		Version int    `json:"version"`
	} `json:"schema"`
	Verb      string   `json:"verb"`
	Outcome   string   `json:"outcome"`
	LatencyMS *float64 `json:"latency_ms"`
	TS        string   `json:"ts"`
}

// ParseHookObservations folds a hook-observation JSONL stream into observations.
// Tolerant by contract: a non-JSON line, a row of a foreign schema family, or a row
// with no measured latency_ms increments skipped and is never fatal — .dos/metrics
// streams are shared append-only files and one bad row must not hide the rollup.
// The only returned error is a real read failure from r.
func ParseHookObservations(r io.Reader) (obs []HookObservation, skipped int, err error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var row hookObservationRow
		if json.Unmarshal(line, &row) != nil || row.Schema.Family != "hook-observation" || row.LatencyMS == nil {
			skipped++
			continue
		}
		at, _ := time.Parse(time.RFC3339, row.TS) // zero time on absent/bad ts, by contract
		obs = append(obs, HookObservation{
			Verb:      row.Verb,
			Outcome:   row.Outcome,
			LatencyMS: *row.LatencyMS,
			At:        at,
		})
	}
	return obs, skipped, sc.Err()
}

// FilterHookObservationsSince keeps only observations provably at-or-after cutoff.
// A zero-At row (no timestamp) is dropped: it cannot witness being inside the
// window, and a session-scoped exit-summary line must not smuggle in another
// session's rows. A zero cutoff keeps everything (the all-time fold).
func FilterHookObservationsSince(obs []HookObservation, cutoff time.Time) []HookObservation {
	if cutoff.IsZero() {
		return obs
	}
	kept := make([]HookObservation, 0, len(obs))
	for _, o := range obs {
		if !o.At.IsZero() && !o.At.Before(cutoff) {
			kept = append(kept, o)
		}
	}
	return kept
}

// HookLatencyStats is the percentile rollup for one verb (or all verbs folded,
// Verb == ""). Percentiles are nearest-rank over the observed samples — with small
// n the p99 IS the max, which is the honest reading of a thin tail.
type HookLatencyStats struct {
	Verb   string
	Count  int
	MeanMS float64
	P50MS  float64
	P90MS  float64
	P99MS  float64
	MaxMS  float64
}

// HookLatencyRollup is the full fold: the all-verbs total (the tail the budget
// judges — pretool and posttool BOTH tax the same tool call) plus a per-verb split
// sorted by verb name so a slow posttool journal write is attributable.
type HookLatencyRollup struct {
	Total  HookLatencyStats
	ByVerb []HookLatencyStats
}

// FoldHookLatency computes the percentile rollup over the observations. Zero
// observations return a zero rollup (Count 0) rather than an error: an empty
// stream is a valid "nothing observed" fact.
func FoldHookLatency(obs []HookObservation) HookLatencyRollup {
	byVerb := map[string][]float64{}
	all := make([]float64, 0, len(obs))
	for _, o := range obs {
		all = append(all, o.LatencyMS)
		byVerb[o.Verb] = append(byVerb[o.Verb], o.LatencyMS)
	}
	r := HookLatencyRollup{Total: foldLatencySamples("", all)}
	verbs := make([]string, 0, len(byVerb))
	for v := range byVerb {
		verbs = append(verbs, v)
	}
	sort.Strings(verbs)
	for _, v := range verbs {
		r.ByVerb = append(r.ByVerb, foldLatencySamples(v, byVerb[v]))
	}
	return r
}

// foldLatencySamples reduces one sample set to its stats row.
func foldLatencySamples(verb string, ms []float64) HookLatencyStats {
	s := HookLatencyStats{Verb: verb, Count: len(ms)}
	if len(ms) == 0 {
		return s
	}
	sorted := append([]float64(nil), ms...)
	sort.Float64s(sorted)
	var sum float64
	for _, v := range sorted {
		sum += v
	}
	s.MeanMS = sum / float64(len(sorted))
	s.P50MS = nearestRank(sorted, 50)
	s.P90MS = nearestRank(sorted, 90)
	s.P99MS = nearestRank(sorted, 99)
	s.MaxMS = sorted[len(sorted)-1]
	return s
}

// nearestRank is the nearest-rank percentile over an ascending-sorted sample:
// the smallest value with at least p% of samples at or below it. Deterministic
// and interpolation-free, so a reported p99 is always a REAL observed latency.
func nearestRank(sorted []float64, p int) float64 {
	rank := (p*len(sorted) + 99) / 100 // ceil(p/100 * n) in integer arithmetic
	if rank < 1 {
		rank = 1
	}
	return sorted[rank-1]
}

// HookLatencyVerdict is the budget judgment over a rollup's total tail.
type HookLatencyVerdict struct {
	// OK is false only on a fired alarm: enough samples AND the p99 over budget.
	OK bool
	// Thin marks an abstained alarm — fewer than MinHookAlarmSamples observations.
	// The stats are still real; the tail is just not trustworthy enough to red on.
	Thin bool
	// Reason is GateLatencyRegression when the alarm fired, "" otherwise — the
	// closed-vocabulary discipline: a breach is a token, never prose.
	Reason        string
	BudgetP99MS   float64
	ObservedP99MS float64
	Count         int
}

// JudgeHookLatency reads the total rollup back against a p99 budget in
// milliseconds. A budgetP99MS of 0 or below means "no budget declared" and, per
// the plane's fail-open coverage contract (see CheckSpan), can never breach.
func JudgeHookLatency(total HookLatencyStats, budgetP99MS float64) HookLatencyVerdict {
	v := HookLatencyVerdict{
		OK:            true,
		BudgetP99MS:   budgetP99MS,
		ObservedP99MS: total.P99MS,
		Count:         total.Count,
	}
	if total.Count < MinHookAlarmSamples {
		v.Thin = true
		return v
	}
	if budgetP99MS > 0 && total.P99MS > budgetP99MS {
		v.OK = false
		v.Reason = GateLatencyRegression
	}
	return v
}
