package scorecard

import (
	"fmt"
	"sort"
	"time"
)

// --- Delivery-flow fold ---
//
// The fleet measures how much work is OPEN (an issue count) but not how work FLOWS: how long
// a ticket waits before anyone starts it, how long it then takes to close, how much work is
// in flight at once, and how much of that in-flight work is even VISIBLE to a dashboard. A
// backlog count answers "how much is left"; it cannot distinguish a lane shipping steadily
// from a lane that has started everything and finished nothing. Those two read identically as
// an issue count and could not be more different to work in.
//
// This fold is the missing flow number. Like the cache-health fold it is PURE over
// caller-supplied facts and imports nothing but fmt/sort/time, so it stays deterministic and
// unit-testable with fixtures: it reads no git, no gh, and no clock. time is used only as a
// value type (durations the caller already measured) -- there is no time.Now() here, which is
// what keeps two clones at one commit scoring identically (the determinism law).
//
// The five families map one-to-one onto rules the repo already states, so each is retired by
// changing reality rather than by weakening the detector:
//
//   - start_latency    -- how long an issue waits between opened and first touched. The
//                         queue-time nobody sees; a ticket sitting unstarted is the cheapest
//                         work to lose track of.
//   - cycle_time       -- how long from first touch to closed. Together with start_latency
//                         this is the full "opened -> started -> closed" span the operator
//                         asked to be able to read.
//   - wip_headroom     -- concurrent in-flight objectives against the WIP cap. This is the
//                         same discipline AGENTS.md's FOCUS_WIP_SATURATED rung enforces at
//                         dispatch time ("converge before broadening"), read as a health.
//   - wip_visibility   -- the fraction of in-flight work that is COMMITTED rather than sitting
//                         as uncommitted local edits. This is the family with no existing
//                         instrument at all: an issue tracker cannot see a working tree, so
//                         half-applied work is invisible WIP -- it is not on any dashboard, and
//                         a reader of the backlog count will conclude the opposite of the truth.
//   - commit_atomicity -- the fraction of closures that landed as exactly ONE commit, the
//                         "one issue, one commit; one commit, one leaf" atom AGENTS.md binds.
//                         A closure smeared across land -> patch -> patch is churn, and churn
//                         is what makes cycle_time unreadable.
//
// Every family is INDIVIDUALLY RETIRABLE: its health is a standalone 0..1 fraction, its KPI a
// standalone builder, and its defect names the real thing to change.

// FlowSchema tags the delivery-flow card, distinct from every other schema so a roster or
// control-pane consumer reads this card apart from the cache/value family.
const FlowSchema = "fak-delivery-flow-scorecard/1"

// FlowDebtKey is the corpus debt integer this card writes: the count of flow families below
// the pass line. Exported so a CLI shell can name it in the shared --json/--compare tail.
const FlowDebtKey = "delivery_flow_debt"

// FlowPassLine is the 0..1 health floor below which a flow family books a defect. It is
// deliberately the same conservative starting floor CacheHealthPassLine uses -- a knob an
// operator TIGHTENS over time, not a per-family tuned threshold. The worst-first worklist
// orders every scored family regardless of this floor; the floor only decides what is debt.
const FlowPassLine = 0.5

// The canonical, ordered flow-family component keys. The order is the deterministic tie-break
// when two families carry equal health AND the enumeration a consumer iterates to address each
// family standalone. It follows the life of one ticket: wait, work, then the three ways a fleet
// loses track of work in flight.
const (
	FlowStartLatency    = "start_latency"    // opened -> first touched
	FlowCycleTime       = "cycle_time"       // first touched -> closed
	FlowWIPHeadroom     = "wip_headroom"     // concurrent in-flight objectives vs the WIP cap
	FlowWIPVisibility   = "wip_visibility"   // in-flight work that is committed vs uncommitted
	FlowCommitAtomicity = "commit_atomicity" // closures that landed as exactly one commit
)

// FlowComponents is the canonical ordered family set. len == the fixed denominator
// components_total; a family is folded only when the caller supplies evidence for it.
var FlowComponents = []string{
	FlowStartLatency,
	FlowCycleTime,
	FlowWIPHeadroom,
	FlowWIPVisibility,
	FlowCommitAtomicity,
}

// flowLabels give each family a human phrase for the worklist detail + the retire hint.
var flowLabels = map[string]string{
	FlowStartLatency:    "opened-to-started latency",
	FlowCycleTime:       "started-to-closed cycle time",
	FlowWIPHeadroom:     "concurrent WIP headroom",
	FlowWIPVisibility:   "in-flight work visibility",
	FlowCommitAtomicity: "one-issue-one-commit atomicity",
}

// flowRetire names, per family, the real change that retires its defect. Kept beside the
// labels so the defect string can never degrade into "raise the number".
var flowRetire = map[string]string{
	FlowStartLatency:    "start fewer things sooner -- pull the oldest unstarted ticket before opening a new one",
	FlowCycleTime:       "finish work already started before starting more; split tickets that cannot close in one pass",
	FlowWIPHeadroom:     "converge before broadening -- close in-flight objectives down to the cap instead of raising it",
	FlowWIPVisibility:   "commit or gate in-flight edits (a build-tagged WIP file is visible; an uncommitted one is not)",
	FlowCommitAtomicity: "green the acceptance criteria before the first commit so a closure lands once, not land-then-patch",
}

// FlowFacts carries each flow family's already-derived 0..1 health (1.0 == healthy). Every
// field is a POINTER so "no evidence this window" (nil) is never conflated with a measured
// 0.0 -- a nil family is EXCLUDED from the fold. The caller fills these from the existing
// signals via the mapper helpers below; this scoring core reads no ledger and no repository.
type FlowFacts struct {
	StartLatency    *float64 // see DurationHealth(observed p-value, target)
	CycleTime       *float64 // see DurationHealth(observed p-value, target)
	WIPHeadroom     *float64 // see WIPHeadroomHealth(inFlight, cap)
	WIPVisibility   *float64 // see VisibilityHealth(committed, uncommitted)
	CommitAtomicity *float64 // see AtomicityHealth(singleCommitClosures, closures)
}

// componentHealth returns the caller-supplied health for one family key, or nil when absent.
func (f FlowFacts) componentHealth(component string) *float64 {
	switch component {
	case FlowStartLatency:
		return f.StartLatency
	case FlowCycleTime:
		return f.CycleTime
	case FlowWIPHeadroom:
		return f.WIPHeadroom
	case FlowWIPVisibility:
		return f.WIPVisibility
	case FlowCommitAtomicity:
		return f.CommitAtomicity
	}
	return nil
}

// DurationPercentile is the nearest-rank percentile of a duration sample, the shape a latency
// or cycle-time family is measured at ("the p85 ticket closed within X"). p is a fraction in
// 0..1; it is clamped, so p<=0 reads the fastest sample and p>=1 the slowest. An empty sample
// returns 0, which the DurationHealth mapper then reads as "no evidence" rather than as a
// perfect zero-second cycle time.
//
// Nearest-rank (rather than an interpolating percentile) is deliberate: it always returns a
// duration that was actually OBSERVED, so a reported p85 is a real ticket a human can go read
// rather than a synthetic midpoint between two.
//
// The input slice is not mutated -- it is sorted through a copy, so a caller can hand the same
// sample to several percentiles and get the same answers in any order.
func DurationPercentile(samples []time.Duration, p float64) time.Duration {
	if len(samples) == 0 {
		return 0
	}
	sorted := make([]time.Duration, len(samples))
	copy(sorted, samples)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	if p <= 0 {
		return sorted[0]
	}
	if p >= 1 {
		return sorted[len(sorted)-1]
	}
	// Nearest-rank: the smallest sample at or above the p-th position, 1-indexed.
	idx := int(float64(len(sorted))*p + 0.999999)
	if idx < 1 {
		idx = 1
	}
	if idx > len(sorted) {
		idx = len(sorted)
	}
	return sorted[idx-1]
}

// DurationHealth maps an observed duration against a target into a 0..1 family health:
// target/observed, clamped, so hitting or beating the target reads 1.0 and taking twice the
// target reads 0.5. A non-positive observed or target is nil (no evidence) -- notably an empty
// DurationPercentile result, so a window with no closed tickets does not score a perfect 0s.
//
// The ratio form (rather than a fixed band) is what lets one mapper serve both latency
// families at whatever target a lane sets, and it degrades smoothly instead of cliff-edging at
// a threshold: a lane drifting from 1.0x to 1.4x the target sees the number move every step.
func DurationHealth(observed, target time.Duration) *float64 {
	if observed <= 0 || target <= 0 {
		return nil
	}
	h := clamp01(float64(target) / float64(observed))
	return &h
}

// WIPHeadroomHealth maps concurrent in-flight objectives against the WIP cap into a 0..1
// family health: at or under the cap reads 1.0, and over it decays as cap/inFlight, so twice
// the cap reads 0.5. A non-positive cap is nil (no cap declared == no evidence); a negative
// inFlight is nil (not a measurement); zero in flight is a healthy 1.0.
func WIPHeadroomHealth(inFlight, cap int) *float64 {
	if cap <= 0 || inFlight < 0 {
		return nil
	}
	if inFlight <= cap {
		h := 1.0
		return &h
	}
	h := clamp01(float64(cap) / float64(inFlight))
	return &h
}

// VisibilityHealth maps in-flight work into a 0..1 family health: the fraction that is
// COMMITTED (and so visible to any git-reading instrument) rather than sitting as uncommitted
// local edits. No in-flight work at all (committed+uncommitted == 0) is nil (no evidence).
// Negative counts are nil (not a measurement).
//
// This is the family that exists because an issue tracker structurally cannot see a working
// tree: uncommitted work is not merely unfinished, it is UNCOUNTED, so every other flow number
// is computed over a population that silently excludes it.
func VisibilityHealth(committed, uncommitted int) *float64 {
	if committed < 0 || uncommitted < 0 {
		return nil
	}
	total := committed + uncommitted
	if total == 0 {
		return nil
	}
	h := clamp01(float64(committed) / float64(total))
	return &h
}

// AtomicityHealth maps closures into a 0..1 family health: the fraction that landed as exactly
// ONE commit, the atom AGENTS.md binds ("one issue, one commit; one commit, one leaf"). No
// closures this window is nil (no evidence). Negative counts, or more single-commit closures
// than closures, are nil (not a coherent measurement).
func AtomicityHealth(singleCommitClosures, closures int) *float64 {
	if closures <= 0 || singleCommitClosures < 0 || singleCommitClosures > closures {
		return nil
	}
	h := clamp01(float64(singleCommitClosures) / float64(closures))
	return &h
}

// FlowRow is one flow family's row in the worst-first worklist: its key, its 0..1 health, the
// pass line, whether it is in debt (below the pass line), and a human detail. The worklist is
// EVERY scored family, sorted worst-first (lowest health first, FlowComponents order breaking
// ties), so a human reads the weakest family at a glance and a gate ratchets on the debt.
type FlowRow struct {
	Component string  `json:"component"`
	Health    float64 `json:"health"`
	PassLine  float64 `json:"pass_line"`
	InDebt    bool    `json:"in_debt"`
	Detail    string  `json:"detail"`
}

// flowDetail renders one family's worklist/KPI detail line.
func flowDetail(component string, health float64) string {
	status := "clears pass line"
	if health+gateEps < FlowPassLine {
		status = "BELOW pass line"
	}
	return fmt.Sprintf("%s health %.3f (pass line %.2f, %s)", flowLabels[component], health, FlowPassLine, status)
}

// Flow is the pure core: it folds the present flow-family healths into the single 0..1
// delivery-flow number (the mean of the families that HAVE evidence) and the worst-first
// worklist (every scored family, lowest health first, canonical order breaking ties). present
// is the count of families folded; when it is 0 the number is 1.0 (nothing is known-unhealthy)
// and the worklist empty. A degraded family lowers the number AND floats to the top.
func Flow(f FlowFacts) (number float64, present int, worklist []FlowRow) {
	rank := map[string]int{}
	for i, c := range FlowComponents {
		rank[c] = i
	}
	var sum float64
	worklist = make([]FlowRow, 0, len(FlowComponents))
	for _, c := range FlowComponents {
		h := f.componentHealth(c)
		if h == nil {
			continue
		}
		v := clamp01(*h)
		sum += v
		present++
		worklist = append(worklist, FlowRow{
			Component: c,
			Health:    Round3(v),
			PassLine:  FlowPassLine,
			InDebt:    v+gateEps < FlowPassLine,
			Detail:    flowDetail(c, v),
		})
	}
	sort.SliceStable(worklist, func(i, j int) bool {
		if worklist[i].Health != worklist[j].Health {
			return worklist[i].Health < worklist[j].Health
		}
		return rank[worklist[i].Component] < rank[worklist[j].Component]
	})
	if present == 0 {
		return 1, 0, worklist
	}
	return sum / float64(present), present, worklist
}

// flowKPI builds one family KPI. Score is 100*health so the Fold composite (mean of scores) is
// exactly 100*number; PassLine feeds the unbounded pressure layer (deficit below the health
// bar). A family below the pass line owns exactly one Defect, so Fold's debt is the count of
// below-pass-line families and ok flips iff any family is in debt. The defect is prefixed with
// the component id so per-row debt stays recoverable from the joined reason string.
func flowKPI(component string, health float64) KPI {
	v := clamp01(health)
	k := KPI{
		Key:      component,
		Group:    "delivery_flow",
		Score:    100 * v,
		PassLine: 100 * FlowPassLine,
		Detail:   flowDetail(component, v),
	}
	if v+gateEps < FlowPassLine {
		k.Defects = []string{fmt.Sprintf("%s: %s health %.3f < %.2f pass line -- %s", component, flowLabels[component], v, FlowPassLine, flowRetire[component])}
	}
	return k
}

// flowKPIs builds one KPI per family that has evidence, in canonical order. With no evidence
// it returns a single healthy INSUFFICIENT KPI so the fold's value stays a coherent 1.0
// (nothing is known-unhealthy) instead of collapsing an empty slice to a spurious 0/F.
func flowKPIs(f FlowFacts) []KPI {
	kpis := make([]KPI, 0, len(FlowComponents))
	for _, c := range FlowComponents {
		if h := f.componentHealth(c); h != nil {
			kpis = append(kpis, flowKPI(c, *h))
		}
	}
	if len(kpis) == 0 {
		return []KPI{{
			Key:      "delivery_flow_evidence",
			Group:    "delivery_flow",
			Score:    100,
			PassLine: 100 * FlowPassLine,
			Detail:   "no flow-family evidence this window -- nothing to fold (INSUFFICIENT); delivery_flow defaults to 1.0",
		}}
	}
	return kpis
}

// ComposeFlow folds the flow families into the delivery-flow control-pane payload (symmetric
// with ComposeCacheHealth). corpus["delivery_flow"] is the 0..1 headline (identical to
// corpus.value by construction); corpus["delivery_flow_worklist"] is the worst-first family
// order; corpus[FlowDebtKey] is the count of families below the pass line; ok == (debt == 0).
// GradeStd is used because this is an OPERATIONAL health card, not a provenance-honesty card.
func ComposeFlow(f FlowFacts) Payload {
	number, present, worklist := Flow(f)
	if worklist == nil {
		worklist = []FlowRow{}
	}
	extra := map[string]any{
		"delivery_flow":          Round3(number),
		"delivery_flow_worklist": worklist,
		"components_present":     present,
		"components_total":       len(FlowComponents),
		"pass_line":              FlowPassLine,
	}
	return Fold(FlowSchema, flowKPIs(f), FlowDebtKey, nil, Messages{
		Finding:         "delivery flow carries debt: a flow family fell below the health pass line -- work the worst-first worklist",
		FindingClean:    "delivery flow clean: every flow family with evidence clears the health pass line",
		NextAction:      "raise the worst-first family: start latency / cycle time / WIP headroom / in-flight visibility / one-issue-one-commit atomicity",
		NextActionClean: "hold the line; keep every flow family above the pass line and tighten the ratchet",
		Grade:           GradeStd,
		ExtraCorpus:     extra,
	})
}
