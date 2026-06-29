package modelroute

// Telemetry feedback for learned routing — per-(aspect,rule) cost/latency/quality
// outcomes (#600, epic #595).
//
// THE GAP IT FILLS. The SOTA routers (RouteLLM, NotDiamond, Unify) LEARN a
// request->model predictor from feedback: they record how each served request
// fared (cheap? fast? right?) and fit a model that picks the engine. fak's
// differentiator is doing that PER-ASPECT under a DETERMINISTIC, AUDITABLE policy
// — never a black-box predictor that silently mutates the route. This file ships
// the FEEDBACK CORPUS that a future learned policy stands on, without abandoning
// determinism:
//
//   - Outcome is the measured result of one served decision — its cost, its
//     latency, and a quality score. It is the OBSERVED half (relayed from a live
//     serve), in contrast to the WITNESSED routing decision (Route/Combine, pure).
//   - OutcomeRecord binds an Outcome to the decision it grades, keyed by the SAME
//     (aspect, rule) pair a route is taken on, plus the decision's content-address
//     digest so a recorded outcome can be replayed against the exact route it
//     describes (#615 round-trip).
//   - OutcomeJournal is the append-only sink — a journal, exactly like the
//     DecisionJournal in observe.go, so the two share their shape: append now,
//     fold later, never mutate.
//   - Aggregate is the PURE FOLD over the journal: a per-(aspect,rule) rollup of
//     mean cost / latency / quality (and the sample count). Same inputs always
//     yield the same aggregate — it is a deterministic fold, not a fitted model.
//
// WHAT IS DELIBERATELY NOT HERE (the next rung, tracked on #600): a LEARNED
// predictor that proposes a manifest rule change. The deliverable here is the
// grounded, auditable feedback corpus + its honest aggregate; the offline run
// that mines the corpus for a reviewable manifest diff is additive ON TOP of this
// sink, and gated by a human (mirroring the RSI ship-gate). Building it now would
// be a black box; the corpus must exist first.
//
// HONESTY (the load-bearing boundary). The recorded Cost/Latency/Quality are
// OBSERVED values relayed from a serve — they are NOT something modelroute
// controls or witnesses, the way the routing decision is. A decision with NO
// recorded outcome contributes NOTHING to the aggregate (it is not counted as a
// zero), so an unserved or unmeasured route never silently drags a mean down.

import (
	"sort"
	"time"
)

// Outcome is the measured result of serving one routing decision — the per-aspect
// feedback signal a learned policy would later fit on. Every field is an OBSERVED
// value relayed from a live serve, never a value modelroute witnesses:
//
//   - Cost is the rough $ the served call actually spent (the cost lens, cost.go,
//     estimates this offline; here it is the observed spend).
//   - Latency is the wall-clock the served call took.
//   - Quality is a 0..1 score for the served answer (1 == ground-truth match, a
//     judge score, a thumbs-up rate, … — the producer's chosen quality signal).
//
// The zero Outcome is a valid "all zero" measurement; a decision with NO outcome
// is represented by the ABSENCE of an OutcomeRecord, not a zero Outcome (see the
// package note), so the two are never conflated.
type Outcome struct {
	Cost    float64       `json:"cost"`       // rough $ the served call spent
	Latency time.Duration `json:"latency_ns"` // wall-clock the served call took
	Quality float64       `json:"quality"`    // 0..1 quality score for the answer
}

// AspectRuleKey is the per-(aspect,rule) key the feedback corpus aggregates on —
// the same two dimensions a route is taken on (Subject.Aspect + the matched
// Rule.Name). It is comparable so it keys a map directly; an empty Rule names the
// fail-closed default and an empty Aspect the un-aspected (whole-request) route.
type AspectRuleKey struct {
	Aspect Aspect `json:"aspect"`
	Rule   string `json:"rule"` // matched rule name; "" == fail-closed default
}

// keyOf derives the (aspect,rule) key a decision is grouped under — the decision's
// subject aspect and the rule it matched (empty rule == default), so an outcome
// keys exactly to the route it grades.
func keyOf(d Decision) AspectRuleKey {
	return AspectRuleKey{Aspect: d.Subject.Aspect, Rule: d.RuleName}
}

// OutcomeRecord binds a measured Outcome to the routing decision it grades. It
// carries the (aspect,rule) key the aggregate folds on and the decision's
// content-address Digest (#615) so a recorded outcome can be replayed against the
// exact route it describes — the corpus stays auditable, never an anonymous bag of
// numbers. A gateway/serve layer fills Outcome from the live call; the key and
// digest come from the Decision, with no live dependency in this leaf.
type OutcomeRecord struct {
	Key     AspectRuleKey `json:"key"`
	Digest  string        `json:"digest,omitempty"` // #615 content-address of the graded decision
	Outcome Outcome       `json:"outcome"`
}

// RecordOutcome builds the corpus record for a decision served under a manifest
// version, given the measured outcome. It derives the (aspect,rule) key and the
// content-address digest from the decision — everything needed to fold and to
// replay — with no live dependency. It is the outcome-side mirror of
// RecordDecision in observe.go.
func RecordOutcome(version string, d Decision, o Outcome) OutcomeRecord {
	return OutcomeRecord{
		Key:     keyOf(d),
		Digest:  d.Digest(version),
		Outcome: o,
	}
}

// OutcomeJournal is the append-only telemetry SINK: the grounded feedback corpus a
// future learned policy stands on. It mirrors DecisionJournal exactly — append
// now, fold later, never mutate — so the decision trail and the outcome trail
// share one shape. It is a plain slice (single-writer per request path); a
// concurrent caller wraps it.
type OutcomeJournal struct {
	records []OutcomeRecord
}

// Append adds an outcome record to the journal.
func (j *OutcomeJournal) Append(r OutcomeRecord) { j.records = append(j.records, r) }

// Record builds the outcome record for a decision served under version with the
// measured outcome and appends it in one call. Returns the appended record so a
// caller can forward it to a live emitter too.
func (j *OutcomeJournal) Record(version string, d Decision, o Outcome) OutcomeRecord {
	r := RecordOutcome(version, d, o)
	j.Append(r)
	return r
}

// Len reports how many outcomes the journal holds.
func (j *OutcomeJournal) Len() int { return len(j.records) }

// Records returns a copy of the journal's records (audit read; the copy keeps the
// caller from mutating the journal's backing array).
func (j *OutcomeJournal) Records() []OutcomeRecord {
	out := make([]OutcomeRecord, len(j.records))
	copy(out, j.records)
	return out
}

// AspectRuleStats is the folded feedback for one (aspect,rule) bucket: the sample
// count and the MEAN cost / latency / quality over the outcomes recorded under
// that key. The means are over the recorded outcomes ONLY — a decision with no
// outcome contributes nothing — so an unserved route never appears as a zero that
// drags a mean down. The sums are retained so two aggregates can be merged or so a
// caller can re-derive a mean at a different grouping without re-reading the
// journal.
type AspectRuleStats struct {
	Count       int           `json:"count"`
	MeanCost    float64       `json:"mean_cost"`
	MeanLatency time.Duration `json:"mean_latency_ns"`
	MeanQuality float64       `json:"mean_quality"`

	// SumCost / SumLatency / SumQuality are the running totals the means divide.
	SumCost    float64       `json:"sum_cost"`
	SumLatency time.Duration `json:"sum_latency_ns"`
	SumQuality float64       `json:"sum_quality"`
}

// Aggregate is the per-(aspect,rule) outcome rollup — a deterministic, pure fold
// over an OutcomeJournal. For each (aspect,rule) key it carries the sample count
// and the mean cost / latency / quality of the outcomes recorded under it.
type Aggregate struct {
	ByKey map[AspectRuleKey]AspectRuleStats `json:"by_key"`
	Total int                               `json:"total"`
}

// Aggregate folds the journal into the per-(aspect,rule) rollup of mean cost /
// latency / quality. It is a PURE fold: same recorded outcomes always yield the
// same Aggregate (no time, no I/O, no map-order dependence in the result — the
// means are exact sums divided by an integer count). A key with no recorded
// outcome never appears; an outcome with a zero value DOES count as a measured
// zero (it is a recorded measurement), in contrast to an absent outcome which
// contributes nothing. This is the grounded feedback corpus folded into the
// signal a learned policy would later fit on — without being that policy.
func (j *OutcomeJournal) Aggregate() Aggregate {
	type acc struct {
		count      int
		sumCost    float64
		sumLatency time.Duration
		sumQuality float64
	}
	by := make(map[AspectRuleKey]*acc, len(j.records))
	for _, r := range j.records {
		a := by[r.Key]
		if a == nil {
			a = &acc{}
			by[r.Key] = a
		}
		a.count++
		a.sumCost += r.Outcome.Cost
		a.sumLatency += r.Outcome.Latency
		a.sumQuality += r.Outcome.Quality
	}
	out := Aggregate{ByKey: make(map[AspectRuleKey]AspectRuleStats, len(by)), Total: len(j.records)}
	for k, a := range by {
		n := float64(a.count)
		out.ByKey[k] = AspectRuleStats{
			Count:       a.count,
			MeanCost:    a.sumCost / n,
			MeanLatency: time.Duration(int64(a.sumLatency) / int64(a.count)),
			MeanQuality: a.sumQuality / n,
			SumCost:     a.sumCost,
			SumLatency:  a.sumLatency,
			SumQuality:  a.sumQuality,
		}
	}
	return out
}

// SortedKeys returns the aggregate's (aspect,rule) keys in a stable order —
// aspect ascending, then rule ascending — so a CLI dump or a comparison over the
// rollup is deterministic across runs regardless of Go's map iteration order.
func (a Aggregate) SortedKeys() []AspectRuleKey {
	out := make([]AspectRuleKey, 0, len(a.ByKey))
	for k := range a.ByKey {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Aspect != out[j].Aspect {
			return out[i].Aspect < out[j].Aspect
		}
		return out[i].Rule < out[j].Rule
	})
	return out
}
