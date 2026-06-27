package modelroute

// Routing observability — the modelroute-side decision journal + rollup (#603,
// epic #595).
//
// Make every routing decision observable AFTER the fact: a gateway exporter
// copies this package's rollup into fak_gateway_* metrics, and the decision
// journal records each route so it is auditable. The determinism/verifiability
// axis that sets fak apart from learned-predictor routers needs an OBSERVABLE
// trail, not just a deterministic function.
//
// LANE BOUNDARY (#603): this file ships the PURE, in-leaf half — a
// DecisionRecord, an append-only DecisionJournal, and a Counts() rollup over the
// per-aspect / per-rule / per-strategy distribution a gateway exporter reads. The
// LIVE /metrics emit (extending internal/gateway/http.go + fak_gateway_*) is
// DEFERRED: that file is peer-broken and out of lane. The exporter is a thin
// copy-out of Counts() — designed here so the gateway side is a projection, never
// new logic.

import (
	"sort"
	"time"
)

// Strategy is the coarse routing strategy a Decision took — the dimension a
// dashboard buckets on ("which strategy served this request?"). It is derived
// from the chosen Plan, not stored on it, so it never drifts from the plan.
type Strategy string

const (
	// StrategyDefault is the fail-closed default plan (no rule matched).
	StrategyDefault Strategy = "default"
	// StrategySingle is a single-model pick (the SOTA router shape).
	StrategySingle Strategy = "single"
	// StrategyEnsemble is a multi-model ensemble folded by a Reduction.
	StrategyEnsemble Strategy = "ensemble"
)

// StrategyOf classifies a Decision into its routing strategy: an unmatched
// decision is StrategyDefault; otherwise an ensemble (len(Members) > 1) is
// StrategyEnsemble and a single member is StrategySingle.
func StrategyOf(d Decision) Strategy {
	if !d.Matched {
		return StrategyDefault
	}
	if d.Plan.IsEnsemble() {
		return StrategyEnsemble
	}
	return StrategySingle
}

// DecisionRecord is the observable, after-the-fact record of one routing
// decision: the matched rule (empty == default), the strategy, the ensemble
// member count, the scout-call count (1 when the plan named a Scout, else 0), the
// routing overhead, and the content-address digest that binds it to evidence
// (#615). A gateway exporter reads these fields straight into fak_gateway_*
// labels/observations; the journal stores them for audit.
type DecisionRecord struct {
	RuleName   string        `json:"rule"`             // matched rule; "" == fail-closed default
	Strategy   Strategy      `json:"strategy"`         // default | single | ensemble
	Aspect     Aspect        `json:"aspect,omitempty"` // the subject's aspect (granularity bucket)
	Members    int           `json:"members"`          // ensemble member count (>=1)
	ScoutCalls int           `json:"scout_calls"`      // 1 when the plan named a scout, else 0
	Overhead   time.Duration `json:"overhead_ns"`      // routing overhead the caller measured
	Digest     string        `json:"digest,omitempty"` // #615 content-address of the decision
}

// RecordDecision builds the observable record for a decision taken under a
// manifest version, given the routing overhead the caller measured. It derives
// the strategy, member count, scout-call count, and the #615 content-address
// digest — everything a /metrics exporter and an audit journal need, with no live
// dependency. overhead is the time the route itself took (the caller times
// Route); pass 0 when not measured.
func RecordDecision(version string, d Decision, overhead time.Duration) DecisionRecord {
	scout := 0
	if d.Plan.Scout != "" {
		scout = 1
	}
	return DecisionRecord{
		RuleName:   d.RuleName,
		Strategy:   StrategyOf(d),
		Aspect:     d.Subject.Aspect,
		Members:    len(d.Plan.Members),
		ScoutCalls: scout,
		Overhead:   overhead,
		Digest:     d.Digest(version),
	}
}

// DecisionJournal is an append-only record of routing decisions — the in-leaf
// decision journal #603 names. A gateway exporter folds Counts() into
// fak_gateway_* periodically; an audit reads Records() to replay routes. It is a
// plain slice (single-writer per request path); a concurrent caller wraps it.
type DecisionJournal struct {
	records []DecisionRecord
}

// Append adds a decision record to the journal.
func (j *DecisionJournal) Append(r DecisionRecord) { j.records = append(j.records, r) }

// Record routes-and-records in one call: derive the DecisionRecord for d under
// version (with measured overhead) and append it. Returns the appended record so
// a caller can forward it to a live emitter too.
func (j *DecisionJournal) Record(version string, d Decision, overhead time.Duration) DecisionRecord {
	r := RecordDecision(version, d, overhead)
	j.Append(r)
	return r
}

// Len reports how many decisions the journal holds.
func (j *DecisionJournal) Len() int { return len(j.records) }

// Records returns a copy of the journal's records (audit read; the copy keeps the
// caller from mutating the journal's backing array).
func (j *DecisionJournal) Records() []DecisionRecord {
	out := make([]DecisionRecord, len(j.records))
	copy(out, j.records)
	return out
}

// Counts is the per-dimension rollup a gateway exporter copies into fak_gateway_*
// metrics: how many decisions fell to each rule, each strategy, and each aspect,
// plus the ensemble/scout totals and the summed routing overhead. Every map is
// keyed by a string so the exporter emits one labeled series per key with no
// modelroute-side knowledge of the metric names.
type Counts struct {
	Total         int            `json:"total"`
	ByRule        map[string]int `json:"by_rule"`     // "" rolled up as "(default)"
	ByStrategy    map[string]int `json:"by_strategy"` // default | single | ensemble
	ByAspect      map[string]int `json:"by_aspect"`   // "" rolled up as "(none)"
	EnsembleHits  int            `json:"ensemble_hits"`
	ScoutCalls    int            `json:"scout_calls"`
	TotalOverhead time.Duration  `json:"total_overhead_ns"`
}

// Counts folds the journal into the per-dimension rollup. Pure over the recorded
// decisions — the gateway exporter's only job is to copy these integers into its
// metric families, so the modelroute side owns the WHAT-to-count and the gateway
// side owns the metric NAMES.
func (j *DecisionJournal) Counts() Counts {
	c := Counts{
		ByRule:     map[string]int{},
		ByStrategy: map[string]int{},
		ByAspect:   map[string]int{},
	}
	for _, r := range j.records {
		c.Total++
		rule := r.RuleName
		if rule == "" {
			rule = "(default)"
		}
		c.ByRule[rule]++
		c.ByStrategy[string(r.Strategy)]++
		asp := string(r.Aspect)
		if asp == "" {
			asp = "(none)"
		}
		c.ByAspect[asp]++
		if r.Strategy == StrategyEnsemble {
			c.EnsembleHits++
		}
		c.ScoutCalls += r.ScoutCalls
		c.TotalOverhead += r.Overhead
	}
	return c
}

// SortedRules returns the rule names in the rollup in descending count order
// (ties broken by name) — the deterministic order a top-N exporter or a CLI
// dump emits, so two runs over the same journal print the same table.
func (c Counts) SortedRules() []string {
	out := make([]string, 0, len(c.ByRule))
	for k := range c.ByRule {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool {
		if c.ByRule[out[i]] != c.ByRule[out[j]] {
			return c.ByRule[out[i]] > c.ByRule[out[j]]
		}
		return out[i] < out[j]
	})
	return out
}
