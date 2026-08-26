package ultracodebench

import (
	"fmt"
	"strings"
	"time"
)

// ScopedPrefixVerdict is the conservative operator decision for micro-context scoping.
type ScopedPrefixVerdict string

const (
	ScopedPrefixEnable  ScopedPrefixVerdict = "ENABLE"
	ScopedPrefixHold    ScopedPrefixVerdict = "HOLD"
	ScopedPrefixDisable ScopedPrefixVerdict = "DISABLE"
	ScopedPrefixAbstain ScopedPrefixVerdict = "ABSTAIN"
)

// ScopedPrefixEnvelope bounds a decision to the workload that produced its evidence.
type ScopedPrefixEnvelope struct {
	Model, Runtime, Tokenizer, Task, WarmthCondition, CampaignVersion string
}

// ScopedPrefixRow is one independently replayable comparison. Avoided-work effects are
// positive when the treatment did less authoritative work than its control.
type ScopedPrefixRow struct {
	Name                 string
	AcceptedOutcomeEqual bool
	ScopeAvoided         float64
	PrefixReadAvoided    float64
	Uncertainty          float64
	NetOverhead          float64
	AuthoritativeMetric  bool
	NegativeControlPass  bool
	SourcesSeparated     bool
	ResetOccurred        bool
	SourceReceipt        string
}

// ScopedPrefixEvidence contains operator evidence, not an attribution summary.
type ScopedPrefixEvidence struct {
	Envelope              ScopedPrefixEnvelope
	Rows                  []ScopedPrefixRow
	IndependentWitness    string
	ObservedAt, ExpiresAt time.Time
}

// ScopedPrefixReport explains a local decision. It never changes the runtime default.
type ScopedPrefixReport struct {
	Verdict                ScopedPrefixVerdict
	Envelope               ScopedPrefixEnvelope
	ContradictoryRows      []string
	SmallestNextExperiment string
	Reason                 string
	RuntimeDefaultChanged  bool
	Rollback               string
	PromotionEvidence      string
	DemotionEvidence       string
	InvalidatingAssumption string
	ScopeShare             float64
	NetAvoided             float64
}

// EvaluateScopedPrefixEvidence turns witnessed factorial evidence into a bounded decision.
func EvaluateScopedPrefixEvidence(e ScopedPrefixEvidence, now time.Time) ScopedPrefixReport {
	r := ScopedPrefixReport{
		Verdict: ScopedPrefixAbstain, Envelope: e.Envelope,
		Rollback:               "leave scoping off, or disable the workload-local gate and replay the full-context control",
		PromotionEvidence:      "promote toward gen/now after ENABLE repeats on operator data inside this named envelope",
		DemotionEvidence:       "demote or retire when negative controls fail, accepted outcomes diverge, or net avoided work is not positive",
		InvalidatingAssumption: "the named model, runtime, tokenizer, task, cache posture, and campaign remain representative",
	}
	missing := missingEnvelope(e.Envelope)
	if e.IndependentWitness == "" || len(e.Rows) == 0 || len(missing) > 0 {
		r.Reason = "independently witnessed row evidence and a complete envelope are required; an attribution summary is insufficient"
		r.SmallestNextExperiment = "capture independently witnessed factorial rows with authoritative telemetry for " + strings.Join(missing, ", ")
		return r
	}
	if e.ObservedAt.IsZero() || e.ExpiresAt.IsZero() || !now.Before(e.ExpiresAt) || now.Before(e.ObservedAt) {
		r.Reason = "evidence is expired or lacks a valid observation window"
		r.SmallestNextExperiment = "replay the smallest factorial cell set inside the named envelope"
		return r
	}

	inconclusive := false
	var scopeTotal, prefixTotal, netTotal float64
	for _, row := range e.Rows {
		if !row.SourcesSeparated {
			r.ContradictoryRows = append(r.ContradictoryRows, row.Name)
			r.Reason = "scope and prefix attribution are not disjoint"
			r.SmallestNextExperiment = "replay row " + row.Name + " with disjoint token attribution"
			return r
		}
		if row.ResetOccurred {
			r.ContradictoryRows = append(r.ContradictoryRows, row.Name)
			r.Reason = "cache reset invalidates the matched comparison"
			r.SmallestNextExperiment = "replay row " + row.Name + " without a cache reset between matched arms"
			return r
		}
		if row.SourceReceipt == "" || !row.AuthoritativeMetric || !row.AcceptedOutcomeEqual {
			r.Verdict = ScopedPrefixAbstain
			r.ContradictoryRows = append(r.ContradictoryRows, row.Name)
			r.Reason = "missing authoritative telemetry, source receipt, or equal accepted outcome"
			r.SmallestNextExperiment = "replay row " + row.Name + " with equal outcomes and authoritative telemetry"
			return r
		}
		if !row.NegativeControlPass {
			r.ContradictoryRows = append(r.ContradictoryRows, row.Name)
			r.Verdict = ScopedPrefixDisable
			r.Reason = "a negative control failed"
			r.SmallestNextExperiment = "keep scoping disabled and diagnose negative control " + row.Name
			return r
		}
		net := row.ScopeAvoided + row.PrefixReadAvoided - row.NetOverhead
		scopeTotal += row.ScopeAvoided
		prefixTotal += row.PrefixReadAvoided
		netTotal += net
		if row.ScopeAvoided <= row.Uncertainty || net <= row.Uncertainty {
			r.ContradictoryRows = append(r.ContradictoryRows, row.Name)
			inconclusive = true
		}
	}
	if scopeTotal+prefixTotal > 0 {
		r.ScopeShare = 100 * scopeTotal / (scopeTotal + prefixTotal)
	}
	r.NetAvoided = netTotal
	if netTotal <= 0 {
		r.Verdict = ScopedPrefixDisable
		r.Reason = "authoritative net avoided work is non-positive"
		r.SmallestNextExperiment = "keep scoping disabled and inspect overhead in the first non-positive row"
		return r
	}
	if inconclusive {
		r.Verdict = ScopedPrefixHold
		r.Reason = "witnessed effects do not establish positive net scope gain beyond uncertainty"
		r.SmallestNextExperiment = "repeat the first contradictory row with enough replicates to narrow uncertainty"
		return r
	}
	r.Verdict = ScopedPrefixEnable
	r.Reason = "equal outcomes and independently witnessed authoritative rows show positive net scope gain"
	r.SmallestNextExperiment = "repeat ENABLE on fresh operator data before widening the envelope"
	return r
}

func missingEnvelope(e ScopedPrefixEnvelope) []string {
	values := []struct{ name, value string }{{"model", e.Model}, {"runtime", e.Runtime}, {"tokenizer", e.Tokenizer}, {"task", e.Task}, {"cache posture", e.WarmthCondition}, {"campaign version", e.CampaignVersion}}
	var missing []string
	for _, v := range values {
		if strings.TrimSpace(v.value) == "" {
			missing = append(missing, v.name)
		}
	}
	return missing
}

// FormatScopedPrefixReport renders the operator readout and generation evidence.
func FormatScopedPrefixReport(r ScopedPrefixReport) string {
	contradictions := "none"
	if len(r.ContradictoryRows) > 0 {
		contradictions = strings.Join(r.ContradictoryRows, ", ")
	}
	return fmt.Sprintf("decision: %s\nreason: %s\nenvelope: model=%s runtime=%s tokenizer=%s task=%s cache=%s campaign=%s\ncontradictory rows: %s\nsmallest next experiment: %s\nruntime default changed: %t\nrollback: %s\npromotion evidence: %s\ndemotion evidence: %s\ninvalidating assumption: %s\n", r.Verdict, r.Reason, r.Envelope.Model, r.Envelope.Runtime, r.Envelope.Tokenizer, r.Envelope.Task, r.Envelope.WarmthCondition, r.Envelope.CampaignVersion, contradictions, r.SmallestNextExperiment, r.RuntimeDefaultChanged, r.Rollback, r.PromotionEvidence, r.DemotionEvidence, r.InvalidatingAssumption)
}
