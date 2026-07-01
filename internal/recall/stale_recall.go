package recall

import "fmt"

type StaleRecallAction string

const (
	StaleRecallRefreshSource StaleRecallAction = "refresh_source"
	StaleRecallQueryUser     StaleRecallAction = "query_user"
)

type StaleRecallDecision struct {
	Step      int               `json:"step"`
	Digest    string            `json:"digest"`
	Axis      string            `json:"axis"`
	Action    StaleRecallAction `json:"action"`
	Reason    string            `json:"reason"`
	SourceRef string            `json:"source_ref"`
}

type StaleRecallReport struct {
	EffectSafe bool                  `json:"effect_safe"`
	Decisions  []StaleRecallDecision `json:"decisions,omitempty"`
}

func PlanStaleRecallRefresh(syndromes []PageSyndrome) StaleRecallReport {
	report := StaleRecallReport{EffectSafe: true}
	for _, syn := range syndromes {
		if syn.Reusable() {
			continue
		}
		ev, ok := staleRecallEvidence(syn)
		if !ok {
			continue
		}
		report.EffectSafe = false
		report.Decisions = append(report.Decisions, StaleRecallDecision{
			Step:      syn.Step,
			Digest:    syn.Digest,
			Axis:      ev.Axis.String(),
			Action:    staleRecallAction(ev),
			Reason:    ev.Reason,
			SourceRef: fmt.Sprintf("recall:page:%d:%s", syn.Step, ev.Axis),
		})
	}
	return report
}

func staleRecallEvidence(syn PageSyndrome) (Evidence, bool) {
	for _, axis := range []EvidenceAxis{EvidenceTrustEpoch, EvidenceWitness} {
		ev, ok := syn.EvidenceFor(axis)
		if ok && ev.blocks() {
			return ev, true
		}
	}
	return Evidence{}, false
}

func staleRecallAction(ev Evidence) StaleRecallAction {
	if ev.Axis == EvidenceTrustEpoch || ev.Axis == EvidenceWitness {
		return StaleRecallRefreshSource
	}
	return StaleRecallQueryUser
}
