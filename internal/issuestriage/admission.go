package issuestriage

import (
	"sort"
	"time"
)

// Provenance records enough information to re-derive an ingested triage item.
type Provenance struct {
	QueryOrRule string    `json:"query_or_rule"`
	Source      string    `json:"source"`
	Terms       string    `json:"terms,omitempty"`
	RetrievedAt time.Time `json:"retrieved_at"`
	ToolVersion string    `json:"tool_version"`
}

// Signal is one independent classifier's answer.
type Signal string

const (
	SignalAdmit   Signal = "ADMIT"
	SignalReject  Signal = "REJECT"
	SignalMissing Signal = "MISSING"
)

// Outcome is deliberately ternary. ABSTAIN is persisted, never represented by omission.
type Outcome string

const (
	Admit   Outcome = "ADMIT"
	Reject  Outcome = "REJECT"
	Abstain Outcome = "ABSTAIN"
)

// AbstainReason is closed and aliases the existing issues-triage refusal vocabulary.
type AbstainReason string

const (
	NeedsArea       AbstainReason = "needs-area"
	NeedsKind       AbstainReason = "needs-kind"
	LikelyDuplicate AbstainReason = "likely-dup"
)

// Admission is the stored adjudication attached to an ingested item.
type Admission struct {
	SignalA    Signal        `json:"signal_a"`
	SignalB    Signal        `json:"signal_b"`
	Outcome    Outcome       `json:"outcome"`
	Reason     AbstainReason `json:"reason,omitempty"`
	Provenance Provenance    `json:"provenance"`
}

// TwoSignalAdmission admits or rejects only corroborated answers. Any missing or
// disagreeing signal is a first-class ABSTAIN with a closed reason.
func TwoSignalAdmission(a, b Signal, reason AbstainReason, p Provenance) Admission {
	out := Abstain
	if a == SignalAdmit && b == SignalAdmit {
		out = Admit
	}
	if a == SignalReject && b == SignalReject {
		out = Reject
	}
	if out != Abstain {
		reason = ""
	}
	return Admission{SignalA: a, SignalB: b, Outcome: out, Reason: reason, Provenance: p}
}

// AdmissionAudit measures the migration and exposes legacy single-signal admissions.
type AdmissionAudit struct {
	BeforeSingleSignalAdmitted int   `json:"before_single_signal_admitted"`
	AfterAdmitted              int   `json:"after_admitted"`
	AfterAbstained             int   `json:"after_abstained"`
	SingleSignalIssueNumbers   []int `json:"single_signal_issue_numbers"`
}

// AuditAdmissions answers "which admitted items rest on a single signal?" and
// reports the before/after counts for the supplied real issues-triage records.
func AuditAdmissions(items map[int]Admission) AdmissionAudit {
	var r AdmissionAudit
	for n, x := range items {
		single := x.SignalA == SignalMissing || x.SignalB == SignalMissing
		if single && (x.SignalA == SignalAdmit || x.SignalB == SignalAdmit) {
			r.BeforeSingleSignalAdmitted++
			r.SingleSignalIssueNumbers = append(r.SingleSignalIssueNumbers, n)
		}
		if x.Outcome == Admit {
			r.AfterAdmitted++
		}
		if x.Outcome == Abstain {
			r.AfterAbstained++
		}
	}
	sort.Ints(r.SingleSignalIssueNumbers)
	return r
}

// IngestAction applies two-signal admission to the real issues-triage input.
// The action produced by the garden scan is signal A; caller corroboration is
// signal B. Legacy callers that cannot corroborate pass SignalMissing and the
// action remains stored as ABSTAIN for audit instead of silently becoming fact.
func IngestAction(action Action, corroboration Signal, reason AbstainReason, p Provenance) Admission {
	first := SignalMissing
	if action.Number > 0 && action.Kind != "" {
		first = SignalAdmit
	}
	return TwoSignalAdmission(first, corroboration, reason, p)
}
