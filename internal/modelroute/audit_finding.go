package modelroute

import (
	"fmt"
	"sort"
	"strings"
)

// FindingAction is the closed vocabulary of issue mutations the cross-audit
// finding planner proposes for one audit receipt. It never widens: an untrusted
// verdict maps to exactly one action, and ESCALATE (not CREATE) is the sink for
// non-actionable INCONCLUSIVE/UNAVAILABLE receipts so a transient auditor outage
// or model-only uncertainty never becomes a public "this code is broken" claim.
type FindingAction string

const (
	// FindingNoop records that a receipt needs no issue mutation — a clean PASS
	// with no open finding, or a repeated identical REFUTE already captured on an
	// open finding (the dedupe path).
	FindingNoop FindingAction = "NOOP"
	// FindingCreate opens one new finding issue for a REFUTE whose key has no
	// existing finding.
	FindingCreate FindingAction = "CREATE"
	// FindingUpdate appends new REFUTE evidence to an existing open finding whose
	// last recorded receipt differs.
	FindingUpdate FindingAction = "UPDATE"
	// FindingComment records an independent re-audit PASS on an open finding
	// without closing it — closure still requires a witnessed fix.
	FindingComment FindingAction = "COMMENT"
	// FindingReopen reopens a closed finding whose subject a fresh REFUTE recurs
	// on: the resolution did not hold.
	FindingReopen FindingAction = "REOPEN"
	// FindingEscalate queues an INCONCLUSIVE/UNAVAILABLE receipt for human review
	// instead of asserting a corruption finding.
	FindingEscalate FindingAction = "ESCALATE"
)

// Valid reports whether a is a member of the closed FindingAction vocabulary.
func (a FindingAction) Valid() bool {
	switch a {
	case FindingNoop, FindingCreate, FindingUpdate, FindingComment, FindingReopen, FindingEscalate:
		return true
	default:
		return false
	}
}

// Mutating reports whether the action would change GitHub state (so a live
// adapter can bound and count only the mutations).
func (a FindingAction) Mutating() bool {
	switch a {
	case FindingCreate, FindingUpdate, FindingComment, FindingReopen:
		return true
	default:
		return false
	}
}

// FindingReason is the closed vocabulary explaining WHY the planner chose an
// action. It is separate from the receipt's free-text reason so a supervisor can
// route on the decision class without parsing prose.
const (
	FindingReasonNewRefute          = "NEW_REFUTE"              // CREATE
	FindingReasonDuplicateReceipt   = "DUPLICATE_RECEIPT"       // NOOP
	FindingReasonNewEvidence        = "NEW_EVIDENCE"            // UPDATE
	FindingReasonResolvedRecurrence = "RESOLVED_RECURRENCE"     // REOPEN
	FindingReasonReauditPassed      = "REAUDIT_PASSED"          // COMMENT
	FindingReasonCleanPass          = "CLEAN_PASS"              // NOOP
	FindingReasonInconclusive       = "INCONCLUSIVE_ESCALATION" // ESCALATE
	FindingReasonUnavailable        = "AUDITOR_UNAVAILABLE"     // ESCALATE
)

// FindingState is the open/closed lifecycle of an existing finding issue.
type FindingState string

const (
	FindingStateOpen   FindingState = "open"
	FindingStateClosed FindingState = "closed"
)

// normalizeFindingState defaults anything that is not explicitly "closed" to
// open — an unset state on a finding row is treated as still-open (fail-safe:
// a REFUTE on an unknown-state finding updates in place rather than reopening).
func normalizeFindingState(s FindingState) FindingState {
	if strings.EqualFold(strings.TrimSpace(string(s)), string(FindingStateClosed)) {
		return FindingStateClosed
	}
	return FindingStateOpen
}

// ExistingFinding is the durable state of a finding issue already filed for a
// finding key: whether it is open or closed, and the receipt digest last
// recorded on it. The cmd adapter parses these from issue-body markers; the
// planner tests inject them directly. The planner never fetches anything.
type ExistingFinding struct {
	Key           string       `json:"key"`
	IssueNumber   int          `json:"issue_number"`
	State         FindingState `json:"state"`
	ReceiptDigest string       `json:"receipt_digest,omitempty"`
}

const (
	// FindingPlanSchema stamps the typed plan output.
	FindingPlanSchema = "fak-crossaudit-finding-plan/v1"
	// FindingKeyPrefix namespaces the stable per-subject finding key.
	FindingKeyPrefix = "crossaudit/finding/"
)

// FindingKey is the stable, human-readable dedupe key for one audit subject's
// finding. It binds the audited issue number and the subject bundle digest (the
// exact issue+diff that was audited), so an identical re-audit of the same
// closed change maps to the same finding, while a new closing change produces a
// new key. The form matches issuecontract's marker-key grammar.
func FindingKey(subject IssueAuditSubject) string {
	digest := shortDigest(subject.Digest)
	if digest == "" {
		digest = "unknown"
	}
	return fmt.Sprintf("%s%d/%s", FindingKeyPrefix, subject.IssueNumber, digest)
}

// FindingPlanItem is one planned action for one receipt. It carries everything
// the adapter needs to render a candidate (CREATE), edit/comment an existing
// finding (UPDATE/COMMENT/REOPEN), or queue an escalation (ESCALATE) — without
// the adapter re-deriving any decision.
type FindingPlanItem struct {
	Action        FindingAction        `json:"action"`
	Reason        string               `json:"reason"`
	Key           string               `json:"key"`
	AuditedIssue  int                  `json:"audited_issue"`
	TargetIssue   int                  `json:"target_issue,omitempty"`
	Verdict       CrossAuditVerdict    `json:"verdict"`
	Severity      AuditFindingSeverity `json:"severity,omitempty"`
	ReceiptDigest string               `json:"receipt_digest,omitempty"`
	AuditKey      string               `json:"audit_key,omitempty"`
	Subject       IssueAuditSubject    `json:"subject"`
	Detail        string               `json:"detail,omitempty"`
	EvidenceRefs  []EvidenceRef        `json:"evidence_refs,omitempty"`
}

// FindingPlan is the deterministic output of the planner: one item per input
// receipt (in receipt order) plus per-action counts.
type FindingPlan struct {
	Schema string                `json:"schema"`
	Items  []FindingPlanItem     `json:"items"`
	Counts map[FindingAction]int `json:"counts"`
}

// PlanCrossAuditFindings is the pure finding planner. Given a batch of verified
// audit receipts (in ledger order) and the set of findings already filed, it
// returns exactly one action per receipt under these rules:
//
//	REFUTE, no finding for the key      -> CREATE   (NEW_REFUTE)
//	REFUTE, finding closed              -> REOPEN   (RESOLVED_RECURRENCE)
//	REFUTE, finding open, same receipt  -> NOOP     (DUPLICATE_RECEIPT)
//	REFUTE, finding open, new receipt   -> UPDATE   (NEW_EVIDENCE)
//	PASS,   finding open                -> COMMENT  (REAUDIT_PASSED, does not close)
//	PASS,   otherwise                   -> NOOP     (CLEAN_PASS)
//	INCONCLUSIVE                        -> ESCALATE (INCONCLUSIVE_ESCALATION)
//	UNAVAILABLE / anything else         -> ESCALATE (AUDITOR_UNAVAILABLE)
//
// The in-batch finding state is advanced after each item so repeated identical
// receipts in one run collapse onto a single finding (the first REFUTE creates,
// the rest dedupe), and a recurrence followed by a duplicate does not double up.
// Closure is never auto-applied here: a PASS re-audit only comments, because the
// done-condition requires a witnessed fix in addition to an independent re-audit.
func PlanCrossAuditFindings(receipts []IssueAuditReceipt, existing []ExistingFinding) FindingPlan {
	state := make(map[string]ExistingFinding, len(existing))
	for _, f := range existing {
		key := strings.TrimSpace(f.Key)
		if key == "" {
			continue
		}
		state[key] = ExistingFinding{
			Key:           key,
			IssueNumber:   f.IssueNumber,
			State:         normalizeFindingState(f.State),
			ReceiptDigest: strings.TrimSpace(f.ReceiptDigest),
		}
	}

	plan := FindingPlan{Schema: FindingPlanSchema, Counts: map[FindingAction]int{}}
	for _, receipt := range receipts {
		item := planOneFinding(receipt, state)
		advanceFindingState(state, item)
		plan.Items = append(plan.Items, item)
		plan.Counts[item.Action]++
	}
	return plan
}

func planOneFinding(receipt IssueAuditReceipt, state map[string]ExistingFinding) FindingPlanItem {
	key := FindingKey(receipt.Subject)
	item := FindingPlanItem{
		Key:           key,
		AuditedIssue:  receipt.Subject.IssueNumber,
		Verdict:       receipt.Verdict,
		Severity:      receipt.Severity,
		ReceiptDigest: strings.TrimSpace(receipt.ReceiptDigest),
		AuditKey:      strings.TrimSpace(receipt.AuditKey),
		Subject:       receipt.Subject,
		Detail:        strings.TrimSpace(receipt.Reason),
		EvidenceRefs:  receipt.EvidenceRefs,
	}
	existing, known := state[key]

	switch receipt.Verdict {
	case CrossAuditRefute:
		switch {
		case !known:
			item.Action = FindingCreate
			item.Reason = FindingReasonNewRefute
		case existing.State == FindingStateClosed:
			item.Action = FindingReopen
			item.Reason = FindingReasonResolvedRecurrence
			item.TargetIssue = existing.IssueNumber
		case existing.ReceiptDigest != "" && existing.ReceiptDigest == item.ReceiptDigest:
			item.Action = FindingNoop
			item.Reason = FindingReasonDuplicateReceipt
			item.TargetIssue = existing.IssueNumber
		default:
			item.Action = FindingUpdate
			item.Reason = FindingReasonNewEvidence
			item.TargetIssue = existing.IssueNumber
		}
	case CrossAuditPass:
		if known && existing.State == FindingStateOpen {
			item.Action = FindingComment
			item.Reason = FindingReasonReauditPassed
			item.TargetIssue = existing.IssueNumber
		} else {
			item.Action = FindingNoop
			item.Reason = FindingReasonCleanPass
			if known {
				item.TargetIssue = existing.IssueNumber
			}
		}
	case CrossAuditInconclusive:
		item.Action = FindingEscalate
		item.Reason = FindingReasonInconclusive
		if known {
			item.TargetIssue = existing.IssueNumber
		}
	default: // UNAVAILABLE or any out-of-vocabulary verdict: escalate, never accuse.
		item.Action = FindingEscalate
		item.Reason = FindingReasonUnavailable
		if known {
			item.TargetIssue = existing.IssueNumber
		}
	}
	return item
}

// advanceFindingState folds one planned item back into the in-batch state so a
// following receipt for the same key sees the effect of this one. A CREATE
// records an open finding with issue number 0 (not filed yet) whose recorded
// digest is this receipt's, which is exactly what makes a second identical
// receipt dedupe to NOOP.
func advanceFindingState(state map[string]ExistingFinding, item FindingPlanItem) {
	switch item.Action {
	case FindingCreate:
		state[item.Key] = ExistingFinding{
			Key:           item.Key,
			IssueNumber:   0,
			State:         FindingStateOpen,
			ReceiptDigest: item.ReceiptDigest,
		}
	case FindingUpdate, FindingReopen:
		f := state[item.Key]
		f.Key = item.Key
		f.IssueNumber = item.TargetIssue
		f.State = FindingStateOpen
		f.ReceiptDigest = item.ReceiptDigest
		state[item.Key] = f
	default:
		// NOOP / COMMENT / ESCALATE do not change the recorded finding state.
	}
}

// FindingPlanActionCounts returns the per-action counts as an ordered slice so a
// renderer can print them deterministically without depending on map order.
func FindingPlanActionCounts(plan FindingPlan) []struct {
	Action FindingAction
	Count  int
} {
	actions := []FindingAction{
		FindingCreate, FindingUpdate, FindingReopen, FindingComment, FindingEscalate, FindingNoop,
	}
	out := make([]struct {
		Action FindingAction
		Count  int
	}, 0, len(actions))
	seen := map[FindingAction]bool{}
	for _, a := range actions {
		seen[a] = true
		out = append(out, struct {
			Action FindingAction
			Count  int
		}{a, plan.Counts[a]})
	}
	// Surface any out-of-vocabulary action that somehow entered the counts.
	var extra []FindingAction
	for a := range plan.Counts {
		if !seen[a] {
			extra = append(extra, a)
		}
	}
	sort.Slice(extra, func(i, j int) bool { return extra[i] < extra[j] })
	for _, a := range extra {
		out = append(out, struct {
			Action FindingAction
			Count  int
		}{a, plan.Counts[a]})
	}
	return out
}
