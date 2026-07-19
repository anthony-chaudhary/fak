// Admission wiring for the steerpr overlay-maintenance loop (#5023). The
// loop's tick claims "every commit in base..head is assigned to exactly one
// unit or reported as an orphan, and every touched unit's band was
// re-folded". That claim is admissible only against an EXTERNAL witness —
// dos commit-audit over the same range — so the wiring here binds the turn to
// CriterionCommitAudit over base..head and Adjudicate does the rest: admitted
// only on OutcomeWitnessed, re-armed with ReasonDoneUnwitnessed otherwise.
//
// This file deliberately takes plain fields, not a steerpr type: the gate
// stays free of sibling-leaf imports (steerpr is pureRoot and cannot import
// loopgate back, so the seam is data, not types).
package loopgate

import "strings"

// TurnForSteerprTick shapes one steerpr overlay tick's done-claim as a
// loopgate turn: the claim text, and the commit-audit witness criterion over
// the tick's base..head range (just head for a rangeless genesis tick). A
// tick that is not claiming done passes claimedDone=false and simply
// continues (NOT_YET) without owing a witness call.
func TurnForSteerprTick(base, head, claim string, claimedDone bool) Turn {
	base = strings.TrimSpace(base)
	head = strings.TrimSpace(head)
	ref := head
	if base != "" && head != "" {
		ref = base + ".." + head
	}
	return Turn{
		ClaimedDone: claimedDone,
		Claim:       claim,
		HeadRef:     head,
		Criterion:   Criterion{Kind: CriterionCommitAudit, Ref: ref},
	}
}

// SteerprAuditRef renders the external witness reference an ADMITTED
// tick records in its ledger row's witness field: the re-runnable dos argv
// the decision was witnessed against. It returns "" for any non-witnessed
// decision, so the ledger can never carry a fabricated witness binding — the
// row's witness field is downstream of the gate's verdict, never of the
// loop's self-report.
func SteerprAuditRef(d Decision) string {
	if d.Verdict != VerdictWitnessed {
		return ""
	}
	return "dos " + strings.Join(d.Request.Argv(), " ")
}
