package gardenbundle

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// walk_triage.go applies the decenter-the-human doctrine to the walk's REVIEW bucket.
// PlanWalk (walk.go) splits every attention item into DispAct (a ready command the fleet
// runs) and DispReview — commented "needs human judgment". But that label over-centers the
// person: a review that only needs an AREA label, a KIND label, or a dedup check is knowable
// classification an agent can drive in a fresh context, not an authority call. The only
// review that genuinely waits on a person is one that names a PRIORITY / policy / release
// authority — the small irreducible remainder.
//
// TriageDecision folds each emitted WalkDecision through internal/choicetriage so the
// disposition emerges from the shared lexicon rather than a private "review == human" rule.
// It mirrors cmd/fak's triageIssueAction Signal shape exactly (Detail = the condition/reason,
// Action = the ready command), so the garden walk and the issues pane agree on what pages a
// person for the same item. DecisionNeedsHuman is the enforce-mode replacement for "review ==
// needs-you": a DispAct is TAKE_OBVIOUS (the fleet's), a needs-area/needs-kind/likely-dup
// review is FRESH_CONTEXT (the fleet's), and only an unset-priority review stays NeedsHuman.
//
// Pure, deterministic, no I/O — like the rest of walk.go. It soaks behind
// FAK_GARDENWALK_TRIAGE_GATE (read at the CLI): the default worklist is unchanged until
// enforce splits the review count into needs-you vs fleet-drives.

// TriageDecision folds one emitted walk decision into its choicetriage disposition. The
// Signal is built from the decision's own fields — the ready Command is the strongest
// TAKE_OBVIOUS signal, and the Reason (the condition tags, e.g. "needs-area, likely-dup")
// carries the authority test: a reason naming priority/policy/release routes to
// HUMAN_RESIDUAL, everything else to FRESH_CONTEXT. Source is the neutral "garden" — never a
// token choicetriage reads as authority.
func TriageDecision(d WalkDecision) choicetriage.Verdict {
	severity := "action"
	if d.Disposition == DispReview {
		severity = "decision"
	}
	return choicetriage.Triage(choicetriage.Signal{
		Severity:    severity,
		Source:      "garden",
		Question:    fmt.Sprintf("#%d: %s", d.ID, firstNonEmptyStr(d.Action, string(d.Disposition))),
		Detail:      d.Reason,
		Action:      d.Command,
		OptionCount: 2,
	})
}

// DecisionNeedsHuman reports whether an emitted decision genuinely waits on a person — the
// enforce-mode replacement for "DispReview == needs you". True only for a real
// priority/policy/release authority; a ready-command act and a knowable classification review
// both return false even though PlanWalk emits the latter under DispReview.
func DecisionNeedsHuman(d WalkDecision) bool {
	return TriageDecision(d).NeedsHuman
}

// WalkAttentionSplit partitions the plan's emitted worklist into the decisions that genuinely
// wait on a person and the ones the fleet drives itself. Deterministic: it walks
// plan.Decisions in their already-sorted worst-first order.
func WalkAttentionSplit(p WalkPlan) (needHuman, fleetDrives []WalkDecision) {
	for _, d := range p.Decisions {
		if DecisionNeedsHuman(d) {
			needHuman = append(needHuman, d)
		} else {
			fleetDrives = append(fleetDrives, d)
		}
	}
	return needHuman, fleetDrives
}

// GardenWalkTriageEnforced reports whether the decenter split is active for the given mode
// string. Only "enforce" (case-insensitive) turns it on; "", "warn" and anything else leave
// the walk's worklist byte-for-byte unchanged so the fold can soak. Mirrors every other
// decenter seam's enforce/warn switch.
func GardenWalkTriageEnforced(mode string) bool {
	switch mode {
	case "enforce", "ENFORCE", "Enforce":
		return true
	default:
		return false
	}
}

// AttentionTriageLine renders the decenter split for a walk plan as one readout line: of the
// emitted worklist, how many genuinely wait on a person vs how many the fleet drives. Returns
// "" when nothing was emitted (an empty or all-skipped walk has no attention to split).
// Rendered only under FAK_GARDENWALK_TRIAGE_GATE=enforce.
func AttentionTriageLine(p WalkPlan) string {
	if len(p.Decisions) == 0 {
		return ""
	}
	needHuman, fleetDrives := WalkAttentionSplit(p)
	return fmt.Sprintf("attention-triage: %d needs-you (priority/policy authority), %d fleet-drives (act + knowable review)",
		len(needHuman), len(fleetDrives))
}

// TriageSelfcheck is the deterministic, no-I/O proof of the walk fold: a ready-command act is
// the fleet's to run, a needs-area / needs-kind / likely-dup review is the fleet's to drive in
// a fresh context, and only an unset-priority review genuinely waits on a person. It is the
// witness the CLI surfaces as the walk's triage selfcheck.
func TriageSelfcheck() error {
	// A ready-command act -> TAKE_OBVIOUS, not a person.
	act := WalkDecision{ID: 1, Disposition: DispAct, Action: "mark-stale", Command: "gh issue edit 1 --add-label stale", Reason: "idle 90d"}
	if v := TriageDecision(act); v.NeedsHuman || v.Disposition != choicetriage.TakeObvious {
		return fmt.Errorf("a ready-command act must be TAKE_OBVIOUS and not need a person, got %s", v.Disposition)
	}
	// Knowable-classification reviews -> FRESH_CONTEXT, not a person.
	for _, reason := range []string{"needs-area", "needs-kind", "likely-dup", "needs-area, likely-dup", "orphan", "bare"} {
		rev := WalkDecision{ID: 2, Disposition: DispReview, Action: "review", Reason: reason}
		v := TriageDecision(rev)
		if v.NeedsHuman {
			return fmt.Errorf("review %q is knowable classification — it must NOT wait on a person, got %s", reason, v.Disposition)
		}
		if v.Disposition != choicetriage.FreshContext {
			return fmt.Errorf("review %q must route to FRESH_CONTEXT, got %s", reason, v.Disposition)
		}
	}
	// An unset-priority review -> HUMAN_RESIDUAL: a prioritization authority a person holds.
	prio := WalkDecision{ID: 3, Disposition: DispReview, Action: "review", Reason: "needs-priority, needs-area"}
	if v := TriageDecision(prio); !v.NeedsHuman || v.Disposition != choicetriage.HumanResidual {
		return fmt.Errorf("an unset-priority review must be HUMAN_RESIDUAL and wait on a person, got %s", v.Disposition)
	}
	// The split covers every emitted decision, and the enforce switch flips only on "enforce".
	plan := WalkPlan{Decisions: []WalkDecision{act, prio}}
	nh, fd := WalkAttentionSplit(plan)
	if len(nh)+len(fd) != len(plan.Decisions) {
		return fmt.Errorf("attention split lost a decision: %d+%d != %d", len(nh), len(fd), len(plan.Decisions))
	}
	if !GardenWalkTriageEnforced("enforce") || GardenWalkTriageEnforced("") || GardenWalkTriageEnforced("warn") {
		return fmt.Errorf("GardenWalkTriageEnforced must flip only on \"enforce\"")
	}
	return nil
}

// firstNonEmptyStr returns the first non-empty string, or "".
func firstNonEmptyStr(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
