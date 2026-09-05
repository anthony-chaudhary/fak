package trendreport

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// TriageEnforced checks whether automated triage gating is enabled for the specified mode.
func TriageEnforced(mode string) bool {
	return strings.EqualFold(strings.TrimSpace(mode), "enforce")
}

// TriageEnvelope determines whether an envelope's next action requires human intervention.
func TriageEnvelope(label string, e Envelope) choicetriage.Verdict {
	return choicetriage.Triage(choicetriage.Signal{
		Source:   label,
		Question: e.Finding,
		Detail:   e.Reason,
		Action:   e.NextAction,
	})
}

// AdvisoryGateTriaged evaluates advisory gate conditions while delegating routable actions.
func AdvisoryGateTriaged(label, finding, reason, nextAction, unmeasuredFinding string, enforce bool) GateVerdict {
	base := AdvisoryGate(label, finding, reason, unmeasuredFinding)
	if base.Exit == 0 || !enforce {
		return base
	}
	v := choicetriage.Triage(choicetriage.Signal{
		Source:   label,
		Question: finding,
		Detail:   reason,
		Action:   nextAction,
	})
	if v.NeedsHuman {
		return base
	}
	return GateVerdict{Exit: 0, Message: label + " ROUTED: " + reason + " — " + v.Resolve}
}

// TriageSelfcheck validates triage gating behavior across enforce and warn modes.
func TriageSelfcheck() error {
	const (
		label  = "CADENCE"
		unmeas = "cadence_unmeasured"
		reason = "a dimension could not be measured"
		rerun  = "repair scores, then rerun `fak cadence`"
	)

	if g := AdvisoryGateTriaged(label, unmeas, reason, rerun, unmeas, true); g.Exit != 0 {
		return fmt.Errorf("a runnable incomplete report must route to the fleet under enforce, got exit %d (%s)", g.Exit, g.Message)
	}
	if g := AdvisoryGateTriaged(label, unmeas, reason, rerun, unmeas, false); g.Exit != 1 {
		return fmt.Errorf("warn mode must leave the incomplete page unchanged, got exit %d", g.Exit)
	}
	if g := AdvisoryGateTriaged("RELEASE", "release_unmeasured", "publish decision pending", "approve the tagged release before publish", "release_unmeasured", true); g.Exit != 1 {
		return fmt.Errorf("an authority-bearing incomplete report must still page under enforce, got exit %d", g.Exit)
	}
	if g := AdvisoryGateTriaged(label, "cadence_recorded", "recorded", "", unmeas, true); g.Exit != 0 {
		return fmt.Errorf("a measured report must stay OK, got exit %d", g.Exit)
	}
	if !TriageEnforced("enforce") || TriageEnforced("") || TriageEnforced("warn") {
		return fmt.Errorf("TriageEnforced must flip only on \"enforce\"")
	}
	return nil
}
