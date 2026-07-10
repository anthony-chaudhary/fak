package checkpointscore

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

// dispatch.go maps checkpoint-debt Gaps onto the shared dogfoodissues.ActionItem backlog
// input, so `fak checkpoint-debt-dispatch` fans each gap out to a sink (stdout / local-db /
// GitHub) exactly the way unwired-debt-dispatch fans out orphaned packages. Keys are
// content-stable (no timestamp) so re-runs dedup and converge.

// Key is the stable dedup key for a gap: one issue per (subsystem, axis).
func (g Gap) Key() string {
	return "checkpoint-debt/" + slug(g.Subsystem+"-"+g.Axis)
}

// Title is the one-line issue subject.
func (g Gap) Title() string {
	switch g.Axis {
	case "crash_recovery":
		return "checkpoint: " + g.Subsystem + " does not persist resumable WIP state"
	case "status":
		return "checkpoint: " + g.Subsystem + " exposes no witnessed status surface"
	case "planned":
		return "checkpoint: build " + g.Subsystem + " (unified worker-state checkpoint/restore)"
	default:
		return "checkpoint: " + g.Subsystem + " has a checkpoint gap"
	}
}

func (g Gap) grade() string {
	if g.Axis == "planned" {
		return "F" // an entire missing recovery subsystem is the loudest debt
	}
	return "D"
}

// ToActionItem maps a gap onto the shared dogfoodissues.ActionItem, reusing the exact backlog
// contract every other fak dispatcher speaks so one dedup/sync path serves them all.
func (g Gap) ToActionItem(evidencePath string) dogfoodissues.ActionItem {
	axis := strings.ReplaceAll(g.Axis, "_", " ")
	return dogfoodissues.ActionItem{
		Key:          g.Key(),
		Title:        g.Title(),
		SourceProbe:  "checkpoint-scorecard",
		ScoreName:    "checkpoint_gap",
		Score:        g.Axis,
		Grade:        g.grade(),
		DebtName:     DebtKey,
		DebtCount:    1,
		EvidencePath: evidencePath,
		NextAction:   g.nextAction(),
		Finding:      g.Key(),
		ParentRef:    "fak checkpoint-scorecard",
		CurrentState: "The checkpoint scorecard found the long-running subsystem " + g.Subsystem +
			" (" + g.Role + ") is missing its " + axis + " affordance: " + g.Detail + ". A process that " +
			"runs unbounded without this loses its work-in-progress on a mid-task crash, or leaves peers " +
			"unable to tell it stopped.",
		WhyNow: "Crash-recovery and witnessed status are the two affordances that keep in-flight work " +
			"from evaporating silently. This subsystem does long-running work but cannot survive/report a " +
			"mid-task crash on this axis, so the loss is invisible until a real crash proves it.",
		WorkingSpine:   g.workingSpine(),
		WorkUnit:       "leaf",
		ExpectedSteps:  4,
		Assumptions:    []string{"The subsystem's existing surface names its intended durable store / status fold, so the missing affordance can be added in-package."},
		ConfusionRisks: []string{"Do not satisfy the probe by adding the token in a comment or a test; the affordance must be real, non-test source.", "If the subsystem legitimately does not own this axis, record a documented roster exception in internal/checkpointscore rather than a fake affordance."},
		Coordination:   []string{"One generated issue owns one (subsystem, axis) gap."},
		Trigger:        "Checkpoint scorecard reports " + g.Subsystem + " missing " + axis + ".",
		BatchPolicy:    "One issue per (subsystem, axis); reruns update by stable marker.",
		InScope:        g.workingSpine(),
		OutOfScope:     "Do not change the checkpoint scorecard's roster tokens or grade thresholds to clear the gap, and do not touch other subsystems in the same change.",
		DoneCondition:  "A re-run of `fak checkpoint-scorecard --json` no longer lists the `" + g.Key() + "` gap.",
		Witness:        "fak checkpoint-scorecard --json",
		AcceptanceGate: "go build ./... && go test ./internal/checkpointscore",
		Lane:           "",
		Paths:          []string{g.Dir + "/**"},
		Labels:         []string{"checkpoint-debt", "tech-debt"},
		BoundaryNotes:  []string{"Public subsystem-source evidence only; no private or lab-local artifacts."},
		ClosureBinding: "Resolving commit cites `#N` in the subject and carries a `(fak <leaf>)` trailer for the subsystem's lane.",
	}
}

func (g Gap) nextAction() string {
	switch g.Axis {
	case "crash_recovery":
		return "Add a durable, resumable WIP store to " + g.Dir + " (an append-only journal/ledger the subsystem can replay after a crash)."
	case "status":
		return "Add a witnessed status surface to " + g.Dir + " (a fold a peer can read without tailing logs)."
	case "planned":
		return "Build " + g.Dir + ": a unified worker-state checkpoint that composes shadowgit, the loop ledger, and the session journal into a resumable replay."
	default:
		return "Close the checkpoint gap in " + g.Dir + "."
	}
}

func (g Gap) workingSpine() string {
	switch g.Axis {
	case "crash_recovery":
		return "Persist the subsystem's in-flight state to a durable append-only store keyed by unit of work, and add a resume path that replays it after a crash."
	case "status":
		return "Fold the subsystem's live state into a witnessed status value (a closed status vocabulary or a summary), exposed on a default path."
	case "planned":
		return "Compose the existing write-ledger (shadowgit), run ledger (loopmgr), and boot-epoch crash journal (sessionjournal) into a single checkpoint that can resume an agent exactly where it crashed."
	default:
		return "Close the checkpoint gap."
	}
}

// ActionItems maps a set of gaps onto dogfoodissues.ActionItems, the backlog input.
func ActionItems(gaps []Gap, evidencePath string) []dogfoodissues.ActionItem {
	items := make([]dogfoodissues.ActionItem, 0, len(gaps))
	for _, g := range gaps {
		items = append(items, g.ToActionItem(evidencePath))
	}
	return items
}

// slug lowercases and dash-collapses an arbitrary string for a stable issue key.
func slug(s string) string {
	s = strings.ToLower(s)
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	return strings.Trim(b.String(), "-")
}
