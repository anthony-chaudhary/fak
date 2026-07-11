package releasestatus

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// status_triage.go applies the decenter-the-human doctrine at the release-status next-action
// seam. loopStatus (releasestatus.go) folds the single operator move into one ACTION/OK
// verdict: every kind in actionNextActions reads ACTION, so a loop or operator is paged
// identically whether the move is "cut a release" or "fix a red CI run". But those are not
// one job:
//
//   - Cutting a rolling release, promoting to the stable channel, or promoting the release
//     branch is a genuine PUBLISH/RELEASE authority — minting an outward-facing tag a person
//     holds the authority to approve. So is a CI billing wall: a GitHub Actions spend/budget
//     limit is an account authority no agent can clear. These are the irreducible
//     HUMAN_RESIDUAL, and the fold keeps the page for them.
//   - Fixing a red CI run, confirming a green signal, repairing a workflow YAML, reconciling
//     VERSION topology, repairing stable-tag evidence, cleaning a dirty worktree, clearing a
//     blocker, or weighing whether a soaked tag is worth promoting are all KNOWABLE
//     engineering — the answer is not obvious but it is discoverable, so each wants a fresh
//     context window at the top model tier, not a human tap on the shoulder.
//
// TriageNextAction folds each gating next-action's own nature — NOT its incidental detail
// prose (which mentions "release" even in fix_ci's "before cutting a release") — through
// internal/choicetriage, so the disposition emerges from the shared classifier rather than a
// private table. NextActionNeedsHuman is the enforce-mode replacement for "verdict==ACTION"
// as the thing a loop pages on: only a real PUBLISH/RELEASE/BUDGET authority waits on a
// person; the knowable-engineering moves are the fleet's to drive. It soaks behind
// FAK_RELEASESTATUS_TRIAGE_GATE (read at the CLI): the default readout and check are
// unchanged until enforce.

// nextActionSignals maps each gating next_action kind (the actionNextActions set) to the
// choicetriage.Signal that truthfully describes THAT KIND's nature, phrased so the shared
// classifier lands it on the right disposition. Crucially the text describes the action
// class, not release_status.py's incidental detail line — fix_ci's real detail says "before
// cutting a release", whose stray "release" would misroute it to a human; the Signal here
// says only what fix_ci actually is (a knowable CI repair), so it folds to FRESH_CONTEXT.
//
// The two advisory non-gating kinds the fold also emits (wait, pause_auto_release) have no
// entry: they already read OK and never page, so there is nothing to decenter.
var nextActionSignals = map[string]choicetriage.Signal{
	// --- genuine human authority: minting/publishing an outward-facing release, or an
	//     account budget wall. Each names RELEASE / PUBLISH / BUDGET / SPEND on purpose. ---
	"cut_release": {
		Severity: "decision",
		Question: "cut and publish a rolling release tag",
		Detail:   "minting an outward-facing release build is a publish decision a person holds",
	},
	"cut_release_hot_tree": {
		Severity: "decision",
		Question: "cut and publish a rolling release from a detached trunk checkout",
		Detail:   "minting an outward-facing release build is a publish decision a person holds",
	},
	"promote_stable": {
		Severity: "decision",
		Question: "promote a soaked rolling tag to the stable channel",
		Detail:   "a stable-channel release is an outward-facing publish authority a person holds",
	},
	"promote_release_branch": {
		Severity: "decision",
		Question: "promote the development branch onto the release branch",
		Detail:   "advancing the public release branch is a publish authority a person holds",
	},
	"fix_ci_billing": {
		Severity: "decision",
		Question: "restore CI after a billing wall",
		Detail:   "a GitHub Actions spend/budget limit blocks CI — an account budget authority a person holds, not a code fix",
	},

	// --- knowable engineering: not obvious, but discoverable end-to-end in a fresh context.
	//     Phrased WITHOUT any authority token so each folds to FRESH_CONTEXT. ---
	"fix_ci": {
		Question: "diagnose and fix the red main CI run",
		Detail:   "a knowable CI failure to diagnose and repair before the pipeline can proceed",
	},
	"confirm_ci": {
		Question: "restore or confirm a green main CI signal",
		Detail:   "a knowable check — read the CI state and re-establish a green signal",
	},
	"fix_workflow": {
		Question: "repair the GitHub workflow YAML",
		Detail:   "a knowable YAML repair before the pipeline can proceed",
	},
	"fix_version_topology": {
		Question: "reconcile the VERSION file with the semver tag graph",
		Detail:   "a knowable reconciliation of VERSION against the tag topology",
	},
	"repair_stable_evidence": {
		Question: "repair stable-tag evidence frontmatter",
		Detail:   "a knowable evidence/frontmatter reconciliation against the tag it documents",
	},
	"clean_worktree": {
		Question: "commit, shelve, or remove the dirty paths blocking trunk evidence",
		Detail:   "a knowable worktree cleanup before status can be treated as trunk evidence",
	},
	"hold": {
		Question: "a blocker is holding the pipeline",
		Detail:   "a knowable blocker to investigate and clear before proceeding",
	},
	"consider_stable": {
		Question: "weigh whether the soaked rolling tag is worth promoting",
		Detail:   "an evaluation of whether the stable channel should advance yet",
	},
}

// TriageNextAction folds a next-action into its choicetriage disposition. The bool is false
// for a non-gating kind (wait / pause_auto_release / unknown / ""), which already reads OK
// and has no attention to decenter; true for a gating kind, with the classifier's Verdict.
func TriageNextAction(a NextAction) (choicetriage.Verdict, bool) {
	sig, ok := nextActionSignals[a.Kind]
	if !ok {
		return choicetriage.Verdict{}, false
	}
	sig.Source = "cadence" // a neutral pane name — deliberately NOT "release" (which is itself an authority token in choicetriage)
	return choicetriage.Triage(sig), true
}

// NextActionNeedsHuman reports whether the single operator move genuinely waits on a person —
// the enforce-mode replacement for "verdict==ACTION" as a loop's page condition. True only
// for a real PUBLISH/RELEASE/BUDGET authority (cut/promote/billing); the knowable-engineering
// moves fold to the fleet and return false even though the base loopStatus calls them ACTION.
func NextActionNeedsHuman(a NextAction) bool {
	v, ok := TriageNextAction(a)
	return ok && v.NeedsHuman
}

// ReleaseStatusTriageEnforced reports whether the decenter split is active for the given mode
// string. Only "enforce" (case-insensitive) narrows the loop page-condition to the human
// residual; "warn", "" and anything else leave loopStatus's single ACTION gate untouched so
// the change can soak. Mirrors the enforce/warn switch every other decenter seam reads.
func ReleaseStatusTriageEnforced(mode string) bool {
	return strings.EqualFold(strings.TrimSpace(mode), "enforce")
}

// AttentionTriageLine renders the decenter split for the status's single next-action as one
// readout line: whether the move genuinely waits on a person (needs-you) or is the fleet's to
// drive (fleet-drives), with the kind and disposition named. It returns "" for a non-gating
// action (nothing to page, nothing to decenter). Rendered only under
// FAK_RELEASESTATUS_TRIAGE_GATE=enforce.
func AttentionTriageLine(s Status) string {
	v, ok := TriageNextAction(s.NextAction)
	if !ok {
		return ""
	}
	side := "fleet-drives"
	if v.NeedsHuman {
		side = "needs-you"
	}
	return fmt.Sprintf("next-action-triage: %s (%s=%s)", side, s.NextAction.Kind, v.Disposition)
}

// TriageSelfcheck is the deterministic, no-I/O proof of the release-status fold: cutting or
// promoting a release, and a CI billing wall, genuinely wait on a person; fixing CI,
// repairing a workflow, reconciling version topology, cleaning the worktree, and weighing a
// stable promotion are the fleet's to drive; and a non-gating wait surfaces no split. It is
// the witness the CLI surfaces as `fak release triage-selfcheck`.
func TriageSelfcheck() error {
	human := []string{"cut_release", "cut_release_hot_tree", "promote_stable", "promote_release_branch", "fix_ci_billing"}
	for _, kind := range human {
		if !NextActionNeedsHuman(NextAction{Kind: kind}) {
			return fmt.Errorf("kind %q must wait on a person (a publish/release/budget authority)", kind)
		}
	}
	fleet := []string{"fix_ci", "confirm_ci", "fix_workflow", "fix_version_topology", "repair_stable_evidence", "clean_worktree", "hold", "consider_stable"}
	for _, kind := range fleet {
		v, ok := TriageNextAction(NextAction{Kind: kind})
		if !ok {
			return fmt.Errorf("kind %q must be a gating action the fold classifies", kind)
		}
		if v.NeedsHuman {
			return fmt.Errorf("kind %q is knowable engineering — it must NOT wait on a person, got %s", kind, v.Disposition)
		}
	}
	// Every gating kind in actionNextActions must have a Signal, or the fold silently
	// defaults a real ACTION to the fleet without ever classifying it.
	for kind := range actionNextActions {
		if _, ok := nextActionSignals[kind]; !ok {
			return fmt.Errorf("gating kind %q has no triage Signal — it would go unclassified", kind)
		}
	}
	// A non-gating move surfaces no split (no page, no fleet churn).
	if _, ok := TriageNextAction(NextAction{Kind: "wait"}); ok {
		return fmt.Errorf("a non-gating wait must not be triaged as an attention item")
	}
	if line := AttentionTriageLine(Status{NextAction: NextAction{Kind: "wait"}}); line != "" {
		return fmt.Errorf("a non-gating wait must surface no triage line, got %q", line)
	}
	if !ReleaseStatusTriageEnforced("enforce") || ReleaseStatusTriageEnforced("") || ReleaseStatusTriageEnforced("warn") {
		return fmt.Errorf("ReleaseStatusTriageEnforced must flip only on \"enforce\"")
	}
	return nil
}

// TriagedGatingKinds returns the gating next_action kinds sorted, split into the ones that
// wait on a person and the ones the fleet drives — a small helper for a caller (or a doc
// generator) that wants to show the whole split at once.
func TriagedGatingKinds() (needHuman, fleetDrives []string) {
	for kind := range nextActionSignals {
		if NextActionNeedsHuman(NextAction{Kind: kind}) {
			needHuman = append(needHuman, kind)
		} else {
			fleetDrives = append(fleetDrives, kind)
		}
	}
	sort.Strings(needHuman)
	sort.Strings(fleetDrives)
	return needHuman, fleetDrives
}
