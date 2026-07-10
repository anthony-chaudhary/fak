package scorecardpane

// qadogfoodpanel.go — issue #1982: QA dogfood issues are filed and closed, but the
// operator has no compact view of how the standing set is doing — how many are still
// open, how many have gone stale, how many carry a real closure witness, and what
// fraction still name the at-origin "root-point" fields that make a dogfood issue
// dispatchable instead of an after-the-fact cleanup item. This file adds that panel.
//
// Shape follows the gather/fold split contexthealth.go / contextstatus.go already
// draw: scorecardpane stays tier-1 and stdlib-only (see internal/architest's tier
// map), so it never imports internal/dogfoodissues and never shells `gh`. The caller
// (cmd/fak, reading `gh issue list --json number,state,body,updatedAt`) decodes each
// tracker row into one QADogfoodIssue; FoldQADogfoodPanel is the pure, deterministic
// fold from those rows into the four counts + the root-point percentage the issue's
// done condition names.

import (
	"fmt"
	"strings"
	"time"
)

// QADogfoodPanelSchema tags the machine-readable panel payload so a consumer keyed on
// the schema string can recognize it, mirroring the Schema convention controlpane.go
// and hygiene.go already use.
const QADogfoodPanelSchema = "fak-scorecard-qa-dogfood-panel/1"

// DefaultQADogfoodStaleHorizon is the age past which an OPEN QA dogfood issue with no
// tracker movement is counted stale. Fourteen days matches the "avoid broad cleanup,
// fix the root control first" cadence: an origin control that has not moved in two
// weeks is drifting into after-the-fact debt.
const DefaultQADogfoodStaleHorizon = 14 * 24 * time.Hour

// QADogfoodIssue is one QA-dogfood-spine issue's health-relevant fields, already read
// from the tracker by the caller. The impure gather (shelling `gh`, parsing the issue
// body's markdown sections) lives in cmd/fak; this struct is the pure fold's input,
// the same separation collect.go draws from Fold.
type QADogfoodIssue struct {
	// Number is the GitHub issue number, carried for render/debug context only.
	Number int
	// Open is true when the tracker state is open (not closed).
	Open bool
	// Stale is true when the issue is Open AND untouched past the staleness horizon.
	// The caller derives it (it needs a clock) — QADogfoodStale is the shared helper.
	Stale bool
	// ClosureWitness is the issue's declared `## Witness` / acceptance-gate command
	// (the go test line that binds its closure); "" when the issue names none.
	ClosureWitness string
	// WitnessRun is true when that closure-witness command was actually EXECUTED (or
	// bound to the resolving commit's test_run_witness) — not merely declared. The
	// impure run/binding lives in the caller (cmd/fak, sandboxed); this field carries
	// its recorded outcome into the pure fold. False for a witness that never ran —
	// the common case until the binding source (issue #3838) exists, which is why a
	// declared-but-unrun witness is honestly counted as UNPROVEN, not clean.
	WitnessRun bool
	// WitnessPassed is true when the executed witness PASSED. Only meaningful when
	// WitnessRun; a declared-but-unrun or a ran-and-failed witness leaves it false.
	WitnessPassed bool
	// RootPointChange is the issue's `## Root-point change` statement — the named
	// at-origin control the work moves earlier, not the after-the-fact symptom.
	RootPointChange string
	// DoneCondition is the issue's `## Done condition`.
	DoneCondition string
}

// HasClosureWitness reports whether the issue declares a closure-witness command.
// Declaration alone is NOT proof — see WitnessProven for the upgraded signal.
func (i QADogfoodIssue) HasClosureWitness() bool {
	return strings.TrimSpace(i.ClosureWitness) != ""
}

// WitnessProven reports whether the issue's closure is actually PROVEN: it declares a
// witness command, that command was run, and it passed. This is the upgrade #3839
// makes over HasClosureWitness — a witness that was declared but never run, or ran
// and failed, is theater rather than proof, and must NOT count as a clean closure
// witness. A closed issue that is not WitnessProven is closure debt (see
// QADogfoodPanel.UnprovenClosureCount).
func (i QADogfoodIssue) WitnessProven() bool {
	return i.HasClosureWitness() && i.WitnessRun && i.WitnessPassed
}

// HasRootPointFields reports whether the issue carries the at-origin "root-point"
// fields that make it dispatchable at origin: a root-point change statement AND a
// done condition. An issue missing either is an after-the-fact cleanup item, not an
// at-origin control — exactly the distinction the QA dogfood spine (#1961) polices.
func (i QADogfoodIssue) HasRootPointFields() bool {
	return strings.TrimSpace(i.RootPointChange) != "" &&
		strings.TrimSpace(i.DoneCondition) != ""
}

// QADogfoodStale reports whether an issue is stale: open, with a known last-touch
// time, untouched for longer than horizon. A zero updatedAt (unknown) is never stale
// — the panel undercounts rather than inventing staleness it cannot witness. A
// non-positive horizon falls back to DefaultQADogfoodStaleHorizon.
func QADogfoodStale(open bool, updatedAt, now time.Time, horizon time.Duration) bool {
	if !open || updatedAt.IsZero() {
		return false
	}
	if horizon <= 0 {
		horizon = DefaultQADogfoodStaleHorizon
	}
	return now.Sub(updatedAt) > horizon
}

// QADogfoodPanel is the folded health of a QA dogfood issue set: the total tracked,
// how many are open, how many are stale, how many carry a closure witness, and the
// count + percent that carry the root-point fields. Field tags let cmd/fak emit it as
// a `--json` payload alongside the other native scorecard panels.
type QADogfoodPanel struct {
	Schema     string `json:"schema"`
	Total      int    `json:"total"`
	OpenCount  int    `json:"open_count"`
	StaleCount int    `json:"stale_count"`
	// ClosureWitnessCount is how many issues DECLARE a closure-witness command
	// (present, not necessarily run). Kept as-is for continuity; it is the "field
	// present" signal #3839 upgrades but does not remove.
	ClosureWitnessCount int `json:"closure_witness_count"`
	// WitnessRunCount is how many witness-declaring issues actually had that witness
	// executed/bound (regardless of pass/fail) — the "was it run at all" signal.
	WitnessRunCount int `json:"witness_run_count"`
	// WitnessPassCount is how many issues are WitnessProven (declared + run + passed):
	// the clean closure-witness count, distinct from the merely-declared count above.
	WitnessPassCount int `json:"witness_pass_count"`
	// UnprovenClosureCount is the debt: CLOSED issues that declared a closure witness
	// which was NOT run-and-passed (ran and failed, or never executed) — a closure
	// that reads as witnessed but was never proven. This is the gap #3839 surfaces.
	UnprovenClosureCount int     `json:"unproven_closure_count"`
	RootPointCount       int     `json:"root_point_count"`
	RootPointPercent     float64 `json:"root_point_percent"`
}

// FoldQADogfoodPanel folds a set of QA dogfood issues into the panel counts. Pure and
// deterministic: the same rows always produce the same panel, independent of order.
// RootPointPercent is over Total and is 0 for an empty set — never a divide-by-zero.
func FoldQADogfoodPanel(issues []QADogfoodIssue) QADogfoodPanel {
	p := QADogfoodPanel{Schema: QADogfoodPanelSchema, Total: len(issues)}
	for _, it := range issues {
		if it.Open {
			p.OpenCount++
		}
		if it.Stale {
			p.StaleCount++
		}
		if it.HasClosureWitness() {
			p.ClosureWitnessCount++
		}
		if it.HasClosureWitness() && it.WitnessRun {
			p.WitnessRunCount++
		}
		if it.WitnessProven() {
			p.WitnessPassCount++
		}
		// Closure debt: a closed issue that declared a witness but never proved it
		// (failed or never run) reads as witnessed yet was never confirmed green.
		if !it.Open && it.HasClosureWitness() && !it.WitnessProven() {
			p.UnprovenClosureCount++
		}
		if it.HasRootPointFields() {
			p.RootPointCount++
		}
	}
	if p.Total > 0 {
		p.RootPointPercent = 100 * float64(p.RootPointCount) / float64(p.Total)
	}
	return p
}

// RenderQADogfoodPanel renders the panel as ONE concise, deterministic line — the
// compact control-pane card the issue asks for, in the done-condition's order (open,
// stale, closure-witness, then the root-point coverage). The closure-witness figure
// now DISTINGUISHES declared from proven: "N closure-witness (P proven, U
// unproven-closed)" so a declared-but-unrun-or-failed witness can never masquerade as
// a clean closure. It performs no I/O and never panics: an empty panel still renders
// a well-formed "0 tracked" line.
func RenderQADogfoodPanel(p QADogfoodPanel) string {
	return fmt.Sprintf(
		"qa-dogfood issue health — %d tracked · %d open · %d stale · %d closure-witness (%d proven, %d unproven-closed) · %d root-point (%.0f%% of tracked)",
		p.Total, p.OpenCount, p.StaleCount, p.ClosureWitnessCount, p.WitnessPassCount, p.UnprovenClosureCount, p.RootPointCount, p.RootPointPercent,
	)
}
