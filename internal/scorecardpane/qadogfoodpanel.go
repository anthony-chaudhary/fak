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
	// RootPointChange is the issue's `## Root-point change` statement — the named
	// at-origin control the work moves earlier, not the after-the-fact symptom.
	RootPointChange string
	// DoneCondition is the issue's `## Done condition`.
	DoneCondition string
}

// HasClosureWitness reports whether the issue declares a closure-witness command.
func (i QADogfoodIssue) HasClosureWitness() bool {
	return strings.TrimSpace(i.ClosureWitness) != ""
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
	Schema              string  `json:"schema"`
	Total               int     `json:"total"`
	OpenCount           int     `json:"open_count"`
	StaleCount          int     `json:"stale_count"`
	ClosureWitnessCount int     `json:"closure_witness_count"`
	RootPointCount      int     `json:"root_point_count"`
	RootPointPercent    float64 `json:"root_point_percent"`
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
// stale, closure-witness, then the root-point coverage). It performs no I/O and never
// panics: an empty panel still renders a well-formed "0 tracked" line.
func RenderQADogfoodPanel(p QADogfoodPanel) string {
	return fmt.Sprintf(
		"qa-dogfood issue health — %d tracked · %d open · %d stale · %d closure-witness · %d root-point (%.0f%% of tracked)",
		p.Total, p.OpenCount, p.StaleCount, p.ClosureWitnessCount, p.RootPointCount, p.RootPointPercent,
	)
}
