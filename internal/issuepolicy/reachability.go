package issuepolicy

// Reachability is the guard surface for the one failure mode a blanket change to
// this contract can cause: stranding the EXISTING GitHub backlog out of dispatch.
//
// IsDispatchable (internal/dispatchtick) gates every open issue through
// ReviewIssueDraft. That makes the required-section/label set here a load-bearing
// schema: tighten it (add a required section, add a label to the non-leaf set) and
// every already-filed ticket that lacks the new requirement silently becomes
// unreachable for dispatch. This file gives a future author two things a plain
// unit test does not:
//
//  1. ReachabilityContract() — a behavior-DERIVED manifest of exactly which
//     sections and labels gate dispatch reachability, computed by probing the
//     reviewer so it can never drift from the real gate. A pinned test turns any
//     blanket change into a loud, reviewed diff.
//  2. SummarizeReachability() — the offline core behind `fak issue contract
//     --from-issues`, so a representative corpus can assert a dispatchable-share
//     floor and fail CI when a change collapses it, instead of the collapse only
//     showing up as a starved live dispatch loop.
//
// Neither replaces the live audit; they make its regression case enforceable
// offline, before the change ships.

import (
	"sort"
	"strings"
)

// reachabilitySection is one heading+body block of the canonical dispatchable
// issue used to probe the reachability gate.
type reachabilitySection struct {
	Heading string
	Body    string
}

// canonicalDispatchableSections renders every issue-body section a fully-scoped
// leaf carries, in the order CandidateFromIssueDraft reads them. It is the probe
// input for ReachabilityContract: removing one section and re-reviewing tells us
// whether that section GATES dispatch reachability. The bodies deliberately avoid
// every private-boundary needle so that removing a section is the only thing that
// can change the verdict.
func canonicalDispatchableSections() []reachabilitySection {
	return []reachabilitySection{
		{"Parent context", "Parent: dispatch reachability guard; this is the next scoped leaf under it."},
		{"Current state", "The dispatch gate reviews every open issue through the shared issue contract."},
		{"Why this is next", "A blanket contract change can strand the existing backlog, so it is guarded now."},
		{"Working spine", "Existing ticket -> issue contract review -> dispatch lane stays reachable end to end."},
		{"In scope", "Keep the representative backlog dispatchable under the current contract."},
		{"Out of scope", "Do not loosen the contract or rewrite the router; only guard the regression."},
		{"Done condition", "The canonical leaf reviews as dispatchable and the corpus floor holds."},
		{"Witness", "go test ./internal/issuecontract -run TestDispatchReachability"},
		{"Acceptance gate", "go test ./internal/issuecontract"},
		{"Closure binding", "Resolving commit cites #N and carries a matching (fak <leaf>) trailer."},
		{"Likely files", "`internal/issuecontract/reachability.go`"},
		{"Lane", "tools"},
	}
}

func renderReachabilitySections(sections []reachabilitySection) string {
	var b strings.Builder
	for _, s := range sections {
		b.WriteString("## ")
		b.WriteString(s.Heading)
		b.WriteString("\n\n")
		b.WriteString(s.Body)
		b.WriteString("\n\n")
	}
	return b.String()
}

// CanonicalDispatchableDraft is a fully-scoped leaf issue that reviews as
// dispatchable under the current contract. It is the fixed point the reachability
// probe measures against and is exported so a guard test can assert the base case
// stays dispatchable before trusting the probe.
func CanonicalDispatchableDraft() IssueDraft {
	return IssueDraft{
		Number: 9000,
		Title:  "guard: keep the existing backlog reachable for dispatch",
		Body:   renderReachabilitySections(canonicalDispatchableSections()),
	}
}

// ReachabilityProbeOrdinaryLabels are common area/type/priority labels an existing
// ticket routinely carries. NONE of them may gate dispatch reachability: a blanket
// change that adds one to a non-leaf/exclusion set would strand a whole slice of
// the backlog, and the pinned test fails when one starts gating.
var ReachabilityProbeOrdinaryLabels = []string{
	"agentic-serving", "bug", "compute", "docs", "documentation", "enhancement",
	"gateway", "gen/now", "generation", "good first issue", "help wanted",
	"priority/P0", "priority/P1", "substrate",
}

// ReachabilityProbeTriageLabels are the labels that MUST hold a ticket out of
// worker dispatch (epics/research/triage umbrellas). If a change stops one from
// gating, un-scoped work would dispatch as a leaf, so the pinned test fails.
var ReachabilityProbeTriageLabels = []string{
	"epic", "idea-scout", "needs-scope", "needs-triage", "research",
	"triage-only", "triage_only",
}

// DispatchReachabilityContract is the behavior-derived description of the blanket
// knobs that decide whether an existing GitHub ticket is reachable for dispatch:
// the issue-body sections whose absence strands a ticket (GatingSections), the
// labels that hold a ticket out of dispatch (GatingLabels), and the ordinary
// labels proven not to gate (SafeLabels). It is computed, never hand-maintained,
// so it cannot silently diverge from the real gate.
type DispatchReachabilityContract struct {
	GatingSections []string `json:"gating_sections"`
	GatingLabels   []string `json:"gating_labels"`
	SafeLabels     []string `json:"safe_labels"`
}

// ReachabilityContract probes ReviewIssueDraft against CanonicalDispatchableDraft
// to derive the current dispatch-reachability contract. It removes each canonical
// section (and adds each probe label) one at a time and records whether the
// verdict flips away from dispatchable — that is the exact, drift-proof set a
// blanket change would move.
func ReachabilityContract() DispatchReachabilityContract {
	base := CanonicalDispatchableDraft()
	out := DispatchReachabilityContract{}
	if ReviewIssueDraft(base, Options{}).Dispatchability != Dispatchable {
		// The base must itself be dispatchable or the probe is meaningless; leave the
		// sets empty so the guard test's separate base-case assertion is what fails,
		// with a clearer message than a mangled manifest.
		return out
	}

	sections := canonicalDispatchableSections()
	for i := range sections {
		variant := base
		variant.Body = renderReachabilitySections(removeReachabilitySection(sections, i))
		if ReviewIssueDraft(variant, Options{}).Dispatchability != Dispatchable {
			out.GatingSections = append(out.GatingSections, normalizeHeading(sections[i].Heading))
		}
	}

	probe := append(append([]string(nil), ReachabilityProbeOrdinaryLabels...), ReachabilityProbeTriageLabels...)
	for _, label := range probe {
		variant := base
		variant.Labels = []IssueLabel{{Name: label}}
		if ReviewIssueDraft(variant, Options{}).Dispatchability != Dispatchable {
			out.GatingLabels = append(out.GatingLabels, label)
		} else {
			out.SafeLabels = append(out.SafeLabels, label)
		}
	}

	sort.Strings(out.GatingSections)
	sort.Strings(out.GatingLabels)
	sort.Strings(out.SafeLabels)
	return out
}

func removeReachabilitySection(sections []reachabilitySection, skip int) []reachabilitySection {
	out := make([]reachabilitySection, 0, len(sections)-1)
	for i, s := range sections {
		if i == skip {
			continue
		}
		out = append(out, s)
	}
	return out
}

// ReachabilitySummary is an aggregate reachability readout over a set of existing
// GitHub issue rows: how many are dispatchable versus held, and the
// closed-vocabulary reasons the held ones carry. It is the offline-testable core
// behind `fak issue contract --from-issues`, reused by the corpus floor guard so a
// blanket change that collapses the dispatchable share fails a test.
type ReachabilitySummary struct {
	Total        int            `json:"total"`
	Dispatchable int            `json:"dispatchable"`
	TriageOnly   int            `json:"triage_only"`
	Refused      int            `json:"refused"`
	ByReason     map[string]int `json:"by_reason"`
}

// DispatchableFraction is the share of reviewed rows that are dispatchable, in
// [0,1]. An empty set is 0 so a floor assertion over no rows fails loudly rather
// than passing vacuously.
func (s ReachabilitySummary) DispatchableFraction() float64 {
	if s.Total == 0 {
		return 0
	}
	return float64(s.Dispatchable) / float64(s.Total)
}

// SummarizeReachability reviews every row and tallies its dispatchability and
// reasons. opt is passed through so a caller can measure the live-producer view
// (Options.Live) as well as the default open-backlog view.
func SummarizeReachability(rows []IssueDraft, opt Options) ReachabilitySummary {
	sum := ReachabilitySummary{ByReason: map[string]int{}}
	for _, row := range rows {
		review := ReviewIssueDraft(row, opt)
		sum.Total++
		switch review.Dispatchability {
		case Dispatchable:
			sum.Dispatchable++
		case TriageOnly:
			sum.TriageOnly++
		case Refused:
			sum.Refused++
		}
		for _, reason := range review.Reasons {
			sum.ByReason[reason]++
		}
	}
	return sum
}
