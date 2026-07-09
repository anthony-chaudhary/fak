package blockerpost

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// This file is the decenter-the-human fold at the blocker-feed seam. FoldIssues
// pages the operator (SeverityOperator) the moment ANY issue is unowned — but
// "unowned" is not the same as "needs a person". Most unowned blockers are
// fleet-routable: a fresh context window can pick them up, they carry an obvious
// runnable next step, or they are oversized and decompose into a ticket. Only an
// issue that names authority a person actually holds (policy / auth / release /
// priority / trust) is a genuine operator page.
//
// FoldIssuesTriaged runs each unowned issue through internal/choicetriage and
// keeps the operator page only when at least one unowned issue is human-residual.
// It is the blocker-seam analogue of operatorbrief.TriageHumanBucket, and like the
// operator brief it soaks behind a mode switch (TriageEnforced) so the paging
// change can be observed before it flips fleet-wide.

// TriageEnforced reports whether the decenter-the-human fold is active for the
// given mode string. Only "enforce" (case-insensitive) flips paging; "warn", ""
// and anything else leave FoldIssues' ownership-only paging unchanged so the
// change can soak. Mirrors operatorbrief's enforce/warn soak switch.
func TriageEnforced(mode string) bool {
	return strings.EqualFold(strings.TrimSpace(mode), "enforce")
}

// issueSignal projects one issue onto a choicetriage.Signal. The title is the
// surfaced question and the labels are the "why"; both feed the authority test,
// so an issue whose title or labels name a policy/auth/release/priority decision
// triages to HumanResidual. The issue title is NOT mapped onto Action — an issue
// is a description of a problem, not a producer-identified runnable step.
func issueSignal(i Issue) choicetriage.Signal {
	var labels []string
	for _, l := range i.Labels {
		if n := strings.TrimSpace(l.Name); n != "" {
			labels = append(labels, n)
		}
	}
	return choicetriage.Signal{
		Source:   "blockers",
		Question: i.Title,
		Detail:   strings.Join(labels, " "),
	}
}

// FoldIssuesTriaged folds the backlog exactly like FoldIssues, then decenters the
// human: an unowned issue only keeps the operator page when it names authority a
// person holds. If every unowned issue is fleet-routable, the roll-up is recorded
// as background status — the fleet picks them up — and no one is paged. A backlog
// that is already clear or all-owned is returned unchanged (there is no page to
// decenter). The listed rows and triage link are preserved so an operator who
// looks anyway still sees the full backlog.
func FoldIssuesTriaged(issues []Issue, label, repoURL string) Blocker {
	b := FoldIssues(issues, label, repoURL)
	if b.Severity != SeverityOperator {
		return b
	}

	var unowned, residual int
	for _, i := range issues {
		if i.owned() {
			continue
		}
		unowned++
		if choicetriage.Triage(issueSignal(i)).NeedsHuman {
			residual++
		}
	}

	if label == "" {
		label = "blocked"
	}
	if residual == 0 {
		// Every unowned blocker is knowable, obvious, or decomposable — none is a
		// person's call. Record it, do not page.
		b.Severity = SeverityStatus
		b.Title = fmt.Sprintf("%d unowned blocker(s) — fleet-routable", unowned)
		b.Detail = fmt.Sprintf("%d open `%s` issue(s) unowned, but none name authority a person holds — the fleet picks them up, no page.", unowned, label)
		return b
	}
	// At least one genuine authority decision remains: keep paging, and name how
	// many of the unowned issues actually need a person.
	b.Detail += fmt.Sprintf(" %d of the %d unowned name authority only a person holds.", residual, unowned)
	return b
}

// TriageSelfcheck is the deterministic, no-I/O proof of the blocker-seam fold: a
// fleet-routable unowned backlog stops paging under triage, an authority-bearing
// one still pages, and the ownership-only baseline is unchanged. It mirrors
// operatorbrief.TriageSelfcheck at this seam.
func TriageSelfcheck() error {
	routable := []Issue{{Number: 1, Title: "regenerate the cadence report and rerun `fak cadence`"}}
	authority := []Issue{{Number: 2, Title: "approve the tagged release before publish", Labels: []Label{{Name: "release"}}}}

	// Baseline: the ownership-only fold pages on any unowned issue.
	if got := FoldIssues(routable, "blocked", "").Severity; got != SeverityOperator {
		return fmt.Errorf("baseline FoldIssues on an unowned issue = %q, want %q", got, SeverityOperator)
	}
	// Triaged: a purely fleet-routable unowned backlog must not page.
	if got := FoldIssuesTriaged(routable, "blocked", "").Severity; got != SeverityStatus {
		return fmt.Errorf("a fleet-routable unowned backlog triaged = %q, want %q (no page)", got, SeverityStatus)
	}
	// Triaged: an unowned issue that names authority still pages.
	if got := FoldIssuesTriaged(authority, "blocked", "").Severity; got != SeverityOperator {
		return fmt.Errorf("an authority-bearing unowned issue triaged = %q, want %q (page)", got, SeverityOperator)
	}
	// The soak switch only flips on "enforce".
	if TriageEnforced("") || TriageEnforced("warn") || !TriageEnforced("enforce") {
		return fmt.Errorf("TriageEnforced must flip only on \"enforce\"")
	}
	return nil
}
