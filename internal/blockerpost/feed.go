package blockerpost

import (
	"fmt"
	"net/url"
	"sort"
	"strings"
)

// Issue is the slice of a GitHub issue the blocker fold needs — exactly the fields a
// `gh issue list --json number,title,url,assignees,labels` payload carries. The CLI
// unmarshals the gh JSON into these; the fold itself stays pure (no gh, no network) so
// it is unit-testable.
type Issue struct {
	Number    int        `json:"number"`
	Title     string     `json:"title"`
	URL       string     `json:"url"`
	Assignees []Assignee `json:"assignees"`
	Labels    []Label    `json:"labels"`
}

// Assignee / Label mirror the nested gh JSON shapes (each is an object with a name).
type Assignee struct {
	Login string `json:"login"`
}

type Label struct {
	Name string `json:"name"`
}

// owned reports whether anyone is on the issue. An UNOWNED blocker is the one that needs
// a human to pick it up — that is the signal the feeder surfaces.
func (i Issue) owned() bool { return len(i.Assignees) > 0 }

// maxFeedLines caps how many issues a single roll-up lists so a large backlog does not
// flood the channel; the overflow is summarized.
const maxFeedLines = 12

// FoldIssues folds the open "blocker" backlog into ONE roll-up Blocker. The severity is
// derived from ownership — the honest mapping of the user's "lightweight status ones in
// background, human-operator ones surfaced":
//
//	0 issues                 -> SeverityClear    (all-clear heartbeat, no page)
//	>=1, at least one UNOWNED -> SeverityOperator (someone must pick these up — paged)
//	>=1, all OWNED            -> SeverityStatus   (tracked, in progress — recorded, no page)
//
// label is the issue label the backlog was filtered by (e.g. "blocked"), used in the
// prose and to build the "triage" link. repoURL is the repo's base URL (e.g.
// https://github.com/owner/repo); when set, an operator card links to the filtered
// issue list as its "do this next". Unowned issues are listed first (worst-first), each
// row carrying the number, title, and owner state.
func FoldIssues(issues []Issue, label, repoURL string) Blocker {
	if label == "" {
		label = "blocked"
	}
	if len(issues) == 0 {
		return Blocker{
			Severity: SeverityClear,
			Title:    "no standing blockers",
			Detail:   fmt.Sprintf("0 open `%s` issues — the board is clear.", label),
		}
	}

	// Stable worst-first order: unowned before owned, then by issue number.
	ordered := append([]Issue(nil), issues...)
	sort.SliceStable(ordered, func(a, c int) bool {
		if ordered[a].owned() != ordered[c].owned() {
			return !ordered[a].owned() // unowned first
		}
		return ordered[a].Number < ordered[c].Number
	})

	var unowned int
	for _, i := range ordered {
		if !i.owned() {
			unowned++
		}
	}

	b := Blocker{Ref: fmt.Sprintf("label:%s", label)}
	if unowned > 0 {
		b.Severity = SeverityOperator
		b.Title = fmt.Sprintf("%d blocker(s) need an owner", unowned)
		b.Detail = fmt.Sprintf("%d open `%s` issue(s); %d have no assignee and need a human to pick them up.", len(ordered), label, unowned)
		b.Action = "triage the blocked backlog"
		b.ActionURL = backlogURL(repoURL, label)
	} else {
		b.Severity = SeverityStatus
		b.Title = fmt.Sprintf("%d blocker(s) in progress", len(ordered))
		b.Detail = fmt.Sprintf("%d open `%s` issue(s), all assigned — tracked, no action needed.", len(ordered), label)
	}

	shown := ordered
	if len(shown) > maxFeedLines {
		shown = shown[:maxFeedLines]
	}
	for _, i := range shown {
		b.Lines = append(b.Lines, issueLine(i))
	}
	if len(ordered) > len(shown) {
		b.Lines = append(b.Lines, fmt.Sprintf("…and %d more (unowned-first)", len(ordered)-len(shown)))
	}
	return b
}

// issueLine renders one issue row: number + title, then either the assignees or an
// explicit UNOWNED marker so the human can see at a glance which ones are adrift.
func issueLine(i Issue) string {
	title := strings.TrimSpace(i.Title)
	if title == "" {
		title = "(untitled)"
	}
	ref := fmt.Sprintf("#%d", i.Number)
	if u := strings.TrimSpace(i.URL); u != "" {
		ref = fmt.Sprintf("<%s|#%d>", u, i.Number)
	}
	if !i.owned() {
		return fmt.Sprintf("%s %s · *UNOWNED*", ref, title)
	}
	var who []string
	for _, a := range i.Assignees {
		if n := strings.TrimSpace(a.Login); n != "" {
			who = append(who, "@"+n)
		}
	}
	return fmt.Sprintf("%s %s · %s", ref, title, strings.Join(who, " "))
}

// backlogURL builds the filtered "open + label" issue-list link for an operator card's
// "do this next" button. Returns "" when no repo base is known (the button is then
// simply omitted — the fallback line still names the action).
func backlogURL(repoURL, label string) string {
	repoURL = strings.TrimRight(strings.TrimSpace(repoURL), "/")
	if repoURL == "" {
		return ""
	}
	q := url.QueryEscape(fmt.Sprintf("is:open is:issue label:%s", label))
	return repoURL + "/issues?q=" + q
}
