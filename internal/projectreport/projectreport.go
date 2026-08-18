// Package projectreport folds a GitHub ProjectsV2 board into the same
// schema/ok/verdict/finding/next_action control-pane envelope the milestone report
// uses (internal/milestonereport), so the board — "the fleet's single work pane" per
// .github/workflows/project-board-sync.yml — becomes an operator-visible dimension
// instead of a write-only sync target.
//
// The fold is pure: it takes the board items (each an issue with its Status /
// Generation / Priority single-select values — exactly what
// cmd/fak/dispatch_project_fields.go already reads for dispatch ranking) and counts
// them by field, flags items missing Status or Generation as unclassified, and emits a
// verdict. A board that could not be measured (no read:project scope, or
// FAK_DISPATCH_PROJECT_NUMBER unset) folds to a visible UNMEASURED verdict, never a
// silent green.
package projectreport

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/generation"
	"sort"
	"strings"
)

// Schema is the envelope version, mirroring milestonereport's schema string.
const Schema = "fak.project-report/1"

// unmeasured is the bucket key for an item missing a single-select value.
const unmeasured = "(unset)"

// generationOrder is the board's Generation single-select horizon, in ramp order
// (project-board-sync.yml maps gen/now|next|second-next|future to these options).
var generationOrder = generation.Order()

// Item is one board row: an issue and its single-select field values ("" when the
// field is unset on the board).
type Item struct {
	Issue      int    `json:"issue"`
	Status     string `json:"status,omitempty"`
	Generation string `json:"generation,omitempty"`
	Priority   string `json:"priority,omitempty"`
}

// FoldOpts carries the provenance stamps the ledger/trend need (same shape as
// milestonereport.FoldOpts).
type FoldOpts struct {
	Commit      string
	GeneratedAt string
	Date        string
}

// Report is the folded control-pane envelope.
type Report struct {
	Schema       string         `json:"schema"`
	OK           bool           `json:"ok"`
	Measured     bool           `json:"measured"`
	Verdict      string         `json:"verdict"`
	Finding      string         `json:"finding"`
	Total        int            `json:"total"`
	ByStatus     map[string]int `json:"by_status,omitempty"`
	ByGeneration map[string]int `json:"by_generation,omitempty"`
	ByPriority   map[string]int `json:"by_priority,omitempty"`
	Unclassified []int          `json:"unclassified,omitempty"`
	NextAction   string         `json:"next_action"`
	Commit       string         `json:"commit,omitempty"`
	GeneratedAt  string         `json:"generated_at,omitempty"`
	Date         string         `json:"date,omitempty"`
	// Trend is the per-tick delta vs the previous durable-ledger row (ledger.go),
	// attached by WithTrend after the pure fold so Fold itself stays trend-free. nil
	// until a caller trends the report against docs/project/history.jsonl.
	Trend *Trend `json:"trend,omitempty"`
}

// Unmeasured builds the fail-closed envelope for a board that could not be read. It is
// advisory-OK (not a hard failure) but visibly UNMEASURED, so a dormant board never
// reads as healthy.
func Unmeasured(reason string, opts FoldOpts) Report {
	if strings.TrimSpace(reason) == "" {
		reason = "board unreachable (read:project scope or FAK_DISPATCH_PROJECT_NUMBER absent)"
	}
	return Report{
		Schema:      Schema,
		OK:          true,
		Measured:    false,
		Verdict:     "UNMEASURED",
		Finding:     reason,
		NextAction:  "gh auth refresh -s read:project; set FAK_DISPATCH_PROJECT_NUMBER",
		Commit:      opts.Commit,
		GeneratedAt: opts.GeneratedAt,
		Date:        opts.Date,
	}
}

// Fold counts the measured board items into the envelope. An item missing Status OR
// Generation is unclassified — the drift the sync workflow cannot self-heal (a filed
// issue with no gen/* label yet). Verdict: ACTION on an empty board or any
// unclassified item, else OK.
func Fold(items []Item, opts FoldOpts) Report {
	r := Report{
		Schema:       Schema,
		Measured:     true,
		Total:        len(items),
		ByStatus:     map[string]int{},
		ByGeneration: map[string]int{},
		ByPriority:   map[string]int{},
		Commit:       opts.Commit,
		GeneratedAt:  opts.GeneratedAt,
		Date:         opts.Date,
	}
	for _, it := range items {
		bump(r.ByStatus, it.Status)
		bump(r.ByGeneration, it.Generation)
		bump(r.ByPriority, it.Priority)
		if strings.TrimSpace(it.Status) == "" || strings.TrimSpace(it.Generation) == "" {
			r.Unclassified = append(r.Unclassified, it.Issue)
		}
	}
	sort.Ints(r.Unclassified)

	switch {
	case r.Total == 0:
		r.Verdict = "ACTION"
		r.Finding = "board is empty"
		r.NextAction = "confirm project-board-sync is wired (FAK_PROJECT_URL/FAK_PROJECT_TOKEN)"
	case len(r.Unclassified) > 0:
		r.Verdict = "ACTION"
		r.Finding = fmt.Sprintf("%d/%d item(s) missing Status or Generation", len(r.Unclassified), r.Total)
		r.NextAction = "classify board items: " + joinIssues(r.Unclassified, 10)
	default:
		r.Verdict = "OK"
		r.Finding = fmt.Sprintf("%d item(s) on the board; all classified", r.Total)
		r.NextAction = "no action — board is populated and classified"
	}
	r.OK = r.Verdict == "OK"
	return r
}

func bump(m map[string]int, key string) {
	key = strings.TrimSpace(key)
	if key == "" {
		key = unmeasured
	}
	m[key]++
}

func joinIssues(issues []int, cap int) string {
	if len(issues) > cap {
		issues = issues[:cap]
	}
	parts := make([]string, len(issues))
	for i, n := range issues {
		parts[i] = fmt.Sprintf("#%d", n)
	}
	return strings.Join(parts, " ")
}

// Render is the human one-screen text fold, mirroring milestonereport.Render's shape.
func Render(r Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s (%s) @%s %s\n", r.Verdict, r.Finding, short(r.Commit), r.Date)
	if !r.Measured {
		fmt.Fprintf(&b, "  board     unmeasured\n")
		fmt.Fprintf(&b, "  -> %s\n", r.NextAction)
		return strings.TrimRight(b.String(), "\n")
	}
	fmt.Fprintf(&b, "  total     %d item(s) on the board\n", r.Total)
	if line := distLine(r.ByStatus, nil); line != "" {
		fmt.Fprintf(&b, "  status    %s\n", line)
	}
	if line := distLine(r.ByGeneration, generationOrder); line != "" {
		fmt.Fprintf(&b, "  horizon   %s\n", line)
	}
	if line := distLine(r.ByPriority, nil); line != "" {
		fmt.Fprintf(&b, "  priority  %s\n", line)
	}
	if len(r.Unclassified) > 0 {
		fmt.Fprintf(&b, "  drift     %d unclassified: %s\n", len(r.Unclassified), joinIssues(r.Unclassified, 10))
	}
	fmt.Fprintf(&b, "  -> %s\n", r.NextAction)
	return strings.TrimRight(b.String(), "\n")
}

// distLine renders a "k n · k n" distribution. When order is non-nil those keys lead
// (in the given order); the remainder sort by descending count then key, with the
// (unset) bucket always last.
func distLine(m map[string]int, order []string) string {
	if len(m) == 0 {
		return ""
	}
	seen := map[string]bool{}
	var keys []string
	for _, k := range order {
		if _, ok := m[k]; ok {
			keys = append(keys, k)
			seen[k] = true
		}
	}
	var rest []string
	for k := range m {
		if !seen[k] && k != unmeasured {
			rest = append(rest, k)
		}
	}
	sort.Slice(rest, func(i, j int) bool {
		if m[rest[i]] != m[rest[j]] {
			return m[rest[i]] > m[rest[j]]
		}
		return rest[i] < rest[j]
	})
	keys = append(keys, rest...)
	if _, ok := m[unmeasured]; ok {
		keys = append(keys, unmeasured)
	}
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s %d", k, m[k]))
	}
	return strings.Join(parts, " · ")
}

func short(commit string) string {
	if len(commit) > 12 {
		return commit[:12]
	}
	if commit == "" {
		return "unknown"
	}
	return commit
}
