package benchpost

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// planRow is one bench_plan.py --json entry (a per_machine_next value or a ranked[]
// element — they share a shape). Only the fields the run-request post renders are
// decoded.
type planRow struct {
	MachineID        string `json:"machine_id"`
	WorkloadKind     string `json:"workload_kind"`
	Model            string `json:"model"`
	Precision        string `json:"precision"`
	Intent           string `json:"intent"`
	Feasible         bool   `json:"feasible"`
	SuggestedCommand string `json:"suggested_command"`
}

// Plan is the bench_plan.py --json payload (schema benchmark/plan...). per_machine_next
// is keyed by machine_id; ranked is the global do-next list. honesty carries the
// planner's own PLAN-ONLY banner, surfaced verbatim so the post never reads as "a run
// happened".
type Plan struct {
	Schema  string             `json:"schema"`
	OK      bool               `json:"ok"`
	Now     string             `json:"now"`
	Honesty string             `json:"honesty"`
	PerMach map[string]planRow `json:"per_machine_next"`
	Ranked  []planRow          `json:"ranked"`
}

// LoadPlan reads a bench_plan.py --json payload from disk.
func LoadPlan(path string) (*Plan, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return ParsePlan(b)
}

// ParsePlan decodes a bench_plan.py --json payload from raw bytes (used when the
// planner is invoked on the fly and its stdout folded directly).
func ParsePlan(b []byte) (*Plan, error) {
	var p Plan
	if err := json.Unmarshal(b, &p); err != nil {
		return nil, err
	}
	return &p, nil
}

// RequestFromPlan folds a bench_plan payload into a run-request Post: one line per
// machine's next test (`machine → workload (model/precision) [intent]`) plus a short
// suggested-command hint. The lead carries the planner's PLAN-ONLY honesty banner so
// the channel reads it as a REQUEST to run, not a report of a run. top caps the
// per-machine list at the most useful few (0 = all).
func RequestFromPlan(p *Plan, top int) Post {
	lead := strings.TrimSpace(p.Honesty)
	if lead == "" {
		lead = "PLAN ONLY — no benchmark was run; this is a request for the bench-nodes to run next."
	}
	if p.Now != "" {
		lead = fmt.Sprintf("%s (planned %s)", lead, p.Now)
	}

	// Stable machine order so the post is deterministic.
	machines := make([]string, 0, len(p.PerMach))
	for m := range p.PerMach {
		machines = append(machines, m)
	}
	sort.Strings(machines)
	if top > 0 && top < len(machines) {
		machines = machines[:top]
	}

	post := Post{
		Emoji: ":clipboard:",
		Title: "bench run-request — next test per machine",
		Lead:  lead,
	}
	for _, m := range machines {
		row := p.PerMach[m]
		post.Lines = append(post.Lines, formatPlanRow(row))
	}
	if len(post.Lines) == 0 {
		post.Lines = append(post.Lines, "_no feasible next test in the plan_")
	}
	return post
}

// formatPlanRow renders one plan row as a request line.
func formatPlanRow(row planRow) string {
	mp := row.Model
	if mp == "" || strings.EqualFold(mp, "none") {
		mp = "(agent workload)"
	} else if row.Precision != "" && !strings.EqualFold(row.Precision, "n/a") && !strings.EqualFold(row.Precision, "none") {
		mp = row.Model + "/" + row.Precision
	}
	intent := ""
	if row.Intent != "" {
		intent = " [" + row.Intent + "]"
	}
	line := fmt.Sprintf("`%s` → %s · %s%s", row.MachineID, row.WorkloadKind, mp, intent)
	if cmd := shortCommand(row.SuggestedCommand); cmd != "" {
		line += " — `" + cmd + "`"
	}
	return line
}

// shortCommand trims the planner's "on <machine> (...): <cmd>  # HINT ..." suggestion
// down to the bare command for a compact channel line.
func shortCommand(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	// Strip a leading "on <machine> (...): " prefix.
	if i := strings.Index(s, "): "); i > 0 && strings.HasPrefix(s, "on ") {
		s = s[i+3:]
	}
	// Strip a trailing "  # HINT ..." comment.
	if i := strings.Index(s, "  #"); i > 0 {
		s = s[:i]
	}
	return strings.TrimSpace(s)
}
