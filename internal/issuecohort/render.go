package issuecohort

import (
	"fmt"
	"strings"
)

// Render produces the human-readable summary of a cohort Plan.
func Render(p Plan) string {
	var b strings.Builder
	fmt.Fprintf(&b, "issue-cohort: %d candidate(s) -> %d dispatchable, %d to-split, %d triage, %d refused\n",
		p.Total, p.Dispatchable, p.Subdividable, p.TriageOnly, p.Refused)
	fmt.Fprintf(&b, "  concurrency: %d wave(s), peak %d at once, %d colliding pair(s)\n",
		p.NumWaves, p.PeakConcurrency, p.CollisionPairs)
	if p.ChildIssueTotal > 0 {
		fmt.Fprintf(&b, "  subdivision: %d row(s) expand to ~%d child issue(s)\n",
			p.Subdividable, p.ChildIssueTotal)
	}
	if p.DuplicateKeys > 0 {
		fmt.Fprintf(&b, "  duplicate keys: %d extra occurrence(s) across %d key(s) (rerun should update, not create)\n",
			p.DuplicateKeys, len(p.Duplicates))
	}

	for _, w := range p.Waves {
		fmt.Fprintf(&b, "  wave %d: %d leaf/leaves, step_budget=%d\n", w.Index, w.Size, w.StepBudget)
		for _, m := range w.Members {
			fmt.Fprintf(&b, "    - %s%s%s\n", m.Key, laneSuffix(m.Lane), pathSuffix(m.Paths))
		}
	}
	if len(p.Subdivide) > 0 {
		fmt.Fprintf(&b, "  split-first (%d):\n", len(p.Subdivide))
		for _, s := range p.Subdivide {
			fmt.Fprintf(&b, "    - %s: %s (steps=%d -> ~%d child)\n",
				s.Key, strings.Join(s.Reasons, ","), s.ExpectedSteps, s.ChildIssueBudget)
		}
	}
	if len(p.Triage) > 0 {
		fmt.Fprintf(&b, "  triage (%d):\n", len(p.Triage))
		for _, t := range p.Triage {
			fmt.Fprintf(&b, "    - [%s] %s: %s\n", t.Dispatchability, t.Key, strings.Join(t.Reasons, ","))
		}
	}
	if len(p.Duplicates) > 0 {
		fmt.Fprintf(&b, "  duplicates (%d):\n", len(p.Duplicates))
		for _, d := range p.Duplicates {
			fmt.Fprintf(&b, "    - %s x%d\n", d.Key, d.Count)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

func laneSuffix(lane string) string {
	if lane == "" {
		return ""
	}
	return "  lane=" + lane
}

func pathSuffix(paths []string) string {
	if len(paths) == 0 {
		return ""
	}
	return "  paths=" + strings.Join(paths, ",")
}
