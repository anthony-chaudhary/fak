package issueorchestrator

import (
	"fmt"
	"strings"
	"text/tabwriter"
)

// RenderWaves formats a Plan as human-readable terminal text.
func RenderWaves(plan Plan) string {
	var b strings.Builder

	b.WriteString("=== FAK ISSUE ORCHESTRATOR: CONCURRENT SAFE WAVE PLAN ===\n")
	b.WriteString(fmt.Sprintf("Campaign Scope:      %d total issue(s) evaluated · %d dispatchable · %d subdivide · %d triage\n",
		plan.TotalIssues, plan.Dispatchable, plan.Subdividable, plan.TriageOnly,
	))
	b.WriteString(fmt.Sprintf("Plan Summary:        %d wave(s) planned · %d total worker slot(s) · %d total step budget\n",
		plan.TotalWaves, plan.PlannedIssues, plan.PlannedSteps,
	))
	b.WriteString(fmt.Sprintf("Concurrency Cap:     Max %d concurrent workers per wave · Concurrency safety: pairwise tree-disjoint\n",
		plan.WaveSizeCap,
	))

	if len(plan.HeldIssues) > 0 || len(plan.HeldLanes) > 0 {
		b.WriteString(fmt.Sprintf("Held Excluded:       %d issue(s) excluded due to active lease(s) held in workspace (%s)\n",
			len(plan.HeldIssues), strings.Join(plan.HeldLanes, ", "),
		))
	}
	b.WriteString("\n")

	if len(plan.Waves) == 0 {
		b.WriteString("No waves scheduled (backlog drained, targets reached, or all issues held/triaged).\n")
	} else {
		for _, w := range plan.Waves {
			b.WriteString(fmt.Sprintf("--- [%s] Wave %d (%s) · %d worker(s) · Step Budget: %d ---\n",
				w.ID, w.Index, w.Safety, w.WaveSize, w.StepBudget,
			))
			if len(w.LeaseRegion) > 0 {
				b.WriteString(fmt.Sprintf("  Lease Region: %s\n", strings.Join(w.LeaseRegion, ", ")))
			}
			if len(w.LeaseLanes) > 0 {
				b.WriteString(fmt.Sprintf("  Lease Lanes:  %s\n", strings.Join(w.LeaseLanes, ", ")))
			}

			tw := tabwriter.NewWriter(&b, 0, 0, 2, ' ', 0)
			for _, iss := range w.Issues {
				pathsStr := strings.Join(iss.Paths, ", ")
				if pathsStr == "" {
					pathsStr = "lane-wide"
				}
				laneStr := iss.Lane
				if laneStr == "" {
					laneStr = "unspecified"
				}
				numStr := fmt.Sprintf("#%d", iss.Number)
				if iss.Number == 0 {
					numStr = iss.Key
				}
				fmt.Fprintf(tw, "  %s\t[%s]\t(steps: %d)\t%s\tpaths: %s\n",
					numStr, laneStr, iss.ExpectedSteps, iss.Title, pathsStr,
				)
			}
			tw.Flush()
			b.WriteString("\n")
		}
	}

	if len(plan.Subdivide) > 0 {
		b.WriteString(fmt.Sprintf("Subdivide Queue (%d epics requiring decomposition before dispatch):\n", len(plan.Subdivide)))
		subLimit := len(plan.Subdivide)
		if subLimit > 5 {
			subLimit = 5
		}
		for i := 0; i < subLimit; i++ {
			s := plan.Subdivide[i]
			numStr := fmt.Sprintf("#%d", s.IssueNumber)
			if s.IssueNumber == 0 {
				numStr = s.Key
			}
			b.WriteString(fmt.Sprintf("  - %s: %s (steps: %d, child budget: %d)\n",
				numStr, s.Title, s.ExpectedSteps, s.ChildIssueBudget,
			))
		}
		if len(plan.Subdivide) > subLimit {
			b.WriteString(fmt.Sprintf("  ... and %d more (run with --subdivide to view all)\n", len(plan.Subdivide)-subLimit))
		}
		b.WriteString("\n")
	}

	if len(plan.Triage) > 0 {
		b.WriteString(fmt.Sprintf("Triage Queue (%d issues requiring scope/acceptance repair):\n", len(plan.Triage)))
		triLimit := len(plan.Triage)
		if triLimit > 5 {
			triLimit = 5
		}
		for i := 0; i < triLimit; i++ {
			t := plan.Triage[i]
			numStr := fmt.Sprintf("#%d", t.IssueNumber)
			if t.IssueNumber == 0 {
				numStr = t.Key
			}
			b.WriteString(fmt.Sprintf("  - %s: %s [%s]\n", numStr, t.Title, t.Dispatchability))
		}
		if len(plan.Triage) > triLimit {
			b.WriteString(fmt.Sprintf("  ... and %d more (run with --triage to view all)\n", len(plan.Triage)-triLimit))
		}
		b.WriteString("\n")
	}

	return b.String()
}

// MarkdownWaves formats a Plan as markdown text.
func MarkdownWaves(plan Plan) string {
	var b strings.Builder

	b.WriteString("# Issue Orchestrator: Concurrent Safe Wave Plan\n\n")
	b.WriteString(fmt.Sprintf("- **Campaign Scope**: %d evaluated, %d dispatchable, %d subdivide, %d triage\n",
		plan.TotalIssues, plan.Dispatchable, plan.Subdividable, plan.TriageOnly,
	))
	b.WriteString(fmt.Sprintf("- **Plan Summary**: %d wave(s), %d planned issue(s), %d total step budget\n",
		plan.TotalWaves, plan.PlannedIssues, plan.PlannedSteps,
	))
	b.WriteString(fmt.Sprintf("- **Concurrency**: Max %d workers per wave (pairwise tree-disjoint)\n\n",
		plan.WaveSizeCap,
	))

	if len(plan.Waves) > 0 {
		b.WriteString("## Planned Execution Waves\n\n")
		for _, w := range plan.Waves {
			b.WriteString(fmt.Sprintf("### %s (%s) — %d worker(s), %d steps\n\n",
				w.ID, w.Safety, w.WaveSize, w.StepBudget,
			))
			if len(w.LeaseRegion) > 0 {
				b.WriteString(fmt.Sprintf("- **Lease Region**: `%s`\n", strings.Join(w.LeaseRegion, "`, `")))
			}
			b.WriteString("| Issue | Lane | Steps | Title | Paths |\n")
			b.WriteString("|---|---|---|---|---|\n")
			for _, iss := range w.Issues {
				numStr := fmt.Sprintf("#%d", iss.Number)
				if iss.Number == 0 {
					numStr = iss.Key
				}
				pathsStr := strings.Join(iss.Paths, ", ")
				if pathsStr == "" {
					pathsStr = "lane-wide"
				}
				b.WriteString(fmt.Sprintf("| %s | %s | %d | %s | `%s` |\n",
					numStr, iss.Lane, iss.ExpectedSteps, iss.Title, pathsStr,
				))
			}
			b.WriteString("\n")
		}
	}

	if len(plan.Subdivide) > 0 {
		b.WriteString("## Subdivide Queue (Epics to Decompose)\n\n")
		b.WriteString("| Issue | Title | Steps | Child Budget |\n")
		b.WriteString("|---|---|---|---|\n")
		for _, s := range plan.Subdivide {
			numStr := fmt.Sprintf("#%d", s.IssueNumber)
			if s.IssueNumber == 0 {
				numStr = s.Key
			}
			b.WriteString(fmt.Sprintf("| %s | %s | %d | %d |\n",
				numStr, s.Title, s.ExpectedSteps, s.ChildIssueBudget,
			))
		}
		b.WriteString("\n")
	}

	return b.String()
}
