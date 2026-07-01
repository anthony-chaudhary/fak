package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func runDispatchRoute(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch route", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	root := strings.TrimSpace(*workspace)
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch route: getwd: %v\n", err)
			return 1
		}
		root = wd
	}
	router, err := dispatchRouteIssues(root, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch route: %v\n", err)
		return 1
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, router); err != nil {
			fmt.Fprintf(stderr, "fak dispatch route: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderDispatchRoute(router))
	}
	if router.OK {
		return 0
	}
	return 1
}

func renderDispatchRoute(router dispatchtick.RouterPayload) string {
	var b strings.Builder
	fmt.Fprintf(&b, "dispatch route: %s (%s)\n", router.Verdict, okWord(router.OK))
	fmt.Fprintf(&b, "  %s\n", router.Reason)
	fmt.Fprintf(&b, "  routed=%d steps=%d unrouted=%d skipped=%d coverage=%s\n",
		router.Counts.Routed, router.Counts.RoutedStepBudget, router.Counts.Unrouted, router.Counts.SkippedHumanBlocked, coverageWord(router.Coverage))
	lanes := make([]string, 0, len(router.Lanes))
	for lane := range router.Lanes {
		lanes = append(lanes, lane)
	}
	sort.Strings(lanes)
	for _, lane := range lanes {
		grp := router.Lanes[lane]
		fmt.Fprintf(&b, "  %-16s %3d issue(s) %3d step(s): %s\n", lane, grp.Count, grp.StepBudget, intList(grp.Issues))
		for _, sublane := range grp.SubLanes {
			fmt.Fprintf(&b, "    split %-24s %3d issue(s) %3d step(s): %s\n",
				sublane.Prefix, sublane.Count, sublane.StepBudget, intList(sublane.Issues))
		}
	}
	for _, issue := range router.Issues {
		fmt.Fprintf(&b, "  candidate #%d lane=%s confidence=%s signal=%s",
			issue.Number, routeIssueLane(issue), emptyDash(issue.Confidence), emptyDash(issue.Signal))
		if issue.SignalConflict {
			fmt.Fprintf(&b, " conflict=true")
		}
		if issue.UnroutedReason != "" {
			fmt.Fprintf(&b, " reason=%s", issue.UnroutedReason)
		}
		fmt.Fprintln(&b)
	}
	if len(router.SkippedHumanBlocked) > 0 {
		fmt.Fprintf(&b, "  skipped: %d", len(router.SkippedHumanBlocked))
		if summary := skippedReasonSummary(router.Counts.SkippedByReason); summary != "" {
			fmt.Fprintf(&b, " (%s)", summary)
		}
		fmt.Fprintln(&b)
	}
	if len(router.UnroutableBacklog) > 0 {
		fmt.Fprintf(&b, "  unroutable_backlog: %d\n", len(router.UnroutableBacklog))
		for _, row := range router.UnroutableBacklog {
			fmt.Fprintf(&b, "    #%d bucket=%s reason=%s next=%s\n",
				row.Number, row.Bucket, row.Reason, row.NextAction)
		}
	}
	for _, queue := range router.RepairQueues {
		fmt.Fprintf(&b, "  repair_queue[%s]: %d issue(s) %d step(s)",
			queue.Kind, queue.Count, queue.StepBudget)
		if queue.ChildIssueBudget > 0 {
			fmt.Fprintf(&b, " child_issues=%d", queue.ChildIssueBudget)
		}
		if len(queue.Issues) > 0 {
			fmt.Fprintf(&b, " issues=%s", intList(queue.Issues))
		}
		fmt.Fprintf(&b, " next=%s\n", queue.NextAction)
		if summary := skippedReasonSummary(queue.ByReason); summary != "" {
			fmt.Fprintf(&b, "    reasons: %s\n", summary)
		}
	}
	return b.String()
}

func routeIssueLane(issue dispatchtick.IssueRoute) string {
	if issue.Lane == "" {
		return "(unrouted)"
	}
	return issue.Lane
}

func emptyDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

func skippedReasonSummary(counts map[string]int) string {
	if len(counts) == 0 {
		return ""
	}
	reasons := make([]string, 0, len(counts))
	for reason := range counts {
		reasons = append(reasons, reason)
	}
	sort.Strings(reasons)
	parts := make([]string, 0, len(reasons))
	for _, reason := range reasons {
		parts = append(parts, fmt.Sprintf("%s=%d", reason, counts[reason]))
	}
	return strings.Join(parts, ", ")
}

func coverageWord(c dispatchtick.RouterCoverage) string {
	switch {
	case c.Injected:
		return "injected"
	case c.Truncated:
		return "truncated"
	case c.Complete:
		return "complete"
	default:
		return "unknown"
	}
}

func intList(nums []int) string {
	if len(nums) == 0 {
		return "-"
	}
	parts := make([]string, len(nums))
	for i, n := range nums {
		parts[i] = fmt.Sprint(n)
	}
	return strings.Join(parts, ",")
}
