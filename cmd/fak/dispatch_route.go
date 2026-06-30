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
	fmt.Fprintf(&b, "  routed=%d unrouted=%d skipped-human=%d coverage=%s\n",
		router.Counts.Routed, router.Counts.Unrouted, router.Counts.SkippedHumanBlocked, coverageWord(router.Coverage))
	lanes := make([]string, 0, len(router.Lanes))
	for lane := range router.Lanes {
		lanes = append(lanes, lane)
	}
	sort.Strings(lanes)
	for _, lane := range lanes {
		grp := router.Lanes[lane]
		fmt.Fprintf(&b, "  %-16s %3d issue(s): %s\n", lane, grp.Count, intList(grp.Issues))
	}
	if len(router.SkippedHumanBlocked) > 0 {
		fmt.Fprintf(&b, "  skipped-human-blocked: %d\n", len(router.SkippedHumanBlocked))
	}
	return b.String()
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
