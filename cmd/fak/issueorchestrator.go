package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issueorchestrator"
)

func cmdIssueOrchestrator(argv []string) {
	os.Exit(runIssueOrchestrator(os.Stdout, os.Stderr, argv))
}

func runIssueOrchestrator(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak issue-orchestrator", flag.ContinueOnError)
	fs.SetOutput(stderr)

	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	fromIssues := fs.String("from-issues", "", "path to GitHub issue JSON or - for stdin (gh issue list --json ...)")
	fromPlan := fs.String("from-plan", "", "path to candidate plan JSON")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit plan markdown")
	waveSize := fs.Int("wave-size", 4, "maximum concurrent workers per wave")
	maxWaves := fs.Int("max-waves", 0, "maximum number of waves to plan (0 = all necessary)")
	targetIssues := fs.Int("target-issues", 0, "campaign target number of issues to resolve")
	var targetPoints int
	fs.IntVar(&targetPoints, "target-points", 0, "campaign target step budget points to retire")
	fs.IntVar(&targetPoints, "points", 0, "alias for --target-points")
	excludeIssuesStr := fs.String("exclude-issues", "", "comma-separated list of issue numbers to exclude")
	excludeLanes := fs.String("exclude-lanes", "", "comma-separated list of lanes to exclude")
	noDetectHeld := fs.Bool("no-detect-held", false, "disable auto-detection of currently held leases in .dos")
	comparePath := fs.String("compare", "", "compare against a prior --json baseline payload")
	check := fs.Bool("check", false, "gate mode: exit non-zero if active dispatchable issues remain")
	subdivideOnly := fs.Bool("subdivide", false, "show only the subdivide queue (epics needing decomposition)")
	triageOnly := fs.Bool("triage", false, "show only the triage queue (issues needing scope clarification)")

	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak issue-orchestrator: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	// 1. Load issues from input
	var issues []issueorchestrator.Issue
	var err error

	inputPath := *fromIssues
	if inputPath == "" {
		inputPath = *fromPlan
	}

	issues, err = issueorchestrator.LoadIssues(inputPath, root)
	if err != nil {
		fmt.Fprintf(stderr, "fak issue-orchestrator: %v\n", err)
		return 2
	}

	// 2. Parse exclusion flags
	var excludedIssues []int
	if *excludeIssuesStr != "" {
		for _, part := range strings.Split(*excludeIssuesStr, ",") {
			trimmed := strings.TrimSpace(part)
			trimmed = strings.TrimPrefix(trimmed, "#")
			if num, err := strconv.Atoi(trimmed); err == nil && num > 0 {
				excludedIssues = append(excludedIssues, num)
			}
		}
	}

	var excludedLanesList []string
	if *excludeLanes != "" {
		for _, part := range strings.Split(*excludeLanes, ",") {
			trimmed := strings.TrimSpace(part)
			if trimmed != "" {
				excludedLanesList = append(excludedLanesList, trimmed)
			}
		}
	}

	// 3. Generate wave plan
	plan := issueorchestrator.PlanWaves(issues, issueorchestrator.WavePlanOptions{
		WaveSize:       *waveSize,
		MaxWaves:       *maxWaves,
		TargetIssues:   *targetIssues,
		TargetPoints:   targetPoints,
		ExcludedIssues: excludedIssues,
		ExcludedLanes:  excludedLanesList,
		AutoDetectHeld: !*noDetectHeld,
		WorkspaceRoot:  root,
	})

	// 4. Handle baseline comparison if requested
	if *comparePath != "" {
		baseBytes, err := os.ReadFile(*comparePath)
		if err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator: read compare baseline: %v\n", err)
			return 2
		}
		var base issueorchestrator.Plan
		if err := json.Unmarshal(baseBytes, &base); err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator: decode compare baseline JSON: %v\n", err)
			return 2
		}

		if *asJSON {
			cmpRes := issueorchestrator.Compare(plan, base)
			if err := writeIndentedJSON(stdout, cmpRes); err != nil {
				fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
				return 1
			}
			return 0
		}

		fmt.Fprint(stdout, issueorchestrator.CompareReport(plan, base))
		if *check && plan.PlannedIssues > 0 {
			return 1
		}
		return 0
	}

	// 5. Handle queue filters
	if *subdivideOnly {
		if *asJSON {
			if err := writeIndentedJSON(stdout, plan.Subdivide); err != nil {
				fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
				return 1
			}
			return 0
		}
		if len(plan.Subdivide) == 0 {
			fmt.Fprintln(stdout, "No issues in subdivide queue.")
			return 0
		}
		fmt.Fprintf(stdout, "Subdivide Queue (%d epics requiring decomposition before dispatch):\n", len(plan.Subdivide))
		for _, s := range plan.Subdivide {
			fmt.Fprintf(stdout, "  - #%d: %s (steps: %d, child budget: %d)\n", s.IssueNumber, s.Title, s.ExpectedSteps, s.ChildIssueBudget)
		}
		return 0
	}

	if *triageOnly {
		if *asJSON {
			if err := writeIndentedJSON(stdout, plan.Triage); err != nil {
				fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
				return 1
			}
			return 0
		}
		if len(plan.Triage) == 0 {
			fmt.Fprintln(stdout, "No issues in triage queue.")
			return 0
		}
		fmt.Fprintf(stdout, "Triage Queue (%d issues requiring scope/acceptance repair):\n", len(plan.Triage))
		for _, t := range plan.Triage {
			fmt.Fprintf(stdout, "  - #%d: %s [%s]\n", t.IssueNumber, t.Title, t.Dispatchability)
		}
		return 0
	}

	// 6. Normal output
	switch {
	case *asJSON:
		if err := writeIndentedJSON(stdout, plan); err != nil {
			fmt.Fprintf(stderr, "fak issue-orchestrator: encode json: %v\n", err)
			return 1
		}
	case *asMarkdown:
		fmt.Fprint(stdout, issueorchestrator.MarkdownWaves(plan))
	default:
		fmt.Fprint(stdout, issueorchestrator.RenderWaves(plan))
	}

	if *check && plan.PlannedIssues > 0 {
		return 1
	}

	return 0
}
