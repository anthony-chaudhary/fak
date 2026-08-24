package main

// sota_coverage_scorecard.go is the thin CLI shell over internal/sotacoverage: it resolves
// the workspace, folds the card through the shared pkg/scorecard kernel, and owns the
// standard family surface --json (the control-pane payload) / --markdown (the committed
// snapshot for the published page) / --compare (the prove-the-debt-drop regression gate
// against a prior --json payload). --check keeps owning the exit code, so the gate reds
// only when the operator asks for it.

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/sotacoverage"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func cmdSOTACoverageScorecard(argv []string) {
	os.Exit(runSOTACoverageScorecard(os.Stdout, os.Stderr, argv))
}

func runSOTACoverageScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sota-coverage-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit the control-pane JSON payload")
	asMarkdown := fs.Bool("markdown", false, "emit the committed snapshot body")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	today := fs.String("today", "", "evaluate freshness against this YYYY-MM-DD date")
	check := fs.Bool("check", false, "exit nonzero when there is any HARD sota-debt")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	root := *workspace
	if root == "" {
		root = resolveRoot("")
		if root == "" {
			root = "."
		}
	}
	payload := sotacoverage.Collect(root, *today)

	compareExit := 0
	if *check && !payload.OK {
		compareExit = 1
	}
	if code, done := emitScorecardComparison(stdout, stderr, "sota-coverage-scorecard", *comparePath, compareExit, func(base map[string]any) string {
		return scorecard.Compare(payload, base, sotacoverage.DebtKey)
	}); done {
		return code
	}

	switch {
	case *asJSON:
		if err := writeIndentedJSONNoEscape(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "sota-coverage-scorecard: encode json: %v\n", err)
			return 2
		}
	case *asMarkdown:
		fmt.Fprint(stdout, scorecard.Markdown(payload, sotacoverage.MarkdownDoc(payload)))
	default:
		fmt.Fprintln(stdout, scorecard.Render(payload, sotacoverage.DebtKey))
	}

	if *check && !payload.OK {
		return 1
	}
	return 0
}
