package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/conceptusage"
)

func cmdConceptUsageScore(argv []string) { os.Exit(runConceptUsageScore(os.Stdout, os.Stderr, argv)) }

// runConceptUsageScore scores how much the agentic DEVELOPMENT of fak routes through
// fak's own concepts — the commit-discipline + lane-arbitration usage breadth and the
// verify/improve witness depth — re-derived from git and the .dos journals. Exit codes:
// 0 healthy (no debt), 1 carries debt, 2 usage/IO error.
func runConceptUsageScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak concept-usage-score", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	window := fs.Int("commits", conceptusage.DefaultCommitWindow, "how many recent commits define 'agentic dev now'")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak concept-usage-score: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := conceptusage.Build(conceptusage.Options{
		Root:         root,
		CommitWindow: *window,
	})
	if *comparePath != "" {
		base, ok := readCompareBase(stderr, "fak concept-usage-score", *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprintln(stdout, conceptusage.Compare(payload, base))
		if payload.OK {
			return 0
		}
		return 1
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak concept-usage-score: encode json: %v\n", err)
			return 1
		}
	} else if *asMarkdown {
		fmt.Fprint(stdout, conceptusage.Markdown(payload))
	} else {
		fmt.Fprintln(stdout, conceptusage.Render(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}
