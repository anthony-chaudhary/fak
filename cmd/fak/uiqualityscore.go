package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/uiquality"
)

func cmdUIQualityScore(argv []string) { os.Exit(runUIQualityScore(os.Stdout, os.Stderr, argv)) }

// runUIQualityScore grades fak's terminal UI/UX surface (the `fak console` panes,
// the `fak info` overlay, the `fak guard --split` launcher) on mechanical KPIs and
// emits the control-pane payload the unified scorecard folds. Read-only: it reads
// the git-tracked render sources and writes nothing.
func runUIQualityScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak ui-quality-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown (the committed snapshot body)")
	comparePath := fs.String("compare", "", "compare against a prior --json payload and prove the debt moved")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak ui-quality-scorecard: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := uiquality.Build(uiquality.Options{Root: root})

	if *comparePath != "" {
		base, ok := readCompareBase(stderr, "fak ui-quality-scorecard", *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprint(stdout, uiquality.Compare(payload, base))
		if payload.OK {
			return 0
		}
		return 1
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak ui-quality-scorecard: encode json: %v\n", err)
			return 1
		}
	} else if *asMarkdown {
		fmt.Fprint(stdout, uiquality.Markdown(payload))
	} else {
		fmt.Fprint(stdout, uiquality.Render(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}
