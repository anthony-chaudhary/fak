package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/maturity"
)

func cmdMaturity(argv []string) { os.Exit(runMaturity(os.Stdout, os.Stderr, argv)) }

// runMaturity scores where each declared fak capability sits on its lifecycle
// maturity ladder (proposed → prototyped → tested → dogfooded → benchmarked →
// default) and emits the next work item that would mature each one. Immaturity is
// never a defect; a ladder-skip (high-rung evidence over an unmet lower rung) is.
// Exit codes: 0 no ladder-skips, 1 carries skip-debt, 2 usage/IO error.
func runMaturity(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak maturity", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	asMarkdown := fs.Bool("markdown", false, "emit scorecard markdown")
	next := fs.Bool("next", false, "emit only the next-work backlog (ladder-skips first)")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	// `fak maturity next` is the natural spelling of the backlog view.
	if fs.NArg() == 1 && fs.Arg(0) == "next" {
		*next = true
	} else if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak maturity: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := maturity.Build(maturity.Options{Root: root})

	if *comparePath != "" {
		base, ok := readCompareBase(stderr, "fak maturity", *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprintln(stdout, maturity.Compare(payload, base))
		if payload.OK {
			return 0
		}
		return 1
	}
	switch {
	case *asJSON:
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak maturity: encode json: %v\n", err)
			return 1
		}
	case *asMarkdown:
		fmt.Fprint(stdout, maturity.Markdown(payload))
	case *next:
		fmt.Fprintln(stdout, maturity.RenderNext(payload))
	default:
		fmt.Fprintln(stdout, maturity.Render(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}
