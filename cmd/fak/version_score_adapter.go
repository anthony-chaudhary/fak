package main

// fak version score-adapter converts per-file scorecard output into the flat
// module score map consumed by `fak version modules --scores` (#2466).

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/modver"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func runVersionScoreAdapter(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("version score-adapter", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "per-file scorecard JSON (default: stdin; use - for stdin)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak version score-adapter [--input FILE]")
		return 2
	}
	var (
		data []byte
		err  error
	)
	if *input == "" || *input == "-" {
		data, err = io.ReadAll(os.Stdin)
	} else {
		data, err = os.ReadFile(pathutil.ExpandTilde(*input))
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak version score-adapter: read: %v\n", err)
		return 1
	}
	scores, err := modver.ScorecardFileScores(data)
	if err != nil {
		fmt.Fprintf(stderr, "fak version score-adapter: %v\n", err)
		return 1
	}
	out, err := modver.MarshalModuleScores(scores)
	if err != nil {
		fmt.Fprintf(stderr, "fak version score-adapter: encode: %v\n", err)
		return 1
	}
	if _, err := fmt.Fprintln(stdout, string(out)); err != nil {
		fmt.Fprintf(stderr, "fak version score-adapter: write: %v\n", err)
		return 1
	}
	return 0
}
