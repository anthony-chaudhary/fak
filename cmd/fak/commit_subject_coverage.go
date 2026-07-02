package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/commitsubject"
)

func cmdCommitSubjectCoverage(argv []string) {
	os.Exit(runCommitSubjectCoverage(os.Stdout, os.Stderr, argv))
}

func runCommitSubjectCoverage(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit-subject-coverage", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	last := fs.Int("last", commitsubject.DefaultLast, "commits to scan")
	minCoverage := fs.Float64("min-coverage", -1, "advisory floor pct")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
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
	var floor *float64
	if *minCoverage >= 0 {
		v := *minCoverage
		floor = &v
	}
	payload := commitsubject.Collect(root, *last, floor, nil)
	if *asJSON {
		data, err := json.MarshalIndent(payload, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "commit-subject-coverage: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintln(stdout, commitsubject.Render(payload))
	}
	if !payload.OK {
		return 1
	}
	return 0
}
