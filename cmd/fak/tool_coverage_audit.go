package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/toolcoverage"
)

func cmdToolCoverageAudit(argv []string) { os.Exit(runToolCoverageAudit(os.Stdout, os.Stderr, argv)) }

func runToolCoverageAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tool-coverage-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	minCoverage := fs.Float64("min-coverage", -1, "advisory floor pct on load-bearing coverage")
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
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	var floor *float64
	if *minCoverage >= 0 {
		v := *minCoverage
		floor = &v
	}
	payload, err := toolcoverage.Collect(root, floor)
	if err != nil {
		fmt.Fprintf(stderr, "tool-coverage-audit: %v\n", err)
		return 2
	}
	if *asJSON {
		data, err := toolcoverage.MarshalJSON(payload)
		if err != nil {
			fmt.Fprintf(stderr, "tool-coverage-audit: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(data))
	} else {
		fmt.Fprintln(stdout, toolcoverage.Render(payload))
	}
	if !payload.OK {
		return 1
	}
	return 0
}
