package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/dogfoodcoverage"
)

func cmdDogfoodCoverage(argv []string) {
	os.Exit(runDogfoodCoverage(os.Stdout, os.Stderr, argv))
}

func runDogfoodCoverage(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak dogfood-coverage", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the control-pane JSON payload")
	check := fs.Bool("check", false, "exit 1 if any HARD KPI is unmet")
	workspace := fs.String("workspace", "", "repo root (default: auto)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dogfood-coverage: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}

	payload := dogfoodcoverage.Evaluate(root, nil)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(payload); err != nil {
			fmt.Fprintf(stderr, "fak dogfood-coverage: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, dogfoodcoverage.FormatReport(payload))
	}

	if *check && payload.DogfoodDebt > 0 {
		return 1
	}
	return 0
}
