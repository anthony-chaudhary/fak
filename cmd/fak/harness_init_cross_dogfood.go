package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/harnessinit"
)

var harnessCrossDogfoodRun = harnessinit.CrossDogfood

func runHarnessCrossDogfood(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness cross-dogfood", flag.ContinueOnError)
	fs.SetOutput(stderr)
	selfcheck := fs.Bool("selfcheck", false, "run the offline three-host conformance matrix")
	jsonOut := fs.Bool("json", false, "emit the machine-readable matrix")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if !*selfcheck || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness cross-dogfood --selfcheck [--json]")
		return 2
	}
	repoRoot, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(stderr, "fak harness cross-dogfood: %v\n", err)
		return 1
	}
	matrix, err := harnessCrossDogfoodRun(context.Background(), repoRoot)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness cross-dogfood: %v\n", err)
		return 1
	}
	if *jsonOut {
		encoder := json.NewEncoder(stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(matrix); err != nil {
			fmt.Fprintf(stderr, "fak harness cross-dogfood: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "harness-cross-dogfood OK hosts=%d subsystems_per_host=%d drift_refusals=%d platform=%s\n", matrix.Hosts, matrix.SubsystemsPerHost, matrix.DriftRefusals, matrix.Platform)
	return 0
}
