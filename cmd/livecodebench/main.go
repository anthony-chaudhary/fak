package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/livecodebench"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(argv []string) int {
	fs := flag.NewFlagSet("livecodebench", flag.ContinueOnError)
	fixture := fs.String("fixture", "internal/livecodebench/testdata/fixture.json", "path to committed LiveCodeBench smoke fixture")
	asJSON := fs.Bool("json", false, "print the smoke report as JSON")
	check := fs.Bool("check", false, "fail if the smoke report is not shape-valid or if result_claim_allowed is true")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "livecodebench: unexpected positional arguments")
		return 2
	}
	f, err := livecodebench.LoadFile(*fixture)
	if err != nil {
		fmt.Fprintf(os.Stderr, "livecodebench: %v\n", err)
		return 1
	}
	report := livecodebench.SmokeReport(f)
	if *check {
		if err := livecodebench.ValidateSmokeReport(report); err != nil {
			fmt.Fprintf(os.Stderr, "livecodebench: %v\n", err)
			return 1
		}
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(os.Stderr, "livecodebench: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Printf("LiveCodeBench fixture smoke: %d question(s), %d scenario(s), result_claim_allowed=%v\n",
		report.Questions, len(report.Scenarios), report.ResultClaimAllowed)
	for _, s := range report.Scenarios {
		fmt.Printf("  - %s: %d\n", s.Scenario, s.Questions)
	}
	return 0
}
