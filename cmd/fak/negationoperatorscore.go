package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

func cmdNegationOperatorScore(argv []string) {
	os.Exit(runNegationOperatorScore(os.Stdout, os.Stderr, argv))
}

func runNegationOperatorScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak score negation_operator", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit JSON")
	benchmarkDelta := fs.Float64("benchmark-delta", 0, "witnessed operator-minus-baseline delta in percentage points")
	unknownRate := fs.Float64("unknown-fallback-rate", 0, "witnessed safe UNKNOWN fallback rate [0,1]")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *unknownRate < 0 || *unknownRate > 1 {
		fmt.Fprintln(stderr, "fak score negation_operator: --unknown-fallback-rate must be in [0,1]")
		return 2
	}
	signals := negframe.MeasureNegationOperator(*benchmarkDelta, *unknownRate)
	payload := negframe.BuildNegationOperatorScore(signals)
	if *asJSON {
		b, _ := json.MarshalIndent(payload, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else {
		fmt.Fprintf(stdout, "negation_operator benchmark_delta=%.3f enumerable_domain_coverage=%.3f unknown_fallback_rate=%.3f verdict=%s\n", signals.BenchmarkDelta, signals.DomainCoverage, signals.UnknownFallbackRate, payload.Verdict)
	}
	if payload.OK {
		return 0
	}
	return 1
}
