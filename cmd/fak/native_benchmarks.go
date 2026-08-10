package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/nativebench"
)

func cmdNativeBenchmarks(args []string) { os.Exit(runNativeBenchmarks(os.Stdout, os.Stderr, args)) }

func runNativeBenchmarks(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("native-benchmarks", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit the benchmark-obligation report as JSON")
	check := fs.Bool("check", false, "exit non-zero while any required benchmark witness is missing")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "native-benchmarks accepts no positional arguments")
		return 2
	}
	report := nativebench.Audit()
	if *jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	} else {
		verdict := "INCOMPLETE"
		if report.Complete {
			verdict = "COMPLETE"
		}
		fmt.Fprintf(stdout, "%s: %d/%d native leaves covered by %d comparison contracts; %d classified, %d unclassified, %d findings\n", verdict, report.Coverage.CoveredLeaves, report.Coverage.NativeLeaves, len(report.Contracts), report.Coverage.ClassifiedLeaves, report.Coverage.UnclassifiedLeaves, len(report.Findings))
		for _, f := range report.Findings {
			fmt.Fprintf(stdout, "- %s: %s\n", f.Capability, f.Reason)
		}
	}
	if *check && !report.Complete {
		return 1
	}
	return 0
}
