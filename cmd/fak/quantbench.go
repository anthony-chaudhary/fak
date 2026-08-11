package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/quantbench"
)

func cmdQuantbench(args []string) {
	os.Exit(runQuantbench(os.Stdout, os.Stderr, args))
}

func runQuantbench(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("quantbench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	selfTest := fs.Bool("self-test", false, "emit the machine-readable contract matrix")
	input := fs.String("input", "", "JSON benchmark input file ('-' for stdin)")
	jsonOut := fs.Bool("json", false, "emit JSON (the default contract output)")
	_ = jsonOut
	if err := fs.Parse(args); err != nil {
		return 2
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if *selfTest {
		report := quantbench.SelfTest()
		if err := enc.Encode(report); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		if !report.Pass {
			return 1
		}
		return 0
	}
	if *input == "" {
		fmt.Fprintln(stderr, "usage: fak quantbench --self-test --json | --input FILE --json")
		return 2
	}
	var r io.Reader
	if *input == "-" {
		r = os.Stdin
	} else {
		f, err := os.Open(*input)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
		defer f.Close()
		r = f
	}
	var in quantbench.Input
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		fmt.Fprintf(stderr, "quantbench: decode input: %v\n", err)
		return 2
	}
	result := quantbench.Evaluate(in)
	if err := enc.Encode(result); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if result.Outcome != quantbench.OutcomeBenchmark {
		return 3
	}
	return 0
}
