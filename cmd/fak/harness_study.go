package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/harnesscrossover"
)

func runHarnessStudy(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "crossover" {
		fmt.Fprintln(stderr, "usage: fak harness study crossover --input STUDY.json")
		return 2
	}
	fs := flag.NewFlagSet("harness study crossover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak.harness-crossover-study/v1alpha1 JSON")
	if err := fs.Parse(argv[1:]); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness study crossover --input STUDY.json")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study crossover: %v\n", err)
		return 1
	}
	study, err := harnesscrossover.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study crossover: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(harnesscrossover.Evaluate(study)); err != nil {
		fmt.Fprintf(stderr, "fak harness study crossover: %v\n", err)
		return 1
	}
	return 0
}
