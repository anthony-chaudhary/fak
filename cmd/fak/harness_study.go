package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/harnesscreationreceipt"
	"github.com/anthony-chaudhary/fak/internal/harnesscreationstudy"
	"github.com/anthony-chaudhary/fak/internal/harnesscrossover"
)

func runHarnessStudy(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "receipt" {
		return runHarnessCreationReceipt(stdout, stderr, argv[1:])
	}
	if len(argv) > 0 && argv[0] == "creation" {
		return runHarnessCreationStudy(stdout, stderr, argv[1:])
	}
	if len(argv) == 0 || argv[0] != "crossover" {
		fmt.Fprintln(stderr, "usage: fak harness study <creation|crossover> --input STUDY.json")
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

func runHarnessCreationStudy(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness study creation", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak.harness-creation-study/v1alpha1 JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness study creation --input STUDY.json")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	study, err := harnesscreationstudy.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(harnesscreationstudy.Evaluate(study)); err != nil {
		fmt.Fprintf(stderr, "fak harness study creation: %v\n", err)
		return 1
	}
	return 0
}

func runHarnessCreationReceipt(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("harness study receipt", flag.ContinueOnError)
	fs.SetOutput(stderr)
	input := fs.String("input", "", "fak.harness-creation-receipt/v1alpha1 JSON")
	studyPath := fs.String("study", "", "optional study JSON for duplicate run/participant refusal")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *input == "" || fs.NArg() != 0 {
		fmt.Fprintln(stderr, "usage: fak harness study receipt --input RECEIPT.json")
		return 2
	}
	raw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study receipt: %v\n", err)
		return 1
	}
	receipt, err := harnesscreationreceipt.Parse(raw)
	if err != nil {
		fmt.Fprintf(stderr, "fak harness study receipt: %v\n", err)
		return 1
	}
	result := harnesscreationreceipt.Evaluate(receipt)
	if *studyPath != "" {
		studyRaw, readErr := os.ReadFile(*studyPath)
		if readErr != nil {
			fmt.Fprintf(stderr, "fak harness study receipt: %v\n", readErr)
			return 1
		}
		if uniqueErr := harnesscreationreceipt.CheckUnique(studyRaw, result.Row); uniqueErr != nil {
			fmt.Fprintf(stderr, "fak harness study receipt: %v\n", uniqueErr)
			return 1
		}
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(result); err != nil {
		fmt.Fprintf(stderr, "fak harness study receipt: %v\n", err)
		return 1
	}
	return 0
}
