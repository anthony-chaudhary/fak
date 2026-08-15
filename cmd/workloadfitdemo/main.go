package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/workloadfit"
)

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	fs := flag.NewFlagSet("workloadfitdemo", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	selfcheck := fs.Bool("selfcheck", false, "run coding and legal-review fixture")
	jsonOut := fs.Bool("json", false, "emit machine output")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if !*selfcheck {
		fmt.Fprintln(os.Stderr, "usage: workloadfitdemo -selfcheck [-json]")
		return 2
	}
	coding, legal, err := workloadfit.Selfcheck()
	if err != nil {
		fmt.Fprintf(os.Stderr, "SELFCHECK FAIL: %v\n", err)
		return 1
	}
	if *jsonOut {
		_ = json.NewEncoder(os.Stdout).Encode(struct {
			Schema string                `json:"schema"`
			Coding workloadfit.Selection `json:"coding"`
			Legal  workloadfit.Selection `json:"legal"`
		}{workloadfit.Schema, coding, legal})
		return 0
	}
	fmt.Printf("CODING fit: choose %s\n", coding.Chosen)
	fmt.Printf("LEGAL REVIEW fit: choose %s\n", legal.Chosen)
	for _, a := range legal.Assessments {
		if a.CandidateID != "ponytail@r8" {
			continue
		}
		fmt.Printf("PONYTAIL for legal: %s\n", a.Status)
		for _, f := range a.Findings {
			if f.State != "met" {
				fmt.Printf("  %s: %s — %s [%s]\n", f.Requirement, f.State, f.Reason, f.Source.Reference)
			}
		}
	}
	fmt.Println("BOUNDARY: fitness fixture is not legal certification; domain review remains required")
	fmt.Println("SELFCHECK PASS: compatibility and purpose-specific fitness remain separate")
	return 0
}
