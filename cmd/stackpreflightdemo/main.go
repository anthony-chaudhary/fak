package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stackpreflight"
	"github.com/anthony-chaudhary/fak/internal/stackresolve"
	"github.com/anthony-chaudhary/fak/internal/supportgraph"
	"github.com/anthony-chaudhary/fak/internal/workloadfit"
)

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	fs := flag.NewFlagSet("stackpreflightdemo", flag.ContinueOnError)
	selfcheck := fs.Bool("selfcheck", false, "run accepted and refused preflight fixtures")
	jsonOutput := fs.Bool("json", false, "emit machine-readable receipts")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if !*selfcheck {
		fmt.Fprintln(os.Stderr, "usage: stackpreflightdemo -selfcheck [-json]")
		return 2
	}
	accepted, refused, err := selfcheckReceipts()
	if err != nil {
		fmt.Fprintf(os.Stderr, "SELFCHECK FAIL: %v\n", err)
		return 1
	}
	if *jsonOutput {
		_ = json.NewEncoder(os.Stdout).Encode(struct {
			Schema            string `json:"schema"`
			Accepted, Refused stackpreflight.Result
		}{stackpreflight.Schema, accepted, refused})
		return 0
	}
	fmt.Printf("MINIMUM PATH: %s required=%v\n", accepted.Status, accepted.Required)
	for _, warning := range accepted.Warnings {
		fmt.Printf("  WARN %s\n", warning)
	}
	fmt.Printf("T4 PATH: %s blockers=%v\n", refused.Status, refused.Blockers)
	for _, alternative := range refused.Alternatives {
		fmt.Printf("  ALTERNATIVE %d: %s; impact=%s\n", alternative.Rank, alternative.Action, alternative.Impact)
	}
	fmt.Println("SELFCHECK PASS: mandatory support blocks launch; recommendations remain warnings")
	return 0
}

func selfcheckReceipts() (stackpreflight.Result, stackpreflight.Result, error) {
	stack, _, err := stackresolve.Selfcheck(context.Background())
	if err != nil {
		return stackpreflight.Result{}, stackpreflight.Result{}, err
	}
	_, legal, err := workloadfit.Selfcheck()
	if err != nil {
		return stackpreflight.Result{}, stackpreflight.Result{}, err
	}
	fitness := legal.Assessments[0]
	var graph supportgraph.Graph
	raw, err := os.ReadFile("internal/supportgraph/testdata/awq.json")
	if err != nil {
		return stackpreflight.Result{}, stackpreflight.Result{}, err
	}
	graph, err = supportgraph.Parse(raw)
	if err != nil {
		return stackpreflight.Result{}, stackpreflight.Result{}, err
	}
	asOf := time.Date(2026, 8, 15, 0, 0, 0, 0, time.UTC)
	accepted := stackpreflight.Run(stackpreflight.Input{Stack: stack, Fitness: fitness, Graph: graph, Tuple: graph.Edges[0].Tuple, AsOf: asOf, CapacityTarget: "24GiB model residency"})
	refused := stackpreflight.Run(stackpreflight.Input{Stack: stack, Fitness: fitness, Graph: graph, Tuple: graph.Edges[1].Tuple, AsOf: asOf, CapacityTarget: "24GiB model residency"})
	if accepted.Status != "allow" || refused.Status != "refuse" {
		return accepted, refused, fmt.Errorf("statuses allow=%s refuse=%s", accepted.Status, refused.Status)
	}
	return accepted, refused, nil
}
