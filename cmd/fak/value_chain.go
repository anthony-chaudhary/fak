package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/valuechain"
	"io"
	"os"
)

func cmdValueChain(args []string) { os.Exit(runValueChain(os.Stdout, os.Stderr, args)) }
func runValueChain(out, errOut io.Writer, args []string) int {
	if len(args) == 0 || args[0] != "audit" {
		fmt.Fprintln(errOut, "usage: fak value-chain audit --manifest M --observations O [--json]")
		return 2
	}
	fs := flag.NewFlagSet("value-chain audit", flag.ContinueOnError)
	fs.SetOutput(errOut)
	manifest := fs.String("manifest", "", "manifest JSON")
	observations := fs.String("observations", "", "observations JSON")
	asJSON := fs.Bool("json", false, "emit JSON")
	agenticPacket := fs.String("agentic-packet", "", "graduated AgenticBench result packet")
	agenticStage := fs.String("agentic-stage", "benchmark", "manifest stage for AgenticBench observations")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	if *manifest == "" || *observations == "" {
		fmt.Fprintln(errOut, "--manifest and --observations are required")
		return 2
	}
	m, in, err := valuechain.Read(*manifest, *observations)
	if err == nil && *agenticPacket != "" {
		var packet valuechain.Input
		packet, err = valuechain.ReadAgenticPacket(*agenticPacket, *agenticStage)
		if err == nil {
			in.Observations = append(in.Observations, packet.Observations...)
		}
	}
	if err != nil {
		fmt.Fprintln(errOut, err)
		return 1
	}
	rep, err := valuechain.Audit(m, in)
	if err != nil {
		fmt.Fprintln(errOut, err)
		return 1
	}
	if *asJSON {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(errOut, err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(out, "VALUE CHAIN %s\n", rep.Name)
	for _, a := range rep.Arms {
		fmt.Fprintf(out, "arm=%s traces=%d sessions=%d turns=%d billing_evidence=%d/%d", a.Arm, a.Traces, a.Sessions, a.Turns, a.BillingEvidence.Covered, a.BillingEvidence.Total)
		if a.CostPerTurn != nil {
			fmt.Fprintf(out, " $/turn=%.6f", *a.CostPerTurn)
		} else {
			fmt.Fprint(out, " $/turn=UNKNOWN")
		}
		for id, v := range a.CostPerOutcome {
			fmt.Fprintf(out, " $/%s=%.6f", id, v)
		}
		fmt.Fprintln(out)
	}
	for _, s := range rep.Inventory {
		fmt.Fprintf(out, "stage=%s kind=%s status=%s observations=%d\n", s.Stage, s.Kind, s.Status, s.Observations)
	}
	if rep.Comparison != nil {
		fmt.Fprintf(out, "comparison=%s->%s design=%s paired=%d\n", rep.Comparison.Baseline, rep.Comparison.Candidate, rep.Comparison.Design, rep.Comparison.PairedTraces)
	}
	return 0
}
