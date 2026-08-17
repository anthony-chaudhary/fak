package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/toolcallcontrol"
)

func runToolprocReplay(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("toolproc replay", flag.ContinueOnError)
	fs.SetOutput(stderr)
	trace := fs.String("trace", "", "JSONL trace with independent needed labels")
	asJSON := fs.Bool("json", false, "emit full machine-readable report")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *trace == "" {
		fmt.Fprintln(stderr, "fak toolproc replay: --trace is required")
		return 2
	}
	f, err := os.Open(*trace)
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc replay: %v\n", err)
		return 1
	}
	defer f.Close()
	rows, err := toolcallcontrol.DecodeReplay(f)
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc replay: %v\n", err)
		return 1
	}
	report := toolcallcontrol.Replay(rows)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak toolproc replay: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "TOOL-CALL CONTROL REPLAY — %d labeled proposals, identical trace in every arm\n", report.TraceRows)
	fmt.Fprintln(stdout, "ARM                 EXECUTED  UNNEEDED AVOIDED  NEEDED SUPPRESSED  REPLAY UNITS SAVED  REPLAY² PROXY")
	for _, arm := range report.Arms {
		m := arm.Metrics
		fmt.Fprintf(stdout, "%-20s %8d %18d %18d %19d %14s\n", arm.Name, m.CallsExecuted, m.UnneededAvoided, m.NeededSuppressed, m.ReplayUnitsSaved, m.ReplaySquareProxy)
	}
	fmt.Fprintln(stdout, "Guardrail: NEEDED SUPPRESSED must remain 0; replay units and replay² are exposure proxies, not measured dollars, latency, FLOPs, or provider-billed tokens.")
	return 0
}
