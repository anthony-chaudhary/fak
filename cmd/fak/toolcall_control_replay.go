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
	compactJSON := fs.Bool("compact-json", false, "emit aggregates with links to full records")
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
	if *compactJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report.Compact()); err != nil {
			fmt.Fprintf(stderr, "fak toolproc replay: %v\n", err)
			return 1
		}
		return 0
	}
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
	fmt.Fprintln(stdout, "ARM                 EXECUTED  UNNEEDED AVOIDED  NEEDED SUPPRESSED  GROSS SAVED  CONTROL COST  RECOVERY COST  NET VALUE  BREAK-EVEN  REPLAY2 PROXY")
	for _, arm := range report.Arms {
		m := arm.Metrics
		fmt.Fprintf(stdout, "%-20s %8d %18d %18d %11d %13d %14d %10d %11t %14s\n", arm.Name, m.CallsExecuted, m.UnneededAvoided, m.NeededSuppressed, m.ReplayUnitsSaved, m.ControllerUnits, m.FalseSuppressionRecoveryUnits, m.NetReplayValue, m.BreakEven, m.ReplaySquareProxy)
	}
	fmt.Fprintln(stdout, "\nLONG-CONTEXT GUARDRAIL BY BUCKET")
	for _, arm := range report.Arms {
		for _, bucket := range arm.Buckets {
			if bucket.NeededSuppressed > 0 || bucket.UnneededAvoided > 0 {
				fmt.Fprintf(stdout, "%-20s %-10s avoided=%d NEEDED_SUPPRESSED=%d records=full-report.json#/arms/%s/decisions\n", arm.Name, bucket.Name, bucket.UnneededAvoided, bucket.NeededSuppressed, arm.Name)
			}
		}
	}
	fmt.Fprintln(stdout, "Guardrail: NEEDED SUPPRESSED must remain 0. Net replay value = gross saved - controller units - false-suppression recovery. cost_basis says observed or scenario; replay units and replay2 are not measured dollars, latency, FLOPs, or provider billing.")
	return 0
}
