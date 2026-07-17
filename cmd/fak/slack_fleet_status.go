package main

import (
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func runSlackFleetStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("slack fleet-status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", defaultLoopLedger(), "loop manager ledger path")
	jsonOut := fs.Bool("json", false, "emit machine-readable status")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak slack fleet-status: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	now := time.Now()
	events, integrity, err := loopmgr.LoadPrefix(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak slack fleet-status: %v\n", err)
		return 1
	}
	processes, processErr := collectBackgroundProcesses()
	report := loopfleet.BuildBackgroundStatus(loopmgr.Summarize(events, now), processes, now)
	if *jsonOut {
		if processErr != nil {
			fmt.Fprintf(stderr, "fak slack fleet-status: warning: %v\n", processErr)
		}
		if integrity.Broken {
			fmt.Fprintf(stderr, "fak slack fleet-status: warning: ledger integrity break at line %d\n", integrity.AtLine)
		}
		return encodeJSONOrFail(stdout, stderr, report, "fak slack fleet-status")
	}
	fmt.Fprintf(stdout, "background fleet: %d total, %d live, %d managed, %d process-only, %d stale\n", report.Total, report.Live, report.Managed, report.ProcessOnly, report.Stale)
	for _, row := range report.Loops {
		fmt.Fprintf(stdout, "%-14s %-12s %-7s %s", row.Kind, row.State, row.Source, row.ID)
		if row.PID > 0 {
			fmt.Fprintf(stdout, " pid=%d", row.PID)
		}
		fmt.Fprintln(stdout)
	}
	if processErr != nil {
		fmt.Fprintf(stderr, "fak slack fleet-status: warning: %v\n", processErr)
	}
	return 0
}
