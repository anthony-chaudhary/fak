package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
)

func runDormancy(out, errOut io.Writer, args []string) int {
	fs := flag.NewFlagSet("dormancy", flag.ContinueOnError)
	fs.SetOutput(errOut)
	ledger := fs.String("ledger", "", "loop-manager JSONL ledger")
	asJSON := fs.Bool("json", false, "emit JSON view")
	metrics := fs.Bool("prometheus", false, "emit Prometheus exposition")
	nowText := fs.String("now", "", "RFC3339 observation time (test/replay)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *ledger == "" {
		fmt.Fprintln(errOut, "usage: fak dormancy --ledger LEDGER.jsonl [--json|--prometheus]")
		return 2
	}
	now := time.Now().UTC()
	if *nowText != "" {
		var err error
		now, err = time.Parse(time.RFC3339, *nowText)
		if err != nil {
			fmt.Fprintf(errOut, "dormancy: --now: %v\n", err)
			return 2
		}
	}
	f, err := os.Open(*ledger)
	if err != nil {
		fmt.Fprintf(errOut, "dormancy: %v\n", err)
		return 1
	}
	defer f.Close()
	records, err := dormancy.ReadLedger(f)
	if err != nil {
		fmt.Fprintf(errOut, "dormancy: %v\n", err)
		return 1
	}
	view := dormancy.Fold(records, now)
	if *metrics {
		fmt.Fprint(out, view.Prometheus())
		return 0
	}
	if *asJSON {
		enc := json.NewEncoder(out)
		enc.SetIndent("", "  ")
		if err := enc.Encode(view); err != nil {
			return 1
		}
		return 0
	}
	fmt.Fprintf(out, "loops=%d intentionally-dormant=%d stuck=%d oldest=%s gap=%.0fs\n", len(view.Loops), view.Dormant, view.Stuck, view.OldestLoopID, view.OldestGapSecs)
	for _, l := range view.Loops {
		fmt.Fprintf(out, "%s\t%s\t%s\t%.0fs\trestores=%d\n", l.LoopID, l.Status, l.Horizon, l.GapSeconds, l.RestoreCount)
	}
	return 0
}

func cmdDormancy(args []string) { os.Exit(runDormancy(os.Stdout, os.Stderr, args)) }
