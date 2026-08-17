package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/turnbench"
)

func cmdFanout(args []string) {
	dispatchSubcommands("fanout", "trend", args,
		subcommand{"trend", runFanoutTrend},
	)
}

func runFanoutTrend(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak fanout trend", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", "docs/nightrun/fanout-reuse.jsonl", "fan-out reuse JSONL ledger")
	if !parseFlags(fs, args) {
		return 2
	}

	f, err := os.Open(*ledger)
	if err != nil {
		fmt.Fprintf(stderr, "fak fanout trend: open ledger: %v\n", err)
		return 1
	}
	defer f.Close()

	trend, err := turnbench.FoldFanLedger(f)
	if err != nil {
		fmt.Fprintf(stderr, "fak fanout trend: fold ledger: %v\n", err)
		return 1
	}
	fmt.Fprint(stdout, trend.Render())
	return 0
}
