package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"
)

func main() {
	fs := flag.NewFlagSet("loophealth", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit the fold as JSON")
	journal := fs.String("journal", "", "path to the verdict journal (default: .dos/verdict-journal.jsonl under the workspace)")
	window := fs.Int("window", 0, "consider only the most-recent N journal rows (0 = all)")
	gate := fs.Bool("gate", false, "exit 3 if the current window is worse than the recorded baseline (else exit 0)")
	fleet := fs.Bool("fleet", false, "fold EVERY loop's last-tick/keep-rate across the fragmented loop ledgers (loopmgr/nightrun/dojo/cadence/rsiloop/guardrsi/dispatch); a dark loop exits 3")
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if *window < 0 {
		fmt.Fprintln(os.Stderr, "loophealth: --window must be >= 0")
		os.Exit(2)
	}

	root, err := os.Getwd()
	if err != nil {
		fmt.Fprintln(os.Stderr, "loophealth:", err)
		os.Exit(2)
	}

	// --fleet is the cross-loop fold (#1196): every loop's health from one read,
	// dark loops surfaced and exit-gated. It is its own read-only view, distinct
	// from the self-improve loop-health fold below.
	if *fleet {
		os.Exit(fleetMain(os.Stdout, os.Stderr, root, *asJSON, time.Now()))
	}

	path := resolveJournal(*journal, root)

	h, err := computeHealth(path, *window)
	if err != nil {
		fmt.Fprintln(os.Stderr, "loophealth:", err)
		os.Exit(2)
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(h); err != nil {
			fmt.Fprintln(os.Stderr, "loophealth:", err)
			os.Exit(2)
		}
	} else {
		fmt.Println(render(h))
	}

	if *gate && !h.Healthy {
		os.Exit(3)
	}
}
