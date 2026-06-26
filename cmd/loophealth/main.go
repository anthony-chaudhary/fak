package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

func main() {
	fs := flag.NewFlagSet("loophealth", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit the {closure_rate, regression_rate, window, baseline} fold as JSON")
	journal := fs.String("journal", "", "path to the verdict journal (default: .dos/verdict-journal.jsonl under the workspace)")
	window := fs.Int("window", 0, "consider only the most-recent N journal rows (0 = all)")
	gate := fs.Bool("gate", false, "exit 3 if the current window is worse than the recorded baseline (else exit 0)")
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
