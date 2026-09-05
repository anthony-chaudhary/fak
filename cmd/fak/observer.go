package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/observer"
)

func cmdObserver(argv []string) {
	os.Exit(runObserver(os.Stdout, os.Stderr, argv))
}

func runObserver(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak observer", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit observer status as JSON")

	if !parseFlags(fs, argv) {
		return 2
	}

	screen := observer.NewObserverSemanticScreen(nil)
	screen.Register()

	report := map[string]any{
		"registered": true,
		"component":  "observer.ObserverSemanticScreen",
		"pool":       "default",
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak observer: json encode: %v\n", err)
			return 2
		}
		return 0
	}

	fmt.Fprintln(stdout, "observer: semantic screen registered successfully")
	return 0
}
