package main

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/sessionledger"
)

func runSessionLog(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("session log", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit JSON entries")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: fak session log <trace> [--json]")
		return 2
	}
	l, err := sessionledger.OpenDefault()
	if err != nil {
		fmt.Fprintf(stderr, "fak session log: %v\n", err)
		return 1
	}
	entries, err := l.Chain(fs.Arg(0))
	if err != nil {
		fmt.Fprintf(stderr, "fak session log: %v\n", err)
		return 1
	}
	renderSessionLog(stdout, entries, *jsonOut)
	return 0
}

func renderSessionLog(stdout io.Writer, entries []sessionledger.Entry, jsonOut bool) {
	for _, e := range entries {
		if jsonOut {
			fmt.Fprintf(stdout, "{\"hash\":%q,\"parent\":%q,\"kind\":%q,\"content\":%s}\n", e.Hash, e.Parent, e.Kind, e.Content)
		} else {
			fmt.Fprintf(stdout, "%s %s\n", e.Hash, e.Kind)
		}
	}
}
