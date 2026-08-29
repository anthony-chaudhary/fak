package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agentic"
)

func cmdAgentic(args []string) {
	os.Exit(runAgentic(os.Stdout, os.Stderr, args))
}

func runAgentic(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("agentic", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "emit the deterministic fak-agentic-work/1 plan as JSON")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak agentic [--json] OBJECTIVE...")
		fmt.Fprintln(stderr, "  compile an objective into a bounded read-only/offline expand, experiment, and contract plan")
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak agentic: objective is required")
		fs.Usage()
		return 2
	}

	plan, err := agentic.Compile(strings.Join(fs.Args(), " "))
	if err != nil {
		fmt.Fprintf(stderr, "fak agentic: %v\n", err)
		return 2
	}
	if *jsonOut {
		data, err := agentic.Marshal(plan)
		if err != nil {
			fmt.Fprintf(stderr, "fak agentic: encode plan: %v\n", err)
			return 1
		}
		if _, err := stdout.Write(data); err != nil {
			fmt.Fprintf(stderr, "fak agentic: write plan: %v\n", err)
			return 1
		}
		return 0
	}
	if _, err := io.WriteString(stdout, agentic.Render(plan)); err != nil {
		fmt.Fprintf(stderr, "fak agentic: write plan: %v\n", err)
		return 1
	}
	return 0
}
