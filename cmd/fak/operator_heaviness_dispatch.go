package main

import (
	"fmt"
	"io"
	"os"
)

func cmdOperatorHeaviness(argv []string) {
	os.Exit(runOperatorHeavinessGroup(os.Stdout, os.Stderr, argv))
}

func runOperatorHeavinessGroup(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak operator: missing subcommand (brief, triage, heaviness)")
		return 2
	}
	switch argv[0] {
	case "brief":
		return runOperatorBrief(stdout, stderr, argv[1:])
	case "triage":
		return runOperatorTriage(stdout, stderr, argv[1:])
	case "heaviness":
		return runOperatorHeaviness(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, "usage: fak operator <brief|triage|heaviness>")
		fmt.Fprintln(stdout, "  brief      compact operator pacing snapshot (--full to expand, --json for agents, --check to gate)")
		fmt.Fprintln(stdout, "  triage     decenter the human: page only on genuine authority decisions [--brief FILE] [--json] [--check] [selfcheck]")
		fmt.Fprintln(stdout, "  heaviness  operator-heaviness / steering-effort scorecard [--json] [--markdown] [--compare FILE]")
		return 0
	default:
		fmt.Fprintf(stderr, "fak operator: unknown subcommand %q (want: brief, triage, heaviness)\n", argv[0])
		return 2
	}
}
