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
		fmt.Fprintln(stderr, "fak operator: missing subcommand (heaviness)")
		return 2
	}
	switch argv[0] {
	case "heaviness":
		return runOperatorHeaviness(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, "usage: fak operator heaviness [--json] [--markdown] [--compare FILE]")
		return 0
	default:
		fmt.Fprintf(stderr, "fak operator: unknown subcommand %q (want: heaviness)\n", argv[0])
		return 2
	}
}
