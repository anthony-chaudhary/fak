package main

import (
	"fmt"
	"io"
	"os"
)

// subcommand pairs an argv[0] verb with the handler that runs it. Order matters only
// for the usage hint rendered when a verb is missing or unknown.
type subcommand struct {
	name string
	run  func(stdout, stderr io.Writer, argv []string) int
}

// dispatchSubcommands is the shared `fak <group> <verb>` router: with no verb it prints
// "missing subcommand (<hint>)" and exits 2; with an unknown verb it prints
// "unknown subcommand %q (want: <hint>)" and exits 2; otherwise it runs the matching
// handler over os.Stdout/os.Stderr and exits with its return code. group is the command
// name ("scoreboard"), hint is the human verb list ("post | feed").
func dispatchSubcommands(group, hint string, argv []string, subs ...subcommand) {
	if len(argv) == 0 {
		fmt.Fprintf(os.Stderr, "fak %s: missing subcommand (%s)\n", group, hint)
		os.Exit(2)
	}
	for _, s := range subs {
		if s.name == argv[0] {
			os.Exit(s.run(os.Stdout, os.Stderr, argv[1:]))
		}
	}
	fmt.Fprintf(os.Stderr, "fak %s: unknown subcommand %q (want: %s)\n", group, argv[0], hint)
	os.Exit(2)
}
