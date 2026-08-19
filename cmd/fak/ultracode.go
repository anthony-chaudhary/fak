package main

import (
	"fmt"
	"io"
	"os"
	"strings"
)

const ultracodeUsage = `usage: fak ultracode [status] [orchestration plan flags]

Run a bounded concurrent coding-agent fleet through fak's canonical orchestration path.

  fak ultracode --task-text "split independent checks and reconcile them"
  fak ultracode --task task.json --launch --json
  fak ultracode status [--json]
  fak ultracode --selfcheck

The ultracode profile requires leases, independent effect readback, and reconciliation.
Planning and selfcheck are offline. --launch starts the resolved harness workers.`

func cmdUltracode(args []string) {
	os.Exit(runUltracode(os.Stdout, os.Stderr, args))
}

func runUltracode(stdout, stderr io.Writer, args []string) int {
	if stdout == nil || stderr == nil {
		panic("runUltracode requires writers")
	}
	if len(args) > 0 && (args[0] == "help" || args[0] == "-h" || args[0] == "--help") {
		fmt.Fprintln(stdout, ultracodeUsage)
		return 0
	}
	if len(args) > 0 && args[0] == "status" {
		return runOrchestration(stdout, stderr, args)
	}
	for i, arg := range args {
		if arg == "--profile" || strings.HasPrefix(arg, "--profile=") {
			fmt.Fprintln(stderr, "fak ultracode: --profile is fixed to ultracode; use fak orchestration plan to select another profile")
			return 2
		}
		if arg == "plan" && i == 0 {
			fmt.Fprintln(stderr, "fak ultracode: plan is implicit; pass plan flags directly")
			return 2
		}
	}
	delegated := []string{"plan", "--profile", "ultracode"}
	delegated = append(delegated, args...)
	if hasArg(args, "--selfcheck") && !hasArg(args, "--task") && !hasArg(args, "--task-text") {
		delegated = append(delegated, "--task-text", "fan out independent implementation and review work, then reconcile witnessed effects")
	}
	return runOrchestration(stdout, stderr, delegated)
}

func hasArg(args []string, name string) bool {
	for _, arg := range args {
		if arg == name || strings.HasPrefix(arg, name+"=") {
			return true
		}
	}
	return false
}
