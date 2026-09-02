//go:build !darwin

package main

import (
	"fmt"
	"io"
)

// The incumbent ownership surface observes launchd and BSD process identity, so
// it exists on darwin only; elsewhere it refuses instead of guessing. render and
// the typed classifier stay cross-platform for tests and receipt validation.

func runModelIncumbentPreflight(_, stderr io.Writer, _ []string) int {
	fmt.Fprintln(stderr, "fak model incumbent preflight: requires darwin (launchd + BSD ps/lsof)")
	return 2
}

func runModelIncumbentInstall(_, stderr io.Writer, _ []string) int {
	fmt.Fprintln(stderr, "fak model incumbent install: requires darwin (launchd)")
	return 2
}
