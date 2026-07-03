package main

import (
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchconservation"
)

// cmdDispatchConservation is the worker-unit conservation ledger: over a window,
// units_spent = accounted + leaked, so a worker-unit that dies ungraded reads as
// a LEAK count, not as silence. Read-only; parses the .dispatch-runs artifacts
// directly. `--fail-on-leak N` turns the leak count into a CI exit code.
func cmdDispatchConservation(argv []string) {
	os.Exit(dispatchconservation.Run(argv, dispatchconservation.DefaultAliveProbe, time.Now(), os.Stdout, os.Stderr))
}
