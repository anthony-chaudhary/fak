package main

import (
	"os"

	"github.com/anthony-chaudhary/fak/internal/glm52prefillsweep"
)

// cmdGLM52PrefillSweep is the thin shell over internal/glm52prefillsweep — the GLM-5.2
// pure-fak prefill-latency sweep driver (Go port of the retired
// tools/glm52_prefill_sweep.py). --dry-run (or omitting --endpoint) prints the pure plan;
// --endpoint runs the live sweep and lands per-length benchmark-ledger artifacts.
func cmdGLM52PrefillSweep(argv []string) {
	os.Exit(glm52prefillsweep.Run(os.Stdout, os.Stderr, argv))
}
