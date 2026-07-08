package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/conformance"
)

// cmdConformance runs the standalone fak safety-conformance suite (#453): it recomputes the
// ABI wire contract and re-adjudicates the shipped dogfood verdict matrix against the
// COMPILED kernel, then reports CONFORMANT / NON-CONFORMANT. Exit code is 1 on any
// divergence, so `fak conformance` is a CI-gateable, third-party-runnable attestation —
// any fork self-tests and any auditor verifies a "certified" claim independently.
func cmdConformance(argv []string) {
	fs := flag.NewFlagSet("conformance", flag.ExitOnError)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON instead of a human report")
	_ = fs.Parse(argv)

	rep := conformance.Run()

	if *asJSON {
		b, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak conformance: %v\n", err)
			os.Exit(2)
		}
		os.Stdout.Write(b)
		os.Stdout.Write([]byte("\n"))
	} else {
		fmt.Print(conformance.Render(rep))
	}

	if !rep.Pass {
		os.Exit(1)
	}
}
