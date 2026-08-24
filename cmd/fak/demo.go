package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/demo"
)

// `fak demo` — the zero-flag 60-second proof. It runs fak's canonical offline
// scenario end-to-end through the REAL kernel and narrates one verdict per call
// class: a safe read is ALLOWED, an irreversible/destructive call is DENIED, and a
// poisoned tool RESULT is QUARANTINED (held out of context). Every verdict is a live
// kernel decision — the same adjudicator + result-admitter chain a guarded session
// arms — not a scripted string. It launches no agent, needs no key or network, and
// writes nothing; run it to SEE the floor work in seconds.
//
//   fak demo           the plain-words walkthrough (default)
//   fak demo --json    the machine-readable result (the three verdicts)

func cmdDemo(argv []string) {
	os.Exit(runDemo(os.Stdout, os.Stderr, argv))
}

func emitResultOrError[T any](stdout, stderr io.Writer, label string, asJSON bool, result T, err error) (int, bool) {
	if err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", label, err)
		return 1, true
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, result, label), true
	}
	return 0, false
}

func runDemo(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("demo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the three real kernel verdicts as JSON and exit")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	res, err := demo.Run(context.Background())
	if code, done := emitResultOrError(stdout, stderr, "fak demo", *asJSON, res, err); done {
		return code
	}
	res.RenderText(stdout)
	return 0
}
