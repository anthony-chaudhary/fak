package main

// Test fixture ONLY — never compiled (Go ignores testdata/). It is scanned as
// TEXT by internal/ctxknobs to exercise the flag/env walker: two context knobs
// (ctx-view-budget, FAK_CONTEXT_TOKENS) and two non-context registrations that
// must be ignored (verbose, HOME).

import "os"

func sample() {
	_ = os.Getenv("FAK_CONTEXT_TOKENS") // context knob -> operator-debug
	_ = os.Getenv("HOME")               // not a context knob -> ignored

	var budget int
	fs.IntVar(&budget, "ctx-view-budget", 8000, "planned context view budget")
	fs.Bool("verbose", false, "loud output") // not a context knob -> ignored
}
