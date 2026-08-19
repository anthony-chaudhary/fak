package main

import (
	"context"
	"os"

	"github.com/anthony-chaudhary/fak/internal/flowmetrics"
)

// `fak score flow` is the READOUT hop for the eight flow KPIs (#6198, epic #6194).
// Its local-WIP section also names recently written paths and duplicate symbols;
// repeat --touch for planned edits to surface exact path overlap before authoring.
//
// internal/flowmetrics already folds issue rows against commit rows into started/closed
// spans and grades eight Little's-Law axes, but nothing called it: the numbers were
// reachable only from a Go test, so they could not change a decision. This route is the
// caller and nothing else.
//
// The whole hop — flags, gather, fold, render — lives in flowmetrics.RunScore rather
// than here, so the argv path is exercised by the package's own tests instead of only
// by a cmd/fak test binary. Per the #2247 convention the verb lands under
// `fak score <name>` and never as another top-level *-scorecard verb; score_test.go
// pins that route table.
//
// It is deliberately NOT a gate: flow debt is a measurement, and the verb exits 0 even
// on a DEFECT verdict so a control-pane collection reads the payload rather than a
// failure. Refusal on a threshold is a separate child of #6194.
func cmdFlowScore(argv []string) {
	os.Exit(flowmetrics.RunScore(context.Background(), os.Stdout, os.Stderr, argv, repoRoot()))
}
