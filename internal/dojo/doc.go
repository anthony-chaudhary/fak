// Package dojo is fak's prediction-vs-reality gym: the closed loop that turns a
// token-saving THEORY ("this lever saves X") into a scored, trended verdict
// against billed reality.
//
// The loop has four rungs, mirroring how `fak resume validate` back-tests one
// lever today, generalized so any optimization can register:
//
//	declare a prediction  ->  run the scenario  ->  measure ground truth  ->  score the gap
//	   (the theory)            (a workload)          (the provider's bill)     (calibration)
//
// A Lever declares a Prediction (the Claimed number) before reality is
// consulted; a Scenario is the workload it runs against (an offline corpus of
// real transcripts today, a live session feed later); the Outcome is the
// measured Realized number lifted from billed reality; and Score folds the two
// into an Episode whose Verdict says whether reality met the claim (CALIBRATED),
// fell short of it (OVER_CLAIM), or exceeded it (UNDER_CLAIM).
//
// Fold rolls a run's episodes into one control-pane envelope (the same
// schema/ok/verdict/finding/reason/next_action shape the cadence report uses),
// and the durable JSONL ledger (TrendVsLast) trends the mean calibration error
// across runs -- so the gym answers not just "what did this lever save" but "are
// our predictors getting better calibrated over time". Provenance keeps every
// number honest about whether fak WITNESSED it or merely OBSERVED it from the
// provider.
//
// This package is pure (stdlib only, no clock, no I/O): the corpus scan and the
// concrete levers live in the cmd/fak/dojo.go shell, exactly as resume's
// back-test keeps its transcript scan in the cmd shell.
package dojo
