package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopindex"
)

// TestLoopIndexSpineGreen is the spine regression sentinel for #1152 (the dev-ex
// epic #1148 keystone): the committed tree must keep every loop stage WIRED at its
// floor, so the loop-index headline holds at loopindex_debt 0 over 6-of-6 wired
// stages. A stage UN-wiring — a regressed default such as recall re-verify (#1158),
// collision-priced fan-out (#1154), the false-done STOP refusal (#1157), the
// green-gate latency budget (#1155), or the consuming RSI loop (#1161) — reds this
// test, the same regression the portfolio ratchet catches fleet-wide
// (tools/scorecard_control_pane.py, baseline loopindex=0). Deterministic and
// re-runnable from a clean clone: it scores the tracked tree, never a live metric,
// so a peer reproduces the verdict.
func TestLoopIndexSpineGreen(t *testing.T) {
	rep := loopindex.Score(collectLoopIndex(repoRoot()))
	if rep.Corpus.LoopIndexDebt != 0 {
		t.Fatalf("loopindex_debt = %d, want 0 — a loop stage regressed below its floor\n  finding: %s\n  next: %s",
			rep.Corpus.LoopIndexDebt, rep.Finding, rep.NextAction)
	}
	if rep.Corpus.WiredStages != 6 {
		t.Fatalf("wired stages = %d/6, want 6 — a loop stage lost its load-bearing witness", rep.Corpus.WiredStages)
	}
}
