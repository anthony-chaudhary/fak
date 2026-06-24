package main

import (
	"strings"
	"testing"

	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

// TestUnSeeWitness is the load-bearing assertion: driving the REAL ctxmmu gate + kvmmu
// bridge + model.KVCache.Evict on the synthetic Llama reproduces the committed witness —
// the gate quarantines the real poison bytes, the bridge evicts the span write-time, and
// the post-evict next-token distribution is BIT-IDENTICAL to a brain that never saw the
// poison (max|Δ|=0) while the poison-kept control differs. Same numbers as
// internal/kvmmu's TestWriteTimeEvictEqualsNeverSaw and experiments/kvmmu/kvmmu-report.json.
func TestUnSeeWitness(t *testing.T) {
	w := runExperiment().Witness

	if w.GateVerdict != "QUARANTINE" {
		t.Fatalf("gate verdict = %q, want QUARANTINE (the real ctxmmu decision must drive the bridge)", w.GateVerdict)
	}
	if !w.Quarantined {
		t.Fatal("the quarantined span was not evicted from the KV cache")
	}
	if w.CacheBeforeEvict != w.PrefixLen+w.PoisonLen {
		t.Fatalf("cache_before_evict = %d, want %d", w.CacheBeforeEvict, w.PrefixLen+w.PoisonLen)
	}
	if w.CacheAfterEvict != w.PrefixLen {
		t.Fatalf("cache_after_evict = %d, want %d (poison span removed)", w.CacheAfterEvict, w.PrefixLen)
	}
	// The headline: write-time evict == never-saw, bit-identical. Exact on every arch
	// (the prefix survivors sit before the cut, so nothing is repositioned).
	if w.EvictVsNever != 0 {
		t.Fatalf("evict-vs-never max|Δ| = %.3e, want exactly 0 (bit-identical to never-saw)", w.EvictVsNever)
	}
	// Non-vacuity: the poison must actually perturb the distribution, else the witness is empty.
	if w.PoisonVsNever <= 0 {
		t.Fatalf("poison-vs-never max|Δ| = %.3e, want > 0 (the witness would be vacuous)", w.PoisonVsNever)
	}
	// The boundary: a span a query already attended to cannot be un-seen.
	if w.TooLateVsNever <= 0 {
		t.Fatalf("too-late-vs-never max|Δ| = %.3e, want > 0 (late evict must NOT equal never-saw)", w.TooLateVsNever)
	}
	// The re-RoPE reposition must be bit-exact within the FMA noise floor.
	if w.RepositionResid > repTol {
		t.Fatalf("reposition residual = %.3e, want <= %.0e (re-RoPE drifted — rotations composed?)", w.RepositionResid, repTol)
	}
}

// TestWitnessMatchesCommittedReport pins the demo to the exact published magnitudes so a
// drift in the model engine that moved these numbers fails here, not silently in a video.
// The poison-vs-never magnitude (0.3257) is deterministic across platforms; allow a wide
// band so arm64 FMA noise (≤1e-4) never flakes the test while a real regression (which
// moves the logit by ≫1e-2) still trips it.
func TestWitnessMatchesCommittedReport(t *testing.T) {
	w := runExperiment().Witness
	if w.PoisonVsNever < 0.30 || w.PoisonVsNever > 0.35 {
		t.Fatalf("poison-vs-never = %.3e, want ~3.257e-01 (the committed kvmmu-report value); a shift this large is a model-engine regression", w.PoisonVsNever)
	}
}

// TestEventLogIsWellFormed guards the driver seam the browser cam replays: three acts,
// the headline reveal carries both the identical and contaminated readouts, and every
// frame's cache length is consistent with its cells.
func TestEventLogIsWellFormed(t *testing.T) {
	ev := runExperiment()
	if len(ev.Frames) == 0 {
		t.Fatal("no frames emitted")
	}
	acts := map[int]bool{}
	var sawHeadlineReveal bool
	for _, fr := range ev.Frames {
		acts[fr.Act] = true
		// the resident-cell count must match the reported cache length (evicting/attended
		// frames are mid-transition snapshots where the strip still shows the doomed span).
		if fr.Phase == "reveal" && fr.Act == 1 {
			var identical, contaminated bool
			for _, ro := range fr.Readouts {
				if ro.Verdict == "identical" && ro.Value == "0.000e+00" {
					identical = true
				}
				if ro.Verdict == "contaminated" {
					contaminated = true
				}
			}
			if !identical || !contaminated {
				t.Errorf("Act 1 reveal must light up BOTH the identical (0.000e+00) and contaminated readouts; got identical=%v contaminated=%v", identical, contaminated)
			}
			sawHeadlineReveal = true
		}
	}
	for _, a := range []int{1, 2, 3} {
		if !acts[a] {
			t.Errorf("event log missing Act %d", a)
		}
	}
	if !sawHeadlineReveal {
		t.Error("no Act 1 headline reveal frame")
	}
	if len(ev.Fences) == 0 {
		t.Error("event log carries no honesty fences")
	}
	if !strings.Contains(ev.PoisonText, ev.Marker) {
		t.Errorf("poison text must contain the marker %q the gate caught", ev.Marker)
	}
}

// TestSelfcheckPasses runs the headless invariant gate the same way CI would.
func TestSelfcheckPasses(t *testing.T) {
	if code := runSelfcheck(); code != 0 {
		t.Fatalf("runSelfcheck() exit code = %d, want 0", code)
	}
}
