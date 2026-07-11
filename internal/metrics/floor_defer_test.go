package metrics

import "testing"

// TestFloorDeferFragmentGoldenLine pins the exact bytes of the guard per-turn
// exit-summary fragment (#3531) for both deferral states, from a synthetic
// footprint + defer count — no network, no gateway. It is the acceptance test
// the issue names: "renders from a synthetic footprint + defer count".
func TestFloorDeferFragmentGoldenLine(t *testing.T) {
	// Deferral ON: a floor of 1200 system + 3400 tools tokens, of which 2100
	// tools tokens are the deferrable cold tail; 7 cold defs deferred this turn.
	// The fragment shows the floor split, the witnessed defer count, and the
	// provider-side Δtools reduction the deferral drove.
	on := FloorDeferFragment(FloorFootprint{SystemTokens: 1200, ToolsTokens: 3400, ColdToolTokens: 2100}, 7)
	if want := "floor=1200+3400 defer=7 Δtools=2100"; on != want {
		t.Fatalf("deferral-on fragment = %q, want %q", on, want)
	}

	// Deferral OFF: same floor, but 0 cold defs deferred. defer=0 and Δtools=0 —
	// the fragment never claims a saving the lever did not take, even though the
	// footprint still carries a 2100-token cold tail.
	off := FloorDeferFragment(FloorFootprint{SystemTokens: 1200, ToolsTokens: 3400, ColdToolTokens: 2100}, 0)
	if want := "floor=1200+3400 defer=0 Δtools=0"; off != want {
		t.Fatalf("deferral-off fragment = %q, want %q", off, want)
	}
}

// TestFloorDeferFragmentClampsNonsense confirms negative inputs (never expected
// from a real footprint) floor to 0 rather than rendering "defer=-1", so a
// malformed footprint degrades to an honest zero instead of a nonsense line.
func TestFloorDeferFragmentClampsNonsense(t *testing.T) {
	got := FloorDeferFragment(FloorFootprint{SystemTokens: -5, ToolsTokens: -10, ColdToolTokens: -1}, -3)
	if want := "floor=0+0 defer=0 Δtools=0"; got != want {
		t.Fatalf("clamped fragment = %q, want %q", got, want)
	}
}
