package ctxplan

import (
	"math"
	"testing"
)

func sel(id string, cost int, pinned bool) Selection {
	return Selection{ID: id, Cost: cost, Pinned: pinned}
}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// TestSignalNoise_TokenWeighted is the core property: noise is weighted by the TOKENS it
// occupies, so one large stale blob dominates many small live spans — exactly the bloat
// the cache-hit number hides.
func TestSignalNoise_TokenWeighted(t *testing.T) {
	p := Plan{Selected: []Selection{
		sel("a", 100, false),     // referenced -> signal
		sel("b", 100, false),     // referenced -> signal
		sel("blob", 9000, false), // resident, never touched -> noise (the bloat)
	}}
	o := Outcome{Hits: []string{"a", "b"}, Wasted: []string{"blob"}}
	sn := ComputeSignalNoise(p, o)

	if sn.SignalTokens != 200 {
		t.Errorf("signal = %d, want 200", sn.SignalTokens)
	}
	if sn.NoiseTokens != 9000 {
		t.Errorf("noise = %d, want 9000", sn.NoiseTokens)
	}
	if sn.ResidentTokens != 9200 {
		t.Errorf("resident = %d, want 9200", sn.ResidentTokens)
	}
	// The blob dominates: only ~2% of the window pulled weight, even though 2 of 3 spans
	// were referenced. A span-COUNT metric would say 67%; the token metric says the truth.
	if !approx(sn.Ratio(), 200.0/9200.0) {
		t.Errorf("ratio = %v, want %v", sn.Ratio(), 200.0/9200.0)
	}
	if sn.Grade() != "bloated" {
		t.Errorf("a 2%%-signal window must grade bloated, got %q", sn.Grade())
	}
}

// TestSignalNoise_InvariantToCaching is the THESIS test: cache-hit % is irrelevant to
// S/N. We model two turns with IDENTICAL resident content (so identical cache behavior),
// one where the big span is referenced (signal) and one where it idles (noise). S/N tells
// them apart; cache-hit — which depends only on what bytes are resident, not whether they
// were used — cannot.
func TestSignalNoise_InvariantToCaching(t *testing.T) {
	resident := []Selection{sel("small", 100, false), sel("big", 9000, false)}
	plan := Plan{Selected: resident}

	used := ComputeSignalNoise(plan, Outcome{Hits: []string{"small", "big"}})
	idle := ComputeSignalNoise(plan, Outcome{Hits: []string{"small"}, Wasted: []string{"big"}})

	// Same bytes resident => a provider would report the SAME cache-hit for both. But:
	if !approx(used.Ratio(), 1.0) {
		t.Errorf("fully-referenced window ratio = %v, want 1.0", used.Ratio())
	}
	if used.Ratio() <= idle.Ratio() {
		t.Fatalf("S/N must distinguish used (%v) from idle (%v) context that caches identically",
			used.Ratio(), idle.Ratio())
	}
	if idle.Ratio() >= 0.5 {
		t.Errorf("idle big-span window must be bloated (ratio %v)", idle.Ratio())
	}
}

// TestSignalNoise_PinsAreSignal: a pinned span is signal even if a given turn did not cite
// it — the turn cannot proceed without it, so it is never idle noise.
func TestSignalNoise_PinsAreSignal(t *testing.T) {
	p := Plan{Selected: []Selection{
		sel("pin", 500, true), // pinned, NOT in Hits this turn
		sel("noise", 500, false),
	}}
	o := Outcome{Wasted: []string{"noise"}} // pin not listed anywhere
	sn := ComputeSignalNoise(p, o)
	if sn.SignalTokens != 500 {
		t.Errorf("pin must count as signal: signal=%d, want 500", sn.SignalTokens)
	}
	if sn.NoiseTokens != 500 {
		t.Errorf("noise=%d, want 500", sn.NoiseTokens)
	}
}

// TestSignalNoise_FaultsAreSeparateAxis: trimming a needed span out of the window must NOT
// raise the ratio — it moves cost to the fault axis (under-resident), graded "starving".
func TestSignalNoise_FaultsAreSeparateAxis(t *testing.T) {
	// Lean resident view (all signal), but a big needed span was elided and faulted back.
	p := Plan{
		Selected: []Selection{sel("a", 100, false)},
		Elided:   []Elision{{ID: "needed", Cost: 5000, Reason: ElideOverBudget}},
	}
	o := Outcome{Hits: []string{"a"}, Faults: []string{"needed"}}
	sn := ComputeSignalNoise(p, o)

	if !approx(sn.Ratio(), 1.0) {
		t.Errorf("resident ratio = %v, want 1.0 (every resident token was signal)", sn.Ratio())
	}
	if sn.FaultTokens != 5000 {
		t.Errorf("fault cost = %d, want 5000 (the elided needed span)", sn.FaultTokens)
	}
	// A perfect resident ratio but heavy faulting is OVER-trimmed, not healthy.
	if sn.Grade() != "starving" {
		t.Errorf("lean-but-faulting window must grade starving, got %q", sn.Grade())
	}
	if sn.FaultRatio() <= 0.9 {
		t.Errorf("fault ratio = %v, want > 0.9", sn.FaultRatio())
	}
}

// TestSignalNoise_Unaccounted: a resident span the Outcome never labeled is neither signal
// nor noise — it sits in the denominator as honest "unknown", so the ratio can't over-claim.
func TestSignalNoise_Unaccounted(t *testing.T) {
	p := Plan{Selected: []Selection{
		sel("hit", 100, false),
		sel("mystery", 100, false), // not in Hits, not in Wasted
	}}
	sn := ComputeSignalNoise(p, Outcome{Hits: []string{"hit"}})
	if sn.UnaccountedTokens != 100 {
		t.Errorf("unaccounted = %d, want 100", sn.UnaccountedTokens)
	}
	if sn.SignalTokens != 100 || sn.NoiseTokens != 0 {
		t.Errorf("unlabeled span must not become signal or noise: sig=%d noise=%d", sn.SignalTokens, sn.NoiseTokens)
	}
	// Ratio is signal/resident = 100/200 = 0.5, NOT 1.0 — the unknown sits in the denom.
	if !approx(sn.Ratio(), 0.5) {
		t.Errorf("ratio = %v, want 0.5 (unaccounted in denominator)", sn.Ratio())
	}
}

func TestSignalNoise_EmptyAndGrades(t *testing.T) {
	// Empty resident view: nothing to curate => ratio 1.0 (fail-to-best).
	if got := ComputeSignalNoise(Plan{}, Outcome{}).Ratio(); !approx(got, 1.0) {
		t.Errorf("empty plan ratio = %v, want 1.0", got)
	}
	// Lean: high signal, no faults.
	lean := SignalNoise{SignalTokens: 90, ResidentTokens: 100}
	if lean.Grade() != "lean" {
		t.Errorf("90%% signal, 0 fault => lean, got %q", lean.Grade())
	}
	// Stale id naming no span teaches nothing (fail-closed).
	p := Plan{Selected: []Selection{sel("a", 100, false)}}
	sn := ComputeSignalNoise(p, Outcome{Hits: []string{"ghost"}, Wasted: []string{"phantom"}})
	if sn.SignalTokens != 0 || sn.NoiseTokens != 0 || sn.UnaccountedTokens != 100 {
		t.Errorf("ids naming no resident span must not move signal/noise: %+v", sn)
	}
}
