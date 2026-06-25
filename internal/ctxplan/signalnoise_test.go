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

// TestSignalNoiseFromAttention_SplitsByMass: the witnessed a_s ∈ [0,1] splits a span's
// cost — round(mass·cost) is signal, the remainder noise — granularity the boolean hit
// (all-or-nothing) cannot express. A half-attended span is half signal, half noise.
func TestSignalNoiseFromAttention_SplitsByMass(t *testing.T) {
	p := Plan{Selected: []Selection{
		sel("hot", 1000, false),  // fully attended
		sel("half", 1000, false), // half attended
		sel("cold", 1000, false), // never attended (witnessed mass 0)
	}}
	attr := Attribution{"hot": 1.0, "half": 0.5, "cold": 0.0}
	sn := SignalNoiseFromAttention(p, attr, Outcome{})

	if sn.SignalTokens != 1500 { // 1000 + 500 + 0
		t.Errorf("signal = %d, want 1500 (1.0+0.5+0.0 of 1000 each)", sn.SignalTokens)
	}
	if sn.NoiseTokens != 1500 { // 0 + 500 + 1000
		t.Errorf("noise = %d, want 1500", sn.NoiseTokens)
	}
	if sn.ResidentTokens != 3000 {
		t.Errorf("resident = %d, want 3000", sn.ResidentTokens)
	}
	if !approx(sn.Ratio(), 0.5) {
		t.Errorf("ratio = %v, want 0.5", sn.Ratio())
	}
}

// TestSignalNoiseFromAttention_PinIsSignalAtZeroMass: a pin stays pure signal even when the
// witness placed zero attention on it — same pin-is-signal rule the boolean path applies.
func TestSignalNoiseFromAttention_PinIsSignalAtZeroMass(t *testing.T) {
	p := Plan{Selected: []Selection{
		sel("pin", 500, true), // pinned, witnessed mass 0
		sel("seen", 500, false),
	}}
	attr := Attribution{"pin": 0.0, "seen": 1.0}
	sn := SignalNoiseFromAttention(p, attr, Outcome{})
	if sn.SignalTokens != 1000 {
		t.Errorf("pin must be signal at mass 0: signal=%d, want 1000", sn.SignalTokens)
	}
	if sn.NoiseTokens != 0 {
		t.Errorf("noise=%d, want 0", sn.NoiseTokens)
	}
}

// TestSignalNoiseFromAttention_DivergesFromInferred is the THESIS test for #854: attention
// sees what lexical overlap misses. The boolean path (rung 0) marks a lexically-similar
// span as Hit -> signal; the witnessed path sees the turn never ATTENDED it (mass 0) and
// drops it from signal to noise. Both ratios are computed side by side and asserted to
// DIVERGE — proving the witness is not a no-op relabel of the inferred metric.
func TestSignalNoiseFromAttention_DivergesFromInferred(t *testing.T) {
	// "decoy" is lexically similar to the query (so the bench's lexical-overlap heuristic
	// flags it Hit) but the model never actually attended to it.
	p := Plan{Selected: []Selection{
		sel("real", 1000, false),  // genuinely used
		sel("decoy", 1000, false), // lexically similar, NEVER attended
	}}

	// Rung 0: the inferred boolean path. Lexical overlap marks BOTH as hits.
	inferred := ComputeSignalNoise(p, Outcome{Hits: []string{"real", "decoy"}})
	// Rung 1→2: the witnessed path. Attention saw all weight on "real", none on "decoy".
	witnessed := SignalNoiseFromAttention(p, Attribution{"real": 1.0, "decoy": 0.0}, Outcome{})

	if !approx(inferred.Ratio(), 1.0) {
		t.Fatalf("inferred ratio = %v, want 1.0 (lexical overlap marks both spans signal)", inferred.Ratio())
	}
	if !approx(witnessed.Ratio(), 0.5) {
		t.Fatalf("witnessed ratio = %v, want 0.5 (only the truly-attended span is signal)", witnessed.Ratio())
	}
	if approx(inferred.Ratio(), witnessed.Ratio()) {
		t.Fatalf("witness must DIVERGE from inferred: both %v — attention added no information", inferred.Ratio())
	}
	// The decoy moved from signal (inferred) to noise (witnessed): the whole point of #854.
	if witnessed.NoiseTokens != 1000 {
		t.Errorf("the lexically-similar-but-unattended span must be noise: noise=%d, want 1000", witnessed.NoiseTokens)
	}
	if inferred.NoiseTokens != 0 {
		t.Errorf("inferred path counts the decoy as signal (its blind spot): noise=%d, want 0", inferred.NoiseTokens)
	}
	t.Logf("side by side: inferred S/N = %.3f (decoy counted as signal) vs witnessed S/N = %.3f (decoy is noise)",
		inferred.Ratio(), witnessed.Ratio())
}

// TestSignalNoiseFromAttention_UnattributedIsUnaccounted: a resident span the Attribution
// does not name (no witness on it) is honest unknown — denominator only, not signal/noise —
// so the witnessed ratio cannot over-claim when the witness is partial.
func TestSignalNoiseFromAttention_UnattributedIsUnaccounted(t *testing.T) {
	p := Plan{Selected: []Selection{
		sel("seen", 100, false),
		sel("unwitnessed", 100, false), // not in the attribution map at all
	}}
	sn := SignalNoiseFromAttention(p, Attribution{"seen": 1.0}, Outcome{})
	if sn.UnaccountedTokens != 100 {
		t.Errorf("unwitnessed resident span must be unaccounted: %d, want 100", sn.UnaccountedTokens)
	}
	if sn.SignalTokens != 100 || sn.NoiseTokens != 0 {
		t.Errorf("an absent-from-witness span must not become signal or noise: sig=%d noise=%d", sn.SignalTokens, sn.NoiseTokens)
	}
	if !approx(sn.Ratio(), 0.5) {
		t.Errorf("ratio = %v, want 0.5 (unaccounted in denominator)", sn.Ratio())
	}
}

// TestSignalNoiseFromAttention_FaultAxisMatchesInferred: fault cost (under-resident) is read
// from the same Outcome the boolean path uses — attention covers resident spans only, so the
// witnessed and inferred paths report IDENTICAL fault cost for the same elided-faulted span.
func TestSignalNoiseFromAttention_FaultAxisMatchesInferred(t *testing.T) {
	p := Plan{
		Selected: []Selection{sel("a", 100, false)},
		Elided:   []Elision{{ID: "needed", Cost: 5000, Reason: ElideOverBudget}},
	}
	o := Outcome{Faults: []string{"needed"}}
	witnessed := SignalNoiseFromAttention(p, Attribution{"a": 1.0}, o)
	inferred := ComputeSignalNoise(p, Outcome{Hits: []string{"a"}, Faults: []string{"needed"}})
	if witnessed.FaultTokens != 5000 || inferred.FaultTokens != 5000 {
		t.Errorf("fault cost must match across paths: witnessed=%d inferred=%d, want 5000 each",
			witnessed.FaultTokens, inferred.FaultTokens)
	}
}

// TestSignalNoiseFromAttention_ClampsMalformedMass: a witness mass outside [0,1] is clamped,
// so a normalization slip can never push a span past pure signal or below pure noise.
func TestSignalNoiseFromAttention_ClampsMalformedMass(t *testing.T) {
	p := Plan{Selected: []Selection{sel("over", 100, false), sel("under", 100, false)}}
	sn := SignalNoiseFromAttention(p, Attribution{"over": 1.7, "under": -0.3}, Outcome{})
	if sn.SignalTokens != 100 { // over clamps to 1.0 -> 100; under clamps to 0 -> 0
		t.Errorf("clamped signal = %d, want 100", sn.SignalTokens)
	}
	if sn.NoiseTokens != 100 {
		t.Errorf("clamped noise = %d, want 100", sn.NoiseTokens)
	}
}

// TestSignalNoise_BooleanPathUnchanged: the rung-0 fallback (ComputeSignalNoise) is the
// byte-identical offline path — with no attention witness available, the caller uses it and
// gets exactly today's behavior. This pins that the witnessed addition did not perturb it.
func TestSignalNoise_BooleanPathUnchanged(t *testing.T) {
	p := Plan{Selected: []Selection{
		sel("a", 100, false),
		sel("b", 100, false),
		sel("blob", 9000, false),
	}}
	o := Outcome{Hits: []string{"a", "b"}, Wasted: []string{"blob"}}
	sn := ComputeSignalNoise(p, o)
	if sn.SignalTokens != 200 || sn.NoiseTokens != 9000 || sn.ResidentTokens != 9200 {
		t.Errorf("boolean fallback perturbed: %+v", sn)
	}
	if !approx(sn.Ratio(), 200.0/9200.0) {
		t.Errorf("boolean ratio = %v, want %v", sn.Ratio(), 200.0/9200.0)
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
