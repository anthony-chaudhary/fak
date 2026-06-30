package frontierswe

import "testing"

// longHorizon is a representative FrontierSWE geometry: 2,000 turns whose resident
// context grows to ~200k tokens — the regime the TTS floor exists to model (the
// quadratic re-prefill integral over a 20-hour trajectory). Prefix 8k + 2000·(64+32)
// = 8000 + 192000 = 200000 tokens of resident context at the end of the trajectory.
func longHorizon() TaskGeometry {
	return TaskGeometry{Name: "git-to-zig", Prefix: 8000, Turns: 2000, Decode: 64, Result: 32}
}

func TestMaxContextReaches200k(t *testing.T) {
	g := longHorizon()
	if got := g.MaxContext(); got != 200000 {
		t.Fatalf("MaxContext = %d, want 200000 (8000 + 2000*(64+32))", got)
	}
}

// TestPrefillWorkShape pins the WITNESSED-by-construction integrals: A is the
// quadratic naive re-prefill sum and is far larger than the linear per-agent-KV B,
// so the turn-tax A/B is large for a long-horizon trajectory.
func TestPrefillWorkShape(t *testing.T) {
	g := longHorizon()
	a, b := PrefillWork(g)

	// A = Σ_{t=0..T-1}(P + t·(D+R)) = T·P + (D+R)·T·(T-1)/2.
	P, T, DR := int64(g.Prefix), int64(g.Turns), int64(g.Decode+g.Result)
	wantA := T*P + DR*T*(T-1)/2
	if a != wantA {
		t.Fatalf("A = %d, want %d (quadratic re-prefill integral)", a, wantA)
	}
	// B = P + (T-1)·R.
	wantB := P + (T-1)*int64(g.Result)
	if b != wantB {
		t.Fatalf("B = %d, want %d (prefix + incremental result ingest)", b, wantB)
	}
	if a <= b {
		t.Fatalf("A (%d) must dominate B (%d) for a long-horizon trajectory", a, b)
	}
	if wantA == 0 {
		t.Fatal("sanity: A should be non-zero")
	}
}

// TestEndpointArms checks the C arm collapses to A at r=0 (no reuse, no speedup)
// and to the per-agent-KV floor B at r=1 (full turn-tax removed), so the TTS ratio
// runs from 1.0 down to B/A.
func TestEndpointArms(t *testing.T) {
	m := DefaultTTSModel()
	g := longHorizon()
	a, b := PrefillWork(g)

	w0 := m.Derive(g, 0)
	if int64(w0.C) != a {
		t.Errorf("r=0: C = %v, want A = %d (naive fallback)", w0.C, a)
	}
	if w0.AOverC != 1.0 {
		t.Errorf("r=0: A/C = %v, want 1.0", w0.AOverC)
	}
	if w0.TTSRatio != 1.0 {
		t.Errorf("r=0: TTS ratio = %v, want 1.0 (no speedup with no reuse)", w0.TTSRatio)
	}

	w1 := m.Derive(g, 1)
	if int64(w1.C) != b {
		t.Errorf("r=1: C = %v, want B = %d (per-agent-KV floor)", w1.C, b)
	}
	wantTTS := float64(b) / float64(a)
	if w1.TTSRatio != wantTTS {
		t.Errorf("r=1: TTS ratio = %v, want B/A = %v", w1.TTSRatio, wantTTS)
	}
	// At full reuse the net work-elimination equals the turn-tax ceiling.
	if w1.AOverC != w1.AOverB {
		t.Errorf("r=1: A/C (%v) should equal the turn-tax A/B (%v)", w1.AOverC, w1.AOverB)
	}
}

// TestTurnTaxRIndependent checks A/B (the structural turn-tax) does not move with
// the reuse rate — it is the ceiling A/C climbs toward, fixed by the geometry.
func TestTurnTaxRIndependent(t *testing.T) {
	m := DefaultTTSModel()
	g := longHorizon()
	var prev float64
	for i, r := range []float64{0, 0.25, 0.5, 0.75, 1.0} {
		w := m.Derive(g, r)
		if i > 0 && w.AOverB != prev {
			t.Fatalf("A/B moved with r: %v != %v at r=%v", w.AOverB, prev, r)
		}
		prev = w.AOverB
	}
	if prev <= 1.0 {
		t.Fatalf("turn-tax A/B = %v, want > 1 for a long-horizon trajectory", prev)
	}
}

// TestMonotoneInReuse is the acceptance assertion: A/C rises monotonically and the
// TTS ratio falls monotonically as the reuse rate r increases — the value curve the
// floor projects.
func TestMonotoneInReuse(t *testing.T) {
	m := DefaultTTSModel()
	g := longHorizon()
	rs := []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

	var prevAOverC, prevTTS float64
	for i, r := range rs {
		w := m.Derive(g, r)
		if i > 0 {
			if !(w.AOverC > prevAOverC) {
				t.Errorf("A/C not strictly increasing at r=%v: %v <= %v", r, w.AOverC, prevAOverC)
			}
			if !(w.TTSRatio < prevTTS) {
				t.Errorf("TTS ratio not strictly decreasing at r=%v: %v >= %v", r, w.TTSRatio, prevTTS)
			}
		}
		// TTS ratio and A/C are reciprocals by construction (TTSRatio = C/A, A/C = A/C).
		if w.AOverC != 0 {
			recip := 1.0 / w.AOverC
			if diff := recip - w.TTSRatio; diff > 1e-9 || diff < -1e-9 {
				t.Errorf("r=%v: TTS ratio %v should be 1/(A/C) = %v", r, w.TTSRatio, recip)
			}
		}
		prevAOverC, prevTTS = w.AOverC, w.TTSRatio
	}

	// The free-function TTSRatio agrees with the model and is itself monotone.
	if got := TTSRatio(g, 1.0); got != m.Derive(g, 1.0).TTSRatio {
		t.Errorf("TTSRatio() free fn = %v disagrees with model %v", got, m.Derive(g, 1.0).TTSRatio)
	}
	if TTSRatio(g, 0.5) >= TTSRatio(g, 0.25) {
		t.Errorf("TTSRatio not decreasing in r: %v >= %v", TTSRatio(g, 0.5), TTSRatio(g, 0.25))
	}
}

// TestReuseClamped checks r outside [0,1] is clamped, so callers cannot project a
// speedup beyond the per-agent-KV floor or worse than naive.
func TestReuseClamped(t *testing.T) {
	m := DefaultTTSModel()
	g := longHorizon()
	if m.Derive(g, -0.5).TTSRatio != m.Derive(g, 0).TTSRatio {
		t.Error("negative r should clamp to r=0")
	}
	if m.Derive(g, 2.0).TTSRatio != m.Derive(g, 1).TTSRatio {
		t.Error("r>1 should clamp to r=1")
	}
}
