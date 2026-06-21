package model

import "testing"

// greedyContinue drives a session greedily from an initial logit vector (the last
// prefilled token's distribution), returning n decoded ids.
func greedyContinue(s *Session, logits []float32, n int) []int {
	out := make([]int, 0, n)
	for i := 0; i < n; i++ {
		nx := argmax(logits)
		out = append(out, nx)
		if s.M.Cfg.IsEOS(nx) {
			break
		}
		logits = s.Step(nx)
	}
	return out
}

func eq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestEvictRepositionsWithLayerSpecificRopeTheta(t *testing.T) {
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		RopeThetaPerLayer: []float64{10000, 1000000},
		BlockTopology:     SandwichNorm,
	}
	m := NewSynthetic(cfg)
	s := m.NewSession()
	all := []int{3, 17, 5, 23, 41, 2, 19}
	s.Prefill(all)
	s.Cache.Evict(2, 2)

	w, hd, nKV := s.Cache.kvStride(), cfg.HeadDim, cfg.NumKVHeads
	for i := 0; i < s.Cache.Len(); i++ {
		for l := 0; l < cfg.NumLayers; l++ {
			c, sn := ropeRowForLayer(cfg, l, i)
			for h := 0; h < nKV; h++ {
				want := append([]float32(nil), s.Cache.Kraw[l][i*w+h*hd:i*w+(h+1)*hd]...)
				applyRopeRow(want, c, sn)
				got := s.Cache.K[l][i*w+h*hd : i*w+(h+1)*hd]
				assertFloat32BitsEqual(t, "layer-specific evict L"+itoa(l)+" h"+itoa(h)+" pos"+itoa(i), want, got)
			}
		}
	}
}

// TestKVQuarantineEqualsNeverSaw is the rung-3 fusion-payoff witness — the whole
// point of pulling the model into the kernel. The context-MMU's quarantine, instead
// of being a string filter on result bytes, becomes EVICTION of the poisoned span's
// K/V from the kernel-owned cache. Three things are proven, including the BOUNDARY
// the adversarial review surfaced:
//
//  1. WRITE-TIME quarantine (end-span evict) == the never-saw-it run, token-for-token
//     vs HF. This is the real quarantine scenario: a poisoned tool RESULT is evicted
//     the moment it is produced, BEFORE the model's next turn attends to it.
//  2. The reposition is mechanically BIT-EXACT: every survivor's post-RoPE K equals a
//     single rotation of its pre-RoPE Kraw at its NEW position (what the pre-RoPE
//     store buys — composing two rotations would drift and flip a greedy token).
//  3. The BOUNDARY: a span evicted AFTER downstream tokens already attended to it
//     CANNOT be un-seen — those tokens' hidden states / cached V already absorbed it.
//     So middle-span evict != never. This is precisely WHY quarantine must be
//     write-time (ctxmmu's Admit gate), not a retroactive scrub.
//
// This is the witness build-plan landmine #7 demands: not "mark X poison, assert X
// absent" (proves nothing), but "evict X, assert output == the run that never saw X,
// assert the un-evicted run differs, AND prove the one thing eviction cannot do."
func TestKVQuarantineEqualsNeverSaw(t *testing.T) {
	m, doc := loadFixture(t)
	ev := doc.Eviction
	if len(ev.NeverGreedy) == 0 {
		t.Skip("no eviction fixture; re-run export_oracle.py")
	}
	cfg := m.Cfg
	P, Q := len(ev.PrefixIds), len(ev.PoisonIds)
	n := len(ev.NeverGreedy)

	// ---- (1) WRITE-TIME evict: quarantine the poison BEFORE the query attends -----
	s := m.NewSession()
	s.Prefill(ev.PrefixIds)
	s.Prefill(ev.PoisonIds)
	if s.Cache.Len() != P+Q {
		t.Fatalf("pre-evict cache len %d != %d", s.Cache.Len(), P+Q)
	}
	if removed := s.Cache.Evict(P, Q); removed != Q || s.Cache.Len() != P {
		t.Fatalf("evict removed %d (want %d), cache len %d (want %d)", removed, Q, s.Cache.Len(), P)
	}
	logits := s.Prefill(ev.QueryIds) // query prefilled AFTER eviction — never sees poison
	gotEvict := greedyContinue(s, logits, n)
	t.Logf("EVICT(write-time) go=%v", gotEvict)
	t.Logf("NEVER (HF ref)    hf=%v", ev.NeverGreedy)
	if !eq(gotEvict, ev.NeverGreedy) {
		t.Errorf("write-time KV-evicted continuation != HF never-saw-poison\n  go=%v\n  hf=%v",
			gotEvict, ev.NeverGreedy)
	}

	// ---- (2) reposition is bit-exact: K[i] == RoPE(Kraw[i], i) for every survivor ---
	// Build a cache where the poison sits in the MIDDLE (survivors follow it) so Evict
	// must reposition them, then assert the invariant directly (independent of greedy).
	rep := m.NewSession()
	rep.Prefill(append(append(append([]int{}, ev.PrefixIds...), ev.PoisonIds...), ev.QueryIds...))
	rep.Cache.Evict(P, Q)
	w, hd, nKV := rep.Cache.kvStride(), cfg.HeadDim, cfg.NumKVHeads
	var maxRe float64
	for i := 0; i < rep.Cache.Len(); i++ {
		for l := 0; l < cfg.NumLayers; l++ {
			c, sn := ropeRowForLayer(cfg, l, i)
			for h := 0; h < nKV; h++ {
				want := append([]float32(nil), rep.Cache.Kraw[l][i*w+h*hd:i*w+(h+1)*hd]...)
				applyRopeRow(want, c, sn)
				if d, _ := maxAbsDiff(want, rep.Cache.K[l][i*w+h*hd:i*w+(h+1)*hd]); d > maxRe {
					maxRe = d
				}
			}
		}
	}
	t.Logf("reposition invariant K==RoPE(Kraw,newpos): max|Δ|=%.3e (tol=%.0e)", maxRe, fmaCrossPathTol)
	if maxRe > fmaCrossPathTol {
		// Byte-exact on amd64 (fmaCrossPathTol==0); ≤1e-4 on arches where gc auto-fuses FMA
		// (see fmatol_other_test.go). A drift above this bound means rotations were actually
		// composed (the failure this rung guards against), which moves K by ≫1e-4, not the
		// sub-ULP FMA noise.
		t.Errorf("reposition drifted beyond FMA noise (max|Δ|=%.3e > %.0e) — composing rotations", maxRe, fmaCrossPathTol)
	}

	// ---- (3) the BOUNDARY: evicting AFTER downstream tokens attended can't un-see ---
	ql := ev.QueryIds
	mid := m.NewSession()
	mid.Prefill(append(append(append([]int{}, ev.PrefixIds...), ev.PoisonIds...), ql[:len(ql)-1]...))
	mid.Cache.Evict(P, Q)           // query[:-1] already absorbed the poison
	lmid := mid.Step(ql[len(ql)-1]) // continue from the contaminated context
	gotMid := greedyContinue(mid, lmid, n)
	t.Logf("MIDDLE(too-late) go=%v", gotMid)
	if eq(gotMid, ev.NeverGreedy) {
		t.Errorf("middle-span evict equaled never — the contamination model is wrong")
	} else {
		t.Logf("✓ a span evicted after downstream tokens attended is NOT un-seen — quarantine must be write-time")
	}

	// ---- (4) NEGATIVE control: keep the poison (no quarantine) -> HF poisoned -------
	s2 := m.NewSession()
	lg2 := s2.Prefill(append(append(append([]int{}, ev.PrefixIds...), ev.PoisonIds...), ev.QueryIds...))
	gotPoison := greedyContinue(s2, lg2, len(ev.PoisonedGreedy))
	t.Logf("POISON go=%v", gotPoison)
	t.Logf("POISON hf=%v", ev.PoisonedGreedy)
	if !eq(gotPoison, ev.PoisonedGreedy) {
		t.Errorf("un-evicted continuation != HF poisoned\n  go=%v\n  hf=%v", gotPoison, ev.PoisonedGreedy)
	}
	if eq(gotPoison[:n], ev.NeverGreedy) {
		t.Errorf("poison had NO effect — the quarantine witness would be vacuous")
	} else {
		t.Logf("✓ poison perturbs generation (poisoned != never), so the eviction guarantee is non-trivial")
	}
}
