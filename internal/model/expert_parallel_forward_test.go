package model

import "testing"

// expert_parallel_forward_test.go — the whole-forward correctness gates for ForwardEP
// (expert_parallel_forward.go), the EP twin of tensor_parallel_forward_test.go's ForwardTP
// gates. expert_parallel_test.go proved the EP delta per-LAYER; these pin it through a full
// glm_moe_dsa prefill:
//
//	ForwardEP(ranks=1)  ==(bit-exact, max|Δ|=0)        Forward
//	ForwardEP(ranks=N)  ==(AllReduce reassociation)    Forward   (argmax-stable, cosine ≈ 1)
//
// Forward is gated bit-exact against the HF oracle on this glm_moe_dsa path (glm_test.go /
// oracle_test.go), so ForwardEP inherits it transitively. The fixture is loaded with
// NumExpertsPerTok=2 over its 2 experts so BOTH experts fire every token — a genuine 2-partial
// AllReduceSum at ranks=2 (not the degenerate single-pick case the K=1 default would give).

func epForwardTestIDs() []int { return []int{3, 17, 5, 23} } // all < VocabSize(41)

// TestForwardEPMatchesForward is the end-to-end gate: a full ForwardEP over the glm_moe_dsa
// fixture is bit-identical to Forward at ranks=1 and argmax-stable within reassociation
// round-off at ranks=2 (one expert per rank, both firing).
func TestForwardEPMatchesForward(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	cfg.NumExpertsPerTok = 2 // top-2 over the fixture's 2 experts -> both fire -> genuine multi-partial reduce
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if !cfg.isGLMMoeDsa() || !cfg.IsMoE() {
		t.Fatalf("fixture not glm_moe_dsa MoE: isGLMMoeDsa=%v IsMoE=%v", cfg.isGLMMoeDsa(), cfg.IsMoE())
	}

	ids := epForwardTestIDs()
	ref := m.Forward(ids) // the monolith reference (HF-oracle-gated on this path)

	for _, ranks := range []int{1, 2} {
		act, err := m.ForwardEP(ids, EPConfig{Ranks: ranks, Coll: LocalCollective{}})
		if err != nil {
			t.Fatalf("ranks=%d: ForwardEP: %v", ranks, err)
		}
		if len(act.Logits) != len(ref.Logits) {
			t.Fatalf("ranks=%d: ForwardEP returned %d logit rows, want %d", ranks, len(act.Logits), len(ref.Logits))
		}
		for pos := range ref.Logits {
			if ranks == 1 {
				// ranks=1: every expert in one band, identity reduce -> bit-exact vs the monolith.
				if mx := epMaxAbs(act.Logits[pos], ref.Logits[pos]); mx != 0 {
					t.Fatalf("ForwardEP(ranks=1) logits[%d] != Forward: max|Δ|=%.3e (want bit-exact 0)", pos, mx)
				}
				continue
			}
			// ranks>1: the routed sum is regrouped across expert bands -> matches the monolith
			// within the single AllReduceSum's reassociation round-off (argmax must not move).
			if argmaxF32(act.Logits[pos]) != argmaxF32(ref.Logits[pos]) {
				t.Fatalf("ForwardEP(ranks=%d) argmax[%d]=%d != Forward argmax=%d",
					ranks, pos, argmaxF32(act.Logits[pos]), argmaxF32(ref.Logits[pos]))
			}
			if cos := epCosine(act.Logits[pos], ref.Logits[pos]); cos < 0.99999 {
				t.Fatalf("ForwardEP(ranks=%d) logits[%d] vs Forward cosine=%.8f, want ≥ 0.99999 (reassociation only)", ranks, pos, cos)
			}
		}
		t.Logf("ForwardEP(ranks=%d) over glm_moe_dsa: ranks=1 bit-exact / ranks>1 argmax-stable vs Forward", ranks)
	}
}

// TestForwardEPViaBackendCollective pins that ForwardEP driven by the compute.CollectiveBackend
// HAL bridge (the NCCL plug-in seam) reproduces ForwardEP driven by LocalCollective
// bit-for-bit — so a real device collective dropped in behind BackendCollective moves no number
// (the EP twin of collective_bridge_test.go's TestForwardTPViaBackendCollective).
func TestForwardEPViaBackendCollective(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true, true)
	cfg.NumExpertsPerTok = 2
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	ids := epForwardTestIDs()
	bc := mustBackendColl(t) // cpu-ref CollectiveBackend behind the HAL bridge (the device-collective drop-in seam)

	for _, ranks := range []int{1, 2} {
		refAct, err := m.ForwardEP(ids, EPConfig{Ranks: ranks, Coll: LocalCollective{}})
		if err != nil {
			t.Fatalf("ranks=%d: ForwardEP via Local: %v", ranks, err)
		}
		gotAct, err := m.ForwardEP(ids, EPConfig{Ranks: ranks, Coll: bc})
		if err != nil {
			t.Fatalf("ranks=%d: ForwardEP via BackendCollective: %v", ranks, err)
		}
		for pos := range refAct.Logits {
			if mx := epMaxAbs(gotAct.Logits[pos], refAct.Logits[pos]); mx != 0 {
				t.Fatalf("ranks=%d: ForwardEP via BackendCollective != via LocalCollective at logits[%d]: max|Δ|=%.3e, want 0", ranks, pos, mx)
			}
		}
	}
	t.Logf("ForwardEP via BackendCollective == via LocalCollective, bit-exact (the device-collective drop-in seam)")
}

// TestForwardEPFailsClosed pins the fail-closed boundary: ForwardEP refuses a dense (non-MoE)
// glm_moe_dsa config and a non-glm MoE model rather than mis-serving an unsupported decomposition.
func TestForwardEPFailsClosed(t *testing.T) {
	ids := epForwardTestIDs()

	// Dense glm_moe_dsa (withMoE=false) -> not an MoE config -> rejected.
	densePath, denseCfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, false /*withMoE*/, false)
	dense, err := LoadSafetensors(densePath, denseCfg)
	if err != nil {
		t.Fatalf("LoadSafetensors(dense glm): %v", err)
	}
	if dense.Cfg.IsMoE() {
		t.Fatalf("test setup: dense glm fixture should not be MoE")
	}
	if _, err := dense.ForwardEP(ids, EPConfig{Ranks: 1, Coll: LocalCollective{}}); err == nil {
		t.Fatalf("ForwardEP should reject a dense (non-MoE) glm_moe_dsa config")
	}

	// Non-glm MoE model (generic synthetic MoE) -> not glm_moe_dsa -> rejected (it is ForwardTP's
	// expert-parallel job only once an MLA-aware path exists; the EP forward is glm-specific).
	gen := epGenMoeModel(8, 4)
	if gen.Cfg.isGLMMoeDsa() {
		t.Fatalf("test setup: generic MoE should not be glm_moe_dsa")
	}
	if _, err := gen.ForwardEP(ids, EPConfig{Ranks: 2, Coll: LocalCollective{}}); err == nil {
		t.Fatalf("ForwardEP should reject a non-glm_moe_dsa MoE model")
	}
}
