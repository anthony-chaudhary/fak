package engine

// cache_conformance_test.go — the SHARED adapter conformance harness, cache-frontier
// "Next 50" item 36 (#1554), External-engines lane under epic #1490. This is the
// conformance test the forward reference in llama.go ("item 36's conformance test")
// promised: one table-driven boundary proof over the three "fronted" external-engine
// observe adapters (vLLM #1551, SGLang #1552, llama.cpp #1553) that a base-url proxy
// observe-adapter can NEVER accidentally claim active cache integration.
//
// Two rungs per adapter:
//
//   - fail-closed default: an un-driven observer (constructed, Observe never called)
//     reports the explicit CacheUnknown label — never an inferred positive, and never
//     an Active() class; and
//   - observed stays passive: even a PRESENT/observed pure signal maps to at most
//     CachePassiveObserve — a fronted adapter that sees cache symptoms still cannot
//     report active integration.
//
// Plus one table-independent rung pinning that "active" is a CLOSED, named class of
// the CacheVerdict vocabulary (exactly warm / exact-evict / prefix-clone), so the
// boundary the adapters provably never cross is itself enum-enforced. Any NEW
// external-engine observe adapter must be added to the table below and pass both
// rungs before it may ship.

import "testing"

// TestFrontedAdaptersNeverClaimActiveCacheIntegration drives every fronted
// external-engine observe adapter through the two conformance rungs: the un-driven
// producer fails closed to CacheUnknown, and a present/observed signal caps at
// CachePassiveObserve. Both rungs assert the core property !Verdict.Active() — the
// "cannot claim active integration" boundary — plus ProvenanceProvider,
// ColdPathCorrect, and well-formedness.
func TestFrontedAdaptersNeverClaimActiveCacheIntegration(t *testing.T) {
	cases := []struct {
		name   string
		engine string
		// unobserved is the producer as a base-url proxy would first hold it:
		// constructed from config only, Observe never called.
		unobserved CacheCapabilityProducer
		// observed is the capability mapped from a PRESENT/observed pure signal —
		// the strongest evidence a fronted adapter can ever hold.
		observed CacheCapability
	}{
		{
			name:       "vllm",
			engine:     VLLMEngineID,
			unobserved: NewVLLMCacheObserver(VLLMConfig{BaseURL: "http://example.invalid/v1"}),
			observed:   VLLMPrefixCacheSignal{Present: true, Queries: 10, Hits: 7}.Capability(),
		},
		{
			name:       "sglang",
			engine:     SGLangEngineID,
			unobserved: NewSGLangCacheObserver(SGLangConfig{BaseURL: "http://example.invalid"}),
			observed:   SGLangRadixCacheSignal{Present: true, HitRatio: 0.7}.Capability(),
		},
		{
			name:       "llama.cpp",
			engine:     LlamaEngineID,
			unobserved: NewLlamaCacheObserver(LlamaConfig{BaseURL: "http://example.invalid"}),
			observed:   LlamaSessionCache{Present: true, Slots: 2, Fields: []string{"cache_tokens", "n_past"}}.Capability(),
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Rung 1 — fail-closed default: the un-driven base-url proxy reports the
			// explicit unknown label, never an inferred positive.
			t.Run("fail-closed-default", func(t *testing.T) {
				cap := tc.unobserved.CacheCapability()
				if cap.Engine != tc.engine {
					t.Errorf("Engine = %q, want %q", cap.Engine, tc.engine)
				}
				if cap.Verdict != CacheUnknown {
					t.Errorf("un-observed Verdict = %q, want %q (fail closed)", cap.Verdict, CacheUnknown)
				}
				if cap.Verdict.Active() {
					t.Errorf("a fronted adapter's fail-closed default must never claim active cache integration: %+v", cap)
				}
				if cap.Provenance != ProvenanceProvider {
					t.Errorf("Provenance = %q, want %q (observed provider surface, not a kernel witness)", cap.Provenance, ProvenanceProvider)
				}
				if !cap.ColdPathCorrect {
					t.Errorf("ColdPathCorrect must be true: observation changes no serving path")
				}
				if !cap.Valid() {
					t.Errorf("fail-closed capability must be well-formed: %+v", cap)
				}
			})

			// Rung 2 — even WITH a present/observed signal the mapping caps at
			// passive-observe: seeing cache symptoms is still not active integration.
			t.Run("observed-stays-passive", func(t *testing.T) {
				cap := tc.observed
				if cap.Engine != tc.engine {
					t.Errorf("Engine = %q, want %q", cap.Engine, tc.engine)
				}
				if cap.Verdict != CachePassiveObserve {
					t.Errorf("observed-signal Verdict = %q, want %q (fronted adapters cap at passive-observe)", cap.Verdict, CachePassiveObserve)
				}
				if cap.Verdict.Active() {
					t.Errorf("a fronted adapter that observed a signal still must not claim active cache integration: %+v", cap)
				}
				if !cap.ColdPathCorrect {
					t.Errorf("ColdPathCorrect must stay true on the observed path")
				}
				if !cap.Valid() {
					t.Errorf("observed capability must be well-formed: %+v", cap)
				}
			})
		})
	}
}

// TestActiveVerdictClassIsClosed pins the boundary the enum itself enforces: exactly
// {CacheActiveWarm, CacheExactEvict, CachePrefixClone} report Active() == true, and
// every other member of the closed CacheVerdict vocabulary reports false. This is the
// named class the fronted adapters above provably never enter — a new verdict cannot
// silently join it, and passive/unknown can never drift into it.
func TestActiveVerdictClassIsClosed(t *testing.T) {
	activeClass := map[CacheVerdict]bool{
		CacheActiveWarm:  true,
		CacheExactEvict:  true,
		CachePrefixClone: true,
	}
	seenActive := 0
	for _, v := range CacheVerdicts() {
		if got, want := v.Active(), activeClass[v]; got != want {
			t.Errorf("Active(%q) = %v, want %v", v, got, want)
		}
		if v.Active() {
			seenActive++
		}
	}
	if seenActive != len(activeClass) {
		t.Errorf("vocabulary reports %d active verdicts, want exactly %d (warm / exact-evict / prefix-clone)", seenActive, len(activeClass))
	}
	// Every member of the active class must itself be in the closed vocabulary —
	// the class cannot name a verdict the enum does not admit.
	for v := range activeClass {
		if !v.Valid() {
			t.Errorf("active-class verdict %q is not a member of the closed vocabulary", v)
		}
	}
}
