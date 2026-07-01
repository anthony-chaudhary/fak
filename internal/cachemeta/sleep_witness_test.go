package cachemeta

import "testing"

// sleep_witness_test.go — issue #1730 acceptance witness.
//
// Proves the cache-loss evidence for vLLM sleep/pause/wake without any engine,
// GPU, or clock: a pure decision test over the witness lowering.

func TestWitnessDormancySleepForgetsKVAndDemotesWarm(t *testing.T) {
	// Acceptance #1: sleep(level=1/2) records kv_cache=forgotten and demotes warm
	// prefixes; the engine must not report serving while asleep.
	for _, lvl := range []EngineSleepLevel{SleepLevel1, SleepLevel2} {
		w := WitnessDormancy(DormancySleep, lvl)
		if w.KV != KVForgotten {
			t.Fatalf("sleep level %d: KV=%s, want forgotten", lvl, w.KV)
		}
		if !w.WarmPrefixesCold {
			t.Fatalf("sleep level %d: warm prefixes must be demoted cold", lvl)
		}
		if w.Serving {
			t.Fatalf("sleep level %d: engine must NOT report serving while asleep", lvl)
		}
		if w.Phase != PhaseSleeping {
			t.Fatalf("sleep level %d: phase=%s, want sleeping", lvl, w.Phase)
		}
	}
}

func TestWitnessDormancyPausePreservesKV(t *testing.T) {
	w := WitnessDormancy(DormancyPause, SleepNone)
	if w.KV != KVPreserved {
		t.Fatalf("pause: KV=%s, want preserved", w.KV)
	}
	if w.WarmPrefixesCold {
		t.Fatalf("pause: warm prefixes must stay warm (KV preserved)")
	}
	if w.Serving {
		t.Fatalf("pause: engine is not serving while paused")
	}
	if w.Phase != PhasePaused {
		t.Fatalf("pause: phase=%s, want paused", w.Phase)
	}
}

func TestWitnessDormancyResetForgetsKVButStaysUp(t *testing.T) {
	w := WitnessDormancy(DormancyReset, SleepNone)
	if w.KV != KVForgotten || !w.WarmPrefixesCold {
		t.Fatalf("reset: want forgotten+cold, got KV=%s cold=%v", w.KV, w.WarmPrefixesCold)
	}
	if !w.Serving {
		t.Fatalf("reset: engine stays up and serving after a prefix-cache reset")
	}
}

func TestWitnessDormancySleepWithoutLevelFailsClosed(t *testing.T) {
	w := WitnessDormancy(DormancySleep, SleepNone)
	if w.Phase != PhaseError {
		t.Fatalf("sleep with no level: phase=%s, want error", w.Phase)
	}
	if w.KV != KVDispositionUnknown || !w.WarmPrefixesCold {
		t.Fatalf("sleep with no level must fail closed: KV=%s cold=%v", w.KV, w.WarmPrefixesCold)
	}
}

func TestWarmHitGateRefusesWarmHitAfterSleepUntilRevalidated(t *testing.T) {
	// Acceptance #2: resume refuses a warm hit after a vLLM sleep/reset unless a
	// fresh BlockStored/cache signal revalidates it.
	var g WarmHitGate
	if g.WarmHitAllowed() {
		t.Fatalf("a fresh gate must not allow a warm hit before any cache signal")
	}
	g.ObserveCacheSignal() // a vLLM BlockStored proves the KV is resident
	if !g.WarmHitAllowed() {
		t.Fatalf("a fresh BlockStored signal should allow a warm hit")
	}
	g.ObserveDormancy(WitnessDormancy(DormancySleep, SleepLevel1))
	if g.WarmHitAllowed() {
		t.Fatalf("resume must NOT report a warm hit after a vLLM sleep")
	}
	g.ObserveDormancy(WitnessDormancy(DormancyWake, SleepNone))
	if g.WarmHitAllowed() {
		t.Fatalf("a bare wake must not re-warm the belief without a fresh cache signal")
	}
	g.ObserveCacheSignal() // a fresh BlockStored after wake revalidates
	if !g.WarmHitAllowed() {
		t.Fatalf("a fresh BlockStored after wake should revalidate the warm belief")
	}
}

func TestWitnessDormancyResetAlsoInvalidatesWarmGate(t *testing.T) {
	var g WarmHitGate
	g.ObserveCacheSignal()
	g.ObserveDormancy(WitnessDormancy(DormancyReset, SleepNone))
	if g.WarmHitAllowed() {
		t.Fatalf("a prefix-cache reset must invalidate the warm belief")
	}
}
