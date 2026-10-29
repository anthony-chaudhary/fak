package modelengine

import (
	"context"
	"testing"
)

// clearServeKnobEnv pins every scheduler env knob to its disabled value so a test
// sees only the knob it sets.
func clearServeKnobEnv(t *testing.T) {
	t.Helper()
	for _, name := range []string{
		"FAK_MODEL_DIR", "FAK_NATIVE_MAX_RUNNING", "FAK_NATIVE_IN_BATCH_PREFIX_DEDUP",
		"FAK_NATIVE_KV_MAX_BLOCKS", "FAK_NATIVE_KV_BLOCK_TOKENS", "FAK_NATIVE_KV_PREEMPT_MODE",
		"FAK_NATIVE_KV_VICTIM_RULE", envNativeMaxTokensPerStep, envNativeDeterministic,
		envNativeKVAutosize, envNativeKVElasticBlocks,
	} {
		t.Setenv(name, "")
	}
}

func TestServeKnobMaxTokensPerStepReachesScheduler(t *testing.T) {
	clearServeKnobEnv(t)
	if got := New().nativeScheduler().maxTokensPerStep; got != 0 {
		t.Fatalf("default maxTokensPerStep = %d, want 0 (off)", got)
	}
	t.Setenv(envNativeMaxTokensPerStep, "64")
	if got := New().nativeScheduler().maxTokensPerStep; got != 64 {
		t.Fatalf("%s=64 maxTokensPerStep = %d, want 64", envNativeMaxTokensPerStep, got)
	}
}

func TestServeKnobDeterministicModeReachesScheduler(t *testing.T) {
	clearServeKnobEnv(t)
	if New().nativeScheduler().DeterministicMode() {
		t.Fatal("default DeterministicMode = true, want false (off)")
	}
	t.Setenv(envNativeDeterministic, "1")
	if !New().nativeScheduler().DeterministicMode() {
		t.Fatalf("%s=1 DeterministicMode = false, want true", envNativeDeterministic)
	}
}

func TestServeKnobKVAutosizeGatesFreeBytesProbe(t *testing.T) {
	clearServeKnobEnv(t)
	prev := nativeKVFreeBytesProbe
	t.Cleanup(func() { nativeKVFreeBytesProbe = prev })
	calls := 0
	nativeKVFreeBytesProbe = func() (uint64, bool) { calls++; return 1 << 30, true }

	if got := New().nativeScheduler().preemption.MaxBlocks; got != 0 || calls != 0 {
		t.Fatalf("default MaxBlocks = %d probe calls = %d, want 0/0 (probe not consulted)", got, calls)
	}
	t.Setenv(envNativeKVAutosize, "1")
	if got := New().nativeScheduler().preemption.MaxBlocks; got <= 0 || calls != 1 {
		t.Fatalf("%s=1 MaxBlocks = %d probe calls = %d, want >0/1", envNativeKVAutosize, got, calls)
	}
}

func TestKVFreeBytesFromBackendNilIsUnknown(t *testing.T) {
	if free, ok := kvFreeBytesFromBackend(nil); ok || free != 0 {
		t.Fatalf("kvFreeBytesFromBackend(nil) = %d,%v, want 0,false", free, ok)
	}
}

func TestServeKnobLowestPriorityVictimRule(t *testing.T) {
	clearServeKnobEnv(t)
	if got := nativePreemptionPolicyFromEnv().VictimRule; got != NativePreemptVictimMostRecent {
		t.Fatalf("default VictimRule = %v, want most-recent", got)
	}
	for _, v := range []string{"lowest-priority", "priority"} {
		t.Setenv("FAK_NATIVE_KV_VICTIM_RULE", v)
		if got := nativePreemptionPolicyFromEnv().VictimRule; got != NativePreemptVictimLowestPriority {
			t.Fatalf("FAK_NATIVE_KV_VICTIM_RULE=%s VictimRule = %v, want lowest-priority", v, got)
		}
	}
}

func TestServeKnobElasticLedgerReachesAllocator(t *testing.T) {
	clearServeKnobEnv(t)
	if _, ok := New().nativeScheduler().ElasticLedgerStats(); ok {
		t.Fatal("default elastic ledger armed, want off")
	}
	t.Setenv(envNativeKVElasticBlocks, "64")
	e := New()
	r, err := e.Admit(context.Background(), inlineCall("search_flights", `{"from":"SFO"}`))
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}
	for range r.Tokens() {
	}
	if _, err := r.Result(); err != nil {
		t.Fatalf("Result: %v", err)
	}
	st, ok := e.nativeScheduler().ElasticLedgerStats()
	if !ok {
		t.Fatalf("%s=64 elastic ledger not armed", envNativeKVElasticBlocks)
	}
	if st.AdmittedLanes != 1 || st.AllocatedBlocks == 0 || st.CachedReleases != st.AllocatedBlocks {
		t.Fatalf("ledger stats = %+v, want 1 lane with every allocated block released cached", st)
	}
	if st.Allocator.AllocatedBlocks != 0 || st.Allocator.CurrentTotalBlocks != 64 {
		t.Fatalf("allocator stats = %+v, want 0 allocated of 64", st.Allocator)
	}
	a := e.nativeScheduler().elastic.alloc
	a.mu.Lock()
	cached := len(a.cachedBlocks)
	a.mu.Unlock()
	if int64(cached) != st.CachedReleases {
		t.Fatalf("allocator cached FIFO len = %d, want %d (ReleaseCachedBlock path)", cached, st.CachedReleases)
	}
}
