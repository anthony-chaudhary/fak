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

// TestServeKnobTokenBudgetDefaultArmedWhenEnvUnset pins the default-armed
// per-step token budget on the serve path (Engine.nativeScheduler): env unset
// arms a positive budget derived from the model's declared context window, an
// explicit env value wins verbatim, an explicit 0 keeps the unlimited behavior,
// and a model that declares no window keeps the historical unlimited promotion
// (which is why TestServeKnobMaxTokensPerStepReachesScheduler's default-0
// assertion stays true). The exact fraction is pinned at the derivation helper,
// not at the serve-path assertion, so the fraction can move without rewriting
// the behavioral contract.
func TestServeKnobTokenBudgetDefaultArmedWhenEnvUnset(t *testing.T) {
	clearServeKnobEnv(t)

	withDeclaredCtx := func() *Engine {
		e := &Engine{cfg: SyntheticConfig()}
		e.cfg.MaxPositionEmbeddings = 4096
		return e
	}

	// (1) Env unset + a declared context window: the serve path arms a positive
	// budget derived from the model context, bounded by the window.
	if got := withDeclaredCtx().nativeScheduler().maxTokensPerStep; got <= 0 || got > 4096 {
		t.Fatalf("env unset, declared context 4096: maxTokensPerStep = %d, want a positive budget derived from the context (0 < b <= 4096)", got)
	}

	// (2) An explicit env value wins verbatim over the derived default.
	t.Setenv(envNativeMaxTokensPerStep, "77")
	if got := withDeclaredCtx().nativeScheduler().maxTokensPerStep; got != 77 {
		t.Fatalf("%s=77 with a declared context: maxTokensPerStep = %d, want 77 verbatim", envNativeMaxTokensPerStep, got)
	}

	// (3) An explicit 0 keeps today's unlimited behavior even with a declared
	// context: the default derivation must not override an explicit value.
	t.Setenv(envNativeMaxTokensPerStep, "0")
	if got := withDeclaredCtx().nativeScheduler().maxTokensPerStep; got != 0 {
		t.Fatalf("%s=0 with a declared context: maxTokensPerStep = %d, want 0 (unlimited)", envNativeMaxTokensPerStep, got)
	}

	// No declared window (the synthetic default): nothing to derive from, the
	// scheduler keeps today's unlimited promotion byte-for-byte.
	t.Setenv(envNativeMaxTokensPerStep, "")
	if got := New().nativeScheduler().maxTokensPerStep; got != 0 {
		t.Fatalf("env unset, no declared window: maxTokensPerStep = %d, want 0 (unlimited)", got)
	}

	// The derivation contract: one quarter of the declared window, minimum 1;
	// 0 for a model that declares no (positive) window.
	for _, tc := range []struct{ ctx, want int }{
		{ctx: 0, want: 0}, {ctx: -5, want: 0},
		{ctx: 3, want: 1}, {ctx: 4, want: 1},
		{ctx: 4096, want: 1024}, {ctx: 262144, want: 65536},
	} {
		if got := nativeTokenBudgetDefault(tc.ctx); got != tc.want {
			t.Fatalf("nativeTokenBudgetDefault(%d) = %d, want %d", tc.ctx, got, tc.want)
		}
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
