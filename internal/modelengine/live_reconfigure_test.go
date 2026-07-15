package modelengine

import (
	"context"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestEngineLiveKVPreemptionPolicyAfterAdmit(t *testing.T) {
	t.Setenv("FAK_NATIVE_KV_MAX_BLOCKS", "0")
	e := New()
	ctx := context.Background()
	first, err := e.Admit(ctx, inlineCall("live_first", `{"prompt":"alpha"}`))
	if err != nil {
		t.Fatal(err)
	}
	// The first Admit has initialized the scheduler with the unchanged env boot path.
	if got := e.KVPreemptionStats().MaxBlocks; got != 0 {
		t.Fatalf("boot max blocks=%d, want env default 0", got)
	}
	e.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptRecompute, MaxBlocks: 1, BlockTokens: 128})
	second, err := e.Admit(ctx, inlineCall("live_second", `{"prompt":"bravo"}`))
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for _, r := range []abi.EngineRequest{first, second} {
		wg.Add(1)
		go func(r abi.EngineRequest) {
			defer wg.Done()
			for range r.Tokens() {
			}
		}(r)
	}
	wg.Wait()
	for _, r := range []abi.EngineRequest{first, second} {
		if _, err := r.Result(); err != nil {
			t.Fatal(err)
		}
	}
	stats := e.KVPreemptionStats()
	if stats.MaxBlocks != 1 || stats.Preemptions == 0 || stats.RecomputeCount == 0 {
		t.Fatalf("live stats=%+v, want shrink-triggered recompute preemption", stats)
	}
}

func TestEngineSetMaxRunningReachesInitializedScheduler(t *testing.T) {
	e := New()
	_ = e.nativeScheduler()
	e.SetMaxRunning(1)
	ctx := context.Background()
	var reqs []abi.EngineRequest
	for _, tool := range []string{"one", "two", "three"} {
		r, err := e.Admit(ctx, inlineCall(tool, `{}`))
		if err != nil {
			t.Fatal(err)
		}
		reqs = append(reqs, r)
	}
	var wg sync.WaitGroup
	for _, r := range reqs {
		wg.Add(1)
		go func(r abi.EngineRequest) {
			defer wg.Done()
			for range r.Tokens() {
			}
		}(r)
	}
	wg.Wait()
	if peak := e.nativeScheduler().MaxObservedRunning(); peak != 1 {
		t.Fatalf("peak=%d, want 1", peak)
	}
}
