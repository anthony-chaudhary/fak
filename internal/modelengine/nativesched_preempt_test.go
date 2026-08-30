package modelengine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

// TestNativeSchedulerPreemptionSwapAndRecomputePreserveOutput is the issue-#31
// scheduler witness: under a paged-KV block budget, admitting two lanes exhausts the
// single-block budget, so the loop preempts one lane, keeps making forward progress on
// the survivor, readmits the victim, and produces the same temp-0 token streams as an
// unpreempted run. Both preemption modes are exercised.
func TestNativeSchedulerPreemptionSwapAndRecomputePreserveOutput(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	calls := issue31Calls()
	want := drainIssue31Scheduler(t, m, calls, NativePreemptionPolicy{})

	for _, tc := range []struct {
		name string
		mode NativePreemptionMode
	}{
		{name: "swap", mode: NativePreemptSwap},
		{name: "recompute", mode: NativePreemptRecompute},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, stats := drainIssue31SchedulerWithStats(t, m, calls, NativePreemptionPolicy{
				Mode:        tc.mode,
				MaxBlocks:   1,
				BlockTokens: 128,
			})
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%s preemption changed generated tokens:\n got %v\nwant %v", tc.name, got, want)
			}
			if stats.Preemptions != 1 || stats.Readmitted != 1 || stats.SwappedOut != 0 {
				t.Fatalf("%s stats = %+v, want one preemption, one readmit, no live swapped victims", tc.name, stats)
			}
			if stats.VictimReason != nativePreemptVictimMostRecent {
				t.Fatalf("%s victim reason = %q, want %q", tc.name, stats.VictimReason, nativePreemptVictimMostRecent)
			}
			switch tc.mode {
			case NativePreemptSwap:
				if stats.SwapPreemptions != 1 || stats.SwapBytes == 0 || stats.SwapRestoredBytes != stats.SwapBytes {
					t.Fatalf("swap stats = %+v, want one byte-bearing swap round trip", stats)
				}
			case NativePreemptRecompute:
				if stats.RecomputeCount != 1 || stats.SwapBytes != 0 || stats.SwapRestoredBytes != 0 {
					t.Fatalf("recompute stats = %+v, want one recompute and no swap bytes", stats)
				}
			}
		})
	}
}

// TestNativeSchedulerPreemptsGeneratedLanePreserveOutput covers the non-trivial readmit
// case: the victim has already emitted tokens, so swap must restore the KV after those
// generated tokens and recompute must re-prefill prompt+generated-so-far.
func TestNativeSchedulerPreemptsGeneratedLanePreserveOutput(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	calls := issue31Calls()
	prepare := func(context.Context, *abi.ToolCall, *model.Model) schedPrepare {
		return schedPrepare{prompt: []int{7}}
	}
	drain := func(p NativePreemptionPolicy) ([][]int, NativePreemptionStats) {
		t.Helper()
		s := newNativeScheduler(m, prepare)
		if p.MaxBlocks > 0 {
			s.SetKVPreemptionPolicy(p)
		}
		defer s.Close()
		out := drainIssue31Requests(t, s, calls)
		return out, s.KVPreemptionStats()
	}
	want, _ := drain(NativePreemptionPolicy{})

	for _, tc := range []struct {
		name string
		mode NativePreemptionMode
	}{
		{name: "swap", mode: NativePreemptSwap},
		{name: "recompute", mode: NativePreemptRecompute},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, stats := drain(NativePreemptionPolicy{
				Mode:        tc.mode,
				MaxBlocks:   3,
				BlockTokens: 2,
			})
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%s generated-lane preemption changed tokens:\n got %v\nwant %v", tc.name, got, want)
			}
			if stats.Preemptions == 0 || stats.Readmitted == 0 {
				t.Fatalf("%s stats = %+v, want a generated-lane preemption and readmit", tc.name, stats)
			}
		})
	}
}

// TestNativeSchedulerPreemptionRequiresPagedBlockBudget proves the strict dependency gate:
// without a positive paged-KV block budget, the scheduler does not preempt at all and the
// old co-batched running set is preserved.
func TestNativeSchedulerPreemptionRequiresPagedBlockBudget(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	calls := issue31Calls()
	s := NewNativeScheduler(m)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 0, BlockTokens: 128})
	defer s.Close()

	_ = drainIssue31Requests(t, s, calls)
	if stats := s.KVPreemptionStats(); stats.Preemptions != 0 || stats.SwapPreemptions != 0 || stats.RecomputeCount != 0 {
		t.Fatalf("preemption fired without a paged block budget: %+v", stats)
	}
	if peak := s.MaxObservedRunning(); peak != len(calls) {
		t.Fatalf("unarmed scheduler peak running = %d, want %d (old co-batch path)", peak, len(calls))
	}
}

func TestNativeSchedulerQ4KSlabConfigReachesNewAndRestoredSessions(t *testing.T) {
	t.Setenv("FAK_Q4K_GATEUP_SLAB", "0")
	s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
	s.SetQ4KGateUpOutputSlab(true)

	s.SetResidentQ4K(true)
	fresh := s.newLaneSession(true, NativeSessionFresh)
	defer fresh.Close()
	if !fresh.Q4KGateUpOutputSlab || fresh.Quant || !fresh.Q4K || !fresh.MetalQ4K {
		t.Fatal("explicit scheduler Q4_K/Metal/slab settings did not reach fresh session")
	}

	seed := s.m.NewSession()
	cache := seed.Cache
	seed.Cache = nil
	seed.Close()
	restored := s.sessionFromCache(cache, true)
	defer restored.Close()
	if !restored.Q4KGateUpOutputSlab || restored.Quant || !restored.Q4K || !restored.MetalQ4K {
		t.Fatal("explicit scheduler Q4_K/Metal/slab settings did not reach restored session")
	}
}

func TestNativeSchedulerForcedSwapReadmissionCreatesRestoredSession(t *testing.T) {
	m := nativeSchedulerQwenSwapModel()
	prompts := [][]int{
		{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
		{32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
	}
	control := NewNativeScheduler(m)
	want := drainExactTokenRequests(t, control, prompts)
	if err := control.CloseAndWait(context.Background()); err != nil {
		t.Fatalf("control teardown: %v", err)
	}

	s := NewNativeScheduler(m)
	var lifecycles []NativeSessionLifecycle
	var lifecycleMu sync.Mutex
	s.SetSessionProfilerFactory(func(lifecycle NativeSessionLifecycle) *model.PhaseProfiler {
		lifecycleMu.Lock()
		lifecycles = append(lifecycles, lifecycle)
		lifecycleMu.Unlock()
		return model.NewPhaseProfiler()
	})
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 1, BlockTokens: 16})

	got := drainExactTokenRequests(t, s, prompts)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("forced swap outputs changed:\n got %v\nwant %v", got, want)
	}
	stats := s.KVPreemptionStats()
	if stats.SwapPreemptions == 0 || stats.Readmitted == 0 || stats.SwapBytes == 0 || stats.SwapRestoredBytes != stats.SwapBytes || stats.RecomputeCount != 0 {
		t.Fatalf("forced swap stats=%+v", stats)
	}
	if peak := s.MaxObservedRunning(); peak != 2 {
		t.Fatalf("forced swap peak running=%d, want overlapping two-request admission", peak)
	}
	if err := s.CloseAndWait(context.Background()); err != nil {
		t.Fatalf("forced swap teardown: %v", err)
	}
	lifecycleMu.Lock()
	defer lifecycleMu.Unlock()
	if len(lifecycles) < 3 || lifecycles[0] != NativeSessionFresh || lifecycles[1] != NativeSessionFresh {
		t.Fatalf("session lifecycles=%v, want two fresh then restored", lifecycles)
	}
	restored := 0
	for _, lifecycle := range lifecycles {
		if lifecycle == NativeSessionRestored {
			restored++
		}
	}
	if restored == 0 {
		t.Fatalf("session lifecycles=%v, want observed restored owner", lifecycles)
	}
}

func drainExactTokenRequests(t *testing.T, s *NativeScheduler, prompts [][]int) [][]int {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	reqs := make([]abi.EngineRequest, 0, len(prompts))
	for i, prompt := range prompts {
		req, err := s.AdmitTokenIDs(ctx, fmt.Sprintf("request-%d", i+1), prompt)
		if err != nil {
			t.Fatalf("AdmitTokenIDs %d: %v", i, err)
		}
		reqs = append(reqs, req)
	}
	out := make([][]int, len(reqs))
	var wg sync.WaitGroup
	for i, req := range reqs {
		wg.Add(1)
		go func(i int, req abi.EngineRequest) {
			defer wg.Done()
			for token := range req.Tokens() {
				out[i] = append(out[i], token.ID)
			}
		}(i, req)
	}
	wg.Wait()
	for i, req := range reqs {
		result, err := req.Result()
		if err != nil || result == nil || result.Status != abi.StatusOK {
			t.Fatalf("Result %d=%+v err=%v", i, result, err)
		}
	}
	return out
}

func TestNativeSchedulerPreemptionMetrics(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	s := NewNativeScheduler(m)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{
		Mode:        NativePreemptSwap,
		MaxBlocks:   1,
		BlockTokens: 128,
	})
	defer s.Close()

	_ = drainIssue31Requests(t, s, issue31Calls())
	var b strings.Builder
	s.WriteKVPreemptionMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"# TYPE fak_sched_preempt_running gauge",
		"fak_sched_preempt_running 0",
		"fak_sched_preempt_max_blocks 1",
		"fak_sched_preempt_total 1",
		"fak_sched_preempt_swap_total 1",
		"fak_sched_preempt_recompute_total 0",
		"fak_sched_preempt_swap_bytes_total ",
		"fak_sched_preempt_readmitted_total 1",
		"fak_sched_preempt_victim_rule 0",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("native scheduler metrics missing %q\n--- got ---\n%s", want, out)
		}
	}
}

func TestNativeSchedulerCostAwareVictimUsesReuseHintsAndPins(t *testing.T) {
	s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{
		Mode:        NativePreemptRecompute,
		VictimRule:  NativePreemptVictimCostAware,
		MaxBlocks:   2,
		BlockTokens: 10,
	})
	s.lanes = []*schedLane{
		nativePreemptTestLane("pinned", 1, 10, 0, true),
		nativePreemptTestLane("cold", 2, 10, 0, false),
		nativePreemptTestLane("hot", 3, 10, 8, false),
	}
	defer func() {
		for _, ln := range s.lanes {
			ln.cancel()
		}
		for _, ln := range s.preempted {
			ln.cancel()
		}
	}()

	if idx := s.mostRecentPreemptibleLaneLocked(); idx != 2 {
		t.Fatalf("fixture sanity: most-recent victim = %d, want 2 (hot/newest)", idx)
	}
	idx := s.preemptibleLaneLocked()
	if idx != 1 {
		t.Fatalf("cost-aware victim = %d (%s), want 1 (cold, unpinned, lowest reuse)", idx, s.lanes[idx].tool)
	}
	stats := s.preemptStats
	if stats.LastCandidates != 3 || stats.LastPinned != 1 || stats.PinnedSkipped != 1 {
		t.Fatalf("cost-aware selection observability = %+v, want 3 candidates / 1 pinned skip", stats)
	}
	if stats.LastVictimHits != 0 || stats.LastVictimTokens != 10 || stats.LastVictimBlocks != 1 || stats.LastVictimCost != 1 {
		t.Fatalf("last victim stats = %+v, want cold 10-token one-block cost=1", stats)
	}
}

func TestNativeSchedulerCostAwarePreemptionMetrics(t *testing.T) {
	s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{
		Mode:        NativePreemptRecompute,
		VictimRule:  NativePreemptVictimCostAware,
		MaxBlocks:   2,
		BlockTokens: 10,
	})
	s.lanes = []*schedLane{
		nativePreemptTestLane("pinned", 1, 10, 0, true),
		nativePreemptTestLane("cold", 2, 10, 0, false),
		nativePreemptTestLane("hot", 3, 10, 8, false),
	}
	defer func() {
		for _, ln := range s.lanes {
			ln.cancel()
		}
		for _, ln := range s.preempted {
			ln.cancel()
		}
	}()

	s.enforcePreemptionLocked()
	if len(s.preempted) != 1 || s.preempted[0].tool != "cold" {
		t.Fatalf("preempted lanes = %+v, want only cold", preemptedTools(s.preempted))
	}
	if got := runningTools(s.lanes); !reflect.DeepEqual(got, []string{"pinned", "hot"}) {
		t.Fatalf("running lanes after cost-aware preemption = %v, want pinned+hot", got)
	}
	stats := s.KVPreemptionStats()
	if stats.CostAwareVictims != 1 || stats.RecomputeCount != 1 || stats.VictimRule != NativePreemptVictimCostAware {
		t.Fatalf("cost-aware preemption stats = %+v, want one cost-aware recompute", stats)
	}
	var b strings.Builder
	s.WriteKVPreemptionMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"fak_sched_preempt_victim_rule 2",
		"fak_sched_preempt_cost_aware_total 1",
		"fak_sched_preempt_pinned_skipped_total 1",
		"fak_sched_preempt_last_candidates 3",
		"fak_sched_preempt_last_pinned 1",
		"fak_sched_preempt_last_expired_pins 0",
		"fak_sched_preempt_pin_expired_total 0",
		"fak_sched_preempt_last_victim_cost 1",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("cost-aware metrics missing %q\n--- got ---\n%s", want, out)
		}
	}
}

func TestNativeSchedulerCostAwarePinTTLExpires(t *testing.T) {
	s := NewNativeScheduler(model.NewSynthetic(SyntheticConfig()))
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{
		Mode:        NativePreemptRecompute,
		VictimRule:  NativePreemptVictimCostAware,
		MaxBlocks:   2,
		BlockTokens: 10,
	})
	expiredPin := nativePreemptTestLane("expired-pin", 1, 10, 0, true)
	expiredPin.kvPinUntil = time.Now().Add(-time.Millisecond)
	s.lanes = []*schedLane{
		expiredPin,
		nativePreemptTestLane("cold", 2, 10, 0, false),
		nativePreemptTestLane("hot", 3, 10, 8, false),
	}
	defer func() {
		for _, ln := range s.lanes {
			ln.cancel()
		}
		for _, ln := range s.preempted {
			ln.cancel()
		}
	}()

	idx := s.preemptibleLaneLocked()
	if idx != 0 {
		got := "<none>"
		if idx >= 0 && idx < len(s.lanes) {
			got = s.lanes[idx].tool
		}
		t.Fatalf("expired pin victim = %d (%s), want expired-pin to become evictable", idx, got)
	}
	stats := s.preemptStats
	if stats.LastPinned != 0 || stats.PinnedSkipped != 0 || stats.LastExpiredPins != 1 || stats.ExpiredPins != 1 {
		t.Fatalf("expired pin observability = %+v, want expired=1 and active pinned=0", stats)
	}
	var b strings.Builder
	s.WriteKVPreemptionMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"fak_sched_preempt_last_expired_pins 1",
		"fak_sched_preempt_pin_expired_total 1",
		"fak_sched_preempt_last_pinned 0",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("expired pin metrics missing %q\n--- got ---\n%s", want, out)
		}
	}
}

// TestNativeSchedulerCostAwarePreemptionPreservesOutputAndPins is the live #2666 R2
// witness: the cost-aware picker runs inside the real scheduler under KV pressure, skips a
// pinned lane, chooses the cold one-shot lane instead of the newer hot lane, readmits the
// victim, and still produces the same generated streams as the unpreempted scheduler.
func TestNativeSchedulerCostAwarePreemptionPreservesOutputAndPins(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	calls := issue2666KVBMCalls()
	want := drainIssue31Scheduler(t, m, calls, NativePreemptionPolicy{})

	for _, tc := range []struct {
		name string
		mode NativePreemptionMode
	}{
		{name: "swap", mode: NativePreemptSwap},
		{name: "recompute", mode: NativePreemptRecompute},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, stats := drainIssue31SchedulerWithStats(t, m, calls, NativePreemptionPolicy{
				Mode:        tc.mode,
				VictimRule:  NativePreemptVictimCostAware,
				MaxBlocks:   2,
				BlockTokens: 128,
			})
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("%s cost-aware preemption changed generated tokens:\n got %v\nwant %v", tc.name, got, want)
			}
			if stats.Preemptions != 1 || stats.CostAwareVictims != 1 || stats.Readmitted != 1 {
				t.Fatalf("%s cost-aware stats = %+v, want one preemption, cost-aware victim, and readmit", tc.name, stats)
			}
			if stats.LastCandidates != 3 || stats.LastPinned != 1 || stats.PinnedSkipped != 1 {
				t.Fatalf("%s cost-aware candidate stats = %+v, want 3 candidates / 1 pinned skip", tc.name, stats)
			}
			if stats.LastVictimHits != 0 || stats.LastVictimBlocks != 1 || stats.LastVictimTokens <= 0 || stats.LastVictimCost <= 0 {
				t.Fatalf("%s last victim stats = %+v, want cold one-block victim with zero reuse and positive cost", tc.name, stats)
			}
			switch tc.mode {
			case NativePreemptSwap:
				if stats.SwapPreemptions != 1 || stats.SwapBytes == 0 || stats.SwapRestoredBytes != stats.SwapBytes {
					t.Fatalf("swap cost-aware stats = %+v, want byte-bearing swap round trip", stats)
				}
			case NativePreemptRecompute:
				if stats.RecomputeCount != 1 || stats.SwapBytes != 0 || stats.SwapRestoredBytes != 0 {
					t.Fatalf("recompute cost-aware stats = %+v, want recompute without swap bytes", stats)
				}
			}
		})
	}
}

func TestNativeSchedulerReadmitsOversizeVictimWhenAlone(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	calls := issue31Calls()
	prepare := func(context.Context, *abi.ToolCall, *model.Model) schedPrepare {
		return schedPrepare{prompt: []int{1, 2, 3, 4, 5}}
	}
	ref := newNativeScheduler(m, prepare)
	want := drainIssue31Requests(t, ref, calls)
	ref.Close()

	s := newNativeScheduler(m, prepare)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{
		Mode:        NativePreemptRecompute,
		MaxBlocks:   1,
		BlockTokens: 2,
	})
	defer s.Close()

	got := drainIssue31Requests(t, s, calls)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("oversize-victim readmit changed tokens:\n got %v\nwant %v", got, want)
	}
	stats := s.KVPreemptionStats()
	if stats.Preemptions == 0 || stats.Readmitted == 0 || stats.SwappedOut != 0 {
		t.Fatalf("oversize-victim stats = %+v, want preempt/readmit/no swapped victims", stats)
	}
}

func TestNativePreemptionPolicyFromEnv(t *testing.T) {
	t.Setenv("FAK_NATIVE_KV_MAX_BLOCKS", "7")
	t.Setenv("FAK_NATIVE_KV_BLOCK_TOKENS", "4")
	t.Setenv("FAK_NATIVE_KV_PREEMPT_MODE", "recompute")
	t.Setenv("FAK_NATIVE_KV_VICTIM_RULE", "cost-aware")

	p := nativePreemptionPolicyFromEnv()
	if p.Mode != NativePreemptRecompute || p.VictimRule != NativePreemptVictimCostAware || p.MaxBlocks != 7 || p.BlockTokens != 4 {
		t.Fatalf("native preemption policy from env = %+v, want recompute max=7 block=4", p)
	}
}

func TestNativeKVBMHintsFromMeta(t *testing.T) {
	now := time.Unix(100, 0)
	hits, pinned, until := nativeKVBMHintsFromMetaAt(map[string]string{
		"kv.reuse_hits":         "4",
		"kv.preempt_pin":        "true",
		"kv.preempt_pin_ttl_ms": "1500",
	}, now)
	if hits != 4 || !pinned || until.Sub(now) != 1500*time.Millisecond {
		t.Fatalf("dot-form hints = hits %d pinned %v ttl %s, want 4/true/1500ms", hits, pinned, until.Sub(now))
	}
	hits, pinned, until = nativeKVBMHintsFromMetaAt(map[string]string{
		"kv.reuse_hits":  "4",
		"kv.preempt_pin": "true",
	}, now)
	if hits != 4 || !pinned || until.Sub(now) != nativeKVBMDefaultPinTTL {
		t.Fatalf("default pin ttl = hits %d pinned %v ttl %s, want 4/true/%s",
			hits, pinned, until.Sub(now), nativeKVBMDefaultPinTTL)
	}
	hits, pinned, until = nativeKVBMHintsFromMetaAt(map[string]string{
		"kv_reuse_hits": "-4",
		"kv_pin":        "false",
	}, now)
	if hits != 0 || pinned || !until.IsZero() {
		t.Fatalf("defensive hints = hits %d pinned %v until %v, want 0/false/zero", hits, pinned, until)
	}
}

func TestNativeKVBMHintsFromMetaAssignsDefaultPinTTL(t *testing.T) {
	hits, pinned, until := nativeKVBMHintsFromMeta(map[string]string{
		"kv.reuse_hits": "4",
		"kv.pin":        "true",
	})
	if hits != 4 || !pinned || until.IsZero() {
		t.Fatalf("native hints = hits %d pinned %v until %v, want pinned with deadline", hits, pinned, until)
	}
}

func nativePreemptTestLane(tool string, seq int64, tokens, hits int, pinned bool) *schedLane {
	ctx, cancel := context.WithCancel(context.Background())
	var pinUntil time.Time
	if pinned {
		pinUntil = time.Now().Add(nativeKVBMDefaultPinTTL)
	}
	return &schedLane{
		ctx:         ctx,
		cancel:      cancel,
		sess:        &model.Session{},
		tool:        tool,
		promptLen:   tokens,
		seqNo:       seq,
		kvReuseHits: hits,
		kvPinned:    pinned,
		kvPinUntil:  pinUntil,
		done:        make(chan struct{}),
		tokens:      make(chan abi.EngineToken),
	}
}

func runningTools(lanes []*schedLane) []string {
	out := make([]string, 0, len(lanes))
	for _, ln := range lanes {
		out = append(out, ln.tool)
	}
	return out
}

func preemptedTools(lanes []*schedLane) []string {
	return runningTools(lanes)
}

func issue31Calls() []*abi.ToolCall {
	return []*abi.ToolCall{
		inlineCall("issue31_first", `{"prompt":"alpha"}`),
		inlineCall("issue31_second", `{"prompt":"bravo"}`),
	}
}

func issue2666KVBMCalls() []*abi.ToolCall {
	pinned := inlineCall("issue2666_pinned", `{"prompt":"alpha"}`)
	pinned.Meta = map[string]string{"kv_pin": "true"}
	cold := inlineCall("issue2666_cold", `{"prompt":"bravo"}`)
	hot := inlineCall("issue2666_hot", `{"prompt":"charlie"}`)
	hot.Meta = map[string]string{"kv_reuse_hits": "8"}
	return []*abi.ToolCall{pinned, cold, hot}
}

func drainIssue31Scheduler(t *testing.T, m *model.Model, calls []*abi.ToolCall, p NativePreemptionPolicy) [][]int {
	t.Helper()
	got, _ := drainIssue31SchedulerWithStats(t, m, calls, p)
	return got
}

func drainIssue31SchedulerWithStats(t *testing.T, m *model.Model, calls []*abi.ToolCall, p NativePreemptionPolicy) ([][]int, NativePreemptionStats) {
	t.Helper()
	s := NewNativeScheduler(m)
	if p.MaxBlocks > 0 {
		s.SetKVPreemptionPolicy(p)
	}
	defer s.Close()
	out := drainIssue31Requests(t, s, calls)
	return out, s.KVPreemptionStats()
}

func drainIssue31Requests(t *testing.T, s *NativeScheduler, calls []*abi.ToolCall) [][]int {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	reqs := make([]abi.EngineRequest, len(calls))
	for i, c := range calls {
		r, err := s.Admit(ctx, c)
		if err != nil {
			t.Fatalf("Admit %d: %v", i, err)
		}
		reqs[i] = r
	}

	out := make([][]int, len(reqs))
	var wg sync.WaitGroup
	for i, r := range reqs {
		wg.Add(1)
		go func(i int, r abi.EngineRequest) {
			defer wg.Done()
			for tok := range r.Tokens() {
				out[i] = append(out[i], tok.ID)
			}
		}(i, r)
	}
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatalf("timed out draining scheduler requests: %v", ctx.Err())
	}
	for i, r := range reqs {
		res, err := r.Result()
		if err != nil {
			t.Fatalf("Result %d: %v", i, err)
		}
		if res == nil || res.Status != abi.StatusOK {
			t.Fatalf("Result %d = %+v, want StatusOK", i, res)
		}
	}
	return out
}

func TestNativeSchedulerQwenSwapPreemptionResumes(t *testing.T) {
	m := nativeSchedulerQwenSwapModel()
	calls := issue31Calls()
	want := drainIssue31Scheduler(t, m, calls, NativePreemptionPolicy{})
	ledger := t.TempDir() + "/qwen-swap.jsonl"
	got, stats := drainIssue31SchedulerWithStats(t, m, calls, NativePreemptionPolicy{
		MaxBlocks:       1,
		BlockTokens:     16,
		Mode:            NativePreemptSwap,
		UsageLedgerPath: ledger,
	})
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Qwen swap output = %v, want %v", got, want)
	}
	if stats.SwapPreemptions == 0 || stats.Readmitted == 0 || stats.SwapBytes == 0 || stats.SwapRestoredBytes == 0 {
		t.Fatalf("Qwen swap stats = %+v, want nonzero swap/readmission counters", stats)
	}
	rows := readQwenSwapUsageRows(t, ledger)
	if len(rows) != 2 || rows[0].Direction != modelperfobs.QwenSwapDirectionOut || rows[1].Direction != modelperfobs.QwenSwapDirectionIn {
		t.Fatalf("Qwen swap production rows = %+v, want swap-out then restore-in", rows)
	}
	for i, row := range rows {
		if row.Version != modelperfobs.QwenSwapCodecVersion || row.Outcome != modelperfobs.QwenSwapOutcomeSuccess || row.Result != modelperfobs.QwenSwapResultCommitted || row.Bytes <= 0 {
			t.Fatalf("Qwen swap production row %d = %+v, want byte-bearing v1 committed success", i, row)
		}
	}
	if rows[0].Bytes != stats.SwapBytes || rows[1].Bytes != stats.SwapRestoredBytes {
		t.Fatalf("Qwen swap row bytes out/in=%d/%d, want encoded output/input=%d/%d", rows[0].Bytes, rows[1].Bytes, stats.SwapBytes, stats.SwapRestoredBytes)
	}
	fold, err := modelperfobs.FoldQwenSwapUsage(ledger)
	if err != nil {
		t.Fatal(err)
	}
	if len(fold) != 1 || fold[0].Invocations != 2 || fold[0].SwapOut != 1 || fold[0].RestoreIn != 1 || fold[0].Succeeded != 2 || fold[0].Refused != 0 || fold[0].Errors != 0 {
		t.Fatalf("Qwen swap production fold = %+v, want two committed invocations", fold)
	}
	t.Logf("Qwen swap production rows=%+v fold=%+v", rows, fold)
}

func TestNativeSchedulerQwenSwapUsageAppendFailureIsObservational(t *testing.T) {
	calls := issue31Calls()
	policy := NativePreemptionPolicy{MaxBlocks: 1, BlockTokens: 16, Mode: NativePreemptSwap}
	want, wantStats := drainIssue31SchedulerWithStats(t, nativeSchedulerQwenSwapModel(), calls, policy)

	dir := t.TempDir()
	blocker := dir + "/not-a-directory"
	if err := os.WriteFile(blocker, []byte("block ledger parent"), 0o600); err != nil {
		t.Fatal(err)
	}
	policy.UsageLedgerPath = blocker + "/qwen-swap.jsonl"
	got, gotStats := drainIssue31SchedulerWithStats(t, nativeSchedulerQwenSwapModel(), calls, policy)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unwritable ledger changed scheduler output:\n got %v\nwant %v", got, want)
	}
	if !reflect.DeepEqual(gotStats, wantStats) {
		t.Fatalf("unwritable ledger changed scheduler lifecycle/stats:\n got %+v\nwant %+v", gotStats, wantStats)
	}
	if gotStats.SwapPreemptions == 0 || gotStats.Readmitted == 0 || gotStats.SwappedOut != 0 || gotStats.SwapBytes == 0 || gotStats.SwapRestoredBytes != gotStats.SwapBytes {
		t.Fatalf("unwritable ledger scheduler receipt = %+v, want completed byte-exact swap/readmit", gotStats)
	}
	if _, err := os.Stat(policy.UsageLedgerPath); err == nil {
		t.Fatal("unwritable ledger unexpectedly exists")
	}

	s := NewNativeScheduler(nativeSchedulerQwenSwapModel())
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 8, BlockTokens: 4, UsageLedgerPath: policy.UsageLedgerPath})
	ln := nativeSchedulerQwenReadmitLane(t, s, []int{3, 7, 11, 5}, 2)
	defer ln.cancel()
	if err := s.preemptLaneLocked(ln); err != nil {
		t.Fatalf("unwritable-ledger swap preempt: %v", err)
	}
	ln.hostKV[0] ^= 0xff
	s.readmitPreemptedLocked()
	if ln.err == nil || !strings.Contains(ln.err.Error(), "checksum mismatch") || strings.Contains(ln.err.Error(), "record Qwen swap usage") {
		t.Fatalf("unwritable ledger changed primary codec error: %v", ln.err)
	}
}

func TestNativeSchedulerQwenSwapReadmitRestoresTokenLineageAndContinuation(t *testing.T) {
	m := nativeSchedulerQwenSwapModel()
	s := NewNativeScheduler(m)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 8, BlockTokens: 4})
	ln := nativeSchedulerQwenReadmitLane(t, s, []int{3, 7, 11, 5}, 2)
	defer ln.cancel()
	history := append(append([]int(nil), ln.prompt...), ln.gen...)
	wantLogits := copyF32(ln.logits)

	control := m.NewSession()
	defer control.Close()
	controlLogits := control.Prefill(history)
	if !reflect.DeepEqual(wantLogits, controlLogits) {
		t.Fatal("fixture logits differ before preemption")
	}
	if err := s.preemptLaneLocked(ln); err != nil {
		t.Fatalf("swap preempt: %v", err)
	}
	if ln.sess != nil || len(s.preempted) != 1 {
		t.Fatalf("preempted lane session=%p victims=%d, want nil/1", ln.sess, len(s.preempted))
	}
	s.readmitPreemptedLocked()
	if ln.sess == nil || len(s.lanes) != 1 || len(s.preempted) != 0 {
		t.Fatalf("readmit session=%p running=%d victims=%d, want live/1/0", ln.sess, len(s.lanes), len(s.preempted))
	}
	defer ln.sess.Close()
	if !reflect.DeepEqual(ln.logits, wantLogits) {
		t.Fatal("readmit did not preserve saved logits exactly")
	}
	if _, err := ln.sess.VerifyTokenLineage(history); err != nil {
		t.Fatalf("readmit lineage: %v", err)
	}

	for step := 0; step < 3; step++ {
		gotToken, wantToken := argmax(ln.logits), argmax(controlLogits)
		if gotToken != wantToken {
			t.Fatalf("continuation step %d token=%d, want %d", step, gotToken, wantToken)
		}
		history = append(history, gotToken)
		ln.logits = ln.sess.Step(gotToken)
		controlLogits = control.Step(wantToken)
		if !reflect.DeepEqual(ln.logits, controlLogits) {
			t.Fatalf("continuation step %d logits differ", step)
		}
		if _, err := ln.sess.VerifyTokenLineage(history); err != nil {
			t.Fatalf("continuation step %d lineage: %v", step, err)
		}
	}
	if stats := s.KVPreemptionStats(); stats.Readmitted != 1 || stats.SwapRestoredBytes == 0 || stats.SwapRestoredBytes != stats.SwapBytes {
		t.Fatalf("readmit stats=%+v, want one byte-bearing readmit", stats)
	}
}

func TestNativeSchedulerQwenSwapReadmitLineageMismatchDoesNotPublishSession(t *testing.T) {
	m := nativeSchedulerQwenSwapModel()
	s := NewNativeScheduler(m)
	ledger := t.TempDir() + "/qwen-swap.jsonl"
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 8, BlockTokens: 4, UsageLedgerPath: ledger})
	ln := nativeSchedulerQwenReadmitLane(t, s, []int{3, 7, 11, 5}, 2)
	defer ln.cancel()
	if err := s.preemptLaneLocked(ln); err != nil {
		t.Fatalf("swap preempt: %v", err)
	}
	ln.gen = ln.gen[:len(ln.gen)-1]
	s.readmitPreemptedLocked()

	if !ln.terminal || !ln.reclaimed || ln.sess != nil || len(s.lanes) != 0 || len(s.preempted) != 0 {
		t.Fatalf("refused readmit terminal=%t reclaimed=%t session=%p running=%d victims=%d", ln.terminal, ln.reclaimed, ln.sess, len(s.lanes), len(s.preempted))
	}
	if !errors.Is(ln.err, model.ErrTokenLineageMismatch) {
		t.Fatalf("refused readmit error=%v, want ErrTokenLineageMismatch", ln.err)
	}
	if stats := s.KVPreemptionStats(); stats.Readmitted != 0 || stats.SwapRestoredBytes != 0 {
		t.Fatalf("refused readmit stats=%+v, want zero readmit/restored-byte credit", stats)
	}
	if ln.hostKV != nil || ln.savedLogits != nil {
		t.Fatalf("refused readmit retained host state: host_bytes=%d saved_logits=%d", len(ln.hostKV), len(ln.savedLogits))
	}
	rows := readQwenSwapUsageRows(t, ledger)
	if len(rows) != 2 || rows[1].Outcome != modelperfobs.QwenSwapOutcomeSuccess || rows[1].Result != modelperfobs.QwenSwapResultRefused {
		t.Fatalf("lineage-refused usage rows = %+v, want successful decode with refused publication", rows)
	}
}

func TestNativeSchedulerQwenSwapDecodeFailureDoesNotClaimSuccess(t *testing.T) {
	m := nativeSchedulerQwenSwapModel()
	s := NewNativeScheduler(m)
	ledger := t.TempDir() + "/qwen-swap.jsonl"
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 8, BlockTokens: 4, UsageLedgerPath: ledger})
	ln := nativeSchedulerQwenReadmitLane(t, s, []int{3, 7, 11, 5}, 2)
	defer ln.cancel()
	if err := s.preemptLaneLocked(ln); err != nil {
		t.Fatalf("swap preempt: %v", err)
	}
	ln.hostKV[0] ^= 0xff
	s.readmitPreemptedLocked()
	if !ln.terminal || ln.sess != nil {
		t.Fatalf("failed decode terminal=%t session=%p, want terminal/nil", ln.terminal, ln.sess)
	}
	rows := readQwenSwapUsageRows(t, ledger)
	if len(rows) != 2 || rows[1].Outcome != modelperfobs.QwenSwapOutcomeError || rows[1].Result != modelperfobs.QwenSwapResultRefused || rows[1].Bytes == 0 {
		t.Fatalf("failed decode usage rows = %+v, want byte-bearing error/refused row", rows)
	}
}

func TestNativeSchedulerQwenSwapReadmitLongHistoryDoesNotPublishSession(t *testing.T) {
	m := nativeSchedulerQwenSwapModel()
	s := NewNativeScheduler(m)
	s.SetKVPreemptionPolicy(NativePreemptionPolicy{Mode: NativePreemptSwap, MaxBlocks: 8, BlockTokens: 4})
	ln := nativeSchedulerQwenReadmitLane(t, s, []int{3, 7, 11, 5}, 2)
	defer ln.cancel()
	if err := s.preemptLaneLocked(ln); err != nil {
		t.Fatalf("swap preempt: %v", err)
	}
	ln.gen = append(ln.gen, 34)
	s.readmitPreemptedLocked()

	if !ln.terminal || !ln.reclaimed || ln.sess != nil || len(s.lanes) != 0 || len(s.preempted) != 0 {
		t.Fatalf("refused readmit terminal=%t reclaimed=%t session=%p running=%d victims=%d", ln.terminal, ln.reclaimed, ln.sess, len(s.lanes), len(s.preempted))
	}
	if !errors.Is(ln.err, model.ErrTokenLineageMismatch) {
		t.Fatalf("refused readmit error=%v, want ErrTokenLineageMismatch", ln.err)
	}
	if stats := s.KVPreemptionStats(); stats.Readmitted != 0 || stats.SwapRestoredBytes != 0 {
		t.Fatalf("refused readmit stats=%+v, want zero readmit/restored-byte credit", stats)
	}
	if ln.hostKV != nil || ln.savedLogits != nil {
		t.Fatalf("refused readmit retained host state: host_bytes=%d saved_logits=%d", len(ln.hostKV), len(ln.savedLogits))
	}
}

func nativeSchedulerQwenSwapModel() *model.Model {
	cfg := SyntheticConfig()
	cfg.NumLayers = 4
	cfg.LayerTypes = []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"}
	cfg.FullAttentionInterval = 4
	cfg.LinearConvKernelDim = 3
	cfg.LinearKeyHeadDim = cfg.HeadDim
	cfg.LinearValueHeadDim = cfg.HeadDim
	cfg.LinearNumKeyHeads = cfg.NumKVHeads
	cfg.LinearNumValueHeads = cfg.NumHeads
	cfg.AttnOutputGate = true
	cfg.NormGain1p = true
	return model.NewSynthetic(cfg)
}

func nativeSchedulerQwenReadmitLane(t *testing.T, s *NativeScheduler, prompt []int, generated int) *schedLane {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	sess := s.m.NewSession()
	logits := sess.Prefill(prompt)
	gen := make([]int, 0, generated)
	for range generated {
		next := argmax(logits)
		gen = append(gen, next)
		logits = sess.Step(next)
	}
	return &schedLane{
		sched: s, ctx: ctx, cancel: cancel, sess: sess, logits: logits,
		prompt: append([]int(nil), prompt...), promptLen: len(prompt), gen: gen, emitted: len(gen),
		tokens: make(chan abi.EngineToken, 1), done: make(chan struct{}),
	}
}

func readQwenSwapUsageRows(t *testing.T, path string) []modelperfobs.QwenSwapUsageRow {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var rows []modelperfobs.QwenSwapUsageRow
	for i, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var row modelperfobs.QwenSwapUsageRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			t.Fatalf("usage row %d: %v", i+1, err)
		}
		rows = append(rows, row)
	}
	return rows
}
