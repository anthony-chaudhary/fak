package modelengine

import (
	"context"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
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
		"fak_sched_preempt_last_victim_cost 1",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("cost-aware metrics missing %q\n--- got ---\n%s", want, out)
		}
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
	hits, pinned := nativeKVBMHintsFromMeta(map[string]string{
		"kv.reuse_hits":  "4",
		"kv.preempt_pin": "true",
	})
	if hits != 4 || !pinned {
		t.Fatalf("dot-form hints = hits %d pinned %v, want 4/true", hits, pinned)
	}
	hits, pinned = nativeKVBMHintsFromMeta(map[string]string{
		"kv_reuse_hits": "-4",
		"kv_pin":        "false",
	})
	if hits != 0 || pinned {
		t.Fatalf("defensive hints = hits %d pinned %v, want 0/false", hits, pinned)
	}
}

func nativePreemptTestLane(tool string, seq int64, tokens, hits int, pinned bool) *schedLane {
	ctx, cancel := context.WithCancel(context.Background())
	return &schedLane{
		ctx:         ctx,
		cancel:      cancel,
		sess:        &model.Session{},
		tool:        tool,
		promptLen:   tokens,
		seqNo:       seq,
		kvReuseHits: hits,
		kvPinned:    pinned,
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
