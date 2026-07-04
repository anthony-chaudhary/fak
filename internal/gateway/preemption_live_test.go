package gateway

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

// TestKVPreemptorRendersIntoLiveMetrics proves the #31 metrics are on the actual
// /metrics render path, not only the standalone WriteMetrics fragment.
func TestKVPreemptorRendersIntoLiveMetrics(t *testing.T) {
	srv := newTestServer(t)

	if pre := srv.renderMetrics(); strings.Contains(pre, "fak_sched_preempt_") {
		t.Fatalf("preemption metrics present before SetKVPreemptor:\n%s", pre)
	}

	c := NewKVPreemptor(PreemptionPolicy{Mode: PreemptSwap, Victim: VictimLowestPriority, MaxBlocks: 8, AgingRounds: 1})
	c.Admit(KVSeq{TraceID: "victim", Priority: 9, Blocks: 5, KV: kvOf(20)})
	c.Admit(KVSeq{TraceID: "new", Priority: 0, Blocks: 5, KV: kvOf(1)})
	srv.SetKVPreemptor(c)

	out := srv.renderMetrics()
	for _, want := range []string{
		"fak_sched_preempt_running 1",
		"fak_sched_preempt_used_blocks 5",
		"fak_sched_preempt_swapped_out 1",
		"fak_sched_preempt_total 1",
		"fak_sched_preempt_swap_total 1",
		"fak_sched_preempt_recompute_total 0",
		"fak_sched_preempt_swap_bytes_total 20",
		"fak_sched_preempt_victim_rule 1",
		"# TYPE fak_sched_preempt_running gauge",
		"# TYPE fak_sched_preempt_total counter",
	} {
		if !strings.Contains(out, want+"\n") {
			t.Fatalf("live /metrics surface missing %q\n--- got ---\n%s", want, out)
		}
	}

	srv.SetKVPreemptor(nil)
	if post := srv.renderMetrics(); strings.Contains(post, "fak_sched_preempt_") {
		t.Fatalf("preemption metrics still present after detaching:\n%s", post)
	}
}

func TestNativeSchedulerPreemptionMetricsRenderIntoLiveMetrics(t *testing.T) {
	srv := newTestServer(t)
	sched := modelengine.NewNativeScheduler(model.NewSynthetic(modelengine.SyntheticConfig()))
	t.Cleanup(sched.Close)
	sched.SetKVPreemptionPolicy(modelengine.NativePreemptionPolicy{
		Mode:        modelengine.NativePreemptRecompute,
		MaxBlocks:   3,
		BlockTokens: 4,
	})
	srv.SetKVPreemptionMetrics(sched)

	out := srv.renderMetrics()
	for _, want := range []string{
		"fak_sched_preempt_max_blocks 3",
		"fak_sched_preempt_recompute_total 0",
		"# TYPE fak_sched_preempt_readmitted_total counter",
	} {
		if !strings.Contains(out, want+"\n") {
			t.Fatalf("live native scheduler metrics missing %q\n--- got ---\n%s", want, out)
		}
	}
}

func TestNativeSchedulerCostAwarePreemptionMetricsRenderLiveDecision(t *testing.T) {
	srv := newTestServer(t)
	sched := modelengine.NewNativeScheduler(model.NewSynthetic(modelengine.SyntheticConfig()))
	t.Cleanup(sched.Close)
	sched.SetKVPreemptionPolicy(modelengine.NativePreemptionPolicy{
		Mode:        modelengine.NativePreemptRecompute,
		VictimRule:  modelengine.NativePreemptVictimCostAware,
		MaxBlocks:   2,
		BlockTokens: 128,
	})
	drainNativeSchedulerRequests(t, sched, []*abi.ToolCall{
		kvbmMetricsCall("issue2666_pinned", `{"prompt":"alpha"}`, map[string]string{"kv_pin": "true"}),
		kvbmMetricsCall("issue2666_cold", `{"prompt":"bravo"}`, nil),
		kvbmMetricsCall("issue2666_hot", `{"prompt":"charlie"}`, map[string]string{"kv_reuse_hits": "8"}),
	})
	srv.SetKVPreemptionMetrics(sched)

	out := srv.renderMetrics()
	for _, want := range []string{
		"fak_sched_preempt_victim_rule 2",
		"fak_sched_preempt_cost_aware_total 1",
		"fak_sched_preempt_pinned_skipped_total 1",
		"fak_sched_preempt_last_candidates 3",
		"fak_sched_preempt_last_pinned 1",
		"fak_sched_preempt_last_expired_pins 0",
		"fak_sched_preempt_pin_expired_total 0",
		"fak_sched_preempt_last_victim_cost ",
		"fak_sched_preempt_recompute_total 1",
		"fak_sched_preempt_readmitted_total 1",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("live cost-aware native scheduler metrics missing %q\n--- got ---\n%s", want, out)
		}
	}
}

func kvbmMetricsCall(tool, args string, meta map[string]string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args), Len: int64(len(args))},
		Meta: meta,
	}
}

func drainNativeSchedulerRequests(t *testing.T, sched *modelengine.NativeScheduler, calls []*abi.ToolCall) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	reqs := make([]abi.EngineRequest, len(calls))
	for i, c := range calls {
		r, err := sched.Admit(ctx, c)
		if err != nil {
			t.Fatalf("Admit %d: %v", i, err)
		}
		reqs[i] = r
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
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
		t.Fatalf("timed out draining native scheduler requests: %v", ctx.Err())
	}
	for i, r := range reqs {
		if _, err := r.Result(); err != nil {
			t.Fatalf("Result %d: %v", i, err)
		}
	}
}
