package gateway

import (
	"strings"
	"testing"

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
