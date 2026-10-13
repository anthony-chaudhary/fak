package modelengine

import (
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestEnginePreemptionMetricWriterCapturesLiveScheduler(t *testing.T) {
	t.Setenv("FAK_NATIVE_KV_MAX_BLOCKS", "0")
	t.Setenv("FAK_MODEL_DIR", "")
	e := New()
	if e.sched != nil {
		t.Fatal("fresh engine already has a scheduler")
	}
	w := e.KVPreemptionMetricWriter()
	s := e.sched
	if s == nil || w.(nativePreemptionMetricWriter).scheduler != s {
		t.Fatal("startup writer did not capture the engine's scheduler")
	}
	t.Cleanup(s.Close)
	var b strings.Builder
	w.WriteKVPreemptionMetrics(&b)
	if b.Len() != 0 {
		t.Fatal("disabled preemption emitted metrics")
	}
	e.SetKVPreemptionPolicy(NativePreemptionPolicy{MaxBlocks: 3, BlockTokens: 128})
	w.WriteKVPreemptionMetrics(&b)
	if !strings.Contains(b.String(), "fak_sched_preempt_max_blocks 3\n") {
		t.Fatal("captured writer did not observe the live policy")
	}
	if e.sched != s {
		t.Fatal("observation replaced the scheduler")
	}
	e.SetKVPreemptionPolicy(NativePreemptionPolicy{})
	b.Reset()
	w.WriteKVPreemptionMetrics(&b)
	if b.Len() != 0 {
		t.Fatal("disarmed preemption retained metrics")
	}
	var absent *Engine
	if absent.KVPreemptionMetricWriter() != nil {
		t.Fatal("nil engine returned a writer")
	}
	w.WriteKVPreemptionMetrics(nil)
	(nativePreemptionMetricWriter{}).WriteKVPreemptionMetrics(&b)
	if b.Len() != 0 {
		t.Fatal("absent scheduler emitted metrics")
	}
}
