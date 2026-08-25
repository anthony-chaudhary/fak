package model

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestQwen35DecodeGraphTraceTokenHiddenQLifecycleExactlyOnce(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	cfg.NumLayers = 2
	cfg.LayerTypes = []string{"linear_attention", "full_attention"}
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	if len(m.q4kw) == 0 {
		t.Fatal("q4k witness has no resident q4k weights")
	}

	direct := m.NewSession()
	direct.Q4K = true
	direct.EnableQwen35DecodeGraphTrace()
	_ = direct.tokenHiddenQ(1, 7)
	assertQwen35GraphLifecycle(t, direct, 1, 1, false)
	directTrace := direct.LastQwen35DecodeGraphTrace()
	assertQwen35GraphTrace(t, directTrace, 7)

	delegated := m.NewSession()
	delegated.Quant = true
	delegated.Q4K = true
	delegated.EnableQwen35DecodeGraphTrace()
	_ = delegated.tokenHidden(2, 9)
	assertQwen35GraphLifecycle(t, delegated, 1, 1, false)
	delegatedTrace := delegated.LastQwen35DecodeGraphTrace()
	assertQwen35GraphTrace(t, delegatedTrace, 9)

	// Returned traces own their node and dependency storage, across sessions and snapshots.
	directTrace.Nodes[0].Operation = "mutated"
	directTrace.Nodes[0].DependsOn[0] = "mutated"
	again := direct.LastQwen35DecodeGraphTrace()
	if again.Nodes[0].Operation != "linear-attention" || again.Nodes[0].DependsOn[0] != "layer-input" {
		t.Fatalf("snapshot aliases recorder: %+v", again.Nodes[0])
	}
}

func TestQwen35DecodeGraphTraceAbortCleansActiveOwner(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	cfg.NumLayers = 1
	cfg.LayerTypes = []string{"linear_attention"}
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	s := m.NewSession()
	s.Q4K = true
	s.EnableQwen35DecodeGraphTrace()

	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("tokenHiddenQ did not panic for an out-of-range token")
			}
		}()
		_ = s.tokenHiddenQ(cfg.VocabSize, 3)
	}()
	assertQwen35GraphLifecycle(t, s, 1, 1, false)
	if tr := s.LastQwen35DecodeGraphTrace(); !tr.Aborted || tr.Position != 3 {
		t.Fatalf("aborted trace=%+v", tr)
	}

	_ = s.tokenHiddenQ(1, 4)
	assertQwen35GraphLifecycle(t, s, 2, 2, false)
	if tr := s.LastQwen35DecodeGraphTrace(); tr.Aborted || tr.Position != 4 || len(tr.Nodes) != 4 {
		t.Fatalf("recovery trace leaked aborted owner: %+v", tr)
	}
}

func TestQwen35DecodeGraphTraceDoesNotOverwriteActiveOwner(t *testing.T) {
	m := NewSynthetic(qwen35HybridQ4KTestCfg())
	s := m.NewSession()
	s.EnableQwen35DecodeGraphTrace()

	firstFinish := s.beginQwen35DecodeGraph(11)
	s.recordQwen35DecodeGraph(Qwen35DecodeGraphNode{Operation: "first-owner"})
	secondFinish := s.beginQwen35DecodeGraph(12)
	secondFinish(false)
	assertQwen35GraphLifecycleAt(t, s, 1, 0, true, 11)

	firstFinish(false)
	firstFinish(true)
	assertQwen35GraphLifecycle(t, s, 1, 1, false)
	if tr := s.LastQwen35DecodeGraphTrace(); tr.Position != 11 || tr.Aborted || len(tr.Nodes) != 1 {
		t.Fatalf("first owner was overwritten: %+v", tr)
	}

	thirdFinish := s.beginQwen35DecodeGraph(13)
	firstFinish(true) // A stale finish cannot close the new owner.
	assertQwen35GraphLifecycleAt(t, s, 2, 1, true, 13)
	thirdFinish(false)
	assertQwen35GraphLifecycle(t, s, 2, 2, false)
}

func TestQwen35DecodeGraphTraceNativeMetadataUnavailable(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	cfg.NumLayers = 1
	cfg.LayerTypes = []string{"linear_attention"}
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	s := m.NewSession()
	s.Q4K = true
	s.EnableQwen35DecodeGraphTrace()
	_ = s.tokenHiddenQ(1, 5)

	for i, node := range s.LastQwen35DecodeGraphTrace().Nodes {
		if node.NativeAttributionStatus != Qwen35DecodeGraphNativeAttributionUnavailable {
			t.Fatalf("node[%d] native status=%q", i, node.NativeAttributionStatus)
		}
		if node.UnavailableReason != Qwen35DecodeGraphMetalEventSourceUnavailable {
			t.Fatalf("node[%d] unavailable reason=%q", i, node.UnavailableReason)
		}
		if node.HostRead != nil || node.Syncs != nil {
			t.Fatalf("node[%d] fabricated native observations: host_read=%v syncs=%v", i, node.HostRead, node.Syncs)
		}
		encoded, err := json.Marshal(node)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(encoded), `"host_read"`) || strings.Contains(string(encoded), `"syncs"`) {
			t.Fatalf("node[%d] serialized unavailable observations: %s", i, encoded)
		}
	}

	// An observed false/zero remains representable and serializes differently from unavailable.
	hostRead, syncs := false, 0
	observed := Qwen35DecodeGraphNode{
		NativeAttributionStatus: Qwen35DecodeGraphNativeAttributionAvailable,
		HostRead:                &hostRead,
		Syncs:                   &syncs,
	}
	encoded, err := json.Marshal(observed)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"host_read":false`) || !strings.Contains(string(encoded), `"syncs":0`) {
		t.Fatalf("observed zero values were lost: %s", encoded)
	}
}

func assertQwen35GraphTrace(t *testing.T, trace Qwen35DecodeGraphTrace, position int) {
	t.Helper()
	if trace.Position != position || trace.Aborted {
		t.Fatalf("trace=%+v", trace)
	}
	wantOperations := []string{
		"linear-attention", "post-attention-residual-norm", "mlp", "next-layer-handoff",
		"full-attention", "post-attention-residual-norm", "mlp", "next-layer-handoff",
	}
	wantDependencies := []string{
		"layer-input", "linear-attention", "post-attention-residual-norm", "mlp",
		"layer-input", "full-attention", "post-attention-residual-norm", "mlp",
	}
	if len(trace.Nodes) != len(wantOperations) {
		t.Fatalf("nodes=%d want %d: %+v", len(trace.Nodes), len(wantOperations), trace.Nodes)
	}
	for i := range wantOperations {
		if trace.Nodes[i].Operation != wantOperations[i] {
			t.Fatalf("node[%d]=%q want %q", i, trace.Nodes[i].Operation, wantOperations[i])
		}
		if len(trace.Nodes[i].DependsOn) != 1 || trace.Nodes[i].DependsOn[0] != wantDependencies[i] {
			t.Fatalf("node[%d] dependencies=%v want [%q]", i, trace.Nodes[i].DependsOn, wantDependencies[i])
		}
		if strings.Contains(trace.Nodes[i].Operation, "fused") {
			t.Fatalf("node[%d] falsely labels an unfused model operation: %q", i, trace.Nodes[i].Operation)
		}
	}
}

func assertQwen35GraphLifecycle(t *testing.T, s *Session, begins, finishes uint64, active bool) {
	t.Helper()
	assertQwen35GraphLifecycleAt(t, s, begins, finishes, active, 0)
}

func assertQwen35GraphLifecycleAt(t *testing.T, s *Session, begins, finishes uint64, active bool, activePosition int) {
	t.Helper()
	r := s.qwen35DecodeGraph
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.begins != begins || r.finishes != finishes || (r.active != nil) != active {
		t.Fatalf("lifecycle begins=%d finishes=%d active=%v; want %d/%d/%v", r.begins, r.finishes, r.active != nil, begins, finishes, active)
	}
	if active && r.active.trace.Position != activePosition {
		t.Fatalf("active position=%d want %d", r.active.trace.Position, activePosition)
	}
}
