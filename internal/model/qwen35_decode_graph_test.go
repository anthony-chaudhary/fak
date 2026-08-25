package model

import "testing"

func TestQwen35DecodeGraphTraceProductionOrderAndIsolation(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	cfg.NumLayers = 2
	cfg.LayerTypes = []string{"linear_attention", "full_attention"}
	m := NewSynthetic(cfg)
	a, b := m.NewSession(), m.NewSession()
	a.EnableQwen35DecodeGraphTrace()
	b.EnableQwen35DecodeGraphTrace()
	_ = a.tokenHidden(1, 7)
	_ = b.tokenHidden(2, 9)
	ta, tb := a.LastQwen35DecodeGraphTrace(), b.LastQwen35DecodeGraphTrace()
	if ta.Position != 7 || tb.Position != 9 || ta.Aborted || tb.Aborted {
		t.Fatalf("traces a=%+v b=%+v", ta, tb)
	}
	want := []string{"linear-attention", "post-attention-residual-norm", "mlp", "next-layer-handoff", "full-attention", "post-attention-residual-norm", "mlp", "next-layer-handoff"}
	if len(ta.Nodes) != len(want) {
		t.Fatalf("nodes=%d want %d: %+v", len(ta.Nodes), len(want), ta.Nodes)
	}
	for i, w := range want {
		if ta.Nodes[i].Operation != w {
			t.Fatalf("node[%d]=%q want %q", i, ta.Nodes[i].Operation, w)
		}
	}
	if &ta.Nodes[0] == &tb.Nodes[0] {
		t.Fatal("sessions share trace storage")
	}
	ta.Nodes[0].Operation = "mutated"
	if got := a.LastQwen35DecodeGraphTrace().Nodes[0].Operation; got != "linear-attention" {
		t.Fatalf("snapshot aliases recorder: %q", got)
	}
}

func TestQwen35DecodeGraphTraceAbortCleansActiveOwner(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	s := m.NewSession()
	s.EnableQwen35DecodeGraphTrace()
	finish := s.beginQwen35DecodeGraph(3)
	s.recordQwen35DecodeGraph(Qwen35DecodeGraphNode{Operation: "partial"})
	finish(true)
	tr := s.LastQwen35DecodeGraphTrace()
	if !tr.Aborted || len(tr.Nodes) != 1 {
		t.Fatalf("trace=%+v", tr)
	}
	finish = s.beginQwen35DecodeGraph(4)
	finish(false)
	tr = s.LastQwen35DecodeGraphTrace()
	if tr.Aborted || tr.Position != 4 || len(tr.Nodes) != 0 {
		t.Fatalf("second trace leaked first: %+v", tr)
	}
}
