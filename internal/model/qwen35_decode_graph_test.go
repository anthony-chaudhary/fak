package model

import (
	"reflect"
	"testing"
)

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

type fakeQwen35ProjectionBatcher struct {
	events                  Qwen35NativeEvents
	begins, calls, finishes int
	aborts, closes          int
	panicInFullAttention    bool
}

func (b *fakeQwen35ProjectionBatcher) Begin(*Session) bool {
	b.begins++
	return true
}

func (b *fakeQwen35ProjectionBatcher) MulGroup(s *Session, _ int, names []string, x []float32, outs []int) ([][]float32, Qwen35NativeEvents, bool) {
	b.calls++
	if b.panicInFullAttention {
		panic("fake mixed projection failure")
	}
	k := sessionQ4KKernel{s: s}
	out := make([][]float32, len(names))
	for i, name := range names {
		out[i] = k.mul(name, x, outs[i], len(x))
	}
	return out, b.events, true
}

func (b *fakeQwen35ProjectionBatcher) Finish(*Session) { b.finishes++ }
func (b *fakeQwen35ProjectionBatcher) Abort(*Session)  { b.aborts++ }
func (b *fakeQwen35ProjectionBatcher) Close(*Session)  { b.closes++ }

func qwen35MixedBatchTestModel(t *testing.T) *Model {
	t.Helper()
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	name := layerName(cfg.NumLayers-1, "self_attn.v_proj.weight")
	meta := m.manifest[name]
	out, in := meta.Shape[0], meta.Shape[1]
	m.q4kw = map[string]*q4kTensor{
		name: quantizeQ4KFromRaw(make([]byte, out*(in/qkK)*q4kBlockBytes), out, in),
	}
	return m
}

func installFakeQwen35Batch(s *Session, b *fakeQwen35ProjectionBatcher) {
	s.qwen35ProjectionBatch = &qwen35ProjectionBatchState{batcher: b}
}

func TestQwen35MixedProjectionBatchDefaultIdentityAndProductionReachability(t *testing.T) {
	m := qwen35MixedBatchTestModel(t)
	control := m.NewSession()
	control.Q4K = true
	want := control.tokenHiddenQ(1, 0)

	candidate := m.NewSession()
	candidate.Q4K = true
	fake := &fakeQwen35ProjectionBatcher{events: Qwen35NativeEvents{CommandBuffers: 1, Commits: 1, Waits: 1}}
	installFakeQwen35Batch(candidate, fake)
	got := candidate.tokenHiddenQ(1, 0)
	if !reflect.DeepEqual(got, want) {
		t.Fatal("explicit candidate changed Qwen35 tokenHiddenQ output")
	}
	if fake.begins != 1 || fake.calls != 1 || fake.finishes != 1 || fake.aborts != 0 {
		t.Fatalf("candidate lifecycle = begin:%d call:%d finish:%d abort:%d", fake.begins, fake.calls, fake.finishes, fake.aborts)
	}
}

func TestQwen35MixedProjectionBatchSessionIsolationAbortAndClose(t *testing.T) {
	m := qwen35MixedBatchTestModel(t)
	a, b := m.NewSession(), m.NewSession()
	a.Q4K, b.Q4K = true, true
	fa := &fakeQwen35ProjectionBatcher{}
	fb := &fakeQwen35ProjectionBatcher{}
	installFakeQwen35Batch(a, fa)
	installFakeQwen35Batch(b, fb)
	_ = a.tokenHiddenQ(1, 0)
	if fa.calls != 1 || fb.calls != 0 {
		t.Fatalf("session batch state leaked: a=%d b=%d", fa.calls, fb.calls)
	}
	b.Close()
	if fb.closes != 1 || b.qwen35ProjectionBatch != nil {
		t.Fatalf("Backend=nil Close did not clean batch owner: closes=%d state=%v", fb.closes, b.qwen35ProjectionBatch)
	}

	panicBatch := &fakeQwen35ProjectionBatcher{panicInFullAttention: true}
	c := m.NewSession()
	c.Q4K = true
	installFakeQwen35Batch(c, panicBatch)
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("expected candidate panic")
			}
		}()
		_ = c.tokenHiddenQ(1, 0)
	}()
	if panicBatch.aborts != 1 || panicBatch.finishes != 0 || c.qwen35ProjectionBatch.active {
		t.Fatalf("abort cleanup = abort:%d finish:%d active:%v", panicBatch.aborts, panicBatch.finishes, c.qwen35ProjectionBatch.active)
	}
	c.Close()
	if panicBatch.closes != 1 || c.qwen35ProjectionBatch != nil {
		t.Fatalf("close cleanup = closes:%d state=%v", panicBatch.closes, c.qwen35ProjectionBatch)
	}
}

func TestQwen35MixedProjectionBatchNativeEventReduction(t *testing.T) {
	m := qwen35MixedBatchTestModel(t)
	run := func(events uint64) Qwen35DecodeGraphTrace {
		s := m.NewSession()
		s.Q4K = true
		s.Quant = true
		s.EnableQwen35DecodeGraphTrace()
		installFakeQwen35Batch(s, &fakeQwen35ProjectionBatcher{events: Qwen35NativeEvents{
			CommandBuffers: events,
			Commits:        events,
			Waits:          events,
		}})
		_ = s.tokenHidden(1, 0)
		return s.LastQwen35DecodeGraphTrace()
	}
	control := run(2)
	candidate := run(1)
	if control.NativeEvents.CommandBuffers != 2 || candidate.NativeEvents.CommandBuffers != 1 {
		t.Fatalf("native command buffers control=%+v candidate=%+v", control.NativeEvents, candidate.NativeEvents)
	}
	syncs := func(tr Qwen35DecodeGraphTrace) int {
		for _, n := range tr.Nodes {
			if n.Operation == "full-attention" {
				return n.Syncs
			}
		}
		return -1
	}
	if syncs(control) != 2 || syncs(candidate) != 1 {
		t.Fatalf("native-derived syncs control=%+v candidate=%+v", control.Nodes, candidate.Nodes)
	}
}
