package ggufload

import (
	"math/rand"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=5ms lane=default
func TestInkling2ReadaheadChunksHandsOutEachChunkOnceWithinWindow(t *testing.T) {
	buf := make([]byte, 40)
	p := newInkling2ReadaheadChunks([][]byte{buf[0:10], buf[10:35], nil, buf[35:40]}, 16, 8)
	// Chunks: 8,2 | 8,8,8,1 | 5 at upload offsets 0,8,10,18,26,34,35.
	wantStarts := []int64{0, 8, 10, 18, 26, 34, 35}
	if len(p.starts) != len(wantStarts) {
		t.Fatalf("starts %v, want %v", p.starts, wantStarts)
	}
	for i, s := range wantStarts {
		if p.starts[i] != s {
			t.Fatalf("starts %v, want %v", p.starts, wantStarts)
		}
	}
	step := func(consumed int64, want int) {
		t.Helper()
		if got := len(p.due(consumed)); got != want {
			t.Fatalf("due(%d) = %d spans, want %d", consumed, got, want)
		}
	}
	step(0, 3)  // starts 0, 8, 10 < 16
	step(0, 0)  // never twice
	step(10, 1) // 18 < 26
	step(40, 3)
	step(80, 0)
	var total int
	for _, s := range p.spans {
		if len(s) > 8 || cap(s) != len(s) {
			t.Fatalf("span len %d cap %d escapes its chunk", len(s), cap(s))
		}
		total += len(s)
	}
	if total != len(buf) {
		t.Fatalf("spans cover %d bytes, want %d", total, len(buf))
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2UploadReadaheadStaysWindowAheadAndStopsOnClose(t *testing.T) {
	buf := make([]byte, 64)
	p := newInkling2ReadaheadChunks([][]byte{buf}, 16, 8)
	got := make(chan int, 16)
	r := startInkling2UploadReadahead(p, func(b []byte) { got <- len(b) })
	recv := func(n int) {
		t.Helper()
		for i := 0; i < n; i++ {
			select {
			case <-got:
			case <-time.After(5 * time.Second):
				t.Fatalf("advice %d of %d never came", i+1, n)
			}
		}
	}
	recv(2) // window 16 covers chunks at 0 and 8
	r.advance(32)
	recv(4) // through 40 < 48
	r.close()
	if len(got) != 0 {
		t.Fatalf("%d advice calls past the window", len(got))
	}
	// A goroutine that ran out of spans has already exited; close must not hang.
	q := startInkling2UploadReadahead(newInkling2ReadaheadChunks([][]byte{buf[:4]}, 16, 8), func([]byte) {})
	q.close()
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2UploadItemsFollowBindOrderAndSortByMapping(t *testing.T) {
	mapping := make([]byte, 1<<10)
	q := func(name string, typ TensorType, off, n, rows, cols int) *inkling2QuantMatrix {
		return &inkling2QuantMatrix{name: name, typ: typ, raw: mapping[off : off+n], rows: rows, cols: cols}
	}
	out := q("out", TensorQ4_K, 900, 100, 1, 256)
	wq0 := q("wq0", TensorQ4_K, 100, 50, 1, 256)
	router := q("router", TensorF32, 0, 8, 1, 2) // small dense: stays host
	wq1 := q("wq1", TensorQ4_K, 600, 50, 1, 256)
	gu0 := &inkling2QuantExperts{typ: TensorQ4_K, raw: mapping[200:300]}
	gu1 := &inkling2QuantExperts{typ: TensorQ4_K, raw: mapping[700:800]}
	m := &Inkling2LoopModel{
		HP: model.Inkling2LoopHParams{Passes: 2, CoreStart: 1, CoreEnd: 2},
		Weights: &model.Inkling2LoopWeights{Output: out, Layers: []model.Inkling2LoopLayerWeights{
			{Wq: wq0, Router: router, GateUp: gu0},
			{Wq: wq1, GateUp: gu1},
		}},
	}
	name := func(it inkling2UploadItem) string {
		if it.mat != nil {
			return (*it.mat).(*inkling2QuantMatrix).name
		}
		if (*it.exps).(*inkling2QuantExperts) == gu0 {
			return "gu0"
		}
		return "gu1"
	}
	check := func(items []inkling2UploadItem, want ...string) {
		t.Helper()
		if len(items) != len(want) {
			t.Fatalf("%d items, want %v", len(items), want)
		}
		for i, it := range items {
			if name(it) != want[i] {
				t.Fatalf("item %d = %s, want order %v", i, name(it), want)
			}
		}
	}
	// Bind order: output, layer matrices, core layer 1's experts before layer 0's.
	items := inkling2UploadItems(m, []bool{true, true})
	check(items, "out", "wq0", "router", "wq1", "gu1", "gu0")
	inkling2SortUploadsByMapping(items)
	check(items, "wq0", "gu0", "wq1", "gu1", "out", "router")
	if items[5].raw() != nil {
		t.Fatal("host-staying router reported an upload payload")
	}
	// A spilled layer's experts are never uploaded, so never read ahead.
	check(inkling2UploadItems(m, []bool{false, true}), "out", "wq0", "router", "wq1", "gu1")
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2StreamedUploadSwapsTheSameWeightsAndFreesThem(t *testing.T) {
	be := &inkling2FreeCounter{Backend: compute.Default()}
	r := rand.New(rand.NewSource(5))
	quant := func(typ TensorType, rows, cols int) *inkling2QuantMatrix {
		rb, _ := inkling2RowBytes(typ, cols)
		return &inkling2QuantMatrix{name: typ.String(), typ: typ, raw: randomInkling2Raw(t, r, typ, rows, cols), rows: rows, cols: cols, rowBytes: rb}
	}
	const experts, rows, cols = 2, 4, 128
	rb, _ := inkling2RowBytes(TensorNVFP4, cols)
	exps := &inkling2QuantExperts{typ: TensorNVFP4, raw: randomInkling2NVFP4(r, experts*rows, cols), experts: experts, rows: rows, cols: cols, rowLen: rb}
	router := quant(TensorF32, 8, 16)
	m := &Inkling2LoopModel{Weights: &model.Inkling2LoopWeights{
		Output: quant(TensorQ8_0, 5, 64),
		Layers: []model.Inkling2LoopLayerWeights{{Wq: quant(TensorQ8_0, 3, 64), Router: router, GateUp: exps}},
	}}
	s := &inkling2DeviceSet{be: be}
	if err := s.upload(m, []bool{true}, 64); err != nil {
		t.Fatal(err)
	}
	if _, ok := m.Weights.Output.(*inkling2DeviceMatrix); !ok {
		t.Fatalf("output stayed %T", m.Weights.Output)
	}
	if _, ok := m.Weights.Layers[0].Wq.(*inkling2DeviceMatrix); !ok {
		t.Fatalf("wq stayed %T", m.Weights.Layers[0].Wq)
	}
	if m.Weights.Layers[0].Router != model.Inkling2LoopMatrix(router) {
		t.Fatalf("small F32 router moved to %T", m.Weights.Layers[0].Router)
	}
	if _, ok := m.Weights.Layers[0].GateUp.(*inkling2DeviceExperts); !ok {
		t.Fatalf("experts stayed %T", m.Weights.Layers[0].GateUp)
	}
	if len(s.tensors) != 3 {
		t.Fatalf("set owns %d tensors, want 3", len(s.tensors))
	}
	be.freed = 0
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if be.freed != 3 {
		t.Fatalf("close freed %d tensors, want 3", be.freed)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2UploadReadaheadWindowGatesOnCUDAAndEnv(t *testing.T) {
	env := func(kv map[string]string) func(string) (string, bool) {
		return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
	}
	cuda := inkling2TierBackend{name: "cuda", tier: "sm_120"}
	def := int64(0)
	if inkling2UploadAdviseOn {
		def = inkling2UploadReadaheadDefault
	}
	cases := []struct {
		be   compute.Backend
		kv   map[string]string
		want int64
	}{
		{cuda, nil, def},
		{inkling2TierBackend{name: "vulkan"}, nil, 0},
		{nil, nil, 0},
		{cuda, map[string]string{"FAK_INKLING2_UPLOAD_READAHEAD_MB": "0"}, 0},
		{cuda, map[string]string{"FAK_INKLING2_UPLOAD_READAHEAD_MB": "junk"}, 0},
	}
	if inkling2UploadAdviseOn {
		cases = append(cases, struct {
			be   compute.Backend
			kv   map[string]string
			want int64
		}{cuda, map[string]string{"FAK_INKLING2_UPLOAD_READAHEAD_MB": "256"}, 256 << 20})
	}
	for i, c := range cases {
		if got := inkling2UploadReadaheadWindow(c.be, env(c.kv)); got != c.want {
			t.Fatalf("case %d: window %d, want %d", i, got, c.want)
		}
	}
}
