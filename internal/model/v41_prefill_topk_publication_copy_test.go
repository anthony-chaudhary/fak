package model

import (
	"runtime"
	"testing"
)

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41PrefillTopKPublicationCopy(t *testing.T) {
	const sources, seq, width, hd = 8, 512, 512, 4
	m := &Model{Cfg: Config{NumLayers: sources, DeepSeekV41: &DeepSeekV41Config{
		CompressRatios:      []int{2, 2, 2, 2, 2, 2, 2, 2},
		IndexSourceLayerIDs: []int{0, 1, 2, 3, 4, 5, 6, 7},
	}}}
	kv := make([][]float32, seq)
	for i := range kv {
		kv[i] = []float32{1, 2, 3, 4}
	}
	input := make([]int32, seq*width)
	for i := range input {
		input[i] = int32(i%257) - 1
	}
	publish := func(indices []int32) *v41ForwardState {
		st := &v41ForwardState{}
		for l := 0; l < sources; l++ {
			src := v41LayerSources{plan: V41AttentionPlan{Layer: l, KVSourceLayer: l, Ratio: 2, TopKWidth: width}}
			if err := m.v41LayerPublishAttentionState(l, seq, hd, src, v41LayerQKV{kvRows: kv}, nil, indices, st); err != nil {
				t.Fatal(err)
			}
		}
		return st
	}
	// Exercise the production layer publication, rather than a stand-in row builder.
	st := publish(input)
	if !st.attn.topkSet || st.attn.topkRatio != 2 || len(st.attn.topk) != seq {
		t.Fatal("missing source publication")
	}
	for row, ids := range st.attn.topk {
		if len(ids) != width {
			t.Fatalf("row %d width = %d", row, len(ids))
		}
		for col, id := range ids {
			if want := input[row*width+col]; id != want {
				t.Fatalf("row %d column %d: got %d want %d", row, col, id, want)
			}
		}
	}
	for i := range input {
		input[i] = -99
	}
	for row, ids := range st.attn.topk {
		for col, id := range ids {
			if want := int32((row*width+col)%257) - 1; id != want {
				t.Fatalf("input mutation changed owned row %d column %d", row, col)
			}
		}
	}
	// The same real seam with nil indices isolates all non-top-k layer setup.
	// Only two row-header slabs and one owned payload per row may be added per source.
	var retained *v41ForwardState
	baseAllocs := testing.AllocsPerRun(5, func() { retained = publish(nil) })
	topKAllocs := testing.AllocsPerRun(5, func() { retained = publish(input) })
	bytesFor := func(indices []int32) uint64 {
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		retained = publish(indices)
		runtime.ReadMemStats(&after)
		return after.TotalAlloc - before.TotalAlloc
	}
	baseBytes, topKBytes := bytesFor(nil), bytesFor(input)
	runtime.KeepAlive(retained)
	addedAllocs, addedBytes := topKAllocs-baseAllocs, topKBytes-baseBytes
	t.Logf("actual publication seam: sources=%d P=%d K=%d no-top-k=%g allocs/%d bytes top-k=%g allocs/%d bytes added=%g allocs/%d bytes", sources, seq, width, baseAllocs, baseBytes, topKAllocs, topKBytes, addedAllocs, addedBytes)
	if want := float64(sources * (seq + 2)); addedAllocs != want {
		t.Fatalf("top-k allocations = %g, want %g (one owned copy per row)", addedAllocs, want)
	}
	// Leave room for allocator rounding of the two header slabs, but not another payload copy.
	if maxBytes := uint64(sources * (seq*width*4 + 3*seq*24)); addedBytes > maxBytes {
		t.Fatalf("top-k allocated %d bytes, max %d", addedBytes, maxBytes)
	}
}
