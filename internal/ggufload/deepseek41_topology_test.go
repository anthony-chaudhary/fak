package ggufload

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

var ds41PublishedRatios = []int32{
	0, 0,
	2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
	1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
	0, 0, 0,
}

// ds41TopologyFile is the staged vcruz shape: the published compress schedule and
// RoPE base, compressor and index-key tensors on the KV sources, indexer queries
// on the index sources, and no Engram or topology metadata keys at all.
func ds41TopologyFile(kvSources, indexSources []int) *File {
	meta := ds41Meta("deepseek41")
	for key := range meta {
		if strings.Contains(key, "engram") {
			delete(meta, key)
		}
	}
	meta["deepseek41.attention.compress_ratios"] = vcruzI32Array(ds41PublishedRatios)
	meta["deepseek41.attention.compress_rope_freq_base"] = Value{Type: TypeFloat32, Value: float32(160000)}
	f := &File{Metadata: meta}
	for _, l := range kvSources {
		for _, leaf := range []string{"attn_compressor_kv", "attn_compressor_gate", "indexer.attn_k"} {
			f.Tensors = append(f.Tensors, TensorInfo{Name: fmt.Sprintf("blk.%d.%s.weight", l, leaf), Dims: []uint64{5120, 512}, Type: TensorF32})
		}
	}
	for _, l := range indexSources {
		f.Tensors = append(f.Tensors, TensorInfo{Name: fmt.Sprintf("blk.%d.indexer.attn_q_b.weight", l), Dims: []uint64{1280, 4096}, Type: TensorF32})
	}
	return f
}

// A GGUF with no Engram still carries the compressed RoPE base, the
// tensor-derived KV/index sources and the published candidate geometry.
// fak-test:runtime fast est=10ms
func TestDeepSeek41GGUFTopologyCarriesPublishedSchedule(t *testing.T) {
	kv, index := []int{2, 8, 14, 20}, []int{2, 8, 14, 20, 24, 28, 32, 36}
	cfg, err := ds41TopologyFile(kv, index).Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	d41 := cfg.DeepSeekV41
	if d41 == nil {
		t.Fatal("deepseek41 GGUF with a compress schedule produced no V4.1 metadata")
	}
	got := []any{d41.CompressRopeTheta, d41.KVSourceLayerIDs, d41.IndexSourceLayerIDs,
		d41.CandidateSourceLayerID, d41.CandidateTopKBlocks, d41.CandidateBlockSize, d41.HCMult, len(d41.CompressRatios)}
	want := []any{160000.0, kv, index, 20, 2048, 8, 4, len(ds41PublishedRatios)}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("V4.1 topology = %v, want %v", got, want)
	}
	if len(d41.EngramLayerIDs) != 0 {
		t.Fatalf("Engram-free GGUF declared Engram layers %v", d41.EngramLayerIDs)
	}
}

// A header-only file adopts the published sources; a tensor directory that
// disagrees with the published schedule is refused, never run as another topology.
// fak-test:runtime fast est=10ms
func TestDeepSeek41GGUFTopologyCrossChecksPublishedSources(t *testing.T) {
	cfg, err := ds41TopologyFile(nil, nil).Config()
	if err != nil {
		t.Fatalf("header-only Config: %v", err)
	}
	if d41 := cfg.DeepSeekV41; d41 == nil || !reflect.DeepEqual(d41.KVSourceLayerIDs, []int{2, 8, 14, 20}) || d41.CandidateSourceLayerID != 20 {
		t.Fatalf("header-only file did not adopt the published topology: %+v", d41)
	}
	for _, c := range []struct{ kv, index []int }{
		{[]int{2, 3, 8, 14, 20}, []int{2, 8, 14, 20, 24, 28, 32, 36}},
		{[]int{2, 8, 14, 20}, []int{2, 8, 14, 20}},
		{[]int{2, 8, 14}, []int{2, 8, 14, 20, 24, 28, 32, 36}},     // ratio-one layer 20 would read ratio-two layer 14
		{[]int{3, 8, 14, 20}, []int{2, 8, 14, 20, 24, 28, 32, 36}}, // layer 2 reads before any KV source publishes
	} {
		if _, err := ds41TopologyFile(c.kv, c.index).Config(); !errors.Is(err, model.ErrV41ConfigAdmission) {
			t.Errorf("sources kv=%v index=%v: err = %v, want ErrV41ConfigAdmission", c.kv, c.index, err)
		}
	}
	extraKeyOwner := ds41TopologyFile([]int{2, 8, 14, 20}, []int{2, 8, 14, 20, 24, 28, 32, 36})
	extraKeyOwner.Tensors = append(extraKeyOwner.Tensors, TensorInfo{Name: "blk.24.indexer.attn_k.weight", Dims: []uint64{512, 128}, Type: TensorF32})
	if _, err := extraKeyOwner.Config(); !errors.Is(err, model.ErrV41ConfigAdmission) {
		t.Errorf("index-key owner outside the KV sources: err = %v, want ErrV41ConfigAdmission", err)
	}
}
