package ggufload

import (
	"bytes"
	"io"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestApplyLazyQ4KStoresRangeWithoutPayload(t *testing.T) {
	cfg := model.Config{HiddenSize: 256}
	b := model.NewQuantBuilder(cfg, false)
	payload := make([]byte, blockQ4KBytes)
	info := TensorInfo{Name: "blk.0.ffn_down.weight", Type: TensorQ4_K, FileOffset: 7, Dims: []uint64{256, 1}}
	tw := tensorWork{pending: []pendingTensor{{lazyQ4K: true, name: "model.layers.0.mlp.down_proj.weight", shape: []int{1, 256}, sourceInfo: info, lazyReader: bytes.NewReader(append(make([]byte, 7), payload...))}}}
	if err := applyQ4KTensorWork(tw, nil, cfg, b, map[int]glmKVBHalf{}, false); err != nil {
		t.Fatal(err)
	}
	if err := b.AddF32Tensor("model.layers.0.self_attn.v_proj.weight", []int{1, 256}, make([]float32, 256)); err != nil {
		t.Fatal(err)
	}
	m, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	if !m.Q4KLazy("model.layers.0.mlp.down_proj.weight") {
		t.Fatal("Q4_K tensor was not lazy")
	}
}

type lazyReadCounter struct {
	r     io.ReaderAt
	reads atomic.Int64
}

func (r *lazyReadCounter) ReadAt(p []byte, off int64) (int, error) {
	r.reads.Add(1)
	return r.r.ReadAt(p, off)
}

func TestLazyDenseQ4KIndexDoesNotReadPayload(t *testing.T) {
	payload := make([]byte, blockQ4KBytes)
	reader := &lazyReadCounter{r: bytes.NewReader(payload)}
	info := TensorInfo{
		Name:       "blk.0.ffn_down.weight",
		Type:       TensorQ4_K,
		FileOffset: 0,
		Dims:       []uint64{256, 1},
	}
	ws := &WeightSource{
		File:   &File{Tensors: []TensorInfo{info}},
		r:      reader,
		size:   int64(len(payload)),
		byName: map[string]int{info.Name: 0},
	}

	tw := ws.lazyDenseQ4KTensorWork(info, "model.layers.0.mlp.down_proj.weight", int64(len(payload)))
	if tw.err != nil {
		t.Fatal(tw.err)
	}
	if got := reader.reads.Load(); got != 0 {
		t.Fatalf("lazy indexing made %d payload reads, want zero", got)
	}
	if len(tw.pending) != 1 || !tw.pending[0].lazyQ4K || tw.pending[0].lazyReader != reader {
		t.Fatalf("lazy tensor descriptor = %+v", tw.pending)
	}
}

type rangeReadCounter struct {
	r        io.ReaderAt
	lo, hi   int64
	overlaps atomic.Int64
}

func (r *rangeReadCounter) ReadAt(p []byte, off int64) (int, error) {
	if off < r.hi && off+int64(len(p)) > r.lo {
		r.overlaps.Add(1)
	}
	return r.r.ReadAt(p, off)
}

func TestQwen38StreamedDenseQ4KLoaderBuildsWithoutQ4KPayloadRead(t *testing.T) {
	const tensorName = "blk.0.ffn_down.weight"
	q4Payload := make([]byte, blockQ4KBytes)
	f32Payload := make([]byte, 256*4)
	blob := append(append([]byte(nil), q4Payload...), f32Payload...)
	reader := &rangeReadCounter{r: bytes.NewReader(blob), lo: 0, hi: int64(len(q4Payload))}
	meta := map[string]Value{
		"general.architecture":                   {Type: TypeString, Value: "qwen2"},
		"general.name":                           {Type: TypeString, Value: "Qwen3.8-27B-Q4_K_M"},
		"qwen2.embedding_length":                 {Type: TypeUint32, Value: uint32(256)},
		"qwen2.block_count":                      {Type: TypeUint32, Value: uint32(0)},
		"qwen2.attention.head_count":             {Type: TypeUint32, Value: uint32(1)},
		"qwen2.attention.head_count_kv":          {Type: TypeUint32, Value: uint32(1)},
		"qwen2.feed_forward_length":              {Type: TypeUint32, Value: uint32(256)},
		"qwen2.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},
	}
	file := &File{Metadata: meta, Tensors: []TensorInfo{
		{Name: tensorName, Type: TensorQ4_K, Dims: []uint64{256, 1}},
		{Name: "blk.0.attn_v.weight", Type: TensorF32, Dims: []uint64{256, 1}, FileOffset: int64(len(q4Payload))},
	}}
	ws, err := NewWeightSource(file, reader, int64(len(blob)))
	if err != nil {
		t.Fatal(err)
	}
	m, err := ws.QuantModelQ4KProfileOptions(nil, WithStreamedDenseQ4K(true))
	if err != nil {
		t.Fatal(err)
	}
	if got := reader.overlaps.Load(); got != 0 {
		t.Fatalf("streamed model build made %d reads overlapping the Q4_K payload, want zero", got)
	}
	if got := m.Cfg.Name; got != "Qwen3.8-27B-Q4_K_M" {
		t.Fatalf("checkpoint identity = %q", got)
	}
	if !m.Q4KLazy("model.layers.0.mlp.down_proj.weight") {
		t.Fatal("exact-model dense Q4_K tensor was not checkpoint-backed")
	}
}
