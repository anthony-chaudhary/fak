package ggufload

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func writeIssue9073MappedQ4KFixture(t *testing.T) string {
	t.Helper()
	const dim = 256
	const payloadBytes = 144
	var b bytes.Buffer
	writeMinimalHeader(&b, 1, 8)
	writeKVString(&b, "general.architecture", "llama")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "llama.embedding_length", dim)
	writeKVUint32(&b, "llama.block_count", 1)
	writeKVUint32(&b, "llama.attention.head_count", 1)
	writeKVUint32(&b, "llama.attention.key_length", dim)
	writeKVUint32(&b, "llama.feed_forward_length", dim)
	writeKVFloat32(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)
	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, 1}, TensorQ4_K, 0)
	padToAlignment(&b, 32)
	for i := 0; i < payloadBytes; i++ {
		b.WriteByte(byte(i*17 + 3))
	}
	page := os.Getpagesize()
	for b.Len()%page != 0 {
		b.WriteByte(0)
	}
	path := filepath.Join(t.TempDir(), "issue-9073.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestIssue9073StreamedDenseQ4KCarriesBoundsCheckedMappedShardView(t *testing.T) {
	const payloadBytes = 144
	path := writeIssue9073MappedQ4KFixture(t)
	t.Setenv("FAK_GGUF_MMAP", "1")
	resetGGUFMmapGateForTest()
	t.Cleanup(resetGGUFMmapGateForTest)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	if ws.data == nil {
		t.Skip("GGUF mmap unavailable")
	}
	info := ws.File.Tensors[0]
	tw := ws.lazyDenseQ4KTensorWork(info, "model.layers.0.self_attn.v_proj.weight", 0)
	if tw.err != nil || len(tw.pending) != 1 {
		t.Fatalf("lazy work = %+v", tw)
	}
	mapped, ok := tw.pending[0].lazyReader.(*mappedQ4KReaderAt)
	if !ok {
		t.Fatalf("lazy reader = %T, want mapped Q4_K view", tw.pending[0].lazyReader)
	}
	if mapped.offset != int(info.FileOffset) || len(mapped.span)%os.Getpagesize() != 0 {
		t.Fatalf("mapped view offset=%d span=%d", mapped.offset, len(mapped.span))
	}
	if end := mapped.offset + payloadBytes; end > len(mapped.span) {
		t.Fatalf("mapped payload end=%d exceeds span=%d", end, len(mapped.span))
	}
}

// TestLoadModelQ4KRoutesByIdentityNorm is the small-scale (27B-free) integration test for
// the resident-Q4_K loader. It writes a real 1-layer GGUF with four Q4_K matmul tensors,
// loads it via LoadModelQ4K, and asserts the loader's routing:
//
//   - identity-normalized weights (self_attn.v_proj, mlp.down_proj, lm_head) → resident q4kw
//     (raw Q4_K bytes, no round-trip).
//   - the normalize-sensitive self_attn.q_proj (rotary) → the proven dequant→normalize→Q8
//     path (q8w), NOT q4kw — this is the critical correctness gate: storing q_proj raw would
//     feed wrongly-laid-out weights to the forward.
//
// It also checks ResidentReport reflects the 3/1 split. This is the cheapest witness that
// the loader's NEW routing logic (eligibility + type gate + CanonicalTensorName mapping +
// the dequant+normalize fallback) all fire together correctly on a real GGUF file.
func TestLoadModelQ4KRoutesByIdentityNorm(t *testing.T) {
	const dim = 256 // reduction dim a multiple of qkK → 1 super-block/row
	// Each Q4_K tensor [dim,dim]: 256 super-blocks × 144 B = 36864 B. All-zero blocks are
	// valid Q4_K (d=min=0 → dequant to all zeros); the test checks ROUTING, not values.
	const blkBytes = 36864

	var b bytes.Buffer
	writeMinimalHeader(&b, 4, 8)
	writeKVString(&b, "general.architecture", "llama")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "llama.embedding_length", dim)
	writeKVUint32(&b, "llama.block_count", 1)
	writeKVUint32(&b, "llama.attention.head_count", 1)
	writeKVUint32(&b, "llama.attention.key_length", dim)
	writeKVUint32(&b, "llama.feed_forward_length", dim)
	writeKVFloat32(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)

	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, dim}, TensorQ4_K, 0)
	writeTensorInfoForTest(&b, "blk.0.attn_q.weight", []uint64{dim, dim}, TensorQ4_K, blkBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_down.weight", []uint64{dim, dim}, TensorQ4_K, 2*blkBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, dim}, TensorQ4_K, 3*blkBytes)
	padToAlignment(&b, 32)
	for i := 0; i < 4; i++ {
		b.Write(bytes.Repeat([]byte{0}, blkBytes))
	}

	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadModelQ4K(path)
	if err != nil {
		t.Fatalf("LoadModelQ4K: %v", err)
	}

	// Identity-normalized weights → q4kw (raw), NOT q8w.
	for _, name := range []string{
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
		"lm_head.weight",
	} {
		if !m.HasQ4K(name) {
			t.Errorf("expected %s in q4kw (identity-norm → raw)", name)
		}
		if m.HasQ8(name) {
			t.Errorf("%s must NOT be in q8w (it is identity-normalized)", name)
		}
	}
	// Normalize-sensitive q_proj → q8w (dequant→normalize→Q8), NOT q4kw.
	qp := "model.layers.0.self_attn.q_proj.weight"
	if !m.HasQ8(qp) {
		t.Errorf("expected %s in q8w (rotary → normalize-sensitive)", qp)
	}
	if m.HasQ4K(qp) {
		t.Errorf("%s must NOT be in q4kw (rotary weights held raw would corrupt the forward)", qp)
	}
	r := m.ResidentReport()
	if r.Q4KTensors != 3 || r.Q8Tensors != 1 {
		t.Errorf("resident split: q4k=%d q8=%d, want 3/1", r.Q4KTensors, r.Q8Tensors)
	}
}

func TestLoadModelQ4KProfileTicksProgress(t *testing.T) {
	const dim = 256
	const blkBytes = 36864

	var b bytes.Buffer
	writeMinimalHeader(&b, 4, 8)
	writeKVString(&b, "general.architecture", "llama")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "llama.embedding_length", dim)
	writeKVUint32(&b, "llama.block_count", 1)
	writeKVUint32(&b, "llama.attention.head_count", 1)
	writeKVUint32(&b, "llama.attention.key_length", dim)
	writeKVUint32(&b, "llama.feed_forward_length", dim)
	writeKVFloat32(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)

	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, dim}, TensorQ4_K, 0)
	writeTensorInfoForTest(&b, "blk.0.attn_q.weight", []uint64{dim, dim}, TensorQ4_K, blkBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_down.weight", []uint64{dim, dim}, TensorQ4_K, 2*blkBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, dim}, TensorQ4_K, 3*blkBytes)
	padToAlignment(&b, 32)
	for i := 0; i < 4; i++ {
		b.Write(bytes.Repeat([]byte{0}, blkBytes))
	}

	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	prof := NewLoadProfiler()
	var progress bytes.Buffer
	prof.Progress = &progress
	if _, err := LoadModelQ4KProfile(path, prof); err != nil {
		t.Fatalf("LoadModelQ4KProfile: %v", err)
	}
	if prof.Total != 4 || prof.ggufSeen != 4 {
		t.Fatalf("progress tensors total/seen = %d/%d, want 4/4", prof.Total, prof.ggufSeen)
	}
	if prof.cumBytes != 4*blkBytes {
		t.Fatalf("progress bytes = %d, want %d", prof.cumBytes, 4*blkBytes)
	}
	if out := progress.String(); !strings.Contains(out, "100% (4/4 tensors") {
		t.Fatalf("progress output missing final 100%% line:\n%s", out)
	}
}

func TestQ4KLoadOptionCanRouteDenseKQuantToQ8(t *testing.T) {
	cfg := model.Config{ModelType: "qwen35", HiddenSize: 256, IntermediateSize: 256}
	name := "model.layers.0.mlp.down_proj.weight"
	shape := []int{256, 256}
	// Q6_K is the exact mixed-quant type used by Qwen3.6 q4_k_m down_proj tensors.
	raw := make([]byte, (shape[0]*shape[1]/256)*210)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	info := TensorInfo{Name: "blk.0.ffn_down.weight", Dims: []uint64{256, 256}, Type: TensorQ6_K}
	data, err := dequantF32(info, raw)
	if err != nil {
		t.Fatalf("dequant Q6_K fixture: %v", err)
	}
	b := model.NewQuantBuilder(cfg, false)
	if err := b.AddF32Tensor(name, shape, data); err != nil {
		t.Fatalf("route dense Q6_K through Q8: %v", err)
	}
	m, err := b.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if !m.HasQ8(name) {
		t.Fatalf("dense Q6_K %q missing from Q8 store", name)
	}
	if m.HasQ4K(name) {
		t.Fatalf("dense Q6_K %q unexpectedly entered Q4_K store", name)
	}

	opts, err := resolveQ4KLoadOptions(cfg, []Q4KLoadOption{WithDenseKQuantResident(false)})
	if err != nil {
		t.Fatalf("resolve option: %v", err)
	}
	if opts.residentDenseKQuant {
		t.Fatal("WithDenseKQuantResident(false) did not disable dense raw k-quant residency")
	}
}

type countingCloser struct {
	closes atomic.Int32
}

func (c *countingCloser) Close() error {
	c.closes.Add(1)
	return nil
}

func TestLoadModelQ4KProfileOptionsContextClosesReaderOnceBeforeReturn(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	closer := &countingCloser{}
	open := func(string) (*WeightSource, error) {
		cancel()
		return &WeightSource{File: &File{}, closers: []io.Closer{closer}}, nil
	}

	_, err := loadModelQ4KProfileOptionsContext(ctx, "ignored", nil, open)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("LoadModelQ4KProfileOptionsContext error = %v, want context.Canceled", err)
	}
	if got := closer.closes.Load(); got != 1 {
		t.Fatalf("reader closes before return = %d, want exactly 1", got)
	}
}
