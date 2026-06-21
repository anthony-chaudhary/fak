package ggufload

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// gguf_split_test.go covers the multi-file GGUF split path (the
// "-NNNNN-of-MMMMM.gguf" shards that HuggingFace emits for large models). The
// fixtures here are two tiny synthetic shards written through the same low-level
// byte builders the rest of this package's tests use, so the test is a real
// end-to-end exercise of OpenWeights → merge → per-shard tensor routing with no
// network and no real 7B download.

// splitTensor describes one tensor to place into a synthetic shard's data section.
type splitTensor struct {
	name string
	dims []uint64
	typ  TensorType
	data []byte // already-aligned payload bytes, offset 0
}

// writeSplitShard builds a single GGUF shard. When withConfig is set the shard
// carries a minimal qwen2 config (so shard 1 yields a usable Config()); the
// split.* metadata is always present so the parent can route the shard.
func writeSplitShard(t *testing.T, shardNo, count, totalTensors uint32, withConfig bool, tensors []splitTensor) []byte {
	t.Helper()
	const align = 32

	kvs := uint64(3) // split.no, split.count, split.tensors.count
	if withConfig {
		kvs += 8 // general.alignment + general.architecture + 6 qwen2.* dims
	}
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), kvs)
	if withConfig {
		writeKVUint32(&b, "general.alignment", align)
		writeKVString(&b, "general.architecture", "qwen2")
		writeKVUint64(&b, "qwen2.embedding_length", 32)
		writeKVUint64(&b, "qwen2.block_count", 2)
		writeKVUint64(&b, "qwen2.feed_forward_length", 64)
		writeKVUint64(&b, "qwen2.attention.head_count", 4)
		writeKVUint64(&b, "qwen2.attention.head_count_kv", 2)
		writeKVFloat32(&b, "qwen2.attention.layer_norm_rms_epsilon", 1e-5)
	}
	writeKVUint32(&b, "split.no", shardNo)
	writeKVUint32(&b, "split.count", count)
	writeKVUint32(&b, "split.tensors.count", totalTensors)

	for _, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, tt.dims, tt.typ, 0)
	}
	padToAlignment(&b, align)
	for _, tt := range tensors {
		dataStart := b.Len()
		b.Write(tt.data)
		padToLen(&b, dataStart+align)
	}
	return b.Bytes()
}

func f32Payload(vs ...float32) []byte {
	var buf bytes.Buffer
	for _, v := range vs {
		_ = binary.Write(&buf, binary.LittleEndian, math.Float32bits(v))
	}
	return buf.Bytes()
}

// TestOpenWeightsLoadsSplitShardsFromShardOne is the happy path: hand
// OpenWeights shard 1 of a 2-shard split and confirm both shards' tensors are
// merged and read back from the correct files, and that Config() resolves from
// shard 1's metadata.
func TestOpenWeightsLoadsSplitShardsFromShardOne(t *testing.T) {
	dir := t.TempDir()
	shard1 := writeSplitShard(t, 1, 2, 2, true, []splitTensor{
		{name: "a.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(1.25, -2.5)},
	})
	shard2 := writeSplitShard(t, 2, 2, 2, false, []splitTensor{
		{name: "b.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(3.5, 7.0)},
	})
	p1 := filepath.Join(dir, "tiny-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "tiny-00002-of-00002.gguf")
	if err := os.WriteFile(p1, shard1, 0o644); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := os.WriteFile(p2, shard2, 0o644); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}

	ws, err := OpenWeights(p1)
	if err != nil {
		t.Fatalf("OpenWeights(shard1): %v", err)
	}
	defer ws.Close()

	if got := len(ws.File.Tensors); got != 2 {
		t.Fatalf("merged tensor count=%d, want 2", got)
	}
	assertF32 := func(name string, want ...float32) {
		t.Helper()
		got, _, err := ws.TensorF32(name)
		if err != nil {
			t.Fatalf("TensorF32(%s): %v", name, err)
		}
		if len(got) != len(want) {
			t.Fatalf("%s len=%d, want %d", name, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d]=%v, want %v", name, i, got[i], want[i])
			}
		}
	}
	assertF32("a.weight", 1.25, -2.5) // served by shard 1
	assertF32("b.weight", 3.5, 7.0)   // served by shard 2

	// Config must resolve from shard 1's metadata even though tensors span both.
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.ModelType != "qwen2" || cfg.HiddenSize != 32 {
		t.Fatalf("config not resolved from shard 1: %#v", cfg)
	}
}

// TestOpenWeightsResolvesShardOneFromLaterShard confirms that handing
// OpenWeights a non-first shard still loads the whole model: shard 1's path is
// derived from the -N-of-M.gguf suffix so the config-carrying shard is opened.
func TestOpenWeightsResolvesShardOneFromLaterShard(t *testing.T) {
	dir := t.TempDir()
	shard1 := writeSplitShard(t, 1, 2, 2, true, []splitTensor{
		{name: "a.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(1.25, -2.5)},
	})
	shard2 := writeSplitShard(t, 2, 2, 2, false, []splitTensor{
		{name: "b.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(3.5, 7.0)},
	})
	p1 := filepath.Join(dir, "tiny-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "tiny-00002-of-00002.gguf")
	if err := os.WriteFile(p1, shard1, 0o644); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := os.WriteFile(p2, shard2, 0o644); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}

	ws, err := OpenWeights(p2) // later shard entry
	if err != nil {
		t.Fatalf("OpenWeights(shard2): %v", err)
	}
	defer ws.Close()

	got, _, err := ws.TensorF32("a.weight")
	if err != nil {
		t.Fatalf("TensorF32(a.weight) via later-shard entry: %v", err)
	}
	if math.Float32bits(got[0]) != math.Float32bits(1.25) {
		t.Fatalf("a.weight[0]=%v, want 1.25", got[0])
	}
}

// TestOpenWeightsLoadsHFZeroIndexedSplit mirrors the real HuggingFace GGUF
// split writer, which is 0-indexed: the first shard declares split.no=0 (not 1,
// as llama.cpp does). This is the convention the downloaded 7B/14B/32B Qwen
// checkpoints use, so it must load identically to the 1-indexed path.
func TestOpenWeightsLoadsHFZeroIndexedSplit(t *testing.T) {
	dir := t.TempDir()
	shard1 := writeSplitShard(t, 0, 2, 2, true, []splitTensor{
		{name: "a.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(1.25, -2.5)},
	})
	shard2 := writeSplitShard(t, 1, 2, 2, false, []splitTensor{
		{name: "b.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(3.5, 7.0)},
	})
	p1 := filepath.Join(dir, "hf-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "hf-00002-of-00002.gguf")
	if err := os.WriteFile(p1, shard1, 0o644); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := os.WriteFile(p2, shard2, 0o644); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}

	ws, err := OpenWeights(p1)
	if err != nil {
		t.Fatalf("OpenWeights(0-indexed shard1): %v", err)
	}
	defer ws.Close()

	got, _, err := ws.TensorF32("b.weight")
	if err != nil {
		t.Fatalf("TensorF32(b.weight): %v", err)
	}
	if math.Float32bits(got[0]) != math.Float32bits(3.5) {
		t.Fatalf("b.weight[0]=%v, want 3.5", got[0])
	}
}

// TestShardPathHelpers pins the suffix-derivation helpers that the split path
// depends on: width is preserved, the count comes from the filename, and a
// non-shard path is rejected.
func TestShardPathHelpers(t *testing.T) {
	t.Run("firstShardPath preserves width", func(t *testing.T) {
		got, err := firstShardPath("/models/qwen-00005-of-00009.gguf")
		if err != nil {
			t.Fatalf("firstShardPath: %v", err)
		}
		want := "/models/qwen-00001-of-00009.gguf"
		if got != want {
			t.Fatalf("firstShardPath=%q, want %q", got, want)
		}
	})
	t.Run("shardPaths expands full set", func(t *testing.T) {
		got, err := shardPaths("/models/q-00001-of-00003.gguf", 3)
		if err != nil {
			t.Fatalf("shardPaths: %v", err)
		}
		want := []string{
			"/models/q-00001-of-00003.gguf",
			"/models/q-00002-of-00003.gguf",
			"/models/q-00003-of-00003.gguf",
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("shardPaths[%d]=%q, want %q (full=%v)", i, got[i], want[i], got)
			}
		}
	})
	t.Run("non-shard path rejected", func(t *testing.T) {
		if _, err := firstShardPath("/models/single.gguf"); err == nil {
			t.Fatal("firstShardPath accepted a non-shard path")
		}
		if _, err := shardPaths("/models/single.gguf", 1); err == nil {
			t.Fatal("shardPaths accepted a non-shard path")
		}
	})
}

// TestOpenWeightsRejectsDuplicateTensorAcrossShards guards the merge integrity
// check: if two shards claim the same tensor name, the load must fail rather
// than silently route to one shard.
func TestOpenWeightsRejectsDuplicateTensorAcrossShards(t *testing.T) {
	dir := t.TempDir()
	shard1 := writeSplitShard(t, 1, 2, 2, true, []splitTensor{
		{name: "dup.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(1, 2)},
	})
	shard2 := writeSplitShard(t, 2, 2, 2, false, []splitTensor{
		{name: "dup.weight", dims: []uint64{2}, typ: TensorF32, data: f32Payload(3, 4)},
	})
	p1 := filepath.Join(dir, "tiny-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "tiny-00002-of-00002.gguf")
	if err := os.WriteFile(p1, shard1, 0o644); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := os.WriteFile(p2, shard2, 0o644); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}
	if _, err := OpenWeights(p1); err == nil {
		t.Fatal("OpenWeights accepted duplicate tensor across shards")
	}
}
