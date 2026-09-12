package ggufload

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// deepseek41_test.go — the witness suite for issue #12903 (DeepSeek-V4.1-Flash
// GGUF header/metadata + shard-loading slice). It binds the four load-bearing
// claims of deepseek41.go to runnable evidence:
//
//  1. ConfigMapsMetadata  — the raw "<spelling>." metadata axes land in
//     model.Config field-for-field, and the Engram tables are retained on the
//     File under DeepSeek41Engram.
//  2. CanonicalArchSpellings — every sibling spelling normalizes onto the single
//     internal arch "deepseek41" and no sibling (deepseek2/llama) is
//     over-normalized.
//  3. EngramRoutesToDedicatedNamespace — an Engram tensor maps to its own
//     model.engram.<L>.* root (never a generic self_attn./mlp. name), a normal
//     MLA suffix maps correctly, and deepseek41 is NOT in the MLA+MoE loader
//     layout gate.
//  4. MalformedEngramFailsBeforeAllocation — an internally inconsistent Engram
//     declaration is refused at Config() with the offending key named, before
//     any large allocation.
//  5. SplitShardsLoadPackedExpertBytes — a two-shard synthetic split carries a
//     real Q4_K batched routed-expert blob whose per-expert byte pattern
//     survives the merge byte-for-byte (still packed, never dequantized).

// ds41Meta is the real V4.1-Flash header geometry, keyed under the RAW arch
// prefix the test passes in (proving Config reads the file's own "<arch>." keys
// while recognition normalizes the arch string).
func ds41Meta(arch string) map[string]Value {
	p := arch + "."
	meta := map[string]Value{
		"general.architecture": {Type: TypeString, Value: arch},

		p + "embedding_length":                 {Type: TypeUint64, Value: uint64(5120)},
		p + "block_count":                      {Type: TypeUint64, Value: uint64(40)},
		p + "attention.head_count":             {Type: TypeUint64, Value: uint64(128)},
		p + "attention.head_count_kv":          {Type: TypeUint64, Value: uint64(128)},
		p + "feed_forward_length":              {Type: TypeUint64, Value: uint64(18432)},
		p + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},

		// MoE FFN axis.
		p + "expert_count":                      {Type: TypeUint64, Value: uint64(384)},
		p + "expert_used_count":                 {Type: TypeUint64, Value: uint64(6)},
		p + "expert_feed_forward_length":        {Type: TypeUint64, Value: uint64(2304)},
		p + "expert_shared_count":               {Type: TypeUint64, Value: uint64(1)},
		p + "expert_shared_feed_forward_length": {Type: TypeUint64, Value: uint64(2304)},

		// MLA latent-attention axis.
		p + "attention.q_lora_rank":      {Type: TypeUint64, Value: uint64(1280)},
		p + "attention.kv_lora_rank":     {Type: TypeUint64, Value: uint64(512)},
		p + "attention.key_length_mla":   {Type: TypeUint64, Value: uint64(512)},
		p + "attention.value_length_mla": {Type: TypeUint64, Value: uint64(512)},
		p + "attention.qk_nope_head_dim": {Type: TypeUint64, Value: uint64(448)},
		p + "attention.qk_rope_head_dim": {Type: TypeUint64, Value: uint64(64)},

		// DSA-style learned indexer axis.
		p + "attention.indexer.head_count": {Type: TypeUint64, Value: uint64(32)},
		p + "attention.indexer.key_length": {Type: TypeUint64, Value: uint64(128)},
		p + "attention.indexer.top_k":      {Type: TypeUint64, Value: uint64(512)},

		// V4 grouped low-rank output + hyper-connection + compression.
		p + "attention.output_group_count":         {Type: TypeUint64, Value: uint64(8)},
		p + "attention.output_lora_rank":           {Type: TypeUint64, Value: uint64(1024)},
		p + "attention.compress_ratios":            {Type: TypeArray, Value: []Value{{Type: TypeInt32, Value: int32(1)}, {Type: TypeInt32, Value: int32(2)}, {Type: TypeInt32, Value: int32(4)}}},
		p + "hyper_connection.count":               {Type: TypeUint64, Value: uint64(4)},
		p + "hyper_connection.sinkhorn_iterations": {Type: TypeUint64, Value: uint64(20)},
		p + "hyper_connection.epsilon":             {Type: TypeFloat32, Value: float32(1e-6)},

		// Engram declaration (GUESSED namespace; i32/i64 arrays both read via IntArray).
		p + "engram_layer_ids":             {Type: TypeArray, Value: []Value{{Type: TypeInt32, Value: int32(1)}, {Type: TypeInt32, Value: int32(14)}}},
		p + "engram_num_embeddings":        {Type: TypeArray, Value: []Value{{Type: TypeInt32, Value: int32(384006168)}, {Type: TypeInt32, Value: int32(384016682)}}},
		p + "engram_max_ngram_size":        {Type: TypeUint64, Value: uint64(3)},
		p + "engram_vocab_size":            {Type: TypeUint64, Value: uint64(384016682)},
		p + "engram_n_heads":               {Type: TypeUint64, Value: uint64(8)},
		p + "engram_head_dim":              {Type: TypeUint64, Value: uint64(128)},
		p + "engram_pad_token_id":          {Type: TypeUint64, Value: uint64(0)},
		p + "engram_compressed_vocab_size": {Type: TypeUint64, Value: uint64(262144)},
	}
	return meta
}

// TestDeepSeek41GGUFConfigMapsMetadata is witness (1): the raw V4.1 metadata axes
// (keyed under the file's OWN "deepseek_v41." prefix) land on model.Config
// field-for-field, and the Engram tables are retained on the File.
func TestDeepSeek41GGUFConfigMapsMetadata(t *testing.T) {
	const raw = "deepseek_v41"
	f := &File{Metadata: ds41Meta(raw)}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	checks := []struct {
		name string
		got  int
		want int
	}{
		{"NumLayers", cfg.NumLayers, 40},
		{"HiddenSize", cfg.HiddenSize, 5120},
		{"HeadDim", cfg.HeadDim, 512},
		{"NumExperts", cfg.NumExperts, 384},
		{"NumExpertsPerTok", cfg.NumExpertsPerTok, 6},
		{"NSharedExperts", cfg.NSharedExperts, 1},
		{"MoEIntermediateSize", cfg.MoEIntermediateSize, 2304},
		{"QLoraRank", cfg.QLoraRank, 1280},
		{"QKRopeHeadDim", cfg.QKRopeHeadDim, 64},
		{"QKNopeHeadDim", cfg.QKNopeHeadDim, 448},
		{"OGroups", cfg.OGroups, 8},
		{"OLoraRank", cfg.OLoraRank, 1024},
		{"HCMult", cfg.HCMult, 4},
		{"HCSinkhornIters", cfg.HCSinkhornIters, 20},
		{"IndexNHeads", cfg.IndexNHeads, 32},
		{"IndexTopK", cfg.IndexTopK, 512},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
		}
	}
	if cfg.ModelType != "deepseek41" {
		t.Fatalf("ModelType = %q, want deepseek41 (canonicalized from %q)", cfg.ModelType, raw)
	}
	if math.Abs(cfg.HCEps-1e-6) > 1e-12 {
		t.Errorf("HCEps = %g, want 1e-6", cfg.HCEps)
	}
	if len(cfg.CompressRatios) == 0 {
		t.Errorf("CompressRatios = %v, want the declared attention.compress_ratios", cfg.CompressRatios)
	}
	if f.DeepSeek41Engram == nil {
		t.Fatal("f.DeepSeek41Engram is nil; the Engram declaration was not retained")
	}
	if got := f.DeepSeek41Engram.LayerIDs; len(got) != 2 || got[0] != 1 || got[1] != 14 {
		t.Errorf("Engram.LayerIDs = %v, want [1 14]", got)
	}
	wantEmb := []int{384006168, 384016682}
	if got := f.DeepSeek41Engram.NumEmbeddings; len(got) != 2 || got[0] != wantEmb[0] || got[1] != wantEmb[1] {
		t.Errorf("Engram.NumEmbeddings = %v, want %v", got, wantEmb)
	}
}

// TestDeepSeek41GGUFCanonicalArchSpellings is witness (2): every sibling spelling
// normalizes onto the single internal "deepseek41", while deepseek2/llama are
// passed through untouched (no sibling regression / over-normalization).
func TestDeepSeek41GGUFCanonicalArchSpellings(t *testing.T) {
	v41 := []string{
		"deepseek4", "deepseek-v4", "deepseek_v4", "deepseekv4",
		"deepseek41", "deepseek_v41", "deepseek-v41", "deepseek-v4.1", "deepseek_v41_text",
	}
	for _, spelling := range v41 {
		if got := canonicalGGUFArch(spelling); got != "deepseek41" {
			t.Errorf("canonicalGGUFArch(%q) = %q, want deepseek41", spelling, got)
		}
	}
	// Sibling families must NOT collapse into deepseek41.
	for _, spelling := range []string{"deepseek2", "deepseek-v3", "llama"} {
		if got := canonicalGGUFArch(spelling); got == "deepseek41" {
			t.Errorf("canonicalGGUFArch(%q) = deepseek41; over-normalized a sibling family", spelling)
		}
	}
	if got := canonicalGGUFArch("deepseek2"); got != "deepseek2" {
		t.Errorf("canonicalGGUFArch(deepseek2) = %q, want deepseek2", got)
	}
	if got := canonicalGGUFArch("llama"); got != "llama" {
		t.Errorf("canonicalGGUFArch(llama) = %q, want llama", got)
	}
}

// TestDeepSeek41GGUFEngramRoutesToDedicatedNamespace is witness (3): an Engram
// tensor maps to the dedicated model.engram.<L>.* root, a normal MLA suffix maps
// to the expected canonical name, and deepseek41 is kept OUT of the MLA+MoE
// loader layout gate (so the glm KV-b merge never runs for a V4 file).
func TestDeepSeek41GGUFEngramRoutesToDedicatedNamespace(t *testing.T) {
	got, ok := CanonicalTensorNameArch("blk.1.engram_table", "deepseek41")
	if !ok {
		t.Fatal("CanonicalTensorNameArch(blk.1.engram_table, deepseek41) not mapped")
	}
	want := "model.engram.1.engram_table.weight"
	if got != want {
		t.Errorf("Engram canonical name = %q, want %q", got, want)
	}
	if strings.Contains(got, "self_attn.") || strings.Contains(got, "mlp.") {
		t.Errorf("Engram mapped into a generic attention/MLP namespace: %q", got)
	}

	mla, ok := CanonicalTensorNameArch("blk.0.attn_kv.weight", "deepseek41")
	if !ok {
		t.Fatal("CanonicalTensorNameArch(blk.0.attn_kv.weight, deepseek41) not mapped")
	}
	if want := "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"; mla != want {
		t.Errorf("attn_kv canonical name = %q, want %q", mla, want)
	}

	if archUsesMLAMoELayout("deepseek41") {
		t.Error("archUsesMLAMoELayout(deepseek41) = true; the glm KV-b merge must never run for a V4 file")
	}
}

// TestDeepSeek41GGUFMalformedEngramFailsBeforeAllocation is witness (4): two
// internally inconsistent Engram declarations are refused by Config() with the
// offending key named, before any large allocation.
func TestDeepSeek41GGUFMalformedEngramFailsBeforeAllocation(t *testing.T) {
	base := func() map[string]Value {
		m := map[string]Value{
			"general.architecture":                          {Type: TypeString, Value: "deepseek_v41"},
			"deepseek_v41.embedding_length":                 {Type: TypeUint64, Value: uint64(5120)},
			"deepseek_v41.block_count":                      {Type: TypeUint64, Value: uint64(40)},
			"deepseek_v41.attention.head_count":             {Type: TypeUint64, Value: uint64(128)},
			"deepseek_v41.feed_forward_length":              {Type: TypeUint64, Value: uint64(18432)},
			"deepseek_v41.attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},
		}
		return m
	}

	t.Run("length mismatch", func(t *testing.T) {
		m := base()
		m["deepseek_v41.engram_layer_ids"] = Value{Type: TypeArray, Value: []Value{
			{Type: TypeInt32, Value: int32(1)}, {Type: TypeInt32, Value: int32(14)},
		}}
		m["deepseek_v41.engram_num_embeddings"] = Value{Type: TypeArray, Value: []Value{
			{Type: TypeInt32, Value: int32(384006168)},
		}}
		f := &File{Metadata: m}
		_, err := f.Config()
		if err == nil {
			t.Fatal("Config accepted engram_layer_ids/engram_num_embeddings length mismatch")
		}
		if !strings.Contains(err.Error(), "engram_layer_ids") || !strings.Contains(err.Error(), "engram_num_embeddings") {
			t.Errorf("error %q does not name both offending keys (engram_layer_ids / engram_num_embeddings)", err)
		}
	})

	t.Run("out-of-range layer id", func(t *testing.T) {
		m := base()
		m["deepseek_v41.engram_layer_ids"] = Value{Type: TypeArray, Value: []Value{
			{Type: TypeInt32, Value: int32(1)}, {Type: TypeInt32, Value: int32(99)},
		}}
		m["deepseek_v41.engram_num_embeddings"] = Value{Type: TypeArray, Value: []Value{
			{Type: TypeInt32, Value: int32(384006168)}, {Type: TypeInt32, Value: int32(384016682)},
		}}
		f := &File{Metadata: m}
		_, err := f.Config()
		if err == nil {
			t.Fatal("Config accepted engram layer id 99 >= block_count 40")
		}
		if !strings.Contains(err.Error(), "engram_layer_ids") {
			t.Errorf("error %q does not name engram_layer_ids", err)
		}
	})
}

// writeDeepSeek41SplitShard builds one synthetic shard of a DeepSeek-V4.1 split.
// It mirrors writeSplitShard but declares deepseek_v41 (with the axes Config()
// needs) instead of qwen2, so the shard path can compute a Config(). Only the
// config-carrying shard passes config=true.
func writeDeepSeek41SplitShard(t *testing.T, shardNo, count, totalTensors uint32, config bool, tensors []splitTensor) []byte {
	t.Helper()
	const align = 32

	var kvs []func(*bytes.Buffer)
	if config {
		kvs = append(kvs,
			func(b *bytes.Buffer) { writeKVUint32(b, "general.alignment", align) },
			func(b *bytes.Buffer) { writeKVString(b, "general.architecture", "deepseek_v41") },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.embedding_length", 256) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.block_count", 1) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.attention.head_count", 4) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.feed_forward_length", 512) },
			func(b *bytes.Buffer) { writeKVFloat32(b, "deepseek_v41.attention.layer_norm_rms_epsilon", 1e-6) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.expert_count", 4) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.expert_used_count", 1) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.expert_feed_forward_length", 256) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.expert_shared_count", 1) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.attention.q_lora_rank", 32) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.attention.kv_lora_rank", 32) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.attention.qk_nope_head_dim", 56) },
			func(b *bytes.Buffer) { writeKVUint64(b, "deepseek_v41.attention.qk_rope_head_dim", 8) },
			func(b *bytes.Buffer) { writeKVIntArrayForTest(b, "deepseek_v41.engram_layer_ids", []int32{0}) },
			func(b *bytes.Buffer) { writeKVIntArrayForTest(b, "deepseek_v41.engram_num_embeddings", []int32{1024}) },
		)
	}
	kvs = append(kvs,
		func(b *bytes.Buffer) { writeKVUint32(b, "split.no", shardNo) },
		func(b *bytes.Buffer) { writeKVUint32(b, "split.count", count) },
		func(b *bytes.Buffer) { writeKVUint32(b, "split.tensors.count", totalTensors) },
	)

	offsets := make([]uint64, len(tensors))
	off := 0
	for i, tt := range tensors {
		offsets[i] = uint64(off)
		b, err := splitTensorPayloadBytes(tt)
		if err != nil {
			t.Fatalf("payload bytes for %s: %v", tt.name, err)
		}
		off = (off + b + align - 1) / align * align
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), uint64(len(kvs)))
	for _, kv := range kvs {
		kv(&b)
	}
	for i, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, tt.dims, tt.typ, offsets[i])
	}
	padToAlignment(&b, align)
	for _, tt := range tensors {
		dataStart := b.Len()
		b.Write(tt.data)
		padToLen(&b, dataStart+align)
	}
	return b.Bytes()
}

// splitTensorPayloadBytes returns the byte length of a splitTensor's payload.
func splitTensorPayloadBytes(tt splitTensor) (int, error) {
	switch tt.typ {
	case TensorF32:
		n := 1
		for _, d := range tt.dims {
			n *= int(d)
		}
		return n * 4, nil
	default:
		return qwen3MoEPayloadBytes(tt.typ, tt.dims), nil
	}
}

// TestDeepSeek41GGUFSplitShardsLoadPackedExpertBytes is witness (5): a two-shard
// synthetic split declares deepseek_v41, places a real Q4_K batched routed-expert
// blob in shard 2 with a DISTINCT per-expert byte pattern, and the merge must
// preserve those packed bytes byte-for-byte (still Q4_K, never dequantized).
func TestDeepSeek41GGUFSplitShardsLoadPackedExpertBytes(t *testing.T) {
	const E, I, H = 4, 256, 256
	const expertName = "blk.0.ffn_gate_exps.weight"

	total := qwen3MoEPayloadBytes(TensorQ4_K, []uint64{uint64(H), uint64(I), uint64(E)})
	per := total / E
	if total%blockQ4KBytes != 0 {
		t.Fatalf("fixture payload %d is not a whole number of %d-byte super-blocks", total, blockQ4KBytes)
	}
	expertPayload := make([]byte, total)
	for x := 0; x < E; x++ {
		for i := 0; i < per; i++ {
			expertPayload[x*per+i] = expertPatternByte(x, i)
		}
	}

	shard1 := writeDeepSeek41SplitShard(t, 1, 2, 2, true, []splitTensor{
		{name: "blk.0.attn_norm.weight", dims: []uint64{4}, typ: TensorF32, data: f32Payload(1.5, 2.5, 3.5, 4.5)},
	})
	shard2 := writeDeepSeek41SplitShard(t, 2, 2, 2, false, []splitTensor{
		{name: expertName, dims: []uint64{uint64(H), uint64(I), uint64(E)}, typ: TensorQ4_K, data: expertPayload},
	})

	dir := t.TempDir()
	p1 := filepath.Join(dir, "ds41-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "ds41-00002-of-00002.gguf")
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

	// (a) Both shards' tensors are merged: the shard-2 expert and the shard-1 f32.
	if _, ok := ws.Tensor(expertName); !ok {
		t.Fatalf("merged source is missing shard-2 tensor %q", expertName)
	}
	if _, ok := ws.Tensor("blk.0.attn_norm.weight"); !ok {
		t.Fatal("merged source is missing shard-1 tensor blk.0.attn_norm.weight")
	}

	// The merged Config must resolve from shard 1's deepseek_v41 metadata.
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config on merged split: %v", err)
	}
	if cfg.ModelType != "deepseek41" {
		t.Fatalf("merged ModelType = %q, want deepseek41", cfg.ModelType)
	}

	// (b) TensorBytes returns the exact packed Q4_K bytes written, byte-for-byte —
	// proving the split load preserved the still-quantized expert blob.
	got, info, err := ws.TensorBytes(expertName)
	if err != nil {
		t.Fatalf("TensorBytes(%s): %v", expertName, err)
	}
	if info.Type != TensorQ4_K {
		t.Fatalf("TensorBytes type = %s, want Q4_K (must not be dequantized)", info.Type)
	}
	if !bytes.Equal(got, expertPayload) {
		t.Fatalf("packed expert bytes differ from source: got %d bytes, want %d (first-got=%d first-want=%d)",
			len(got), len(expertPayload), firstByte(got), expertPayload[0])
	}
	// The resident bytes must be expert 0..E-1's distinct patterns, not one repeated blob.
	for x := 0; x < E; x++ {
		if got[x*per] != expertPatternByte(x, 0) {
			t.Fatalf("expert %d region first byte = %d, want pattern %d", x, got[x*per], expertPatternByte(x, 0))
		}
	}
}

func firstByte(b []byte) byte {
	if len(b) == 0 {
		return 0
	}
	return b[0]
}
