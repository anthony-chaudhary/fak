package ggufload

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func b11870WriteStr(b *bytes.Buffer, s string) {
	_ = binary.Write(b, binary.LittleEndian, uint64(len(s)))
	_, _ = b.WriteString(s)
}

func b11870WriteKV(b *bytes.Buffer, key string, typ ValueType, val any) {
	b11870WriteStr(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(typ))
	switch typ {
	case TypeUint16:
		_ = binary.Write(b, binary.LittleEndian, val.(uint16))
	case TypeUint32:
		_ = binary.Write(b, binary.LittleEndian, val.(uint32))
	case TypeInt32:
		_ = binary.Write(b, binary.LittleEndian, val.(int32))
	case TypeFloat32:
		_ = binary.Write(b, binary.LittleEndian, math.Float32bits(val.(float32)))
	case TypeBool:
		var v uint8
		if val.(bool) {
			v = 1
		}
		_ = binary.Write(b, binary.LittleEndian, v)
	case TypeString:
		b11870WriteStr(b, val.(string))
	}
}

func b11870WriteKVArray(b *bytes.Buffer, key string, elem ValueType, vals any) {
	b11870WriteStr(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(elem))
	switch elem {
	case TypeUint32:
		xs := vals.([]uint32)
		_ = binary.Write(b, binary.LittleEndian, uint64(len(xs)))
		for _, x := range xs {
			_ = binary.Write(b, binary.LittleEndian, x)
		}
	case TypeInt32:
		xs := vals.([]int32)
		_ = binary.Write(b, binary.LittleEndian, uint64(len(xs)))
		for _, x := range xs {
			_ = binary.Write(b, binary.LittleEndian, x)
		}
	case TypeUint64:
		xs := vals.([]uint64)
		_ = binary.Write(b, binary.LittleEndian, uint64(len(xs)))
		for _, x := range xs {
			_ = binary.Write(b, binary.LittleEndian, x)
		}
	case TypeFloat32:
		xs := vals.([]float32)
		_ = binary.Write(b, binary.LittleEndian, uint64(len(xs)))
		for _, x := range xs {
			_ = binary.Write(b, binary.LittleEndian, math.Float32bits(x))
		}
	case TypeString:
		xs := vals.([]string)
		_ = binary.Write(b, binary.LittleEndian, uint64(len(xs)))
		for _, x := range xs {
			b11870WriteStr(b, x)
		}
	}
}

type b11870KVEntry struct {
	key  string
	typ  ValueType
	elem ValueType
	val  any
}

type b11870TensorSpec struct {
	name string
	dims []uint64
	typ  TensorType
	data []byte
}

func buildB11870GGUFBytes(t *testing.T, kvs []b11870KVEntry, tensors []b11870TensorSpec) []byte {
	t.Helper()
	const align = 32
	var hdr bytes.Buffer
	_ = binary.Write(&hdr, binary.LittleEndian, uint32(0x46554747))
	_ = binary.Write(&hdr, binary.LittleEndian, uint32(3))
	_ = binary.Write(&hdr, binary.LittleEndian, uint64(len(tensors)))
	_ = binary.Write(&hdr, binary.LittleEndian, uint64(len(kvs)))
	for _, kv := range kvs {
		if kv.typ == TypeArray {
			b11870WriteKVArray(&hdr, kv.key, kv.elem, kv.val)
		} else {
			b11870WriteKV(&hdr, kv.key, kv.typ, kv.val)
		}
	}
	var payload bytes.Buffer
	for _, ts := range tensors {
		for payload.Len()%align != 0 {
			payload.WriteByte(0)
		}
		off := uint64(payload.Len())
		payload.Write(ts.data)
		b11870WriteStr(&hdr, ts.name)
		_ = binary.Write(&hdr, binary.LittleEndian, uint32(len(ts.dims)))
		for _, d := range ts.dims {
			_ = binary.Write(&hdr, binary.LittleEndian, d)
		}
		_ = binary.Write(&hdr, binary.LittleEndian, uint32(ts.typ))
		_ = binary.Write(&hdr, binary.LittleEndian, off)
	}
	for hdr.Len()%align != 0 {
		hdr.WriteByte(0)
	}
	hdr.Write(payload.Bytes())
	return hdr.Bytes()
}

func b11870PublishedKVs(tokCount uint32, primes []uint64, mults []uint64, ratios []int32, tokenMap []int32, splits ...uint16) []b11870KVEntry {
	const p = "deepseek41."
	clamp := make([]float32, 40)
	for i := range clamp {
		clamp[i] = 10
	}
	toks := make([]string, tokCount)
	tokTypes := make([]int32, tokCount)
	for i := range toks {
		toks[i] = string(rune('a' + (i % 26)))
		tokTypes[i] = 1
	}
	headOffsets := make([]uint64, len(primes))
	var acc uint64
	for i, pr := range primes {
		headOffsets[i] = acc
		acc += pr
	}
	kvs := []b11870KVEntry{
		{"general.alignment", TypeUint32, 0, uint32(32)},
		{"general.architecture", TypeString, 0, "deepseek41"},
		{p + "block_count", TypeUint32, 0, uint32(40)},
		{p + "context_length", TypeUint32, 0, uint32(1048576)},
		{p + "embedding_length", TypeUint32, 0, uint32(5120)},
		{p + "attention.head_count", TypeUint32, 0, uint32(64)},
		{p + "attention.head_count_kv", TypeUint32, 0, uint32(1)},
		{p + "attention.key_length", TypeUint32, 0, uint32(512)},
		{p + "attention.value_length", TypeUint32, 0, uint32(512)},
		{p + "rope.dimension_count", TypeUint32, 0, uint32(64)},
		{p + "rope.freq_base", TypeFloat32, 0, float32(10000)},
		{p + "rope.scaling.type", TypeString, 0, "yarn"},
		{p + "rope.scaling.factor", TypeFloat32, 0, float32(16)},
		{p + "rope.scaling.original_context_length", TypeUint32, 0, uint32(65536)},
		{p + "rope.scaling.yarn_beta_fast", TypeFloat32, 0, float32(32)},
		{p + "rope.scaling.yarn_beta_slow", TypeFloat32, 0, float32(1)},
		{p + "attention.layer_norm_rms_epsilon", TypeFloat32, 0, float32(1e-20)},
		{p + "attention.q_lora_rank", TypeUint32, 0, uint32(1280)},
		{p + "attention.output_group_count", TypeUint32, 0, uint32(8)},
		{p + "attention.output_lora_rank", TypeUint32, 0, uint32(1024)},
		{p + "attention.sliding_window", TypeUint32, 0, uint32(128)},
		{p + "attention.indexer.head_count", TypeUint32, 0, uint32(32)},
		{p + "attention.indexer.key_length", TypeUint32, 0, uint32(128)},
		{p + "attention.indexer.top_k", TypeUint32, 0, uint32(512)},
		{p + "attention.compress_ratios", TypeArray, TypeInt32, ratios},
		{p + "attention.compress_rope_freq_base", TypeFloat32, 0, float32(160000)},
		{p + "attention.kv_source_layers", TypeArray, TypeUint32, []uint32{2, 8, 14, 20}},
		{p + "attention.indexer.source_layers", TypeArray, TypeUint32, []uint32{2, 8, 14, 20, 24, 28, 32, 36}},
		{p + "attention.indexer.candidate_source_layer", TypeInt32, 0, int32(20)},
		{p + "attention.indexer.candidate_top_k_blocks", TypeUint32, 0, uint32(2048)},
		{p + "attention.indexer.candidate_block_size", TypeUint32, 0, uint32(8)},
		{p + "expert_count", TypeUint32, 0, uint32(384)},
		{p + "expert_used_count", TypeUint32, 0, uint32(6)},
		{p + "expert_shared_count", TypeUint32, 0, uint32(1)},
		{p + "expert_feed_forward_length", TypeUint32, 0, uint32(2304)},
		{p + "expert_weights_scale", TypeFloat32, 0, float32(1.5)},
		{p + "expert_weights_norm", TypeBool, 0, true},
		{p + "expert_gating_func", TypeUint32, 0, uint32(4)},
		{p + "swiglu_clamp_exp", TypeArray, TypeFloat32, clamp},
		{p + "swiglu_clamp_shexp", TypeArray, TypeFloat32, clamp},
		{p + "hyper_connection.count", TypeUint32, 0, uint32(4)},
		{p + "hyper_connection.sinkhorn_iterations", TypeUint32, 0, uint32(20)},
		{p + "hyper_connection.epsilon", TypeFloat32, 0, float32(1e-6)},
		{p + "hash_layer_count", TypeUint32, 0, uint32(0)},
		{p + "ple.layers", TypeArray, TypeUint32, []uint32{1, 14}},
		{p + "ple.ngram_size", TypeUint32, 0, uint32(4)},
		{p + "ple.heads_per_ngram", TypeUint32, 0, uint32(8)},
		{p + "embedding_length_per_layer_input", TypeUint32, 0, uint32(256)},
		{p + "ple.layer_multipliers", TypeArray, TypeUint64, mults},
		{p + "ple.head_offsets", TypeArray, TypeUint64, headOffsets},
		{p + "ple.head_vocab_sizes", TypeArray, TypeUint64, primes},
		{p + "engram.pad_token_id", TypeUint32, 0, uint32(2)},
		{p + "engram.token_map", TypeArray, TypeInt32, tokenMap},
		{"tokenizer.ggml.model", TypeString, 0, "gpt2"},
		{"tokenizer.ggml.pre", TypeString, 0, "joyai-llm"},
		{"tokenizer.ggml.tokens", TypeArray, TypeString, toks},
		{"tokenizer.ggml.token_type", TypeArray, TypeInt32, tokTypes},
	}
	if len(splits) == 3 {
		kvs = append(kvs,
			b11870KVEntry{"split.no", TypeUint16, 0, splits[0]},
			b11870KVEntry{"split.count", TypeUint16, 0, splits[1]},
			b11870KVEntry{"split.tensors.count", TypeInt32, 0, int32(splits[2])},
		)
	}
	return kvs
}

func makeTestIQ4NLRow(tag byte) []byte {
	row := make([]byte, qkK/qkIQ4NL*blockIQ4NLBytes) // 8 * 18 = 144 bytes
	for b := 0; b < 8; b++ {
		off := b * blockIQ4NLBytes
		// f16 1.0 = 0x3c00, distinguishable by tag in nibble bytes
		row[off] = 0x00
		row[off+1] = 0x3c
		for j := 2; j < blockIQ4NLBytes; j++ {
			row[off+j] = tag ^ byte(b+j)
		}
	}
	return row
}

// TestDeepSeek41B11870PLEMetadataAndEngramGate parses llama.cpp b11870 ple.* metadata,
// validates global-to-local offset normalization, and maps blk.<L>.engram_gate.weight +
// per_layer_token_embd.weight onto the native forward's canonical names.
// fak-test:runtime fast est=50ms
func TestDeepSeek41B11870PLEMetadataAndEngramGate(t *testing.T) {
	primes := make([]uint64, 48)
	var sum0, sum1 int
	for i := range primes {
		primes[i] = uint64(11 + i*2)
		if i < 24 {
			sum0 += int(primes[i])
		} else {
			sum1 += int(primes[i])
		}
	}
	mults := []uint64{11, 13, 15, 17, 19, 21, 23, 25}
	ratios := make([]int32, 43)
	for i := 2; i < 20; i++ {
		ratios[i] = 2
	}
	for i := 20; i < 40; i++ {
		ratios[i] = 1
	}
	tokenMap := []int32{0, 1, 2, 3, 4, 3}
	dir := t.TempDir()
	shard1 := filepath.Join(dir, "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00002.gguf")
	shard2 := filepath.Join(dir, "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00002.gguf")
	var table bytes.Buffer
	for r := 0; r < sum0+sum1; r++ {
		table.Write(makeTestIQ4NLRow(byte((r + 1) & 0x7f)))
	}
	gateBytes := make([]byte, 5120*4*4)
	tiny := make([]byte, 16)
	s2Tensors := []b11870TensorSpec{
		{name: "blk.1.engram_gate.weight", dims: []uint64{5120, 4}, typ: TensorF32, data: gateBytes},
		{name: "blk.14.engram_gate.weight", dims: []uint64{5120, 4}, typ: TensorF32, data: gateBytes},
		{name: "blk.1.engram_kv.weight", dims: []uint64{4}, typ: TensorF32, data: tiny},
		{name: "blk.1.attn_q_a_norm.weight", dims: []uint64{4}, typ: TensorF32, data: tiny},
		{name: "blk.1.exp_probs_b_vl.bias", dims: []uint64{4}, typ: TensorF32, data: tiny},
		{name: "per_layer_token_embd.weight", dims: []uint64{256, uint64(sum0 + sum1)}, typ: TensorIQ4_NL, data: table.Bytes()},
	}
	for _, l := range []int{2, 8, 14, 20} {
		s2Tensors = append(s2Tensors,
			b11870TensorSpec{name: fmt.Sprintf("blk.%d.attn_compressor_kv.weight", l), dims: []uint64{4}, typ: TensorF32, data: tiny},
			b11870TensorSpec{name: fmt.Sprintf("blk.%d.attn_compressor_gate.weight", l), dims: []uint64{4}, typ: TensorF32, data: tiny},
			b11870TensorSpec{name: fmt.Sprintf("blk.%d.indexer.attn_k.weight", l), dims: []uint64{4}, typ: TensorF32, data: tiny},
		)
	}
	for _, l := range []int{2, 8, 14, 20, 24, 28, 32, 36} {
		s2Tensors = append(s2Tensors,
			b11870TensorSpec{name: fmt.Sprintf("blk.%d.indexer.attn_q_b.weight", l), dims: []uint64{4}, typ: TensorF32, data: tiny},
		)
	}
	if err := os.WriteFile(shard1, buildB11870GGUFBytes(t, b11870PublishedKVs(6, primes, mults, ratios, tokenMap, 0, 2, uint16(len(s2Tensors))), nil), 0o644); err != nil {
		t.Fatal(err)
	}
	s2KVs := []b11870KVEntry{
		{"general.alignment", TypeUint32, 0, uint32(32)},
		{"split.no", TypeUint16, 0, uint16(1)},
		{"split.count", TypeUint16, 0, uint16(2)},
		{"split.tensors.count", TypeInt32, 0, int32(len(s2Tensors))},
	}
	if err := os.WriteFile(shard2, buildB11870GGUFBytes(t, s2KVs, s2Tensors), 0o644); err != nil {
		t.Fatal(err)
	}

	ws, err := OpenWeights(shard1)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("File.Config: %v", err)
	}
	if cfg.DeepSeekV41 == nil {
		t.Fatal("cfg.DeepSeekV41 == nil")
	}
	eng := ws.File.DeepSeek41Engram
	if eng == nil {
		t.Fatal("ws.File.DeepSeek41Engram == nil")
	}
	if !reflect.DeepEqual(eng.LayerIDs, []int{1, 14}) {
		t.Fatalf("eng.LayerIDs = %v, want [1 14]", eng.LayerIDs)
	}
	if !reflect.DeepEqual(eng.NumEmbeddings, []int{sum0, sum1}) {
		t.Fatalf("eng.NumEmbeddings = %v, want [%d %d]", eng.NumEmbeddings, sum0, sum1)
	}
	if !reflect.DeepEqual(cfg.DeepSeekV41.KVSourceLayerIDs, []int{2, 8, 14, 20}) {
		t.Fatalf("KVSourceLayerIDs = %v", cfg.DeepSeekV41.KVSourceLayerIDs)
	}
	if cfg.DeepSeekV41.CandidateSourceLayerID != 20 || cfg.DeepSeekV41.CandidateTopKBlocks != 2048 || cfg.DeepSeekV41.CandidateBlockSize != 8 {
		t.Fatalf("candidate = (%d,%d,%d)", cfg.DeepSeekV41.CandidateSourceLayerID, cfg.DeepSeekV41.CandidateTopKBlocks, cfg.DeepSeekV41.CandidateBlockSize)
	}

	if got, ok := CanonicalTensorNameArch("blk.1.engram_gate.weight", "deepseek41"); !ok || got != "model.engram.1.engram_folded_gate.weight" {
		t.Fatalf("CanonicalTensorNameArch(engram_gate) = (%q, %v), want model.engram.1.engram_folded_gate.weight", got, ok)
	}
	if !deepseek41EngramTableTensor("per_layer_token_embd.weight") {
		t.Fatal("deepseek41EngramTableTensor(per_layer_token_embd.weight) = false, want true")
	}
	if !deepseek41HasEngramTable(ws.File) {
		t.Fatal("deepseek41HasEngramTable = false, want true")
	}

	src0, rows0, rb0, err := openV41EngramRowSource(ws, 1)
	if err != nil {
		t.Fatalf("openV41EngramRowSource(1): %v", err)
	}
	src1, rows1, rb1, err := openV41EngramRowSource(ws, 14)
	if err != nil {
		t.Fatalf("openV41EngramRowSource(14): %v", err)
	}
	if rows0 != sum0 || rows1 != sum1 || rb0 != qkK*4 || rb1 != qkK*4 {
		t.Fatalf("rows/rb = (%d,%d) (%d,%d), want (%d,%d)", rows0, rb0, rows1, rb1, sum0, sum1)
	}
	buf0 := make([]byte, rb0)
	buf1 := make([]byte, rb1)
	if _, err := src0.ReadRows(0, 1, buf0); err != nil {
		t.Fatalf("src0.ReadRows(0): %v", err)
	}
	if _, err := src1.ReadRows(0, 1, buf1); err != nil {
		t.Fatalf("src1.ReadRows(0): %v", err)
	}
	if bytes.Equal(buf0, buf1) {
		t.Fatalf("layer 1 row 0 and layer 14 row 0 decoded identically; layer 14 slice did not apply ple.head_offsets base %d", sum0)
	}

	pf := BuildManifestPreflight(ManifestPreflightInput{File: ws.File})
	if pf.Keys.Unread != 0 {
		t.Fatalf("ManifestPreflight unread keys = %d: %v", pf.Keys.Unread, pf.Render())
	}
	if pf.Tensors.Orphan != 0 {
		t.Fatalf("ManifestPreflight orphan tensors = %d: %v", pf.Tensors.Orphan, pf.Render())
	}

	if got, ok := CanonicalTensorNameArch("per_layer_token_embd.weight", "deepseek41"); !ok || got != "model.engram.per_layer_token_embd.weight" {
		t.Fatalf("CanonicalTensorNameArch(per_layer_token_embd) = (%q, %v)", got, ok)
	}

	// Corrupt ple.head_offsets must fail closed instead of mis-slicing the combined table.
	badKVs := b11870PublishedKVs(6, primes, mults, ratios, tokenMap)
	for i := range badKVs {
		if badKVs[i].key == "deepseek41.ple.head_offsets" {
			offs := append([]uint64(nil), badKVs[i].val.([]uint64)...)
			offs[1]++
			badKVs[i].val = offs
		}
	}
	badPath := filepath.Join(dir, "bad.gguf")
	if err := os.WriteFile(badPath, buildB11870GGUFBytes(t, badKVs, nil), 0o644); err != nil {
		t.Fatal(err)
	}
	badFile, err := Open(badPath)
	if err != nil {
		t.Fatalf("Open(bad): %v", err)
	}
	if _, err := badFile.Config(); err == nil {
		t.Fatal("corrupt ple.head_offsets unexpectedly succeeded")
	}
}

// fak-test:runtime fast est=50ms
func TestDeepSeek41B11870RealShard1Optional(t *testing.T) {
	p := os.Getenv("FAK_TEST_DS41_SHARD1")
	if p == "" {
		t.Skip("FAK_TEST_DS41_SHARD1 unset")
	}
	f, err := Open(p)
	if err != nil {
		t.Fatalf("Open(%s): %v", p, err)
	}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.DeepSeekV41 == nil || !reflect.DeepEqual(cfg.DeepSeekV41.KVSourceLayerIDs, []int{2, 8, 14, 20}) {
		t.Fatalf("unexpected DeepSeekV41 topology: %+v", cfg.DeepSeekV41)
	}
	eng := f.DeepSeek41Engram
	if eng == nil || eng.HeadDim != 256 || !reflect.DeepEqual(eng.LayerIDs, []int{1, 14}) || eng.PadTokenID != 2 || len(eng.TokenMap) != 129280 || len(eng.Primes) != 48 || len(eng.Offsets) != 48 || len(eng.Multipliers) != 8 {
		t.Fatalf("unexpected Engram config: %+v", eng)
	}
	gt, ok := f.GGMLTokenizer()
	if !ok || gt.Pre != "joyai-llm" || len(gt.Tokens) != 129280 {
		t.Fatalf("tokenizer ok=%v Pre=%q tokens=%d", ok, gt.Pre, len(gt.Tokens))
	}
	pf := BuildManifestPreflight(ManifestPreflightInput{File: f})
	if pf.Keys.Unread != 0 {
		t.Fatalf("unread keys on real shard1 = %d:\n%s", pf.Keys.Unread, pf.Render())
	}
}
