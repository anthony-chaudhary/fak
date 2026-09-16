package ggufload

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
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

// TestDeepSeek41GGUFReadsDS4EngramMetadata is the fail-before/pass-after
// converter fixture for #12918. Keys and element types match antirez/ds4
// deepseek41_metadata.py:115-123 at bd66c402070042bf0a79ad6ece8242de4c93680c.
func TestDeepSeek41GGUFReadsDS4EngramMetadata(t *testing.T) {
	meta := ds41Meta("deepseek41")
	for _, key := range []string{
		"engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size",
		"engram_vocab_size", "engram_n_heads", "engram_head_dim",
		"engram_pad_token_id", "engram_compressed_vocab_size",
	} {
		delete(meta, "deepseek41."+key)
	}
	meta["deepseek41.engram.encoding"] = Value{Type: TypeString, Value: "e4m3_e8m0_32_row264"}
	meta["deepseek41.engram.layer_ids"] = intArrayValue(TypeUint32, []uint64{1, 14})
	meta["deepseek41.engram.rows"] = intArrayValue(TypeUint32, []uint64{384006168, 384016682})
	meta["deepseek41.engram.compressed_vocab_size"] = Value{Type: TypeUint32, Value: uint32(99092)}
	meta["deepseek41.engram.pad_id"] = Value{Type: TypeUint32, Value: uint32(0)}
	meta["deepseek41.engram.token_map"] = intArrayValue(TypeUint32, []uint64{7, 8, 9})
	primes := make([]uint64, 48)
	for i := range primes {
		primes[i] = 16000057 + uint64(i)
	}
	meta["deepseek41.engram.primes"] = intArrayValue(TypeUint32, primes)
	meta["deepseek41.engram.multipliers"] = intArrayValue(TypeUint64, []uint64{
		35184372088831, 35184372088829, 35184372088827, 35184372088825,
		35184372088823, 35184372088821, 35184372088819, 35184372088817,
	})
	// Conflicting provisional keys prove that presence, including a valid zero
	// pad ID, never lets the legacy namespace override converter output.
	meta["deepseek41.engram_layer_ids"] = intArrayValue(TypeUint32, []uint64{2})
	meta["deepseek41.engram_num_embeddings"] = intArrayValue(TypeUint32, []uint64{99})
	meta["deepseek41.engram_pad_token_id"] = Value{Type: TypeUint32, Value: uint32(99)}
	meta["deepseek41.engram_compressed_vocab_size"] = Value{Type: TypeUint32, Value: uint32(1)}

	f := &File{Metadata: meta}
	if _, err := f.Config(); err != nil {
		t.Fatalf("Config with ds4 converter keys: %v", err)
	}
	eng := f.DeepSeek41Engram
	encoding, encodingOK := deepSeek41EngramField[string](eng, "Encoding")
	tokenMap, tokenMapOK := deepSeek41EngramField[[]int](eng, "TokenMap")
	primesGot, primesOK := deepSeek41EngramField[[]int](eng, "Primes")
	multipliers, multipliersOK := deepSeek41EngramField[[]uint64](eng, "Multipliers")
	if eng == nil || !encodingOK || encoding != "e4m3_e8m0_32_row264" || eng.MaxNgramSize != 4 || eng.NHeads != 8 ||
		len(eng.LayerIDs) != 2 || len(eng.NumEmbeddings) != 2 || !tokenMapOK || len(tokenMap) != 3 ||
		!primesOK || len(primesGot) != 48 || !multipliersOK || len(multipliers) != 8 || multipliers[0] != 35184372088831 ||
		eng.PadTokenID != 0 || eng.CompressedVocabSize != 99092 {
		t.Fatalf("ds4 Engram metadata not retained: %+v", eng)
	}

	delete(meta, "deepseek41.engram.layer_ids")
	if _, err := (&File{Metadata: meta}).Config(); err == nil || !strings.Contains(err.Error(), "deepseek41.engram.layer_ids") {
		t.Fatalf("partial converter metadata error = %v, want missing nested layer_ids despite legacy fallback", err)
	}
	meta["deepseek41.engram.layer_ids"] = intArrayValue(TypeUint32, []uint64{1, 14})

	delete(meta, "deepseek41.engram.primes")
	if _, err := (&File{Metadata: meta}).Config(); err == nil || !strings.Contains(err.Error(), "deepseek41.engram.primes") {
		t.Fatalf("missing converter primes error = %v, want named key", err)
	}
}

// deepSeek41EngramField keeps this symptom witness source-compatible with the
// parent DeepSeek41Engram type. Missing fields are therefore a behavioral RED
// instead of an unrelated parent compilation failure.
func deepSeek41EngramField[T any](eng *DeepSeek41Engram, name string) (T, bool) {
	var zero T
	if eng == nil {
		return zero, false
	}
	field := reflect.ValueOf(eng).Elem().FieldByName(name)
	if !field.IsValid() || !field.CanInterface() {
		return zero, false
	}
	value, ok := field.Interface().(T)
	return value, ok
}

func intArrayValue(kind ValueType, values []uint64) Value {
	items := make([]Value, len(values))
	for i, value := range values {
		switch kind {
		case TypeUint32:
			items[i] = Value{Type: kind, Value: uint32(value)}
		case TypeUint64:
			items[i] = Value{Type: kind, Value: value}
		}
	}
	return Value{Type: TypeArray, Value: items}
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

// TestDeepSeek41GGUFMoEOnlyHeaderDerivesConfig is the fail-before/pass-after
// witness for issue #12979. A real vcruz305 deepseek41 GGUF is MoE-only: it
// declares expert_feed_forward_length (2304) and
// expert_shared_feed_forward_length but NO dense feed_forward_length. Before the
// fix Config() called requiredInt(p+"feed_forward_length") unconditionally and
// refused every published artifact with
// "gguf: missing deepseek41.feed_forward_length" before applyDeepSeek41Config
// ran, so none of the V4.1 axes were reached.
//
// This fixture is the vcruz305 Engram key set with the dense FFN key REMOVED,
// which is exactly how the real converter materializes. Config() must now derive
// the dense width from the expert width, retain the Engram declaration, and reach
// every V4.1 axis. A header declaring NEITHER width must still fail loud naming
// the dense key.
func TestDeepSeek41GGUFMoEOnlyHeaderDerivesConfig(t *testing.T) {
	meta, p := vcruzEngramMeta()
	delete(meta, p+"feed_forward_length")
	meta[p+"expert_count"] = Value{Type: TypeUint64, Value: uint64(384)}
	meta[p+"expert_used_count"] = Value{Type: TypeUint64, Value: uint64(6)}
	meta[p+"expert_feed_forward_length"] = Value{Type: TypeUint64, Value: uint64(2304)}
	meta[p+"expert_shared_count"] = Value{Type: TypeUint64, Value: uint64(1)}
	meta[p+"expert_shared_feed_forward_length"] = Value{Type: TypeUint64, Value: uint64(2304)}

	f := &File{Metadata: meta}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config on MoE-only deepseek41 header: %v (must derive the dense FFN width)", err)
	}
	if cfg.ModelType != "deepseek41" {
		t.Fatalf("ModelType = %q, want deepseek41", cfg.ModelType)
	}
	// The dense width is derived from the expert width, and the MoE axis is read.
	if cfg.IntermediateSize != 2304 {
		t.Errorf("IntermediateSize = %d, want 2304 (derived from expert_feed_forward_length)", cfg.IntermediateSize)
	}
	if cfg.MoEIntermediateSize != 2304 {
		t.Errorf("MoEIntermediateSize = %d, want 2304", cfg.MoEIntermediateSize)
	}
	// The V4.1-specific axes were actually reached (applyDeepSeek41Config ran).
	if cfg.NumExperts != 384 || cfg.NumExpertsPerTok != 6 || cfg.NSharedExperts != 1 {
		t.Errorf("MoE axes = experts:%d topk:%d shared:%d, want 384/6/1", cfg.NumExperts, cfg.NumExpertsPerTok, cfg.NSharedExperts)
	}
	if cfg.SharedIntermediateSize != 2304 {
		t.Errorf("SharedIntermediateSize = %d, want 2304", cfg.SharedIntermediateSize)
	}
	if f.DeepSeek41Engram == nil {
		t.Fatal("f.DeepSeek41Engram is nil; the Engram declaration was not retained")
	}
	if len(f.DeepSeek41Engram.LayerIDs) != 2 || f.DeepSeek41Engram.NHeads != 8 {
		t.Errorf("Engram not retained: layer_ids=%v n_heads=%d", f.DeepSeek41Engram.LayerIDs, f.DeepSeek41Engram.NHeads)
	}

	// Fail-loud preservation: a header with NEITHER dense nor expert FFN width must
	// still be refused, and the refusal must name the dense key.
	delete(meta, p+"expert_feed_forward_length")
	delete(meta, p+"expert_shared_feed_forward_length")
	if _, err := (&File{Metadata: meta}).Config(); err == nil {
		t.Fatal("Config accepted a header with neither dense nor expert feed_forward_length")
	} else if !strings.Contains(err.Error(), "feed_forward_length") {
		t.Fatalf("refusal %q does not name feed_forward_length", err)
	}
}

// TestRequiredDenseFFNLendDoesNotLeakToDenseArchs is the regression guard for the
// issue #13010 out-of-scope fence: "Do NOT relax any guard that would let an
// incompatible artifact silently load." requiredDenseFFNLen derives the dense FFN
// width from the expert width for a MoE-only deepseek41 artifact (correct), but its
// derive branch keys only on the PRESENCE of "<p>expert_feed_forward_length" — it
// never checks that the file is a MoE architecture. Before the fix, a plain dense
// arch (e.g. gemma3, llama) that happened to carry an expert width but no dense
// width silently admitted and projected IntermediateSize from the EXPERT width
// instead of failing loud on its own missing dense key.
//
// The guard is deliberately narrow: deepseek41/MoE spellings still derive, and a
// genuinely dense arch that declares no expert width still fails loud naming the
// dense key.
func TestRequiredDenseFFNLendDoesNotLeakToDenseArchs(t *testing.T) {
	// A dense (non-MoE) arch with hidden/layers/heads/rms but NO dense
	// feed_forward_length, carrying only an expert width. It must NOT admit.
	dense := func(arch string) map[string]Value {
		p := arch + "."
		return map[string]Value{
			"general.architecture":                 {Type: TypeString, Value: arch},
			p + "embedding_length":                 {Type: TypeUint64, Value: uint64(4096)},
			p + "block_count":                      {Type: TypeUint64, Value: uint64(32)},
			p + "attention.head_count":             {Type: TypeUint64, Value: uint64(32)},
			p + "attention.head_count_kv":          {Type: TypeUint64, Value: uint64(8)},
			p + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-5)},
			p + "expert_feed_forward_length":       {Type: TypeUint64, Value: uint64(2304)},
		}
	}

	for _, arch := range []string{"gemma3", "llama", "qwen2"} {
		meta := dense(arch)
		_, err := (&File{Metadata: meta}).Config()
		if err == nil {
			t.Fatalf("arch %q: Config admitted a dense header that declared only an expert width; the derive leaked outside the MoE/deepseek41 gate", arch)
		}
		if !strings.Contains(err.Error(), "feed_forward_length") {
			t.Fatalf("arch %q: refusal %q does not name feed_forward_length", arch, err)
		}
	}

	// The legitimate derive must still work for a deepseek41 MoE-only header.
	meta, p := vcruzEngramMeta()
	delete(meta, p+"feed_forward_length")
	meta[p+"expert_count"] = Value{Type: TypeUint64, Value: uint64(384)}
	meta[p+"expert_used_count"] = Value{Type: TypeUint64, Value: uint64(6)}
	meta[p+"expert_feed_forward_length"] = Value{Type: TypeUint64, Value: uint64(2304)}
	meta[p+"expert_shared_count"] = Value{Type: TypeUint64, Value: uint64(1)}
	meta[p+"expert_shared_feed_forward_length"] = Value{Type: TypeUint64, Value: uint64(2304)}
	cfg, err := (&File{Metadata: meta}).Config()
	if err != nil {
		t.Fatalf("deepseek41 MoE-only header must still derive its dense width: %v", err)
	}
	if cfg.IntermediateSize != 2304 {
		t.Errorf("deepseek41 MoE-only IntermediateSize = %d, want 2304 (derived)", cfg.IntermediateSize)
	}
}

// vcruzQ2KReceiptMeta reconstructs the EXACT header shape the physical loader
// receipt recorded for the published vcruz305/DeepSeek-V4.1-Flash-GGUF Q2_K
// shard, built from a clean public HEAD at commit f75ae93f6
// (docs/benchmarks/receipts/deepseek-v4-loader-admission-strix3-20260913-hardware.json
// "observed_metadata"). That run REFUSED an otherwise-loadable artifact with
// `gguf: missing deepseek41.feed_forward_length`. The observed set is used
// verbatim: an MoE-only deepseek41 header (expert_count/used/shared/ffn), the
// vcruz Engram dialect (nested head_count/key_length/max_ngram_size, INT32
// layer_ids), the indexer + compress_ratios + hyper_connection axes, and NO
// dense feed_forward_length and NO ds4 engram.encoding/rows/compressed_vocab_size.
func vcruzQ2KReceiptMeta() map[string]Value {
	const arch = "deepseek41"
	p := arch + "."
	primes := make([]uint64, 2*(4-1)*8)
	for i := range primes {
		primes[i] = 16000057 + uint64(i)
	}
	return map[string]Value{
		"general.architecture":                 {Type: TypeString, Value: arch},
		p + "embedding_length":                 {Type: TypeUint64, Value: uint64(5120)},
		p + "block_count":                      {Type: TypeUint64, Value: uint64(40)},
		p + "attention.head_count":             {Type: TypeUint64, Value: uint64(128)},
		p + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},
		// No p+"feed_forward_length": the publisher emits none (the receipt's
		// dense_feed_forward_length_present=false).

		// MoE FFN axis (observed).
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

		// Indexer axis present (observed attention.indexer.present=true).
		p + "attention.indexer.head_count": {Type: TypeUint64, Value: uint64(32)},
		p + "attention.indexer.key_length": {Type: TypeUint64, Value: uint64(128)},
		p + "attention.indexer.top_k":      {Type: TypeUint64, Value: uint64(512)},

		// Compression ratios present (observed) + hyper-connection axis.
		p + "attention.compress_ratios": vcruzI32Array([]int32{1, 2, 4}),
		p + "hyper_connection.count":    {Type: TypeUint64, Value: uint64(4)},
		p + "hyper_connection.epsilon":  {Type: TypeFloat32, Value: float32(1e-6)},

		// vcruz Engram dialect (observed); NO encoding/rows/compressed_vocab_size.
		p + "engram.layer_ids":      vcruzI32Array([]int32{1, 14}),
		p + "engram.head_count":     {Type: TypeUint32, Value: uint32(8)},
		p + "engram.key_length":     {Type: TypeUint32, Value: uint32(256)},
		p + "engram.max_ngram_size": {Type: TypeUint32, Value: uint32(4)},
		p + "engram.primes":         vcruzU64Array(primes),
		p + "engram.multipliers":    vcruzU64Array([]uint64{35184372088831, 35184372088829}),
		p + "engram.offsets":        vcruzU64Array([]uint64{0, 1}),
		p + "engram.token_map":      vcruzI32Array([]int32{7, 8, 9}),
		p + "engram.pad_id":         {Type: TypeUint32, Value: uint32(0)},
	}
}

// TestDeepSeek41Q2KAdmission is the acceptance witness the #13010 tracking issue
// names. It reproduces the EXACT published-artifact header that physical commit
// f75ae93f6 REFUSED on Strix Halo (`missing deepseek41.feed_forward_length`) and
// asserts the current loader admits it and reaches every V4.1 axis. It is the
// regression guard for issue #13010's dense-FFN derive, and it fails loud if the
// conditional derive or the Engram converter-key read is ever removed.
func TestDeepSeek41Q2KAdmission(t *testing.T) {
	meta := vcruzQ2KReceiptMeta()
	f := &File{Metadata: meta}

	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config on the exact f75ae93f6 Q2_K header: %v (the physical receipt refused here; current trunk must admit)", err)
	}
	if cfg.ModelType != "deepseek41" {
		t.Errorf("ModelType = %q, want deepseek41", cfg.ModelType)
	}
	if cfg.NumLayers != 40 || cfg.HiddenSize != 5120 {
		t.Errorf("geometry = layers:%d hidden:%d, want 40/5120", cfg.NumLayers, cfg.HiddenSize)
	}
	// The dense width is derived from the expert width (the exact f75ae93f6
	// refusal site), and the MoE axes are actually read.
	if cfg.IntermediateSize != 2304 {
		t.Errorf("IntermediateSize = %d, want 2304 (derived from expert_feed_forward_length)", cfg.IntermediateSize)
	}
	if cfg.NumExperts != 384 || cfg.NumExpertsPerTok != 6 || cfg.NSharedExperts != 1 {
		t.Errorf("MoE axes = experts:%d topk:%d shared:%d, want 384/6/1", cfg.NumExperts, cfg.NumExpertsPerTok, cfg.NSharedExperts)
	}
	// The V4.1-specific axes the receipt recorded as present were reached.
	if len(cfg.CompressRatios) != 3 {
		t.Errorf("CompressRatios = %v, want 3 entries from attention.compress_ratios", cfg.CompressRatios)
	}
	if cfg.IndexNHeads != 32 || cfg.IndexHeadDim != 128 || cfg.IndexTopK != 512 {
		t.Errorf("indexer axes = n_heads:%d head_dim:%d top_k:%d, want 32/128/512", cfg.IndexNHeads, cfg.IndexHeadDim, cfg.IndexTopK)
	}
	if cfg.HCMult != 4 {
		t.Errorf("HCMult = %d, want 4", cfg.HCMult)
	}
	if f.DeepSeek41Engram == nil {
		t.Fatal("f.DeepSeek41Engram is nil; the vcruz Engram declaration was not retained")
	}
	if len(f.DeepSeek41Engram.LayerIDs) != 2 || f.DeepSeek41Engram.NHeads != 8 || f.DeepSeek41Engram.HeadDim != 256 {
		t.Errorf("Engram not retained: layer_ids=%v n_heads=%d head_dim=%d",
			f.DeepSeek41Engram.LayerIDs, f.DeepSeek41Engram.NHeads, f.DeepSeek41Engram.HeadDim)
	}

	// Negative control: without ANY FFN width the header must still refuse, and
	// the refusal must name the dense key (no silent derive, no zero-fill).
	delete(meta, "deepseek41.expert_feed_forward_length")
	delete(meta, "deepseek41.expert_shared_feed_forward_length")
	if _, err := (&File{Metadata: meta}).Config(); err == nil {
		t.Fatal("Config admitted a header with neither dense nor expert feed_forward_length")
	} else if !strings.Contains(err.Error(), "feed_forward_length") {
		t.Fatalf("refusal %q does not name feed_forward_length", err)
	}
}

// vcruzI32Array builds a GGUF INT32 metadata array (the width the vcruz305
// converter writes for engram.layer_ids and engram.token_map).
func vcruzI32Array(values []int32) Value {
	items := make([]Value, len(values))
	for i, v := range values {
		items[i] = Value{Type: TypeInt32, Value: v}
	}
	return Value{Type: TypeArray, Value: items}
}

// vcruzU64Array builds a GGUF UINT64 metadata array (engram.primes/offsets/multipliers).
func vcruzU64Array(values []uint64) Value {
	items := make([]Value, len(values))
	for i, v := range values {
		items[i] = Value{Type: TypeUint64, Value: v}
	}
	return Value{Type: TypeArray, Value: items}
}

// vcruzEngramMeta is the Engram header shape a real vcruz305-converted
// deepseek41 GGUF writes: nine nested {arch}.engram.* keys, at their real value
// types (layer_ids/token_map INT32, primes/offsets/multipliers UINT64,
// head_count/key_length/max_ngram_size/pad_id UINT32), and NO
// encoding/rows/compressed_vocab_size. The primes count obeys the published
// hash geometry len(layer_ids)*(max_ngram_size-1)*head_count so the guard sees a
// consistent declaration.
func vcruzEngramMeta() (map[string]Value, string) {
	const arch = "deepseek41"
	p := arch + "."
	primes := make([]uint64, 2*(4-1)*8)
	for i := range primes {
		primes[i] = 16000057 + uint64(i)
	}
	meta := map[string]Value{
		"general.architecture":                 {Type: TypeString, Value: arch},
		p + "embedding_length":                 {Type: TypeUint64, Value: uint64(5120)},
		p + "block_count":                      {Type: TypeUint64, Value: uint64(40)},
		p + "attention.head_count":             {Type: TypeUint64, Value: uint64(128)},
		p + "feed_forward_length":              {Type: TypeUint64, Value: uint64(18432)},
		p + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},

		p + "engram.layer_ids":      vcruzI32Array([]int32{1, 14}),
		p + "engram.head_count":     {Type: TypeUint32, Value: uint32(8)},
		p + "engram.key_length":     {Type: TypeUint32, Value: uint32(256)},
		p + "engram.max_ngram_size": {Type: TypeUint32, Value: uint32(4)},
		p + "engram.primes":         vcruzU64Array(primes),
		p + "engram.multipliers":    vcruzU64Array([]uint64{35184372088831, 35184372088829}),
		p + "engram.offsets":        vcruzU64Array([]uint64{0, 1}),
		p + "engram.token_map":      vcruzI32Array([]int32{7, 8, 9}),
		p + "engram.pad_id":         {Type: TypeUint32, Value: uint32(0)},
	}
	return meta, p
}

// TestDeepSeek41GGUFReadsVcruzEngramMetadata is the fail-before/pass-after fixture
// for the vcruz305 DeepSeek-V4.1-Flash-GGUF converter dialect. Before the
// reconciliation the loader refused a valid file with
// "deepseek41.engram.layer_ids must be a uint32 array" (the real converter writes
// INT32) and left all four engram_* tensors unmapped.
func TestDeepSeek41GGUFReadsVcruzEngramMetadata(t *testing.T) {
	meta, _ := vcruzEngramMeta()
	f := &File{Metadata: meta}
	if _, err := f.Config(); err != nil {
		t.Fatalf("Config with vcruz converter keys: %v", err)
	}
	eng := f.DeepSeek41Engram
	if eng == nil {
		t.Fatal("f.DeepSeek41Engram is nil; the vcruz Engram declaration was not retained")
	}
	if len(eng.LayerIDs) != 2 || eng.LayerIDs[0] != 1 || eng.LayerIDs[1] != 14 {
		t.Errorf("Engram.LayerIDs = %v, want [1 14]", eng.LayerIDs)
	}
	if eng.NHeads != 8 {
		t.Errorf("Engram.NHeads = %d, want 8 (engram.head_count)", eng.NHeads)
	}
	if eng.HeadDim != 256 {
		t.Errorf("Engram.HeadDim = %d, want 256 (engram.key_length)", eng.HeadDim)
	}
	if eng.MaxNgramSize != 4 {
		t.Errorf("Engram.MaxNgramSize = %d, want 4 (engram.max_ngram_size)", eng.MaxNgramSize)
	}
	if len(eng.TokenMap) != 3 || eng.TokenMap[0] != 7 {
		t.Errorf("Engram.TokenMap = %v, want [7 8 9]", eng.TokenMap)
	}
	if len(eng.Primes) != 48 {
		t.Errorf("Engram.Primes has %d entries, want 48", len(eng.Primes))
	}
	if len(eng.Multipliers) != 2 || eng.Multipliers[0] != 35184372088831 {
		t.Errorf("Engram.Multipliers = %v, want the 47-bit odd constants", eng.Multipliers)
	}
	if len(eng.Offsets) != 2 || eng.Offsets[0] != 0 || eng.Offsets[1] != 1 {
		t.Errorf("Engram.Offsets = %v, want [0 1]", eng.Offsets)
	}
	if eng.PadTokenID != 0 {
		t.Errorf("Engram.PadTokenID = %d, want 0", eng.PadTokenID)
	}
	// The vcruz dialect writes no encoding/rows/compressed_vocab_size; they must
	// not be fabricated from the absent keys.
	if eng.Encoding != "" || eng.CompressedVocabSize != 0 {
		t.Errorf("vcruz Engram fabricated ds4-only fields: encoding=%q compressed=%d", eng.Encoding, eng.CompressedVocabSize)
	}
}

// TestDeepSeek41GGUFVcruzEngramTensorSuffixes proves the four real vcruz engram_*
// tensor suffixes map into the dedicated model.engram.<L>.* namespace, and never
// into a generic self_attn./mlp. name (which would let an Engram table fall
// through to the wrong forward).
//
// The three projection-side suffixes (engram_wkv / engram_q / engram_k) are
// normalized onto the exact canonical leaves the reduced native forward consumes
// (engram_kv / engram_q_norm / engram_k_norm) - see
// internal/model/v41_forward.go:295 and v41_forward_engram.go:212. The Engram
// table (engram_embd) keeps its own leaf, because the forward reaches it through
// the packed-row source rather than a named projection tensor.
func TestDeepSeek41GGUFVcruzEngramTensorSuffixes(t *testing.T) {
	want := map[string]string{
		"engram_embd": "model.engram.1.engram_embd.weight",
		"engram_k":    "model.engram.1.engram_k_norm.weight",
		"engram_q":    "model.engram.1.engram_q_norm.weight",
		"engram_wkv":  "model.engram.1.engram_kv.weight",
	}
	for suffix, wantName := range want {
		got, ok := CanonicalTensorNameArch("blk.1."+suffix+".weight", "deepseek41")
		if !ok {
			t.Errorf("CanonicalTensorNameArch(blk.1.%s.weight, deepseek41) not mapped", suffix)
			continue
		}
		if got != wantName {
			t.Errorf("suffix %s canonical name = %q, want %q", suffix, got, wantName)
		}
		if strings.Contains(got, "self_attn.") || strings.Contains(got, "mlp.") {
			t.Errorf("suffix %s mapped into a generic attention/MLP namespace: %q", suffix, got)
		}
	}
}

// TestDeepSeek41GGUFVcruzMalformedFailsLoud proves the reconciled reader still
// fails closed: a vcruz file with inconsistent hash geometry, or with a
// non-integer array where an integer array is required, is refused with the
// offending key named.
func TestDeepSeek41GGUFVcruzMalformedFailsLoud(t *testing.T) {
	t.Run("inconsistent hash geometry", func(t *testing.T) {
		meta, _ := vcruzEngramMeta()
		// Drop primes below the required len(layer_ids)*(max_ngram-1)*head_count.
		meta["deepseek41.engram.primes"] = vcruzU64Array([]uint64{16000057})
		if _, err := (&File{Metadata: meta}).Config(); err == nil || !strings.Contains(err.Error(), "hash geometry") {
			t.Fatalf("Config error = %v, want inconsistent hash geometry refusal", err)
		}
	})
	t.Run("non-integer array", func(t *testing.T) {
		meta, _ := vcruzEngramMeta()
		meta["deepseek41.engram.layer_ids"] = Value{Type: TypeArray, Value: []Value{{Type: TypeString, Value: "not-an-int"}}}
		if _, err := (&File{Metadata: meta}).Config(); err == nil || !strings.Contains(err.Error(), "deepseek41.engram.layer_ids") {
			t.Fatalf("Config error = %v, want integer-array refusal naming layer_ids", err)
		}
	})
}

// deepSeek41V41RealSuffixes is the GGUF per-layer suffix set the real converted
// DeepSeek-V4.1 GGUF emits, each paired with the exact canonical name the
// "deepseek41" arch MUST resolve it to. Kept as the single source of truth for
// the suffix-map tests below so every one of the eight is exercised identically.
var deepSeek41V41RealSuffixes = []struct {
	suffix string
	want   func(l int) string
}{
	{"attn_kv_a_norm.weight", func(l int) string { return layerName(l, "self_attn.kv_a_layernorm.weight") }},
	{"attn_compressor_gate.weight", func(l int) string { return layerName(l, "self_attn.compressor.wgate.weight") }},
	{"attn_compressor_kv.weight", func(l int) string { return layerName(l, "self_attn.compressor.wkv.weight") }},
	{"attn_compressor_norm.weight", func(l int) string { return layerName(l, "self_attn.compressor.norm.weight") }},
	{"attn_sinks.weight", func(l int) string { return fmt.Sprintf("model.layers.%d.attn.attn_sink", l) }},
	{"exp_probs_b_vl.bias", func(l int) string { return layerName(l, "mlp.gate.e_score_correction_bias_vl") }},
	{"indexer.attn_k.weight", func(l int) string { return layerName(l, "self_attn.indexer.wk.weight") }},
	{"indexer.k_norm.weight", func(l int) string { return layerName(l, "self_attn.indexer.k_norm.weight") }},
}

// layerName formats a canonical per-layer name for layer l under a suffix.
func layerName(layer int, suffix string) string {
	return fmt.Sprintf("model.layers.%d.%s", layer, suffix)
}

// canonicalFor maps a raw "blk.<L>.<suffix>" GGUF name through
// CanonicalTensorNameArch for arch "deepseek41", failing the test if the map
// refuses a name the spec says must resolve.
func canonicalFor(t *testing.T, ggufName string) string {
	t.Helper()
	got, ok := CanonicalTensorNameArch(ggufName, "deepseek41")
	if !ok {
		t.Fatalf("CanonicalTensorNameArch(%q, deepseek41) = ok=false; the real V4.1 suffix must resolve", ggufName)
	}
	return got
}

// TestDeepSeek41GGUFV41RealArtifactSuffixMap binds the eight per-layer suffixes
// the real converted DeepSeek-V4.1 GGUF emits to their exact canonical names.
// Each suffix is asserted at layer 0 AND a non-zero layer (7) so the layer index
// is genuinely substituted rather than hardcoded.
func TestDeepSeek41GGUFV41RealArtifactSuffixMap(t *testing.T) {
	for _, tc := range deepSeek41V41RealSuffixes {
		for _, layer := range []int{0, 7} {
			ggufName := fmt.Sprintf("blk.%d.%s", layer, tc.suffix)
			got := canonicalFor(t, ggufName)
			want := tc.want(layer)
			if got != want {
				t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = %q, want %q", ggufName, got, want)
			}
		}
	}
}

// TestDeepSeek41GGUFV41FixtureResolvesEveryTensor builds a real-artifact-shaped
// File from the existing vcruzQ2KReceiptMeta header and proves EVERY one of the
// eight V4.1 per-layer suffixes resolves through the arch map for both a layer
// the header declares and a non-declared layer, so a single omitted map arm
// cannot hide behind the table test.
func TestDeepSeek41GGUFV41FixtureResolvesEveryTensor(t *testing.T) {
	f := &File{Metadata: vcruzQ2KReceiptMeta()}
	if _, err := f.Config(); err != nil {
		t.Fatalf("Config on the vcruz Q2_K fixture: %v", err)
	}
	// No frozen table-length assertion here: the table's 8 members are the
	// artifact inventory, and every one is exercised below. Asserting the count
	// would pin today's total rather than the relation under test.
	for _, layer := range []int{0, 7} {
		for _, tc := range deepSeek41V41RealSuffixes {
			ggufName := fmt.Sprintf("blk.%d.%s", layer, tc.suffix)
			got, ok := CanonicalTensorNameArch(ggufName, "deepseek41")
			if !ok {
				t.Errorf("fixture tensor %q did not resolve under deepseek41", ggufName)
				continue
			}
			if want := tc.want(layer); got != want {
				t.Errorf("fixture tensor %q = %q, want %q", ggufName, got, want)
			}
		}
	}
}

// TestDeepSeek41GGUFV41SuffixMapEdges probes the adversarial edges of the V4.1
// suffix map: sibling indexer names must not collide, the new kv_a norm must not
// shadow or be shadowed by the legacy kv_norm (and must never become the kv_a
// projection), an unknown suffix must stay refused, and no compressor/indexer/
// sink arm may produce a kv_b_proj leaf.
func TestDeepSeek41GGUFV41SuffixMapEdges(t *testing.T) {
	t.Run("indexer names do not collide", func(t *testing.T) {
		names := map[string]string{
			"indexer.attn_k.weight":   canonicalFor(t, "blk.0.indexer.attn_k.weight"),
			"indexer.k_norm.weight":   canonicalFor(t, "blk.0.indexer.k_norm.weight"),
			"indexer.attn_q_b.weight": canonicalFor(t, "blk.0.indexer.attn_q_b.weight"),
			"indexer.proj.weight":     canonicalFor(t, "blk.0.indexer.proj.weight"),
		}
		seen := map[string]string{}
		for suffix, canonical := range names {
			if canonical == "" {
				t.Errorf("indexer suffix %s resolved to an empty canonical name", suffix)
			}
			if prev, dup := seen[canonical]; dup {
				t.Errorf("indexer suffixes %s and %s collide on canonical name %q", prev, suffix, canonical)
			}
			seen[canonical] = suffix
		}
	})

	t.Run("kv_a norm and legacy kv_norm agree and are not the projection", func(t *testing.T) {
		newNorm := canonicalFor(t, "blk.0.attn_kv_a_norm.weight")
		legacyNorm := canonicalFor(t, "blk.0.attn_kv_norm.weight")
		if newNorm != legacyNorm {
			t.Errorf("attn_kv_a_norm canonical = %q, legacy attn_kv_norm canonical = %q; both must resolve to the same kv norm", newNorm, legacyNorm)
		}
		if want := "model.layers.0.self_attn.kv_a_layernorm.weight"; newNorm != want {
			t.Errorf("kv norm canonical = %q, want %q", newNorm, want)
		}
		if proj := "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"; newNorm == proj {
			t.Errorf("kv norm canonical = %q; a norm must never map to the kv_a projection", proj)
		}
	})

	t.Run("unknown suffix stays refused", func(t *testing.T) {
		if got, ok := CanonicalTensorNameArch("blk.0.totally_unknown_suffix.weight", "deepseek41"); ok {
			t.Errorf("unknown suffix resolved to %q; the deepseek41 map must not be a catch-all", got)
		}
	})

	t.Run("no kv_b_proj leaf in compressor indexer sink names", func(t *testing.T) {
		for _, suffix := range []string{
			"attn_compressor_gate.weight",
			"attn_compressor_kv.weight",
			"attn_compressor_norm.weight",
			"indexer.attn_k.weight",
			"indexer.k_norm.weight",
			"attn_sinks.weight",
		} {
			got := canonicalFor(t, "blk.0."+suffix)
			if strings.Contains(got, "kv_b_proj") {
				t.Errorf("suffix %s canonical name %q carries a kv_b_proj leaf; the V4 single-attn_kv invariant must hold", suffix, got)
			}
		}
	})
}

// TestDeepSeek41GGUFV41CompressorStaysForwardClassified is the ticket's required
// negative arm. The admission functions live in internal/model (a different
// package), so this is a comment-documented structural assertion instead: the
// loader map resolving a compressor tensor is NOT sufficient to make the layer
// admissible, because the compressor canonical names live under the
// self_attn.compressor.* namespace that the reduced forward's supported tensor
// set does not contain.
func TestDeepSeek41GGUFV41CompressorStaysForwardClassified(t *testing.T) {
	compressorNames := []string{
		canonicalFor(t, "blk.0.attn_compressor_gate.weight"),
		canonicalFor(t, "blk.0.attn_compressor_kv.weight"),
		canonicalFor(t, "blk.0.attn_compressor_norm.weight"),
	}
	for _, name := range compressorNames {
		if !strings.Contains(name, "self_attn.compressor.") {
			t.Errorf("compressor canonical name %q is not under the self_attn.compressor.* namespace; a reduced forward could fall through and wrongly admit the stage", name)
		}
	}
}

// TestDeepSeek41GGUFV41SuffixMapClassifierConsistency asserts the namespace
// classification that keeps each V4.1 tensor distinguishable from a generic
// MLP/attention fall-through, without calling the forward package: compressor
// and indexer names land under a self_attn. namespace, while attn_sinks lands
// under the raw attn. namespace. It also re-proves the unknown-suffix arm that
// guards against the map becoming a catch-all for a non-V4 arch.
func TestDeepSeek41GGUFV41SuffixMapClassifierConsistency(t *testing.T) {
	subSelfAttn := []string{
		canonicalFor(t, "blk.0.attn_compressor_gate.weight"),
		canonicalFor(t, "blk.0.attn_compressor_kv.weight"),
		canonicalFor(t, "blk.0.attn_compressor_norm.weight"),
		canonicalFor(t, "blk.0.indexer.attn_k.weight"),
		canonicalFor(t, "blk.0.indexer.k_norm.weight"),
	}
	for _, name := range subSelfAttn {
		if !strings.Contains(name, "self_attn.") {
			t.Errorf("canonical name %q is not under a self_attn. namespace", name)
		}
		if !strings.HasPrefix(name, "model.layers.0.") {
			t.Errorf("canonical name %q is not rooted at model.layers.0.", name)
		}
	}

	sink := canonicalFor(t, "blk.0.attn_sinks.weight")
	if !strings.Contains(sink, ".attn.") {
		t.Errorf("attn_sinks canonical name %q is not under the raw attn. namespace", sink)
	}
	if strings.Contains(sink, "self_attn.") {
		t.Errorf("attn_sinks canonical name %q landed under self_attn.; the sink must stay distinguishable", sink)
	}

	if got, ok := CanonicalTensorNameArch("blk.0.totally_unknown_suffix.weight", "deepseek41"); ok {
		t.Errorf("unknown suffix resolved to %q; the deepseek41 map must not be a catch-all", got)
	}
}

// TestDeepSeek41GGUFV41SuffixesDoNotLeakToSiblings is the ticket #13112 negative
// arm: the V4.1 arms are reached ONLY inside the deepseek41 branch of
// CanonicalTensorNameArch (gated on archIsDeepSeek41). A sibling arch must
// therefore resolve no suffix that the V4.1 branch ALONE introduced.
//
// Fence: an arch that reaches the shared glm_moe_dsa MLA+MoE map
// (archUsesMLAMoELayout: glm_moe_dsa, deepseek2) legitimately resolves the three
// suffixes that map ALREADY shares with V4.1 — attn_kv_a_norm.weight,
// indexer.attn_k.weight, indexer.k_norm.weight. That coverage PREDATES #13112
// (glmGGUFAttnKVANorm / glmGGUFIndexerWK / glmGGUFIndexerKNorm), so it is not a
// leak this ticket introduced; asserting ok=false for those would pin a
// pre-existing sibling behavior this ticket has no scope to change. The five
// suffixes UNIQUE to V4.1 must stay refused by every sibling, and llama/qwen2
// (which do not reach the glm map) must refuse all eight.
func TestDeepSeek41GGUFV41SuffixesDoNotLeakToSiblings(t *testing.T) {
	// Suffixes the shared glm_moe_dsa/deepseek2 map already carries at base.
	sharedWithSiblingMLA := map[string]bool{
		"attn_kv_a_norm.weight": true,
		"indexer.attn_k.weight": true,
		"indexer.k_norm.weight": true,
	}
	siblings := []string{"llama", "qwen2", "deepseek2", "glm_moe_dsa"}
	for _, sibling := range siblings {
		t.Run(sibling, func(t *testing.T) {
			for _, tc := range deepSeek41V41RealSuffixes {
				ggufName := "blk.0." + tc.suffix
				got, ok := CanonicalTensorNameArch(ggufName, sibling)
				if !ok {
					continue // refused: the strongest form of no-leak
				}
				reachesGlmMLA := sibling == "deepseek2" || sibling == "glm_moe_dsa"
				if reachesGlmMLA && sharedWithSiblingMLA[tc.suffix] {
					// Pre-existing sibling coverage, not a #13112 leak. Pin the
					// canonical it must keep producing so a future refactor that
					// silently re-routes it is still caught.
					if want := tc.want(0); got != want {
						t.Errorf("sibling %q resolved shared suffix %q to %q, want %q",
							sibling, tc.suffix, got, want)
					}
					continue
				}
				t.Errorf("V4.1 arm LEAKED into sibling arch %q: CanonicalTensorNameArch(%q, %q) = %q, want ok=false; the deepseek41 branch must stay gated on archIsDeepSeek41",
					sibling, ggufName, sibling, got)
			}
		})
	}
}

// TestDeepSeek41GGUFV41HyperconnectionTapsMap is the regression guard for the
// hc_* hyper-connection tap spellings the real converted artifact emits. The
// published vcruz305 Q2_K shard carries blk.N.hc_attn_{fn,base,scale}.weight and
// blk.N.hc_ffn_{fn,base,scale}.weight (parsed directly from the staged header).
// The earlier map arms omitted the trailing ".weight", so every one of these
// tensors was refused by CanonicalTensorNameArch and the shard load hard-failed
// with "gguf: no canonical mapping for tensor blk.0.hc_attn_fn.weight" - the
// same failure class the attn_kv_a_norm fix retired. Assert both the resolve and
// the exact canonical leaf, at two layer indices.
func TestDeepSeek41GGUFV41HyperconnectionTapsMap(t *testing.T) {
	taps := []string{
		"hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
		"hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight",
	}
	for _, tap := range taps {
		for _, layer := range []int{0, 7} {
			ggufName := fmt.Sprintf("blk.%d.%s", layer, tap)
			got, ok := CanonicalTensorNameArch(ggufName, "deepseek41")
			if !ok {
				t.Errorf("published V4.1 hyper-connection tap %q did not resolve under deepseek41; the shard load hard-fails", ggufName)
				continue
			}
			// The map strips the leading "hc_" and re-nests the tap under the
			// per-layer hc.<leaf>.weight namespace, so hc_attn_fn.weight becomes
			// model.layers.<L>.hc.attn_fn.weight.
			want := layerName(layer, "hc."+strings.TrimPrefix(strings.TrimSuffix(tap, ".weight"), "hc_")+".weight")
			if got != want {
				t.Errorf("CanonicalTensorNameArch(%q, deepseek41) = %q, want %q", ggufName, got, want)
			}
		}
	}
}
