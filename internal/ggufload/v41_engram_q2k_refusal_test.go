package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// v41_engram_q2k_refusal_test.go is the bounded, deterministic witness for the
// residual clause of issue #12996: the EXACT shape of the published
// DeepSeek-V4.1-Flash Q2_K GGUF (vcruz305/DeepSeek-V4.1-Flash-GGUF) must FAIL
// CLOSED at the V4.1 Engram row-source seam, with a refusal that names the
// divergence - never a silent load, never a panic.
//
// The physical admission receipt (commit f75ae93f6,
// docs/benchmarks/receipts/deepseek-v41-loader-admission-strix3-20260913-hardware.json)
// records the published artifact's real header facts:
//
//   - dense_feed_forward_length_present:  false   (MoE-only; expert FFN only)
//   - engram_encoding_key_present:        false   (no {arch}.engram.encoding)
//   - blk.1.engram_embd.weight: type 10 (Q2_K), dims [256, 384006168]
//
// #12979 already landed the conditional dense-FFN derive and #12976 reconciled
// the vcruz reader dialect, so Config() now reaches the V4.1 axes. What was NOT
// witnessed is the composite loader path on the real artifact: Config() success
// followed by V41EngramGGUFOpen, which must refuse because the published table
// is Q2_K rather than the row264 raw-I8 encoding the row source reads.
//
// This fixture is reconstructed from those recorded header facts only. It is a
// fail-closed contract witness, not an admission claim: admitting the published
// artifact requires dequantizing (or re-publishing) the Q2_K Engram table, which
// this leaf deliberately does not do.

// v41Q2KRoWS is a small row count so the synthetic table stays tiny; the real
// artifact declares 384006168 rows. The type and the missing encoding key are
// the load-bearing facts, not the magnitude.
const v41Q2KRoWS = 4

// v41PublishedArtifactMeta builds the published Q2_K artifact's metadata shape:
// the vcruz305 Engram dialect (no encoding/rows/compressed_vocab_size), an
// MoE-only dense-FFN gap, and the full V4.1 axes. It is Config()-valid, so a
// refusal downstream is attributable to the Engram table, not the header.
func v41PublishedArtifactMeta(arch string) (map[string]Value, string) {
	p := arch + "."
	primes := make([]uint64, 2*(4-1)*8) // len(layer_ids)*(max_ngram-1)*head_count
	for i := range primes {
		primes[i] = 16000057 + uint64(i)
	}
	meta := map[string]Value{
		"general.architecture":                 {Type: TypeString, Value: arch},
		p + "embedding_length":                 {Type: TypeUint64, Value: uint64(5120)},
		p + "block_count":                      {Type: TypeUint64, Value: uint64(40)},
		p + "attention.head_count":             {Type: TypeUint64, Value: uint64(128)},
		p + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},

		// MoE-only: expert widths present, dense width deliberately absent.
		p + "expert_count":                      {Type: TypeUint64, Value: uint64(384)},
		p + "expert_used_count":                 {Type: TypeUint64, Value: uint64(6)},
		p + "expert_feed_forward_length":        {Type: TypeUint64, Value: uint64(2304)},
		p + "expert_shared_count":               {Type: TypeUint64, Value: uint64(1)},
		p + "expert_shared_feed_forward_length": {Type: TypeUint64, Value: uint64(2304)},

		// vcruz305 Engram dialect: these keys, and NO engram.encoding.
		p + "engram.layer_ids":      vcruzI32Array([]int32{1, 14}),
		p + "engram.head_count":     {Type: TypeUint32, Value: uint32(8)},
		p + "engram.key_length":     {Type: TypeUint32, Value: uint32(256)},
		p + "engram.max_ngram_size": {Type: TypeUint32, Value: uint32(4)},
		p + "engram.primes":         vcruzU64Array(primes),
		p + "engram.multipliers":    vcruzU64Array([]uint64{35184372088831, 35184372088829}),
		p + "engram.offsets":        vcruzU64Array([]uint64{0, 1}),
		p + "engram.pad_id":         {Type: TypeUint32, Value: uint32(0)},
		p + "engram.token_map":      vcruzI32Array([]int32{7, 8, 9}),
	}
	return meta, p
}

// v41PublishedMetaKeys is the fixed write order for the published-shape header.
var v41PublishedMetaKeys = []string{
	"general.architecture",
	"deepseek41.embedding_length",
	"deepseek41.block_count",
	"deepseek41.attention.head_count",
	"deepseek41.attention.layer_norm_rms_epsilon",
	"deepseek41.expert_count",
	"deepseek41.expert_used_count",
	"deepseek41.expert_feed_forward_length",
	"deepseek41.expert_shared_count",
	"deepseek41.expert_shared_feed_forward_length",
	"deepseek41.engram.layer_ids",
	"deepseek41.engram.head_count",
	"deepseek41.engram.key_length",
	"deepseek41.engram.max_ngram_size",
	"deepseek41.engram.primes",
	"deepseek41.engram.multipliers",
	"deepseek41.engram.offsets",
	"deepseek41.engram.pad_id",
	"deepseek41.engram.token_map",
	"deepseek41.engram.rows",
	"deepseek41.engram.encoding",
}

// writeMetaValueForTest emits one GGUF metadata pair of any supported shape.
func writeMetaValueForTest(t *testing.T, b *bytes.Buffer, key string, v Value) {
	t.Helper()
	switch v.Type {
	case TypeString:
		writeKVString(b, key, v.Value.(string))
	case TypeUint64:
		writeKVUint64(b, key, v.Value.(uint64))
	case TypeUint32:
		writeKVUint32(b, key, v.Value.(uint32))
	case TypeFloat32:
		writeKVFloat32(b, key, v.Value.(float32))
	case TypeArray:
		items, _ := v.Value.([]Value)
		if len(items) == 0 {
			t.Fatalf("empty array for %s", key)
		}
		switch items[0].Type {
		case TypeInt32:
			vals := make([]int32, len(items))
			for i, it := range items {
				vals[i] = it.Value.(int32)
			}
			writeKVIntArrayForTest(b, key, vals)
		case TypeUint64:
			vals := make([]uint64, len(items))
			for i, it := range items {
				vals[i] = it.Value.(uint64)
			}
			v41WriteKVUint64Array(b, key, vals)
		default:
			t.Fatalf("unhandled array element type %v for %s", items[0].Type, key)
		}
	default:
		t.Fatalf("unhandled metadata value %v for %s", v.Type, key)
	}
}

// v41BuildPublishedShardWithMeta writes a two-shard checkpoint whose shard 2
// carries a blk.1.engram_embd.weight table at the given tensor type. The
// injected metadata map lets one builder serve both the Q2_K artifact shape and
// a raw-I8 control, so the assertion isolates the tensor encoding.
func v41BuildPublishedShardWithMeta(t *testing.T, meta map[string]Value, tableType TensorType) (shard1, shard2 []byte) {
	t.Helper()
	const align = 32

	written := 0
	for _, key := range v41PublishedMetaKeys {
		if _, ok := meta[key]; ok {
			written++
		}
	}

	var s1 bytes.Buffer
	writeMinimalHeader(&s1, 0, uint64(written+3))
	for _, key := range v41PublishedMetaKeys {
		v, ok := meta[key]
		if !ok {
			continue
		}
		writeMetaValueForTest(t, &s1, key, v)
	}
	writeKVUint32(&s1, "split.no", 1)
	writeKVUint32(&s1, "split.count", 2)
	writeKVUint32(&s1, "split.tensors.count", 1)
	padToAlignment(&s1, align)

	var infos bytes.Buffer
	writeTensorInfoForTest(&infos, "blk.1.engram_embd.weight",
		[]uint64{v41EngramRowBytes, v41Q2KRoWS}, tableType, 0)

	var s2 bytes.Buffer
	writeMinimalHeader(&s2, 1, 4)
	writeKVUint32(&s2, "general.alignment", align)
	writeKVUint32(&s2, "split.no", 2)
	writeKVUint32(&s2, "split.count", 2)
	writeKVUint32(&s2, "split.tensors.count", 1)
	s2.Write(infos.Bytes())
	padToAlignment(&s2, align)
	for i := 0; i < v41Q2KRoWS; i++ {
		row := v41TestRow(byte(0x40 + i))
		dataStart := s2.Len()
		s2.Write(row)
		padToLen(&s2, dataStart+align)
	}
	return s1.Bytes(), s2.Bytes()
}

// v41OpenPublishedShards writes and opens the two-shard fixture, returning the
// open WeightSource; the cleanup closes it.
func v41OpenPublishedShards(t *testing.T, tableType TensorType) *WeightSource {
	t.Helper()
	meta, _ := v41PublishedArtifactMeta("deepseek41")
	shard1, shard2 := v41BuildPublishedShardWithMeta(t, meta, tableType)
	dir := t.TempDir()
	p1 := filepath.Join(dir, "v41-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "v41-00002-of-00002.gguf")
	if err := os.WriteFile(p1, shard1, 0o644); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := os.WriteFile(p2, shard2, 0o644); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}
	ws, err := OpenWeights(p1)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	t.Cleanup(func() { ws.Close() })
	return ws
}

// TestV41PublishedQ2KEngramTableRefusesFailClosed is the #12996 residual witness:
// the published Q2_K artifact's Config() must succeed (dense-FFN derive and the
// vcruz Engram dialect are landed), while V41EngramGGUFOpen must refuse the
// Q2_K (type 10) Engram table with a named refusal. A silent acceptance here
// would interpret Q2_K quantization blocks as row264 E4M3+E8M0 rows - a silent
// mock of the row-source contract.
func TestV41PublishedQ2KEngramTableRefusesFailClosed(t *testing.T) {
	ws := v41OpenPublishedShards(t, TensorQ2_K)

	// Config() must succeed on the real artifact shape: #12979 derives the dense
	// width and #12976 reads the vcruz Engram dialect. If this fails, the seam
	// under test was never reached.
	if _, err := ws.File.Config(); err != nil {
		t.Fatalf("Config on published-shape metadata: %v (dense-FFN derive / vcruz dialect regressed)", err)
	}
	if ws.File.DeepSeek41Engram == nil {
		t.Fatal("DeepSeek41Engram is nil after Config; the vcruz Engram declaration was not retained")
	}
	if got := ws.File.DeepSeek41Engram.Encoding; got != "" {
		t.Fatalf("published artifact has no engram.encoding key, but Encoding = %q", got)
	}

	// The row source MUST fail closed. The encoding check fires first for the
	// real artifact (no engram.encoding key), so the refusal names the encoding.
	_, err := V41EngramGGUFOpen(ws, 1)
	if err == nil {
		t.Fatal("V41EngramGGUFOpen accepted a Q2_K (type 10) Engram table with no engram.encoding key; the row-source contract was silently mocked")
	}
	if !strings.Contains(err.Error(), "encoding") {
		t.Fatalf("refusal %q does not name the Engram encoding divergence", err)
	}
}

// TestV41RawI8EngramTableStillOpens is the positive control: the same fixture
// builder, with a raw-I8 (type 24) table AND the required engram.encoding key,
// must still open. This proves the refusal above is about the published
// artifact's table encoding, not a fixture or builder defect.
func TestV41RawI8EngramTableStillOpens(t *testing.T) {
	meta, p := v41PublishedArtifactMeta("deepseek41")
	meta[p+"engram.encoding"] = Value{Type: TypeString, Value: v41EngramEncoding}
	// A row source needs a per-layer row count; the vcruz dialect omits it, so the
	// positive control supplies engram.rows positionally against layer_ids [1 14].
	meta[p+"engram.rows"] = Value{Type: TypeArray, Value: []Value{
		{Type: TypeUint64, Value: uint64(v41Q2KRoWS)},
		{Type: TypeUint64, Value: uint64(1000)},
	}}

	shard1, shard2 := v41BuildPublishedShardWithMeta(t, meta, v41EngramRawI8Type)
	dir := t.TempDir()
	p1 := filepath.Join(dir, "v41-00001-of-00002.gguf")
	p2 := filepath.Join(dir, "v41-00002-of-00002.gguf")
	if err := os.WriteFile(p1, shard1, 0o644); err != nil {
		t.Fatalf("write shard 1: %v", err)
	}
	if err := os.WriteFile(p2, shard2, 0o644); err != nil {
		t.Fatalf("write shard 2: %v", err)
	}
	ws, err := OpenWeights(p1)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	src, err := V41EngramGGUFOpen(ws, 1)
	if err != nil {
		t.Fatalf("V41EngramGGUFOpen on raw-I8 control: %v", err)
	}
	if src.RowBytes() != v41EngramRowBytes || src.TableRows() != v41Q2KRoWS {
		t.Fatalf("RowBytes/TableRows = %d/%d, want %d/%d", src.RowBytes(), src.TableRows(), v41EngramRowBytes, v41Q2KRoWS)
	}
}
