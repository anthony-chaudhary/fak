package ggufload

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// v41_engram_rows_test.go is the witness suite for issue #12936: a bounded,
// shard-aware model.V41EngramRowSource over a real GGUF engram_embd.weight table.
//
// The fixture is a two-shard split checkpoint built through the same low-level
// byte writers the rest of this package uses:
//
//   - shard 1 carries the deepseek41 base geometry plus the complete eight-key
//     nested Engram metadata (engram.encoding, .layer_ids, .rows,
//     .compressed_vocab_size, .pad_id, .token_map, .primes, .multipliers);
//   - shard 2 carries one type-24 [264,3] blk.1.engram_embd.weight payload whose
//     three rows are byte-distinct.
//
// The witness opens the split through OpenWeights, builds the row source, feeds
// row IDs [2,0,2] through one bounded row cache with three columns, and gathers
// through model.GatherV41EngramRows. That proves owning-shard routing, byte-exact
// non-contiguous reads, and a cache hit on the repeated row. It also asserts every
// malformed-input refusal fires before any payload read.

const (
	v41TestEncode = "e4m3_e8m0_32_row264"
	v41TestRows   = 3
)

// v41TestEngramArch is the raw arch spelling the fixture declares so Config reads
// the file's own "<arch>." keys while recognition canonicalizes to deepseek41.
const v41TestEngramArch = "deepseek_v41"

// v41TestRow returns a deterministic 264-byte row whose first byte is a unique
// per-row tag, so a wrong row (or a wrong stride) is immediately visible.
func v41TestRow(tag byte) []byte {
	row := make([]byte, v41EngramRowBytes)
	for i := range row {
		row[i] = tag
	}
	// Seed a few interior bytes so a truncated read cannot alias the tag alone.
	row[v41EngramRowBytes-1] = tag ^ 0xFF
	row[v41EngramRowBytes/2] = tag + 1
	return row
}

// v41WriteKVUint32Array writes a uint32 GGUF metadata array (the element type the
// converter Engram keys require).
func v41WriteKVUint32Array(b *bytes.Buffer, key string, values []uint32) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeUint32))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, v := range values {
		_ = binary.Write(b, binary.LittleEndian, v)
	}
}

// v41WriteKVUint64Array writes a uint64 GGUF metadata array for engram.multipliers.
func v41WriteKVUint64Array(b *bytes.Buffer, key string, values []uint64) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeUint64))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, v := range values {
		_ = binary.Write(b, binary.LittleEndian, v)
	}
}

// v41BuildShard1 constructs shard 1 with an explicit KV count.
func v41BuildShard1(t *testing.T, count int) []byte {
	t.Helper()
	const align = 32
	arch := v41TestEngramArch
	p := arch + "."
	tensors := 0
	// count the exact KV entries we write below.
	kvPairs := [][2]any{}
	addU64 := func(k string, v uint64) { kvPairs = append(kvPairs, [2]any{k, v}) }
	addU32 := func(k string, v uint32) { kvPairs = append(kvPairs, [2]any{k, v}) }
	addF32 := func(k string, v float32) { kvPairs = append(kvPairs, [2]any{k, v}) }
	addStr := func(k, v string) { kvPairs = append(kvPairs, [2]any{k, v}) }
	addU32Arr := func(k string, v []uint32) { kvPairs = append(kvPairs, [2]any{k, v}) }
	addU64Arr := func(k string, v []uint64) { kvPairs = append(kvPairs, [2]any{k, v}) }
	addI32Arr := func(k string, v []int32) { kvPairs = append(kvPairs, [2]any{k, v}) }

	addU32("general.alignment", align)
	addStr("general.architecture", arch)

	addU64(p+"embedding_length", 5120)
	addU64(p+"block_count", 40)
	addU64(p+"attention.head_count", 128)
	addU64(p+"attention.head_count_kv", 128)
	addU64(p+"feed_forward_length", 18432)
	addF32(p+"attention.layer_norm_rms_epsilon", 1e-6)

	addU64(p+"expert_count", 384)
	addU64(p+"expert_used_count", 6)
	addU64(p+"expert_feed_forward_length", 2304)
	addU64(p+"expert_shared_count", 1)
	addU64(p+"expert_shared_feed_forward_length", 2304)

	addU64(p+"attention.q_lora_rank", 1280)
	addU64(p+"attention.kv_lora_rank", 512)
	addU64(p+"attention.key_length_mla", 512)
	addU64(p+"attention.value_length_mla", 512)
	addU64(p+"attention.qk_nope_head_dim", 448)
	addU64(p+"attention.qk_rope_head_dim", 64)

	addU64(p+"attention.indexer.head_count", 32)
	addU64(p+"attention.indexer.key_length", 128)
	addU64(p+"attention.indexer.top_k", 512)

	addU64(p+"attention.output_group_count", 8)
	addU64(p+"attention.output_lora_rank", 1024)
	addI32Arr(p+"attention.compress_ratios", []int32{1, 2, 4})
	addU64(p+"hyper_connection.count", 4)
	addU64(p+"hyper_connection.sinkhorn_iterations", 20)
	addF32(p+"hyper_connection.epsilon", 1e-6)

	// Complete eight-key nested Engram declaration.
	// ngram=4 => multipliers = layerIDs(2) * 4 = 8; nheads=8 =>
	// primes = 2 * (4-1) * 8 = 48.
	layerIDs := []uint32{1, 14}
	multipliers := make([]uint64, len(layerIDs)*4)
	for i := range multipliers {
		multipliers[i] = uint64(i + 1)
	}
	primes := make([]uint32, len(layerIDs)*3*8)
	for i := range primes {
		primes[i] = uint32(17 + i)
	}
	tokenMap := []uint32{0, 1, 2, 3}
	addStr(p+"engram.encoding", v41TestEncode)
	addU32Arr(p+"engram.layer_ids", layerIDs)
	addU32Arr(p+"engram.rows", []uint32{v41TestRows, 1000})
	addU32(p+"engram.compressed_vocab_size", 262144)
	addU32(p+"engram.pad_id", 0)
	addU32Arr(p+"engram.token_map", tokenMap)
	addU32Arr(p+"engram.primes", primes)
	addU64Arr(p+"engram.multipliers", multipliers)

	addU32("split.no", 1)
	addU32("split.count", uint32(count))
	addU32("split.tensors.count", 1)

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(tensors), uint64(len(kvPairs)))
	for _, kv := range kvPairs {
		switch v := kv[1].(type) {
		case uint64:
			writeKVUint64(&b, kv[0].(string), v)
		case uint32:
			writeKVUint32(&b, kv[0].(string), v)
		case float32:
			writeKVFloat32(&b, kv[0].(string), v)
		case string:
			writeKVString(&b, kv[0].(string), v)
		case []uint32:
			v41WriteKVUint32Array(&b, kv[0].(string), v)
		case []uint64:
			v41WriteKVUint64Array(&b, kv[0].(string), v)
		case []int32:
			writeKVIntArrayForTest(&b, kv[0].(string), v)
		default:
			t.Fatalf("unhandled metadata value %T for %s", v, kv[0])
		}
	}
	padToAlignment(&b, align)
	return b.Bytes()
}

// v41BuildShard2 builds a tensor shard carrying one type-24 [264,3] payload.
func v41BuildShard2(t *testing.T, name string, rows [][]byte) []byte {
	t.Helper()
	const align = 32

	var infos bytes.Buffer
	writeTensorInfoForTest(&infos, name, []uint64{v41EngramRowBytes, uint64(len(rows))}, v41TensorRawI8(), 0)

	var b bytes.Buffer
	writeMinimalHeader(&b, 1, 4)
	writeKVUint32(&b, "general.alignment", align)
	writeKVUint32(&b, "split.no", uint32(2))
	writeKVUint32(&b, "split.count", 2)
	writeKVUint32(&b, "split.tensors.count", 1)
	b.Write(infos.Bytes())
	padToAlignment(&b, align)
	for _, row := range rows {
		if len(row) != v41EngramRowBytes {
			t.Fatalf("fixture row width %d, want %d", len(row), v41EngramRowBytes)
		}
		dataStart := b.Len()
		b.Write(row)
		padToLen(&b, dataStart+align)
	}
	return b.Bytes()
}

func v41TensorRawI8() TensorType { return v41EngramRawI8Type }

// TestV41EngramGGUFRowSourceFeedsGather is the witness: real GGUF parsing,
// owning-shard selection, random-access row IO, cache, and gather.
func TestV41EngramGGUFRowSourceFeedsGather(t *testing.T) {
	rows := [][]byte{v41TestRow(0xA1), v41TestRow(0xB2), v41TestRow(0xC3)}
	dir := t.TempDir()
	shard1 := v41BuildShard1(t, 2)
	shard2 := v41BuildShard2(t, "blk.1.engram_embd.weight", rows)
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
		t.Fatalf("V41EngramGGUFOpen: %v", err)
	}
	if src.RowBytes() != v41EngramRowBytes || src.TableRows() != v41TestRows {
		t.Fatalf("RowBytes/TableRows = %d/%d, want %d/%d", src.RowBytes(), src.TableRows(), v41EngramRowBytes, v41TestRows)
	}

	// PrefetchRows=2 so reading row 0 eagerly reads rows 1 and 2 into the bounded
	// cache; the repeated row 2 then serves from cache with no second IO.
	cache, err := model.NewV41EngramRowCache(src, model.V41EngramRowCacheOptions{
		TableRows:    v41TestRows,
		RowBytes:     v41EngramRowBytes,
		BudgetBytes:  int64(v41EngramRowBytes) * 3,
		PrefetchRows: 2,
	})
	if err != nil {
		t.Fatalf("NewV41EngramRowCache: %v", err)
	}

	// Rows [0,2,2] through one cache and three columns: row 0 is a miss whose
	// prefetch pulls rows 1 and 2; the two later row-2 reads serve from cache.
	got, err := model.GatherV41EngramRows([]*model.V41EngramRowCache{cache}, []uint32{0, 2, 2}, 3)
	if err != nil {
		t.Fatalf("GatherV41EngramRows: %v", err)
	}
	want := [][]byte{rows[0], rows[2], rows[2]}
	if len(got) != len(want) {
		t.Fatalf("gather returned %d rows, want %d", len(got), len(want))
	}
	for i := range want {
		if !bytes.Equal(got[i], want[i]) {
			t.Fatalf("gathered row %d mismatch: got tag %#x, want tag %#x", i, got[i][0], want[i][0])
		}
	}
	stats := cache.Stats()
	if stats.Hits < 2 {
		t.Fatalf("expected two cache hits on the repeated row, stats=%+v", stats)
	}
	// One miss (row 0) plus a two-row prefetch = three distinct rows read once.
	if stats.BytesRead != int64(v41TestRows*v41EngramRowBytes) {
		t.Fatalf("BytesRead = %d, want %d", stats.BytesRead, v41TestRows*v41EngramRowBytes)
	}
}

// TestV41EngramGGUFRefusalsFailBeforeRead proves the malformed-input refusals are
// typed and happen before any payload read.
func TestV41EngramGGUFRefusalsFailBeforeRead(t *testing.T) {
	rows := [][]byte{v41TestRow(1), v41TestRow(2), v41TestRow(3)}
	dir := t.TempDir()
	shard1 := v41BuildShard1(t, 2)
	shard2 := v41BuildShard2(t, "blk.1.engram_embd.weight", rows)
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

	if _, err := V41EngramGGUFOpen(ws, 2); err == nil {
		t.Fatal("layer 2 is not an Engram layer, want refusal")
	}
	src, err := V41EngramGGUFOpen(ws, 1)
	if err != nil {
		t.Fatalf("V41EngramGGUFOpen(layer 1): %v", err)
	}
	if _, err := src.ReadRows(2, 2, make([]byte, 2*v41EngramRowBytes)); err == nil {
		t.Fatal("ReadRows past the table end, want refusal")
	}
	if _, err := src.ReadRows(0, 0, nil); err != nil {
		t.Fatalf("zero-count read must succeed: %v", err)
	}
	if _, err := src.ReadRows(-1, 1, make([]byte, v41EngramRowBytes)); err == nil {
		t.Fatal("negative start, want refusal")
	}
	if _, err := src.ReadRows(0, 1, make([]byte, v41EngramRowBytes-1)); err == nil {
		t.Fatal("undersized destination, want refusal")
	}
}

// ensure fmt stays used even if assertions change.
var _ = fmt.Sprintf
