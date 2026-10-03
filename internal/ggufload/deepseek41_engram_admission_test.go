package ggufload

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// writeV41UnboundEngramFixture writes an admission-only, two-shard checkpoint.
// Its vcruz metadata grammar is grounded in the 2026-10-01 header inspection:
// Q2_K rows have width 256; heads=8, max_ngram=4, pad_id=2; encoding, rows and
// compressed_vocab_size are absent. split.no is zero-based, matching the
// inspected artifact (filenames remain one-based). Hidden width, layer count, vocabulary,
// hash constants and table length are deliberately synthetic and bounded.
// This is not a full decoder or a replica of the 40-layer artifact.
//
// output.weight supplies one ordinary quantizable tensor, so an Engram error
// cannot be replaced by the unrelated "no quantizable weights found" error.
// The Engram mixing tensors are intentionally absent: a successful load must
// not erase that declared but unexecutable stage.
func writeV41UnboundEngramFixture(t *testing.T, withEngram bool) string {
	t.Helper()
	const hidden, vocab, rows, align = 64, 4, 72, 32
	const prefix = "deepseek41."
	meta := map[string]Value{
		"general.architecture":                      {Type: TypeString, Value: "deepseek41"},
		"general.alignment":                         {Type: TypeUint32, Value: uint32(align)},
		prefix + "embedding_length":                 {Type: TypeUint32, Value: uint32(hidden)},
		prefix + "block_count":                      {Type: TypeUint32, Value: uint32(2)},
		prefix + "attention.head_count":             {Type: TypeUint32, Value: uint32(2)},
		prefix + "feed_forward_length":              {Type: TypeUint32, Value: uint32(hidden)},
		prefix + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},
	}
	if withEngram {
		primes, offsets := make([]uint64, 24), make([]uint64, 24)
		for i := range primes {
			primes[i], offsets[i] = 3, uint64(3*i)
		}
		meta[prefix+"engram.layer_ids"] = vcruzI32Array([]int32{1})
		meta[prefix+"engram.head_count"] = Value{Type: TypeUint32, Value: uint32(8)}
		meta[prefix+"engram.key_length"] = Value{Type: TypeUint32, Value: uint32(256)}
		meta[prefix+"engram.max_ngram_size"] = Value{Type: TypeUint32, Value: uint32(4)}
		meta[prefix+"engram.pad_id"] = Value{Type: TypeUint32, Value: uint32(2)}
		meta[prefix+"engram.token_map"] = vcruzI32Array([]int32{0, 1, 2, 3})
		meta[prefix+"engram.primes"] = vcruzU64Array(primes)
		meta[prefix+"engram.multipliers"] = vcruzU64Array([]uint64{1, 3, 5, 7})
		meta[prefix+"engram.offsets"] = vcruzU64Array(offsets)
	}
	keys := make([]string, 0, len(meta))
	for key := range meta {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	total := uint32(1)
	if withEngram {
		total++
	}
	var first bytes.Buffer
	writeMinimalHeader(&first, 1, uint64(len(meta)+3))
	for _, key := range keys {
		writeMetaValueForTest(t, &first, key, meta[key])
	}
	writeKVUint32(&first, "split.no", 0)
	writeKVUint32(&first, "split.count", 2)
	writeKVUint32(&first, "split.tensors.count", total)
	writeTensorInfoForTest(&first, "output.weight", []uint64{hidden, vocab}, TensorF32, 0)
	padToAlignment(&first, align)
	first.Write(make([]byte, hidden*vocab*4))

	var second bytes.Buffer
	writeMinimalHeader(&second, uint64(total-1), 4)
	writeKVUint32(&second, "general.alignment", align)
	writeKVUint32(&second, "split.no", 1)
	writeKVUint32(&second, "split.count", 2)
	writeKVUint32(&second, "split.tensors.count", total)
	if withEngram {
		writeTensorInfoForTest(&second, "blk.1.engram_embd.weight", []uint64{256, rows}, TensorQ2_K, 0)
	}
	padToAlignment(&second, align)
	if withEngram {
		block, _ := q2KFixtureBlock()
		for i := 0; i < rows; i++ {
			second.Write(block)
		}
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "engram-00001-of-00002.gguf")
	for name, data := range map[string][]byte{
		"engram-00001-of-00002.gguf": first.Bytes(),
		"engram-00002-of-00002.gguf": second.Bytes(),
	} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

// TestDeepSeek41GGUFDeclaredEngramAttachesOnLoad verifies the #13662 behavior
// change: an ordinary Q4K streamed load of a fixture that DECLARES Engram and
// ships a supported Q2_K table now succeeds and returns a model carrying the
// attached stage, while the identical no-Engram geometry still loads. The
// declared-but-unopenable refusal lives in
// deepseek41_engram_serve_test.go (TestDeepSeek41EngramServingAttachment).
// fak-test:runtime fast est=200ms lane=default
func TestDeepSeek41GGUFDeclaredEngramAttachesOnLoad(t *testing.T) {
	path := writeV41UnboundEngramFixture(t, true)
	m, err := LoadModelQ4KStreamedDenseContext(context.Background(), path, nil)
	if m != nil {
		defer m.CloseWeights()
	}
	if err != nil {
		t.Fatalf("declared-Engram load refused: %v", err)
	}
	if m == nil {
		t.Fatal("declared-Engram load returned a nil model")
	}
	if !m.V41EngramAttached() {
		t.Fatal("loaded model does not report an attached Engram stage")
	}
	if d41 := m.Cfg.DeepSeekV41; d41 == nil || len(d41.EngramLayerIDs) != 1 || d41.EngramLayerIDs[0] != 1 {
		t.Fatalf("returned config does not carry the declared Engram layer: %+v", m.Cfg.DeepSeekV41)
	}

	// Anti-over-refusal: identical ordinary load geometry without any Engram
	// declaration/table is still admitted.
	plain, err := LoadModelQ4KStreamedDenseContext(context.Background(), writeV41UnboundEngramFixture(t, false), nil)
	if err != nil || plain == nil {
		t.Fatalf("no-Engram control refused: model=%v err=%v", plain, err)
	}
	if err := plain.CloseWeights(); err != nil {
		t.Fatal(err)
	}
}
