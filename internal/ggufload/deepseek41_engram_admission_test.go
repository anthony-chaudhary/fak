package ggufload

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
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

// TestDeepSeek41GGUFDeclaredEngramCannotLoadUnbound isolates the ordinary
// GGUF load-admission seam, not full-checkpoint correctness. It compiles against
// the unmodified parent: only existing loader/model APIs and test writers are used.
// Parent behavior drops the table and returns a Model despite missing Engram
// mixing tensors and runtime attachment. A named typed refusal is required.
// fak-test:runtime fast est=100ms lane=default
func TestDeepSeek41GGUFDeclaredEngramCannotLoadUnbound(t *testing.T) {
	path := writeV41UnboundEngramFixture(t, true)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("fixture header must be admitted before load: %v", err)
	}
	eng := ws.File.DeepSeek41Engram
	if cfg.NumLayers != 2 || cfg.HiddenSize != 64 || eng == nil ||
		len(eng.LayerIDs) != 1 || eng.LayerIDs[0] != 1 ||
		eng.NHeads != 8 || eng.HeadDim != 256 || eng.MaxNgramSize != 4 || eng.PadTokenID != 2 {
		t.Fatalf("fixture metadata drift: config=%+v Engram=%+v", cfg, eng)
	}
	if eng.Encoding != "" || len(eng.NumEmbeddings) != 0 || eng.CompressedVocabSize != 0 {
		t.Fatalf("absent vcruz keys were fabricated: Engram=%+v", eng)
	}
	source, err := V41EngramQ2KOpen(ws, 1)
	if err != nil {
		t.Fatalf("fixture must carry a supported Q2_K table: %v", err)
	}
	if source.RowBytes() != 1024 || source.TableRows() != 72 {
		t.Fatalf("fixture source rows=%d bytes=%d, want 72/1024", source.TableRows(), source.RowBytes())
	}

	m, err := LoadModelQ4KStreamedDenseContext(context.Background(), path, nil)
	if m != nil {
		defer m.CloseWeights()
	}
	if err == nil {
		t.Fatal("GGUF loader returned a model after dropping a declared, unbound Engram stage")
	}
	if !errors.Is(err, model.ErrV41NativeUnsupported) || !strings.Contains(strings.ToLower(err.Error()), "engram") {
		t.Fatalf("load error=%v, want an Engram-named ErrV41NativeUnsupported", err)
	}
	if m != nil {
		t.Fatal("refused Engram load returned a partial model")
	}

	// Anti-over-refusal: identical ordinary load geometry without any Engram
	// declaration/table is still admitted. It is not used for forward execution.
	plain, err := LoadModelQ4KStreamedDenseContext(context.Background(), writeV41UnboundEngramFixture(t, false), nil)
	if err != nil || plain == nil {
		t.Fatalf("no-Engram control refused: model=%v err=%v", plain, err)
	}
	if err := plain.CloseWeights(); err != nil {
		t.Fatal(err)
	}
}
