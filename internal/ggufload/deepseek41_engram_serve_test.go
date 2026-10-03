package ggufload

// deepseek41_engram_serve_test.go is the named #13662 witness: an ordinary Q4K
// streamed serve load of a supported V4.1 GGUF attaches an executable, verified
// Engram stage to the returned model, and a declared-but-unsupported Engram still
// refuses with a typed model.ErrV41NativeUnsupported. It is the attachment /
// admission seam only — it makes no forward numeric, parity, or throughput claim.
//
// The fixture is the two-shard family from deepseek41_engram_admission_test.go.
// Its Engram mixing tensors (engram_kv / engram_q_norm / engram_k_norm) are
// intentionally absent, so the reduced forward cannot inject; the acceptance is
// the ATTACHMENT, not a forward run.

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

// TestDeepSeek41EngramServingAttachment is the #13662 witness. It loads the
// Engram fixture through the ordinary streamed Q4K entrypoint and asserts the
// returned model carries the declared Engram layer AND a wired, verified stage;
// then it asserts a declared-but-unopenable Engram still refuses.
// fak-test:runtime fast est=200ms lane=default
func TestDeepSeek41EngramServingAttachment(t *testing.T) {
	path := writeV41UnboundEngramFixture(t, true)

	// Precondition: the header really declares a supported Q2_K table with a
	// declared-rows-absent (dims-derived) count, so a load failure is the
	// fence/attachment, not a malformed fixture.
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := ws.File.Config()
	if err != nil {
		ws.Close()
		t.Fatalf("fixture header must be admitted before load: %v", err)
	}
	eng := ws.File.DeepSeek41Engram
	if cfg.NumLayers != 2 || cfg.HiddenSize != 64 || eng == nil ||
		len(eng.LayerIDs) != 1 || eng.LayerIDs[0] != 1 ||
		eng.NHeads != 8 || eng.HeadDim != 256 || eng.MaxNgramSize != 4 || eng.PadTokenID != 2 {
		ws.Close()
		t.Fatalf("fixture metadata drift: config=%+v Engram=%+v", cfg, eng)
	}
	src, err := V41EngramQ2KOpen(ws, 1)
	if err != nil {
		ws.Close()
		t.Fatalf("fixture must carry a supported Q2_K table: %v", err)
	}
	if src.RowBytes() != 1024 || src.TableRows() != 72 {
		ws.Close()
		t.Fatalf("fixture source rows=%d bytes=%d, want 72/1024", src.TableRows(), src.RowBytes())
	}
	ws.Close()

	// The ordinary streamed serve load must now attach the declared Engram stage
	// and return a model carrying it.
	m, err := LoadModelQ4KStreamedDenseContext(context.Background(), path, nil)
	if m != nil {
		defer m.CloseWeights()
	}
	if err != nil {
		t.Fatalf("declared-Engram load refused (fence not lifted / attach failed): %v", err)
	}
	if m == nil {
		t.Fatal("declared-Engram load returned a nil model")
	}
	if !m.V41EngramAttached() {
		t.Fatal("returned model does not report an attached Engram stage")
	}
	d41 := m.Cfg.DeepSeekV41
	if d41 == nil || len(d41.EngramLayerIDs) != 1 || d41.EngramLayerIDs[0] != 1 ||
		d41.EngramMaxNgramSize != 4 || d41.EngramNHeads != 8 || d41.EngramHeadDim != 256 {
		t.Fatalf("returned config does not carry the declared Engram geometry: %+v", d41)
	}
	// The stage must have used the VERIFIED binding route, not the unverified
	// prepared reader: every wired source reports a non-zero artifact binding.
	bindings, ok := m.V41EngramStageBindings()
	if !ok || len(bindings) != 1 {
		t.Fatalf("stage bindings = %v ok=%t, want one verified binding", bindings, ok)
	}
	if bindings[0].Rows != 72 || bindings[0].RowBytes != model.V41EngramF32RowBytes ||
		!strings.HasPrefix(bindings[0].Digest, "sha256:") || len(bindings[0].Digest) != len("sha256:")+64 {
		t.Fatalf("wired source binding is not the verified Q2_K f32 binding: %+v", bindings[0])
	}

	// Declarative control: a declared Engram whose table is absent (no openable,
	// verifiable source) must still refuse with a typed
	// model.ErrV41NativeUnsupported, never a silent weightless load.
	badPath := writeV41DeclaredEngramNoTableFixture(t)
	bad, berr := LoadModelQ4KStreamedDenseContext(context.Background(), badPath, nil)
	if bad != nil {
		bad.CloseWeights()
		t.Fatal("declared-but-unopenable Engram returned a partial model")
	}
	if berr == nil {
		t.Fatal("declared-but-unopenable Engram loaded instead of refusing")
	}
	if !errors.Is(berr, model.ErrV41NativeUnsupported) {
		t.Fatalf("unopenable-Engram error = %v, want errors.Is ErrV41NativeUnsupported", berr)
	}
	if !strings.Contains(strings.ToLower(berr.Error()), "engram") {
		t.Fatalf("unopenable-Engram error %q does not name the Engram stage", berr)
	}
}

// writeV41DeclaredEngramNoTableFixture writes the same two-shard header family
// as writeV41UnboundEngramFixture, but shard 2 ships NO blk.1.engram_embd.weight
// tensor while the metadata still declares layer 1 as an Engram layer. That is
// the declared-but-unsupported class: the loader must refuse, not attach.
func writeV41DeclaredEngramNoTableFixture(t *testing.T) string {
	t.Helper()
	const hidden, vocab, align = 64, 4, 32
	const prefix = "deepseek41."
	primes := make([]uint64, 24)
	offsets := make([]uint64, 24)
	for i := range primes {
		primes[i], offsets[i] = 3, uint64(3*i)
	}
	meta := map[string]Value{
		"general.architecture":                      {Type: TypeString, Value: "deepseek41"},
		"general.alignment":                         {Type: TypeUint32, Value: uint32(align)},
		prefix + "embedding_length":                 {Type: TypeUint32, Value: uint32(hidden)},
		prefix + "block_count":                      {Type: TypeUint32, Value: uint32(2)},
		prefix + "attention.head_count":             {Type: TypeUint32, Value: uint32(2)},
		prefix + "feed_forward_length":              {Type: TypeUint32, Value: uint32(hidden)},
		prefix + "attention.layer_norm_rms_epsilon": {Type: TypeFloat32, Value: float32(1e-6)},
		prefix + "engram.layer_ids":                 vcruzI32Array([]int32{1}),
		prefix + "engram.head_count":                {Type: TypeUint32, Value: uint32(8)},
		prefix + "engram.key_length":                {Type: TypeUint32, Value: uint32(256)},
		prefix + "engram.max_ngram_size":            {Type: TypeUint32, Value: uint32(4)},
		prefix + "engram.pad_id":                    {Type: TypeUint32, Value: uint32(2)},
		prefix + "engram.token_map":                 vcruzI32Array([]int32{0, 1, 2, 3}),
		prefix + "engram.primes":                    vcruzU64Array(primes),
		prefix + "engram.multipliers":               vcruzU64Array([]uint64{1, 3, 5, 7}),
		prefix + "engram.offsets":                   vcruzU64Array(offsets),
	}
	keys := make([]string, 0, len(meta))
	for key := range meta {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	var first bytes.Buffer
	writeMinimalHeader(&first, 1, uint64(len(meta)+3))
	for _, key := range keys {
		writeMetaValueForTest(t, &first, key, meta[key])
	}
	writeKVUint32(&first, "split.no", 0)
	writeKVUint32(&first, "split.count", 2)
	writeKVUint32(&first, "split.tensors.count", 1)
	writeTensorInfoForTest(&first, "output.weight", []uint64{hidden, vocab}, TensorF32, 0)
	padToAlignment(&first, align)
	first.Write(make([]byte, hidden*vocab*4))

	var second bytes.Buffer
	writeMinimalHeader(&second, 0, 4)
	writeKVUint32(&second, "general.alignment", align)
	writeKVUint32(&second, "split.no", 1)
	writeKVUint32(&second, "split.count", 2)
	writeKVUint32(&second, "split.tensors.count", 1)
	padToAlignment(&second, align)

	dir := t.TempDir()
	for name, data := range map[string][]byte{
		"engram-00001-of-00002.gguf": first.Bytes(),
		"engram-00002-of-00002.gguf": second.Bytes(),
	} {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return filepath.Join(dir, "engram-00001-of-00002.gguf")
}
