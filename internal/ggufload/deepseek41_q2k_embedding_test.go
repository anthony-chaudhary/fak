package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// writeV41EmbeddingFixture writes a single-file DeepSeek-V4.1 checkpoint with a
// token_embd.weight of embType and a distinct F32 output.weight.
func writeV41EmbeddingFixture(t *testing.T, embType TensorType) string {
	t.Helper()
	const hidden, vocab, align = 256, 4, 32
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
	keys := make([]string, 0, len(meta))
	for key := range meta {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var embData []byte
	if embType == TensorQ2_K {
		block, _ := q2KFixtureBlock()
		for i := 0; i < vocab; i++ {
			embData = append(embData, block...)
		}
	} else {
		embData = make([]byte, hidden*vocab*4)
	}
	outOff := uint64((len(embData) + align - 1) / align * align)
	var b bytes.Buffer
	writeMinimalHeader(&b, 2, uint64(len(meta)))
	for _, key := range keys {
		writeMetaValueForTest(t, &b, key, meta[key])
	}
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{hidden, vocab}, embType, 0)
	writeTensorInfoForTest(&b, "output.weight", []uint64{hidden, vocab}, TensorF32, outOff)
	padToAlignment(&b, align)
	b.Write(embData)
	padToAlignment(&b, align)
	b.Write(make([]byte, hidden*vocab*4))
	path := filepath.Join(t.TempDir(), "v41-embed.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// A V4.1 Q2_K token table qualifies for packed residency and loads without the
// F32 expansion; an F32 table does not qualify.
// fak-test:runtime fast est=100ms lane=default
func TestDeepSeek41Q2KEmbeddingResident(t *testing.T) {
	for _, tc := range []struct {
		name string
		typ  TensorType
		want bool
	}{{"q2k", TensorQ2_K, true}, {"f32", TensorF32, false}} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeV41EmbeddingFixture(t, tc.typ)
			ws, err := OpenWeights(path)
			if err != nil {
				t.Fatal(err)
			}
			got := ws.DeepSeek41Q2KEmbeddingResident()
			ws.Close()
			if got != tc.want {
				t.Fatalf("DeepSeek41Q2KEmbeddingResident() = %v, want %v", got, tc.want)
			}
			if !tc.want {
				return
			}
			m, err := LoadModelQ4KProfileOptions(path, nil, WithQ2KEmbeddingResident(true))
			if err != nil {
				t.Fatalf("load with packed embedding: %v", err)
			}
			if !m.HasQ2KEmbedding() {
				t.Fatal("V4.1 load did not retain the packed Q2_K embedding")
			}
			if m.HasF32("model.embed_tokens.weight") {
				t.Fatal("V4.1 load also expanded the embedding to F32")
			}
		})
	}
}
