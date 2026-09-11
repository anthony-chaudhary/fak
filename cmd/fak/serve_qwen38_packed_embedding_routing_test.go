package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// TestServeQwen38PackedEmbeddingRouting exercises the ordinary serve loader with
// real GGUF tensor data. The resulting model layout is the oracle: a selected
// checkpoint retains token_embd.weight in packed Q4_K form, while non-target
// paths continue to expand the same bytes to F32.
func TestServeQwen38PackedEmbeddingRouting(t *testing.T) {
	originalMetalAvailable := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = originalMetalAvailable })
	serveMetalAvailable = func() bool { return true }
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "0")
	t.Setenv("FAK_METAL_STREAM_Q4K", "0")

	t.Run("exact artifact on native Metal retains packed rows", func(t *testing.T) {
		path := writeServePackedEmbeddingFixture(t, "Qwen3.8-27B-Q4_K_M.gguf", true)
		m, residentQ4K, _, _ := loadServeInKernelModel(path, nil, false, 0, nil, 1, nil)
		if m == nil || !residentQ4K {
			t.Fatalf("serve load = model %v residentQ4K=%v, want non-nil/true", m, residentQ4K)
		}
		if m.Q2KEmbedding == nil || m.Q2KEmbedding.Format() != "Q4_K" {
			t.Fatalf("token embedding = %v, want packed Q4_K", m.Q2KEmbedding)
		}
		if m.HasF32("model.embed_tokens.weight") {
			t.Fatal("selected native Metal artifact expanded token embedding to F32")
		}
	})

	t.Run("different artifact keeps prior F32 layout", func(t *testing.T) {
		path := writeServePackedEmbeddingFixture(t, "other-Q4_K_M.gguf", false)
		m, residentQ4K, _, _ := loadServeInKernelModel(path, nil, false, 0, nil, 1, nil)
		if m == nil || !residentQ4K {
			t.Fatalf("serve load = model %v residentQ4K=%v, want non-nil/true", m, residentQ4K)
		}
		assertServeEmbeddingExpanded(t, m.Q2KEmbedding, m.HasF32("model.embed_tokens.weight"))
	})

	t.Run("explicit backend keeps prior F32 layout", func(t *testing.T) {
		path := writeServePackedEmbeddingFixture(t, "Qwen3.8-27B-Q4_K_M.gguf", true)
		backend := serveCapBackend{Backend: compute.Default(), uploadDtype: true}
		m, residentQ4K, _, _ := loadServeInKernelModel(path, backend, false, 0, nil, 1, nil)
		if m == nil || !residentQ4K {
			t.Fatalf("serve load = model %v residentQ4K=%v, want non-nil/true", m, residentQ4K)
		}
		assertServeEmbeddingExpanded(t, m.Q2KEmbedding, m.HasF32("model.embed_tokens.weight"))
	})

	t.Run("resident Q4K disabled keeps prior F32 layout", func(t *testing.T) {
		t.Setenv("FAK_Q4K", "0")
		path := writeServePackedEmbeddingFixture(t, "Qwen3.8-27B-Q4_K_M.gguf", true)
		m, residentQ4K, _, _ := loadServeInKernelModel(path, nil, false, 0, nil, 1, nil)
		if m == nil || residentQ4K {
			t.Fatalf("serve load = model %v residentQ4K=%v, want non-nil/false", m, residentQ4K)
		}
		assertServeEmbeddingExpanded(t, m.Q2KEmbedding, m.HasF32("model.embed_tokens.weight"))
	})
}

func assertServeEmbeddingExpanded(t *testing.T, packed *fakmodel.Q2KEmbedding, hasF32 bool) {
	t.Helper()
	if packed != nil {
		t.Fatalf("non-target token embedding unexpectedly retained as %s", packed.Format())
	}
	if !hasF32 {
		t.Fatal("non-target token embedding missing prior F32 layout")
	}
}

// writeServePackedEmbeddingFixture writes the smallest dense Qwen hybrid GGUF
// that can distinguish packed-row residency from F32 expansion. A sparse tail
// supplies the production artifact's witnessed byte size without allocating or
// reading a 27B checkpoint in this deterministic routing test.
func writeServePackedEmbeddingFixture(t *testing.T, base string, witnessedSize bool) string {
	t.Helper()
	const (
		dim          = 256
		vocab        = 3
		q4BlockBytes = 144
		q6BlockBytes = 210
	)
	embedBytes := uint64(vocab * q4BlockBytes)
	outputBytes := uint64(vocab * q6BlockBytes)
	linearBytes := uint64(dim * q4BlockBytes)
	align32 := func(n uint64) uint64 { return (n + 31) &^ 31 }

	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 5, 10)
	writeKVStringForTest(&b, "general.architecture", "qwen35")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "qwen35.embedding_length", dim)
	writeKVUint32ForTest(&b, "qwen35.block_count", 2)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count_kv", 1)
	writeKVUint32ForTest(&b, "qwen35.full_attention_interval", 4)
	writeKVUint32ForTest(&b, "qwen35.attention.key_length", dim)
	writeKVUint32ForTest(&b, "qwen35.feed_forward_length", dim)
	writeKVFloat32ForTest(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-6)

	offset := uint64(0)
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{dim, vocab}, uint32(ggufload.TensorQ4_K), offset)
	offset = align32(offset + embedBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, vocab}, uint32(ggufload.TensorQ6_K), offset)
	offset = align32(offset + outputBytes)
	for _, name := range []string{"blk.0.attn_v.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"} {
		writeTensorInfoForTest(&b, name, []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), offset)
		offset = align32(offset + linearBytes)
	}
	padToAlignmentForTest(&b, 32)
	b.Write(make([]byte, int(offset)))

	path := filepath.Join(t.TempDir(), base)
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if witnessedSize {
		if err := os.Truncate(path, qwen38Q4KMArtifactBytes); err != nil {
			t.Fatal(err)
		}
	}
	return path
}
