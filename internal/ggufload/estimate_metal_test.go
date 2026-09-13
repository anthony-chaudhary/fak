package ggufload

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// The retained Qwen artifact uses these GGUF source names on linear-attention
// layers. Their model storage names differ, so a suffix-only estimate is wrong.
func TestEstimateQ4KRemappedProjectionStorage(t *testing.T) {
	const dim = 256
	const packed = dim * dim / 256 * 144
	var b bytes.Buffer
	writeMinimalHeader(&b, 3, 10)
	writeKVString(&b, "general.architecture", "qwen35")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "qwen35.embedding_length", dim)
	writeKVUint32(&b, "qwen35.block_count", 2)
	writeKVUint32(&b, "qwen35.attention.head_count", 1)
	writeKVUint32(&b, "qwen35.attention.head_count_kv", 1)
	writeKVUint32(&b, "qwen35.full_attention_interval", 4)
	writeKVUint32(&b, "qwen35.attention.key_length", dim)
	writeKVUint32(&b, "qwen35.feed_forward_length", dim)
	writeKVFloat32(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-6)
	for i, name := range []string{"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "output.weight"} {
		writeTensorInfoForTest(&b, name, []uint64{dim, dim}, TensorQ4_K, uint64(i*packed))
	}
	padToAlignment(&b, 32)
	b.Write(make([]byte, 3*packed))
	path := filepath.Join(t.TempDir(), "remapped.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	m, err := LoadModelQ4KProfileOptions(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.CloseWeights() })
	if !m.HasQ8("model.layers.0.linear_attn.in_proj_qkv.weight") || !m.HasQ8("model.layers.0.linear_attn.in_proj_z.weight") {
		t.Fatal("loader did not materialize both remapped source projections as native Q8")
	}
	const want = 2*dim*dim/32*36 + packed
	if r := m.ResidentReport(); r.Q8Tensors != 2 || r.Q4KTensors != 1 || r.TotalResidentBytes != want {
		t.Fatalf("resident=%+v, want two Q8 projections plus packed head, %d bytes", r, want)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	plan, err := ws.EstimateQ4KLoadMemoryPlan()
	if err != nil {
		t.Fatalf("loader-supported remapped projections must be estimable: %v", err)
	}
	if plan.Total() != want {
		t.Fatalf("estimated=%d, want actual independently counted storage %d", plan.Total(), want)
	}
}

// TestEstimateMetalTransformedWeightResidency reproduces #11962 using actual
// GGUF loads. It measures logical stored weights, not startup allocation peaks,
// page-rounded backing, Metal copies, KV/GDN state, or physical process footprint.
func TestEstimateMetalTransformedWeightResidency(t *testing.T) {
	const dim, vocab = 256, 4
	// The shared fixture has a Q2_K embedding, a Q4_K output projection, and
	// three Q4_K layer projections. The default loader expands the embedding
	// to F32. Keep byte expectations independent of estimator geometry helpers.
	path := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ2_K, TensorQ4_K, false, false)
	const embeddingBytes = vocab * dim * 4
	const projectionElements = 3*dim*dim + vocab*dim
	const packedBytes = projectionElements / 256 * 144
	const q8Bytes = projectionElements + projectionElements/32*4

	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	raw, err := ws.EstimateLoadBytes()
	if err != nil {
		t.Fatal(err)
	}
	rawPlan, err := ws.EstimateLoadMemoryPlan()
	if err != nil || rawPlan.Total() != raw || raw != packedBytes+vocab*84 {
		t.Fatalf("raw payload contract: bytes=%d plan=%v error=%v", raw, rawPlan, err)
	}

	t.Run("resident-q4k", func(t *testing.T) {
		plan, err := ws.EstimateQ4KLoadMemoryPlan()
		if err != nil {
			t.Fatal(err)
		}
		m, err := LoadModelQ4KProfileOptions(path, nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = m.CloseWeights() })
		r := m.ResidentReport()
		const want = packedBytes + embeddingBytes
		if r.Q4KTensors != 4 || r.Q4KBytes != packedBytes || r.F32Bytes != embeddingBytes || r.TotalResidentBytes != want {
			t.Fatalf("fixture resident layout = %+v; want four packed projections (%d bytes), F32 embedding (%d bytes), total %d", r, packedBytes, embeddingBytes, want)
		}
		if got := plan.Total(); got != r.TotalResidentBytes {
			t.Errorf("resident-q4k weight estimate mismatch: got %d bytes, actual stored weights %d (independent count %d)", got, r.TotalResidentBytes, want)
		}
	})

	t.Run("lean-q8", func(t *testing.T) {
		plan, err := ws.EstimateQ8LoadMemoryPlan()
		if err != nil {
			t.Fatal(err)
		}
		m, err := LoadModelQuantProfile(path, nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = m.CloseWeights() })
		r := m.ResidentReport()
		const want = q8Bytes + embeddingBytes
		if r.Q8Tensors != 4 || r.Q8Bytes != q8Bytes || r.F32Bytes != embeddingBytes || r.TotalResidentBytes != want {
			t.Fatalf("fixture resident layout = %+v; want four Q8 projections (%d bytes, F32 scales), F32 embedding (%d bytes), total %d", r, q8Bytes, embeddingBytes, want)
		}
		if got := plan.Total(); got != r.TotalResidentBytes {
			t.Errorf("lean-q8 weight estimate mismatch: got %d bytes, actual stored weights %d (independent count %d)", got, r.TotalResidentBytes, want)
		}
	})

	t.Run("loader-options", func(t *testing.T) {
		mixedPath := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ2_K, TensorQ6_K, false, false)
		mixed, err := OpenWeights(mixedPath)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = mixed.Close() })
		const layerPacked = 3 * dim * dim / 256 * 144
		for _, tc := range []struct {
			name string
			opts []Q4KLoadOption
			want int64
		}{
			{"retained-kquant", nil, layerPacked + vocab*210 + embeddingBytes},
			{"q8-fallback", []Q4KLoadOption{WithDenseKQuantResident(false)}, layerPacked + vocab*dim/32*36 + embeddingBytes},
			{"packed-embedding", []Q4KLoadOption{WithQ2KEmbeddingResident(true)}, layerPacked + vocab*210 + vocab*84},
		} {
			t.Run(tc.name, func(t *testing.T) {
				plan, err := mixed.EstimateQ4KLoadMemoryPlan(tc.opts...)
				if err != nil {
					t.Fatal(err)
				}
				m, err := LoadModelQ4KProfileOptions(mixedPath, nil, tc.opts...)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = m.CloseWeights() })
				if got := m.ResidentReport().TotalResidentBytes; got != tc.want || plan.Total() != tc.want {
					t.Fatalf("stored=%d estimated=%d independent=%d", got, plan.Total(), tc.want)
				}
			})
		}
		if _, err := mixed.EstimateQ4KLoadMemoryPlan(WithStreamedExperts(0)); !errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
			t.Fatalf("streaming error=%v, want explicit unsupported", err)
		}
	})
}
