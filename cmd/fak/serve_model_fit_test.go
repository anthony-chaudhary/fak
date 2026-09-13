package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func TestServeQ4KWeightAdmissionUsesTransformedPlan(t *testing.T) {
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	path := writeServePackedEmbeddingFixture(t, "mixed-qwen.gguf", false)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	arm := resolveMetalServeLoadArm(ws)
	if arm != serveLoadArmResidentQ4K {
		t.Fatalf("actual header selected %q, want resident-Q4K", arm)
	}
	m, err := ggufload.LoadModelQ4KProfileOptions(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.CloseWeights() })
	// Three 256x256 Q4_K projections, a three-row Q6_K head, and
	// an embedding expanded to F32 by the loader's default options.
	const want = 3*256*144 + 3*210 + 3*256*4
	if got := m.ResidentReport().TotalResidentBytes; got != want {
		t.Fatalf("actual stored weights=%d, independent count=%d", got, want)
	}
	plan, err := serveGGUFWeightMemoryPlanForArm(ws, arm)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Total() != want {
		t.Errorf("central weight admission=%d, actual transformed storage=%d", plan.Total(), want)
	}
	raw, err := ws.EstimateLoadBytes()
	if err != nil || raw >= want {
		t.Fatalf("fixture must distinguish raw payload from storage: raw=%d stored=%d error=%v", raw, want, err)
	}

	// Streaming is outside the estimator's qualified transformed-storage route.
	// Per the #11962 contract it retains the prior conservative raw-payload
	// estimate rather than an exactness claim, so the live streamed Metal serve
	// keeps working instead of failing the pre-load fit.
	t.Setenv("FAK_STREAM_Q4K", "1")
	err = fitServeGGUFPathOnReportedHostForArm(path, arm, 16, 0, 0, false, nil)
	if err != nil {
		t.Errorf("streamed resident admission must retain the conservative fallback, got %v", err)
	}
	streamedPlan, err := serveGGUFWeightMemoryPlanForArm(ws, arm, serveQ4KFitOptions(path, ws, nil, arm)...)
	if err != nil {
		t.Fatalf("streamed conservative fallback: %v", err)
	}
	if streamedPlan.Total() != raw {
		t.Errorf("streamed fallback total=%d, want the prior raw payload estimate %d", streamedPlan.Total(), raw)
	}
}

func TestServeResidentQ4KSharedOptionsMatchStoredWeights(t *testing.T) {
	original := serveMetalAvailable
	serveMetalAvailable = func() bool { return true }
	t.Cleanup(func() { serveMetalAvailable = original })
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	t.Setenv("FAK_GGUF_MMAP", "0")
	for _, tc := range []struct {
		name   string
		q2     bool
		device bool
		want   int64
	}{
		{"packed-q2", true, false, 2*256*144 + 256*84 + 3*210 + 3*84},
		{"packed-q4", false, false, 3*256*144 + 3*210 + 3*144},
		{"explicit-backend-q8-fallback", true, true, 2*256*144 + 256*256/32*36 + 3*256/32*36 + 3*256*4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var path string
			if tc.q2 {
				path = writeServePackedQ2EmbeddingFixture(t, "Qwen3.8-27B-UD-Q2_K_XL.gguf", true)
			} else {
				path = writeServePackedEmbeddingFixture(t, "Qwen3.8-27B-Q4_K_M.gguf", true)
			}
			ws, err := ggufload.OpenWeights(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ws.Close() })
			var backend compute.Backend
			if tc.device {
				backend = compute.Default()
			}
			artifact := ggufload.ClassifyTensorQuant(ws.File.Tensors)
			loadOpts := serveResidentQ4KLoadOptions(backend, path, true, artifact)
			m, err := ggufload.LoadModelQ4KProfileOptions(path, nil, loadOpts...)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = m.CloseWeights() })
			fitOpts := serveQ4KFitOptions(path, ws, backend, serveLoadArmResidentQ4K)
			plan, err := serveGGUFWeightMemoryPlanForArm(ws, serveLoadArmResidentQ4K, fitOpts...)
			if err != nil {
				t.Fatal(err)
			}
			if got := m.ResidentReport().TotalResidentBytes; got != tc.want || plan.Total() != tc.want {
				t.Fatalf("loader=%d admission=%d independent=%d", got, plan.Total(), tc.want)
			}
		})
	}
}

// The fixture is a physical, loadable storage slice with complete architecture
// metadata and payloads. It does not claim full-model generation qualification.
func TestServeResidentDenseStandardArchitectureAdmissionMatchesLoader(t *testing.T) {
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	t.Setenv("FAK_W3_MLP", "")
	t.Setenv("FAK_GGUF_MMAP", "0")
	for _, arch := range []string{"llama", "qwen2"} {
		t.Run(arch, func(t *testing.T) {
			path := createTestQ4KGGUF(t)
			if arch == "qwen2" {
				b, err := os.ReadFile(path)
				if err != nil {
					t.Fatal(err)
				}
				// Both architecture names have five bytes. Replace the architecture and
				// its seven metadata prefixes without changing directory/payload offsets.
				if got := bytes.Count(b, []byte("llama")); got != 8 {
					t.Fatalf("fixture architecture occurrences=%d, want 8", got)
				}
				b = bytes.ReplaceAll(b, []byte("llama"), []byte("qwen2"))
				if err := os.WriteFile(path, b, 0600); err != nil {
					t.Fatal(err)
				}
			}
			ws, err := ggufload.OpenWeights(path)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = ws.Close() })
			cfg, err := ws.File.Config()
			if err != nil {
				t.Fatal(err)
			}
			if cfg.ModelType != arch || cfg.TieWordEmbeddings || cfg.IsMoE() {
				t.Fatalf("expected untied dense %s configuration, got %+v", arch, cfg)
			}
			arm := resolveMetalServeLoadArm(ws)
			if arm != serveLoadArmResidentQ4K {
				t.Fatalf("selected load arm=%q, want resident Q4_K", arm)
			}
			opts := serveResidentQ4KLoadOptions(nil, path, true, ggufload.ClassifyTensorQuant(ws.File.Tensors))
			m, err := ggufload.LoadModelQ4KProfileOptions(path, nil, opts...)
			if err != nil {
				t.Fatalf("actual supported loader failed before admission comparison: %v", err)
			}
			t.Cleanup(func() { _ = m.CloseWeights() })
			// Three matrices remain Q4_K; rotary-sensitive q_proj is normalized into
			// native Q8 with F32 scales (36 bytes per 32 values, not GGUF's 34).
			const wantStored int64 = 3*256*144 + 256*256/32*36
			report := m.ResidentReport()
			if report.Q4KTensors != 3 || report.Q8Tensors != 1 || !m.HasQ8("model.layers.0.self_attn.q_proj.weight") || report.TotalResidentBytes != wantStored {
				t.Fatalf("actual loader report=%+v, want three Q4/one normalized Q8 and %d bytes", report, wantStored)
			}
			raw, err := ws.EstimateLoadBytes()
			if err != nil || raw != 4*256*144 {
				t.Fatalf("raw bytes=%d error=%v, want147456", raw, err)
			}
			t.Logf("actual %s loader succeeded: raw=%d stored=%d Q4=%d Q8=%d", arch, raw, report.TotalResidentBytes, report.Q4KTensors, report.Q8Tensors)
			plan, err := serveGGUFWeightMemoryPlanForArm(ws, arm, serveQ4KFitOptions(path, ws, nil, arm)...)
			if err != nil {
				t.Fatalf("admission rejected supported %s loader storage (%d bytes): %v", arch, wantStored, err)
			}
			if plan.Total() != wantStored {
				t.Fatalf("admission=%d actual stored=%d", plan.Total(), wantStored)
			}
		})
	}
}

// writeServePackedQ2EmbeddingFixture writes the smallest dense Qwen hybrid GGUF
// whose token_embd.weight is Q2_K (matching the UD-Q2_K_XL artifact) plus a Q4_K
// tensor so the artifact stays on the resident-Q4_K serve arm. It optionally
// truncates the file to the exact witnessed UD-Q2_K_XL byte size so the
// file-name/size gate admits it, without allocating or reading a 27B checkpoint.
func writeServePackedQ2EmbeddingFixture(t *testing.T, base string, witnessedSize bool) string {
	t.Helper()
	const (
		dim          = 256
		vocab        = 3
		q2BlockBytes = 84
		q4BlockBytes = 144
		q6BlockBytes = 210
	)
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
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{dim, vocab}, uint32(ggufload.TensorQ2_K), offset)
	offset = align32(offset + vocab*q2BlockBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, vocab}, uint32(ggufload.TensorQ6_K), offset)
	offset = align32(offset + vocab*q6BlockBytes)
	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), offset)
	offset = align32(offset + dim*q4BlockBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_up.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ2_K), offset)
	offset = align32(offset + dim*q2BlockBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_down.weight", []uint64{dim, dim}, uint32(ggufload.TensorQ4_K), offset)
	offset = align32(offset + dim*q4BlockBytes)
	padToAlignmentForTest(&b, 32)
	b.Write(make([]byte, int(offset)))

	path := filepath.Join(t.TempDir(), base)
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if witnessedSize {
		if err := os.Truncate(path, qwen38UDQ2KXLArtifactBytes); err != nil {
			t.Fatal(err)
		}
	}
	return path
}
