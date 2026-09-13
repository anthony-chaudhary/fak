package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// Darwin's existing host-capacity overrides make the real startup stage's
// context boundary deterministic. The software loader is the storage witness;
// overriding Metal availability selects the route without executing GPU kernels.
func TestServeRuntimeContextPreservesPackedEmbeddingPath(t *testing.T) {
	original := serveMetalAvailable
	serveMetalAvailable = func() bool { return true }
	t.Cleanup(func() { serveMetalAvailable = original })
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	t.Setenv("FAK_W3_MLP", "")
	t.Setenv("FAK_GGUF_MMAP", "0")
	t.Setenv("FAK_HYBRID_KV", "0")
	const (
		// Two Q4 matrices, one Q2 matrix, and a Q6 output head.
		projections = int64(2*256*144 + 256*84 + 3*210)
		packed      = projections + 3*84
		expanded    = projections + 3*256*4
		// Three recurrent layers: one 2x2 state and one convolution row
		// containing two key vectors and one value vector, all F32.
		fixed = int64(3 * (1*2*2 + (2*2 + 2)) * 4)
		// Four layers with twelve 256-element transient outputs, final
		// normalization, residual activation, and three full logits.
		scratch  = int64((4*12*256+256)*4 + (256+3)*4)
		perToken = int64(1 * 1 * 256 * 3 * 4) // one full-attention K/Kraw/V layer
		avail    = packed + fixed + scratch + 1024*perToken
		base     = (avail*100 + 84) / 85 // ceil(avail / 0.85)
	)
	if got := compute.BudgetAfterHeadroom(base, 0.15); got != avail {
		t.Fatalf("fixture headroom rounding: got %d want %d", got, avail)
	}
	t.Setenv("FAK_UP_MEMORY_BYTES", strconv.FormatInt(base, 10))
	t.Setenv("FAK_UP_AVAILABLE_BYTES", strconv.FormatInt(base, 10))
	for _, exact := range []bool{true, false} {
		name, filename, wantBytes, wantTokens := "ordinary-name", "ordinary-q2.gguf", expanded, 1023
		if exact {
			name, filename, wantBytes, wantTokens = "qualified-name-size", "Qwen3.8-27B-UD-Q2_K_XL.gguf", packed, 1024
		}
		t.Run(name, func(t *testing.T) {
			path := writeServeRuntimeQ2ContextFixture(t, filename, exact)
			ws, err := ggufload.OpenWeights(path)
			if err != nil {
				t.Fatal(err)
			}
			defer ws.Close()
			cfg, err := ws.File.Config()
			if err != nil {
				t.Fatalf("fixture Config: %v", err)
			}
			csc := cfg.ContextSizeConfig()
			if csc.MaxContext != 4096 || csc.SessionState.DeviceTotal() != fixed ||
				compute.EstimateHALTransientMemoryPlan(csc.Scratch).Total() != scratch ||
				compute.EstimateKVStoreBytes(csc.KV, 1) != perToken {
				t.Fatalf("fixture geometry disagrees with independent arithmetic: %+v", csc)
			}
			opts := serveResidentQ4KLoadOptions(nil, path, true, ggufload.ClassifyTensorQuant(ws.File.Tensors))
			model, err := ggufload.LoadModelQ4KProfileOptions(path, nil, opts...)
			if err != nil {
				t.Fatalf("actual software load: %v", err)
			}
			defer model.CloseWeights()
			actual := model.ResidentReport().TotalResidentBytes
			if actual != wantBytes || (model.Q2KEmbedding != nil) != exact {
				t.Fatalf("actual storage=%d packed=%v want storage=%d packed=%v", actual, model.Q2KEmbedding != nil, wantBytes, exact)
			}
			if expected := (avail - actual - fixed - scratch) / perToken; expected != int64(wantTokens) || expected < 512 {
				t.Fatalf("independent context boundary=%d want %d", expected, wantTokens)
			}
			_, sf := newServeFlagSet()
			*sf.ggufPath = path
			*sf.nativeContextTokens = 0
			rt := &serveRuntime{useMetal: true}
			if err := rt.resolveNativeContext(sf, 1); err != nil {
				t.Fatalf("real runtime stage: %v", err)
			}
			t.Logf("actual storage=%d runtime context=%d expected=%d exact=%v base=%d available=%d", actual, rt.nativeContext.ResolvedTokens, wantTokens, exact, base, avail)
			if rt.nativeContext.ResolvedTokens != wantTokens || rt.nativeContext.RequestedTokens != 0 ||
				rt.nativeContext.ModelDeclaredTokens != 4096 || rt.nativeContext.Source != "auto" {
				t.Errorf("runtime context=%+v want auto/%d declared4096", rt.nativeContext, wantTokens)
			}
			if *sf.nativeAdmissionTokenBudget != wantTokens || sf.nativeAdmissionProvenance != "context" {
				t.Errorf("scheduler budget=%d provenance=%q want %d/context", *sf.nativeAdmissionTokenBudget, sf.nativeAdmissionProvenance, wantTokens)
			}
			if rt.fitBudget == nil || rt.fitBudget.Base != base || rt.fitBudget.Headroom != 0.15 || rt.fitBudget.avail() != avail {
				t.Errorf("retained fit snapshot=%+v want base=%d headroom=0.15 available=%d", rt.fitBudget, base, avail)
			}
		})
	}
}

func writeServeRuntimeQ2ContextFixture(t *testing.T, base string, qualified bool) string {
	t.Helper()
	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, 5, 16)
	writeUpBackendAutoStringArray(&b, "tokenizer.ggml.tokens", []string{"a", "b", "c"})
	writeKVStringForTest(&b, "general.architecture", "qwen35")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "qwen35.embedding_length", 256)
	writeKVUint32ForTest(&b, "qwen35.block_count", 4)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count", 1)
	writeKVUint32ForTest(&b, "qwen35.attention.head_count_kv", 1)
	writeKVUint32ForTest(&b, "qwen35.full_attention_interval", 4)
	writeKVUint32ForTest(&b, "qwen35.attention.key_length", 256)
	writeKVUint32ForTest(&b, "qwen35.feed_forward_length", 256)
	writeKVFloat32ForTest(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVUint32ForTest(&b, "qwen35.context_length", 4096)
	writeKVUint32ForTest(&b, "qwen35.ssm.conv_kernel", 2)
	writeKVUint32ForTest(&b, "qwen35.ssm.state_size", 2)
	writeKVUint32ForTest(&b, "qwen35.ssm.group_count", 1)
	writeKVUint32ForTest(&b, "qwen35.ssm.time_step_rank", 1)
	var offset uint64
	for _, tensor := range []struct {
		name  string
		rows  uint64
		kind  uint32
		block uint64
	}{
		{"token_embd.weight", 3, uint32(ggufload.TensorQ2_K), 84},
		{"output.weight", 3, uint32(ggufload.TensorQ6_K), 210},
		{"blk.0.attn_v.weight", 256, uint32(ggufload.TensorQ4_K), 144},
		{"blk.0.ffn_up.weight", 256, uint32(ggufload.TensorQ2_K), 84},
		{"blk.0.ffn_down.weight", 256, uint32(ggufload.TensorQ4_K), 144},
	} {
		writeTensorInfoForTest(&b, tensor.name, []uint64{256, tensor.rows}, tensor.kind, offset)
		offset = (offset + tensor.rows*tensor.block + 31) &^ 31
	}
	padToAlignmentForTest(&b, 32)
	b.Write(make([]byte, int(offset)))
	path := filepath.Join(t.TempDir(), base)
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if qualified {
		if err := os.Truncate(path, qwen38UDQ2KXLArtifactBytes); err != nil {
			t.Fatal(err)
		}
	}
	return path
}
