//go:build vulkan && (windows || linux) && cgo

package model

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func requiredVulkanSequenceBackend(t *testing.T) (compute.Backend, Qwen35SequencePrefillBackend) {
	t.Helper()
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required physical Vulkan backend is unavailable")
		}
		t.Skip("Vulkan unavailable; FAK_VULKAN_REQUIRE_DEVICE=1 prohibits this skip")
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(be.Tier()), strings.ToLower(expected)) {
		t.Fatalf("Vulkan device %q does not match %q", be.Tier(), expected)
	}
	sequence, ok := be.(Qwen35SequencePrefillBackend)
	if !ok || sequence.Qwen35SequencePrefillPath() != compute.Qwen35SequencePrefillPath {
		t.Fatalf("physical Vulkan backend %T lacks whole-sequence prefill", be)
	}
	t.Logf("engine=fak-native backend=%s device=%s path=%s", be.Name(), be.Tier(), sequence.Qwen35SequencePrefillPath())
	return be, sequence
}

func compareVulkanSequenceVector(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) || len(want) == 0 {
		t.Fatalf("%s lengths got=%d want=%d", label, len(got), len(want))
	}
	var dot, gn, wn, maxAbs float64
	for i := range want {
		g, w := float64(got[i]), float64(want[i])
		if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("%s[%d] nonfinite got=%g want=%g", label, i, g, w)
		}
		dot += g * w
		gn += g * g
		wn += w * w
		maxAbs = math.Max(maxAbs, math.Abs(g-w))
		if math.Abs(g-w) > 2e-3+2e-3*math.Abs(w) {
			t.Fatalf("%s[%d]=%.8g want %.8g", label, i, g, w)
		}
	}
	if gn > 0 && wn > 0 {
		c := dot / math.Sqrt(gn*wn)
		if c < compute.Qwen35SequenceParityCosineMin {
			t.Fatalf("%s cosine=%.9f", label, c)
		}
		t.Logf("%s cosine=%.9f max_abs=%g", label, c, maxAbs)
	}
}

func TestVulkanQwen35SequencePrefillMatchesCPUAndPersistsDecodeState(t *testing.T) {
	be, sequence := requiredVulkanSequenceBackend(t)
	cfg := qwen35HybridTestCfg()
	cfg.PartialRotaryFactor = .5
	m := NewSynthetic(cfg)
	cpu := m.NewSession()
	device, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	invalid := device.qwen35SequencePrefillRequest([]int{3, m.Cfg.VocabSize}, true)
	_, err = sequence.Qwen35SequencePrefill(invalid)
	var sequenceErr *compute.Qwen35SequenceError
	if !errors.As(err, &sequenceErr) || device.halKV.Len() != 0 {
		t.Fatalf("invalid token must fail before KV mutation: err=%v kv=%d", err, device.halKV.Len())
	}
	// Two panels exercise both zero-origin and nonzero-prefix attention, while
	// seven initial tokens fill and roll the three-tap recurrent convolution.
	for panel, ids := range [][]int{{3, 7, 11, 5, 17, 19, 23}, {29, 31, 37}} {
		var want []float32
		for _, id := range ids {
			// Step reuses its logits buffer; preserve the oracle before another step.
			want = append([]float32(nil), cpu.Step(id)...)
		}
		req := device.qwen35SequencePrefillRequest(ids, panel != 0)
		start := device.halKV.Len()
		handles := make([][2]compute.Buffer, len(req.States))
		for l, st := range req.States {
			handles[l] = [2]compute.Buffer{st.Conv.Buf(), st.Recurrent.Buf()}
		}
		result, err := sequence.Qwen35SequencePrefill(req)
		if err != nil {
			t.Fatalf("panel %d: %v", panel, err)
		}
		if result.Tokens != len(ids) || device.halKV.Len() != start+len(ids) || !result.LastHidden.Ready() || (req.NeedLogits && !result.Logits.Ready()) {
			t.Fatalf("invalid panel result tokens=%d kv=%d start=%d", result.Tokens, device.halKV.Len(), start)
		}
		logits := result.Logits
		if !req.NeedLogits {
			if logits.Ready() {
				t.Fatal("no-logits sequence unexpectedly materialized logits")
			}
			// Read back the normalized hidden result through the ordinary head;
			// the no-logits sequence must preserve exactly the same model state.
			logits = be.MatMul(req.Output, result.LastHidden)
		}
		got := be.Read(logits)
		compareVulkanSequenceVector(t, fmt.Sprintf("panel_%d_logits", panel), got, want)
		if argmaxF32(got) != argmaxF32(want) {
			t.Fatalf("panel %d greedy token mismatch", panel)
		}
		for l, st := range req.States {
			if !req.Layers[l].Linear {
				continue
			}
			if handles[l] != [2]compute.Buffer{st.Conv.Buf(), st.Recurrent.Buf()} {
				t.Fatalf("layer %d persistent state handle changed", l)
			}
			oracle := cpu.Cache.linear.layers[l]
			var wantConv, wantRecurrent []float32
			for _, row := range oracle.conv {
				wantConv = append(wantConv, row...)
			}
			for _, row := range oracle.recurrent {
				wantRecurrent = append(wantRecurrent, row...)
			}
			compareVulkanSequenceVector(t, fmt.Sprintf("panel_%d_layer_%d_conv", panel, l), be.Read(st.Conv), wantConv)
			compareVulkanSequenceVector(t, fmt.Sprintf("panel_%d_layer_%d_recurrent", panel, l), be.Read(st.Recurrent), wantRecurrent)
		}
		if recycler, ok := be.(interface{ Recycle() }); ok {
			recycler.Recycle()
		}
	}
	want := append([]float32(nil), cpu.Step(41)...)
	got := device.Step(41)
	compareVulkanSequenceVector(t, "decode_after_sequence", got, want)
	if argmaxF32(got) != argmaxF32(want) || device.halKV.Len() != 11 {
		t.Fatalf("decode continuity failed: kv=%d", device.halKV.Len())
	}
}

func TestVulkanQwen35SequencePrefillQ2KEmbeddingRowsMatchCPU(t *testing.T) {
	be, _ := requiredVulkanSequenceBackend(t)
	rows, ok := be.(compute.Qwen35SequenceEmbeddingRowsBackend)
	if !ok || rows.Qwen35SequenceEmbeddingRowsPath() != compute.Qwen35SequenceEmbeddingRowsPath {
		t.Fatalf("physical Vulkan backend %T lacks exact embedding-row capability", be)
	}

	cfg := qwen35HybridTestCfg()
	cfg.HiddenSize = 256
	cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim = 4, 2, 64
	cfg.IntermediateSize = 512
	cfg.LinearKeyHeadDim, cfg.LinearNumKeyHeads = 64, 2
	cfg.LinearValueHeadDim, cfg.LinearNumValueHeads = 64, 4
	cfg.VocabSize = 4097
	m := NewSynthetic(cfg)
	q2k, err := NewQ2KEmbedding(makeTestQ2KPayload(cfg.VocabSize, cfg.HiddenSize), cfg.VocabSize, cfg.HiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	dequantized, err := q2k.DequantizeTable()
	if err != nil {
		t.Fatal(err)
	}
	// The independent CPU oracle uses an ordinary f32 embedding containing the
	// exact Q2_K dequantization. Keep the same values as the device output head,
	// so the only route difference under test is bounded row-panel admission.
	oracle := NewSynthetic(cfg)
	copy(oracle.tensor("model.embed_tokens.weight"), dequantized)
	copy(m.tensor("model.embed_tokens.weight"), dequantized)
	m.Q2KEmbedding = q2k
	// Preserve the synthetic untied output matrix while proving that the input
	// embedding has no f32 vocabulary table available to the device route.
	m.manifest["lm_head.weight"] = m.manifest["model.embed_tokens.weight"]
	delete(m.manifest, "model.embed_tokens.weight")

	ids := []int{0, cfg.VocabSize / 2, cfg.VocabSize - 1, cfg.VocabSize / 2}
	cpu := oracle.NewSession()
	var want []float32
	for _, id := range ids {
		want = append([]float32(nil), cpu.Step(id)...)
	}
	device, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()
	result, used, err := device.tryQwen35SequencePrefill(ids, true)
	if err != nil || !used {
		t.Fatalf("Q2_K row-panel sequence used=%t err=%v", used, err)
	}
	got := be.Read(result.Logits)
	compareVulkanSequenceVector(t, "q2k_embedding_rows_logits", got, want)
	if argmaxF32(got) != argmaxF32(want) {
		t.Fatal("Q2_K row-panel greedy token mismatch")
	}
	status, present := device.Qwen35SequencePrefillRouteStatus()
	wantBytes := int64(len(ids) * cfg.HiddenSize * compute.F32.Bytes())
	productionTableBytes, valid := f32TensorBytes([]int{248320, 5120})
	if !valid || wantBytes >= productionTableBytes {
		t.Fatalf("invalid bounded memory envelope panel=%d production_table=%d valid=%t", wantBytes, productionTableBytes, valid)
	}
	if !present || status.EffectivePath != compute.Qwen35SequencePrefillPath || status.FallbackActive || !status.NativePerformanceQualifying || !status.PackedEmbeddingRows || status.EmbeddingPanelBytes != wantBytes {
		t.Fatalf("Q2_K Vulkan route status=%+v present=%t, want native %d-byte panel", status, present, wantBytes)
	}
	if result.Transfers.H2DBytes != 0 || result.Transfers.D2HBytes != 0 || result.Transfers.ActivationH2DBytes != 0 || result.Transfers.ActivationD2HBytes != 0 {
		t.Fatalf("sequence transferred host activations after row-panel admission: %+v", result.Transfers)
	}
	if _, ok := m.manifest["model.embed_tokens.weight"]; ok {
		t.Fatal("physical route unexpectedly retained an f32 input embedding table")
	}
	maxBufferBytes := int64(0)
	if caps, ok := be.(interface {
		VulkanDebugResourceCaps() (int64, int64, int64)
	}); ok {
		maxBufferBytes, _, _ = caps.VulkanDebugResourceCaps()
	}
	t.Logf("model_fixture=synthetic-qwen35-hybrid-q2k-v1 q2k_rows=%d vocab=%d hidden=%d embedding_panel_bytes=%d production_f32_table_bytes=%d device_max_buffer_bytes=%d transfers=%+v source_rev=%s", len(ids), cfg.VocabSize, cfg.HiddenSize, wantBytes, productionTableBytes, maxBufferBytes, result.Transfers, os.Getenv("FAK_VULKAN_SOURCE_REV"))
}
