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
