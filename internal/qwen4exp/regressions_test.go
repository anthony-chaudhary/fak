package qwen4exp

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestFP8PLEDequantizesExactRowsAndRejectsNaN(t *testing.T) {
	// Transformers #48349-#48351: PLE rows can be FP8-quantized independently.
	got, err := DequantizeFP8E4M3FNPLE([]byte{0x00, 0x01, 0x38, 0x3c, 0x40, 0xb8, 0x7e}, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 1.0 / 256, 2, 3, 4, -2, 896}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got=%v want=%v", got, want)
	}
	if _, err := DequantizeFP8E4M3FNPLE([]byte{0x7f}, 1); err == nil {
		t.Fatal("FP8 NaN accepted")
	}
}

func TestTextModelExclusionsArePrefixBoundaries(t *testing.T) {
	excluded := []string{"visual", "vision_tower."}
	for name, want := range map[string]bool{"visual.patch_embed": false, "vision_tower.blocks.0": false, "model.layers.0.visual_gate": true, "language_model.layers.0": true} {
		if got := TextModelTensor(name, excluded); got != want {
			t.Errorf("%s got=%v want=%v", name, got, want)
		}
	}
}

func TestAbsentTPPlanIsSafeAndCopied(t *testing.T) {
	if got := NormalizeTPPlan(nil); got == nil || len(got) != 0 {
		t.Fatalf("nil plan normalized to %#v", got)
	}
	in := map[string]string{"q_proj": "colwise"}
	got := NormalizeTPPlan(in)
	got["q_proj"] = "changed"
	if in["q_proj"] == got["q_proj"] {
		t.Fatal("caller plan aliased")
	}
}

func TestThinkingToolParserTokenZeroLoopHalts(t *testing.T) {
	var g ToolTokenGuard
	if err := g.Observe(0, false, true); err != nil {
		t.Fatal(err)
	}
	if err := g.Observe(0, false, true); err == nil {
		t.Fatal("token-zero loop accepted")
	}
	if err := g.Observe(0, true, true); err != nil {
		t.Fatal(err)
	}
	if err := g.Observe(0, false, true); err != nil {
		t.Fatal("progress did not reset guard")
	}
}

func TestIncompatibleQSAGDNKernelPairRefusesTyped(t *testing.T) {
	if err := ValidateQSAAndGDNKernels("cuda-exact", "cuda-exact"); err != nil {
		t.Fatal(err)
	}
	err := ValidateQSAAndGDNKernels("cuda-exact", "metal-exact")
	if !errors.Is(err, ErrUnsupportedKernelPair) {
		t.Fatalf("err=%v", err)
	}
}

func TestFP8PLEAllFiniteEncodingsStayFinite(t *testing.T) {
	for i := 0; i < 256; i++ {
		if i&0x7f == 0x7f {
			continue
		}
		got, err := DequantizeFP8E4M3FNPLE([]byte{byte(i)}, 1)
		if err != nil || math.IsNaN(float64(got[0])) || math.IsInf(float64(got[0]), 0) {
			t.Fatalf("bits=%02x got=%v err=%v", i, got, err)
		}
	}
}
