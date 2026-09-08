//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestVulkanQuantKVDirectFragment(t *testing.T) {
	plan, err := PlanVulkanQuantKVDirectFragment("gfx1151", QuantizedKVQ8_0, 64, 40, 8, 128)
	if err != nil {
		t.Fatalf("PlanVulkanQuantKVDirectFragment: %v", err)
	}
	if plan.FullCacheScratchBytes != 0 || plan.FullCacheScratchWrites != 0 {
		t.Fatalf("full-cache scratch = %d bytes/%d writes, want zero", plan.FullCacheScratchBytes, plan.FullCacheScratchWrites)
	}
	if plan.FallbackCount != 0 {
		t.Fatalf("fallback_count = %d, want zero", plan.FallbackCount)
	}
	if plan.LDSBytes > directFragmentLDSLimitBytes || plan.EstimatedVGPRPerLane > directFragmentVGPRLimit {
		t.Fatalf("resource plan exceeds admission: LDS=%d/%d VGPR=%d/%d",
			plan.LDSBytes, directFragmentLDSLimitBytes, plan.EstimatedVGPRPerLane, directFragmentVGPRLimit)
	}
	if _, err := PlanVulkanQuantKVDirectFragment("gfx1100", QuantizedKVQ8_0, 64, 40, 8, 128); err == nil {
		t.Fatal("non-gfx1151 plan admitted instead of failing closed")
	}
	if _, err := PlanVulkanQuantKVDirectFragment("gfx1151", QuantizedKVQ8_0, 64, 34, 2, 128); err == nil {
		t.Fatal("17-head GQA group admitted instead of failing closed")
	}

	shaderPath := filepath.Join("shaders", "flash_attn_dequant.comp")
	shader, err := os.ReadFile(shaderPath)
	if err != nil {
		t.Fatalf("read %s: %v", shaderPath, err)
	}
	src := string(shader)
	start := strings.Index(src, "FAK_DIRECT_FRAGMENT_BEGIN")
	end := strings.Index(src, "FAK_DIRECT_FRAGMENT_END")
	if start < 0 || end <= start {
		t.Fatal("direct-fragment shader markers missing")
	}
	direct := src[start:end]
	for _, forbidden := range []string{"OutputScratchpad", "scratch["} {
		if strings.Contains(direct, forbidden) {
			t.Fatalf("direct-fragment shader contains full-cache scratch token %q", forbidden)
		}
	}
	for _, required := range []string{
		"GL_KHR_cooperative_matrix", "dequantKV", "coopMatLoad", "coopMatMulAdd",
		"MODE_QK", "MODE_PV", "RawQuantizedKV", "FragmentOutput",
	} {
		if !strings.Contains(direct, required) {
			t.Fatalf("direct-fragment shader missing contract token %q", required)
		}
	}

	if glslc, err := exec.LookPath("glslc"); err == nil {
		spv := filepath.Join(t.TempDir(), "flash_attn_direct_fragment.spv")
		cmd := exec.Command(glslc, "-DFAK_DIRECT_FRAGMENT=1", "--target-env=vulkan1.2", "-fshader-stage=comp", shaderPath, "-o", spv)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("compile direct-fragment shader: %v\n%s", err, out)
		}
	} else {
		t.Log("glslc unavailable; structural shader contract checked")
	}

	const (
		nPos    = 16
		nQ      = 4
		nKV     = 2
		headDim = 32
	)
	rng := rand.New(rand.NewSource(12228))
	q := randomUnitValues(rng, nQ*headDim)
	f32K := randomUnitValues(rng, nKV*nPos*headDim)
	f32V := randomUnitValues(rng, nKV*nPos*headDim)

	for _, format := range []QuantizedKVType{QuantizedKVQ8_0, QuantizedKVQ4_0, QuantizedKVQ4_K} {
		t.Run(string(format), func(t *testing.T) {
			rawK, rawV := quantizeDirectFragmentFixture(t, format, f32K, f32V)
			refK := make([]float32, len(f32K))
			refV := make([]float32, len(f32V))
			if err := DequantizeQuantizedKV(refK, rawK, len(refK), format); err != nil {
				t.Fatalf("dequantize reference K: %v", err)
			}
			if err := DequantizeQuantizedKV(refV, rawV, len(refV), format); err != nil {
				t.Fatalf("dequantize reference V: %v", err)
			}
			want := cpuReferenceMultiHeadAttention(q, refK, refV, nPos, nQ, nKV, headDim)
			got, receipt, err := ExecuteVulkanQuantKVDirectFragmentReference(q, rawK, rawV, "gfx1151", nPos, nQ, nKV, headDim, format)
			if err != nil {
				t.Fatalf("ExecuteVulkanQuantKVDirectFragmentReference: %v", err)
			}
			if receipt.FullCacheScratchWrites != 0 || receipt.FallbackCount != 0 {
				t.Fatalf("receipt scratch_writes=%d fallback_count=%d, want zero", receipt.FullCacheScratchWrites, receipt.FallbackCount)
			}
			if cosine := CosineSimilarity(got, want); cosine < 0.999 {
				t.Fatalf("attention cosine = %.8f, want >= 0.999", cosine)
			}
			for i := range got {
				if delta := float32(math.Abs(float64(got[i] - want[i]))); delta > 1e-5 {
					t.Fatalf("output[%d] delta = %g, want <= 1e-5", i, delta)
				}
			}
		})
	}
}

func randomUnitValues(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

func quantizeDirectFragmentFixture(t *testing.T, format QuantizedKVType, k, v []float32) ([]byte, []byte) {
	t.Helper()
	var rawK, rawV []byte
	var err error
	switch format {
	case QuantizedKVQ8_0:
		rawK, err = QuantizeF32ToQ8_0(k)
		if err == nil {
			rawV, err = QuantizeF32ToQ8_0(v)
		}
	case QuantizedKVQ4_0:
		rawK, err = QuantizeF32ToQ4_0(k)
		if err == nil {
			rawV, err = QuantizeF32ToQ4_0(v)
		}
	case QuantizedKVQ4_K:
		rawK = directFragmentQ4KFixture(len(k), 17)
		rawV = directFragmentQ4KFixture(len(v), 53)
	default:
		t.Fatalf("unsupported fixture format %q", format)
	}
	if err != nil {
		t.Fatalf("quantize %s fixture: %v", format, err)
	}
	return rawK, rawV
}

func directFragmentQ4KFixture(elements, seed int) []byte {
	raw := make([]byte, QuantizedKVTotalBytes(QuantizedKVQ4_K, elements))
	for base := 0; base < len(raw); base += 144 {
		binary.LittleEndian.PutUint16(raw[base:base+2], Float32ToFloat16Bits(0.01))
		binary.LittleEndian.PutUint16(raw[base+2:base+4], Float32ToFloat16Bits(0.003))
		for i := 4; i < 144; i++ {
			raw[base+i] = byte((base + i*seed) % 251)
		}
	}
	return raw
}
