//go:build vulkan && (windows || linux) && cgo

package model

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const vulkanGDNRequiredEnv = "FAK_VULKAN_GDN_REQUIRED"

type vulkanGDNParityBackend interface {
	compute.Backend
	Qwen35GDNBackend
}

func requiredVulkanGDNParityBackend(t *testing.T) vulkanGDNParityBackend {
	t.Helper()
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" {
		if os.Getenv(vulkanGDNRequiredEnv) == "1" {
			t.Fatalf("%s=1: real Vulkan GDN parity is required, but exact backend vulkan is not registered", vulkanGDNRequiredEnv)
		}
		t.Skip("exact vulkan backend not registered")
	}
	gdn, ok := be.(vulkanGDNParityBackend)
	if !ok {
		t.Fatalf("registered vulkan backend %T lacks the GDN decode contract", be)
	}
	if got := gdn.Qwen35GDNPath(); got != Qwen35GDNVulkanPath {
		t.Fatalf("Vulkan GDN path = %q, want %q", got, Qwen35GDNVulkanPath)
	}
	return gdn
}

func compareVulkanGDNProductionVector(t *testing.T, label string, want, got []float32, maxAbsLimit float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d", label, len(got), len(want))
	}
	var dot, wantNorm2, gotNorm2, maxAbs float64
	for i := range want {
		w, g := float64(want[i]), float64(got[i])
		if math.IsNaN(w) || math.IsInf(w, 0) || math.IsNaN(g) || math.IsInf(g, 0) {
			t.Fatalf("%s[%d] must be finite: got=%g want=%g", label, i, got[i], want[i])
		}
		dot += w * g
		wantNorm2 += w * w
		gotNorm2 += g * g
		if d := math.Abs(w - g); d > maxAbs {
			maxAbs = d
		}
	}
	if wantNorm2 == 0 || gotNorm2 == 0 {
		t.Fatalf("%s has degenerate norm: got=%g want=%g", label, math.Sqrt(gotNorm2), math.Sqrt(wantNorm2))
	}
	cosine := dot / math.Sqrt(wantNorm2*gotNorm2)
	if cosine < Qwen35GDNParityCosineMin || maxAbs > maxAbsLimit {
		t.Fatalf("%s parity: cosine=%0.9f max_abs=%0.3e, want cosine >= %0.3f max_abs <= %0.3e",
			label, cosine, maxAbs, Qwen35GDNParityCosineMin, maxAbsLimit)
	}
	t.Logf("%s cosine=%0.9f max_abs=%0.3e", label, cosine, maxAbs)
}

func flattenVulkanGDNState(rows [][]float32) []float32 {
	var n int
	for _, row := range rows {
		n += len(row)
	}
	out := make([]float32, 0, n)
	for _, row := range rows {
		out = append(out, row...)
	}
	return out
}

func deterministicVulkanGDNVector(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = (rng.Float32()*2 - 1) * 0.2
	}
	return out
}

// TestVulkanQwen35GDNMatchesProductionCPUMultiStep is independent of the
// compute-package shader oracle: its reference is Session.linearAttnStep, the
// production CPU path. It starts from nonzero carried state and compares three
// sequential state transitions, projected hidden output, and vocabulary logits.
func TestVulkanQwen35GDNMatchesProductionCPUMultiStep(t *testing.T) {
	t.Setenv("FAK_DISABLE_VECTOR_GDN", "0")
	be := requiredVulkanGDNParityBackend(t)
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	p := func(suffix string) string { return layerName(0, suffix) }
	nK, nV, kHd, vHd, _, valDim, convDim := cfg.linearAttnDims()
	hidden, kernel := cfg.HiddenSize, cfg.LinearConvKernelDim

	upload := func(shape []int, data []float32, class compute.MemoryClass, site string) compute.Tensor {
		resident := uploadHostF32Class(be, shape, data, class, site)
		t.Cleanup(func() { be.Free(resident) })
		return resident
	}
	weight := func(name string, shape []int) compute.Tensor {
		return upload(shape, m.tensor(name), compute.MemoryWeights, "qwen35-gdn-weight "+name)
	}

	inQKV := weight(p("linear_attn.in_proj_qkv.weight"), []int{convDim, hidden})
	inZ := weight(p("linear_attn.in_proj_z.weight"), []int{valDim, hidden})
	inB := weight(p("linear_attn.in_proj_b.weight"), []int{nV, hidden})
	inA := weight(p("linear_attn.in_proj_a.weight"), []int{nV, hidden})
	convW := weight(p("linear_attn.conv1d.weight"), []int{convDim, 1, kernel})
	aLog := weight(p("linear_attn.A_log"), []int{nV})
	dtBias := weight(p("linear_attn.dt_bias"), []int{nV})
	norm := weight(p("linear_attn.norm.weight"), []int{vHd})
	outW := weight(p("linear_attn.out_proj.weight"), []int{hidden, valDim})

	convSeed := deterministicVulkanGDNVector((kernel-1)*convDim, 1202601)
	recurrentSeed := deterministicVulkanGDNVector(nV*kHd*vHd, 1202602)
	convState := upload([]int{kernel - 1, convDim}, convSeed, compute.MemoryKVCache, "qwen35-gdn-conv-state")
	recurrentState := upload([]int{nV, kHd, vHd}, recurrentSeed, compute.MemoryKVCache, "qwen35-gdn-recurrent-state")
	convIdentity, recurrentIdentity := convState.Buf(), recurrentState.Buf()

	cpu := m.NewSession()
	cpu.Cache.linear = newLinearAttnCache(cfg)
	cpuLayer := cpu.Cache.linear.layer(cfg, 0)
	cpuLayer.conv = make([][]float32, kernel-1)
	for row := range cpuLayer.conv {
		cpuLayer.conv[row] = append([]float32(nil), convSeed[row*convDim:(row+1)*convDim]...)
	}
	for head := range cpuLayer.recurrent {
		copy(cpuLayer.recurrent[head], recurrentSeed[head*kHd*vHd:(head+1)*kHd*vHd])
	}

	head := m.tensor(m.headName())
	for step := 0; step < 3; step++ {
		input := deterministicVulkanGDNVector(hidden, int64(1202610+step))
		wantOutput := cpu.linearAttnStep(0, input, f32Kernel{m})
		wantLogits := parMatRows(head, wantOutput, cfg.VocabSize, hidden)

		x := uploadHostF32Class(be, []int{hidden}, input, compute.MemoryActivation, "qwen35-gdn-step-input")
		gotOutputDev, nextConv, nextRecurrent, err := be.Qwen35GDNDecode(
			x, inQKV, inZ, inB, inA, convW, aLog, dtBias, norm, outW,
			convState, recurrentState, nK, nV, kHd, vHd, kernel, float32(cfg.RMSNormEps),
		)
		if err != nil {
			be.Free(x)
			t.Fatalf("step %d Vulkan GDN decode: %v", step, err)
		}
		gotOutput := be.Read(gotOutputDev)
		gotConv := be.Read(nextConv)
		gotRecurrent := be.Read(nextRecurrent)
		be.Free(gotOutputDev)
		be.Free(x)
		if nextConv.Buf() != convIdentity || nextRecurrent.Buf() != recurrentIdentity {
			t.Fatalf("step %d persistent state identity changed", step)
		}

		compareVulkanGDNProductionVector(t, "output", wantOutput, gotOutput, 2e-3)
		compareVulkanGDNProductionVector(t, "convolution state", flattenVulkanGDNState(cpuLayer.conv), gotConv, 2e-4)
		compareVulkanGDNProductionVector(t, "recurrent state", flattenVulkanGDNState(cpuLayer.recurrent), gotRecurrent, 2e-3)
		gotLogits := parMatRows(head, gotOutput, cfg.VocabSize, hidden)
		compareVulkanGDNProductionVector(t, "logits", wantLogits, gotLogits, 4e-3)
		if wantToken, gotToken := argmax(wantLogits), argmax(gotLogits); gotToken != wantToken {
			t.Fatalf("step %d greedy token = %d, want %d", step, gotToken, wantToken)
		}

		convState, recurrentState = nextConv, nextRecurrent
	}
}
