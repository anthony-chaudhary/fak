//go:build vulkan && (windows || linux) && cgo

package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type vulkanQKNormDebug interface {
	VulkanDebugTransferBytes() (h2d, d2h uint64)
	VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	VulkanDebugResetDispatchProfile()
}

func TestQwen35VulkanResidentQKNormMatchesCPUWithZeroTransfers(t *testing.T) {
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
	dbg, ok := be.(vulkanQKNormDebug)
	if !ok {
		t.Fatalf("physical Vulkan backend %T lacks transfer/dispatch observability", be)
	}

	// These are the production Qwen3.5-27B/Qwen3.8 attention dimensions. The
	// fixture deliberately contains only this operation's weights and inputs: it
	// proves the production method and shader dispatch without pretending to be a
	// full-checkpoint quality or throughput run.
	const (
		nH    = 24
		nKV   = 4
		hd    = 256
		layer = 0
	)
	cfg := Config{NumHeads: nH, NumKVHeads: nKV, HeadDim: hd, QKNorm: true, QKNormEps: 1e-6, NormGain1p: true}
	m := &Model{Cfg: cfg, manifest: make(map[string]tensorMeta)}
	qWeight, kWeight := make([]float32, hd), make([]float32, hd)
	for i := 0; i < hd; i++ {
		qWeight[i] = -0.15 + 0.3*float32(i)/float32(hd-1)
		kWeight[i] = 0.12 - 0.24*float32(i)/float32(hd-1)
	}
	tpInjectTensors(m, map[string]tpTensor{
		layerName(layer, "self_attn.q_norm.weight"): {shape: []int{hd}, vals: qWeight},
		layerName(layer, "self_attn.k_norm.weight"): {shape: []int{hd}, vals: kWeight},
	})
	s := &Session{M: m, Backend: be, halW: make(map[string]compute.Tensor), borrowedHALW: make(map[string]struct{})}
	defer func() {
		s.Close()
		if err := m.CloseWeights(); err != nil {
			t.Errorf("CloseWeights: %v", err)
		}
	}()

	qHost, kHost := make([]float32, nH*hd), make([]float32, nKV*hd)
	for i := range qHost {
		qHost[i] = float32(math.Sin(float64(i)*0.017))*2.25 + float32(i%13)/17
	}
	for i := range kHost {
		kHost[i] = float32(math.Cos(float64(i)*0.023))*1.75 - float32(i%7)/19
	}
	q := uploadHostF32Class(be, []int{nH * hd}, qHost, compute.MemoryActivation, "qwen35-qknorm-witness-q")
	k := uploadHostF32Class(be, []int{nKV * hd}, kHost, compute.MemoryActivation, "qwen35-qknorm-witness-k")
	// Stage immutable gains before opening the measured operation window.
	_ = s.normWeightHAL(layerName(layer, "self_attn.q_norm.weight"))
	_ = s.normWeightHAL(layerName(layer, "self_attn.k_norm.weight"))

	dbg.VulkanDebugResetDispatchProfile()
	h2dBefore, d2hBefore := dbg.VulkanDebugTransferBytes()
	profileBefore := dbg.VulkanDebugDispatchProfileSnapshot()
	qNorm, kNorm, err := s.qwen35ResidentQKNorm(layer, q, k)
	if err != nil {
		t.Fatal(err)
	}
	h2dAfter, d2hAfter := dbg.VulkanDebugTransferBytes()
	profileAfter := dbg.VulkanDebugDispatchProfileSnapshot()
	h2dDelta, d2hDelta := h2dAfter-h2dBefore, d2hAfter-d2hBefore
	normDispatches := profileAfter.OtherNormDispatches - profileBefore.OtherNormDispatches
	h2dSubmits := profileAfter.OneShotH2DSubmits - profileBefore.OneShotH2DSubmits
	d2hSubmits := profileAfter.OneShotD2HSubmits - profileBefore.OneShotD2HSubmits
	if h2dDelta != 0 || d2hDelta != 0 || h2dSubmits != 0 || d2hSubmits != 0 {
		t.Fatalf("QKNorm transfer delta h2d_bytes=%d d2h_bytes=%d h2d_submits=%d d2h_submits=%d, want all zero", h2dDelta, d2hDelta, h2dSubmits, d2hSubmits)
	}
	if normDispatches != 2 {
		t.Fatalf("QKNorm norm dispatches=%d, want exactly 2", normDispatches)
	}

	qWant, kWant := append([]float32(nil), qHost...), append([]float32(nil), kHost...)
	m.applyLayerQKNorm(layer, qWant, kWant)
	qGot, kGot := be.Read(qNorm), be.Read(kNorm)
	qCos, qMax := qknormVectorAgreement(qGot, qWant)
	kCos, kMax := qknormVectorAgreement(kGot, kWant)
	if qCos < 0.999999 || kCos < 0.999999 || qMax > 2e-5 || kMax > 2e-5 {
		t.Fatalf("QKNorm parity q_cos=%.9f q_max_abs=%g k_cos=%.9f k_max_abs=%g", qCos, qMax, kCos, kMax)
	}
	t.Logf("engine=fak-native backend=%s device=%s model_fixture=qwen3.5-27b-qwen3.8-qknorm-geometry source=working-tree n_heads=%d n_kv_heads=%d head_dim=%d qk_eps=%g norm_gain_1p=true norm_dispatches=%d h2d_bytes=%d d2h_bytes=%d h2d_submits=%d d2h_submits=%d q_cos=%.9f k_cos=%.9f q_max_abs=%g k_max_abs=%g implicit_fallback=false",
		be.Name(), be.Tier(), nH, nKV, hd, cfg.QKNormEps, normDispatches, h2dDelta, d2hDelta, h2dSubmits, d2hSubmits, qCos, kCos, qMax, kMax)
}

func qknormVectorAgreement(got, want []float32) (cosine, maxAbs float64) {
	if len(got) != len(want) || len(got) == 0 {
		return 0, math.Inf(1)
	}
	var dot, gn, wn float64
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		dot += g * w
		gn += g * g
		wn += w * w
		maxAbs = math.Max(maxAbs, math.Abs(g-w))
	}
	if gn == 0 || wn == 0 {
		return 0, maxAbs
	}
	return dot / math.Sqrt(gn*wn), maxAbs
}
