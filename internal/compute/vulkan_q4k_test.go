//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math/rand"
	"testing"
)

func TestVulkanQ4KMatMulMatchesCPUReference(t *testing.T) {
	v, ok := Pick("vulkan").(*vulkanBackend)
	if !ok {
		t.Skip("Vulkan backend unavailable")
	}
	const out, in = 12, 768
	raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	rng := rand.New(rand.NewSource(9715))
	for b := 0; b < out*(in/q4kSuper); b++ {
		randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	hw := NewQ4K(Default(), []int{out, in}, raw)
	dw := v.Upload(hw, Q4_K)
	defer v.Free(dw)
	dx := v.Upload(NewF32(Default(), []int{in}, x), F32)
	defer v.Free(dx)
	dy := v.MatMul(dw, dx)
	defer v.Free(dy)
	got := v.Read(dy)
	want := Default().Read(Default().MatMul(hw, NewF32(Default(), []int{in}, x)))
	if a, b := argmaxF32(got), argmaxF32(want); a != b {
		t.Fatalf("argmax=%d want %d", a, b)
	}
	if c := cosineC(got, want); c < 0.995 {
		t.Fatalf("cosine %.8f < 0.995", c)
	}
}
func TestVulkanQ4KRMSNormMatMul2MatchesCPUReference(t *testing.T) {
	v, ok := Pick("vulkan").(*vulkanBackend)
	if !ok {
		t.Skip("Vulkan backend unavailable")
	}
	const out0, out1, in = 12, 16, 768
	rng := rand.New(rand.NewSource(9716))
	newWeight := func(out int) Tensor {
		raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
		for b := 0; b < out*(in/q4kSuper); b++ {
			randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
		}
		return NewQ4K(Default(), []int{out, in}, raw)
	}
	hw0, hw1 := newWeight(out0), newWeight(out1)
	x := make([]float32, in)
	norm := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
		norm[i] = 0.5 + rng.Float32()
	}
	dw0 := v.Upload(hw0, Q4_K)
	defer v.Free(dw0)
	dw1 := v.Upload(hw1, Q4_K)
	defer v.Free(dw1)
	dx := v.Upload(NewF32(Default(), []int{in}, x), F32)
	defer v.Free(dx)
	dnorm := v.Upload(NewF32(Default(), []int{in}, norm), F32)
	defer v.Free(dnorm)
	const eps = float32(1e-6)
	got0, got1 := v.RMSNormMatMul2(dw0, dw1, dx, dnorm, eps)
	defer v.Free(got0)
	defer v.Free(got1)
	hx := NewF32(Default(), []int{in}, x)
	hnorm := NewF32(Default(), []int{in}, norm)
	xn := Default().RMSNorm(hx, hnorm, eps)
	want0 := Default().Read(Default().MatMul(hw0, xn))
	want1 := Default().Read(Default().MatMul(hw1, xn))
	for name, pair := range map[string][2][]float32{
		"projection 0": {v.Read(got0), want0},
		"projection 1": {v.Read(got1), want1},
	} {
		if c := cosineC(pair[0], pair[1]); c < 0.995 {
			t.Fatalf("%s cosine %.8f < 0.995", name, c)
		}
	}
}
func TestVulkanQ4KSwiGLUMatMulAddInPlaceMatchesCPUReference(t *testing.T) {
	v, ok := Pick("vulkan").(*vulkanBackend)
	if !ok {
		t.Skip("Vulkan backend unavailable")
	}
	const out, in = 16, 768
	rng := rand.New(rand.NewSource(9717))
	raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	for b := 0; b < out*(in/q4kSuper); b++ {
		randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}
	gate, up := make([]float32, in), make([]float32, in)
	dst := make([]float32, out)
	for i := range gate {
		gate[i], up[i] = rng.Float32()*2-1, rng.Float32()*2-1
	}
	for i := range dst {
		dst[i] = rng.Float32()*2 - 1
	}
	hw := NewQ4K(Default(), []int{out, in}, raw)
	dw := v.Upload(hw, Q4_K)
	defer v.Free(dw)
	dgate := v.Upload(NewF32(Default(), []int{in}, gate), F32)
	defer v.Free(dgate)
	dup := v.Upload(NewF32(Default(), []int{in}, up), F32)
	defer v.Free(dup)
	ddst := v.Upload(NewF32(Default(), []int{out}, dst), F32)
	defer v.Free(ddst)
	v.SwiGLUMatMulAddInPlace(ddst, dw, dgate, dup)
	sw := Default().SwiGLU(NewF32(Default(), []int{in}, gate), NewF32(Default(), []int{in}, up))
	proj := Default().Read(Default().MatMul(hw, sw))
	want := append([]float32(nil), dst...)
	for i := range want {
		want[i] += proj[i]
	}
	if c := cosineC(v.Read(ddst), want); c < 0.995 {
		t.Fatalf("cosine %.8f < 0.995", c)
	}
}
