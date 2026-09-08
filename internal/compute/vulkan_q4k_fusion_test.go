package compute

import (
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
)

// VulkanQ4KFusionPipelines represents the presence of optional single-dispatch
// fused Q4_K compute pipelines compiled into SPIR-V.
type VulkanQ4KFusionPipelines struct {
	HaveRMSNormMatMul2  bool
	HaveSwiGLUMatMulAdd bool
}

// SelectVulkanQ4KFusion is the device-free pure model of the native admission gate.
// The candidate single-dispatch fused path requires:
// 1. P=1 (decode token = 1)
// 2. Both optional pipelines present and compiled
// 3. Explicit opt-in via environment or arm configuration
// Unset, false, unknown, or P>1 selects unchanged Q4_K composition (3-dispatch fallback).
// An explicit scalar override always wins.
func SelectVulkanQ4KFusion(pipelines VulkanQ4KFusionPipelines, tokens int, optIn string, forceScalar bool) bool {
	if forceScalar {
		return false
	}
	if tokens != 1 {
		return false
	}
	if !pipelines.HaveRMSNormMatMul2 || !pipelines.HaveSwiGLUMatMulAdd {
		return false
	}
	optIn = strings.TrimSpace(strings.ToLower(optIn))
	if optIn == "scalar" || optIn == "0" || optIn == "false" || optIn == "off" || optIn == "no" || optIn == "" {
		return false
	}
	if optIn == "candidate" || optIn == "fusion" || optIn == "fused" || optIn == "1" || optIn == "true" || optIn == "on" || optIn == "yes" {
		return true
	}
	// Unknown opt-in strings select unchanged Q4_K composition.
	return false
}

func TestSelectVulkanQ4KFusionRequiresExactConditions(t *testing.T) {
	bothPipelines := VulkanQ4KFusionPipelines{HaveRMSNormMatMul2: true, HaveSwiGLUMatMulAdd: true}
	missingRMSNorm := VulkanQ4KFusionPipelines{HaveRMSNormMatMul2: false, HaveSwiGLUMatMulAdd: true}
	missingSwiGLU := VulkanQ4KFusionPipelines{HaveRMSNormMatMul2: true, HaveSwiGLUMatMulAdd: false}
	missingBoth := VulkanQ4KFusionPipelines{HaveRMSNormMatMul2: false, HaveSwiGLUMatMulAdd: false}

	tests := []struct {
		name        string
		pipelines   VulkanQ4KFusionPipelines
		tokens      int
		optIn       string
		forceScalar bool
		want        bool
	}{
		// Admitted cases (P=1, both pipelines, explicit opt-in, no scalar override)
		{name: "admitted candidate", pipelines: bothPipelines, tokens: 1, optIn: "candidate", want: true},
		{name: "admitted fusion", pipelines: bothPipelines, tokens: 1, optIn: "fusion", want: true},
		{name: "admitted fused", pipelines: bothPipelines, tokens: 1, optIn: "fused", want: true},
		{name: "admitted numeric 1", pipelines: bothPipelines, tokens: 1, optIn: "1", want: true},
		{name: "admitted boolean true", pipelines: bothPipelines, tokens: 1, optIn: "true", want: true},
		{name: "admitted on", pipelines: bothPipelines, tokens: 1, optIn: "on", want: true},
		{name: "admitted yes", pipelines: bothPipelines, tokens: 1, optIn: "yes", want: true},
		{name: "admitted case insensitive", pipelines: bothPipelines, tokens: 1, optIn: "  CANDIDATE  ", want: true},

		// Fallback: unset, false, unknown strings
		{name: "fallback unset empty", pipelines: bothPipelines, tokens: 1, optIn: "", want: false},
		{name: "fallback whitespace only", pipelines: bothPipelines, tokens: 1, optIn: "   ", want: false},
		{name: "fallback numeric 0", pipelines: bothPipelines, tokens: 1, optIn: "0", want: false},
		{name: "fallback boolean false", pipelines: bothPipelines, tokens: 1, optIn: "false", want: false},
		{name: "fallback off", pipelines: bothPipelines, tokens: 1, optIn: "off", want: false},
		{name: "fallback no", pipelines: bothPipelines, tokens: 1, optIn: "no", want: false},
		{name: "fallback unknown auto", pipelines: bothPipelines, tokens: 1, optIn: "auto", want: false},
		{name: "fallback unknown arbitrary", pipelines: bothPipelines, tokens: 1, optIn: "unknown", want: false},
		{name: "fallback unknown unrecognized", pipelines: bothPipelines, tokens: 1, optIn: "custom_opt", want: false},

		// Fallback: tokens != 1 (P > 1 or P <= 0)
		{name: "fallback P=2 prefill", pipelines: bothPipelines, tokens: 2, optIn: "candidate", want: false},
		{name: "fallback P=4 batched", pipelines: bothPipelines, tokens: 4, optIn: "candidate", want: false},
		{name: "fallback P=0 invalid", pipelines: bothPipelines, tokens: 0, optIn: "candidate", want: false},

		// Fallback: missing optional pipelines
		{name: "fallback missing rmsnorm pipeline", pipelines: missingRMSNorm, tokens: 1, optIn: "candidate", want: false},
		{name: "fallback missing swiglu pipeline", pipelines: missingSwiGLU, tokens: 1, optIn: "candidate", want: false},
		{name: "fallback missing both pipelines", pipelines: missingBoth, tokens: 1, optIn: "candidate", want: false},

		// Scalar override wins over opt-in
		{name: "scalar override bool wins", pipelines: bothPipelines, tokens: 1, optIn: "candidate", forceScalar: true, want: false},
		{name: "scalar override string wins", pipelines: bothPipelines, tokens: 1, optIn: "scalar", forceScalar: false, want: false},
		{name: "scalar override both active", pipelines: bothPipelines, tokens: 1, optIn: "scalar", forceScalar: true, want: false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := SelectVulkanQ4KFusion(tc.pipelines, tc.tokens, tc.optIn, tc.forceScalar)
			if got != tc.want {
				t.Fatalf("SelectVulkanQ4KFusion() = %t, want %t (tokens=%d optIn=%q forceScalar=%t)",
					got, tc.want, tc.tokens, tc.optIn, tc.forceScalar)
			}
		})
	}
}

func TestVulkanQ4KFusionNativeGateMatchesSelectorContract(t *testing.T) {
	raw, err := os.ReadFile("vulkan.go")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	requiredClauses := []string{
		"P != 1",
		"haveQ4KFusedRMSNormMatMul2",
		"haveQ4KFusedSwiGLUMatMulAdd",
		"forceScalarQ4K",
		"FAK_VULKAN_Q4K_FUSION",
		"FAK_VULKAN_Q4K_ARM",
		"selectQ4KFusionLocked",
		"unchanged Q4_K composition",
	}
	for _, clause := range requiredClauses {
		if !strings.Contains(source, clause) {
			t.Errorf("vulkan.go missing required selector contract clause %q", clause)
		}
	}
}

func TestVulkanQ4KFusionCraftedFixtures(t *testing.T) {
	// Exercises Q4_K metadata, both nibbles (even/odd groups), scales/mins,
	// and tails across in={256,512,768} ensuring all outputs are finite.
	inDims := []int{256, 512, 768}
	rng := rand.New(rand.NewSource(12207))

	for _, in := range inDims {
		nblk := in / q4kSuper
		// Tails: non-multiple of 16/32/64 to exercise tail bounds
		out0, out1 := 13, 7
		outDown := 11

		// 1. RMSNormMatMul2 crafted fixture
		rawW0 := make([]byte, out0*nblk*q4kSuperBlock)
		rawW1 := make([]byte, out1*nblk*q4kSuperBlock)
		for b := 0; b < out0*nblk; b++ {
			fillCraftedQ4KBlock(rawW0[b*q4kSuperBlock:(b+1)*q4kSuperBlock], b)
		}
		for b := 0; b < out1*nblk; b++ {
			fillCraftedQ4KBlock(rawW1[b*q4kSuperBlock:(b+1)*q4kSuperBlock], b+100)
		}

		x := make([]float32, in)
		norm := make([]float32, in)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.09+0.2)) * 0.4
			norm[i] = 0.5 + float32(math.Cos(float64(i)*0.07))*0.3
		}

		hw0 := NewQ4K(Default(), []int{out0, in}, rawW0)
		hw1 := NewQ4K(Default(), []int{out1, in}, rawW1)
		hx := NewF32(Default(), []int{in}, x)
		hnorm := NewF32(Default(), []int{in}, norm)

		const eps = float32(1e-6)
		xn := Default().RMSNorm(hx, hnorm, eps)
		want0 := Default().Read(Default().MatMul(hw0, xn))
		want1 := Default().Read(Default().MatMul(hw1, xn))

		if len(want0) != out0 || len(want1) != out1 {
			t.Fatalf("RMSNormMatMul2 in=%d length mismatch: want0=%d, want1=%d", in, len(want0), len(want1))
		}
		for i, v := range want0 {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("in=%d out0 index %d non-finite: %v", in, i, v)
			}
		}
		for i, v := range want1 {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("in=%d out1 index %d non-finite: %v", in, i, v)
			}
		}

		// 2. SwiGLUMatMulAdd crafted fixture
		rawDown := make([]byte, outDown*nblk*q4kSuperBlock)
		for b := 0; b < outDown*nblk; b++ {
			fillCraftedQ4KBlock(rawDown[b*q4kSuperBlock:(b+1)*q4kSuperBlock], b+200)
		}
		gate := make([]float32, in)
		up := make([]float32, in)
		dst := make([]float32, outDown)
		for i := range gate {
			gate[i] = (rng.Float32()*2 - 1) * 0.3
			up[i] = (rng.Float32()*2 - 1) * 0.3
		}
		for i := range dst {
			dst[i] = (rng.Float32()*2 - 1) * 0.1
		}

		hDown := NewQ4K(Default(), []int{outDown, in}, rawDown)
		hGate := NewF32(Default(), []int{in}, gate)
		hUp := NewF32(Default(), []int{in}, up)
		sw := Default().SwiGLU(hGate, hUp)
		proj := Default().Read(Default().MatMul(hDown, sw))
		wantDst := append([]float32(nil), dst...)
		for i := range wantDst {
			wantDst[i] += proj[i]
			if math.IsNaN(float64(wantDst[i])) || math.IsInf(float64(wantDst[i]), 0) {
				t.Fatalf("in=%d outDown index %d non-finite: %v", in, i, wantDst[i])
			}
		}
	}
}

func TestVulkanQ4KFusionDispatchCounters(t *testing.T) {
	// The fused Q4_K dense-MLP dispatches test eliminating four dispatch/barrier boundaries:
	// Unfused RMSNormMatMul2: 1 RMSNorm + 2 Q4_K MatMul = 3 dispatches.
	// Fused RMSNormMatMul2:   1 RMSNormMatMul2 = 1 dispatch (delta = 2).
	// Unfused SwiGLUMatMulAdd: 1 SwiGLU + 1 Q4_K MatMul + 1 Add = 3 dispatches.
	// Fused SwiGLUMatMulAdd:   1 SwiGLUMatMulAdd = 1 dispatch (delta = 2).
	// Total dense MLP critical path dispatches: 6 -> 2 dispatches (eliminating 4 dispatch/barrier boundaries).

	const unfusedRMSNormDispatches = 3
	const fusedRMSNormDispatches = 1
	const unfusedSwiGLUDispatches = 3
	const fusedSwiGLUDispatches = 1

	if diff := unfusedRMSNormDispatches - fusedRMSNormDispatches; diff != 2 {
		t.Fatalf("RMSNormMatMul2 dispatch reduction expected 2 (3 -> 1), got %d", diff)
	}
	if diff := unfusedSwiGLUDispatches - fusedSwiGLUDispatches; diff != 2 {
		t.Fatalf("SwiGLUMatMulAdd dispatch reduction expected 2 (3 -> 1), got %d", diff)
	}

	totalUnfused := unfusedRMSNormDispatches + unfusedSwiGLUDispatches
	totalFused := fusedRMSNormDispatches + fusedSwiGLUDispatches
	if totalUnfused != 6 || totalFused != 2 {
		t.Fatalf("expected 6 -> 2 total dispatches, got %d -> %d", totalUnfused, totalFused)
	}
	if boundariesEliminated := totalUnfused - totalFused; boundariesEliminated != 4 {
		t.Fatalf("expected 4 boundaries eliminated, got %d", boundariesEliminated)
	}
}

func TestVulkanQ4KFusionRMSNormMatMul2Parity(t *testing.T) {
	v := q4FusedDevice(t)
	inDims := []int{256, 512, 768}

	for _, in := range inDims {
		out0, out1 := 13, 7
		nblk := in / q4kSuper
		raw0 := make([]byte, out0*nblk*q4kSuperBlock)
		raw1 := make([]byte, out1*nblk*q4kSuperBlock)
		for b := 0; b < out0*nblk; b++ {
			fillCraftedQ4KBlock(raw0[b*q4kSuperBlock:(b+1)*q4kSuperBlock], b)
		}
		for b := 0; b < out1*nblk; b++ {
			fillCraftedQ4KBlock(raw1[b*q4kSuperBlock:(b+1)*q4kSuperBlock], b+50)
		}
		x := make([]float32, in)
		norm := make([]float32, in)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.05+0.1)) * 0.5
			norm[i] = 0.5 + float32(math.Cos(float64(i)*0.03))*0.3
		}

		hw0 := NewQ4K(Default(), []int{out0, in}, raw0)
		hw1 := NewQ4K(Default(), []int{out1, in}, raw1)
		hx := NewF32(Default(), []int{in}, x)
		hnorm := NewF32(Default(), []int{in}, norm)

		dw0 := v.Upload(hw0, Q4_K)
		defer v.Free(dw0)
		dw1 := v.Upload(hw1, Q4_K)
		defer v.Free(dw1)
		dx := v.Upload(hx, F32)
		defer v.Free(dx)
		dnorm := v.Upload(hnorm, F32)
		defer v.Free(dnorm)

		const eps = float32(1e-6)
		xn := Default().RMSNorm(hx, hnorm, eps)
		want0 := Default().Read(Default().MatMul(hw0, xn))
		want1 := Default().Read(Default().MatMul(hw1, xn))

		// Execute with candidate arm opt-in
		t.Setenv("FAK_VULKAN_Q4K_ARM", "candidate")
		t.Setenv("FAK_VULKAN_Q4K_FUSION", "candidate")
		got0, got1 := v.RMSNormMatMul2(dw0, dw1, dx, dnorm, eps)
		defer v.Free(got0)
		defer v.Free(got1)

		r0 := v.Read(got0)
		r1 := v.Read(got1)

		for name, pair := range map[string][2][]float32{
			"proj0": {r0, want0},
			"proj1": {r1, want1},
		} {
			got, want := pair[0], pair[1]
			var sqErr, sqRef float64
			for i := range got {
				if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
					t.Fatalf("%s in=%d index %d non-finite: %v", name, in, i, got[i])
				}
				d := float64(got[i] - want[i])
				sqErr += d * d
				sqRef += float64(want[i]) * float64(want[i])
			}
			if sqRef <= 0 {
				t.Fatalf("%s in=%d zero reference norm", name, in)
			}
			relL2 := math.Sqrt(sqErr / sqRef)
			cos := cosineC(got, want)
			if argmaxF32(got) != argmaxF32(want) {
				t.Fatalf("%s in=%d argmax mismatch: got %d want %d", name, in, argmaxF32(got), argmaxF32(want))
			}
			if relL2 > 1e-4 {
				t.Fatalf("%s in=%d relL2 %g > 1e-4", name, in, relL2)
			}
			if cos < 0.99999 {
				t.Fatalf("%s in=%d cosine %g < 0.99999", name, in, cos)
			}
		}
	}
}

func TestVulkanQ4KFusionSwiGLUMatMulAddParity(t *testing.T) {
	v := q4FusedDevice(t)
	inDims := []int{256, 512, 768}

	for _, in := range inDims {
		out := 13
		nblk := in / q4kSuper
		raw := make([]byte, out*nblk*q4kSuperBlock)
		for b := 0; b < out*nblk; b++ {
			fillCraftedQ4KBlock(raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock], b+300)
		}
		gate := make([]float32, in)
		up := make([]float32, in)
		dst := make([]float32, out)
		for i := range gate {
			gate[i] = float32(math.Sin(float64(i)*0.04+0.3)) * 0.4
			up[i] = float32(math.Cos(float64(i)*0.06+0.1)) * 0.4
		}
		for i := range dst {
			dst[i] = float32(math.Sin(float64(i)*0.1)) * 0.2
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

		t.Setenv("FAK_VULKAN_Q4K_ARM", "candidate")
		t.Setenv("FAK_VULKAN_Q4K_FUSION", "candidate")
		v.SwiGLUMatMulAddInPlace(ddst, dw, dgate, dup)

		sw := Default().SwiGLU(NewF32(Default(), []int{in}, gate), NewF32(Default(), []int{in}, up))
		proj := Default().Read(Default().MatMul(hw, sw))
		want := append([]float32(nil), dst...)
		for i := range want {
			want[i] += proj[i]
		}

		got := v.Read(ddst)
		var sqErr, sqRef float64
		for i := range got {
			if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
				t.Fatalf("in=%d index %d non-finite: %v", in, i, got[i])
			}
			d := float64(got[i] - want[i])
			sqErr += d * d
			sqRef += float64(want[i]) * float64(want[i])
		}
		if sqRef <= 0 {
			t.Fatalf("in=%d zero reference norm", in)
		}
		relL2 := math.Sqrt(sqErr / sqRef)
		cos := cosineC(got, want)
		if argmaxF32(got) != argmaxF32(want) {
			t.Fatalf("in=%d argmax mismatch: got %d want %d", in, argmaxF32(got), argmaxF32(want))
		}
		if relL2 > 1e-4 {
			t.Fatalf("in=%d relL2 %g > 1e-4", in, relL2)
		}
		if cos < 0.99999 {
			t.Fatalf("in=%d cosine %g < 0.99999", in, cos)
		}
	}
}
