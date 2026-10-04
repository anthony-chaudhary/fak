//go:build vulkan && (windows || linux) && cgo

package compute

// Filename sorts after vulkan_plan_test.go on purpose: its weight uploads would otherwise hide the allocation delta TestVulkanWeightArenaSuballocation measures.

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Decode-kernel parity witnesses for the two optional Vulkan decode kernels:
//   - rmsnorm_q8_matmul2_coop (cooperative Q8 gate/up): must be BIT-IDENTICAL to
//     rmsnorm_q8_matmul2 (one thread per output), and close to the CPU reference.
//   - q2k_matvec (single-token Q2_K): must match q2k_matmul and the CPU reference
//     within a reduction-order tolerance, including the fused Q2_K decode paths.
// Each case runs the old kernel (oracle) and the new kernel (target) in one process via
// the test-only selection hooks, restoring the previous selection on cleanup.

// decodeParityDump writes <name>.oracle.json and <name>.target.json when
// FAK_DECODE_PARITY_DUMP_DIR is set; otherwise it is a no-op.
func decodeParityDump(t *testing.T, name string, oracle, target []float32) {
	t.Helper()
	dir := os.Getenv("FAK_DECODE_PARITY_DUMP_DIR")
	if dir == "" {
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Errorf("parity dump: mkdir %s: %v", dir, err)
		return
	}
	safe := strings.NewReplacer("/", "_", "\\", "_", " ", "_", ":", "_").Replace(name)
	for suffix, data := range map[string][]float32{"oracle": oracle, "target": target} {
		vals := make([]float64, len(data))
		for i, x := range data {
			vals[i] = float64(x)
		}
		raw, err := json.Marshal(vals)
		if err != nil {
			t.Errorf("parity dump: marshal %s.%s: %v", safe, suffix, err)
			continue
		}
		path := filepath.Join(dir, safe+"."+suffix+".json")
		if err := os.WriteFile(path, raw, 0o644); err != nil {
			t.Errorf("parity dump: write %s: %v", path, err)
		}
	}
}

// requireDecodeKernel enables the new kernel via sel, skips when its pipeline was not
// built, restores the previous selection on cleanup, and leaves the new kernel selected.
func requireDecodeKernel(t *testing.T, label string, sel func(bool) bool) {
	t.Helper()
	prev := sel(true)
	t.Cleanup(func() { sel(prev) })
	if !sel(true) {
		t.Skipf("%s pipeline unavailable on this device", label)
	}
}

func copyRead(v *vulkanBackend, x Tensor) []float32 {
	return append([]float32(nil), v.Read(x)...)
}

func assertFinite(t *testing.T, label string, xs []float32) {
	t.Helper()
	for i, x := range xs {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			t.Fatalf("%s: nonfinite output at %d: %g", label, i, x)
		}
	}
}

func assertBitIdentical(t *testing.T, label string, oracle, target []float32) {
	t.Helper()
	if len(oracle) != len(target) {
		t.Fatalf("%s: length oracle=%d target=%d", label, len(oracle), len(target))
	}
	for i := range oracle {
		if math.Float32bits(oracle[i]) != math.Float32bits(target[i]) {
			t.Fatalf("%s: element %d not bit-identical: oracle=%g (0x%08x) target=%g (0x%08x)",
				label, i, oracle[i], math.Float32bits(oracle[i]), target[i], math.Float32bits(target[i]))
		}
	}
}

// assertDecodeClose enforces cosine >= 0.99999 and max|delta| <= 1e-3*(max|ref|+1).
func assertDecodeClose(t *testing.T, label string, ref, got []float32) {
	t.Helper()
	if len(ref) != len(got) {
		t.Fatalf("%s: length ref=%d got=%d", label, len(ref), len(got))
	}
	var maxRef float64
	for _, x := range ref {
		maxRef = math.Max(maxRef, math.Abs(float64(x)))
	}
	c := cosine(ref, got)
	d := maxAbsDelta(ref, got)
	if c < 0.99999 || d > 1e-3*(maxRef+1) {
		t.Fatalf("%s: cosine=%.9f (want >= 0.99999) max|delta|=%g (bound %g)", label, c, d, 1e-3*(maxRef+1))
	}
}

// fak-test:runtime fast est=2s
func TestVulkanQ8GateUpCoopBitParity(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		t.Skip("Vulkan Q8 int8 path unavailable")
	}
	requireDecodeKernel(t, "rmsnorm_q8_matmul2_coop", vulkanDebugSelectQ8GateUpCoop)
	const eps = 1e-6
	for _, tc := range []struct {
		name           string
		out0, out1, in int
	}{
		{"in2048_out37_45", 37, 45, 2048},
		{"in2048_out1029_1031", 1029, 1031, 2048},
		{"in5120_out37_45", 37, 45, 5120},
		{"in17408_out13_21", 13, 21, 17408},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := cpu()
			s := lcg(0x51ed + uint64(tc.in)*31 + uint64(tc.out0))
			x := randVec(&s, tc.in)
			norm := randVec(&s, tc.in)
			for i := range norm {
				norm[i] = 1 + 0.2*norm[i]
			}
			q0 := QuantizeQ8(c, []int{tc.out0, tc.in}, randVec(&s, tc.out0*tc.in), 32)
			q1 := QuantizeQ8(c, []int{tc.out1, tc.in}, randVec(&s, tc.out1*tc.in), 32)

			hx, hn := NewF32(c, []int{tc.in}, x), NewF32(c, []int{tc.in}, norm)
			xn := c.RMSNorm(hx, hn, eps)
			ref0, ref1 := c.Read(c.MatMul(q0, xn)), c.Read(c.MatMul(q1, xn))

			dw0, dw1 := v.Upload(q0, Q8_0), v.Upload(q1, Q8_0)
			dx, dn := v.Upload(hx, F32), v.Upload(hn, F32)
			defer func() {
				v.Free(dw0)
				v.Free(dw1)
				v.Free(dx)
				v.Free(dn)
				v.Recycle()
			}()

			vulkanDebugSelectQ8GateUpCoop(false)
			o0, o1 := v.RMSNormMatMul2(dw0, dw1, dx, dn, eps)
			oracle0, oracle1 := copyRead(v, o0), copyRead(v, o1)
			vulkanDebugSelectQ8GateUpCoop(true)
			g0, g1 := v.RMSNormMatMul2(dw0, dw1, dx, dn, eps)
			target0, target1 := copyRead(v, g0), copyRead(v, g1)

			decodeParityDump(t, t.Name()+"_y0", oracle0, target0)
			decodeParityDump(t, t.Name()+"_y1", oracle1, target1)

			assertFinite(t, "oracle y0", oracle0)
			assertFinite(t, "oracle y1", oracle1)
			assertBitIdentical(t, "y0 coop vs one-thread", oracle0, target0)
			assertBitIdentical(t, "y1 coop vs one-thread", oracle1, target1)
			for _, p := range []struct {
				label    string
				ref, got []float32
			}{{"y0 vs CPU", ref0, target0}, {"y1 vs CPU", ref1, target1}} {
				if cs := cosine(p.ref, p.got); cs < 0.9999 {
					t.Fatalf("%s: cosine %.8f < 0.9999 (max|delta|=%g)", p.label, cs, maxAbsDelta(p.ref, p.got))
				}
			}
		})
	}
}

// q2kParityWeight builds a Q2_K [out,in] host weight with small d/dmin.
func q2kParityWeight(seed int64, out, in int) Tensor {
	rng := rand.New(rand.NewSource(seed))
	raw := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	for b := 0; b < out*(in/q2kSuper); b++ {
		blk := raw[b*q2kSuperBlock : (b+1)*q2kSuperBlock]
		for i := 0; i < 80; i++ { // scales[16] + qs[64]
			blk[i] = byte(rng.Intn(256))
		}
		binaryPutFloat16(blk[80:82], float32(0.01+rng.Float64()*0.05))
		binaryPutFloat16(blk[82:84], float32(0.005+rng.Float64()*0.02))
	}
	return NewQ2K(Default(), []int{out, in}, raw)
}

// fak-test:runtime fast est=2s
func TestVulkanQ2KMatvecParity(t *testing.T) {
	v := vk(t)
	requireDecodeKernel(t, "q2k_matvec", vulkanDebugSelectQ2KMatvec)
	const out, in = 37, 5120
	const eps = 1e-6
	ref := Default()

	t.Run("matmul", func(t *testing.T) {
		s := lcg(0x2a2a)
		hw := q2kParityWeight(7001, out, in)
		hx := NewF32(ref, []int{in}, randVec(&s, in))
		want := ref.Read(ref.MatMul(hw, hx))
		dw, dx := v.Upload(hw, Q2_K), v.Upload(hx, F32)
		defer func() {
			v.Free(dw)
			v.Free(dx)
			v.Recycle()
		}()
		vulkanDebugSelectQ2KMatvec(false)
		oracle := copyRead(v, v.MatMul(dw, dx))
		vulkanDebugSelectQ2KMatvec(true)
		target := copyRead(v, v.MatMul(dw, dx))
		decodeParityDump(t, t.Name(), oracle, target)
		assertFinite(t, "target", target)
		assertDecodeClose(t, "matvec vs q2k_matmul", oracle, target)
		assertDecodeClose(t, "matvec vs CPU", want, target)
		assertDecodeClose(t, "q2k_matmul vs CPU", want, oracle)
	})

	t.Run("rmsnorm_matmul2_fused", func(t *testing.T) {
		t.Setenv("FAK_VULKAN_Q2K_FUSION", "candidate")
		const out1 = 45
		s := lcg(0x3b3b)
		hw0, hw1 := q2kParityWeight(7002, out, in), q2kParityWeight(7003, out1, in)
		hx := NewF32(ref, []int{in}, randVec(&s, in))
		normData := randVec(&s, in)
		for i := range normData {
			normData[i] = 1 + 0.2*normData[i]
		}
		hn := NewF32(ref, []int{in}, normData)
		xn := ref.RMSNorm(hx, hn, eps)
		want0, want1 := ref.Read(ref.MatMul(hw0, xn)), ref.Read(ref.MatMul(hw1, xn))
		dw0, dw1 := v.Upload(hw0, Q2_K), v.Upload(hw1, Q2_K)
		dx, dn := v.Upload(hx, F32), v.Upload(hn, F32)
		defer func() {
			v.Free(dw0)
			v.Free(dw1)
			v.Free(dx)
			v.Free(dn)
			v.Recycle()
		}()
		vulkanDebugSelectQ2KMatvec(false)
		a0, a1 := v.RMSNormMatMul2(dw0, dw1, dx, dn, eps)
		oracle0, oracle1 := copyRead(v, a0), copyRead(v, a1)
		vulkanDebugSelectQ2KMatvec(true)
		b0, b1 := v.RMSNormMatMul2(dw0, dw1, dx, dn, eps)
		target0, target1 := copyRead(v, b0), copyRead(v, b1)
		decodeParityDump(t, t.Name()+"_y0", oracle0, target0)
		decodeParityDump(t, t.Name()+"_y1", oracle1, target1)
		assertFinite(t, "target y0", target0)
		assertFinite(t, "target y1", target1)
		assertDecodeClose(t, "y0 matvec vs original", oracle0, target0)
		assertDecodeClose(t, "y1 matvec vs original", oracle1, target1)
		assertDecodeClose(t, "y0 vs CPU", want0, target0)
		assertDecodeClose(t, "y1 vs CPU", want1, target1)
	})

	t.Run("swiglu_matmul_add", func(t *testing.T) {
		s := lcg(0x4c4c)
		hw := q2kParityWeight(7004, out, in)
		hg := NewF32(ref, []int{in}, randVec(&s, in))
		hu := NewF32(ref, []int{in}, randVec(&s, in))
		residual := randVec(&s, out)
		hr := NewF32(ref, []int{out}, residual)
		want := ref.Read(ref.MatMul(hw, ref.SwiGLU(hg, hu)))
		for i := range want {
			want[i] += residual[i]
		}
		dw, dg, du := v.Upload(hw, Q2_K), v.Upload(hg, F32), v.Upload(hu, F32)
		dOracle, dTarget := v.Upload(hr, F32), v.Upload(hr, F32)
		defer func() {
			v.Free(dw)
			v.Free(dg)
			v.Free(du)
			v.Free(dOracle)
			v.Free(dTarget)
			v.Recycle()
		}()
		vulkanDebugSelectQ2KMatvec(false)
		v.SwiGLUMatMulAddInPlace(dOracle, dw, dg, du)
		oracle := copyRead(v, dOracle)
		vulkanDebugSelectQ2KMatvec(true)
		v.SwiGLUMatMulAddInPlace(dTarget, dw, dg, du)
		target := copyRead(v, dTarget)
		decodeParityDump(t, t.Name(), oracle, target)
		assertFinite(t, "target", target)
		assertDecodeClose(t, "matvec vs original", oracle, target)
		assertDecodeClose(t, "matvec vs CPU", want, target)
	})
}
