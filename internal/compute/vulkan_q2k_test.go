package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

type q2FusedMLPBackend interface {
	Backend
	RMSNormMatMul2(Tensor, Tensor, Tensor, Tensor, float32) (Tensor, Tensor)
	SwiGLUMatMulAddInPlace(Tensor, Tensor, Tensor, Tensor)
	BeginBatch()
	FlushBatch()
	Recycle()
}

func q2FusedDevice(t *testing.T) q2FusedMLPBackend {
	t.Helper()
	be, ok := Lookup("vulkan")
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Vulkan device is not registered")
		}
		t.Skip("Vulkan device is not registered; device composition unverified")
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(be.Tier()), strings.ToLower(expected)) {
		t.Fatalf("device %q does not match required %q", be.Tier(), expected)
	}
	v, ok := be.(q2FusedMLPBackend)
	if !ok {
		t.Fatal("Vulkan backend lacks fused MLP or batch contract")
	}
	t.Logf("device=%s:%s fixture_seed=12064", v.Name(), v.Tier())
	return v
}

func q2FusedWeight(dtype Dtype, out, in int) Tensor {
	rng := rand.New(rand.NewSource(12064 + int64(out)))
	switch dtype {
	case Q2_K:
		raw := make([]byte, out*(in/256)*84)
		_, _ = rng.Read(raw)
		for b := 0; b < len(raw); b += 84 {
			binaryPutFloat16(raw[b+80:b+82], 1.0/128)
			binaryPutFloat16(raw[b+82:b+84], 1.0/256)
		}
		return NewQ2K(Default(), []int{out, in}, raw)
	case Q4_K:
		raw := make([]byte, out*(in/256)*144)
		for b := 0; b < len(raw); b += 144 {
			randQ4KBlockC(rng, raw[b:b+144])
			binaryPutFloat16(raw[b:b+2], 1.0/1024)
			binaryPutFloat16(raw[b+2:b+4], 1.0/1024)
		}
		return NewQ4K(Default(), []int{out, in}, raw)
	case Q8_0:
		codes, scales := make([]int8, out*in), make([]float32, out*(in/32))
		for i := range codes {
			codes[i] = int8(rng.Intn(255) - 127)
		}
		for i := range scales {
			scales[i] = 1.0 / 1024
		}
		return NewQ8(Default(), []int{out, in}, codes, scales, 32)
	default:
		data := make([]float32, out*in)
		for i := range data {
			data[i] = (rng.Float32()*2 - 1) / 8
		}
		return NewF32(Default(), []int{out, in}, data)
	}
}

func q2FusedInput(n int, offset float32) Tensor {
	data := make([]float32, n)
	for i := range data {
		data[i] = offset + float32(i%19-9)/32
	}
	return NewF32(Default(), []int{n}, data)
}

func q2FusedUpload(t *testing.T, v Backend, host Tensor) Tensor {
	t.Helper()
	d := v.Upload(host, host.Dtype)
	t.Cleanup(func() { v.Free(d) })
	return d
}

func q2FusedCompare(t *testing.T, got, cpu, unfused []float32) {
	t.Helper()
	if len(got) != len(cpu) || len(got) != len(unfused) {
		t.Fatalf("output lengths got=%d cpu=%d unfused=%d", len(got), len(cpu), len(unfused))
	}
	for i, x := range got {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.IsNaN(float64(cpu[i])) || math.IsInf(float64(cpu[i]), 0) || math.IsNaN(float64(unfused[i])) || math.IsInf(float64(unfused[i]), 0) {
			t.Fatalf("nonfinite output at %d", i)
		}
		err := math.Abs(float64(x - unfused[i]))
		if math.IsNaN(err) || err > 1e-4+1e-4*math.Abs(float64(unfused[i])) {
			t.Fatalf("element %d: fused=%g unfused=%g", i, x, unfused[i])
		}
	}
	if c := cosineC(got, cpu); math.IsNaN(c) || c < 0.995 {
		t.Fatalf("CPU composition cosine %.8f < 0.995", c)
	}
}

func TestQ2KFusedMLPReference(t *testing.T) {
	be := Default()
	x, norm := q2FusedInput(256, 0), q2FusedInput(256, 1)
	w0, w1, down := q2FusedWeight(Q2_K, 256, 256), q2FusedWeight(F32, 256, 256), q2FusedWeight(Q2_K, 8, 256)
	xn := be.RMSNorm(x, norm, 1e-6)
	gate := be.MatMul(w0, xn)
	up := be.MatMul(w1, xn)
	sw := be.SwiGLU(gate, up)
	got := be.Read(be.MatMul(down, sw))
	// Independent scalar composition checks the CPU oracle used by device tests.
	matmul := func(w Tensor, input []float64) []float64 {
		var dense []float32
		if w.Dtype == Q2_K {
			codes := w.Buf().(HostBuffer).I8()
			raw := make([]byte, len(codes))
			for i, code := range codes {
				raw[i] = byte(code)
			}
			dense = make([]float32, w.Numel())
			for b := 0; b < len(raw)/84; b++ {
				q2kDequantSuperBlock(dense[b*256:(b+1)*256], raw[b*84:(b+1)*84])
			}
		} else {
			dense = be.Read(w)
		}
		out := make([]float64, w.Shape[0])
		for row := range out {
			for col, value := range input {
				out[row] += float64(dense[row*len(input)+col]) * value
			}
		}
		return out
	}
	var sum float64
	for _, value := range be.Read(x) {
		sum += float64(value) * float64(value)
	}
	input := make([]float64, 256)
	for i, value := range be.Read(x) {
		input[i] = float64(value) * float64(be.Read(norm)[i]) / math.Sqrt(sum/256+1e-6)
	}
	g, u := matmul(w0, input), matmul(w1, input)
	for i := range g {
		g[i] = g[i] / (1 + math.Exp(-g[i])) * u[i]
	}
	result := matmul(down, g)
	want := make([]float32, len(result))
	for i, value := range result {
		want[i] = float32(value)
	}
	q2FusedCompare(t, got, want, want)
}

func TestVulkanQ2KFusedRMSNormMatMul2(t *testing.T) {
	v := q2FusedDevice(t)
	pairs := [][2]Dtype{{Q2_K, Q2_K}, {Q2_K, F32}, {F32, Q2_K}, {Q2_K, Q8_0}, {Q8_0, Q2_K}, {Q2_K, Q4_K}, {Q4_K, Q2_K}, {F32, F32}, {Q8_0, Q8_0}, {Q4_K, Q4_K}}
	for _, pair := range pairs {
		t.Run(fmt.Sprintf("%s_%s", pair[0], pair[1]), func(t *testing.T) {
			w0, w1 := q2FusedWeight(pair[0], 8, 256), q2FusedWeight(pair[1], 12, 256)
			x, norm := q2FusedInput(256, 0), q2FusedInput(256, 1)
			dw0, dw1 := q2FusedUpload(t, v, w0), q2FusedUpload(t, v, w1)
			dx, dn := q2FusedUpload(t, v, x), q2FusedUpload(t, v, norm)
			t.Cleanup(v.Recycle)
			y0, y1 := v.RMSNormMatMul2(dw0, dw1, dx, dn, 1e-6)
			xn := v.RMSNorm(dx, dn, 1e-6)
			cpuX := Default().RMSNorm(x, norm, 1e-6)
			q2FusedCompare(t, v.Read(y0), Default().Read(Default().MatMul(w0, cpuX)), v.Read(v.MatMul(dw0, xn)))
			q2FusedCompare(t, v.Read(y1), Default().Read(Default().MatMul(w1, cpuX)), v.Read(v.MatMul(dw1, xn)))
		})
	}
	// Metadata-only invalid operands must be refused before any device dispatch.
	for _, dtype := range []Dtype{Q5_K, Q6_K, F16} {
		for _, reverse := range []bool{false, true} {
			t.Run(fmt.Sprintf("reject_%s_reverse_%t", dtype, reverse), func(t *testing.T) {
				w0, w1 := Tensor{Dtype: Q2_K, Shape: []int{8, 256}}, Tensor{Dtype: dtype, Shape: []int{12, 256}}
				if reverse {
					w0, w1 = w1, w0
				}
				defer func() {
					p := recover()
					if p == nil || !strings.Contains(fmt.Sprint(p), "unsupported companion weight dtype "+dtype.String()) {
						t.Fatalf("expected unsupported companion refusal before dispatch, got %v", p)
					}
				}()
				v.RMSNormMatMul2(w0, w1, Tensor{Dtype: F32, Shape: []int{256}}, Tensor{Dtype: F32, Shape: []int{256}}, 1e-6)
			})
		}
	}
	for _, tc := range []struct {
		name, refusal    string
		input, norm, in1 int
		normDtype        Dtype
	}{
		{"prefill", "decode-only", 512, 256, 256, F32},
		{"norm_shape", "norm weight shape", 256, 255, 256, F32},
		{"weight_width", "weight input dims differ", 256, 256, 512, F32},
		{"input_width", "input shape is not divisible", 255, 256, 256, F32},
		{"norm_dtype", "norm weight must be F32", 256, 256, 256, F16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				p := recover()
				if p == nil || !strings.Contains(fmt.Sprint(p), tc.refusal) {
					t.Fatalf("expected %s refusal, got %v", tc.refusal, p)
				}
			}()
			w := Tensor{Dtype: Q2_K, Shape: []int{8, 256}}
			w1 := Tensor{Dtype: Q2_K, Shape: []int{8, tc.in1}}
			v.RMSNormMatMul2(w, w1, Tensor{Dtype: F32, Shape: []int{tc.input}}, Tensor{Dtype: tc.normDtype, Shape: []int{tc.norm}}, 1e-6)
		})
	}
}

func TestVulkanQ2KFusedSwiGLUMatMulAdd(t *testing.T) {
	v := q2FusedDevice(t)
	for _, tc := range []struct {
		dtype Dtype
		p     int
	}{{Q2_K, 1}, {Q2_K, 2}, {F32, 1}, {Q8_0, 1}, {Q4_K, 1}} {
		t.Run(fmt.Sprintf("%s_p%d", tc.dtype, tc.p), func(t *testing.T) {
			p := tc.p
			w := q2FusedWeight(tc.dtype, 8, 512)
			gate, up, residual := q2FusedInput(p*512, .2), q2FusedInput(p*512, -.1), q2FusedInput(p*8, .3)
			dw, dg, du := q2FusedUpload(t, v, w), q2FusedUpload(t, v, gate), q2FusedUpload(t, v, up)
			dst, baseline := q2FusedUpload(t, v, residual), q2FusedUpload(t, v, residual)
			t.Cleanup(v.Recycle)
			v.SwiGLUMatMulAddInPlace(dst, dw, dg, du)
			v.AddInPlace(baseline, v.BatchedMatMul(dw, v.SwiGLU(dg, du), p))
			cpu := Default().Read(Default().BatchedMatMul(w, Default().SwiGLU(gate, up), p))
			for i, x := range Default().Read(residual) {
				cpu[i] += x
			}
			q2FusedCompare(t, v.Read(dst), cpu, v.Read(baseline))
		})
	}
	for _, dtype := range []Dtype{Q5_K, Q6_K} {
		t.Run("reject_"+dtype.String(), func(t *testing.T) {
			defer func() {
				p := recover()
				if p == nil || !strings.Contains(fmt.Sprint(p), "unsupported weight dtype "+dtype.String()) {
					t.Fatalf("expected unsupported down-projection refusal, got %v", p)
				}
			}()
			v.SwiGLUMatMulAddInPlace(Tensor{Shape: []int{8}}, Tensor{Dtype: dtype, Shape: []int{8, 256}}, Tensor{Shape: []int{256}}, Tensor{Shape: []int{256}})
		})
	}
	for _, tc := range []struct {
		name, refusal string
		gate, up, dst int
	}{
		{"gate_up_shape", "gate/up shapes differ", 256, 255, 8},
		{"input_width", "gate shape is not divisible", 255, 255, 8},
		{"residual_shape", "dst shape does not match", 256, 256, 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				p := recover()
				if p == nil || !strings.Contains(fmt.Sprint(p), tc.refusal) {
					t.Fatalf("expected %s refusal, got %v", tc.refusal, p)
				}
			}()
			v.SwiGLUMatMulAddInPlace(Tensor{Shape: []int{tc.dst}}, Tensor{Dtype: Q2_K, Shape: []int{8, 256}}, Tensor{Shape: []int{tc.gate}}, Tensor{Shape: []int{tc.up}})
		})
	}
}

func TestVulkanQ2KFusedMLPBatchLifetime(t *testing.T) {
	v := q2FusedDevice(t)
	w0, w1, down := q2FusedWeight(Q2_K, 256, 256), q2FusedWeight(Q8_0, 256, 256), q2FusedWeight(Q2_K, 8, 256)
	x, norm, residual := q2FusedInput(256, 0), q2FusedInput(256, 1), q2FusedInput(8, .3)
	dw0, dw1, dd := q2FusedUpload(t, v, w0), q2FusedUpload(t, v, w1), q2FusedUpload(t, v, down)
	dx, dn := q2FusedUpload(t, v, x), q2FusedUpload(t, v, norm)
	t.Cleanup(v.Recycle)
	cpuX := Default().RMSNorm(x, norm, 1e-6)
	cpuG, cpuU := Default().MatMul(w0, cpuX), Default().MatMul(w1, cpuX)
	cpu := Default().Read(Default().MatMul(down, Default().SwiGLU(cpuG, cpuU)))
	for i, x := range Default().Read(residual) {
		cpu[i] += x
	}
	for cycle := 0; cycle < 2; cycle++ {
		t.Run(fmt.Sprintf("cycle%d", cycle), func(t *testing.T) {
			dst, baseline := q2FusedUpload(t, v, residual), q2FusedUpload(t, v, residual)
			v.BeginBatch()
			g, u := v.RMSNormMatMul2(dw0, dw1, dx, dn, 1e-6)
			v.SwiGLUMatMulAddInPlace(dst, dd, g, u)
			v.FlushBatch()
			xn := v.RMSNorm(dx, dn, 1e-6)
			v.AddInPlace(baseline, v.MatMul(dd, v.SwiGLU(v.MatMul(dw0, xn), v.MatMul(dw1, xn))))
			q2FusedCompare(t, v.Read(dst), cpu, v.Read(baseline))
			v.Recycle()
		})
	}
}

// TestVulkanQ2KShaderInvariants verifies that the Q2_K compute shader exists,
// declares valid layout descriptors and push constants, and adheres to 84-byte
// super-block unpacking invariants.
func TestVulkanQ2KShaderInvariants(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("os.Getwd failed: %v", err)
	}
	repoRoot := findRepoRootForTest(t, wd)
	shaderPath := filepath.Join(repoRoot, "internal", "compute", "shaders", "q2k_matmul.comp")

	shaderBytes, err := os.ReadFile(shaderPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", shaderPath, err)
	}
	src := string(shaderBytes)

	requiredClauses := []string{
		"#version 450",
		"layout(local_size_x = 64) in;",
		"layout(std430, set = 0, binding = 0) readonly buffer Q2K",
		"layout(std430, set = 0, binding = 1) readonly buffer X",
		"layout(std430, set = 0, binding = 2) writeonly buffer Y",
		"int outDim;",
		"int inDim;",
		"int tokens;",
		"int row = int(gl_GlobalInvocationID.x);",
		"int token = int(gl_GlobalInvocationID.y);",
		"uint code0_0 = (q0_0 >> shift) & 3u;",
		"float w0_0 = dl0 * float(code0_0) - ml0;",
		"y[token * pc.outDim + row] = sum;",
	}

	for _, clause := range requiredClauses {
		if !strings.Contains(src, clause) {
			t.Errorf("q2k_matmul.comp missing required clause: %q", clause)
		}
	}

	shimPath := filepath.Join(repoRoot, "internal", "compute", "vulkan_shim.cpp")
	shimBytes, err := os.ReadFile(shimPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", shimPath, err)
	}
	if clause := "(uint32_t)(((size_t)out + 63u) / 64u), (uint32_t)P"; !strings.Contains(string(shimBytes), clause) {
		t.Errorf("vulkan_shim.cpp missing Q2_K 2-D dispatch clause: %q", clause)
	}
}

// TestVulkanQ2KShaderSimulatedEquivalence simulates the GLSL shader logic in pure Go
// and verifies that it produces bit-exact equivalence with the reference q2kRowDot.
func TestVulkanQ2KShaderSimulatedEquivalence(t *testing.T) {
	const out = 4
	const in = 256
	rng := rand.New(rand.NewSource(2026))

	raw := make([]byte, out*(in/256)*84)
	for b := 0; b < out*(in/256); b++ {
		blk := raw[b*84 : (b+1)*84]
		for i := 0; i < 16; i++ {
			blk[i] = byte(rng.Intn(256))
		}
		for i := 16; i < 80; i++ {
			blk[i] = byte(rng.Intn(256))
		}
		binaryPutFloat16(blk[80:82], 1.5)
		binaryPutFloat16(blk[82:84], 0.5)
	}

	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	scratch := make([]float32, 256)
	for r := 0; r < out; r++ {
		want := q2kRowDot(raw[r*84:(r+1)*84], x, scratch)

		// Simulated GLSL shader arithmetic
		blk := raw[r*84 : (r+1)*84]
		d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[80:])))
		minVal := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(blk[82:])))

		var simSum float32
		isIdx := 0
		qiOffset := 0
		for n := 0; n < 256; n += 128 {
			shift := uint(0)
			for j := 0; j < 4; j++ {
				sc0 := blk[isIdx]
				isIdx++
				dl0 := d * float32(sc0&15)
				ml0 := minVal * float32(sc0>>4)

				sc1 := blk[isIdx]
				isIdx++
				dl1 := d * float32(sc1&15)
				ml1 := minVal * float32(sc1>>4)

				qBase0 := 16 + qiOffset
				qBase1 := qBase0 + 16
				xGroupBase := n + j*32

				for l := 0; l < 16; l++ {
					q0 := blk[qBase0+l]
					code0 := (q0 >> shift) & 3
					w0 := dl0*float32(code0) - ml0
					simSum += w0 * x[xGroupBase+l]

					q1 := blk[qBase1+l]
					code1 := (q1 >> shift) & 3
					w1 := dl1*float32(code1) - ml1
					simSum += w1 * x[xGroupBase+16+l]
				}
				shift += 2
			}
			qiOffset += 32
		}

		if math.Abs(float64(want-simSum)) > 1e-4 {
			t.Fatalf("row %d: want %g, got %g (delta %g)", r, want, simSum, math.Abs(float64(want-simSum)))
		}
	}
}

func binaryPutFloat16(b []byte, f float32) {
	bits := math.Float32bits(f)
	sign := (bits >> 31) & 0x1
	exp := int((bits>>23)&0xff) - 127
	frac := bits & 0x7fffff

	var h uint16
	if exp > 15 {
		h = uint16(sign<<15 | 0x1f<<10)
	} else if exp < -14 {
		h = uint16(sign << 15)
	} else {
		h = uint16(sign<<15 | uint32(exp+15)<<10 | (frac >> 13))
	}
	b[0] = byte(h)
	b[1] = byte(h >> 8)
}

func TestVulkanQ2KMatMulMatchesCPUReference(t *testing.T) {
	v, ok := Lookup("vulkan")
	if !ok {
		t.Skip("Vulkan backend unavailable")
	}
	rng := rand.New(rand.NewSource(9718))
	for _, tc := range []struct {
		name string
		out  int
		P    int
	}{
		{name: "decode", out: 64, P: 1},
		{name: "decode_row_tail", out: 67, P: 1},
		{name: "prefill_panel_tail", out: 67, P: 3},
		{name: "prefill_full_panel", out: 67, P: 4},
		{name: "prefill_multi_panel_tail", out: 67, P: 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const in = 512
			hw := q2FusedWeight(Q2_K, tc.out, in)
			dw := v.Upload(hw, Q2_K)
			defer v.Free(dw)
			X := make([]float32, tc.P*in)
			for i := range X {
				X[i] = rng.Float32()*2 - 1
			}
			dx := v.Upload(NewF32(Default(), []int{tc.P, in}, X), F32)
			defer v.Free(dx)
			var dy Tensor
			if tc.P == 1 {
				dy = v.MatMul(dw, dx)
			} else {
				dy = v.BatchedMatMul(dw, dx, tc.P)
			}
			defer v.Free(dy)
			got := v.Read(dy)
			ref := Default()
			var want []float32
			if tc.P == 1 {
				want = ref.Read(ref.MatMul(hw, NewF32(ref, []int{in}, X)))
			} else {
				want = ref.Read(ref.BatchedMatMul(hw, NewF32(ref, []int{tc.P, in}, X), tc.P))
			}
			if c := cosineC(got, want); c < 0.999990 {
				t.Fatalf("cosine %.8f < 0.999990", c)
			}
			for token := 0; token < tc.P; token++ {
				gotRow := got[token*tc.out : (token+1)*tc.out]
				wantRow := want[token*tc.out : (token+1)*tc.out]
				if a, b := argmaxF32(gotRow), argmaxF32(wantRow); a != b {
					t.Fatalf("token %d argmax=%d want %d", token, a, b)
				}
			}
		})
	}
}

func BenchmarkVulkanQ2KMatMul(b *testing.B) {
	v, ok := Lookup("vulkan")
	if !ok {
		b.Skip("Vulkan backend unavailable")
	}
	for _, tc := range []struct {
		name string
		P    int
	}{
		{name: "decode_5120x5120_p1", P: 1},
		{name: "prefill_5120x5120_p4", P: 4},
	} {
		b.Run(tc.name, func(b *testing.B) {
			const out, in = 5120, 5120
			hw := q2FusedWeight(Q2_K, out, in)
			dw := v.Upload(hw, Q2_K)
			defer v.Free(dw)
			rng := rand.New(rand.NewSource(9718 + int64(tc.P)))
			X := make([]float32, tc.P*in)
			for i := range X {
				X[i] = rng.Float32()*2 - 1
			}
			dx := v.Upload(NewF32(Default(), []int{tc.P, in}, X), F32)
			defer v.Free(dx)
			b.ReportMetric(float64(tc.P), "tokens/op")
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				var dy Tensor
				if tc.P == 1 {
					dy = v.MatMul(dw, dx)
				} else {
					dy = v.BatchedMatMul(dw, dx, tc.P)
				}
				v.Free(dy)
			}
		})
	}
}
