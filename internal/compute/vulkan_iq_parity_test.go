//go:build vulkan && (windows || linux) && cgo

package compute

// fak-test:runtime slow est=30s

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
)

// Native raw i-quant device-matvec parity witnesses (ticket halo-qwen-default-perf 05f),
// table-driven over every dtype registered in rawIQFormats so a new format is covered
// without a test edit. The oracle is CPU float64 math over DequantRawIQ weights; the
// target is the format's Vulkan iq*_matvec kernel on the verbatim super-block bytes. Both read the same f32
// activations, so the only legitimate difference is reduction order.

const rawIQParityEps = 1e-6

// rawIQParityDtypes returns the registered native raw i-quant dtypes in a stable order.
func rawIQParityDtypes() []Dtype {
	dts := make([]Dtype, 0, len(rawIQFormats))
	for dt := range rawIQFormats {
		dts = append(dts, dt)
	}
	sort.Slice(dts, func(i, j int) bool { return dts[i] < dts[j] })
	return dts
}

// rawIQParityWeight builds a random valid [out,in] host weight of raw i-quant dtype dt:
// every block is random bytes with bytes [0:2] overwritten by a small finite f16 d (all
// i-quants keep d there; every other bit pattern is a valid encoding). Also returns the
// CPU dequant.
func rawIQParityWeight(dt Dtype, seed int64, out, in int) (Tensor, []float32) {
	bb := rawIQFormats[dt].blockBytes
	rng := rand.New(rand.NewSource(seed))
	n := out * (in / rawIQSuper)
	raw := make([]byte, n*bb)
	rng.Read(raw)
	for b := 0; b < n; b++ {
		binaryPutFloat16(raw[b*bb:b*bb+2], float32(0.01+rng.Float64()*0.04))
	}
	return NewRawIQ(Default(), dt, []int{out, in}, raw), DequantRawIQ(dt, out, in, raw)
}

// rawIQOracleMatVec returns W*x for each of the len(x)/in tokens in float64 accumulation.
func rawIQOracleMatVec(deq []float32, out, in int, x []float32) []float32 {
	P := len(x) / in
	y := make([]float32, P*out)
	for p := 0; p < P; p++ {
		xs := x[p*in : (p+1)*in]
		for o := 0; o < out; o++ {
			row := deq[o*in : (o+1)*in]
			var s float64
			for i, w := range row {
				s += float64(w) * float64(xs[i])
			}
			y[p*out+o] = float32(s)
		}
	}
	return y
}

func rawIQOracleRMSNorm(x, norm []float32, eps float64) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(ss/float64(len(x))+eps)
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(float64(v) * inv * float64(norm[i]))
	}
	return out
}

func rawIQOracleSwiGLU(g, u []float32) []float32 {
	out := make([]float32, len(g))
	for i := range g {
		gf := float64(g[i])
		out[i] = float32(gf / (1 + math.Exp(-gf)) * float64(u[i]))
	}
	return out
}

func rawIQArgmax(xs []float32) int {
	best := 0
	for i, x := range xs {
		if x > xs[best] {
			best = i
		}
	}
	return best
}

// rawIQAssertClose enforces |got-want| <= rel*max(1, max|want|) per output.
func rawIQAssertClose(t *testing.T, label string, want, got []float32, rel float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: length want=%d got=%d", label, len(want), len(got))
	}
	assertFinite(t, label, got)
	var maxWant float64
	for _, w := range want {
		maxWant = math.Max(maxWant, math.Abs(float64(w)))
	}
	bound := rel * math.Max(1, maxWant)
	for i := range want {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > bound {
			t.Fatalf("%s: output %d got %g want %g |delta| %g > bound %g (max|want| %g)",
				label, i, got[i], want[i], d, bound, maxWant)
		}
	}
}

// rawIQAssertArgmax requires the target's argmax to be the oracle's, unless the oracle's
// own values at the two indices are closer than the reduction-order bound (a true tie).
func rawIQAssertArgmax(t *testing.T, label string, want, got []float32, rel float64) {
	t.Helper()
	aw, ag := rawIQArgmax(want), rawIQArgmax(got)
	if aw == ag {
		return
	}
	var maxWant float64
	for _, w := range want {
		maxWant = math.Max(maxWant, math.Abs(float64(w)))
	}
	if gap := float64(want[aw]) - float64(want[ag]); gap > rel*math.Max(1, maxWant) {
		t.Fatalf("%s: argmax got %d want %d (oracle gap %g)", label, ag, aw, gap)
	}
}

// requireRawIQNative enables dt's native kernel for the test and skips when its pipeline
// was not built; the previous selection is restored on cleanup.
func requireRawIQNative(t *testing.T, dt Dtype) {
	t.Helper()
	requireDecodeKernel(t, dt.String()+"_matvec", func(on bool) bool { return vulkanDebugSelectIQMatvec(dt, on) })
}

func rawIQUploadF32(v *vulkanBackend, shape []int, data []float32) Tensor {
	return v.Upload(NewF32(Default(), shape, data), F32)
}

// fak-test:runtime slow est=30s
func TestVulkanRawIQMatvecParity(t *testing.T) {
	v := vk(t)
	for _, dt := range rawIQParityDtypes() {
		t.Run(dt.String(), func(t *testing.T) {
			requireRawIQNative(t, dt)
			rawIQMatvecParityCases(t, v, dt)
		})
	}
}

func rawIQMatvecParityCases(t *testing.T, v *vulkanBackend, dt Dtype) {
	const tol = 1e-4
	for _, tc := range []struct{ out, in int }{
		{37, 256},
		{1029, 2048},
		{5120, 17408},
		{17408, 5120},
	} {
		t.Run(fmt.Sprintf("out%d_in%d", tc.out, tc.in), func(t *testing.T) {
			out, in := tc.out, tc.in
			seed := int64(0x1f4 + out*7 + in + int(dt)*1000003)
			hw, deq := rawIQParityWeight(dt, seed, out, in)
			dw := v.Upload(hw, dt)
			defer func() {
				v.Free(dw)
				v.Recycle()
			}()
			if dw.Dtype != dt {
				t.Skipf("native %v upload unavailable (uploaded dtype %v)", dt, dw.Dtype)
			}
			s := lcg(uint64(seed))

			t.Run("matmul", func(t *testing.T) {
				x := randVec(&s, in)
				want := rawIQOracleMatVec(deq, out, in, x)
				dx := rawIQUploadF32(v, []int{in}, x)
				defer func() { v.Free(dx); v.Recycle() }()
				got := copyRead(v, v.MatMul(dw, dx))
				decodeParityDump(t, t.Name(), want, got)
				rawIQAssertClose(t, "MatMul vs CPU", want, got, tol)
				rawIQAssertArgmax(t, "MatMul argmax", want, got, tol)
			})

			t.Run("batched_p3", func(t *testing.T) {
				const P = 3
				X := randVec(&s, P*in)
				want := rawIQOracleMatVec(deq, out, in, X)
				dX := rawIQUploadF32(v, []int{P, in}, X)
				defer func() { v.Free(dX); v.Recycle() }()
				got := copyRead(v, v.BatchedMatMul(dw, dX, P))
				decodeParityDump(t, t.Name(), want, got)
				rawIQAssertClose(t, "BatchedMatMul vs CPU", want, got, tol)
				for p := 0; p < P; p++ {
					rawIQAssertArgmax(t, fmt.Sprintf("BatchedMatMul token %d argmax", p),
						want[p*out:(p+1)*out], got[p*out:(p+1)*out], tol)
				}
			})

			rmsInputs := func() (x, norm, xn []float32) {
				x = randVec(&s, in)
				norm = randVec(&s, in)
				for i := range norm {
					norm[i] = 1 + 0.2*norm[i]
				}
				return x, norm, rawIQOracleRMSNorm(x, norm, rawIQParityEps)
			}

			t.Run("rmsnorm_matmul2_pair", func(t *testing.T) {
				out1 := out - out/4 + 3
				hw1, deq1 := rawIQParityWeight(dt, seed+1, out1, in)
				dw1 := v.Upload(hw1, dt)
				x, norm, xn := rmsInputs()
				dx, dn := rawIQUploadF32(v, []int{in}, x), rawIQUploadF32(v, []int{in}, norm)
				defer func() {
					v.Free(dw1)
					v.Free(dx)
					v.Free(dn)
					v.Recycle()
				}()
				if dw1.Dtype != dt {
					t.Fatalf("second %v upload dtype %v, want %v", dt, dw1.Dtype, dt)
				}
				want0 := rawIQOracleMatVec(deq, out, in, xn)
				want1 := rawIQOracleMatVec(deq1, out1, in, xn)
				y0, y1 := v.RMSNormMatMul2(dw, dw1, dx, dn, rawIQParityEps)
				got0, got1 := copyRead(v, y0), copyRead(v, y1)
				decodeParityDump(t, t.Name()+"_y0", want0, got0)
				decodeParityDump(t, t.Name()+"_y1", want1, got1)
				rawIQAssertClose(t, "pair y0 vs CPU", want0, got0, tol)
				rawIQAssertClose(t, "pair y1 vs CPU", want1, got1, tol)
			})

			t.Run("rmsnorm_matmul2_mixed_q8", func(t *testing.T) {
				const out1 = 45
				c := Default()
				q1 := QuantizeQ8(c, []int{out1, in}, randVec(&s, out1*in), 32)
				dw1 := v.Upload(q1, Q8_0)
				x, norm, xn := rmsInputs()
				dx, dn := rawIQUploadF32(v, []int{in}, x), rawIQUploadF32(v, []int{in}, norm)
				defer func() {
					v.Free(dw1)
					v.Free(dx)
					v.Free(dn)
					v.Recycle()
				}()
				want0 := rawIQOracleMatVec(deq, out, in, xn)
				want1 := c.Read(c.MatMul(q1, NewF32(c, []int{in}, xn)))
				y0, y1 := v.RMSNormMatMul2(dw, dw1, dx, dn, rawIQParityEps)
				got0, got1 := copyRead(v, y0), copyRead(v, y1)
				decodeParityDump(t, t.Name()+"_y0", want0, got0)
				decodeParityDump(t, t.Name()+"_y1", want1, got1)
				rawIQAssertClose(t, "mixed raw-IQ y0 vs CPU", want0, got0, tol)
				// The Q8_0 companion may quantize its activation; bound it accordingly.
				rawIQAssertClose(t, "mixed Q8_0 y1 vs CPU", want1, got1, 2e-2)
			})

			t.Run("rmsnorm_matmul2_mixed_f32", func(t *testing.T) {
				const out1 = 45
				w1 := randVec(&s, out1*in)
				dw1 := rawIQUploadF32(v, []int{out1, in}, w1)
				x, norm, xn := rmsInputs()
				dx, dn := rawIQUploadF32(v, []int{in}, x), rawIQUploadF32(v, []int{in}, norm)
				defer func() {
					v.Free(dw1)
					v.Free(dx)
					v.Free(dn)
					v.Recycle()
				}()
				want0 := rawIQOracleMatVec(deq, out, in, xn)
				want1 := rawIQOracleMatVec(w1, out1, in, xn)
				y0, y1 := v.RMSNormMatMul2(dw, dw1, dx, dn, rawIQParityEps)
				got0, got1 := copyRead(v, y0), copyRead(v, y1)
				decodeParityDump(t, t.Name()+"_y0", want0, got0)
				decodeParityDump(t, t.Name()+"_y1", want1, got1)
				rawIQAssertClose(t, "mixed raw-IQ y0 vs CPU", want0, got0, tol)
				rawIQAssertClose(t, "mixed F32 y1 vs CPU", want1, got1, tol)
			})

			for _, P := range []int{1, 2} {
				t.Run(fmt.Sprintf("swiglu_matmul_add_p%d", P), func(t *testing.T) {
					g, u := randVec(&s, P*in), randVec(&s, P*in)
					residual := randVec(&s, P*out)
					want := rawIQOracleMatVec(deq, out, in, rawIQOracleSwiGLU(g, u))
					for i := range want {
						want[i] = float32(float64(want[i]) + float64(residual[i]))
					}
					shapeIn, shapeOut := []int{P, in}, []int{P, out}
					if P == 1 {
						shapeIn, shapeOut = []int{in}, []int{out}
					}
					dg, du := rawIQUploadF32(v, shapeIn, g), rawIQUploadF32(v, shapeIn, u)
					dst := rawIQUploadF32(v, shapeOut, residual)
					defer func() {
						v.Free(dg)
						v.Free(du)
						v.Free(dst)
						v.Recycle()
					}()
					v.SwiGLUMatMulAddInPlace(dst, dw, dg, du)
					got := copyRead(v, dst)
					decodeParityDump(t, t.Name(), want, got)
					rawIQAssertClose(t, "SwiGLUMatMulAddInPlace vs CPU", want, got, tol)
				})
			}
		})
	}
}

// fak-test:runtime slow est=5s
func TestVulkanRawIQKillSwitchFallsBackToQ8(t *testing.T) {
	v := vk(t)
	for _, dt := range rawIQParityDtypes() {
		t.Run(dt.String(), func(t *testing.T) {
			prev := vulkanDebugSelectIQMatvec(dt, false)
			t.Cleanup(func() { vulkanDebugSelectIQMatvec(dt, prev) })
			const out, in = 1029, 2048
			hw, deq := rawIQParityWeight(dt, 0x5a5a+int64(dt), out, in)
			dw := v.Upload(hw, dt)
			s := lcg(0x5a5a)
			x := randVec(&s, in)
			dx := rawIQUploadF32(v, []int{in}, x)
			defer func() {
				v.Free(dw)
				v.Free(dx)
				v.Recycle()
			}()
			if dw.Dtype != Q8_0 {
				t.Fatalf("kill-switched %v upload dtype = %v, want Q8_0", dt, dw.Dtype)
			}
			want := rawIQOracleMatVec(deq, out, in, x)
			got := copyRead(v, v.MatMul(dw, dx))
			decodeParityDump(t, t.Name(), want, got)
			rawIQAssertClose(t, "Q8 fallback MatMul vs CPU", want, got, 2e-2)
		})
	}
}

// BenchmarkVulkanRawIQMatvec times single-token MatMul on each raw i-quant's native
// kernel versus the Q8_0 fallback representation of the same weight, reporting
// effective weight GB/s (sub-benchmarks <dtype>/<shape>/{native,q8}).
func BenchmarkVulkanRawIQMatvec(b *testing.B) {
	be, ok := Lookup("vulkan")
	if !ok {
		b.Skip("vulkan backend not registered (no reachable Vulkan device)")
	}
	v := be.(*vulkanBackend)
	for _, dt := range rawIQParityDtypes() {
		blockBytes, _ := RawIQBlockBytes(dt)
		for _, sh := range []struct{ out, in int }{{17408, 5120}, {5120, 17408}} {
			hw, _ := rawIQParityWeight(dt, int64(sh.out+sh.in), sh.out, sh.in)
			s := lcg(uint64(sh.out))
			x := randVec(&s, sh.in)
			for _, mode := range []struct {
				name      string
				native    bool
				wantDtype Dtype
				bytes     float64
			}{
				{"native", true, dt, float64(sh.out) * float64(sh.in) / 256 * float64(blockBytes)},
				{"q8", false, Q8_0, float64(sh.out) * float64(sh.in) * 34 / 32},
			} {
				b.Run(fmt.Sprintf("%s/out%d_in%d/%s", dt, sh.out, sh.in, mode.name), func(b *testing.B) {
					prev := vulkanDebugSelectIQMatvec(dt, mode.native)
					dw := v.Upload(hw, dt)
					vulkanDebugSelectIQMatvec(dt, prev)
					dx := rawIQUploadF32(v, []int{sh.in}, x)
					defer func() {
						v.Free(dw)
						v.Free(dx)
						v.Recycle()
					}()
					if dw.Dtype != mode.wantDtype {
						b.Skipf("upload dtype %v, want %v (kernel unavailable)", dw.Dtype, mode.wantDtype)
					}
					if mode.native {
						// Already-uploaded native weights need the kernel selected to run.
						p := vulkanDebugSelectIQMatvec(dt, true)
						defer vulkanDebugSelectIQMatvec(dt, p)
					}
					_ = v.Read(v.MatMul(dw, dx)) // warm the pipeline
					v.Recycle()

					const chunk = 64
					b.ResetTimer()
					var y Tensor
					for i := 0; i < b.N; {
						n := chunk
						if b.N-i < n {
							n = b.N - i
						}
						v.BeginBatch()
						for j := 0; j < n; j++ {
							y = v.MatMul(dw, dx)
						}
						v.FlushBatch()
						i += n
						if i < b.N {
							v.Recycle()
						}
					}
					_ = v.Read(y)
					b.StopTimer()
					v.Recycle()
					if sec := b.Elapsed().Seconds(); sec > 0 {
						b.ReportMetric(mode.bytes*float64(b.N)/sec/1e9, "GB/s")
					}
				})
			}
		}
	}
}
