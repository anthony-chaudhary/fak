//go:build cuda && cgo && fak_cuda_v41_compressor_pool

package compute

import (
	"errors"
	"fmt"
	"math"
	"testing"
)

// Host oracle: serial binary32 operations, F64 exp narrowed to F32. The
// candidate's Cody-Waite exp is approximate; acceptance uses 8 ulps of carried
// magnitude with F64 weights, matching the existing shader witness contract.
// There is no subnormal tolerance floor; zero carried magnitude requires equality.
func cudaPoolWitnessReference(kv, sc []float32, ratio, width int) ([]float32, []float64) {
	out := make([]float32, width)
	magnitude := make([]float64, width)
	for d := 0; d < width; d++ {
		mx := sc[d]
		for r := 1; r < ratio; r++ {
			if sc[r*width+d] > mx {
				mx = sc[r*width+d]
			}
		}
		weights := make([]float32, ratio)
		var denom float32
		for r := 0; r < ratio; r++ {
			shift := float32(sc[r*width+d] - mx)
			weights[r] = float32(math.Exp(float64(shift)))
			denom = float32(denom + weights[r])
		}
		for r := 0; r < ratio; r++ {
			q := float32(weights[r] / denom)
			prod := float32(kv[r*width+d] * q)
			out[d] = float32(out[d] + prod)
		}
		// Acceptance magnitude is independently evaluated in F64, exactly as
		// TestVulkanV41CompressorPool; do not substitute rounded F32 weights.
		var denominator64 float64
		for r := 0; r < ratio; r++ {
			denominator64 += math.Exp(float64(sc[r*width+d]) - float64(mx))
		}
		for r := 0; r < ratio; r++ {
			magnitude[d] += math.Abs(float64(kv[r*width+d])) * math.Exp(float64(sc[r*width+d])-float64(mx)) / denominator64
		}
	}
	return out, magnitude
}

// TestCUDAV41CompressorPoolDeviceWitness is an opt-in physical-device witness.
// Absence of a registered CUDA backend is a FAIL, never a successful skip.
// No throughput or model-level acceptance follows from this component test.
func TestCUDAV41CompressorPoolDeviceWitness(t *testing.T) {
	c, ok := Pick("cuda").(*cudaBackend)
	if !ok {
		t.Fatal("CUDA device required for compressor witness")
	}
	if c.SupportsV41CompressorPool() {
		t.Fatal("candidate must remain publicly unqualified")
	}
	for _, geom := range [][2]int{{2, 1}, {2, 63}, {4, 64}, {4, 65}, {128, 127}, {128, 128}, {128, 129}, {4096, 1}} {
		for _, mode := range []string{"finite", "ties", "extreme", "subnormal", "signed-zero"} {
			ratio, width := geom[0], geom[1]
			t.Run(fmt.Sprintf("r%d_w%d_%s", ratio, width, mode), func(t *testing.T) {
				kv, sc := make([]float32, ratio*width), make([]float32, ratio*width)
				for i := range kv {
					kv[i] = float32(i%17-8) / 16
					sc[i] = float32(i%13-6) / 4
					switch mode {
					case "ties":
						sc[i] = 3
					case "extreme":
						kv[i] = math.MaxFloat32 / 4
						if i/width%2 == 0 {
							sc[i] = math.MaxFloat32
						} else {
							sc[i] = -math.MaxFloat32
						}
					case "subnormal":
						kv[i] = 1
						sc[i] = -100
						if i/width == 0 {
							sc[i] = 0
							kv[i] = 0
						}
					case "signed-zero":
						kv[i] = math.Float32frombits(uint32(i%2) << 31)
						sc[i] = 0
					}
				}
				upload := func(v []float32) Tensor {
					x := c.UploadClass(NewF32(Default(), []int{ratio, width}, v), F32, MemoryActivation, "compressor-device-witness")
					t.Cleanup(func() { c.Free(x) })
					return x
				}
				a, b := upload(kv), upload(sc)
				out, err := c.cudaV41CompressorPool(a, b, ratio, width, true)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { c.Free(out) })
				ob := out.buf.(*cudaBuf)
				if ob.ptr == a.buf.(*cudaBuf).ptr || ob.ptr == b.buf.(*cudaBuf).ptr {
					t.Fatal("output aliases input")
				}
				got, err := c.cudaV41PoolWitnessRead(out, width)
				if err != nil {
					t.Fatal(err)
				}
				want, mag := cudaPoolWitnessReference(kv, sc, ratio, width)
				for d := range got {
					limit := 8 * mag[d] * 0x1p-23
					if math.IsNaN(float64(got[d])) || math.IsInf(float64(got[d]), 0) || math.Abs(float64(float32(got[d]-want[d]))) > limit {
						t.Fatalf("d=%d got=%g want=%g tolerance=%g", d, got[d], want[d], limit)
					}
				}
				// Read storage is independent; mutating it cannot affect device output.
				// Public dispatch still refuses even after a private witness succeeds.
				if _, err := c.V41CompressorPool(a, b, ratio, width); err == nil {
					t.Fatal("unqualified public dispatch accepted")
				}
				original := got[0]
				got[0] = 123
				again, err := c.cudaV41PoolWitnessRead(out, width)
				if err != nil {
					t.Fatal(err)
				}
				if math.Float32bits(again[0]) != math.Float32bits(original) {
					t.Fatal("host read aliases device output")
				}
			})
		}
	}
	t.Run("typed-rejections", func(t *testing.T) {
		check := func(out Tensor, err error) {
			t.Helper()
			var typed *BackendError
			if err == nil || !errors.As(err, &typed) || out.buf != nil {
				t.Fatalf("expected typed failure without output: %v", err)
			}
		}
		for _, g := range [][2]int{{1, 1}, {4097, 1}, {2, 0}, {2, math.MaxInt32}} {
			out, err := c.cudaV41CompressorPool(Tensor{}, Tensor{}, g[0], g[1], true)
			check(out, err)
		}
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			a := c.UploadClass(NewF32(Default(), []int{2, 1}, []float32{bad, 1}), F32, MemoryActivation, "compressor-negative")
			b := c.UploadClass(NewF32(Default(), []int{2, 1}, []float32{0, 0}), F32, MemoryActivation, "compressor-negative")
			out, err := c.cudaV41CompressorPool(a, b, 2, 1, true)
			check(out, err)
			out, err = c.cudaV41CompressorPool(b, a, 2, 1, true)
			check(out, err)
			c.Free(a)
			c.Free(b)
		}
		host := NewF32(Default(), []int{2, 1}, []float32{0, 0})
		out, err := c.cudaV41CompressorPool(host, host, 2, 1, true)
		check(out, err)
	})
}
