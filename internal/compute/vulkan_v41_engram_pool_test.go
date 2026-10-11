//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

func v41EngramRefBF16(v float32) float32 {
	bits := math.Float32bits(v)
	if bits&0x7f800000 != 0x7f800000 {
		bits += 0x7fff + ((bits >> 16) & 1)
	}
	return math.Float32frombits(bits & 0xffff0000)
}

// v41EngramGateRef is the host Model.v41EngramInjectPrepared full-geometry
// arithmetic: separately rounded binary32 products and sums, F64-narrowed
// reciprocal roots, square root and exp.
func v41EngramGateRef(streams, projected, norms []float32, rows, H int, eps float32) (next, status []float32) {
	next = make([]float32, rows*4*H)
	status = make([]float32, rows*4)
	rsqrt := func(x float32) float32 { return float32(1 / math.Sqrt(float64(x))) }
	for r := 0; r < rows; r++ {
		for s := 0; s < 4; s++ {
			h := streams[(r*4+s)*H : (r*4+s+1)*H]
			var h2, k2, dotSum float32
			for i := 0; i < H; i++ {
				key := v41EngramRefBF16(projected[r*5*H+s*H+i])
				value := v41EngramRefBF16(projected[r*5*H+4*H+i])
				if math.IsInf(float64(key), 0) || math.IsNaN(float64(key)) || math.IsInf(float64(value), 0) || math.IsNaN(float64(value)) {
					status[r*4+s] = 1
				}
				hh := h[i] * h[i]
				h2 += hh
				kk := key * key
				k2 += kk
				qk := norms[s*H+i] * norms[4*H+s*H+i]
				t1 := h[i] * qk
				t2 := t1 * key
				dotSum += t2
			}
			d1 := dotSum * rsqrt(h2/float32(H)+eps)
			d2 := d1 * rsqrt(k2/float32(H)+eps)
			dot := d2 * rsqrt(float32(H))
			a := dot
			if a < 0 {
				a = -a
			}
			floor := float32(1e-6)
			if !(a > floor) {
				a = floor
			}
			z := float32(math.Copysign(float64(float32(math.Sqrt(float64(a)))), float64(dot)))
			e := float32(math.Exp(float64(-z)))
			gate := float32(1) / (1 + e)
			for i := 0; i < H; i++ {
				value := v41EngramRefBF16(projected[r*5*H+4*H+i])
				delta := gate * value
				sum := h[i] + delta
				next[(r*4+s)*H+i] = v41EngramRefBF16(sum)
			}
		}
	}
	return next, status
}

// v41EngramGateFixture fills BF16 streams, an F32 projection and norms. Row 1
// (when present) has all-zero streams; row 2 has keys equal to its streams and
// unit norms, so its gate argument is sqrt(sqrt(H)).
func v41EngramGateFixture(rng *rand.Rand, rows, H int) (streams, projected, norms []float32) {
	streams = make([]float32, rows*4*H)
	projected = make([]float32, rows*5*H)
	norms = make([]float32, 8*H)
	for i := range norms {
		norms[i] = float32(0.5 + rng.Float64())
	}
	for r := 0; r < rows; r++ {
		scale := math.Ldexp(1, rng.Intn(8)-4)
		for i := r * 4 * H; i < (r+1)*4*H; i++ {
			if r != 1 {
				streams[i] = v41EngramRefBF16(float32(rng.NormFloat64() * scale))
			}
		}
		for i := r * 5 * H; i < (r+1)*5*H; i++ {
			projected[i] = float32(rng.NormFloat64() * scale * 3)
		}
		if r == 2 {
			for s := 0; s < 4; s++ {
				copy(projected[r*5*H+s*H:r*5*H+(s+1)*H], streams[(r*4+s)*H:(r*4+s+1)*H])
			}
		}
	}
	if rows > 2 {
		for i := range norms {
			norms[i] = 1
		}
	}
	return streams, projected, norms
}

func v41BF16ULPs(a, b float32) int {
	x, y := int64(math.Float32bits(a)>>16), int64(math.Float32bits(b)>>16)
	if (x >= 0x8000) != (y >= 0x8000) {
		return int(x&0x7fff + y&0x7fff)
	}
	d := x - y
	if d < 0 {
		d = -d
	}
	return int(d)
}

// fak-test:runtime integration est=5s lane=optin
// Ref #13811. Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no
// physical witness. The three serial reductions, the reciprocal and square
// roots and every BF16 publication are bit-exact; the gate's exp is a
// Cody-Waite F32 evaluation within about one ulp of the host's F64-narrowed exp,
// so a published element may differ from the host by at most one BF16 ulp, and
// status must match exactly.
func TestVulkanV41EngramGate(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41EngramGate() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("registered Vulkan build lacks the V4.1 Engram gate module or binary32 float controls")
		}
		t.Skip("V4.1 Engram gate module not loaded")
	}
	upload := func(t *testing.T, shape []int, x []float32) Tensor {
		d := v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "V4.1 Engram gate witness input")
		t.Cleanup(func() { v.Free(d) })
		return d
	}
	rng := rand.New(rand.NewSource(13811))
	for _, geom := range [][2]int{{1, 7}, {3, 64}, {3, 4096}, {5, 4096}} {
		rows, H := geom[0], geom[1]
		t.Run(fmt.Sprintf("rows%d_hidden%d", rows, H), func(t *testing.T) {
			streams, projected, norms := v41EngramGateFixture(rng, rows, H)
			const eps = float32(1e-6)
			if rows == 5 {
				projected[4*5*H+3] = math.MaxFloat32 // row 4 BF16 key overflow: status only
				projected[3*5*H+4*H+1] = float32(math.Inf(1))
			}
			wantNext, wantStatus := v41EngramGateRef(streams, projected, norms, rows, H, eps)
			next, status, err := v.V41EngramGate(upload(t, []int{rows, 4, H}, streams), upload(t, []int{rows, 5 * H}, projected), upload(t, []int{8 * H}, norms), rows, H, eps)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { v.Free(next); v.Free(status) })
			gotStatus := append([]float32(nil), v.Read(status)...)
			gotNext := append([]float32(nil), v.Read(next)...)
			for i := range wantStatus {
				if gotStatus[i] != wantStatus[i] {
					t.Fatalf("status[%d] = %g, want %g", i, gotStatus[i], wantStatus[i])
				}
			}
			worst, differ := 0, 0
			for r := 0; r < rows; r++ {
				for i := r * 4 * H; i < (r+1)*4*H; i++ {
					if wantStatus[r*4+(i-r*4*H)/H] != 0 {
						continue // a flagged stream is refused before publication
					}
					if math.Float32bits(gotNext[i]) != math.Float32bits(wantNext[i]) {
						differ++
						worst = max(worst, v41BF16ULPs(gotNext[i], wantNext[i]))
					}
				}
			}
			t.Logf("Engram gate: %d/%d published elements differ, max %d BF16 ulp", differ, rows*4*H, worst)
			if worst > 1 {
				t.Fatalf("published streams differ from the host by %d BF16 ulps (> 1)", worst)
			}
		})
	}
	t.Run("rejects", func(t *testing.T) {
		s, p, n := v41EngramGateFixture(rng, 1, 8)
		ds, dp, dn := upload(t, []int{1, 4, 8}, s), upload(t, []int{40}, p), upload(t, []int{64}, n)
		for name, call := range map[string]func() error{
			"short projection": func() error { _, _, err := v.V41EngramGate(ds, dn, dn, 1, 8, 1e-6); return err },
			"non-finite eps":   func() error { _, _, err := v.V41EngramGate(ds, dp, dn, 1, 8, float32(math.NaN())); return err },
			"host operand": func() error {
				_, _, err := v.V41EngramGate(NewF32(Default(), []int{1, 4, 8}, s), dp, dn, 1, 8, 1e-6)
				return err
			},
		} {
			if err := call(); err == nil {
				t.Fatalf("%s accepted", name)
			}
		}
	})
}

// v41CompressorPoolRef is the host V41CompressorPool.Push group emission.
func v41CompressorPoolRef(kv, scores []float32, ratio, width int) []float32 {
	out := make([]float32, width)
	for d := 0; d < width; d++ {
		m := scores[d]
		for t := 1; t < ratio; t++ {
			if s := scores[t*width+d]; s > m {
				m = s
			}
		}
		var denom float32
		w := make([]float32, ratio)
		for t := 0; t < ratio; t++ {
			w[t] = float32(math.Exp(float64(scores[t*width+d] - m)))
			denom += w[t]
		}
		var acc float32
		for t := 0; t < ratio; t++ {
			q := w[t] / denom
			p := kv[t*width+d] * q
			acc += p
		}
		out[d] = acc
	}
	return out
}

// fak-test:runtime integration est=3s lane=optin
// Ref #13812. Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no
// physical witness. The max, sums, products and quotients are separately
// rounded binary32 in token order; exp is a Cody-Waite F32 evaluation within
// about one ulp of the host's F64-narrowed exp, so each pooled value must lie
// within 8 ulps of the magnitude sum_t |kv_t|*w_t its weights carry.
func TestVulkanV41CompressorPool(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41CompressorPool() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("registered Vulkan build lacks the V4.1 compressor pool module or binary32 float controls")
		}
		t.Skip("V4.1 compressor pool module not loaded")
	}
	upload := func(t *testing.T, shape []int, x []float32) Tensor {
		d := v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "V4.1 compressor pool witness input")
		t.Cleanup(func() { v.Free(d) })
		return d
	}
	rng := rand.New(rand.NewSource(13812))
	for _, geom := range [][2]int{{2, 7}, {4, 512}, {128, 512}} {
		for _, scoreScale := range []float64{1, 40} {
			ratio, width := geom[0], geom[1]
			t.Run(fmt.Sprintf("ratio%d_width%d_scores%g", ratio, width, scoreScale), func(t *testing.T) {
				kv, scores := make([]float32, ratio*width), make([]float32, ratio*width)
				for i := range kv {
					kv[i] = float32(rng.NormFloat64())
					scores[i] = float32(rng.NormFloat64() * scoreScale)
				}
				want := v41CompressorPoolRef(kv, scores, ratio, width)
				out, err := v.V41CompressorPool(upload(t, []int{ratio, width}, kv), upload(t, []int{ratio, width}, scores), ratio, width)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { v.Free(out) })
				got := append([]float32(nil), v.Read(out)...)
				worst, exact := 0.0, 0
				for d := 0; d < width; d++ {
					m := float64(scores[d])
					for r := 1; r < ratio; r++ {
						m = math.Max(m, float64(scores[r*width+d]))
					}
					var denom, carried float64
					for r := 0; r < ratio; r++ {
						denom += math.Exp(float64(scores[r*width+d]) - m)
					}
					for r := 0; r < ratio; r++ {
						carried += math.Abs(float64(kv[r*width+d])) * math.Exp(float64(scores[r*width+d])-m) / denom
					}
					ulps := math.Abs(float64(got[d]-want[d])) / (carried * 0x1p-23)
					worst = math.Max(worst, ulps)
					if got[d] == want[d] {
						exact++
					}
				}
				t.Logf("compressor pool: %d/%d bit-exact, max %.2f ulps of the carried magnitude", exact, width, worst)
				if worst > 8 {
					t.Fatalf("pooled values differ by %.2f ulps of the carried magnitude (> 8)", worst)
				}
			})
		}
	}
	t.Run("rejects", func(t *testing.T) {
		kv := upload(t, []int{2, 4}, make([]float32, 8))
		for name, call := range map[string]func() error{
			"ratio one":    func() error { _, err := v.V41CompressorPool(kv, kv, 1, 8); return err },
			"short scores": func() error { _, err := v.V41CompressorPool(kv, kv, 4, 4); return err },
			"host operand": func() error {
				_, err := v.V41CompressorPool(NewF32(Default(), []int{2, 4}, make([]float32, 8)), kv, 2, 4)
				return err
			},
		} {
			if err := call(); err == nil {
				t.Fatalf("%s accepted", name)
			}
		}
	})
}
