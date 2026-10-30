//go:build cuda && cgo && fak_cuda_v41_engram_gate

package compute

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

// Oracle and fixture copied without arithmetic changes from
// vulkan_v41_engram_pool_test.go at be41c0a42f9878971df73a4043e58ee7eb2800d2.
// Local names avoid collisions in combined CUDA/Vulkan builds.
func cudaEngramWitnessBF16(v float32) float32 {
	bits := math.Float32bits(v)
	if bits&0x7f800000 != 0x7f800000 {
		bits += 0x7fff + ((bits >> 16) & 1)
	}
	return math.Float32frombits(bits & 0xffff0000)
}

// cudaEngramWitnessRef is the host Model.v41EngramInjectPrepared full-geometry
// arithmetic: separately rounded binary32 products and sums, F64-narrowed
// reciprocal roots, square root and exp.
func cudaEngramWitnessRef(streams, projected, norms []float32, rows, H int, eps float32) (next, status []float32) {
	next = make([]float32, rows*4*H)
	status = make([]float32, rows*4)
	rsqrt := func(x float32) float32 { return float32(1 / math.Sqrt(float64(x))) }
	for r := 0; r < rows; r++ {
		for s := 0; s < 4; s++ {
			h := streams[(r*4+s)*H : (r*4+s+1)*H]
			var h2, k2, dotSum float32
			for i := 0; i < H; i++ {
				key := cudaEngramWitnessBF16(projected[r*5*H+s*H+i])
				value := cudaEngramWitnessBF16(projected[r*5*H+4*H+i])
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
				value := cudaEngramWitnessBF16(projected[r*5*H+4*H+i])
				delta := gate * value
				sum := h[i] + delta
				next[(r*4+s)*H+i] = cudaEngramWitnessBF16(sum)
			}
		}
	}
	return next, status
}

// cudaEngramWitnessFixture fills BF16 streams, an F32 projection and norms. Row 1
// (when present) has all-zero streams; row 2 has keys equal to its streams and
// unit norms, so its gate argument is sqrt(sqrt(H)).
func cudaEngramWitnessFixture(rng *rand.Rand, rows, H int) (streams, projected, norms []float32) {
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
				streams[i] = cudaEngramWitnessBF16(float32(rng.NormFloat64() * scale))
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

func cudaEngramWitnessULPs(a, b float32) int {
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
// Candidate-only witness. No public capability activation or hardware claim.
func TestCUDAV41EngramGateDeviceWitness(t *testing.T) {
	if os.Getenv("FAK_CUDA_ENGRAM_GATE_WITNESS") != "1" {
		t.Skip("explicit hardware opt-in required; skip is not acceptance")
	}
	c, ok := Pick("cuda").(*cudaBackend)
	if !ok || c == nil {
		t.Fatal("required CUDA backend absent")
	}
	if c.SupportsV41EngramGate() {
		t.Fatal("candidate unexpectedly advertises public qualification")
	}
	for _, pin := range []string{"FAK_WITNESS_SOURCE_SHA", "FAK_WITNESS_LIB_SHA256", "FAK_WITNESS_DEVICE_UUID", "FAK_WITNESS_BUILD_RECEIPT"} {
		if os.Getenv(pin) == "" {
			t.Fatalf("required evidence pin %s absent", pin)
		}
		t.Logf("operator evidence %s=%s; must match returned receipts", pin, os.Getenv(pin))
	}
	checkedRead := func(t *testing.T, d Tensor, count int) []float32 {
		t.Helper()
		x, err := c.cudaV41EngramReadback(d, count)
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	exact := func(t *testing.T, label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length mismatch", label)
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d] got %08x want %08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	upload := func(t *testing.T, shape []int, data []float32) Tensor {
		t.Helper()
		d := c.UploadClass(NewF32(Default(), shape, data), F32, MemoryActivation, "engram-gate-witness")
		t.Cleanup(func() { c.Free(d) })
		exact(t, "upload", checkedRead(t, d, len(data)), data)
		return d
	}
	rng := rand.New(rand.NewSource(13811))
	for _, geom := range [][2]int{{1, 7}, {3, 64}, {3, 4096}, {5, 4096}, {1, 1}, {3, 65}, {1, 65536}} {
		rows, H := geom[0], geom[1]
		t.Run(fmt.Sprintf("rows%d_hidden%d", rows, H), func(t *testing.T) {
			s, p, n := cudaEngramWitnessFixture(rng, rows, H)
			if rows == 5 {
				p[4*5*H+3] = math.MaxFloat32
				p[3*5*H+4*H+1] = float32(math.Inf(1))
			}
			// Additional tiny geometry tests BF16 ties, signed zeros and subnormals.
			if H == 65 {
				edges := []uint32{0x3f808000, 0x3f818000, 0x80000000, 0, 0x00008000, 0x00018000, 0x80018000}
				for i, b := range edges {
					p[i] = math.Float32frombits(b)
					s[i] = cudaEngramWitnessBF16(math.Float32frombits(b))
				}
				p[2*5*H+2*H] = float32(math.NaN())
			}
			const eps = float32(1e-6)
			wantNext, wantStatus := cudaEngramWitnessRef(s, p, n, rows, H, eps)
			ds, dp, dn := upload(t, []int{rows, 4, H}, s), upload(t, []int{rows, 5 * H}, p), upload(t, []int{8 * H}, n)
			next, status, err := c.cudaV41EngramGate(ds, dp, dn, rows, H, eps, true)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { c.Free(next); c.Free(status) })
			nb, nok := next.buf.(*cudaBuf)
			sb, sok := status.buf.(*cudaBuf)
			if !nok || !sok || nb == nil || sb == nil || nb.ptr == nil || sb.ptr == nil || nb.ptr == sb.ptr {
				t.Fatal("outputs absent or aliased")
			}
			for _, d := range []Tensor{ds, dp, dn} {
				b := d.buf.(*cudaBuf)
				if nb.ptr == b.ptr || sb.ptr == b.ptr {
					t.Fatal("output aliases immutable input")
				}
			}
			exact(t, "status", checkedRead(t, status, len(wantStatus)), wantStatus)
			got := checkedRead(t, next, len(wantNext))
			worst := 0
			for i, v := range got {
				if wantStatus[i/H] != 0 {
					continue
				} // Same donor refusal scope: flagged streams cannot publish.
				if math.IsInf(float64(v), 0) || math.IsNaN(float64(v)) || math.Float32bits(v)&0xffff != 0 {
					t.Fatalf("published[%d] is not finite BF16", i)
				}
				worst = max(worst, cudaEngramWitnessULPs(v, wantNext[i]))
			}
			if worst > 1 {
				t.Fatalf("published streams differ by %d BF16 ULP (>1)", worst)
			}
			t.Logf("max BF16 ULP=%d; status exact", worst)
			exact(t, "streams immutable", checkedRead(t, ds, len(s)), s)
			exact(t, "projection immutable", checkedRead(t, dp, len(p)), p)
			exact(t, "norms immutable", checkedRead(t, dn, len(n)), n)
		})
	}
	t.Run("rejects", func(t *testing.T) {
		s, p, n := cudaEngramWitnessFixture(rng, 1, 8)
		ds, dp, dn := upload(t, []int{1, 4, 8}, s), upload(t, []int{40}, p), upload(t, []int{64}, n)
		cases := []struct {
			name    string
			s, p, n Tensor
			rows, h int
			eps     float32
			witness bool
		}{
			{"short projection", ds, dn, dn, 1, 8, 1e-6, true},
			{"NaN epsilon", ds, dp, dn, 1, 8, float32(math.NaN()), true},
			{"infinite epsilon", ds, dp, dn, 1, 8, float32(math.Inf(1)), true},
			{"host operand", NewF32(Default(), []int{1, 4, 8}, s), dp, dn, 1, 8, 1e-6, true},
			{"zero rows", ds, dp, dn, 0, 8, 1e-6, true},
			{"hidden bound", ds, dp, dn, 1, V41EngramGateMaxHidden + 1, 1e-6, true},
			{"index overflow", ds, dp, dn, math.MaxInt32, 8, 1e-6, true},
			{"public unqualified", ds, dp, dn, 1, 8, 1e-6, false},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				a, b, err := c.cudaV41EngramGate(tc.s, tc.p, tc.n, tc.rows, tc.h, tc.eps, tc.witness)
				if err == nil || a.buf != nil || b.buf != nil {
					t.Fatal("invalid/unqualified condition did not fail closed with empty outputs")
				}
			})
		}
		exact(t, "rejected streams immutable", checkedRead(t, ds, len(s)), s)
		exact(t, "rejected projection immutable", checkedRead(t, dp, len(p)), p)
		exact(t, "rejected norms immutable", checkedRead(t, dn, len(n)), n)
	})
}
