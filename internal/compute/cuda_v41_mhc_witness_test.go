//go:build cuda && cgo && fak_cuda_v41_mhc

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=3s lane=optin
// Bounded candidate witness, NOT full-model or performance qualification.
// Requires the separately reviewed checked-readback companion; never c.Read.
// Public capability remains false. No parallel tests: native state is global.
func TestCUDAV41MHCCandidateWitness(t *testing.T) {
	if os.Getenv("FAK_CUDA_MHC_WITNESS") != "1" {
		t.Skip("explicit candidate hardware opt-in required; skip is not acceptance")
	}
	c, ok := Pick("cuda").(*cudaBackend)
	if !ok || c == nil {
		t.Fatal("required CUDA backend absent")
	}
	if c.SupportsV41MHC() {
		t.Fatal("candidate unexpectedly advertises public qualification")
	}
	for _, pin := range []string{"FAK_WITNESS_SOURCE_SHA", "FAK_WITNESS_LIB_SHA256", "FAK_WITNESS_DEVICE_UUID", "FAK_WITNESS_BUILD_RECEIPT"} {
		if os.Getenv(pin) == "" {
			t.Fatalf("required external evidence pin %s absent", pin)
		}
		t.Logf("operator evidence %s=%s (must match returned build/device receipts)", pin, os.Getenv(pin))
	}
	check := func(t *testing.T, name string, got, want []float32, bound int64) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s length %d != %d", name, len(got), len(want))
		}
		var worst int64
		for i, v := range got {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("%s[%d] nonfinite", name, i)
			}
			d := int64(math.Float32bits(v)) - int64(math.Float32bits(want[i]))
			if d < 0 {
				d = -d
			}
			if d > worst {
				worst = d
			}
			if d > bound {
				t.Fatalf("%s[%d] got %08x want %08x distance %d > %d", name, i, math.Float32bits(v), math.Float32bits(want[i]), d, bound)
			}
		}
		t.Logf("%s max bit-distance=%d bound=%d", name, worst, bound)
	}
	upload := func(t *testing.T, shape []int, data []float32) Tensor {
		t.Helper()
		d := c.UploadClass(NewF32(Default(), shape, data), F32, MemoryActivation, "mhc-witness")
		t.Cleanup(func() { c.Free(d) })
		got, err := c.cudaV41MHCReadback(d, len(data))
		if err != nil {
			t.Fatal(err)
		}
		check(t, "upload-roundtrip", got, data, 0)
		return d
	}
	row := func(n, salt int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = float32((i*31+salt*17)%97-48) * 0.0371
		}
		edges := []uint32{0x3f808000, 0x3f818000, 0x80000000, 0, 0x00008000, 0x00018000, 0x80018000}
		for i, b := range edges {
			if i < len(x) {
				x[i] = math.Float32frombits(b)
			}
		}
		return x
	}
	for _, g := range [][2]int{{1, 7}, {3, 65}, {2, 4096}} {
		for _, iters := range []int{1, 20} {
			t.Run(fmt.Sprintf("r%d_h%d_iters%d", g[0], g[1], iters), func(t *testing.T) {
				r, h := g[0], g[1]
				const eps = float32(1e-6)
				x, s := row(r*h, 0), row(r*4*h, 1)
				for i := range s {
					s[i] = v41MHCRefBF16(s[i])
				}
				m, scale, base := v41MHCSplitFixture(r)
				coeff := v41MHCRefSplit(m, scale, base, r, iters, eps)
				raw := append([]float32(nil), m...)
				for i := range raw {
					raw[i] *= float32(4*h) * 0.01
				}
				params := append(append([]float32(nil), scale...), base...)
				dx, ds := upload(t, []int{r, h}, x), upload(t, []int{r, 4, h}, s)
				dm, dk := upload(t, []int{r, 24}, m), upload(t, []int{r, 24}, coeff)
				dscale, dbase := upload(t, []int{3}, scale), upload(t, []int{24}, base)
				draw, dp := upload(t, []int{r, 24}, raw), upload(t, []int{27}, params)
				cases := []struct {
					name  string
					op    V41MHCOp
					args  []Tensor
					want  []float32
					bound int64
				}{
					{"initial", V41MHCOpInitial, []Tensor{dx}, v41MHCRefInitial(x, r, h), 0},
					{"split", V41MHCOpSplit, []Tensor{dm, dscale, dbase}, coeff, 32},
					{"pre", V41MHCOpPre, []Tensor{ds, dk}, v41MHCRefPre(s, coeff, r, h), 0},
					{"post", V41MHCOpPost, []Tensor{dx, ds, dk}, v41MHCRefPost(x, s, coeff, r, h), 0},
					{"rms-split", V41MHCOpRMSSplit, []Tensor{draw, ds, dp}, v41MHCRefRMSSplit(raw, s, params, r, h, iters, eps, eps), 32},
				}
				for _, tc := range cases {
					out, err := c.cudaV41MHC(tc.op, r, h, iters, eps, eps, true, tc.args...)
					if err != nil {
						t.Fatalf("%s: %v", tc.name, err)
					}
					t.Cleanup(func() { c.Free(out) })
					for _, in := range tc.args {
						if out.buf == in.buf {
							t.Fatalf("%s aliases input", tc.name)
						}
					}
					got, err := c.cudaV41MHCReadback(out, len(tc.want))
					if err != nil {
						t.Fatalf("%s checked read: %v", tc.name, err)
					}
					check(t, tc.name, got, tc.want, tc.bound)
				}
				for _, v := range []struct {
					d    Tensor
					host []float32
				}{{dx, x}, {ds, s}, {dm, m}, {dk, coeff}, {dscale, scale}, {dbase, base}, {draw, raw}, {dp, params}} {
					got, err := c.cudaV41MHCReadback(v.d, len(v.host))
					if err != nil {
						t.Fatal(err)
					}
					check(t, "immutable-input", got, v.host, 0)
				}
			})
		}
	}
	t.Run("rejects", func(t *testing.T) {
		x := upload(t, []int{1, 7}, row(7, 0))
		reject := func(label string, backend *cudaBackend, input Tensor, rows, hidden int, witness bool) {
			t.Helper()
			out, err := backend.cudaV41MHC(V41MHCOpInitial, rows, hidden, 0, 0, 0, witness, input)
			if err == nil || out.buf != nil {
				t.Fatalf("%s must return error and no output: %v", label, err)
			}
		}
		reject("public-unqualified", c, x, 1, 7, false)
		reject("extent", c, x, 1, 8, true)
		reject("zero-rows", c, x, 0, 7, true)
		reject("host-backend", c, NewF32(Default(), []int{1, 7}, row(7, 0)), 1, 7, true)
		bad := x
		bad.Dtype = F16
		reject("dtype", c, bad, 1, 7, true)
		copied := *(x.buf.(*cudaBuf))
		copied.n = 4
		bad = x
		bad.buf = &copied
		reject("storage-short", c, bad, 1, 7, true)
		copied = *(x.buf.(*cudaBuf))
		copied.managed = true
		bad.buf = &copied
		reject("managed", c, bad, 1, 7, true)
		cudaMu.Lock()
		was := c.capturing
		c.capturing = true
		cudaMu.Unlock()
		func() {
			defer func() { cudaMu.Lock(); c.capturing = was; cudaMu.Unlock() }()
			reject("capture", c, x, 1, 7, true)
		}()
		isolated := &cudaBackend{}
		isolated.faultLatch.Observe(DeviceFaultExecution, "mhc-witness-synthetic", 1)
		reject("sticky-latch", isolated, x, 1, 7, true)
		stale := c.UploadClass(NewF32(Default(), []int{1, 7}, row(7, 0)), F32, MemoryActivation, "mhc-stale")
		c.Free(stale)
		reject("stale", c, stale, 1, 7, true)
	})
	t.Log("Native launch/completion failure injection and failed-drain quarantine require separate native fault-harness evidence; NOT established by this fixture")
}
