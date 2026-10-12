//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"testing"
)

// fak-test:runtime integration est=2s lane=optin
// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no physical
// witness. Tolerances are fixed before execution. Initial streams, pre-collapse
// and post-mix are bit-exact against the host oracle (separately rounded F32
// products and sums in stream order, integer BF16 RNE). The split computes exp
// and division in F32 (Cody-Waite expf, Markstein quotient; about 1 ulp each)
// where the host widens to F64, so each coefficient must agree within
// v41MHCSplitULPBound ulps of its own magnitude.
func TestVulkanV41MHC(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41MHC() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("registered Vulkan build lacks the V4.1 mHC module")
		}
		t.Skip("V4.1 mHC module not loaded")
	}
	if kind := v.v41PhysicalDeviceType(); kind != 1 && kind != 2 {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatalf("required physical GPU not established: deviceType=%d tier=%s", kind, v.Tier())
		}
		t.Skipf("physical GPU not established: deviceType=%d", kind)
	}
	upload := func(t *testing.T, shape []int, x []float32) Tensor {
		d := v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "V4.1 mHC witness input")
		t.Cleanup(func() { v.Free(d) })
		return d
	}
	read := func(t *testing.T, out Tensor, err error) []float32 {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { v.Free(out) })
		return append([]float32(nil), v.Read(out)...)
	}
	for _, geom := range [][2]int{{1, 7}, {3, 4096}} {
		for _, batched := range []bool{false, true} {
			rows, hidden := geom[0], geom[1]
			t.Run(fmt.Sprintf("rows%d_hidden%d_batch%t", rows, hidden, batched), func(t *testing.T) {
				x := v41MHCWitnessRow(rows*hidden, 0)
				streams := v41MHCWitnessRow(rows*4*hidden, 1)
				for i := range streams {
					streams[i] = v41MHCRefBF16(streams[i])
				}
				mixes, scale, base := v41MHCSplitFixture(rows)
				const iters, eps = 20, float32(1e-6)
				wantCoeffs := v41MHCRefSplit(mixes, scale, base, rows, iters, eps)

				dx := upload(t, []int{rows, hidden}, x)
				ds := upload(t, []int{rows, 4, hidden}, streams)
				dm := upload(t, []int{rows, 24}, mixes)
				dscale := upload(t, []int{3}, scale)
				dbase := upload(t, []int{24}, base)
				// Pre/post consume the host oracle's coefficients so their bit
				// witness is independent of the split's F32 exp tolerance.
				dk := upload(t, []int{rows, 24}, wantCoeffs)
				if batched {
					v.BeginBatch()
				}
				init, initErr := v.V41MHCInitialStreams(dx, rows, hidden)
				coeffs, splitErr := v.V41MHCSplit(dm, dscale, dbase, rows, iters, eps)
				// Raw mixes sized like an unscaled projection of the streams.
				raw := make([]float32, len(mixes))
				for i := range raw {
					raw[i] = mixes[i] * float32(4*hidden) * 0.01
				}
				params := append(append([]float32(nil), scale...), base...)
				wantRMS := v41MHCRefRMSSplit(raw, streams, params, rows, hidden, iters, eps, 1e-6)
				draw := upload(t, []int{rows, 24}, raw)
				dparams := upload(t, []int{27}, params)
				rmsCoeffs, rmsErr := v.V41MHCRMSSplit(draw, ds, dparams, rows, hidden, iters, eps, 1e-6)
				pre, preErr := v.V41MHCPre(ds, dk, rows, hidden)
				post, postErr := v.V41MHCPost(dx, ds, dk, rows, hidden)
				gotInit := read(t, init, initErr) // checked read fences the marked batch
				gotCoeffs := read(t, coeffs, splitErr)
				gotPre := read(t, pre, preErr)
				gotPost := read(t, post, postErr)
				gotRMS := read(t, rmsCoeffs, rmsErr)
				if batched {
					v.FlushBatch()
				}
				v41MHCRequireBits(t, "initial", gotInit, v41MHCRefInitial(x, rows, hidden))
				v41MHCRequireBits(t, "pre", gotPre, v41MHCRefPre(streams, wantCoeffs, rows, hidden))
				v41MHCRequireBits(t, "post", gotPost, v41MHCRefPost(x, streams, wantCoeffs, rows, hidden))
				maxULP, at := v41MHCRequireULP(t, "split", gotCoeffs, wantCoeffs, v41MHCSplitULPBound)
				t.Logf("split max ulp=%d at %d (got %g want %g; bound %d)", maxULP, at, gotCoeffs[at], wantCoeffs[at], v41MHCSplitULPBound)
				// The serial sum is bit-exact; the device reciprocal square root
				// adds about one ulp to every scaled mix before the same split.
				rmsULP, rmsAt := v41MHCRequireULP(t, "rms-split", gotRMS, wantRMS, v41MHCSplitULPBound)
				t.Logf("rms-split max ulp=%d at %d", rmsULP, rmsAt)
			})
		}
	}
	t.Run("rejects", func(t *testing.T) {
		dx := upload(t, []int{2, 8}, make([]float32, 16))
		ds := upload(t, []int{2, 4, 8}, make([]float32, 64))
		dk := upload(t, []int{2, 24}, make([]float32, 48))
		before, err := v.BackendExecutionSnapshot()
		if err != nil {
			t.Fatal(err)
		}
		host := NewF32(Default(), []int{2, 8}, make([]float32, 16))
		badShape := dx
		badShape.Shape = []int{16}
		for name, call := range map[string]func() (Tensor, error){
			"host operand":  func() (Tensor, error) { return v.V41MHCPost(host, ds, dk, 2, 8) },
			"flat shape":    func() (Tensor, error) { return v.V41MHCInitialStreams(badShape, 2, 8) },
			"short streams": func() (Tensor, error) { return v.V41MHCPre(dx, dk, 2, 8) },
			"zero iters":    func() (Tensor, error) { return v.V41MHCSplit(dk, dk, dk, 2, 0, 1e-6) },
		} {
			out, err := call()
			if err == nil || out.Buf() != nil {
				t.Fatalf("%s: malformed input did not fail before publishing an output", name)
			}
		}
		// Native extent and alias refusals run behind the Go validation.
		status := func(a, b, c, out *vulkanBuf, rows, hidden int) int {
			vulkanMu.Lock()
			defer vulkanMu.Unlock()
			return v41MHCStatusLocked(V41MHCOpPost, a, b, c, out, rows, hidden, 0, 0, 0)
		}
		xb, sb, kb := dx.buf.(*vulkanBuf), ds.buf.(*vulkanBuf), dk.buf.(*vulkanBuf)
		if got := status(xb, sb, kb, sb, 2, 8); got != 2 {
			t.Fatalf("output aliasing streams: status=%d want 2", got)
		}
		if got := status(xb, sb, kb, xb, 2, 8); got != 2 {
			t.Fatalf("short output allocation: status=%d want 2", got)
		}
		after, err := v.BackendExecutionSnapshot()
		if err != nil || before.Counters != after.Counters || before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes {
			t.Fatalf("rejected inputs dispatched, transferred, or allocated: before=%+v after=%+v err=%v", before, after, err)
		}
	})
}

// v41MHCSplitULPBound is the per-coefficient F32 ulp bound for the split. Each
// device exp and quotient is within about 1 ulp of the host's F64-narrowed
// value; Sinkhorn normalization of a near-degenerate softmax row (the fixture's
// 135-wide logit spread) amplifies those into its smallest entries. Measured
// worst on RX 7600 (2026-10-10): 7 ulps at comb[2] = 1.178e-4. The bound keeps
// 4x headroom for a different device's exp/reciprocal, never a loosened oracle.
const v41MHCSplitULPBound = 32

func v41MHCWitnessRow(n, salt int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*31+salt*17)%97-48) * 0.0371
	}
	if n > 3 {
		out[0] = math.Float32frombits(0x3f808000) // exact BF16 tie, even
		out[1] = math.Float32frombits(0x3f818000) // exact BF16 tie, odd
		out[2] = math.Float32frombits(0x80000000) // signed zero
	}
	return out
}

func v41MHCRequireULP(t *testing.T, name string, got, want []float32, bound int64) (int64, int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d values want %d", name, len(got), len(want))
	}
	var worst int64
	at := 0
	for i := range want {
		if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			t.Fatalf("%s[%d]: non-finite %g", name, i, got[i])
		}
		d := int64(math.Float32bits(got[i])) - int64(math.Float32bits(want[i]))
		if d < 0 {
			d = -d
		}
		if got[i] < 0 != (want[i] < 0) {
			d = math.MaxInt64
		}
		if d > worst {
			worst, at = d, i
		}
		if d > bound {
			t.Fatalf("%s[%d]: got %g want %g (%d ulps > %d)", name, i, got[i], want[i], d, bound)
		}
	}
	return worst, at
}
