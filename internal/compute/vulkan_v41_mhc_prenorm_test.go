//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

func requireV41MHCPrenorm(t *testing.T, v *vulkanBackend) {
	t.Helper()
	if v.SupportsV41MHC() && v.V41MHCPrenormAvailable() {
		return
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		t.Fatal("registered Vulkan build lacks the V4.1 mHC prenorm modes")
	}
	t.Skip("V4.1 mHC module not loaded")
}

func v41MHCPrenormWeight(dtype Dtype, hidden int) Tensor {
	width := V41MHCStreams * hidden
	if dtype == Q2_K {
		rng := rand.New(rand.NewSource(int64(13668 + hidden)))
		raw := make([]byte, V41MHCCoeffWidth*(width/q2kSuper)*q2kSuperBlock)
		_, _ = rng.Read(raw)
		for b := 0; b < len(raw); b += q2kSuperBlock {
			binaryPutFloat16(raw[b+80:b+82], 1.0/128)
			binaryPutFloat16(raw[b+82:b+84], 1.0/256)
		}
		return NewQ2K(Default(), []int{V41MHCCoeffWidth, width}, raw)
	}
	w := make([]float32, V41MHCCoeffWidth*width)
	for i := range w {
		w[i] = decodeSplitValue(i, 7919) * 0.01
	}
	return NewF32(Default(), []int{V41MHCCoeffWidth, width}, w)
}

// fak-test:runtime integration est=3s lane=optin
// Qualification requires FAK_VULKAN_REQUIRE_DEVICE=1; a skip is no physical
// witness. The prenorm path replaces the one-token MatMul + rms-split pair, so
// every coefficient must equal that pair's on the same device bit for bit: one
// row against MatMul, F32 panels against BatchedMatMul, Q2_K panels against
// one MatMul per row (the order the prenorm reproduces).
func TestVulkanV41MHCPrenormSplitBitIdentical(t *testing.T) {
	v := vk(t)
	requireV41MHCPrenorm(t, v)
	const iters, eps, rmsEps = 20, float32(1e-6), float32(1e-6)
	_, scale, base := v41MHCSplitFixture(1)
	params := append(append([]float32(nil), scale...), base...)
	for _, tc := range []struct {
		name         string
		dtype        Dtype
		rows, hidden int
		batched      bool
	}{
		{"f32_one_chunk", F32, 1, 7, false},
		{"f32_partial_chunk", F32, 1, 1000, false},
		{"f32_hidden4096", F32, 1, 4096, false},
		{"f32_v41_flash", F32, 1, 5120, false},
		{"f32_v41_flash_batched", F32, 1, 5120, true},
		{"f32_three_rows", F32, 3, 5120, false},
		{"q2k_one_superblock", Q2_K, 1, 64, false},
		{"q2k_v41_flash", Q2_K, 1, 5120, false},
		{"q2k_v41_flash_batched", Q2_K, 1, 5120, true},
		{"q2k_three_rows", Q2_K, 3, 1024, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if !v.SupportsV41MHCPrenormWeight(v41MHCPrenormWeight(tc.dtype, tc.hidden), tc.hidden) {
				if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
					t.Fatalf("%v prenorm weight is not served", tc.dtype)
				}
				t.Skipf("%v prenorm weight is not served", tc.dtype)
			}
			rows, hidden, width := tc.rows, tc.hidden, V41MHCStreams*tc.hidden
			weight := v.Upload(v41MHCPrenormWeight(tc.dtype, hidden), tc.dtype)
			defer v.Free(weight)
			streams := v41MHCWitnessRow(rows*width, hidden)
			for i := range streams {
				streams[i] = v41MHCRefBF16(streams[i])
			}
			upload := func(shape []int, x []float32) Tensor {
				d := v.UploadClass(NewF32(Default(), shape, x), F32, MemoryActivation, "V4.1 mHC prenorm witness")
				t.Cleanup(func() { v.Free(d) })
				return d
			}
			ds := upload([]int{rows, V41MHCStreams, hidden}, streams)
			dp := upload([]int{V41MHCParamWidth}, params)
			if tc.batched {
				v.BeginBatch()
				defer v.FlushBatch()
			}
			var raw Tensor
			switch {
			case rows == 1:
				x := ds
				x.Shape = []int{width}
				raw = v.MatMul(weight, x)
			case tc.dtype == F32:
				x := ds
				x.Shape = []int{rows, width}
				raw = v.BatchedMatMul(weight, x, rows)
			default:
				panel := make([]float32, 0, rows*V41MHCCoeffWidth)
				for r := 0; r < rows; r++ {
					x := upload([]int{width}, streams[r*width:(r+1)*width])
					y := v.MatMul(weight, x)
					panel = append(panel, v.Read(y)...)
					v.Free(y)
				}
				raw = upload([]int{rows, V41MHCCoeffWidth}, panel)
			}
			view := raw
			view.Shape = []int{rows, V41MHCCoeffWidth}
			wantT, err := v.V41MHCRMSSplit(view, ds, dp, rows, hidden, iters, eps, rmsEps)
			if err != nil {
				t.Fatal(err)
			}
			defer v.Free(wantT)
			before, err := v.BackendExecutionSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			gotT, err := v.V41MHCPrenormSplit(weight, ds, dp, rows, hidden, iters, eps, rmsEps)
			if err != nil {
				t.Fatal(err)
			}
			defer v.Free(gotT)
			after, err := v.BackendExecutionSnapshot()
			if err != nil {
				t.Fatal(err)
			}
			// The shim counts dispatches only under its process-start profile switch.
			if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1" {
				if d := after.Counters.ComputeDispatches - before.Counters.ComputeDispatches; d != 2 {
					t.Fatalf("prenorm-split recorded %d compute dispatches, want projection + split = 2", d)
				}
				if after.Counters.H2DCount != before.Counters.H2DCount || after.Counters.D2HCount != before.Counters.D2HCount {
					t.Fatalf("prenorm-split moved host data: before=%+v after=%+v", before.Counters, after.Counters)
				}
			}
			want := append([]float32(nil), v.Read(wantT)...)
			got := append([]float32(nil), v.Read(gotT)...)
			v41MHCRequireBits(t, tc.name, got, want)
			if rows == 1 || tc.dtype == F32 {
				v.Free(raw)
			}
		})
	}
}

// fak-test:runtime integration est=500ms lane=optin
// The coefficient witness alone can hide a raw-mix difference that the split
// rounds away, so the projection's scratch row is pinned too: raw mixes equal
// the one-token MatMul bit for bit and the sum of squares equals the host's
// serial F32 sum (rmsSplit's contract).
func TestVulkanV41MHCPrenormRawBitIdentical(t *testing.T) {
	v := vk(t)
	requireV41MHCPrenorm(t, v)
	_, scale, base := v41MHCSplitFixture(1)
	for _, dtype := range []Dtype{F32, Q2_K} {
		for _, hidden := range []int{64, 1000, 5120} {
			if dtype == Q2_K && (4*hidden)%q2kSuper != 0 {
				continue
			}
			width := V41MHCStreams * hidden
			hostWeight := v41MHCPrenormWeight(dtype, hidden)
			if !v.SupportsV41MHCPrenormWeight(hostWeight, hidden) {
				t.Fatalf("%v hidden=%d prenorm weight is not served", dtype, hidden)
			}
			weight := v.Upload(hostWeight, dtype)
			streams := v41MHCWitnessRow(width, hidden+1)
			for i := range streams {
				streams[i] = v41MHCRefBF16(streams[i])
			}
			ds := v.UploadClass(NewF32(Default(), []int{1, V41MHCStreams, hidden}, streams), F32, MemoryActivation, "prenorm raw streams")
			dp := v.UploadClass(NewF32(Default(), []int{V41MHCParamWidth}, append(append([]float32(nil), scale...), base...)), F32, MemoryActivation, "prenorm raw params")
			x := ds
			x.Shape = []int{width}
			rawT := v.MatMul(weight, x)
			want := append([]float32(nil), v.Read(rawT)...)
			v.Free(rawT)
			vulkanMu.Lock()
			scratch := v.dallocForClass(4*v41MHCPrenormStride, MemoryActivation, "prenorm raw scratch")
			out := v.dallocForClass(4*V41MHCCoeffWidth, MemoryActivation, "prenorm raw coeffs")
			status := v41MHCPrenormStatusLocked(dtype == Q2_K, weight.buf.(*vulkanBuf), ds.buf.(*vulkanBuf), dp.buf.(*vulkanBuf), scratch, out, 1, hidden, 20, 1e-6, 1e-6)
			vulkanMu.Unlock()
			scratchT := makeTensor(v, F32, RowMajor, []int{v41MHCPrenormStride}, nil, scratch)
			got := append([]float32(nil), v.Read(scratchT)...)
			v.Free(scratchT)
			v.Free(makeTensor(v, F32, RowMajor, []int{V41MHCCoeffWidth}, nil, out))
			v.Free(dp)
			v.Free(ds)
			v.Free(weight)
			if status != 0 {
				t.Fatalf("%v hidden=%d: native status %d", dtype, hidden, status)
			}
			var ss float32
			for _, s := range streams {
				sq := s * s
				ss += sq
			}
			v41MHCRequireBits(t, fmt.Sprintf("%v hidden=%d raw", dtype, hidden), got[:V41MHCCoeffWidth], want)
			v41MHCRequireBits(t, fmt.Sprintf("%v hidden=%d sumsq", dtype, hidden), got[V41MHCCoeffWidth:], []float32{ss})
		}
	}
}

// fak-test:runtime integration est=200ms lane=optin
func TestVulkanV41MHCPrenormSplitRejects(t *testing.T) {
	v := vk(t)
	requireV41MHCPrenorm(t, v)
	const hidden = 64
	weight := v.Upload(v41MHCPrenormWeight(F32, hidden), F32)
	defer v.Free(weight)
	ds := v.UploadClass(NewF32(Default(), []int{2, V41MHCStreams, hidden}, make([]float32, 2*4*hidden)), F32, MemoryActivation, "prenorm reject streams")
	defer v.Free(ds)
	dp := v.UploadClass(NewF32(Default(), []int{V41MHCParamWidth}, make([]float32, V41MHCParamWidth)), F32, MemoryActivation, "prenorm reject params")
	defer v.Free(dp)
	before, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	wide := weight
	wide.Shape = []int{V41MHCCoeffWidth, V41MHCStreams * 16641} // 66 F32 chunks
	unaligned := weight
	unaligned.Dtype, unaligned.Shape = Q2_K, []int{V41MHCCoeffWidth, V41MHCStreams * 65}
	flat := ds
	flat.Shape = []int{2 * V41MHCStreams * hidden}
	for name, call := range map[string]func() (Tensor, error){
		"weight width":    func() (Tensor, error) { return v.V41MHCPrenormSplit(weight, ds, dp, 2, hidden+1, 20, 1e-6, 1e-6) },
		"f32 chunk bound": func() (Tensor, error) { return v.V41MHCPrenormSplit(wide, ds, dp, 2, 16641, 20, 1e-6, 1e-6) },
		"q2k superblock":  func() (Tensor, error) { return v.V41MHCPrenormSplit(unaligned, ds, dp, 2, 65, 20, 1e-6, 1e-6) },
		"flat streams":    func() (Tensor, error) { return v.V41MHCPrenormSplit(weight, flat, dp, 2, hidden, 20, 1e-6, 1e-6) },
		"zero iters":      func() (Tensor, error) { return v.V41MHCPrenormSplit(weight, ds, dp, 2, hidden, 0, 1e-6, 1e-6) },
		"zero rms eps":    func() (Tensor, error) { return v.V41MHCPrenormSplit(weight, ds, dp, 2, hidden, 20, 1e-6, 0) },
	} {
		out, err := call()
		if !errors.Is(err, ErrV41MHCPrenormGeometry) || !errors.Is(err, ErrVulkanInvalidGeometry) || out.Buf() != nil {
			t.Fatalf("%s: err=%v out=%v, want the prenorm geometry refusal and no output", name, err, out.Buf())
		}
	}
	status := func(scratch, out *vulkanBuf) int {
		vulkanMu.Lock()
		defer vulkanMu.Unlock()
		return v41MHCPrenormStatusLocked(false, weight.buf.(*vulkanBuf), ds.buf.(*vulkanBuf), dp.buf.(*vulkanBuf), scratch, out, 2, hidden, 20, 1e-6, 1e-6)
	}
	sb, pb := ds.buf.(*vulkanBuf), dp.buf.(*vulkanBuf)
	if got := status(sb, pb); got != 2 {
		t.Fatalf("scratch aliasing streams: status=%d want 2", got)
	}
	if got := status(pb, sb); got != 2 {
		t.Fatalf("short scratch and output aliasing streams: status=%d want 2", got)
	}
	after, err := v.BackendExecutionSnapshot()
	if err != nil || before.Counters != after.Counters || before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes {
		t.Fatalf("rejected inputs dispatched, transferred, or allocated: before=%+v after=%+v err=%v", before, after, err)
	}
}

// fak-test:runtime integration est=5s lane=optin
// Log-only cost comparison, run with FAK_V41_MHC_PRENORM_TIMING=1: K decode
// rows recorded into one batch per arm, fenced by one read. It reports device
// time per row on this GPU, not a Strix Halo number.
func TestVulkanV41MHCPrenormSplitTiming(t *testing.T) {
	if os.Getenv("FAK_V41_MHC_PRENORM_TIMING") != "1" {
		t.Skip("set FAK_V41_MHC_PRENORM_TIMING=1")
	}
	v := vk(t)
	requireV41MHCPrenorm(t, v)
	const hidden, width, k = 5120, 4 * 5120, 64
	_, scale, base := v41MHCSplitFixture(1)
	dp := v.UploadClass(NewF32(Default(), []int{V41MHCParamWidth}, append(append([]float32(nil), scale...), base...)), F32, MemoryActivation, "prenorm timing params")
	defer v.Free(dp)
	streams := v41MHCWitnessRow(width, 3)
	ds := v.UploadClass(NewF32(Default(), []int{1, V41MHCStreams, hidden}, streams), F32, MemoryActivation, "prenorm timing streams")
	defer v.Free(ds)
	x := ds
	x.Shape = []int{width}
	for _, dtype := range []Dtype{F32, Q2_K} {
		weight := v.Upload(v41MHCPrenormWeight(dtype, hidden), dtype)
		if !v.SupportsV41MHCPrenormWeight(weight, hidden) {
			t.Fatalf("%v prenorm weight is not admitted on this device", dtype)
		}
		arm := func(fused bool) time.Duration {
			var outs []Tensor
			start := time.Now()
			v.BeginBatch()
			for i := 0; i < k; i++ {
				if fused {
					out, err := v.V41MHCPrenormSplit(weight, ds, dp, 1, hidden, 20, 1e-6, 1e-6)
					if err != nil {
						t.Fatal(err)
					}
					outs = append(outs, out)
					continue
				}
				raw := v.MatMul(weight, x)
				view := raw
				view.Shape = []int{1, V41MHCCoeffWidth}
				out, err := v.V41MHCRMSSplit(view, ds, dp, 1, hidden, 20, 1e-6, 1e-6)
				if err != nil {
					t.Fatal(err)
				}
				v.Free(raw)
				outs = append(outs, out)
			}
			_ = v.Read(outs[len(outs)-1])
			v.FlushBatch()
			elapsed := time.Since(start)
			for _, out := range outs {
				v.Free(out)
			}
			return elapsed / k
		}
		arm(false)
		arm(true)
		old, fused := arm(false), arm(true)
		t.Logf("%v hidden=%d: matmul+rms-split %v/row, prenorm-split %v/row (%.2fx)", dtype, hidden, old, fused, float64(old)/float64(fused))
		v.Free(weight)
	}
}

// fak-test:runtime integration est=500ms lane=optin
// Admission is the device's own verdict: the prenorm runs only after the probe
// in the weight's dtype matched MatMul + rms-split bit for bit, raw mixes
// included. A device that differs keeps the MatMul path; that is the Strix Halo
// gate before any default-on claim there.
func TestVulkanV41MHCPrenormSelfCheckAdmitsOnlyBitIdentical(t *testing.T) {
	v := vk(t)
	requireV41MHCPrenorm(t, v)
	for _, dtype := range []Dtype{F32, Q2_K} {
		probe := newV41MHCPrenormProbe(dtype)
		weight := Tensor{Dtype: dtype, Layout: RowMajor, Shape: []int{V41MHCCoeffWidth, V41MHCStreams * probe.hidden}}
		admitted := v.SupportsV41MHCPrenormWeight(weight, probe.hidden)
		vulkanMu.Lock()
		verdict := vulkanV41MHCPrenormVerdict[v41MHCPrenormQ2K(weight)]
		_, image, err := v.v41MHCPrenormSelfCheckImageLocked(dtype == Q2_K)
		vulkanMu.Unlock()
		if err != nil {
			t.Fatalf("%v self-check: %v", dtype, err)
		}
		equal := v41MHCPrenormSelfCheckEqual(probe, image)
		want := vulkanV41MHCPrenormDiffered
		if equal {
			want = vulkanV41MHCPrenormEqual
		}
		if admitted != equal || verdict != want {
			t.Fatalf("%v: admitted=%t verdict=%d, self-check equal=%t", dtype, admitted, verdict, equal)
		}
		nonzero := 0
		for i := 0; i < len(image); i += 4 {
			for j := 0; j+4 <= len(image[i]); j += 4 {
				if f := math.Float32frombits(uint32(image[i][j]) | uint32(image[i][j+1])<<8 | uint32(image[i][j+2])<<16 | uint32(image[i][j+3])<<24); f != 0 && !math.IsNaN(float64(f)) {
					nonzero++
				}
			}
		}
		if nonzero < V41MHCCoeffWidth {
			t.Fatalf("%v self-check reference raw mixes are vacuous: %d nonzero", dtype, nonzero)
		}
		if !equal {
			t.Errorf("%v prenorm is not bit-identical to MatMul + rms-split on this device; it stays off", dtype)
		}
		t.Logf("%v probe hidden=%d rows=%v: self-check equal=%t verdict=%d admitted=%t", dtype, probe.hidden, probe.rows, equal, verdict, admitted)
	}
}

// fak-test:runtime integration est=200ms lane=optin
// A differing verdict closes both the capability and the operation: nothing
// is allocated, dispatched or recorded.
func TestVulkanV41MHCPrenormDifferedVerdictRefuses(t *testing.T) {
	v := vk(t)
	requireV41MHCPrenorm(t, v)
	const hidden = 64
	weight := v.Upload(v41MHCPrenormWeight(F32, hidden), F32)
	defer v.Free(weight)
	ds := v.UploadClass(NewF32(Default(), []int{1, V41MHCStreams, hidden}, make([]float32, V41MHCStreams*hidden)), F32, MemoryActivation, "prenorm verdict streams")
	defer v.Free(ds)
	dp := v.UploadClass(NewF32(Default(), []int{V41MHCParamWidth}, make([]float32, V41MHCParamWidth)), F32, MemoryActivation, "prenorm verdict params")
	defer v.Free(dp)
	vulkanMu.Lock()
	saved := vulkanV41MHCPrenormVerdict
	vulkanV41MHCPrenormVerdict[0] = vulkanV41MHCPrenormDiffered
	vulkanMu.Unlock()
	t.Cleanup(func() {
		vulkanMu.Lock()
		vulkanV41MHCPrenormVerdict = saved
		vulkanMu.Unlock()
	})
	var seen []computetrace.Event
	computetrace.SetObserver(func(e computetrace.Event) { seen = append(seen, e) })
	t.Cleanup(func() { computetrace.SetObserver(nil) })
	before, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if v.SupportsV41MHCPrenormWeight(weight, hidden) {
		t.Fatal("a differing F32 verdict still admits the weight")
	}
	out, err := v.V41MHCPrenormSplit(weight, ds, dp, 1, hidden, 20, 1e-6, 1e-6)
	if !errors.Is(err, ErrVulkanExecutionFailed) || out.Buf() != nil {
		t.Fatalf("differing verdict: err=%v out=%v, want the closed capability refusal and no output", err, out.Buf())
	}
	after, err := v.BackendExecutionSnapshot()
	if err != nil || before.Counters != after.Counters || before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes || len(seen) != 0 {
		t.Fatalf("refused prenorm dispatched, allocated or recorded: before=%+v after=%+v events=%d err=%v", before, after, len(seen), err)
	}
}

// fak-test:runtime integration est=300ms lane=optin
// Each recorded prenorm call emits exactly one kernel record carrying the
// weight it projected; a call that fails emits none.
func TestVulkanV41MHCPrenormRecordsKernelEvent(t *testing.T) {
	v := vk(t)
	requireV41MHCPrenorm(t, v)
	const hidden = 1000
	weight := v.Upload(v41MHCPrenormWeight(F32, hidden), F32)
	defer v.Free(weight)
	if !v.SupportsV41MHCPrenormWeight(weight, hidden) {
		t.Fatal("F32 prenorm weight is not admitted on this device")
	}
	ds := v.UploadClass(NewF32(Default(), []int{2, V41MHCStreams, hidden}, v41MHCWitnessRow(2*V41MHCStreams*hidden, hidden)), F32, MemoryActivation, "prenorm trace streams")
	defer v.Free(ds)
	_, scale, base := v41MHCSplitFixture(1)
	dp := v.UploadClass(NewF32(Default(), []int{V41MHCParamWidth}, append(append([]float32(nil), scale...), base...)), F32, MemoryActivation, "prenorm trace params")
	defer v.Free(dp)
	var seen []computetrace.Event
	computetrace.SetObserver(func(e computetrace.Event) { seen = append(seen, e) })
	t.Cleanup(func() { computetrace.SetObserver(nil) })
	for _, rows := range []int{1, 2} {
		seen = seen[:0]
		streams := ds
		streams.Shape = []int{rows, V41MHCStreams, hidden}
		out, err := v.V41MHCPrenormSplit(weight, streams, dp, rows, hidden, 20, 1e-6, 1e-6)
		if err != nil {
			t.Fatal(err)
		}
		v.Free(out)
		input := []int{V41MHCStreams * hidden}
		if rows != 1 {
			input = []int{rows, V41MHCStreams * hidden}
		}
		if len(seen) != 1 || seen[0].Kernel != "f32_v41_mhc_prenorm" || seen[0].Operation != "matmul" || seen[0].Backend != v.Name() ||
			!reflect.DeepEqual(seen[0].Shapes, [][]int{weight.Shape, input}) {
			t.Fatalf("rows=%d recorded %+v, want one f32_v41_mhc_prenorm matmul record of %v x %v", rows, seen, weight.Shape, input)
		}
	}
	seen = seen[:0]
	if _, err := v.V41MHCPrenormSplit(weight, ds, dp, 2, hidden, 0, 1e-6, 1e-6); !errors.Is(err, ErrV41MHCPrenormGeometry) || len(seen) != 0 {
		t.Fatalf("refused call: err=%v records=%d, want the geometry refusal and no record", err, len(seen))
	}
}
