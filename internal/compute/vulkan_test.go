//go:build vulkan && windows && cgo

package compute

import (
	"math"
	"strings"
	"testing"
)

// vulkan_test.go — op-level witness that the AMD/Vulkan backend's kernels reproduce the
// cpuref Reference within the Approx gate, one primitive at a time. This isolates a shader
// bug (e.g. a wrong attention softmax) to the exact op, instead of surfacing only as a
// forward-pass divergence in hal_vulkan_test.go. Compiled only under -tags vulkan; skips if
// no Vulkan device is registered.
//
// The gate per op is the Approx contract: high cosine + small max-abs-delta for the
// reductions (matmul/rmsnorm/attention), near-exact for the elementwise ops, and EXACT for
// argmax (the cpuref first-max tie-break is reproduced bit-for-bit by the shader).

func vk(t *testing.T) *vulkanBackend {
	b, ok := Lookup("vulkan")
	if !ok {
		t.Skip("vulkan backend not registered (no reachable Vulkan device)")
	}
	return b.(*vulkanBackend)
}

func maxAbs(a, b []float32) float64 {
	m := 0.0
	for i := range a {
		d := math.Abs(float64(a[i]) - float64(b[i]))
		if d > m {
			m = d
		}
	}
	return m
}

func TestVulkanResourceCapCheckNamesOffendingBuffer(t *testing.T) {
	v := &vulkanBackend{
		maxBufferBytes:          64,
		maxStorageBufferRange:   64,
		maxMemoryAllocationSize: 128,
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("checkResourceCap did not panic for an over-cap buffer")
		}
		got, ok := r.(string)
		if !ok {
			t.Fatalf("checkResourceCap panic type = %T, want string", r)
		}
		for _, want := range []string{
			"KV key cache layer 7",
			"65 bytes",
			"64 bytes",
			"maxStorageBufferRange=64",
			"maxMemoryAllocationSize=128",
			"split/chunk",
		} {
			if !strings.Contains(got, want) {
				t.Fatalf("cap error missing %q:\n%s", want, got)
			}
		}
	}()
	v.checkResourceCap(65, "KV key cache layer 7")
}

func TestVulkanResourceCapsAreDiscovered(t *testing.T) {
	v := vk(t)
	maxBufferBytes, maxStorageBufferRange, maxMemoryAllocationSize := v.VulkanDebugResourceCaps()
	total, free, known := DeviceMemoryInfo(v)
	if !known || total <= 0 {
		t.Fatalf("DeviceMemoryInfo = total=%d free=%d known=%v, want positive total/known", total, free, known)
	}
	if free != FreeUnknown && (free < 0 || free > total) {
		t.Fatalf("DeviceMemoryInfo free=%d outside [0,total=%d]", free, total)
	}
	if v.VulkanDebugMemoryBudgetAvailable() && free == FreeUnknown {
		t.Fatalf("Vulkan memory-budget extension is available but free memory is unknown: total=%d free=%d", total, free)
	}
	hostTotal, hostFree, hostKnown := HostMemoryInfo(v)
	if !hostKnown || hostTotal <= 0 {
		t.Fatalf("HostMemoryInfo = total=%d free=%d known=%v, want positive host total/known", hostTotal, hostFree, hostKnown)
	}
	if hostFree != FreeUnknown && (hostFree < 0 || hostFree > hostTotal) {
		t.Fatalf("HostMemoryInfo free=%d outside [0,total=%d]", hostFree, hostTotal)
	}
	if maxStorageBufferRange <= 0 {
		t.Fatalf("maxStorageBufferRange=%d, want positive", maxStorageBufferRange)
	}
	want := maxStorageBufferRange
	if maxMemoryAllocationSize > 0 && maxMemoryAllocationSize < want {
		want = maxMemoryAllocationSize
	}
	if maxBufferBytes != want {
		t.Fatalf("maxBufferBytes=%d, want effective cap %d (storage=%d allocation=%d)",
			maxBufferBytes, want, maxStorageBufferRange, maxMemoryAllocationSize)
	}
}

func TestVulkanAdvertisesDeviceCapacityWhenHeapTotalKnown(t *testing.T) {
	v := &vulkanBackend{totalMem: 24 << 30}
	if !v.Caps().CapacityProbe {
		t.Fatal("positive Vulkan device-local heap total must advertise CapacityProbe")
	}
	total, free, known := DeviceMemoryInfo(v)
	if !known || total != 24<<30 || free != FreeUnknown {
		t.Fatalf("DeviceMemoryInfo = total=%d free=%d known=%v, want 24GiB/free unknown/known", total, free, known)
	}
	v.totalMem = 0
	if v.Caps().CapacityProbe {
		t.Fatal("zero Vulkan heap total must not advertise CapacityProbe")
	}
	if _, _, known := DeviceMemoryInfo(v); known {
		t.Fatal("zero Vulkan heap total must fail open as unknown capacity")
	}
}

// upload host data to the device backend and read it straight back — the residency round-trip.
func TestVulkanResidencyRoundTrip(t *testing.T) {
	v := vk(t)
	var s lcg = 7
	x := randVec(&s, 1024)
	dt := v.Upload(NewF32(cpu(), []int{1024}, x), F32)
	got := v.Read(dt)
	v.Free(dt)
	for i := range x {
		if math.Float32bits(got[i]) != math.Float32bits(x[i]) {
			t.Fatalf("residency round-trip altered element %d: got %v want %v", i, got[i], x[i])
		}
	}
}

func TestVulkanEmbeddingRowCopiesSourceOffset(t *testing.T) {
	v := vk(t)
	c := cpu()
	rows, width := 5, 9
	table := make([]float32, rows*width)
	for i := range table {
		table[i] = float32(i*3 - 17)
	}
	dt := v.Upload(NewF32(c, []int{rows, width}, table), F32)
	for _, row := range []int{0, 2, rows - 1} {
		got := v.Read(v.EmbeddingRow(dt, row))
		want := table[row*width : (row+1)*width]
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("EmbeddingRow(row=%d)[%d]=%v want %v", row, i, got[i], want[i])
			}
		}
	}
	v.Recycle()
	v.Trim()
}

func TestVulkanMatMulApprox(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 11
	out, in := 64, 128
	w := randVec(&s, out*in)
	x := randVec(&s, in)
	ref := c.MatMul(NewF32(c, []int{out, in}, w), NewF32(c, []int{in}, x))
	dw := v.Upload(NewF32(c, []int{out, in}, w), F32)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)
	got := v.Read(v.MatMul(dw, dx))
	if cos := cosine(c.Read(ref), got); cos < 0.9999 {
		t.Fatalf("matmul cosine %.6f < 0.9999", cos)
	}
	if d := maxAbs(c.Read(ref), got); d > 1e-2 {
		t.Fatalf("matmul max|Δ| %.4g > 1e-2", d)
	}
}

func TestVulkanQ8MatMulApprox(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		t.Skip("vulkan device does not expose int8 arithmetic + 8-bit storage")
	}
	c := cpu()
	var s lcg = 13
	out, in, P := 37, 64, 3
	w := randVec(&s, out*in)
	x := randVec(&s, in)
	X := randVec(&s, P*in)
	wq := QuantizeQ8(c, []int{out, in}, w, 32)
	dwq := v.Upload(wq, Q8_0)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)
	dX := v.Upload(NewF32(c, []int{P, in}, X), F32)

	ref := c.Read(c.MatMul(wq, NewF32(c, []int{in}, x)))
	got := v.Read(v.MatMul(dwq, dx))
	if cos := cosine(ref, got); cos < 0.9999 {
		t.Fatalf("q8 matmul cosine %.6f < 0.9999", cos)
	}
	if d := maxAbs(ref, got); d > 1e-3 {
		t.Fatalf("q8 matmul max|Delta| %.4g > 1e-3", d)
	}

	refB := c.Read(c.BatchedMatMul(wq, NewF32(c, []int{P, in}, X), P))
	gotB := v.Read(v.BatchedMatMul(dwq, dX, P))
	if cos := cosine(refB, gotB); cos < 0.9999 {
		t.Fatalf("q8 batched matmul cosine %.6f < 0.9999", cos)
	}
	if d := maxAbs(refB, gotB); d > 1e-3 {
		t.Fatalf("q8 batched matmul max|Delta| %.4g > 1e-3", d)
	}
}

// TestVulkanQ8MatMulWideInput exercises the q8_matmul input-tiling path: input dims past the
// shader's per-window staging cap (SHARED_CAP=2048) must window over the input and still match
// the CPU Q8 reference bit-closely. in=3072 spans two windows (2048 + 1024); in=8960 is the
// real Qwen2.5-1.5B FFN down_proj dim (five windows) that motivated lifting the old in<=2048 cap.
func TestVulkanQ8MatMulWideInput(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		t.Skip("vulkan device does not expose int8 arithmetic + 8-bit storage")
	}
	c := cpu()
	var s lcg = 91
	for _, tc := range []struct {
		out int
		in  int
	}{
		{out: 33, in: 2080}, // just past one window
		{out: 64, in: 3072}, // two windows (2048 + 1024)
		{out: 48, in: 8960}, // the 1.5B FFN down_proj dim — five windows
	} {
		w := randVec(&s, tc.out*tc.in)
		x := randVec(&s, tc.in)
		wq := QuantizeQ8(c, []int{tc.out, tc.in}, w, 32)
		dwq := v.Upload(wq, Q8_0)
		dx := v.Upload(NewF32(c, []int{tc.in}, x), F32)
		ref := c.Read(c.MatMul(wq, NewF32(c, []int{tc.in}, x)))
		got := v.Read(v.MatMul(dwq, dx))
		if cos := cosine(ref, got); cos < 0.9999 {
			t.Fatalf("q8 wide matmul (out=%d,in=%d) cosine %.6f < 0.9999", tc.out, tc.in, cos)
		}
		if d := maxAbs(ref, got); d > 1e-3 {
			t.Fatalf("q8 wide matmul (out=%d,in=%d) max|Delta| %.4g > 1e-3", tc.out, tc.in, d)
		}
	}
}

// TestVulkanQ8MatMulVocabHead exercises the q8_matmul OUTPUT-tiling path at LM-head scale —
// the failure that motivated #471. The original shader launched one workgroup per activation
// row and walked the whole output dimension inside it; a real ~49k-vocab LM head made that one
// workgroup walk the entire vocabulary and tripped a device loss (VK_ERROR_DEVICE_LOST). The
// fix splits the output into 256-wide groups (dispatch = P·ceil(out/256)), so the dispatch must
// span many output groups and still match the CPU Q8 reference. out=49152,in=576 is the real
// SmolLM2-135M tied LM head (192 output groups) — small dims (out≤64) never cross a group and so
// never covered this path, which is why the bug shipped green and only surfaced on the real model.
func TestVulkanQ8MatMulVocabHead(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		t.Skip("vulkan device does not expose int8 arithmetic + 8-bit storage")
	}
	c := cpu()
	var s lcg = 4915
	const out, in = 49152, 576 // real SmolLM2-135M LM head: 192 output groups of 256
	w := randVec(&s, out*in)
	x := randVec(&s, in)
	wq := QuantizeQ8(c, []int{out, in}, w, 32)
	dwq := v.Upload(wq, Q8_0)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)

	ref := c.Read(c.MatMul(wq, NewF32(c, []int{in}, x)))
	got := v.Read(v.MatMul(dwq, dx))
	if cos := cosine(ref, got); cos < 0.9999 {
		t.Fatalf("q8 vocab-head matmul (out=%d,in=%d) cosine %.6f < 0.9999", out, in, cos)
	}
	if d := maxAbs(ref, got); d > 1e-3 {
		t.Fatalf("q8 vocab-head matmul (out=%d,in=%d) max|Delta| %.4g > 1e-3", out, in, d)
	}
}

func TestVulkanMatMulArgmaxMatchesVulkanMatMul(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 71
	for _, tc := range []struct {
		out int
		in  int
	}{
		{out: 1, in: 17},
		{out: 257, in: 64},
		{out: 513, in: 96},
	} {
		w := randVec(&s, tc.out*tc.in)
		x := randVec(&s, tc.in)
		dw := v.Upload(NewF32(c, []int{tc.out, tc.in}, w), F32)
		dx := v.Upload(NewF32(c, []int{tc.in}, x), F32)
		want := v.Argmax(v.MatMul(dw, dx))
		got := v.MatMulArgmax(dw, dx)
		if got != want {
			t.Fatalf("MatMulArgmax(out=%d,in=%d)=%d want Vulkan MatMul+Argmax %d", tc.out, tc.in, got, want)
		}
	}
}

func TestVulkanRMSNormMatMulArgmaxMatchesVulkanChain(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 73
	for _, tc := range []struct {
		out int
		in  int
	}{
		{out: 257, in: 64},
		{out: 513, in: 96},
	} {
		w := randVec(&s, tc.out*tc.in)
		x := randVec(&s, tc.in)
		norm := randVec(&s, tc.in)
		dw := v.Upload(NewF32(c, []int{tc.out, tc.in}, w), F32)
		dx := v.Upload(NewF32(c, []int{tc.in}, x), F32)
		dn := v.Upload(NewF32(c, []int{tc.in}, norm), F32)
		xn := v.RMSNorm(dx, dn, 1e-5)
		want := v.MatMulArgmax(dw, xn)
		got := v.RMSNormMatMulArgmax(dw, dx, dn, 1e-5)
		if got != want {
			t.Fatalf("RMSNormMatMulArgmax(out=%d,in=%d)=%d want Vulkan RMSNorm+MatMulArgmax %d",
				tc.out, tc.in, got, want)
		}
	}
}

func TestVulkanTransientRecycleReusesBuffer(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 72
	out, in := 64, 96
	w := randVec(&s, out*in)
	x := randVec(&s, in)
	dw := v.Upload(NewF32(c, []int{out, in}, w), F32)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)

	y0 := v.MatMul(dw, dx)
	b0 := y0.buf.(*vulkanBuf)
	p0 := b0.ptr
	if p0 == nil {
		t.Fatal("first transient buffer is nil")
	}
	v.Recycle()
	if b0.ptr != nil {
		t.Fatal("Recycle must invalidate stale transient tensor handles")
	}

	y1 := v.MatMul(dw, dx)
	b1 := y1.buf.(*vulkanBuf)
	if b1.ptr == nil {
		t.Fatal("second transient buffer is nil")
	}
	if b1.ptr != p0 {
		t.Fatalf("transient buffer was not reused: got %p want %p", b1.ptr, p0)
	}
	v.Recycle()
	v.Trim()
}

func TestVulkanHostVisibleBufferDoesNotRecycleAsDeviceLocal(t *testing.T) {
	v := vk(t)
	host := v.dallocHostVis(4096)
	if !v.debugBufferHostVisible(host) {
		t.Fatal("host-visible allocation did not report HOST_VISIBLE memory")
	}
	v.Free(makeTensor(v, F32, RowMajor, []int{1024}, nil, host))

	dev := v.dalloc(4096)
	if !v.debugBufferDeviceLocal(dev) {
		t.Fatal("device-local allocation reused a host-visible buffer from the recycle pool")
	}
	v.Free(makeTensor(v, F32, RowMajor, []int{1024}, nil, dev))
	v.Trim()
}

func TestVulkanTransientRecycleDropsHostVisibleBuffer(t *testing.T) {
	v := vk(t)
	host := v.dallocHostVis(4096)
	if !v.debugBufferHostVisible(host) {
		t.Fatal("host-visible allocation did not report HOST_VISIBLE memory")
	}
	v.transient = append(v.transient, host)
	v.Recycle()
	if host.ptr != nil {
		t.Fatal("Recycle must invalidate stale transient tensor handles")
	}

	dev := v.dallocTransient(4096)
	if !v.debugBufferDeviceLocal(dev) {
		t.Fatal("transient pool returned a host-visible buffer for a device-local transient")
	}
	v.Free(makeTensor(v, F32, RowMajor, []int{1024}, nil, dev))
	v.Trim()
}

func TestVulkanBatchedHostVisibleFreeDoesNotRecycleAsDeviceLocal(t *testing.T) {
	v := vk(t)
	host := v.dallocHostVis(4096)
	v.BeginBatch()
	v.Free(makeTensor(v, F32, RowMajor, []int{1024}, nil, host))
	v.FlushBatch()

	dev := v.dalloc(4096)
	if !v.debugBufferDeviceLocal(dev) {
		t.Fatal("batched host-visible free recycled into a later device-local allocation")
	}
	v.Free(makeTensor(v, F32, RowMajor, []int{1024}, nil, dev))
	v.Trim()
}

func TestVulkanBudgetedWeightFreeReleasesDeviceLocalBytes(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		t.Skip("vulkan device does not expose int8 arithmetic + 8-bit storage")
	}
	oldBudget, oldUsed, oldHostvis := v.budgetBytes, v.dlUsed, v.hostvisN
	defer func() {
		v.budgetBytes, v.dlUsed, v.hostvisN = oldBudget, oldUsed, oldHostvis
		v.Trim()
	}()
	v.budgetBytes, v.dlUsed, v.hostvisN = 64, 0, 0

	c := cpu()
	var s lcg = 365
	shape := []int{2, 32} // Q8 code buffer is 64 bytes.
	w := randVec(&s, shape[0]*shape[1])
	dw := v.Upload(NewF32(c, shape, w), Q8_0)
	db := dw.buf.(*vulkanBuf)
	if db.budgetedWeightBytes != 64 {
		t.Fatalf("budget charge=%d want 64", db.budgetedWeightBytes)
	}
	if v.dlUsed != 64 {
		t.Fatalf("dlUsed after first upload=%d want 64", v.dlUsed)
	}
	if v.hostvisN != 0 {
		t.Fatalf("first weight unexpectedly spilled host-visible; hostvisN=%d", v.hostvisN)
	}
	v.Free(dw)
	if v.dlUsed != 0 {
		t.Fatalf("dlUsed after Free=%d want 0", v.dlUsed)
	}
	if db.budgetedWeightBytes != 0 {
		t.Fatalf("freed buffer retained budget charge %d", db.budgetedWeightBytes)
	}

	dw2 := v.Upload(NewF32(c, shape, w), Q8_0)
	if v.hostvisN != 0 {
		t.Fatalf("second weight spilled host-visible after budget release; hostvisN=%d", v.hostvisN)
	}
	v.Free(dw2)
}

func TestVulkanBudgetAccountingUsesActualResidency(t *testing.T) {
	v := vk(t)
	oldBudget, oldUsed, oldHostvis := v.budgetBytes, v.dlUsed, v.hostvisN
	defer func() {
		v.budgetBytes, v.dlUsed, v.hostvisN = oldBudget, oldUsed, oldHostvis
		v.Trim()
	}()
	v.budgetBytes, v.dlUsed, v.hostvisN = 64, 0, 0

	host := v.dallocHostVis(64)
	v.accountWeightPlacement(host, 64)
	if v.dlUsed != 0 {
		t.Fatalf("host-visible weight charged dlUsed=%d, want 0", v.dlUsed)
	}
	if v.hostvisN != 1 {
		t.Fatalf("host-visible weight count=%d, want 1", v.hostvisN)
	}
	hb := host
	v.Free(makeTensor(v, F32, RowMajor, []int{16}, nil, host))
	if v.hostvisN != 0 {
		t.Fatalf("host-visible weight count after Free=%d, want 0", v.hostvisN)
	}
	if hb.hostVisibleWeight {
		t.Fatal("freed host-visible weight retained accounting flag")
	}
}

func TestVulkanMatMulAddInPlaceApprox(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 12
	out, in := 64, 128
	w := randVec(&s, out*in)
	x := randVec(&s, in)
	dst := randVec(&s, out)
	refDst := append([]float32(nil), dst...)
	refProj := c.Read(c.MatMul(NewF32(c, []int{out, in}, w), NewF32(c, []int{in}, x)))
	for i := range refDst {
		refDst[i] += refProj[i]
	}
	dw := v.Upload(NewF32(c, []int{out, in}, w), F32)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)
	ddst := v.Upload(NewF32(c, []int{out}, dst), F32)
	v.MatMulAddInPlace(ddst, dw, dx)
	got := v.Read(ddst)
	if cos := cosine(refDst, got); cos < 0.9999 {
		t.Fatalf("matmul_add cosine %.6f < 0.9999", cos)
	}
	if d := maxAbs(refDst, got); d > 1e-2 {
		t.Fatalf("matmul_add max|Δ| %.4g > 1e-2", d)
	}
}

func TestVulkanMatMul2Approx(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 13
	in, out0, out1 := 128, 96, 64
	w0 := randVec(&s, out0*in)
	w1 := randVec(&s, out1*in)
	x := randVec(&s, in)
	ref0 := c.Read(c.MatMul(NewF32(c, []int{out0, in}, w0), NewF32(c, []int{in}, x)))
	ref1 := c.Read(c.MatMul(NewF32(c, []int{out1, in}, w1), NewF32(c, []int{in}, x)))
	dw0 := v.Upload(NewF32(c, []int{out0, in}, w0), F32)
	dw1 := v.Upload(NewF32(c, []int{out1, in}, w1), F32)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)
	y0, y1 := v.MatMul2(dw0, dw1, dx)
	for name, pair := range map[string]struct{ ref, got []float32 }{
		"y0": {ref0, v.Read(y0)},
		"y1": {ref1, v.Read(y1)},
	} {
		if cos := cosine(pair.ref, pair.got); cos < 0.9999 {
			t.Fatalf("matmul2 %s cosine %.6f < 0.9999", name, cos)
		}
		if d := maxAbs(pair.ref, pair.got); d > 1e-2 {
			t.Fatalf("matmul2 %s max|Δ| %.4g > 1e-2", name, d)
		}
	}
}

func TestVulkanQ8MatMul2Approx(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		t.Skip("vulkan device does not expose int8 arithmetic + 8-bit storage")
	}
	c := cpu()
	var s lcg = 33
	in, out0, out1 := 128, 257, 129
	w0 := randVec(&s, out0*in)
	w1 := randVec(&s, out1*in)
	x := randVec(&s, in)
	wq0 := QuantizeQ8(c, []int{out0, in}, w0, 32)
	wq1 := QuantizeQ8(c, []int{out1, in}, w1, 32)
	ref0 := c.Read(c.MatMul(wq0, NewF32(c, []int{in}, x)))
	ref1 := c.Read(c.MatMul(wq1, NewF32(c, []int{in}, x)))
	dwq0 := v.Upload(wq0, Q8_0)
	dwq1 := v.Upload(wq1, Q8_0)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)
	y0, y1 := v.MatMul2(dwq0, dwq1, dx)
	for name, pair := range map[string]struct{ ref, got []float32 }{
		"y0": {ref0, v.Read(y0)},
		"y1": {ref1, v.Read(y1)},
	} {
		if cos := cosine(pair.ref, pair.got); cos < 0.9999 {
			t.Fatalf("q8 matmul2 %s cosine %.6f < 0.9999", name, cos)
		}
		if d := maxAbs(pair.ref, pair.got); d > 1e-3 {
			t.Fatalf("q8 matmul2 %s max|Delta| %.4g > 1e-3", name, d)
		}
	}
}

func TestVulkanMatMul3Approx(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 14
	in, qOut, kOut, vOut := 128, 64, 32, 32
	wq := randVec(&s, qOut*in)
	wk := randVec(&s, kOut*in)
	wv := randVec(&s, vOut*in)
	x := randVec(&s, in)
	refQ := c.Read(c.MatMul(NewF32(c, []int{qOut, in}, wq), NewF32(c, []int{in}, x)))
	refK := c.Read(c.MatMul(NewF32(c, []int{kOut, in}, wk), NewF32(c, []int{in}, x)))
	refV := c.Read(c.MatMul(NewF32(c, []int{vOut, in}, wv), NewF32(c, []int{in}, x)))
	dwq := v.Upload(NewF32(c, []int{qOut, in}, wq), F32)
	dwk := v.Upload(NewF32(c, []int{kOut, in}, wk), F32)
	dwv := v.Upload(NewF32(c, []int{vOut, in}, wv), F32)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)
	q, k, val := v.MatMul3(dwq, dwk, dwv, dx)
	for name, pair := range map[string]struct{ ref, got []float32 }{
		"q": {refQ, v.Read(q)},
		"k": {refK, v.Read(k)},
		"v": {refV, v.Read(val)},
	} {
		if cos := cosine(pair.ref, pair.got); cos < 0.9999 {
			t.Fatalf("matmul3 %s cosine %.6f < 0.9999", name, cos)
		}
		if d := maxAbs(pair.ref, pair.got); d > 1e-2 {
			t.Fatalf("matmul3 %s max|Δ| %.4g > 1e-2", name, d)
		}
	}
}

func TestVulkanQ8MatMul3Approx(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		t.Skip("vulkan device does not expose int8 arithmetic + 8-bit storage")
	}
	c := cpu()
	var s lcg = 34
	in, qOut, kOut, vOut := 128, 257, 129, 65
	wq := randVec(&s, qOut*in)
	wk := randVec(&s, kOut*in)
	wv := randVec(&s, vOut*in)
	x := randVec(&s, in)
	qw := QuantizeQ8(c, []int{qOut, in}, wq, 32)
	kw := QuantizeQ8(c, []int{kOut, in}, wk, 32)
	vw := QuantizeQ8(c, []int{vOut, in}, wv, 32)
	refQ := c.Read(c.MatMul(qw, NewF32(c, []int{in}, x)))
	refK := c.Read(c.MatMul(kw, NewF32(c, []int{in}, x)))
	refV := c.Read(c.MatMul(vw, NewF32(c, []int{in}, x)))
	dwq := v.Upload(qw, Q8_0)
	dwk := v.Upload(kw, Q8_0)
	dwv := v.Upload(vw, Q8_0)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)
	q, k, val := v.MatMul3(dwq, dwk, dwv, dx)
	for name, pair := range map[string]struct{ ref, got []float32 }{
		"q": {refQ, v.Read(q)},
		"k": {refK, v.Read(k)},
		"v": {refV, v.Read(val)},
	} {
		if cos := cosine(pair.ref, pair.got); cos < 0.9999 {
			t.Fatalf("q8 matmul3 %s cosine %.6f < 0.9999", name, cos)
		}
		if d := maxAbs(pair.ref, pair.got); d > 1e-3 {
			t.Fatalf("q8 matmul3 %s max|Delta| %.4g > 1e-3", name, d)
		}
	}
}

func TestVulkanRMSNormMatMul2Approx(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 15
	in, out0, out1 := 128, 96, 64
	w0 := randVec(&s, out0*in)
	w1 := randVec(&s, out1*in)
	x := randVec(&s, in)
	norm := randVec(&s, in)
	xn := c.RMSNorm(NewF32(c, []int{in}, x), NewF32(c, []int{in}, norm), 1e-5)
	ref0 := c.Read(c.MatMul(NewF32(c, []int{out0, in}, w0), xn))
	ref1 := c.Read(c.MatMul(NewF32(c, []int{out1, in}, w1), xn))
	dw0 := v.Upload(NewF32(c, []int{out0, in}, w0), F32)
	dw1 := v.Upload(NewF32(c, []int{out1, in}, w1), F32)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)
	dn := v.Upload(NewF32(c, []int{in}, norm), F32)
	y0, y1 := v.RMSNormMatMul2(dw0, dw1, dx, dn, 1e-5)
	for name, pair := range map[string]struct{ ref, got []float32 }{
		"y0": {ref0, v.Read(y0)},
		"y1": {ref1, v.Read(y1)},
	} {
		if cos := cosine(pair.ref, pair.got); cos < 0.9999 {
			t.Fatalf("rmsnorm_matmul2 %s cosine %.6f < 0.9999", name, cos)
		}
		if d := maxAbs(pair.ref, pair.got); d > 1e-2 {
			t.Fatalf("rmsnorm_matmul2 %s max|Δ| %.4g > 1e-2", name, d)
		}
	}
	if d := maxAbs(x, v.Read(dx)); d > 0 {
		t.Fatalf("rmsnorm_matmul2 mutated source max|Δ| %.4g", d)
	}
}

func TestVulkanRMSNormMatMul3Approx(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 16
	in, qOut, kOut, vOut := 128, 64, 32, 32
	wq := randVec(&s, qOut*in)
	wk := randVec(&s, kOut*in)
	wv := randVec(&s, vOut*in)
	x := randVec(&s, in)
	norm := randVec(&s, in)
	xn := c.RMSNorm(NewF32(c, []int{in}, x), NewF32(c, []int{in}, norm), 1e-5)
	refQ := c.Read(c.MatMul(NewF32(c, []int{qOut, in}, wq), xn))
	refK := c.Read(c.MatMul(NewF32(c, []int{kOut, in}, wk), xn))
	refV := c.Read(c.MatMul(NewF32(c, []int{vOut, in}, wv), xn))
	dwq := v.Upload(NewF32(c, []int{qOut, in}, wq), F32)
	dwk := v.Upload(NewF32(c, []int{kOut, in}, wk), F32)
	dwv := v.Upload(NewF32(c, []int{vOut, in}, wv), F32)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)
	dn := v.Upload(NewF32(c, []int{in}, norm), F32)
	q, k, val := v.RMSNormMatMul3(dwq, dwk, dwv, dx, dn, 1e-5)
	for name, pair := range map[string]struct{ ref, got []float32 }{
		"q": {refQ, v.Read(q)},
		"k": {refK, v.Read(k)},
		"v": {refV, v.Read(val)},
	} {
		if cos := cosine(pair.ref, pair.got); cos < 0.9999 {
			t.Fatalf("rmsnorm_matmul3 %s cosine %.6f < 0.9999", name, cos)
		}
		if d := maxAbs(pair.ref, pair.got); d > 1e-2 {
			t.Fatalf("rmsnorm_matmul3 %s max|Δ| %.4g > 1e-2", name, d)
		}
	}
	if d := maxAbs(x, v.Read(dx)); d > 0 {
		t.Fatalf("rmsnorm_matmul3 mutated source max|Δ| %.4g", d)
	}
}

// The three tests below witness the fused Q8 decode kernels — the dispatch-count lever for
// GPU parity with llama.cpp on the common case (a quantized model decoding batch-1). Each
// folds RMSNorm (or SwiGLU) into a Q8_0 dequant-GEMV in ONE dispatch, the Q8 analogue of the
// f32 fused tests above. They are W8A8 (the activation is dynamically quantized per 32-block
// from the NORMED activation), so the reference quantizes the weights and feeds the f32-normed
// activation through a plain Q8 MatMul — the same numerics, just unfused. The gate matches the
// existing fused-norm tests: cosine ≥ 0.9999 (direction preserved) + max|Δ| ≤ 1e-2 (the W8A8
// quant error stacked on the f32-reduction reorder exceeds the pure-q8 1e-3 bound). The wide
// in= sub-cases (3072 crosses the 2048-float staging window; 8960 is the real 1.5B FFN
// down_proj dim) exercise the input-windowing path the fused kernels inherit from q8_matmul.

func TestVulkanQ8RMSNormMatMul3Approx(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		t.Skip("vulkan device does not expose int8 arithmetic + 8-bit storage")
	}
	c := cpu()
	var s lcg = 116
	for _, in := range []int{128, 3072, 8960} {
		qOut, kOut, vOut := 64, 32, 32
		wq := randVec(&s, qOut*in)
		wk := randVec(&s, kOut*in)
		wv := randVec(&s, vOut*in)
		x := randVec(&s, in)
		norm := randVec(&s, in)
		// reference: quantize-after-norm — feed the f32 RMSNorm output through a plain Q8 MatMul.
		xn := c.RMSNorm(NewF32(c, []int{in}, x), NewF32(c, []int{in}, norm), 1e-5)
		qw := QuantizeQ8(c, []int{qOut, in}, wq, 32)
		kw := QuantizeQ8(c, []int{kOut, in}, wk, 32)
		vw := QuantizeQ8(c, []int{vOut, in}, wv, 32)
		refQ := c.Read(c.MatMul(qw, xn))
		refK := c.Read(c.MatMul(kw, xn))
		refV := c.Read(c.MatMul(vw, xn))
		dwq := v.Upload(qw, Q8_0)
		dwk := v.Upload(kw, Q8_0)
		dwv := v.Upload(vw, Q8_0)
		dx := v.Upload(NewF32(c, []int{in}, x), F32)
		dn := v.Upload(NewF32(c, []int{in}, norm), F32)
		q, k, val := v.RMSNormMatMul3(dwq, dwk, dwv, dx, dn, 1e-5)
		for name, pair := range map[string]struct{ ref, got []float32 }{
			"q": {refQ, v.Read(q)},
			"k": {refK, v.Read(k)},
			"v": {refV, v.Read(val)},
		} {
			if cos := cosine(pair.ref, pair.got); cos < 0.9999 {
				t.Fatalf("q8 rmsnorm_matmul3 %s (in=%d) cosine %.6f < 0.9999", name, in, cos)
			}
			if d := maxAbs(pair.ref, pair.got); d > 1e-2 {
				t.Fatalf("q8 rmsnorm_matmul3 %s (in=%d) max|Δ| %.4g > 1e-2", name, in, d)
			}
		}
		if d := maxAbs(x, v.Read(dx)); d > 0 {
			t.Fatalf("q8 rmsnorm_matmul3 (in=%d) mutated source max|Δ| %.4g", in, d)
		}
	}
}

func TestVulkanQ8RMSNormMatMul2Approx(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		t.Skip("vulkan device does not expose int8 arithmetic + 8-bit storage")
	}
	c := cpu()
	var s lcg = 117
	for _, in := range []int{128, 3072, 8960} {
		out0, out1 := 96, 64
		w0 := randVec(&s, out0*in)
		w1 := randVec(&s, out1*in)
		x := randVec(&s, in)
		norm := randVec(&s, in)
		xn := c.RMSNorm(NewF32(c, []int{in}, x), NewF32(c, []int{in}, norm), 1e-5)
		q0 := QuantizeQ8(c, []int{out0, in}, w0, 32)
		q1 := QuantizeQ8(c, []int{out1, in}, w1, 32)
		ref0 := c.Read(c.MatMul(q0, xn))
		ref1 := c.Read(c.MatMul(q1, xn))
		dw0 := v.Upload(q0, Q8_0)
		dw1 := v.Upload(q1, Q8_0)
		dx := v.Upload(NewF32(c, []int{in}, x), F32)
		dn := v.Upload(NewF32(c, []int{in}, norm), F32)
		y0, y1 := v.RMSNormMatMul2(dw0, dw1, dx, dn, 1e-5)
		for name, pair := range map[string]struct{ ref, got []float32 }{
			"y0": {ref0, v.Read(y0)},
			"y1": {ref1, v.Read(y1)},
		} {
			if cos := cosine(pair.ref, pair.got); cos < 0.9999 {
				t.Fatalf("q8 rmsnorm_matmul2 %s (in=%d) cosine %.6f < 0.9999", name, in, cos)
			}
			if d := maxAbs(pair.ref, pair.got); d > 1e-2 {
				t.Fatalf("q8 rmsnorm_matmul2 %s (in=%d) max|Δ| %.4g > 1e-2", name, in, d)
			}
		}
		if d := maxAbs(x, v.Read(dx)); d > 0 {
			t.Fatalf("q8 rmsnorm_matmul2 (in=%d) mutated source max|Δ| %.4g", in, d)
		}
	}
}

func TestVulkanQ8SwiGLUMatMulAddInPlaceApprox(t *testing.T) {
	v := vk(t)
	if !v.haveQ8 {
		t.Skip("vulkan device does not expose int8 arithmetic + 8-bit storage")
	}
	c := cpu()
	var s lcg = 118
	for _, in := range []int{128, 8960} {
		out := 64
		w := randVec(&s, out*in)
		g := randVec(&s, in)
		u := randVec(&s, in)
		dst := randVec(&s, out)
		// reference: quantize-after-SwiGLU — feed silu(g)*u through a plain Q8 MatMul, add to dst.
		sw := c.SwiGLU(NewF32(c, []int{in}, g), NewF32(c, []int{in}, u))
		qw := QuantizeQ8(c, []int{out, in}, w, 32)
		proj := c.Read(c.MatMul(qw, sw))
		ref := append([]float32(nil), dst...)
		for i := range ref {
			ref[i] += proj[i]
		}
		dw := v.Upload(qw, Q8_0)
		dg := v.Upload(NewF32(c, []int{in}, g), F32)
		du := v.Upload(NewF32(c, []int{in}, u), F32)
		ddst := v.Upload(NewF32(c, []int{out}, dst), F32)
		v.SwiGLUMatMulAddInPlace(ddst, dw, dg, du)
		got := v.Read(ddst)
		if cos := cosine(ref, got); cos < 0.9999 {
			t.Fatalf("q8 swiglu_matmul_add (in=%d) cosine %.6f < 0.9999", in, cos)
		}
		if d := maxAbs(ref, got); d > 1e-2 {
			t.Fatalf("q8 swiglu_matmul_add (in=%d) max|Δ| %.4g > 1e-2", in, d)
		}
	}
}

func TestVulkanRMSNormMatMulApprox(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 79
	out, in := 257, 96
	w := randVec(&s, out*in)
	x := randVec(&s, in)
	norm := randVec(&s, in)
	dw := v.Upload(NewF32(c, []int{out, in}, w), F32)
	dx := v.Upload(NewF32(c, []int{in}, x), F32)
	dn := v.Upload(NewF32(c, []int{in}, norm), F32)

	want := v.MatMul(dw, v.RMSNorm(dx, dn, 1e-5))
	got := v.RMSNormMatMul(dw, dx, dn, 1e-5)
	wh := v.Read(want)
	gh := v.Read(got)
	if cos := cosine(gh, wh); cos < 0.9999 {
		t.Fatalf("rmsnorm_matmul cosine %.6f < 0.9999", cos)
	}
	if d := maxAbs(gh, wh); d > 1e-2 {
		t.Fatalf("rmsnorm_matmul max|Delta| %.4g > 1e-2", d)
	}
	if d := maxAbs(v.Read(dx), x); d != 0 {
		t.Fatalf("rmsnorm_matmul mutated source max|Delta| %.4g", d)
	}
}

func TestVulkanRMSNormApprox(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 13
	n := 576
	x := randVec(&s, n)
	w := randVec(&s, n)
	ref := c.RMSNorm(NewF32(c, []int{n}, x), NewF32(c, []int{n}, w), 1e-5)
	got := v.Read(v.RMSNorm(v.Upload(NewF32(c, []int{n}, x), F32), v.Upload(NewF32(c, []int{n}, w), F32), 1e-5))
	if d := maxAbs(c.Read(ref), got); d > 1e-3 {
		t.Fatalf("rmsnorm max|Δ| %.4g > 1e-3", d)
	}
}

func TestVulkanRoPEApprox(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 17
	nHeads, hd := 9, 64
	x := randVec(&s, nHeads*hd)
	ref := c.RoPE(NewF32(c, []int{nHeads * hd}, x), 5, nHeads, hd, 10000)
	src := v.Upload(NewF32(c, []int{nHeads * hd}, x), F32)
	got := v.Read(v.RoPE(src, 5, nHeads, hd, 10000))
	if d := maxAbs(c.Read(ref), got); d > 1e-3 {
		t.Fatalf("rope max|Δ| %.4g > 1e-3", d)
	}
	if d := maxAbs(x, v.Read(src)); d > 0 {
		t.Fatalf("rope mutated source max|Δ| %.4g", d)
	}
}

func TestVulkanSwiGLUApprox(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 19
	n := 1536
	g := randVec(&s, n)
	u := randVec(&s, n)
	ref := c.SwiGLU(NewF32(c, []int{n}, g), NewF32(c, []int{n}, u))
	got := v.Read(v.SwiGLU(v.Upload(NewF32(c, []int{n}, g), F32), v.Upload(NewF32(c, []int{n}, u), F32)))
	if d := maxAbs(c.Read(ref), got); d > 1e-3 {
		t.Fatalf("swiglu max|Δ| %.4g > 1e-3", d)
	}
}

func TestVulkanSwiGLUMatMulAddInPlaceApprox(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 21
	out, in := 64, 128
	w := randVec(&s, out*in)
	g := randVec(&s, in)
	u := randVec(&s, in)
	dst := randVec(&s, out)
	sw := c.SwiGLU(NewF32(c, []int{in}, g), NewF32(c, []int{in}, u))
	proj := c.Read(c.MatMul(NewF32(c, []int{out, in}, w), sw))
	ref := append([]float32(nil), dst...)
	for i := range ref {
		ref[i] += proj[i]
	}
	dw := v.Upload(NewF32(c, []int{out, in}, w), F32)
	dg := v.Upload(NewF32(c, []int{in}, g), F32)
	du := v.Upload(NewF32(c, []int{in}, u), F32)
	ddst := v.Upload(NewF32(c, []int{out}, dst), F32)
	v.SwiGLUMatMulAddInPlace(ddst, dw, dg, du)
	got := v.Read(ddst)
	if cos := cosine(ref, got); cos < 0.9999 {
		t.Fatalf("swiglu_matmul_add cosine %.6f < 0.9999", cos)
	}
	if d := maxAbs(ref, got); d > 1e-2 {
		t.Fatalf("swiglu_matmul_add max|Δ| %.4g > 1e-2", d)
	}
}

func TestVulkanArgmaxExact(t *testing.T) {
	v := vk(t)
	c := cpu()
	var s lcg = 23
	for _, n := range []int{1, 7, 49152} {
		x := randVec(&s, n)
		ref := c.Argmax(NewF32(c, []int{n}, x))
		got := v.Argmax(v.Upload(NewF32(c, []int{n}, x), F32))
		if got != ref {
			t.Fatalf("argmax(n=%d): vulkan=%d cpuref=%d (must be exact)", n, got, ref)
		}
	}
}

// TestVulkanAttentionApprox drives the fused decode-attention op through a small KV store
// built the same way the forward loop builds it, vs the cpuref KV/Attention.
func TestVulkanAttentionApprox(t *testing.T) {
	v := vk(t)
	c := cpu()
	cfg := KVConfig{NumLayers: 1, NumKVHeads: 2, HeadDim: 16, RopeTheta: 10000}
	grp, nKV, hd := 3, cfg.NumKVHeads, cfg.HeadDim
	nH := grp * nKV
	w := nKV * hd
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	var s lcg = 29
	nPos := 5

	ckv := c.NewKV(cfg)
	vkv := v.NewKV(cfg)
	for p := 0; p < nPos; p++ {
		kRaw := randVec(&s, w)
		kRoPE := randVec(&s, w)
		val := randVec(&s, w)
		ckv.AppendKV(0, NewF32(c, []int{w}, kRaw), NewF32(c, []int{w}, kRoPE), NewF32(c, []int{w}, val), p)
		vkv.AppendKV(0, v.Upload(NewF32(c, []int{w}, kRaw), F32), v.Upload(NewF32(c, []int{w}, kRoPE), F32), v.Upload(NewF32(c, []int{w}, val), F32), p)
	}
	q := randVec(&s, nH*hd)
	ref := c.Read(c.Attention(NewF32(c, []int{nH * hd}, q), ckv, 0, true, grp, scale))
	got := v.Read(v.Attention(v.Upload(NewF32(c, []int{nH * hd}, q), F32), vkv, 0, true, grp, scale))
	if cos := cosine(ref, got); cos < 0.999 {
		t.Fatalf("attention cosine %.6f < 0.999", cos)
	}
	if d := maxAbs(ref, got); d > 1e-2 {
		t.Fatalf("attention max|Δ| %.4g > 1e-2", d)
	}
}
