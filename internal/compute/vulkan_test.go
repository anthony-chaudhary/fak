//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"math/rand"
	"os"
	"slices"
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
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Vulkan device is not registered")
		}
		t.Skip("vulkan backend not registered (no reachable Vulkan device)")
	}
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
		t.Logf("Vulkan device: %s", b.Tier())
		if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(b.Tier()), strings.ToLower(expected)) {
			t.Fatalf("Vulkan device %q does not match required device %q", b.Tier(), expected)
		}
	}
	return b.(*vulkanBackend)
}

func TestVulkanDispatchProfileDisabledIsZero(t *testing.T) {
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1" {
		t.Skip("profiling enabled for this process")
	}
	v := vk(t)
	v.VulkanDebugResetDispatchProfile()
	c := cpu()
	d := v.Upload(NewF32(c, []int{4}, []float32{1, 2, 3, 4}), F32)
	_ = v.Read(v.RMSNorm(d, d, 1e-5))
	if got := v.VulkanDebugDispatchProfileSnapshot(); got != (VulkanDispatchProfile{}) {
		t.Fatalf("disabled profile = %+v, want zero", got)
	}
}

func TestVulkanQ4KStageGrowthKeepsBatchActiveAndBounded(t *testing.T) {
	v, ok := Pick("vulkan").(*vulkanBackend)
	if !ok {
		t.Skip("Vulkan backend unavailable")
	}
	v.VulkanDebugResetQ4KStage()
	defer v.VulkanDebugResetQ4KStage()
	v.VulkanDebugResetQ4KProfile()
	v.BeginBatch()
	v.VulkanDebugResetDispatchProfile()
	if !v.VulkanDebugBatchActive() {
		v.FlushBatch()
		t.Fatal("batch inactive at BeginBatch")
	}
	const out, in = 12, 768
	rng := rand.New(rand.NewSource(9811))
	raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	for b := 0; b < out*(in/q4kSuper); b++ {
		randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}
	hw := NewQ4K(Default(), []int{out, in}, raw)
	dw := v.Upload(hw, Q4_K)
	defer v.Free(dw)
	// The target failure occurs for streamed host-visible Q4_K weights. Mark this
	// fixture equivalently without changing the allocation or residency budget.
	dw.buf.(*vulkanBuf).hostVisibleWeight = true
	defer func() { dw.buf.(*vulkanBuf).hostVisibleWeight = false }()
	oldStage := v.q4kStage
	oldBudget := v.budgetBytes
	v.budgetBytes = 0 // keep this regression focused on the single-stage batch path
	v.q4kStage = true
	defer func() { v.q4kStage = oldStage; v.budgetBytes = oldBudget }()
	dx := v.Upload(NewF32(Default(), []int{in}, x), F32)
	defer v.Free(dx)
	before := v.VulkanDebugDispatchProfileSnapshot()
	if before.OneShotComputeSubmits != 0 {
		v.FlushBatch()
		t.Fatalf("unexpected precompute one-shot submits=%d", before.OneShotComputeSubmits)
	}
	got := v.MatMul(dw, dx)
	defer v.Free(got)
	if !v.VulkanDebugBatchActive() {
		v.FlushBatch()
		t.Fatal("batch not restored after Q4_K staging growth")
	}
	mid := v.VulkanDebugDispatchProfileSnapshot()
	if mid.OneShotComputeSubmits != 0 {
		v.FlushBatch()
		t.Fatalf("staged Q4_K matmul used one-shot path: %d", mid.OneShotComputeSubmits)
	}
	want := Default().Read(Default().MatMul(hw, NewF32(Default(), []int{in}, x)))
	if c := cosineC(v.Read(got), want); c < 0.995 {
		v.FlushBatch()
		t.Fatalf("staged Q4_K cosine %.8f < 0.995", c)
	}
	v.FlushBatch()
	final := v.VulkanDebugDispatchProfileSnapshot()
	if final.OneShotComputeSubmits != 0 {
		t.Fatalf("flush path regressed to one-shot compute submits=%d", final.OneShotComputeSubmits)
	}
	if final.BatchSubmits == 0 {
		t.Fatalf("flush path did not produce a bounded batch submit: %+v", final)
	}
}
func TestVulkanDispatchProfileCountsClassificationAndReset(t *testing.T) {
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("set FAK_VULKAN_DISPATCH_PROFILE=1")
	}
	v := vk(t)
	c := cpu()
	w := v.Upload(NewF32(c, []int{4, 4}, []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}), F32)
	x := v.Upload(NewF32(c, []int{4}, []float32{1, 2, 3, 4}), F32)

	// Prepare attention and GDN operands before reset so only the operations under
	// classification contribute to the exact family counts below.
	vkv := v.NewKV(KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 4, RopeTheta: 10000})
	vkv.AppendKV(0, x, x, x, 0)
	upload := func(shape []int, data []float32, what string) Tensor {
		return v.UploadClass(NewF32(c, shape, data), F32, MemoryActivation, what)
	}
	mixed := upload([]int{1, 3}, []float32{0.1, -0.2, 0.3}, "profile GDN mixed")
	scalar := upload([]int{1}, []float32{0.1}, "profile GDN scalar")
	conv := upload([]int{3, 2}, []float32{0, 0, 0, 0, 0, 0}, "profile GDN conv")
	convState := upload([]int{1, 3}, []float32{0, 0, 0}, "profile GDN conv state")
	recurrentState := upload([]int{1, 1, 1}, []float32{0}, "profile GDN recurrent state")

	v.VulkanDebugResetDispatchProfile()
	y := v.MatMul(w, x)
	_ = v.RMSNorm(x, x, 1e-5)
	_ = v.RoPEInPlace(x, 0, 1, 4, 10000)
	_ = v.SwiGLU(x, x)
	v.AddInPlace(x, x)
	_ = v.Attention(x, vkv, 0, true, 1, 0.5)
	_ = v.Argmax(x)
	if _, err := v.Qwen35GDNPreprojected(
		mixed, scalar, scalar, scalar, conv, scalar, scalar, scalar, convState, recurrentState,
		1, 1, 1, 1, 1, 2, 1e-5,
	); err != nil {
		t.Fatal(err)
	}
	v.BeginBatch()
	_ = v.MatMul(w, x)
	v.FlushBatch()
	_ = upload([]int{1}, []float32{1}, "profile post-reset upload")
	_ = v.Read(y)
	if _, err := v.CloneTensor(x); err != nil {
		t.Fatal(err)
	}

	got := v.VulkanDebugDispatchProfileSnapshot()
	if got.ComputeDispatches != 10 || got.OtherComputeDispatches != 10 || got.Q4KMatmulDispatches != 0 {
		t.Fatalf("dispatch classification = %+v", got)
	}
	wantFamilies := VulkanDispatchProfile{
		OtherMatmulDispatches:    2,
		OtherNormDispatches:      1,
		OtherRoPEDispatches:      1,
		OtherSwiGLUDispatches:    1,
		OtherAddDispatches:       1,
		OtherAttentionDispatches: 1,
		OtherArgmaxDispatches:    1,
		OtherGDNDispatches:       2,
	}
	if got.OtherMatmulDispatches != wantFamilies.OtherMatmulDispatches ||
		got.OtherNormDispatches != wantFamilies.OtherNormDispatches ||
		got.OtherRoPEDispatches != wantFamilies.OtherRoPEDispatches ||
		got.OtherSwiGLUDispatches != wantFamilies.OtherSwiGLUDispatches ||
		got.OtherAddDispatches != wantFamilies.OtherAddDispatches ||
		got.OtherAttentionDispatches != wantFamilies.OtherAttentionDispatches ||
		got.OtherArgmaxDispatches != wantFamilies.OtherArgmaxDispatches ||
		got.OtherGDNDispatches != wantFamilies.OtherGDNDispatches ||
		got.OtherUnclassifiedDispatches != 0 {
		t.Fatalf("operation family classification = %+v", got)
	}
	otherFamilies := got.OtherMatmulDispatches + got.OtherNormDispatches + got.OtherRoPEDispatches +
		got.OtherSwiGLUDispatches + got.OtherAddDispatches + got.OtherAttentionDispatches +
		got.OtherArgmaxDispatches + got.OtherGDNDispatches + got.OtherUnclassifiedDispatches
	if otherFamilies != got.OtherComputeDispatches {
		t.Fatalf("operation family total = %d, other compute dispatches = %d", otherFamilies, got.OtherComputeDispatches)
	}
	if got.OneShotComputeSubmits != 9 || got.OneShotH2DSubmits != 1 ||
		got.OneShotD2HSubmits != 2 || got.OneShotD2DSubmits != 1 {
		t.Fatalf("one-shot callsite classification = %+v", got)
	}
	oneShotFamilies := got.OneShotComputeSubmits + got.OneShotH2DSubmits + got.OneShotD2HSubmits + got.OneShotD2DSubmits
	if oneShotFamilies != got.OneShotSubmits {
		t.Fatalf("one-shot family total = %d, one-shot submits = %d", oneShotFamilies, got.OneShotSubmits)
	}
	if got.D2DCopies != 1 || got.BatchSubmits != 1 || got.BatchFlushes != 2 {
		t.Fatalf("aggregate operation counters = %+v", got)
	}
	v.VulkanDebugResetDispatchProfile()
	if zero := v.VulkanDebugDispatchProfileSnapshot(); zero != (VulkanDispatchProfile{}) {
		t.Fatalf("after reset = %+v", zero)
	}
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

func TestVulkanQ4KProfileClassification(t *testing.T) {
	tests := []struct {
		name              string
		hostVisibleWeight bool
		deviceLocal       bool
		wantHostVisible   bool
	}{
		{name: "device-local", deviceLocal: true},
		{name: "explicit host-visible", hostVisibleWeight: true, deviceLocal: true, wantHostVisible: true},
		{name: "non-device-local fallback", wantHostVisible: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := vulkanQ4KProfileHostVisible(tt.hostVisibleWeight, tt.deviceLocal); got != tt.wantHostVisible {
				t.Fatalf("vulkanQ4KProfileHostVisible(%v, %v) = %v, want %v",
					tt.hostVisibleWeight, tt.deviceLocal, got, tt.wantHostVisible)
			}
		})
	}
}

func TestVulkanQ4KProfileCountsAndReset(t *testing.T) {
	disabled := &vulkanBackend{}
	disabled.profileQ4KMatMulLocked(144, false, true)
	_, deviceCalls, deviceBytes, hostCalls, hostBytes := disabled.VulkanDebugQ4KProfileSnapshot()
	if deviceCalls != 0 || deviceBytes != 0 || hostCalls != 0 || hostBytes != 0 {
		t.Fatalf("disabled counters = device calls=%d bytes=%d host-visible calls=%d bytes=%d, want all zero",
			deviceCalls, deviceBytes, hostCalls, hostBytes)
	}

	v := &vulkanBackend{q4kProfile: true}
	v.profileQ4KMatMulLocked(144, false, true)
	v.profileQ4KMatMulLocked(288, true, false)
	v.profileQ4KMatMulLocked(432, false, false)

	enabled, deviceCalls, deviceBytes, hostCalls, hostBytes := v.VulkanDebugQ4KProfileSnapshot()
	if !enabled {
		t.Fatal("Q4_K profile snapshot reported disabled")
	}
	if deviceCalls != 1 || deviceBytes != 144 {
		t.Fatalf("device counters = calls=%d bytes=%d, want calls=1 bytes=144", deviceCalls, deviceBytes)
	}
	if hostCalls != 2 || hostBytes != 720 {
		t.Fatalf("host-visible counters = calls=%d bytes=%d, want calls=2 bytes=720", hostCalls, hostBytes)
	}

	v.VulkanDebugResetQ4KProfile()
	enabled, deviceCalls, deviceBytes, hostCalls, hostBytes = v.VulkanDebugQ4KProfileSnapshot()
	if !enabled {
		t.Fatal("Q4_K profile reset changed enabled state")
	}
	if deviceCalls != 0 || deviceBytes != 0 || hostCalls != 0 || hostBytes != 0 {
		t.Fatalf("counters after reset = device calls=%d bytes=%d host-visible calls=%d bytes=%d, want all zero",
			deviceCalls, deviceBytes, hostCalls, hostBytes)
	}
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

func TestVulkanCloneTensorCopiesKVStateIndependently(t *testing.T) {
	v := vk(t)
	c := cpu()
	shape := []int{2, 3, 4}
	values := make([]float32, 24)
	for i := range values {
		values[i] = float32(i*i-7*i+3) / 11
	}

	for _, tc := range []struct {
		name       string
		freeSource bool
	}{
		{name: "source_first", freeSource: true},
		{name: "clone_first", freeSource: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := v.UploadClass(
				NewF32(c, shape, append([]float32(nil), values...)),
				F32,
				MemoryKVCache,
				"qwen38 recurrent state clone test",
			)
			sourceBuf := source.buf.(*vulkanBuf)
			if sourceBuf.class != MemoryKVCache {
				t.Fatalf("source memory class = %q, want %q", sourceBuf.class, MemoryKVCache)
			}

			// Clone while a batch is open: CloneTensor must submit and fence the D2D copy
			// before returning, rather than leaving source lifetime coupled to a later flush.
			v.BeginBatch()
			defer v.FlushBatch()
			clone, err := TensorCloner(v).CloneTensor(source)
			if err != nil {
				v.Free(source)
				t.Fatalf("CloneTensor: %v", err)
			}
			cloneBuf := clone.buf.(*vulkanBuf)
			if cloneBuf == sourceBuf || cloneBuf.ptr == sourceBuf.ptr {
				v.Free(clone)
				v.Free(source)
				t.Fatal("CloneTensor reused the source allocation")
			}
			if cloneBuf.class != MemoryKVCache {
				v.Free(clone)
				v.Free(source)
				t.Fatalf("clone memory class = %q, want %q", cloneBuf.class, MemoryKVCache)
			}
			if clone.Dtype != source.Dtype || clone.Layout != source.Layout || !slices.Equal(clone.Shape, source.Shape) {
				v.Free(clone)
				v.Free(source)
				t.Fatalf("clone metadata = dtype=%s layout=%d shape=%v, want dtype=%s layout=%d shape=%v",
					clone.Dtype, clone.Layout, clone.Shape, source.Dtype, source.Layout, source.Shape)
			}
			clone.Shape[0]++
			if slices.Equal(clone.Shape, source.Shape) {
				v.Free(clone)
				v.Free(source)
				t.Fatal("clone shape aliases source metadata")
			}
			clone.Shape[0]--
			if got := v.Read(clone); !slices.Equal(got, values) {
				v.Free(clone)
				v.Free(source)
				t.Fatalf("clone values = %v, want %v", got, values)
			}
			deltaValues := make([]float32, len(values))
			mutatedValues := make([]float32, len(values))
			for i := range values {
				deltaValues[i] = 1
				mutatedValues[i] = values[i] + deltaValues[i]
			}
			delta := v.UploadClass(
				NewF32(c, shape, deltaValues),
				F32,
				MemoryActivation,
				"qwen38 clone independence delta",
			)
			v.AddInPlace(source, delta)
			v.Free(delta)
			if got := v.Read(clone); !slices.Equal(got, values) {
				v.Free(clone)
				v.Free(source)
				t.Fatalf("clone changed after source mutation: got %v, want %v", got, values)
			}

			if tc.freeSource {
				v.Free(source)
				if sourceBuf.ptr != nil {
					v.Free(clone)
					t.Fatal("Free(source) did not invalidate the source handle")
				}
				if cloneBuf.ptr == nil {
					t.Fatal("Free(source) invalidated the clone handle")
				}
				if got := v.Read(clone); !slices.Equal(got, values) {
					v.Free(clone)
					t.Fatalf("clone after Free(source) = %v, want %v", got, values)
				}
				v.Free(clone)
				return
			}

			v.Free(clone)
			if cloneBuf.ptr != nil {
				v.Free(source)
				t.Fatal("Free(clone) did not invalidate the clone handle")
			}
			if sourceBuf.ptr == nil {
				t.Fatal("Free(clone) invalidated the source handle")
			}
			if got := v.Read(source); !slices.Equal(got, mutatedValues) {
				v.Free(source)
				t.Fatalf("source after Free(clone) = %v, want %v", got, mutatedValues)
			}
			v.Free(source)
		})
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
	v.budgetBytes, v.dlUsed, v.hostvisN = 72, 0, 0

	c := cpu()
	var s lcg = 365
	shape := []int{2, 32} // Q8 code buffer is 64 bytes.
	w := randVec(&s, shape[0]*shape[1])
	dw := v.Upload(NewF32(c, shape, w), Q8_0)
	db := dw.buf.(*vulkanBuf)
	if db.budgetedWeightBytes != 64 {
		t.Fatalf("budget charge=%d want 64", db.budgetedWeightBytes)
	}
	if v.dlUsed != 72 {
		t.Fatalf("dlUsed after first upload=%d want 72", v.dlUsed)
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
	v.budgetBytes, v.dlUsed, v.hostvisN = 72, 0, 0

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

func TestVulkanTeardownResourcesIsIdempotent(t *testing.T) {
	v := vk(t)
	a := v.Upload(NewF32(Default(), []int{4}, []float32{1, 2, 3, 4}), F32)
	b := v.Upload(NewF32(Default(), []int{4}, []float32{4, 3, 2, 1}), F32)
	defer v.Free(a)
	defer v.Free(b)
	v.BeginBatch()
	v.AddInPlace(a, b)
	if err := v.TeardownResources(); err != nil {
		t.Fatalf("first teardown: %v", err)
	}
	if err := v.TeardownResources(); err != nil {
		t.Fatalf("second teardown: %v", err)
	}
	got := v.Read(a)
	for i, x := range got {
		if x != 5 {
			t.Fatalf("result[%d] = %v, want 5", i, x)
		}
	}
	// Resource teardown is not device destruction: later operations remain valid.
	v.AddInPlace(a, b)
	got = v.Read(a)
	want := []float32{9, 8, 7, 6}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("continued result[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestVulkanQ4KTensorHomeReusesCopyAndResets(t *testing.T) {
	v := vk(t)
	v.VulkanDebugResetQ4KStage()
	defer v.VulkanDebugResetQ4KStage()
	const out, in = 8, 256
	rng := rand.New(rand.NewSource(9833))
	raw := make([]byte, out*(in/q4kSuper)*q4kSuperBlock)
	for b := 0; b < out*(in/q4kSuper); b++ {
		randQ4KBlockC(rng, raw[b*q4kSuperBlock:(b+1)*q4kSuperBlock])
	}
	xv := make([]float32, in)
	for i := range xv {
		xv[i] = rng.Float32()*2 - 1
	}
	hw := NewQ4K(Default(), []int{out, in}, raw)
	w := v.Upload(hw, Q4_K)
	defer v.Free(w)
	w.buf.(*vulkanBuf).hostVisibleWeight = true
	x := v.Upload(NewF32(Default(), []int{in}, xv), F32)
	defer v.Free(x)
	oldStage, oldBudget := v.q4kStage, v.budgetBytes
	v.q4kStage, v.budgetBytes = true, v.dlUsed+int64(len(raw))*16+(64<<20)
	defer func() { v.q4kStage, v.budgetBytes = oldStage, oldBudget }()
	v.BeginBatch()
	y1 := v.MatMul(w, x)
	v.FlushBatch()
	defer v.Free(y1)
	h1, m1, b1, e1, r1, c1 := v.VulkanDebugQ4KTensorHomeSnapshot()
	if h1 != 0 || m1 != 1 || b1 != 0 || e1 != 1 || r1 != int64(len(raw)) || c1 != int64(len(raw)) {
		t.Fatalf("first snapshot=%d,%d,%d,%d,%d,%d", h1, m1, b1, e1, r1, c1)
	}
	v.BeginBatch()
	y2 := v.MatMul(w, x)
	v.FlushBatch()
	defer v.Free(y2)
	h2, m2, b2, e2, r2, c2 := v.VulkanDebugQ4KTensorHomeSnapshot()
	if h2 != 1 || m2 != 1 || b2 != 0 || e2 != 1 || r2 != r1 || c2 != c1 {
		t.Fatalf("reuse snapshot=%d,%d,%d,%d,%d,%d", h2, m2, b2, e2, r2, c2)
	}
	want := Default().Read(Default().MatMul(hw, NewF32(Default(), []int{in}, xv)))
	if c := cosineC(v.Read(y2), want); c < 0.995 {
		t.Fatalf("cosine %.8f", c)
	}
	v.VulkanDebugResetQ4KStage()
	_, _, _, entries, resident, _ := v.VulkanDebugQ4KTensorHomeSnapshot()
	if entries != 0 || resident != 0 {
		t.Fatalf("reset entries=%d resident=%d", entries, resident)
	}
}
