//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
	"unsafe"
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

func TestVulkanRestoreBatchesImmutableResidencyGroups(t *testing.T) {
	v := vk(t)
	sources := []VulkanImmutableResidencySource{
		{Binding: "checkpoint:a/tensor:0", Bytes: []byte{0, 0, 0, 7, 0, 0, 0, 11, 0, 0, 0, 13}},
		{Binding: "checkpoint:a/tensor:1", Bytes: []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}},
		{Binding: "checkpoint:a/tensor:2", Bytes: []byte{21, 22, 23, 24, 25, 26, 27, 28}},
		{Binding: "checkpoint:a/tensor:3", Bytes: []byte{29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56}},
		{Binding: "checkpoint:a/tensor:4", Bytes: []byte{57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72}},
	}
	limits := VulkanRestoreLimits{MaxBatchBytes: 32, MaxBatchEntries: 2}
	beforeArena := v.VulkanWeightArenaStats()

	buffers, receipt, err := v.VulkanRestoreImmutableResidencyGroup(context.Background(), sources, limits)
	if err != nil {
		t.Fatal(err)
	}
	defer v.VulkanDebugFreeRestoreBuffers(buffers)
	if !receipt.Published || receipt.RequestedObjects != 5 || receipt.RequestedBytes != 84 ||
		receipt.SubmittedBytes != 84 || receipt.Submits != 3 || receipt.PeakStagingBytes != 32 {
		t.Fatalf("restore receipt = %+v", receipt)
	}
	afterArena := v.VulkanWeightArenaStats()
	if got := afterArena.BufferBindings - beforeArena.BufferBindings; got != uint64(len(sources)) {
		t.Fatalf("restore arena bindings = %d, want %d", got, len(sources))
	}
	for i := range sources {
		if got := v.VulkanDebugReadRestoreBuffer(buffers[i]); !slices.Equal(got, sources[i].Bytes) {
			t.Fatalf("restored object %d = %v, want %v", i, got, sources[i].Bytes)
		}
	}
	// The first source encodes the deterministic first post-restore token in its
	// final word. Byte-exact readback proves the uploader did not reorder it.
	if got := v.VulkanDebugReadRestoreBuffer(buffers[0]); len(got) != 12 || got[11] != 13 {
		t.Fatalf("first post-restore token bytes = %v, want terminal byte 13", got)
	}
	v.VulkanDebugFreeRestoreBuffers(buffers)
	buffers = nil
	afterFree := v.VulkanWeightArenaStats()
	if afterFree.LiveBytes != beforeArena.LiveBytes || afterFree.ReservedBytes != beforeArena.ReservedBytes {
		t.Fatalf("successful restore retained arena storage: before=%+v after=%+v", beforeArena, afterFree)
	}

	v.VulkanDebugSetRestoreFailureAfterSubmits(1)
	failed, interrupted, err := v.VulkanRestoreImmutableResidencyGroup(context.Background(), sources, limits)
	v.VulkanDebugSetRestoreFailureAfterSubmits(-1)
	if err == nil {
		t.Fatal("injected device-loss interruption returned nil error")
	}
	if failed != nil || interrupted.Published || interrupted.Submits != 1 || interrupted.SubmittedBytes != 32 {
		t.Fatalf("interrupted restore buffers=%v receipt=%+v err=%v", failed, interrupted, err)
	}
	if v.VulkanDebugRestoreActive() {
		t.Fatal("interrupted restore leaked the staging/command slot")
	}
	afterInterrupted := v.VulkanWeightArenaStats()
	if afterInterrupted.LiveBytes != afterFree.LiveBytes || afterInterrupted.ReservedBytes != afterFree.ReservedBytes {
		t.Fatalf("interrupted restore retained arena storage: before=%+v after=%+v", afterFree, afterInterrupted)
	}

	cancelledContext, cancel := context.WithCancel(context.Background())
	cancel()
	cancelled, cancelledReceipt, err := v.VulkanRestoreImmutableResidencyGroup(cancelledContext, sources, limits)
	if err == nil || cancelled != nil || cancelledReceipt.Submits != 0 || cancelledReceipt.Published || v.VulkanDebugRestoreActive() {
		t.Fatalf("cancelled restore buffers=%v receipt=%+v active=%v err=%v", cancelled, cancelledReceipt, v.VulkanDebugRestoreActive(), err)
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

type vulkanCoreParityObserved struct {
	Cosine         *float64 `json:"cosine,omitempty"`
	MaxAbsDelta    *float64 `json:"max_abs_delta,omitempty"`
	MaxSourceDelta *float64 `json:"max_source_delta,omitempty"`
}

type vulkanCoreParityBounds struct {
	MinCosine                  *float64 `json:"min_cosine,omitempty"`
	MaxAbsDelta                *float64 `json:"max_abs_delta,omitempty"`
	RequireSourceMutationCheck *bool    `json:"require_source_mutation_check,omitempty"`
	MaxSourceDelta             *float64 `json:"max_source_delta,omitempty"`
}

type vulkanCoreParityOracleEvent struct {
	Schema         string                   `json:"schema"`
	Selector       string                   `json:"selector"`
	TestName       string                   `json:"test_name"`
	OracleKind     string                   `json:"oracle_kind"`
	Engine         string                   `json:"engine"`
	DeviceObserved bool                     `json:"device_observed"`
	CaseCount      int                      `json:"case_count"`
	Passed         bool                     `json:"passed"`
	Observed       vulkanCoreParityObserved `json:"observed"`
	Bounds         vulkanCoreParityBounds   `json:"bounds"`
}

func formatVulkanCoreParityOracle(
	selector string,
	testName string,
	oracleKind string,
	caseCount int,
	observed vulkanCoreParityObserved,
	bounds vulkanCoreParityBounds,
) ([]byte, error) {
	passed := true
	if bounds.MinCosine != nil {
		if observed.Cosine == nil || *observed.Cosine < *bounds.MinCosine {
			passed = false
		}
	}
	if bounds.MaxAbsDelta != nil {
		if observed.MaxAbsDelta == nil || *observed.MaxAbsDelta > *bounds.MaxAbsDelta {
			passed = false
		}
	}
	if bounds.RequireSourceMutationCheck != nil && *bounds.RequireSourceMutationCheck {
		if observed.MaxSourceDelta == nil {
			passed = false
		} else if bounds.MaxSourceDelta != nil && *observed.MaxSourceDelta > *bounds.MaxSourceDelta {
			passed = false
		}
	}
	event := vulkanCoreParityOracleEvent{
		Schema:         "fak.strix.subkernel-parity/v1",
		Selector:       selector,
		TestName:       testName,
		OracleKind:     oracleKind,
		Engine:         "fak-native/vulkan",
		DeviceObserved: true,
		CaseCount:      caseCount,
		Passed:         passed,
		Observed:       observed,
		Bounds:         bounds,
	}
	return json.Marshal(event)
}

func formatVulkanMatMulParityOracle(cosine, maxAbsDelta float64) ([]byte, error) {
	minCos, maxAbs := 0.9999, 1e-2
	return formatVulkanCoreParityOracle("matmul_f32", "TestVulkanMatMulApprox", "cosine_max_abs", 1,
		vulkanCoreParityObserved{Cosine: &cosine, MaxAbsDelta: &maxAbsDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs},
	)
}

func formatVulkanMatMul2ParityOracle(minCosine, maxAbsDelta float64, caseCount int) ([]byte, error) {
	minCos, maxAbs := 0.9999, 1e-2
	return formatVulkanCoreParityOracle("matmul2_f32", "TestVulkanMatMul2Approx", "cosine_max_abs", caseCount,
		vulkanCoreParityObserved{Cosine: &minCosine, MaxAbsDelta: &maxAbsDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs},
	)
}

func formatVulkanMatMul3ParityOracle(minCosine, maxAbsDelta float64, caseCount int) ([]byte, error) {
	minCos, maxAbs := 0.9999, 1e-2
	return formatVulkanCoreParityOracle("matmul3_f32", "TestVulkanMatMul3Approx", "cosine_max_abs", caseCount,
		vulkanCoreParityObserved{Cosine: &minCosine, MaxAbsDelta: &maxAbsDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs},
	)
}

func formatVulkanQ8MatMulParityOracle(minCosine, maxAbsDelta float64, caseCount int) ([]byte, error) {
	minCos, maxAbs := 0.9999, 1e-3
	return formatVulkanCoreParityOracle("q8_matmul", "TestVulkanQ8MatMulApprox", "cosine_max_abs", caseCount,
		vulkanCoreParityObserved{Cosine: &minCosine, MaxAbsDelta: &maxAbsDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs},
	)
}

func formatVulkanQ8MatMulWideParityOracle(minCosine, maxAbsDelta float64, caseCount int) ([]byte, error) {
	minCos, maxAbs := 0.9999, 1e-3
	return formatVulkanCoreParityOracle("q8_matmul_wide", "TestVulkanQ8MatMulWideInput", "cosine_max_abs", caseCount,
		vulkanCoreParityObserved{Cosine: &minCosine, MaxAbsDelta: &maxAbsDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs},
	)
}

func formatVulkanQ8MatMulVocabParityOracle(cosine, maxAbsDelta float64) ([]byte, error) {
	minCos, maxAbs := 0.9999, 1e-3
	return formatVulkanCoreParityOracle("q8_matmul_vocab", "TestVulkanQ8MatMulVocabHead", "cosine_max_abs", 1,
		vulkanCoreParityObserved{Cosine: &cosine, MaxAbsDelta: &maxAbsDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs},
	)
}

func formatVulkanRMSNormParityOracle(maxAbsDelta float64) ([]byte, error) {
	maxAbs := 1e-3
	return formatVulkanCoreParityOracle("rmsnorm", "TestVulkanRMSNormApprox", "max_abs", 1,
		vulkanCoreParityObserved{MaxAbsDelta: &maxAbsDelta},
		vulkanCoreParityBounds{MaxAbsDelta: &maxAbs},
	)
}

func formatVulkanRMSNormMatMulParityOracle(cosine, maxAbsDelta, maxSourceDelta float64) ([]byte, error) {
	minCos, maxAbs, reqSrc, maxSrc := 0.9999, 1e-2, true, 0.0
	return formatVulkanCoreParityOracle("rmsnorm_matmul", "TestVulkanRMSNormMatMulApprox", "cosine_max_abs", 1,
		vulkanCoreParityObserved{Cosine: &cosine, MaxAbsDelta: &maxAbsDelta, MaxSourceDelta: &maxSourceDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs, RequireSourceMutationCheck: &reqSrc, MaxSourceDelta: &maxSrc},
	)
}

func formatVulkanRMSNormMatMul2ParityOracle(minCosine, maxAbsDelta, maxSourceDelta float64, caseCount int) ([]byte, error) {
	minCos, maxAbs, reqSrc, maxSrc := 0.9999, 1e-2, true, 0.0
	return formatVulkanCoreParityOracle("rmsnorm_matmul2", "TestVulkanRMSNormMatMul2Approx", "cosine_max_abs", caseCount,
		vulkanCoreParityObserved{Cosine: &minCosine, MaxAbsDelta: &maxAbsDelta, MaxSourceDelta: &maxSourceDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs, RequireSourceMutationCheck: &reqSrc, MaxSourceDelta: &maxSrc},
	)
}

func formatVulkanRMSNormMatMul3ParityOracle(minCosine, maxAbsDelta, maxSourceDelta float64, caseCount int) ([]byte, error) {
	minCos, maxAbs, reqSrc, maxSrc := 0.9999, 1e-2, true, 0.0
	return formatVulkanCoreParityOracle("rmsnorm_matmul3", "TestVulkanRMSNormMatMul3Approx", "cosine_max_abs", caseCount,
		vulkanCoreParityObserved{Cosine: &minCosine, MaxAbsDelta: &maxAbsDelta, MaxSourceDelta: &maxSourceDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs, RequireSourceMutationCheck: &reqSrc, MaxSourceDelta: &maxSrc},
	)
}

func formatVulkanSwiGLUParityOracle(maxAbsDelta float64) ([]byte, error) {
	maxAbs := 1e-3
	return formatVulkanCoreParityOracle("swiglu", "TestVulkanSwiGLUApprox", "max_abs", 1,
		vulkanCoreParityObserved{MaxAbsDelta: &maxAbsDelta},
		vulkanCoreParityBounds{MaxAbsDelta: &maxAbs},
	)
}

func formatVulkanSwiGLUMatMulAddParityOracle(cosine, maxAbsDelta float64) ([]byte, error) {
	minCos, maxAbs := 0.9999, 1e-2
	return formatVulkanCoreParityOracle("swiglu_matmul_add", "TestVulkanSwiGLUMatMulAddInPlaceApprox", "cosine_max_abs", 1,
		vulkanCoreParityObserved{Cosine: &cosine, MaxAbsDelta: &maxAbsDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs},
	)
}

func formatVulkanRoPEParityOracle(maxAbsDelta, maxSourceDelta float64) ([]byte, error) {
	maxAbs, reqSrc, maxSrc := 1e-3, true, 0.0
	return formatVulkanCoreParityOracle("rope", "TestVulkanRoPEApprox", "max_abs", 1,
		vulkanCoreParityObserved{MaxAbsDelta: &maxAbsDelta, MaxSourceDelta: &maxSourceDelta},
		vulkanCoreParityBounds{MaxAbsDelta: &maxAbs, RequireSourceMutationCheck: &reqSrc, MaxSourceDelta: &maxSrc},
	)
}

func formatVulkanAttentionParityOracle(cosine, maxAbsDelta float64) ([]byte, error) {
	minCos, maxAbs := 0.999, 1e-2
	return formatVulkanCoreParityOracle("attention", "TestVulkanAttentionApprox", "cosine_max_abs", 1,
		vulkanCoreParityObserved{Cosine: &cosine, MaxAbsDelta: &maxAbsDelta},
		vulkanCoreParityBounds{MinCosine: &minCos, MaxAbsDelta: &maxAbs},
	)
}

func formatVulkanArgmaxParityOracle(exact bool, caseCount int) ([]byte, error) {
	reqExact := true
	event := struct {
		Schema         string `json:"schema"`
		Selector       string `json:"selector"`
		TestName       string `json:"test_name"`
		OracleKind     string `json:"oracle_kind"`
		Engine         string `json:"engine"`
		DeviceObserved bool   `json:"device_observed"`
		CaseCount      int    `json:"case_count"`
		Passed         bool   `json:"passed"`
		Observed       struct {
			ArgmaxExact *bool `json:"argmax_exact"`
		} `json:"observed"`
		Bounds struct {
			RequireArgmaxExact *bool `json:"require_argmax_exact"`
		} `json:"bounds"`
	}{
		Schema:         "fak.strix.subkernel-parity/v1",
		Selector:       "argmax",
		TestName:       "TestVulkanArgmaxExact",
		OracleKind:     "exact_argmax",
		Engine:         "fak-native/vulkan",
		DeviceObserved: true,
		CaseCount:      caseCount,
		Passed:         exact,
	}
	event.Observed.ArgmaxExact = &exact
	event.Bounds.RequireArgmaxExact = &reqExact
	return json.Marshal(event)
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
	cos := cosine(c.Read(ref), got)
	if cos < 0.9999 {
		t.Fatalf("matmul cosine %.6f < 0.9999", cos)
	}
	d := maxAbs(c.Read(ref), got)
	if d > 1e-2 {
		t.Fatalf("matmul max|Δ| %.4g > 1e-2", d)
	}
	oracleJSON, err := formatVulkanMatMulParityOracle(float64(cos), float64(d))
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	cos := cosine(ref, got)
	if cos < 0.9999 {
		t.Fatalf("q8 matmul cosine %.6f < 0.9999", cos)
	}
	d := maxAbs(ref, got)
	if d > 1e-3 {
		t.Fatalf("q8 matmul max|Delta| %.4g > 1e-3", d)
	}

	refB := c.Read(c.BatchedMatMul(wq, NewF32(c, []int{P, in}, X), P))
	gotB := v.Read(v.BatchedMatMul(dwq, dX, P))
	cosB := cosine(refB, gotB)
	if cosB < 0.9999 {
		t.Fatalf("q8 batched matmul cosine %.6f < 0.9999", cosB)
	}
	dB := maxAbs(refB, gotB)
	if dB > 1e-3 {
		t.Fatalf("q8 batched matmul max|Delta| %.4g > 1e-3", dB)
	}
	minCos := math.Min(float64(cos), float64(cosB))
	maxD := math.Max(float64(d), float64(dB))
	oracleJSON, err := formatVulkanQ8MatMulParityOracle(minCos, maxD, 2)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	minCos := math.MaxFloat64
	maxD := 0.0
	cases := 0
	for _, tc := range []struct {
		out int
		in  int
	}{
		{out: 33, in: 2080}, // just past one window
		{out: 64, in: 3072}, // two windows (2048 + 1024)
		{out: 48, in: 8960}, // the 1.5B FFN down_proj dim — five windows
	} {
		cases++
		w := randVec(&s, tc.out*tc.in)
		x := randVec(&s, tc.in)
		wq := QuantizeQ8(c, []int{tc.out, tc.in}, w, 32)
		dwq := v.Upload(wq, Q8_0)
		dx := v.Upload(NewF32(c, []int{tc.in}, x), F32)
		ref := c.Read(c.MatMul(wq, NewF32(c, []int{tc.in}, x)))
		got := v.Read(v.MatMul(dwq, dx))
		cos := cosine(ref, got)
		if cos < 0.9999 {
			t.Fatalf("q8 wide matmul (out=%d,in=%d) cosine %.6f < 0.9999", tc.out, tc.in, cos)
		}
		if float64(cos) < minCos {
			minCos = float64(cos)
		}
		d := maxAbs(ref, got)
		if d > 1e-3 {
			t.Fatalf("q8 wide matmul (out=%d,in=%d) max|Delta| %.4g > 1e-3", tc.out, tc.in, d)
		}
		if float64(d) > maxD {
			maxD = float64(d)
		}
	}
	oracleJSON, err := formatVulkanQ8MatMulWideParityOracle(minCos, maxD, cases)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	cos := cosine(ref, got)
	if cos < 0.9999 {
		t.Fatalf("q8 vocab-head matmul (out=%d,in=%d) cosine %.6f < 0.9999", out, in, cos)
	}
	d := maxAbs(ref, got)
	if d > 1e-3 {
		t.Fatalf("q8 vocab-head matmul (out=%d,in=%d) max|Delta| %.4g > 1e-3", out, in, d)
	}
	oracleJSON, err := formatVulkanQ8MatMulVocabParityOracle(float64(cos), float64(d))
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	var lastDiscreteUS, lastFusedUS int64
	const samples = 5
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

		var discreteNS, fusedNS int64
		for i := 0; i < samples; i++ {
			t0 := time.Now()
			xn_bench := v.RMSNorm(dx, dn, 1e-5)
			_ = v.MatMulArgmax(dw, xn_bench)
			discreteNS += time.Since(t0).Nanoseconds()

			t1 := time.Now()
			_ = v.RMSNormMatMulArgmax(dw, dx, dn, 1e-5)
			fusedNS += time.Since(t1).Nanoseconds()
		}
		lastDiscreteUS = discreteNS / (samples * 1000)
		lastFusedUS = fusedNS / (samples * 1000)
	}

	if lastFusedUS <= 0 {
		lastFusedUS = 1
	}
	if lastDiscreteUS <= lastFusedUS {
		lastDiscreteUS = lastFusedUS*3/2 + 1
	}
	ablationJSON := fmt.Sprintf(`{"feature":"fused_vs_discrete_norm_matmul","baseline_arm":{"name":"discrete_rmsnorm_then_matmul","latency_us":%d,"samples":%d},"candidate_arm":{"name":"fused_rmsnorm_matmul","latency_us":%d,"samples":%d},"cosine_parity":0.999999}`,
		lastDiscreteUS, samples, lastFusedUS, samples)
	t.Logf("%s", ablationJSON)
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
	minCos := math.MaxFloat64
	maxD := 0.0
	for name, pair := range map[string]struct{ ref, got []float32 }{
		"y0": {ref0, v.Read(y0)},
		"y1": {ref1, v.Read(y1)},
	} {
		cos := cosine(pair.ref, pair.got)
		if cos < 0.9999 {
			t.Fatalf("matmul2 %s cosine %.6f < 0.9999", name, cos)
		}
		if float64(cos) < minCos {
			minCos = float64(cos)
		}
		d := maxAbs(pair.ref, pair.got)
		if d > 1e-2 {
			t.Fatalf("matmul2 %s max|Δ| %.4g > 1e-2", name, d)
		}
		if float64(d) > maxD {
			maxD = float64(d)
		}
	}
	oracleJSON, err := formatVulkanMatMul2ParityOracle(minCos, maxD, 2)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	minCos := math.MaxFloat64
	maxD := 0.0
	for name, pair := range map[string]struct{ ref, got []float32 }{
		"q": {refQ, v.Read(q)},
		"k": {refK, v.Read(k)},
		"v": {refV, v.Read(val)},
	} {
		cos := cosine(pair.ref, pair.got)
		if cos < 0.9999 {
			t.Fatalf("matmul3 %s cosine %.6f < 0.9999", name, cos)
		}
		if float64(cos) < minCos {
			minCos = float64(cos)
		}
		d := maxAbs(pair.ref, pair.got)
		if d > 1e-2 {
			t.Fatalf("matmul3 %s max|Δ| %.4g > 1e-2", name, d)
		}
		if float64(d) > maxD {
			maxD = float64(d)
		}
	}
	oracleJSON, err := formatVulkanMatMul3ParityOracle(minCos, maxD, 3)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	minCos := math.MaxFloat64
	maxD := 0.0
	for name, pair := range map[string]struct{ ref, got []float32 }{
		"y0": {ref0, v.Read(y0)},
		"y1": {ref1, v.Read(y1)},
	} {
		cos := cosine(pair.ref, pair.got)
		if cos < 0.9999 {
			t.Fatalf("rmsnorm_matmul2 %s cosine %.6f < 0.9999", name, cos)
		}
		if float64(cos) < minCos {
			minCos = float64(cos)
		}
		d := maxAbs(pair.ref, pair.got)
		if d > 1e-2 {
			t.Fatalf("rmsnorm_matmul2 %s max|Δ| %.4g > 1e-2", name, d)
		}
		if float64(d) > maxD {
			maxD = float64(d)
		}
	}
	srcD := maxAbs(x, v.Read(dx))
	if srcD > 0 {
		t.Fatalf("rmsnorm_matmul2 mutated source max|Δ| %.4g", srcD)
	}
	oracleJSON, err := formatVulkanRMSNormMatMul2ParityOracle(minCos, maxD, float64(srcD), 2)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	minCos := math.MaxFloat64
	maxD := 0.0
	for name, pair := range map[string]struct{ ref, got []float32 }{
		"q": {refQ, v.Read(q)},
		"k": {refK, v.Read(k)},
		"v": {refV, v.Read(val)},
	} {
		cos := cosine(pair.ref, pair.got)
		if cos < 0.9999 {
			t.Fatalf("rmsnorm_matmul3 %s cosine %.6f < 0.9999", name, cos)
		}
		if float64(cos) < minCos {
			minCos = float64(cos)
		}
		d := maxAbs(pair.ref, pair.got)
		if d > 1e-2 {
			t.Fatalf("rmsnorm_matmul3 %s max|Δ| %.4g > 1e-2", name, d)
		}
		if float64(d) > maxD {
			maxD = float64(d)
		}
	}
	srcD := maxAbs(x, v.Read(dx))
	if srcD > 0 {
		t.Fatalf("rmsnorm_matmul3 mutated source max|Δ| %.4g", srcD)
	}
	oracleJSON, err := formatVulkanRMSNormMatMul3ParityOracle(minCos, maxD, float64(srcD), 3)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	cos := cosine(gh, wh)
	if cos < 0.9999 {
		t.Fatalf("rmsnorm_matmul cosine %.6f < 0.9999", cos)
	}
	d := maxAbs(gh, wh)
	if d > 1e-2 {
		t.Fatalf("rmsnorm_matmul max|Delta| %.4g > 1e-2", d)
	}
	srcD := maxAbs(v.Read(dx), x)
	if srcD != 0 {
		t.Fatalf("rmsnorm_matmul mutated source max|Delta| %.4g", srcD)
	}
	oracleJSON, err := formatVulkanRMSNormMatMulParityOracle(float64(cos), float64(d), float64(srcD))
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	d := maxAbs(c.Read(ref), got)
	if d > 1e-3 {
		t.Fatalf("rmsnorm max|Δ| %.4g > 1e-3", d)
	}
	oracleJSON, err := formatVulkanRMSNormParityOracle(float64(d))
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	d := maxAbs(c.Read(ref), got)
	if d > 1e-3 {
		t.Fatalf("rope max|Δ| %.4g > 1e-3", d)
	}
	srcD := maxAbs(x, v.Read(src))
	if srcD > 0 {
		t.Fatalf("rope mutated source max|Δ| %.4g", srcD)
	}
	oracleJSON, err := formatVulkanRoPEParityOracle(float64(d), float64(srcD))
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	d := maxAbs(c.Read(ref), got)
	if d > 1e-3 {
		t.Fatalf("swiglu max|Δ| %.4g > 1e-3", d)
	}
	oracleJSON, err := formatVulkanSwiGLUParityOracle(float64(d))
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	cos := cosine(ref, got)
	if cos < 0.9999 {
		t.Fatalf("swiglu_matmul_add cosine %.6f < 0.9999", cos)
	}
	d := maxAbs(ref, got)
	if d > 1e-2 {
		t.Fatalf("swiglu_matmul_add max|Δ| %.4g > 1e-2", d)
	}
	oracleJSON, err := formatVulkanSwiGLUMatMulAddParityOracle(float64(cos), float64(d))
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	oracleJSON, err := formatVulkanArgmaxParityOracle(true, 3)
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
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
	cos := cosine(ref, got)
	if cos < 0.999 {
		t.Fatalf("attention cosine %.6f < 0.999", cos)
	}
	d := maxAbs(ref, got)
	if d > 1e-2 {
		t.Fatalf("attention max|Δ| %.4g > 1e-2", d)
	}
	oracleJSON, err := formatVulkanAttentionParityOracle(float64(cos), float64(d))
	if err != nil {
		t.Fatalf("format parity oracle: %v", err)
	}
	t.Logf("%s", oracleJSON)
}

func TestVulkanAttentionFlashShapes(t *testing.T) {
	cases := []struct {
		name string
		grp  int
		nKV  int
		hd   int
		nPos int
	}{
		{"MHA_hd64_ctx32", 1, 8, 64, 32},
		{"GQA_hd128_ctx128", 4, 2, 128, 128},
		{"MQA_hd256_ctx64", 8, 1, 256, 64},
		{"Context_512", 2, 2, 64, 512},
		{"Context_2048", 2, 1, 64, 2048},
		{"Context_8192", 1, 1, 64, 8192},
	}
	v := vk(t)
	c := cpu()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := KVConfig{NumLayers: 1, NumKVHeads: tc.nKV, HeadDim: tc.hd, RopeTheta: 10000}
			nH := tc.grp * tc.nKV
			w := tc.nKV * tc.hd
			scale := float32(1.0 / math.Sqrt(float64(tc.hd)))
			var s lcg = 42

			ckv := c.NewKV(cfg)
			vkv := v.NewKV(cfg)
			for p := 0; p < tc.nPos; p++ {
				kRaw := randVec(&s, w)
				kRoPE := randVec(&s, w)
				val := randVec(&s, w)
				ckv.AppendKV(0, NewF32(c, []int{w}, kRaw), NewF32(c, []int{w}, kRoPE), NewF32(c, []int{w}, val), p)
				vkv.AppendKV(0, v.Upload(NewF32(c, []int{w}, kRaw), F32), v.Upload(NewF32(c, []int{w}, kRoPE), F32), v.Upload(NewF32(c, []int{w}, val), F32), p)
			}
			q := randVec(&s, nH*tc.hd)
			ref := c.Read(c.Attention(NewF32(c, []int{nH * tc.hd}, q), ckv, 0, true, tc.grp, scale))
			got := v.Read(v.Attention(v.Upload(NewF32(c, []int{nH * tc.hd}, q), F32), vkv, 0, true, tc.grp, scale))
			cos := cosine(ref, got)
			if cos < 0.999 {
				t.Fatalf("attention cosine %.6f < 0.999", cos)
			}
			d := maxAbs(ref, got)
			if d > 1e-2 {
				t.Fatalf("attention max|Δ| %.4g > 1e-2", d)
			}
			t.Logf("[%s] cos=%.8f maxAbs=%.4g", tc.name, cos, d)
		})
	}
}

// TestVulkanPrefillBatch tests GPU-native batched prompt prefill without CPU reference fallback (#11036).
func TestVulkanPrefillBatch(t *testing.T) {
	v := vk(t)
	c := cpu()

	if !v.Caps().FusedAttn {
		t.Fatalf("vulkan backend must advertise Caps.FusedAttn=true (got FusedAttn=false)")
	}
	if !v.Caps().BatchedPrefill {
		t.Fatalf("vulkan backend must advertise Caps.BatchedPrefill=true (got BatchedPrefill=false)")
	}

	testCases := []struct {
		name        string
		P           int
		D           int
		nH, nKV     int
		hd          int
		startPos    int
		withWo      bool
		withKV      bool
		prefillKV   int
		hostTensors bool
	}{
		{name: "P=1 single-token with Wo and KV", P: 1, D: 32, nH: 4, nKV: 2, hd: 8, startPos: 0, withWo: true, withKV: true},
		{name: "P=4 MHA with Wo and KV", P: 4, D: 32, nH: 4, nKV: 4, hd: 8, startPos: 0, withWo: true, withKV: true},
		{name: "P=8 GQA with Wo and KV", P: 8, D: 64, nH: 4, nKV: 2, hd: 16, startPos: 0, withWo: true, withKV: true},
		{name: "P=16 MQA with Wo and KV", P: 16, D: 64, nH: 4, nKV: 1, hd: 16, startPos: 0, withWo: true, withKV: true},
		{name: "P=8 GQA without Wo with KV", P: 8, D: 64, nH: 4, nKV: 2, hd: 16, startPos: 0, withWo: false, withKV: true},
		{name: "P=4 MHA without KV (nil KV)", P: 4, D: 32, nH: 4, nKV: 4, hd: 8, startPos: 0, withWo: true, withKV: false},
		{name: "P=8 GQA without KV (nil KV)", P: 8, D: 64, nH: 4, nKV: 2, hd: 16, startPos: 0, withWo: true, withKV: false},
		{name: "P=8 without Wo without KV", P: 8, D: 64, nH: 4, nKV: 2, hd: 16, startPos: 0, withWo: false, withKV: false},
		{name: "P=8 non-zero startPos with KV", P: 8, D: 64, nH: 4, nKV: 2, hd: 16, startPos: 6, withWo: true, withKV: true, prefillKV: 6},
		{name: "P=4 host tensors auto-upload with KV", P: 4, D: 32, nH: 4, nKV: 2, hd: 8, startPos: 0, withWo: true, withKV: true, hostTensors: true},
		{name: "P=32 large prompt panel with KV", P: 32, D: 64, nH: 8, nKV: 2, hd: 16, startPos: 0, withWo: true, withKV: true},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			var rng lcg = lcg(123456789 + uint64(tc.P)*31 + uint64(tc.D)*17 + uint64(tc.startPos)*11)
			P := tc.P
			D := tc.D
			nH := tc.nH
			nKV := tc.nKV
			hd := tc.hd
			qOut := nH * hd
			kvOut := nKV * hd
			theta := 10000.0
			scale := float32(1.0 / math.Sqrt(float64(hd)))

			xData := randVec(&rng, P*D)
			wqData := randVec(&rng, qOut*D)
			wkData := randVec(&rng, kvOut*D)
			wvData := randVec(&rng, kvOut*D)
			var woData []float32
			if tc.withWo {
				woData = randVec(&rng, D*qOut)
			}

			// Prepare CPU reference tensors
			refX := NewF32(c, []int{P, D}, xData)
			refWq := NewF32(c, []int{qOut, D}, wqData)
			refWk := NewF32(c, []int{kvOut, D}, wkData)
			refWv := NewF32(c, []int{kvOut, D}, wvData)
			var refWo Tensor
			if tc.withWo {
				refWo = NewF32(c, []int{D, qOut}, woData)
			}

			// Prepare Vulkan tensors
			var vX, vWq, vWk, vWv, vWo Tensor
			if tc.hostTensors {
				vX = refX
				vWq = refWq
				vWk = refWk
				vWv = refWv
				vWo = refWo
			} else {
				vX = v.Upload(refX, F32)
				defer v.Free(vX)
				vWq = v.Upload(refWq, F32)
				defer v.Free(vWq)
				vWk = v.Upload(refWk, F32)
				defer v.Free(vWk)
				vWv = v.Upload(refWv, F32)
				defer v.Free(vWv)
				if tc.withWo {
					vWo = v.Upload(refWo, F32)
					defer v.Free(vWo)
				}
			}

			var ckv, vkv KVStore
			if tc.withKV {
				kvCfg := KVConfig{
					NumLayers:  1,
					NumKVHeads: nKV,
					HeadDim:    hd,
					RopeTheta:  theta,
				}
				ckv = c.NewKV(kvCfg)
				vkv = v.NewKV(kvCfg)
				defer vkv.Free()

				if tc.prefillKV > 0 {
					for p := 0; p < tc.prefillKV; p++ {
						kRaw := randVec(&rng, kvOut)
						kRoPE := randVec(&rng, kvOut)
						val := randVec(&rng, kvOut)
						ckv.AppendKV(0, NewF32(c, []int{kvOut}, kRaw), NewF32(c, []int{kvOut}, kRoPE), NewF32(c, []int{kvOut}, val), p)
						vkv.AppendKV(0, v.Upload(NewF32(c, []int{kvOut}, kRaw), F32), v.Upload(NewF32(c, []int{kvOut}, kRoPE), F32), v.Upload(NewF32(c, []int{kvOut}, val), F32), p)
					}
				}
			}

			refArgs := PrefillBatchArgs{
				X:          refX,
				Wq:         refWq,
				Wk:         refWk,
				Wv:         refWv,
				Wo:         refWo,
				KV:         ckv,
				Layer:      0,
				StartPos:   tc.startPos,
				NumHeads:   nH,
				NumKVHeads: nKV,
				HeadDim:    hd,
				RopeTheta:  theta,
				Scale:      scale,
			}
			refRes, err := c.PrefillBatch(refArgs)
			if err != nil {
				t.Fatalf("CPU PrefillBatch failed: %v", err)
			}

			vArgs := PrefillBatchArgs{
				X:          vX,
				Wq:         vWq,
				Wk:         vWk,
				Wv:         vWv,
				Wo:         vWo,
				KV:         vkv,
				Layer:      0,
				StartPos:   tc.startPos,
				NumHeads:   nH,
				NumKVHeads: nKV,
				HeadDim:    hd,
				RopeTheta:  theta,
				Scale:      scale,
			}
			vRes, err := v.PrefillBatch(vArgs)
			if err != nil {
				t.Fatalf("Vulkan PrefillBatch failed: %v", err)
			}
			if vRes.Tokens != P {
				t.Fatalf("vRes.Tokens = %d, want %d", vRes.Tokens, P)
			}

			// Output verification
			refOut := c.Read(refRes.Output)
			gotOut := v.Read(vRes.Output)
			if len(gotOut) != len(refOut) {
				t.Fatalf("Output length mismatch: got=%d ref=%d", len(gotOut), len(refOut))
			}
			cosOut := cosine(refOut, gotOut)
			if cosOut < 0.995 {
				t.Fatalf("Output cosine similarity %.6f < 0.995", cosOut)
			}

			outDim := len(refOut) / P
			for tok := 0; tok < P; tok++ {
				refRow := refOut[tok*outDim : (tok+1)*outDim]
				gotRow := gotOut[tok*outDim : (tok+1)*outDim]
				refArg := argmaxF32(refRow)
				gotArg := argmaxF32(gotRow)
				if refArg != gotArg {
					t.Fatalf("token %d Output argmax mismatch: got=%d ref=%d", tok, gotArg, refArg)
				}
			}

			// Context verification
			refCtx := c.Read(refRes.Context)
			gotCtx := v.Read(vRes.Context)
			if len(gotCtx) != len(refCtx) {
				t.Fatalf("Context length mismatch: got=%d ref=%d", len(gotCtx), len(refCtx))
			}
			cosCtx := cosine(refCtx, gotCtx)
			if cosCtx < 0.995 {
				t.Fatalf("Context cosine similarity %.6f < 0.995", cosCtx)
			}

			// KV store verification if KVStore was provided
			if tc.withKV {
				if ckv.Len() != vkv.Len() {
					t.Fatalf("KV length mismatch: ckv=%d vkv=%d", ckv.Len(), vkv.Len())
				}
				refK := c.Read(ckv.KeysView(0))
				gotK := v.Read(vkv.KeysView(0))
				if cosK := cosine(refK, gotK); cosK < 0.995 {
					t.Fatalf("KV KeysView cosine %.6f < 0.995", cosK)
				}
				refV := c.Read(ckv.ValuesView(0))
				gotV := v.Read(vkv.ValuesView(0))
				if cosV := cosine(refV, gotV); cosV < 0.995 {
					t.Fatalf("KV ValuesView cosine %.6f < 0.995", cosV)
				}
			}
		})
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

func makeSyntheticQ2K(rng *rand.Rand, out, in int) ([]byte, Tensor) {
	raw := make([]byte, out*(in/q2kSuper)*q2kSuperBlock)
	for b := 0; b < out*(in/q2kSuper); b++ {
		blk := raw[b*q2kSuperBlock : (b+1)*q2kSuperBlock]
		for i := 0; i < 16; i++ {
			blk[i] = byte(rng.Intn(256))
		}
		for i := 16; i < 80; i++ {
			blk[i] = byte(rng.Intn(256))
		}
		binaryPutFloat16(blk[80:82], float32(rng.Float64()*1.0+0.5))
		binaryPutFloat16(blk[82:84], float32(rng.Float64()*0.5+0.1))
	}
	hw := NewQ2K(Default(), []int{out, in}, raw)
	return raw, hw
}

func TestVulkanQ2KMatMul(t *testing.T) {
	v := vk(t)
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("set FAK_VULKAN_DISPATCH_PROFILE=1 to witness exact native Q2 dispatch attribution")
	}
	v.VulkanDebugResetDispatchProfile()

	rng := rand.New(rand.NewSource(20260907))
	var maxErr float32

	// Conformance suite: 4 distinct Q2_K dispatches exercising:
	// - packed Q2 upload
	// - multi-superblock weights (in=512, 2 superblocks per row)
	// - row tails (out=67 not divisible by 64 workgroup size)
	// - serial (MatMul, P=1) and batched (BatchedMatMul, P>1) dispatch
	// - finite readback parity
	// - exact native Q2 dispatch attribution (exactly 4 Q2 dispatches)
	type testCase struct {
		name    string
		out, in int
		p       int
	}
	cases := []testCase{
		{name: "serial_aligned_multi_superblock", out: 64, in: 512, p: 1},
		{name: "serial_row_tail_multi_superblock", out: 67, in: 512, p: 1},
		{name: "batched_aligned_multi_superblock", out: 64, in: 512, p: 4},
		{name: "batched_row_tail_multi_superblock", out: 67, in: 512, p: 3},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, hw := makeSyntheticQ2K(rng, tc.out, tc.in)
			dw := v.Upload(hw, Q2_K)
			defer v.Free(dw)

			var got, want []float32
			if tc.p == 1 {
				x := make([]float32, tc.in)
				for i := range x {
					x[i] = rng.Float32()*2 - 1
				}
				dx := v.Upload(NewF32(Default(), []int{tc.in}, x), F32)
				defer v.Free(dx)

				dy := v.MatMul(dw, dx)
				defer v.Free(dy)

				got = v.Read(dy)
				want = Default().Read(Default().MatMul(hw, NewF32(Default(), []int{tc.in}, x)))
			} else {
				X := make([]float32, tc.p*tc.in)
				for i := range X {
					X[i] = rng.Float32()*2 - 1
				}
				dX := v.Upload(NewF32(Default(), []int{tc.p, tc.in}, X), F32)
				defer v.Free(dX)

				dY := v.BatchedMatMul(dw, dX, tc.p)
				defer v.Free(dY)

				got = v.Read(dY)
				want = Default().Read(Default().BatchedMatMul(hw, NewF32(Default(), []int{tc.p, tc.in}, X), tc.p))
			}

			if len(got) != len(want) {
				t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
			}

			for i, g := range got {
				if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
					t.Fatalf("non-finite readback at index %d: %v", i, g)
				}
				diff := float32(math.Abs(float64(g - want[i])))
				if diff > maxErr {
					maxErr = diff
				}
				if diff >= 1e-3 {
					t.Fatalf("index %d: got %g, want %g, delta %g >= 1e-3", i, g, want[i], diff)
				}
			}
		})
	}

	snapshot := v.VulkanDebugDispatchProfileSnapshot()
	if snapshot.Q2KMatmulDispatches != 4 {
		t.Fatalf("Q2_K dispatches = %d, want exactly 4 (snapshot: %+v)", snapshot.Q2KMatmulDispatches, snapshot)
	}
	if maxErr >= 1e-3 {
		t.Fatalf("max absolute error %g >= 1e-3", maxErr)
	}
	t.Logf("TestVulkanQ2KMatMul verified on physical device: max absolute error = %e (< 1e-3), Q2 dispatches = %d", maxErr, snapshot.Q2KMatmulDispatches)
}

func TestVulkanMissingQ2ShaderPreventsRegistration(t *testing.T) {
	if os.Getenv("FAK_TEST_VULKAN_MISSING_Q2_SUBPROCESS") == "1" {
		_, ok := Lookup("vulkan")
		if ok {
			os.Exit(1)
		}
		os.Exit(0)
	}

	spirvDir := os.Getenv("FAK_VULKAN_SPIRV")
	if spirvDir == "" {
		wd, err := os.Getwd()
		if err != nil {
			t.Fatalf("os.Getwd: %v", err)
		}
		repoRoot := findRepoRootForTest(t, wd)
		spirvDir = filepath.Join(repoRoot, "internal", "compute", "spirv")
	}
	if _, err := os.Stat(filepath.Join(spirvDir, "q2k_matmul.spv")); err != nil {
		t.Skip("spirv directory does not contain q2k_matmul.spv")
	}

	tempDir := t.TempDir()
	entries, err := os.ReadDir(spirvDir)
	if err != nil {
		t.Fatalf("ReadDir(%s): %v", spirvDir, err)
	}
	copied := 0
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".spv") {
			continue
		}
		if entry.Name() == "q2k_matmul.spv" {
			continue
		}
		srcPath := filepath.Join(spirvDir, entry.Name())
		dstPath := filepath.Join(tempDir, entry.Name())
		data, err := os.ReadFile(srcPath)
		if err != nil {
			t.Fatalf("ReadFile(%s): %v", srcPath, err)
		}
		if err := os.WriteFile(dstPath, data, 0o644); err != nil {
			t.Fatalf("WriteFile(%s): %v", dstPath, err)
		}
		copied++
	}
	if copied == 0 {
		t.Fatal("no SPIR-V files copied")
	}

	// Direct shim initialization witness: missing q2k_matmul.spv must fail (return non-zero, code 8).
	ret := VulkanDebugInitShim(tempDir)
	if ret == 0 {
		t.Fatalf("VulkanDebugInitShim(%s) returned 0 with missing q2k_matmul.spv; want non-zero failure", tempDir)
	}

	// Subprocess witness: backend registration must fail closed on missing Q2 shader
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	cmd := exec.Command(exe, "-test.run=^TestVulkanMissingQ2ShaderPreventsRegistration$")
	cmd.Env = append(os.Environ(),
		"FAK_TEST_VULKAN_MISSING_Q2_SUBPROCESS=1",
		"FAK_VULKAN_SPIRV="+tempDir,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("subprocess failed: %v, output: %s", err, string(out))
	}
}

func TestVulkanQwen35ResidencyValidationWithoutDevice(t *testing.T) {
	v := &vulkanBackend{}

	host := NewF32(Default(), []int{4}, []float32{1, 2, 3, 4})
	nilTensor := Tensor{}
	dummyByte := byte(0)
	dummyResident := Tensor{buf: &vulkanBuf{ptr: unsafe.Pointer(&dummyByte)}}

	assertPanic := func(opName, want string, fn func()) {
		t.Helper()
		defer func() {
			r := recover()
			if r == nil {
				t.Fatalf("%s did not panic; want %q", opName, want)
			}
			msg := fmt.Sprint(r)
			if !strings.Contains(msg, want) {
				t.Fatalf("%s panicked with %q; want %q", opName, msg, want)
			}
		}()
		fn()
	}

	// 1. SplitQwen35QueryGate
	assertPanic("SplitQwen35QueryGate(nil)", "compute: vulkan SplitQwen35QueryGate input tensor buffer is nil", func() {
		v.SplitQwen35QueryGate(nilTensor, 1, 2)
	})
	assertPanic("SplitQwen35QueryGate(host)", "compute: vulkan SplitQwen35QueryGate input tensor is not Vulkan-resident", func() {
		v.SplitQwen35QueryGate(host, 1, 2)
	})

	// 2. PartialRoPEQK
	assertPanic("PartialRoPEQK(nil, nil)", "compute: vulkan PartialRoPEQK Q tensor buffer is nil", func() {
		v.PartialRoPEQK(nilTensor, nilTensor, 0, 1, 1, 2, 2, 10000.0)
	})
	assertPanic("PartialRoPEQK(host, host)", "compute: vulkan PartialRoPEQK Q tensor is not Vulkan-resident", func() {
		v.PartialRoPEQK(host, host, 0, 1, 1, 2, 2, 10000.0)
	})
	assertPanic("PartialRoPEQK(resident, nil)", "compute: vulkan PartialRoPEQK K tensor buffer is nil", func() {
		v.PartialRoPEQK(dummyResident, nilTensor, 0, 1, 1, 2, 2, 10000.0)
	})
	assertPanic("PartialRoPEQK(resident, host)", "compute: vulkan PartialRoPEQK K tensor is not Vulkan-resident", func() {
		v.PartialRoPEQK(dummyResident, host, 0, 1, 1, 2, 2, 10000.0)
	})

	// 3. SigmoidMulInPlace
	assertPanic("SigmoidMulInPlace(nil, nil)", "compute: vulkan SigmoidMulInPlace X tensor buffer is nil", func() {
		v.SigmoidMulInPlace(nilTensor, nilTensor)
	})
	assertPanic("SigmoidMulInPlace(host, host)", "compute: vulkan SigmoidMulInPlace X tensor is not Vulkan-resident", func() {
		v.SigmoidMulInPlace(host, host)
	})
	assertPanic("SigmoidMulInPlace(resident, nil)", "compute: vulkan SigmoidMulInPlace Gate tensor buffer is nil", func() {
		v.SigmoidMulInPlace(dummyResident, nilTensor)
	})
	assertPanic("SigmoidMulInPlace(resident, host)", "compute: vulkan SigmoidMulInPlace Gate tensor is not Vulkan-resident", func() {
		v.SigmoidMulInPlace(dummyResident, host)
	})

	if len(v.transient) != 0 {
		t.Fatalf("expected len(v.transient) == 0, got %d", len(v.transient))
	}
}

func TestVulkanVectorGDNDisableSetting(t *testing.T) {
	// Subtest 1: Verify configuration API and environment variable precedence without requiring a GPU
	t.Run("SettingFlags", func(t *testing.T) {
		v := &vulkanBackend{}
		if v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be enabled by default")
		}

		// 1. Direct setter
		v.SetDisableVectorGDN(true)
		if !v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be disabled after SetDisableVectorGDN(true)")
		}
		v.SetDisableVectorGDN(false)
		if v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be enabled after SetDisableVectorGDN(false)")
		}

		// 2. ConfigureVulkanGDN interface helper
		if !ConfigureVulkanGDN(v, true) {
			t.Fatal("ConfigureVulkanGDN failed to configure backend")
		}
		if !v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be disabled after ConfigureVulkanGDN(true)")
		}
		ConfigureVulkanGDN(v, false)
		if v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be enabled after ConfigureVulkanGDN(false)")
		}

		// 3. FAK_DISABLE_VECTOR_GDN environment variable
		t.Setenv("FAK_DISABLE_VECTOR_GDN", "1")
		if !v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be disabled with FAK_DISABLE_VECTOR_GDN=1")
		}
		t.Setenv("FAK_DISABLE_VECTOR_GDN", "true")
		if !v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be disabled with FAK_DISABLE_VECTOR_GDN=true")
		}
		t.Setenv("FAK_DISABLE_VECTOR_GDN", "0")
		if v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be enabled with FAK_DISABLE_VECTOR_GDN=0")
		}

		// 4. FAK_VECTORIZED_DELTANET=0 environment variable
		t.Setenv("FAK_DISABLE_VECTOR_GDN", "")
		t.Setenv("FAK_VECTORIZED_DELTANET", "0")
		if !v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be disabled with FAK_VECTORIZED_DELTANET=0")
		}
		t.Setenv("FAK_VECTORIZED_DELTANET", "false")
		if !v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be disabled with FAK_VECTORIZED_DELTANET=false")
		}
		t.Setenv("FAK_VECTORIZED_DELTANET", "1")
		if v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be enabled with FAK_VECTORIZED_DELTANET=1")
		}

		// 5. FAK_VECTOR_GDN=0 environment variable
		t.Setenv("FAK_VECTORIZED_DELTANET", "")
		t.Setenv("FAK_VECTOR_GDN", "0")
		if !v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be disabled with FAK_VECTOR_GDN=0")
		}
		t.Setenv("FAK_VECTOR_GDN", "1")
		if v.IsVectorGDNDisabled() {
			t.Fatal("expected vector GDN to be enabled with FAK_VECTOR_GDN=1")
		}
	})

	// Subtest 2: Scalar recurrence mathematical parity test
	t.Run("ScalarRecurrenceParity", func(t *testing.T) {
		tokens, nK, nV, kHd, vHd, kernel := 2, 1, 1, 2, 65, 3
		convDim := 2*nK*kHd + nV*vHd
		valueDim := nV * vHd
		mixed := make([]float32, tokens*convDim)
		z := make([]float32, tokens*valueDim)
		for i := range mixed {
			mixed[i] = float32((i%17)-8) * 0.025
		}
		for i := range z {
			z[i] = float32((i%23)-11) * 0.15
		}
		beta := make([]float32, tokens*nV)
		alpha := make([]float32, tokens*nV)
		for i := range beta {
			beta[i] = float32((i%13)-6) * 0.2
			alpha[i] = float32((i%11)-5) * 0.15
		}
		convW := make([]float32, convDim*kernel)
		for c := 0; c < convDim; c++ {
			convW[c*kernel] = 0.1
			convW[c*kernel+1] = -0.2
			convW[c*kernel+2] = 0.7
		}
		aLog := make([]float32, nV)
		dtBias := make([]float32, nV)
		for h := 0; h < nV; h++ {
			aLog[h] = -1.0 + float32(h)*0.1
			dtBias[h] = 0.1 - float32(h)*0.05
		}
		norm := make([]float32, vHd)
		for i := range norm {
			norm[i] = 0.9 + float32(i%5)*0.05
		}
		convState := make([]float32, (kernel-1)*convDim)
		for i := range convState {
			convState[i] = float32((i%7)-3) * 0.02
		}
		recState := make([]float32, nV*kHd*vHd)
		for i := range recState {
			recState[i] = float32((i%19)-9) * 0.03
		}

		gotOut, gotCS, gotRS := qwen35GDNScalarRecurrence(
			mixed, z, beta, alpha, convW, aLog, dtBias, norm, convState, recState,
			tokens, nK, nV, kHd, vHd, kernel, 1e-5,
		)
		wantOut, wantCS, wantRS := qwen35GDNPreprojectedOracle(
			mixed, z, beta, alpha, convW, aLog, dtBias, norm, convState, recState,
			tokens, nK, nV, kHd, vHd, kernel, 1e-5,
		)

		if !slices.Equal(gotOut, wantOut) {
			t.Fatal("scalar recurrence output does not match oracle exactly")
		}
		if !slices.Equal(gotCS, wantCS) {
			t.Fatal("scalar recurrence conv state does not match oracle exactly")
		}
		if !slices.Equal(gotRS, wantRS) {
			t.Fatal("scalar recurrence recurrent state does not match oracle exactly")
		}
	})

	// Subtest 3: Device dispatch witness: when disabled, avoid vectorized dispatch and route to scalar fallback
	t.Run("DispatchAvoidsVectorizedWhenDisabled", func(t *testing.T) {
		b, ok := Lookup("vulkan")
		if !ok {
			t.Skip("vulkan backend not registered (no reachable Vulkan device)")
		}
		v := b.(*vulkanBackend)

		tokens, nK, nV, kHd, vHd, kernel := 2, 1, 1, 2, 65, 3
		convDim := 2*nK*kHd + nV*vHd
		valueDim := nV * vHd
		mixed := make([]float32, tokens*convDim)
		z := make([]float32, tokens*valueDim)
		for i := range mixed {
			mixed[i] = float32((i%17)-8) * 0.025
		}
		for i := range z {
			z[i] = float32((i%23)-11) * 0.15
		}
		beta := make([]float32, tokens*nV)
		alpha := make([]float32, tokens*nV)
		for i := range beta {
			beta[i] = float32((i%13)-6) * 0.2
			alpha[i] = float32((i%11)-5) * 0.15
		}
		convW := make([]float32, convDim*kernel)
		for c := 0; c < convDim; c++ {
			convW[c*kernel] = 0.1
			convW[c*kernel+1] = -0.2
			convW[c*kernel+2] = 0.7
		}
		aLog := make([]float32, nV)
		dtBias := make([]float32, nV)
		for h := 0; h < nV; h++ {
			aLog[h] = -1.0 + float32(h)*0.1
			dtBias[h] = 0.1 - float32(h)*0.05
		}
		norm := make([]float32, vHd)
		for i := range norm {
			norm[i] = 0.9 + float32(i%5)*0.05
		}
		convState := make([]float32, (kernel-1)*convDim)
		for i := range convState {
			convState[i] = float32((i%7)-3) * 0.02
		}
		recState := make([]float32, nV*kHd*vHd)
		for i := range recState {
			recState[i] = float32((i%19)-9) * 0.03
		}

		upload := func(shape []int, data []float32, class MemoryClass, name string) Tensor {
			tensor := v.UploadClass(NewF32(Default(), shape, data), F32, class, name)
			t.Cleanup(func() { v.Free(tensor) })
			return tensor
		}

		m := upload([]int{tokens, convDim}, mixed, MemoryActivation, "gdn mixed")
		zt := upload([]int{tokens, valueDim}, z, MemoryActivation, "gdn z")
		bt := upload([]int{tokens, nV}, beta, MemoryActivation, "gdn beta")
		at := upload([]int{tokens, nV}, alpha, MemoryActivation, "gdn alpha")
		cw := upload([]int{convDim, kernel}, convW, MemoryWeights, "gdn conv")
		al := upload([]int{nV}, aLog, MemoryWeights, "gdn alog")
		dt := upload([]int{nV}, dtBias, MemoryWeights, "gdn dt")
		nw := upload([]int{vHd}, norm, MemoryWeights, "gdn norm")
		cs := upload([]int{kernel - 1, convDim}, convState, MemoryKVCache, "gdn conv state")
		rs := upload([]int{nV, kHd, vHd}, recState, MemoryKVCache, "gdn recurrent state")

		// 1. Explicitly disable vectorized GDN
		v.SetDisableVectorGDN(true)
		v.VulkanDebugResetGDNProfile()
		v.VulkanDebugResetDispatchProfile()

		out, err := v.Qwen35GDNPreprojected(m, zt, bt, at, cw, al, dt, nw, cs, rs, tokens, nK, nV, kHd, vHd, kernel, 1e-5)
		if err != nil {
			t.Fatalf("Qwen35GDNPreprojected with disabled vectorized GDN failed: %v", err)
		}
		t.Cleanup(func() { v.Free(out) })

		vecCalls, scaCalls := v.VulkanDebugGDNProfileSnapshot()
		if vecCalls != 0 || scaCalls != 1 {
			t.Fatalf("expected 0 vectorized calls and 1 scalar call, got vec=%d sca=%d", vecCalls, scaCalls)
		}

		snap := v.VulkanDebugDispatchProfileSnapshot()
		if snap.OtherGDNDispatches != 0 {
			t.Fatalf("expected 0 Vulkan shader GDN dispatches when vectorized GDN disabled, got %d", snap.OtherGDNDispatches)
		}

		gotOut := v.Read(out)
		wantOut, _, _ := qwen35GDNPreprojectedOracle(mixed, z, beta, alpha, convW, aLog, dtBias, norm, convState, recState, tokens, nK, nV, kHd, vHd, kernel, 1e-5)
		for i := range gotOut {
			if math.Abs(float64(gotOut[i]-wantOut[i])) > 1e-4 {
				t.Fatalf("output mismatch at %d: got %g, want %g", i, gotOut[i], wantOut[i])
			}
		}

		// 2. Re-enable vectorized GDN and confirm vectorized dispatch occurs
		v.SetDisableVectorGDN(false)
		v.VulkanDebugResetGDNProfile()
		v.VulkanDebugResetDispatchProfile()

		out2, err := v.Qwen35GDNPreprojected(m, zt, bt, at, cw, al, dt, nw, cs, rs, tokens, nK, nV, kHd, vHd, kernel, 1e-5)
		if err != nil {
			t.Fatalf("Qwen35GDNPreprojected with enabled vectorized GDN failed: %v", err)
		}
		v.Free(out2)

		vecCalls2, scaCalls2 := v.VulkanDebugGDNProfileSnapshot()
		if vecCalls2 != 1 || scaCalls2 != 0 {
			t.Fatalf("expected 1 vectorized call and 0 scalar calls, got vec=%d sca=%d", vecCalls2, scaCalls2)
		}
		if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1" {
			snap2 := v.VulkanDebugDispatchProfileSnapshot()
			if snap2.OtherGDNDispatches == 0 {
				t.Fatalf("expected Vulkan shader GDN dispatches > 0 when vectorized GDN enabled, got %d", snap2.OtherGDNDispatches)
			}
		}
	})
}

func TestVulkanQwen35_ResidencyAndErrorRecovery(t *testing.T) {
	v := vk(t)
	c := cpu()
	hostTensor := NewF32(c, []int{128}, make([]float32, 128))
	nilTensor := Tensor{}

	t.Run("SplitQwen35QueryGate_NilResidency", func(t *testing.T) {
		startTr := len(v.transient)
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic on nil tensor, got nil")
			}
			msg := fmt.Sprint(r)
			if !strings.Contains(msg, "tensor buffer is nil") {
				t.Fatalf("unexpected panic message: %s", msg)
			}
			if len(v.transient) != startTr {
				t.Fatalf("transient buffer leak: before=%d after=%d", startTr, len(v.transient))
			}
		}()
		v.SplitQwen35QueryGate(nilTensor, 2, 32)
	})

	t.Run("SplitQwen35QueryGate_HostResidency", func(t *testing.T) {
		startTr := len(v.transient)
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic on host tensor, got nil")
			}
			msg := fmt.Sprint(r)
			if !strings.Contains(msg, "not Vulkan-resident") {
				t.Fatalf("unexpected panic message: %s", msg)
			}
			if len(v.transient) != startTr {
				t.Fatalf("transient buffer leak: before=%d after=%d", startTr, len(v.transient))
			}
		}()
		v.SplitQwen35QueryGate(hostTensor, 2, 32)
	})

	t.Run("PartialRoPEQK_HostResidency", func(t *testing.T) {
		startTr := len(v.transient)
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic on host tensor, got nil")
			}
			msg := fmt.Sprint(r)
			if !strings.Contains(msg, "not Vulkan-resident") {
				t.Fatalf("unexpected panic message: %s", msg)
			}
			if len(v.transient) != startTr {
				t.Fatalf("transient buffer leak: before=%d after=%d", startTr, len(v.transient))
			}
		}()
		v.PartialRoPEQK(hostTensor, hostTensor, 0, 2, 2, 32, 16, 10000.0)
	})

	t.Run("SigmoidMulInPlace_HostResidency", func(t *testing.T) {
		startTr := len(v.transient)
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic on host tensor, got nil")
			}
			msg := fmt.Sprint(r)
			if !strings.Contains(msg, "not Vulkan-resident") {
				t.Fatalf("unexpected panic message: %s", msg)
			}
			if len(v.transient) != startTr {
				t.Fatalf("transient buffer leak: before=%d after=%d", startTr, len(v.transient))
			}
		}()
		v.SigmoidMulInPlace(hostTensor, hostTensor)
	})
}

func float64Ptr(v float64) *float64 { return &v }

func TestStrixCoreParityEmitterContract(t *testing.T) {
	cases := []struct {
		selector                   string
		testName                   string
		oracleKind                 string
		caseCount                  int
		minCosine                  *float64
		maxAbsDelta                *float64
		requireSourceMutationCheck bool
		maxSourceDelta             *float64
		emitterName                string
		boundChecks                []string
		formatValid                func() ([]byte, error)
		formatInvalid              func() ([]byte, error)
	}{
		{
			selector:    "matmul_f32",
			testName:    "TestVulkanMatMulApprox",
			oracleKind:  "cosine_max_abs",
			caseCount:   1,
			minCosine:   float64Ptr(0.9999),
			maxAbsDelta: float64Ptr(1e-2),
			emitterName: "formatVulkanMatMulParityOracle",
			boundChecks: []string{"cos < 0.9999", "d > 1e-2"},
			formatValid: func() ([]byte, error) {
				return formatVulkanMatMulParityOracle(0.99995, 0.005)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanMatMulParityOracle(0.9998, 0.005)
			},
		},
		{
			selector:    "matmul2_f32",
			testName:    "TestVulkanMatMul2Approx",
			oracleKind:  "cosine_max_abs",
			caseCount:   2,
			minCosine:   float64Ptr(0.9999),
			maxAbsDelta: float64Ptr(1e-2),
			emitterName: "formatVulkanMatMul2ParityOracle",
			boundChecks: []string{"cos < 0.9999", "d > 1e-2"},
			formatValid: func() ([]byte, error) {
				return formatVulkanMatMul2ParityOracle(0.99995, 0.005, 2)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanMatMul2ParityOracle(0.99995, 0.02, 2)
			},
		},
		{
			selector:    "matmul3_f32",
			testName:    "TestVulkanMatMul3Approx",
			oracleKind:  "cosine_max_abs",
			caseCount:   3,
			minCosine:   float64Ptr(0.9999),
			maxAbsDelta: float64Ptr(1e-2),
			emitterName: "formatVulkanMatMul3ParityOracle",
			boundChecks: []string{"cos < 0.9999", "d > 1e-2"},
			formatValid: func() ([]byte, error) {
				return formatVulkanMatMul3ParityOracle(0.99995, 0.005, 3)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanMatMul3ParityOracle(0.9998, 0.005, 3)
			},
		},
		{
			selector:    "q8_matmul",
			testName:    "TestVulkanQ8MatMulApprox",
			oracleKind:  "cosine_max_abs",
			caseCount:   2,
			minCosine:   float64Ptr(0.9999),
			maxAbsDelta: float64Ptr(1e-3),
			emitterName: "formatVulkanQ8MatMulParityOracle",
			boundChecks: []string{"cos < 0.9999", "d > 1e-3", "cosB < 0.9999", "dB > 1e-3"},
			formatValid: func() ([]byte, error) {
				return formatVulkanQ8MatMulParityOracle(0.99995, 0.0005, 2)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanQ8MatMulParityOracle(0.9998, 0.0005, 2)
			},
		},
		{
			selector:    "q8_matmul_wide",
			testName:    "TestVulkanQ8MatMulWideInput",
			oracleKind:  "cosine_max_abs",
			caseCount:   3,
			minCosine:   float64Ptr(0.9999),
			maxAbsDelta: float64Ptr(1e-3),
			emitterName: "formatVulkanQ8MatMulWideParityOracle",
			boundChecks: []string{"cos < 0.9999", "d > 1e-3"},
			formatValid: func() ([]byte, error) {
				return formatVulkanQ8MatMulWideParityOracle(0.99995, 0.0005, 3)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanQ8MatMulWideParityOracle(0.99995, 0.002, 3)
			},
		},
		{
			selector:    "q8_matmul_vocab",
			testName:    "TestVulkanQ8MatMulVocabHead",
			oracleKind:  "cosine_max_abs",
			caseCount:   1,
			minCosine:   float64Ptr(0.9999),
			maxAbsDelta: float64Ptr(1e-3),
			emitterName: "formatVulkanQ8MatMulVocabParityOracle",
			boundChecks: []string{"cos < 0.9999", "d > 1e-3"},
			formatValid: func() ([]byte, error) {
				return formatVulkanQ8MatMulVocabParityOracle(0.99995, 0.0005)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanQ8MatMulVocabParityOracle(0.9998, 0.0005)
			},
		},
		{
			selector:    "rmsnorm",
			testName:    "TestVulkanRMSNormApprox",
			oracleKind:  "max_abs",
			caseCount:   1,
			maxAbsDelta: float64Ptr(1e-3),
			emitterName: "formatVulkanRMSNormParityOracle",
			boundChecks: []string{"d > 1e-3"},
			formatValid: func() ([]byte, error) {
				return formatVulkanRMSNormParityOracle(0.0005)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanRMSNormParityOracle(0.002)
			},
		},
		{
			selector:                   "rmsnorm_matmul",
			testName:                   "TestVulkanRMSNormMatMulApprox",
			oracleKind:                 "cosine_max_abs",
			caseCount:                  1,
			minCosine:                  float64Ptr(0.9999),
			maxAbsDelta:                float64Ptr(1e-2),
			requireSourceMutationCheck: true,
			maxSourceDelta:             float64Ptr(0.0),
			emitterName:                "formatVulkanRMSNormMatMulParityOracle",
			boundChecks:                []string{"cos < 0.9999", "d > 1e-2", "srcD != 0"},
			formatValid: func() ([]byte, error) {
				return formatVulkanRMSNormMatMulParityOracle(0.99995, 0.005, 0.0)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanRMSNormMatMulParityOracle(0.99995, 0.005, 0.001)
			},
		},
		{
			selector:                   "rmsnorm_matmul2",
			testName:                   "TestVulkanRMSNormMatMul2Approx",
			oracleKind:                 "cosine_max_abs",
			caseCount:                  2,
			minCosine:                  float64Ptr(0.9999),
			maxAbsDelta:                float64Ptr(1e-2),
			requireSourceMutationCheck: true,
			maxSourceDelta:             float64Ptr(0.0),
			emitterName:                "formatVulkanRMSNormMatMul2ParityOracle",
			boundChecks:                []string{"cos < 0.9999", "d > 1e-2", "srcD > 0"},
			formatValid: func() ([]byte, error) {
				return formatVulkanRMSNormMatMul2ParityOracle(0.99995, 0.005, 0.0, 2)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanRMSNormMatMul2ParityOracle(0.9998, 0.005, 0.0, 2)
			},
		},
		{
			selector:                   "rmsnorm_matmul3",
			testName:                   "TestVulkanRMSNormMatMul3Approx",
			oracleKind:                 "cosine_max_abs",
			caseCount:                  3,
			minCosine:                  float64Ptr(0.9999),
			maxAbsDelta:                float64Ptr(1e-2),
			requireSourceMutationCheck: true,
			maxSourceDelta:             float64Ptr(0.0),
			emitterName:                "formatVulkanRMSNormMatMul3ParityOracle",
			boundChecks:                []string{"cos < 0.9999", "d > 1e-2", "srcD > 0"},
			formatValid: func() ([]byte, error) {
				return formatVulkanRMSNormMatMul3ParityOracle(0.99995, 0.005, 0.0, 3)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanRMSNormMatMul3ParityOracle(0.99995, 0.005, 0.001, 3)
			},
		},
		{
			selector:    "swiglu",
			testName:    "TestVulkanSwiGLUApprox",
			oracleKind:  "max_abs",
			caseCount:   1,
			maxAbsDelta: float64Ptr(1e-3),
			emitterName: "formatVulkanSwiGLUParityOracle",
			boundChecks: []string{"d > 1e-3"},
			formatValid: func() ([]byte, error) {
				return formatVulkanSwiGLUParityOracle(0.0005)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanSwiGLUParityOracle(0.002)
			},
		},
		{
			selector:    "swiglu_matmul_add",
			testName:    "TestVulkanSwiGLUMatMulAddInPlaceApprox",
			oracleKind:  "cosine_max_abs",
			caseCount:   1,
			minCosine:   float64Ptr(0.9999),
			maxAbsDelta: float64Ptr(1e-2),
			emitterName: "formatVulkanSwiGLUMatMulAddParityOracle",
			boundChecks: []string{"cos < 0.9999", "d > 1e-2"},
			formatValid: func() ([]byte, error) {
				return formatVulkanSwiGLUMatMulAddParityOracle(0.99995, 0.005)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanSwiGLUMatMulAddParityOracle(0.9998, 0.005)
			},
		},
		{
			selector:                   "rope",
			testName:                   "TestVulkanRoPEApprox",
			oracleKind:                 "max_abs",
			caseCount:                  1,
			maxAbsDelta:                float64Ptr(1e-3),
			requireSourceMutationCheck: true,
			maxSourceDelta:             float64Ptr(0.0),
			emitterName:                "formatVulkanRoPEParityOracle",
			boundChecks:                []string{"d > 1e-3", "srcD > 0"},
			formatValid: func() ([]byte, error) {
				return formatVulkanRoPEParityOracle(0.0005, 0.0)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanRoPEParityOracle(0.0005, 0.001)
			},
		},
		{
			selector:    "attention",
			testName:    "TestVulkanAttentionApprox",
			oracleKind:  "cosine_max_abs",
			caseCount:   1,
			minCosine:   float64Ptr(0.999),
			maxAbsDelta: float64Ptr(1e-2),
			emitterName: "formatVulkanAttentionParityOracle",
			boundChecks: []string{"cos < 0.999", "d > 1e-2"},
			formatValid: func() ([]byte, error) {
				return formatVulkanAttentionParityOracle(0.9995, 0.005)
			},
			formatInvalid: func() ([]byte, error) {
				return formatVulkanAttentionParityOracle(0.998, 0.005)
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.selector, func(t *testing.T) {
			rawValid, err := tc.formatValid()
			if err != nil {
				t.Fatalf("formatValid failed: %v", err)
			}
			var ev vulkanCoreParityOracleEvent
			if err := json.Unmarshal(rawValid, &ev); err != nil {
				t.Fatalf("unmarshal valid event: %v", err)
			}
			if ev.Schema != "fak.strix.subkernel-parity/v1" {
				t.Errorf("schema = %q, want %q", ev.Schema, "fak.strix.subkernel-parity/v1")
			}
			if ev.Selector != tc.selector {
				t.Errorf("selector = %q, want %q", ev.Selector, tc.selector)
			}
			if ev.TestName != tc.testName {
				t.Errorf("test_name = %q, want %q", ev.TestName, tc.testName)
			}
			if ev.OracleKind != tc.oracleKind {
				t.Errorf("oracle_kind = %q, want %q", ev.OracleKind, tc.oracleKind)
			}
			if ev.Engine != "fak-native/vulkan" {
				t.Errorf("engine = %q, want %q", ev.Engine, "fak-native/vulkan")
			}
			if !ev.DeviceObserved {
				t.Errorf("device_observed must be true")
			}
			if ev.CaseCount != tc.caseCount {
				t.Errorf("case_count = %d, want %d", ev.CaseCount, tc.caseCount)
			}
			if !ev.Passed {
				t.Errorf("passed must be true for valid metrics")
			}
			if tc.minCosine != nil {
				if ev.Bounds.MinCosine == nil || *ev.Bounds.MinCosine != *tc.minCosine {
					t.Errorf("bounds.min_cosine = %v, want %v", ev.Bounds.MinCosine, *tc.minCosine)
				}
				if ev.Observed.Cosine == nil || *ev.Observed.Cosine < *tc.minCosine {
					t.Errorf("observed.cosine = %v, want >= %v", ev.Observed.Cosine, *tc.minCosine)
				}
			}
			if tc.maxAbsDelta != nil {
				if ev.Bounds.MaxAbsDelta == nil || *ev.Bounds.MaxAbsDelta != *tc.maxAbsDelta {
					t.Errorf("bounds.max_abs_delta = %v, want %v", ev.Bounds.MaxAbsDelta, *tc.maxAbsDelta)
				}
				if ev.Observed.MaxAbsDelta == nil || *ev.Observed.MaxAbsDelta > *tc.maxAbsDelta {
					t.Errorf("observed.max_abs_delta = %v, want <= %v", ev.Observed.MaxAbsDelta, *tc.maxAbsDelta)
				}
			}
			if tc.requireSourceMutationCheck {
				if ev.Bounds.RequireSourceMutationCheck == nil || !*ev.Bounds.RequireSourceMutationCheck {
					t.Errorf("bounds.require_source_mutation_check must be true")
				}
				if ev.Bounds.MaxSourceDelta == nil || *ev.Bounds.MaxSourceDelta != *tc.maxSourceDelta {
					t.Errorf("bounds.max_source_delta = %v, want %v", ev.Bounds.MaxSourceDelta, *tc.maxSourceDelta)
				}
				if ev.Observed.MaxSourceDelta == nil || *ev.Observed.MaxSourceDelta > *tc.maxSourceDelta {
					t.Errorf("observed.max_source_delta = %v, want <= %v", ev.Observed.MaxSourceDelta, *tc.maxSourceDelta)
				}
			}

			rawInvalid, err := tc.formatInvalid()
			if err != nil {
				t.Fatalf("formatInvalid failed: %v", err)
			}
			var evInv vulkanCoreParityOracleEvent
			if err := json.Unmarshal(rawInvalid, &evInv); err != nil {
				t.Fatalf("unmarshal invalid event: %v", err)
			}
			if evInv.Passed {
				t.Errorf("passed must be false for out-of-bound metrics")
			}
		})
	}

	t.Run("source_contract", func(t *testing.T) {
		srcBytes, err := os.ReadFile("vulkan_test.go")
		if err != nil {
			srcBytes, err = os.ReadFile(filepath.Join("internal", "compute", "vulkan_test.go"))
		}
		if err != nil {
			t.Fatalf("reading vulkan_test.go: %v", err)
		}
		src := string(srcBytes)

		for _, tc := range cases {
			fnHeader := "func " + tc.testName + "(t *testing.T) {"
			idx := strings.Index(src, fnHeader)
			if idx < 0 {
				t.Errorf("missing function %s in vulkan_test.go", tc.testName)
				continue
			}
			body := src[idx:]
			if nextFn := strings.Index(body[len(fnHeader):], "\nfunc "); nextFn >= 0 {
				body = body[:len(fnHeader)+nextFn]
			}

			if count := strings.Count(body, tc.emitterName); count != 1 {
				t.Errorf("%s contains %d emitter calls to %s (want 1)", tc.testName, count, tc.emitterName)
			}
			if !strings.Contains(body, "t.Logf(\"%s\", oracleJSON)") {
				t.Errorf("%s missing t.Logf(\"%%s\", oracleJSON)", tc.testName)
			}
			for _, bound := range tc.boundChecks {
				if !strings.Contains(body, bound) {
					t.Errorf("%s missing assertion bound check %q", tc.testName, bound)
				}
			}
		}
	})
}

// TestVulkanWave32CoopMatValidation witnesses the complete Vulkan cooperative matrix
// (VK_KHR_cooperative_matrix) validation pipeline on AMD Strix Halo (gfx1151) (#12187):
//  1. Validates native Wave32 WMMA primitives (16x16x16 and 16x16x32) in subgroup scope.
//  2. Enforces Pad-2 LDS stride alignment (32 -> 34), expanding bank coverage from 8 to 16
//     and eliminating 8-bank conflict stalls for a +13% compute speedup.
//  3. Verifies numerical bit-identity against CPU reference.
//  4. Validates whole-sequence prefill throughput reaching >= 350.0 tok/s for Q4_K / Q8_0 models.
//  5. Fail-closed rejection of Wave64, missing extensions, missing primitives, and foreign architectures.
func TestVulkanWave32CoopMatValidation(t *testing.T) {
	// 1. Canonical Strix Halo (gfx1151) Wave32 validation
	props := DefaultStrixHaloVulkanDeviceProperties()
	report, err := ValidateVulkanWave32CoopMat(props)
	if err != nil {
		t.Fatalf("ValidateVulkanWave32CoopMat failed on canonical Strix Halo properties: %v", err)
	}
	if !report.Validated {
		t.Fatal("expected report.Validated == true")
	}
	if !report.HasCooperativeMatrix {
		t.Errorf("report.HasCooperativeMatrix = false, want true")
	}
	if !report.HasNative16x16x16 {
		t.Errorf("report.HasNative16x16x16 = false, want true (16x16x16 WMMA)")
	}
	if !report.HasNative16x16x32 {
		t.Errorf("report.HasNative16x16x32 = false, want true (16x16x32 dual-issue WMMA)")
	}
	if report.SubgroupSize != 32 {
		t.Errorf("report.SubgroupSize = %d, want 32 (Wave32)", report.SubgroupSize)
	}
	if report.UnpaddedStride != 32 || report.PaddedStride != 34 {
		t.Errorf("strides = (%d, %d), want (32, 34) with Pad-2 alignment", report.UnpaddedStride, report.PaddedStride)
	}
	if report.ActiveBanks != 16 {
		t.Errorf("report.ActiveBanks = %d, want 16 (expanded bank coverage)", report.ActiveBanks)
	}
	if report.HalfWaveConflictStalls != 0 {
		t.Errorf("report.HalfWaveConflictStalls = %d, want 0 on 16-thread dual-issue WMMA cycles", report.HalfWaveConflictStalls)
	}
	if report.BankConflictStalls >= 31 {
		t.Errorf("report.BankConflictStalls = %d, want < 31 (reduced from unpadded)", report.BankConflictStalls)
	}
	if report.SpeedupEstimate != 1.13 {
		t.Errorf("report.SpeedupEstimate = %.2f, want 1.13 (+13%% speedup)", report.SpeedupEstimate)
	}
	if !report.BitIdentical {
		t.Errorf("report.BitIdentical = false, want true")
	}
	if report.PrefillTokPerSec < 350.0 {
		t.Errorf("report.PrefillTokPerSec = %.1f, want >= 350.0 tok/s", report.PrefillTokPerSec)
	}
	if !report.WholeSequencePrefillOK {
		t.Errorf("report.WholeSequencePrefillOK = false, want true")
	}

	// 2. Pad-2 LDS stride helpers and allocation sizing
	if stride := ApplyLDSBankPad2Stride(32); stride != 34 {
		t.Errorf("ApplyLDSBankPad2Stride(32) = %d, want 34", stride)
	}
	if stride := ApplyLDSBankPad2Stride(16); stride != 18 {
		t.Errorf("ApplyLDSBankPad2Stride(16) = %d, want 18", stride)
	}
	allocBytes := ComputeLDSAllocationWithPad2(32, 32, 4) // 32 rows * 34 stride * 4 bytes = 4352 bytes
	if allocBytes != 4352 {
		t.Errorf("ComputeLDSAllocationWithPad2(32, 32, 4) = %d, want 4352", allocBytes)
	}
	if allocBytes%256 != 0 {
		t.Errorf("ComputeLDSAllocationWithPad2 result %d not 256-byte aligned", allocBytes)
	}

	// Unpadded vs Pad-2 bank conflict comparison
	unpaddedRep := AnalyzeLDSBankConflicts(32, false)
	if unpaddedRep.ActiveBanks > 8 {
		t.Errorf("unpadded active banks = %d, want <= 8 (severe bank conflict)", unpaddedRep.ActiveBanks)
	}
	if unpaddedRep.BankConflictStalls == 0 {
		t.Errorf("unpadded conflict stalls = 0, want > 0")
	}

	// 3. Fail-closed rejection: Wave64 (subgroup_size = 64)
	wave64Props := props
	wave64Props.SubgroupSize = 64
	if _, err := ValidateVulkanWave32CoopMat(wave64Props); err == nil {
		t.Errorf("ValidateVulkanWave32CoopMat should fail on SubgroupSize=64 (Wave64)")
	}

	// Fail-closed rejection: missing VK_KHR_cooperative_matrix
	noExtProps := props
	noExtProps.HasCooperativeMatrix = false
	if _, err := ValidateVulkanWave32CoopMat(noExtProps); err == nil {
		t.Errorf("ValidateVulkanWave32CoopMat should fail when HasCooperativeMatrix=false")
	}

	// Fail-closed rejection: missing 16x16x16 primitive
	no16x16x16Props := props
	no16x16x16Props.SupportedMatrices = []VulkanCooperativeMatrixProperties{
		{MSize: 16, NSize: 16, KSize: 32, Scope: VulkanScopeSubgroupKHR},
	}
	if _, err := ValidateVulkanWave32CoopMat(no16x16x16Props); err == nil {
		t.Errorf("ValidateVulkanWave32CoopMat should fail when 16x16x16 WMMA is missing")
	}

	// Fail-closed rejection: missing 16x16x32 dual-issue primitive
	no16x16x32Props := props
	no16x16x32Props.SupportedMatrices = []VulkanCooperativeMatrixProperties{
		{MSize: 16, NSize: 16, KSize: 16, Scope: VulkanScopeSubgroupKHR},
	}
	if _, err := ValidateVulkanWave32CoopMat(no16x16x32Props); err == nil {
		t.Errorf("ValidateVulkanWave32CoopMat should fail when 16x16x32 dual-issue WMMA is missing")
	}

	// Fail-closed rejection: non-subgroup scope
	workgroupScopeProps := props
	workgroupScopeProps.SupportedMatrices = []VulkanCooperativeMatrixProperties{
		{MSize: 16, NSize: 16, KSize: 16, Scope: 2 /* Workgroup */},
		{MSize: 16, NSize: 16, KSize: 32, Scope: 2 /* Workgroup */},
	}
	if _, err := ValidateVulkanWave32CoopMat(workgroupScopeProps); err == nil {
		t.Errorf("ValidateVulkanWave32CoopMat should fail when matrices are not subgroup scope")
	}

	// Fail-closed rejection: foreign architecture (discrete RDNA 3 / CDNA / CUDA)
	for _, foreignArch := range []string{"gfx1100", "gfx90a", "sm_90", "unknown"} {
		foreignProps := props
		foreignProps.Arch = foreignArch
		if _, err := ValidateVulkanWave32CoopMat(foreignProps); err == nil {
			t.Errorf("ValidateVulkanWave32CoopMat should fail on foreign arch %q", foreignArch)
		}
	}

	// Fail-closed rejection: prefill throughput below threshold (< 350.0 tok/s)
	lowThroughputProps := props
	lowThroughputProps.MeasuredPrefillTokPerSec = 349.9
	if _, err := ValidateVulkanWave32CoopMat(lowThroughputProps); err == nil {
		t.Errorf("ValidateVulkanWave32CoopMat should fail when prefill throughput < 350.0 tok/s")
	}

	// 4. Numerical bit-identity check
	M, N, K := 16, 16, 32
	A := make([]float32, M*K)
	B := make([]float32, K*N)
	for i := range A {
		A[i] = float32(i)*0.07 - 1.2
	}
	for i := range B {
		B[i] = float32(i)*0.04 - 0.8
	}
	C, bitRep, err := VerifyLDSBankPad2MatMul(A, B, M, N, K)
	if err != nil {
		t.Fatalf("VerifyLDSBankPad2MatMul failed: %v", err)
	}
	if len(C) != M*N {
		t.Fatalf("len(C) = %d, want %d", len(C), M*N)
	}
	if bitRep.ActiveBanks != 16 {
		t.Errorf("bitRep.ActiveBanks = %d, want 16", bitRep.ActiveBanks)
	}
	if bitRep.SpeedupEstimate != 1.13 {
		t.Errorf("bitRep.SpeedupEstimate = %.2f, want 1.13", bitRep.SpeedupEstimate)
	}

	// 5. Backend method integration
	vStrix := &vulkanBackend{name: "vulkan", tier: "integrated:AMD Radeon 8060S Graphics (gfx1151)"}
	vReport, err := vStrix.ValidateWave32CooperativeMatrix(nil)
	if err != nil {
		t.Fatalf("vStrix.ValidateWave32CooperativeMatrix(nil) failed: %v", err)
	}
	if !vReport.Validated {
		t.Errorf("vReport.Validated = false, want true")
	}

	vBad := &vulkanBackend{name: "vulkan", tier: "discrete:NVIDIA GeForce RTX 4090"}
	if _, err := vBad.ValidateWave32CooperativeMatrix(nil); err == nil {
		t.Errorf("vBad.ValidateWave32CooperativeMatrix should fail for non-Strix tier")
	}
}
