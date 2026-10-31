//go:build cuda && cgo && fak_cuda_capture_once_qualification && fak_cuda_capture_once_device_witness

package compute

import (
	"fmt"
	"os"
	"runtime"
	"strings"
	"testing"
)

// fak-test:runtime integration est=3s lane=cuda
// Uses actual native CUDA capture/MatMul/abort, not the host recording fixtures.
func TestCUDAGraphOnceRequiredBorrowedLifetime(t *testing.T) {
	for _, key := range []string{"FAK_CUDA_REQUIRE_DEVICE", "FAK_CUDA_GRAPH", "FAK_CUDA_CAPTURE_ONCE"} {
		if os.Getenv(key) != "1" {
			t.Fatalf("%s=1 required; no skip/fallback", key)
		}
	}
	be, ok := Lookup("cuda")
	if !ok {
		t.Fatal("real CUDA backend missing")
	}
	cb, ok := be.(*cudaBackend)
	if !ok || !cb.Caps().DeviceMemory || !cb.Caps().GraphCompile {
		t.Fatal("actual graph-enabled CUDA backend required")
	}
	if cb.GraphOnceQualificationSnapshot().Owner != 0 {
		t.Fatal("another graph owner is live")
	}
	wdata := make([]float32, 64*64)
	for i := 0; i < 64; i++ {
		wdata[i*64+i] = 1
	}
	w := mkResident(cb, []int{64, 64}, wdata)
	defer cb.Free(w)
	input := make([]float32, 64)
	for i := range input {
		input[i] = float32(i+1) / 64
	}
	x := mkResident(cb, []int{64}, input)
	defer cb.Free(x)
	kv := cb.NewKV(KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 64, RopeTheta: 10000, Precision: KVPrecisionF32, GraphCapture: true})
	defer kv.Free()
	ck, ok := kv.(*cudaKV)
	if !ok || !ck.GraphCaptureReady() {
		t.Fatal("fixed real CUDA KV required")
	}
	cb.Read(cb.MatMul(w, x))
	cb.Recycle() // Warm the actual output allocation/BLAS path.
	before := cb.GraphOnceQualificationSnapshot()
	if cb.GraphOnceBegin(x, kv, ck.maxPos) {
		t.Fatal("out-of-bounds begin accepted")
	}
	if cb.GraphOnceQualificationSnapshot().Owner != 0 {
		t.Fatal("bounds refusal created owner")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	active := false
	defer func() {
		if active {
			cb.GraphOnceAbort()
		}
	}()
	if !cb.GraphOnceBegin(x, kv, 0) {
		t.Fatal("real CUDA capture did not start")
	}
	active = true
	cb.MatMul(w, x)           // Records w as an actual borrowed kernel operand.
	foreign := &cudaBackend{} // A second real backend receiver; never a mock or a device allocator.
	recoverPanic := func(f func()) (v any) { defer func() { v = recover() }(); f(); return }
	original := recoverPanic(func() { foreign.Free(w) })
	if original == nil || !strings.Contains(fmt.Sprint(original), "freeing a captured borrowed weight") {
		t.Fatalf("wrong borrowed-Free refusal: %v", original)
	}
	state := cb.GraphOnceQualificationSnapshot()
	if !state.Invalidated || !state.Capturing || !state.SharedOwner || state.Owner == 0 || state.Captures != before.Captures || state.Replays != before.Replays || w.buf.(*cudaBuf).ptr == nil {
		t.Fatal("borrowed Free lost owner/storage or launched")
	}
	refused := recoverPanic(cb.GraphOnceEnd)
	if refused == nil || !strings.Contains(fmt.Sprint(refused), "invalidated before initial launch") {
		t.Fatalf("wrong initial-launch refusal: %v", refused)
	}
	cb.GraphOnceAbort()
	active = false // Checked real native abort; failures are never swallowed.
	after := cb.GraphOnceQualificationSnapshot()
	if after.Owner != 0 || after.Captures != before.Captures || after.Replays != before.Replays || w.buf.(*cudaBuf).ptr == nil {
		t.Fatal("abort launched or freed borrowed weight")
	}
	cb.Free(w)
	if w.buf.(*cudaBuf).ptr != nil {
		t.Fatal("borrowed weight did not release after checked teardown")
	}
	cb.Recycle()
	t.Logf("actual CUDA borrowed-Free refusal retained owner until checked abort; original refusal=%v", original)
}
