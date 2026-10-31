//go:build cuda && cgo && fak_cuda_capture_once_qualification && fak_cuda_capture_once_capacity_witness

package compute

import (
	"math"
	"os"
	"reflect"
	"runtime"
	"testing"
)

// A separate isolated process exercises a real final KV row, not fabricated metadata.
// fak-test:runtime integration est=5s lane=cuda
func TestCUDAGraphOnceRequiredLastCapacity(t *testing.T) {
	for _, key := range []string{"FAK_CUDA_REQUIRE_DEVICE", "FAK_CUDA_GRAPH", "FAK_CUDA_CAPTURE_ONCE"} {
		if os.Getenv(key) != "1" {
			t.Fatalf("%s=1 required; no skip/fallback", key)
		}
	}
	// Setenv prevents this test or a parent from being marked parallel. Value unchanged.
	t.Setenv("FAK_CUDA_CAPTURE_ONCE", "1")
	be, ok := Lookup("cuda")
	if !ok {
		t.Fatal("real CUDA backend required")
	}
	cb, ok := be.(*cudaBackend)
	if !ok || !cb.Caps().DeviceMemory || !cb.Caps().GraphCompile || !cudaAsyncGraphOnceAllowed() {
		t.Fatal("real compatible graph profile required")
	}
	if cb.GraphOnceQualificationSnapshot().Owner != 0 {
		t.Fatal("isolated process with no live graph owner required")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	a, b, z := make([]float32, 64), make([]float32, 64), make([]float32, 64)
	wdata := make([]float32, 64*64)
	for i := range a {
		a[i] = float32(i+1) / 64
		b[i] = -float32(i+1) / 32
		z[i] = float32(i+1) / 16
		wdata[i*64+i] = 1
	}
	row := mkResident(cb, []int{64}, a)
	defer cb.Free(row)
	x := mkResident(cb, []int{64}, b)
	defer cb.Free(x)
	fresh := mkResident(cb, []int{64}, z)
	defer cb.Free(fresh)
	weight := mkResident(cb, []int{64, 64}, wdata)
	defer cb.Free(weight)
	kv := cb.NewKV(KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 64, RopeTheta: 10000, Precision: KVPrecisionF32, GraphCapture: true})
	defer kv.Free()
	k, ok := kv.(*cudaKV)
	if !ok || !k.GraphCaptureReady() || k.maxPos < 2 || k.maxPos > 4096 {
		t.Fatal("actual fixed capacity must be in [2,4096]; do not alter globals to fit")
	}
	capacity := k.maxPos
	for pos := 0; pos < capacity-1; pos++ {
		kv.AppendKV(0, row, row, row, pos)
	}
	cb.Read(cb.MatMul(weight, x))
	cb.Recycle() // Real warmup and completion before capture.
	if len(k.pos) != capacity-1 || k.K[0].len != (capacity-1)*64 {
		t.Fatal("prefill did not create actual penultimate state")
	}
	before := cb.GraphOnceQualificationSnapshot()
	active := false
	defer func() {
		if active {
			cb.GraphOnceAbort()
		} else {
			cb.GraphOnceInvalidate()
		}
	}()
	if !cb.GraphOnceBegin(x, kv, capacity-1) {
		t.Fatal("last valid row refused before capture")
	}
	active = true
	output := cb.MatMul(weight, x)
	kv.AppendKV(0, x, x, x, capacity-1)
	cb.GraphOnceEnd()
	active = false
	cb.GraphOnceSave(output)
	if got := cb.Read(output); !reflect.DeepEqual(got, b) {
		t.Fatal("real identity output differs")
	}
	full, err := k.SnapshotToHost()
	if err != nil {
		t.Fatal(err)
	}
	if len(full.Pos) != capacity || full.Pos[capacity-1] != capacity-1 {
		t.Fatal("last row metadata not published")
	}
	for _, layers := range [][][]float32{full.K, full.KRaw, full.V} {
		if len(layers) != 1 || len(layers[0]) != capacity*64 {
			t.Fatal("unexpected fixed KV extent")
		}
		for pos := 0; pos < capacity; pos++ {
			want := a
			if pos == capacity-1 {
				want = b
			}
			if full.Pos[pos] != pos {
				t.Fatal("nonsequential KV position")
			}
			for j := 0; j < 64; j++ {
				if math.Float32bits(layers[0][pos*64+j]) != math.Float32bits(want[j]) {
					t.Fatalf("wrong actual row at %d/%d", pos, j)
				}
			}
		}
	}
	// Prove every non-capacity replay prerequisite without weakening/mutating state.
	valid := func() bool {
		cudaMu.Lock()
		defer cudaMu.Unlock()
		if cb.faultLatch.Snapshot().Health.Refusing() {
			return false
		}
		s := cb.graphOnce
		if s == nil || s.kv != k || s.next != capacity || len(k.pos) != capacity || s.capturing || s.invalidated || s.output.buf == nil || cudaGraphOnceOwner != cb || !graphEnabled || cudaKernelProfileOn {
			return false
		}
		if fresh.Backend() != cb || fresh.Dtype != F32 || fresh.Numel() != s.input.Numel() {
			return false
		}
		src, dst := fresh.buf.(*cudaBuf), s.input.buf.(*cudaBuf)
		if src.ptr == nil || dst.ptr == nil || src.n != dst.n || src.device != 0 || src.managed || src.invalidForSubmit() || dst.invalidForSubmit() {
			return false
		}
		if len(k.K) != len(s.keys) || len(k.Kraw) != len(s.raw) || len(k.V) != len(s.values) {
			return false
		}
		for l := range k.K {
			for _, pair := range [][2]dslice{{k.K[l], s.keys[l]}, {k.Kraw[l], s.raw[l]}, {k.V[l], s.values[l]}} {
				q, r := pair[0], pair[1]
				if q.ptr == nil || q.ptr != r.ptr || q.cap != r.cap || q.dtype != r.dtype || q.len != capacity*64 {
					return false
				}
			}
		}
		for q := range s.borrowed {
			if q.ptr == nil || q.invalidForSubmit() {
				return false
			}
		}
		return true
	}()
	if !valid {
		t.Fatal("non-capacity replay prerequisite invalid; cannot attribute refusal to capacity")
	}
	ready := cb.GraphOnceQualificationSnapshot()
	if ready.Captures != before.Captures+1 || ready.Replays != before.Replays || ready.Next != capacity {
		t.Fatal("last valid capture not actually completed exactly once")
	}
	if _, used := cb.GraphOnceReplay(fresh, kv, capacity); used {
		t.Fatal("one-past-capacity replay accepted")
	}
	after := cb.GraphOnceQualificationSnapshot()
	if after.Owner != 0 || len(after.Arena) != 0 || after.Captures != ready.Captures || after.Replays != ready.Replays {
		t.Fatal("capacity refusal launched or retained graph owner")
	}
	unchanged, err := k.SnapshotToHost()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(full, unchanged) {
		t.Fatal("capacity refusal changed complete valid KV state")
	}
	t.Logf("real last-valid row %d captured; one-past %d refused with all other replay prerequisites valid; no throughput claim", capacity-1, capacity)
}
