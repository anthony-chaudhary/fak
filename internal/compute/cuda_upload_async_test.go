//go:build cuda && cgo

package compute

import (
	"math"
	"testing"
)

// cuda_upload_async_test.go — the `-tags cuda` witness for spec-prefetch TICKET-02: the CUDA
// compute.AsyncUploader. Each test compares the async upload against the synchronous Upload of the
// SAME host bytes on the SAME backend, so a torn or early-read copy shows up as a bit mismatch.

func asyncUploadIssuedOrFatal(t *testing.T, before uint64) {
	t.Helper()
	issued, _, pinned := cudaAsyncUploadStats()
	if issued <= before {
		t.Fatalf("no copy went through the copy stream (issued %d -> %d); the async path was never taken", before, issued)
	}
	if pinned <= 0 || pinned > 8*(64<<20) {
		t.Fatalf("pinned staging bytes=%d, want within (0, 512 MiB]", pinned)
	}
}

// fak-test:runtime fast est=2s
func TestCUDAUploadAsyncF32BitEqualToUpload(t *testing.T) {
	cb := cudaOrSkip(t)
	var seed lcg = 0xa5a5
	const out, in = 384, 512
	w := randVec(&seed, out*in)
	host := NewF32(cb, []int{out, in}, w)

	before, _, _ := cudaAsyncUploadStats()
	got, fence := cb.UploadAsync(host, F32)
	fence.Wait()
	fence.Wait() // the contract requires Wait to be safe more than once
	if !fence.Done() {
		t.Fatal("Done()=false after Wait")
	}
	asyncUploadIssuedOrFatal(t, before)

	if got.Dtype != F32 || got.Numel() != out*in {
		t.Fatalf("UploadAsync produced %s numel=%d, want f32 numel=%d", got.Dtype, got.Numel(), out*in)
	}
	back := cb.Read(got)
	for i := range w {
		if math.Float32bits(back[i]) != math.Float32bits(w[i]) {
			t.Fatalf("element %d: async=%08x host=%08x", i, math.Float32bits(back[i]), math.Float32bits(w[i]))
		}
	}
	cb.Free(got)
}

// A GEMM enqueued after the fence reads the new weight: identical output to the synchronous upload.
// fak-test:runtime fast est=2s
func TestCUDAUploadAsyncGEMMAfterFenceMatchesUpload(t *testing.T) {
	cb := cudaOrSkip(t)
	var seed lcg = 0x5a5a
	const out, in = 256, 256
	w := randVec(&seed, out*in)
	x := mkResident(cb, []int{in}, randVec(&seed, in))

	ref := cb.Read(cb.MatMul(cb.Upload(NewF32(cb, []int{out, in}, append([]float32(nil), w...)), F32), x))

	got, fence := cb.UploadAsync(NewF32(cb, []int{out, in}, append([]float32(nil), w...)), F32)
	fence.Wait()
	y := cb.Read(cb.MatMul(got, x))
	for i := range ref {
		if math.Float32bits(y[i]) != math.Float32bits(ref[i]) {
			t.Fatalf("row %d: async-weight GEMM=%v sync-weight GEMM=%v", i, y[i], ref[i])
		}
	}
	cb.Free(got)
}

// Raw Q4_K bytes go async verbatim: the native Q4_K GEMM over the async weight equals the GEMM over
// the synchronous Upload of the same super-block bytes.
// fak-test:runtime fast est=2s
func TestCUDAUploadAsyncQ4KMatchesUpload(t *testing.T) {
	cb := cudaOrSkip(t)
	var seed lcg = 0x4b4c
	const out, in = 320, 512
	raw := randQ4KBytes(&seed, out, in)
	x := mkResident(cb, []int{in}, randVec(&seed, in))

	ref := cb.Read(cb.MatMul(cb.Upload(newQ4KHost(cb, []int{out, in}, raw), Q4_K), x))

	before, _, _ := cudaAsyncUploadStats()
	got, fence := cb.UploadAsync(newQ4KHost(cb, []int{out, in}, raw), Q4_K)
	if got.Dtype != Q4_K {
		t.Fatalf("UploadAsync(_, Q4_K) produced %s", got.Dtype)
	}
	fence.Wait()
	asyncUploadIssuedOrFatal(t, before)
	y := cb.Read(cb.MatMul(got, x))
	for i := range ref {
		if math.Float32bits(y[i]) != math.Float32bits(ref[i]) {
			t.Fatalf("row %d: async Q4_K GEMM=%v sync Q4_K GEMM=%v", i, y[i], ref[i])
		}
	}
	cb.Free(got)
}

// More in-flight copies than slots: the pool never grows past its bound, every result is still
// bit-exact, and Free of an unawaited upload waits for its copy instead of pooling a live target.
// fak-test:runtime fast est=3s
func TestCUDAUploadAsyncSlotExhaustionStaysBoundedAndExact(t *testing.T) {
	cb := cudaOrSkip(t)
	var seed lcg = 0x1234
	const n = 1 << 20 // 4 MiB per copy
	const copies = 24 // > 8 slots
	srcs := make([][]float32, copies)
	outs := make([]Tensor, copies)
	fences := make([]Fence, copies)
	for i := range srcs {
		srcs[i] = randVec(&seed, n)
		outs[i], fences[i] = cb.UploadAsync(NewF32(cb, []int{1, n}, srcs[i]), F32)
	}
	_, _, pinned := cudaAsyncUploadStats()
	if pinned > 8*(64<<20) {
		t.Fatalf("pinned staging grew to %d bytes, past the 8x64 MiB bound", pinned)
	}
	cb.Free(outs[0]) // unawaited: Free must fence it first
	for i := 1; i < copies; i++ {
		fences[i].Wait()
		back := cb.Read(outs[i])
		for j := 0; j < n; j += 4097 {
			if math.Float32bits(back[j]) != math.Float32bits(srcs[i][j]) {
				t.Fatalf("copy %d element %d mismatched", i, j)
			}
		}
		cb.Free(outs[i])
	}
	if !fences[0].Done() {
		t.Fatal("fence of a Freed upload still reads in flight")
	}
}

// A request outside the verbatim shapes (F32 narrowed to F16) is the synchronous Upload with an
// already-landed fence.
// fak-test:runtime fast est=1s
func TestCUDAUploadAsyncNarrowingFallsBackToUpload(t *testing.T) {
	cb := cudaOrSkip(t)
	var seed lcg = 0x77
	const out, in = 64, 128
	got, fence := cb.UploadAsync(NewF32(cb, []int{out, in}, randVec(&seed, out*in)), F16)
	if !fence.Done() {
		t.Fatal("synchronous fallback returned an unlanded fence")
	}
	if got.Dtype != F16 {
		t.Fatalf("UploadAsync(_, F16) produced %s, want f16 (same narrowing as Upload)", got.Dtype)
	}
}

// UploadAsyncClass (compute.ClassedAsyncUploader, the expert ring's path on CUDA) issues on the
// copy stream for MemoryOffload and books the allocation under that class AT the caller's site;
// any other class is the synchronous UploadClass with a nil fence.
// fak-test:runtime fast est=2s
func TestCUDAUploadAsyncClassBooksOffloadAtSite(t *testing.T) {
	cb := cudaOrSkip(t)
	var seed lcg = 0x3c3c
	const out, in = 128, 256
	w := randVec(&seed, out*in)
	const site = "ring-async-class-witness"

	before, _, _ := cudaAsyncUploadStats()
	got, fence := cb.UploadAsyncClass(NewF32(cb, []int{out, in}, append([]float32(nil), w...)), F32, MemoryOffload, site)
	fence.Wait()
	asyncUploadIssuedOrFatal(t, before)
	want := DeviceAllocClassFor(MemoryOffload, site, false).String()
	booked := false
	for _, s := range cb.DeviceAllocLedger().Sites {
		if s.Site == site && s.LiveCount > 0 {
			if s.Class != want {
				t.Fatalf("site %q booked as %q, want %q", site, s.Class, want)
			}
			booked = true
		}
	}
	if !booked {
		t.Fatalf("no live ledger row for site %q", site)
	}
	back := cb.Read(got)
	for i := range w {
		if math.Float32bits(back[i]) != math.Float32bits(w[i]) {
			t.Fatalf("element %d: async=%08x host=%08x", i, math.Float32bits(back[i]), math.Float32bits(w[i]))
		}
	}
	cb.Free(got)

	issued, _, _ := cudaAsyncUploadStats()
	sync, landed := cb.UploadAsyncClass(NewF32(cb, []int{out, in}, append([]float32(nil), w...)), F32, MemoryWeights, "")
	if landed != nil {
		t.Fatal("synchronous non-offload upload returned a fence; the ring would book it as overlap")
	}
	if after, _, _ := cudaAsyncUploadStats(); after != issued {
		t.Fatalf("non-offload class went through the copy stream (issued %d -> %d)", issued, after)
	}
	cb.Free(sync)
}
