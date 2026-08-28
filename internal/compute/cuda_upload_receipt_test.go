//go:build cuda

package compute

import "testing"

func TestCUDAImmutableWeightUploadReceiptCountsActualCacheMisses(t *testing.T) {
	cb := cudaOrSkip(t)
	beforeCalls, beforeTransfer, beforeResident := cb.CUDAImmutableWeightUploadSnapshot()
	assertCacheHit := func(name string, first, second Tensor) {
		t.Helper()
		if first.buf != second.buf {
			t.Fatalf("same %s host weight did not hit CUDA upload cache", name)
		}
	}

	host := NewF32(Default(), []int{2, 2}, []float32{1, 2, 3, 4})
	first := cb.Upload(host, F32)
	assertCacheHit("F32", first, cb.Upload(host, F32))

	f16Host := NewF32(Default(), []int{2, 2}, []float32{5, 6, 7, 8})
	f16 := cb.Upload(f16Host, F16)
	f32Q8Host := NewF32(Default(), []int{1, 32}, make([]float32, 32))
	f32Q8 := cb.Upload(f32Q8Host, Q8_0)
	assertCacheHit("F32-to-Q8", f32Q8, cb.Upload(f32Q8Host, Q8_0))

	residentQ8Host := NewQ8(Default(), []int{1, 32}, make([]int8, 32), []float32{1}, 32)
	residentQ8 := cb.Upload(residentQ8Host, Q8_0)
	assertCacheHit("resident Q8", residentQ8, cb.Upload(residentQ8Host, Q8_0))

	residentQ2Host := NewQ2(Default(), []int{1, 32}, make([]byte, 8), []float32{1}, 32)
	residentQ2 := cb.Upload(residentQ2Host, Q2_0)
	assertCacheHit("resident Q2", residentQ2, cb.Upload(residentQ2Host, Q2_0))
	if buf := residentQ2.buf.(*cudaBuf); buf.hostKeep != residentQ2Host.buf {
		t.Fatalf("resident Q2 upload cache did not retain host owner: got %T, want source HostBuffer", buf.hostKeep)
	}

	rawQ4Host := NewQ4K(Default(), []int{1, 256}, make([]byte, 144))
	rawQ4 := cb.Upload(rawQ4Host, Q4_K)
	assertCacheHit("raw Q4_K", rawQ4, cb.Upload(rawQ4Host, Q4_K))
	rawQ5Host := NewQ5K(Default(), []int{1, 256}, make([]byte, 176))
	rawQ5 := cb.Upload(rawQ5Host, Q5_K)
	assertCacheHit("raw Q5_K", rawQ5, cb.Upload(rawQ5Host, Q5_K))
	rawQ6Host := NewQ6K(Default(), []int{1, 256}, make([]byte, 210))
	rawQ6 := cb.Upload(rawQ6Host, Q6_K)
	assertCacheHit("raw Q6_K", rawQ6, cb.Upload(rawQ6Host, Q6_K))

	activation := cb.UploadClass(NewF32(Default(), []int{4}, []float32{9, 10, 11, 12}), F32, MemoryActivation, "upload-receipt-activation")
	t.Cleanup(func() {
		cb.Free(first)
		cb.Free(f16)
		cb.Free(f32Q8)
		cb.Free(residentQ8)
		cb.Free(residentQ2)
		cb.Free(rawQ4)
		cb.Free(rawQ5)
		cb.Free(rawQ6)
		cb.Free(activation)
	})

	afterCalls, afterTransfer, afterResident := cb.CUDAImmutableWeightUploadSnapshot()
	if got := afterCalls - beforeCalls; got != 8 {
		t.Fatalf("immutable upload calls delta = %d, want 8 (cache hits and activation excluded)", got)
	}
	q8Bytes := 32 + F32.Bytes() // one 32-code block plus its f32 scale sidecar
	q2Bytes := 8 + F32.Bytes()  // eight packed 2-bit code bytes plus one f32 scale
	rawKBytes := 144 + 176 + 210
	if got, want := afterTransfer-beforeTransfer, uint64(2*2*F32.Bytes()+2*2*F32.Bytes()+q8Bytes+q8Bytes+q2Bytes+rawKBytes); got != want {
		t.Fatalf("immutable transfer bytes delta = %d, want %d", got, want)
	}
	if got, want := afterResident-beforeResident, uint64(2*2*F32.Bytes()+2*2*F16.Bytes()+q8Bytes+q8Bytes+q2Bytes+rawKBytes); got != want {
		t.Fatalf("immutable resident bytes delta = %d, want %d", got, want)
	}
}

func TestCUDAImmutableWeightUploadReceiptCountsUncachedRankOneWeights(t *testing.T) {
	cb := cudaOrSkip(t)
	const n = 4
	wantBytes := uint64(n * F32.Bytes())
	host := NewF32(Default(), []int{n}, []float32{1, 2, 3, 4})

	assertDelta := func(name string, beforeCalls, beforeTransfer, beforeResident uint64, wantCalls, wantTransfer, wantResident uint64) {
		t.Helper()
		afterCalls, afterTransfer, afterResident := cb.CUDAImmutableWeightUploadSnapshot()
		if got := afterCalls - beforeCalls; got != wantCalls {
			t.Fatalf("%s immutable upload calls delta = %d, want %d", name, got, wantCalls)
		}
		if got := afterTransfer - beforeTransfer; got != wantTransfer {
			t.Fatalf("%s immutable transfer bytes delta = %d, want %d", name, got, wantTransfer)
		}
		if got := afterResident - beforeResident; got != wantResident {
			t.Fatalf("%s immutable resident bytes delta = %d, want %d", name, got, wantResident)
		}
	}

	beforeCalls, beforeTransfer, beforeResident := cb.CUDAImmutableWeightUploadSnapshot()
	first := cb.UploadClass(host, F32, MemoryWeights, "upload-receipt-rank-one-first")
	assertDelta("first rank-one weight", beforeCalls, beforeTransfer, beforeResident, 1, wantBytes, wantBytes)

	beforeCalls, beforeTransfer, beforeResident = cb.CUDAImmutableWeightUploadSnapshot()
	second := cb.UploadClass(host, F32, MemoryWeights, "upload-receipt-rank-one-second")
	assertDelta("second same-pointer rank-one weight", beforeCalls, beforeTransfer, beforeResident, 1, wantBytes, wantBytes)
	if first.buf == second.buf {
		t.Fatal("same-pointer rank-one weight unexpectedly hit the matrix upload cache")
	}

	beforeCalls, beforeTransfer, beforeResident = cb.CUDAImmutableWeightUploadSnapshot()
	activation := cb.UploadClass(host, F32, MemoryActivation, "upload-receipt-rank-one-activation")
	assertDelta("rank-one activation", beforeCalls, beforeTransfer, beforeResident, 0, 0, 0)

	t.Cleanup(func() {
		cb.Free(first)
		cb.Free(second)
		cb.Free(activation)
	})
}

func TestCUDAImmutableWeightUploadReceiptRefusalDoesNotAdvance(t *testing.T) {
	cb := cudaOrSkip(t)
	beforeCalls, beforeTransfer, beforeResident := cb.CUDAImmutableWeightUploadSnapshot()
	func() {
		defer func() {
			if recover() == nil {
				t.Fatal("invalid rank-one Q8 upload did not refuse")
			}
		}()
		cb.Upload(NewF32(Default(), []int{32}, make([]float32, 32)), Q8_0)
	}()
	afterCalls, afterTransfer, afterResident := cb.CUDAImmutableWeightUploadSnapshot()
	if afterCalls != beforeCalls || afterTransfer != beforeTransfer || afterResident != beforeResident {
		t.Fatalf("refused upload advanced receipt: before=%d/%d/%d after=%d/%d/%d", beforeCalls, beforeTransfer, beforeResident, afterCalls, afterTransfer, afterResident)
	}
}
