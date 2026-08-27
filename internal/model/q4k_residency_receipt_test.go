//go:build darwin && arm64 && cgo

package model

import (
	"bytes"
	"os"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestQ4KResidencyReceiptMappedSuccessAndDecline(t *testing.T) {
	if !metalgemmAvailableForQ4KReceiptTest() {
		t.Fatal("Metal unavailable on darwin/arm64/cgo Q4_K residency witness host")
	}
	t.Setenv("FAK_GGUF_MMAP", "1")
	const (
		out = 1
		in  = qkK
	)
	bytesPerTensor := out * q4kBlockBytes

	m := &Model{}
	page := os.Getpagesize()
	mapped := makePageAlignedResidentBytes(page)
	copy(mapped, make([]byte, bytesPerTensor))
	mappedTensor := &q4kTensor{out: out, in: in, nblk: 1, lazy: &LazyQ4KRange{
		Reader: bytes.NewReader(mapped[:bytesPerTensor]), Bytes: bytesPerTensor,
		MappedSpan: mapped, MappedOffset: 0,
	}}
	if w := m.metalQ4KWeight("mapped", mappedTensor); w == nil || !w.NoCopy() {
		t.Fatal("mapped Q4_K upload did not produce a no-copy handle")
	}
	// The shortened, unaligned span is present but fails mappedRaw validation. The same bytes are
	// available through ReaderAt, deterministically forcing a copied upload that succeeds.
	declinedSpan := mapped[1:]
	declinedTensor := &q4kTensor{out: out, in: in, nblk: 1, lazy: &LazyQ4KRange{
		Reader: bytes.NewReader(mapped[:bytesPerTensor]), Bytes: bytesPerTensor,
		MappedSpan: declinedSpan, MappedOffset: 0,
	}}
	if w := m.metalQ4KWeight("declined", declinedTensor); w == nil {
		t.Fatal("copied Q4_K fallback upload failed")
	}

	want := Q4KResidencyCount{Tensors: 1, Bytes: uint64(bytesPerTensor)}
	got := m.Q4KResidencyReceipt()
	if got.FAKGGUFMMap != "1" || got.MappedSuccess != want || got.MappedDeclineCopiedUpload != want || got.UploadFailure != (Q4KResidencyCount{}) {
		t.Fatalf("receipt = %+v, want one %d-byte mapped success and decline/copy", got, bytesPerTensor)
	}
	if err := ValidateQ4KResidencyReceipt(got); err != nil {
		t.Fatalf("valid receipt rejected: %v", err)
	}

	// Cached lookup and teardown do not alter the captured outcome. Even an abnormal lookup after
	// teardown is deduplicated by the receipt state rather than counted as another tensor.
	if m.metalQ4KWeight("mapped", mappedTensor) == nil {
		t.Fatal("cached mapped handle disappeared")
	}
	beforeTeardown := m.Q4KResidencyReceipt()
	releaseMetalQ4KResidency(m)
	if after := m.Q4KResidencyReceipt(); !reflect.DeepEqual(after, beforeTeardown) {
		t.Fatalf("teardown changed receipt: before=%+v after=%+v", beforeTeardown, after)
	}
	if m.metalQ4KWeight("mapped", mappedTensor) == nil {
		t.Fatal("post-teardown witness re-upload failed")
	}
	if afterReuse := m.Q4KResidencyReceipt(); !reflect.DeepEqual(afterReuse, beforeTeardown) {
		t.Fatalf("post-teardown reuse double-counted: before=%+v after=%+v", beforeTeardown, afterReuse)
	}
	releaseMetalQ4KResidency(m)
}

func TestQ4KResidencyReceiptSeparatesNilUploadFailure(t *testing.T) {
	t.Setenv("FAK_GGUF_MMAP", "1")
	m := &Model{}
	span := makePageAlignedResidentBytes(os.Getpagesize())
	const bytesPerTensor = q4kBlockBytes
	// in is deliberately invalid for Q4_K, forcing both mapped and copied uploads to nil.
	qt := &q4kTensor{out: 1, in: qkK + 1, nblk: 1, lazy: &LazyQ4KRange{
		Reader: bytes.NewReader(make([]byte, bytesPerTensor)), Bytes: bytesPerTensor,
		MappedSpan: span, MappedOffset: 0,
	}}
	if w := m.metalQ4KWeight("failed", qt); w != nil {
		w.Release()
		t.Fatal("invalid Q4_K geometry unexpectedly uploaded")
	}
	got := m.Q4KResidencyReceipt()
	if got.UploadFailure != (Q4KResidencyCount{Tensors: 1, Bytes: bytesPerTensor}) || got.MappedDeclineCopiedUpload != (Q4KResidencyCount{}) {
		t.Fatalf("nil upload misclassified: %+v", got)
	}
}

func TestQ4KResidencyReceiptConcurrentOnceOnlyAndTamper(t *testing.T) {
	t.Setenv("FAK_GGUF_MMAP", "1")
	m := &Model{}
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			recordQ4KResidencyOutcome(m, "same", 144, q4kResidencyMappedSuccess)
		}()
	}
	wg.Wait()
	receipt := m.Q4KResidencyReceipt()
	if receipt.MappedSuccess != (Q4KResidencyCount{Tensors: 1, Bytes: 144}) {
		t.Fatalf("concurrent capture double-counted: %+v", receipt)
	}
	receipt.MappedSuccess.Bytes++
	if err := ValidateQ4KResidencyReceipt(receipt); err == nil {
		t.Fatal("tampered receipt was accepted")
	}
}

// Keeping the availability check behind a helper makes the test failure state explicit while
// leaving the receipt witness independent of a full model artifact.
func metalgemmAvailableForQ4KReceiptTest() bool {
	return metalgemm.Available()
}
