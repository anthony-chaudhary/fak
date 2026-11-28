package compute

import "testing"

func TestPaddedRowSpans(t *testing.T) {
	t.Parallel()
	// DSv4.1 idx_c2a: a 13056-byte block lives in a 16384-byte padded row.
	if PaddedRowPad(13056) != 16384 {
		t.Fatalf("pad(13056) = %d, want 16384", PaddedRowPad(13056))
	}
	spans, err := PaddedRowSpans(PaddedRowLayout{Units: 1, UnitBytes: 13056, HostStride: 16384})
	if err != nil || len(spans) != 1 || spans[0].Len != 13056 {
		t.Fatalf("spans = %+v, err %v; want one span of 13056 bytes (padding is not copied)", spans, err)
	}
	// Multiple units: host and device strides advance by the unit's own bytes.
	spans, err = PaddedRowSpans(PaddedRowLayout{Units: 4, UnitBytes: 3000, HostStride: 12288})
	if err != nil || len(spans) != 4 || spans[3].HostOffset != 9000 || spans[3].DevOffset != 9000 || spans[3].Len != 3000 {
		t.Fatalf("spans = %+v, err %v", spans, err)
	}
	// A replicated leaf copies through unit 0 only.
	spans, err = PaddedRowSpans(PaddedRowLayout{Units: 4, UnitBytes: 4096, HostStride: 4096, Replicated: true})
	if err != nil || len(spans) != 1 || spans[0].Unit != 0 {
		t.Fatalf("replicated spans = %+v, err %v", spans, err)
	}
	// Construction-time validation.
	if _, err := PaddedRowSpans(PaddedRowLayout{Units: 2, UnitBytes: 3000, HostStride: 4096}); err == nil {
		t.Fatal("logical row above the stride must be refused")
	}
	if _, err := PaddedRowSpans(PaddedRowLayout{Units: 1, UnitBytes: 100, HostStride: 5000}); err == nil {
		t.Fatal("stride that is not whole 4 KiB must be refused")
	}
}
