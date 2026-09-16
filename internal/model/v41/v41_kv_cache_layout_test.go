package v41

import (
	"bytes"
	"errors"
	"math"
	"testing"
)

// v41KVTestRow builds a deterministic, finite row whose values are tagged by
// index so any misalignment in the tile or nibble walk is visible.
func v41KVTestRow(width int) []float32 {
	row := make([]float32, width)
	for i := range row {
		row[i] = float32(math.Sin(float64(i)*0.37) * 3.5)
	}
	return row
}

// TestV41KVCacheLayoutPublishedGeometry pins the two V4.1 record layouts from
// FlashMLA tests/quant.py: 528 B/token fp8 (tile 32, 16 NoPE tiles, e8m0) and
// 288 B/token fp4 (tile 16, 32 NoPE tiles, e4m3). It is the leaf's headline
// witness that a typed contract exists and derives its own size.
func TestV41KVCacheLayoutPublishedGeometry(t *testing.T) {
	fp8, ok := V41KVCacheLayoutFor(V41KVFP8)
	if !ok {
		t.Fatal("fp8 layout is not in the published set")
	}
	if fp8.DNoPE != 448 || fp8.DRoPE != 64 || fp8.TileSize != 32 || fp8.NumTiles != 16 {
		t.Fatalf("fp8 geometry = %+v, want NoPE 448 RoPE 64 tile 32 tiles 16", fp8)
	}
	if fp8.ScaleFormat != V41KVScaleE8M0 {
		t.Fatalf("fp8 scale format = %s, want e8m0", fp8.ScaleFormat)
	}
	if fp8.BytesPerToken != 528 {
		t.Fatalf("fp8 bytes/token = %d, want 528", fp8.BytesPerToken)
	}
	if err := fp8.Validate(); err != nil {
		t.Fatalf("fp8 layout failed its own validation: %v", err)
	}

	fp4, ok := V41KVCacheLayoutFor(V41KVFP4)
	if !ok {
		t.Fatal("fp4 layout is not in the published set")
	}
	if fp4.DNoPE != 448 || fp4.DRoPE != 64 || fp4.TileSize != 16 || fp4.NumTiles != 32 {
		t.Fatalf("fp4 geometry = %+v, want NoPE 448 RoPE 64 tile 16 tiles 32", fp4)
	}
	if fp4.ScaleFormat != V41KVScaleE4M3 {
		t.Fatalf("fp4 scale format = %s, want e4m3", fp4.ScaleFormat)
	}
	if fp4.BytesPerToken != 288 {
		t.Fatalf("fp4 bytes/token = %d, want 288", fp4.BytesPerToken)
	}
	if err := fp4.Validate(); err != nil {
		t.Fatalf("fp4 layout failed its own validation: %v", err)
	}

	all := V41KVCacheLayouts()
	if len(all) != 2 || all[0].Format != V41KVFP8 || all[1].Format != V41KVFP4 {
		t.Fatalf("published census = %+v, want fp8 then fp4", all)
	}
}

// TestV41KVCacheLayoutUnknownFormatFailsClosed proves the zero value and any
// unlisted format are refused, so a caller can never fall through to a generic
// or Flash-0731 path for an unknown V4.1 record.
func TestV41KVCacheLayoutUnknownFormatFailsClosed(t *testing.T) {
	if _, ok := V41KVCacheLayoutFor(V41KVUnspecified); ok {
		t.Fatal("unspecified format resolved to a layout")
	}
	if _, ok := V41KVCacheLayoutFor(V41KVQuantFormat(99)); ok {
		t.Fatal("unlisted format resolved to a layout")
	}
	var zero V41KVCacheLayout
	if err := zero.Validate(); !errors.Is(err, ErrV41KVCacheLayout) {
		t.Fatalf("zero layout Validate err = %v, want ErrV41KVCacheLayout", err)
	}
	if _, err := zero.V41KVQuantizeRow(nil, nil); !errors.Is(err, ErrV41KVCacheLayout) {
		t.Fatalf("zero layout Quantize err = %v, want ErrV41KVCacheLayout", err)
	}
}

// TestV41KVCacheLayoutByteCountMismatchFailsClosed is the red control: a layout
// whose declared size disagrees with its published geometry must be refused, not
// silently used to size a cache.
func TestV41KVCacheLayoutByteCountMismatchFailsClosed(t *testing.T) {
	bad, _ := V41KVCacheLayoutFor(V41KVFP8)
	bad.BytesPerToken = 584 // the V4 (not V4.1) fp8 size
	if err := bad.Validate(); !errors.Is(err, ErrV41KVCacheLayout) {
		t.Fatalf("mismatched layout Validate err = %v, want ErrV41KVCacheLayout", err)
	}
	// And a wrong tile multiplicity must also be refused.
	bad2, _ := V41KVCacheLayoutFor(V41KVFP4)
	bad2.TileSize = 32
	if err := bad2.Validate(); !errors.Is(err, ErrV41KVCacheLayout) {
		t.Fatalf("wrong-tile layout Validate err = %v, want ErrV41KVCacheLayout", err)
	}
}

// TestV41KVCacheRowRoundTripFP8 quantizes and dequantizes a deterministic row
// and requires the record to be exactly 528 bytes with the e8m0 scale row and a
// bounded reconstruction error.
func TestV41KVCacheRowRoundTripFP8(t *testing.T) {
	l, _ := V41KVCacheLayoutFor(V41KVFP8)
	nope := v41KVTestRow(l.DNoPE)
	rope := v41KVTestRow(l.DRoPE)

	record, err := l.V41KVQuantizeRow(nope, rope)
	if err != nil {
		t.Fatal(err)
	}
	if len(record) != 528 {
		t.Fatalf("fp8 record = %d bytes, want 528", len(record))
	}

	row, err := l.V41KVDequantizeRow(record)
	if err != nil {
		t.Fatal(err)
	}
	if len(row.NoPE) != l.DNoPE || len(row.RoPE) != l.DRoPE {
		t.Fatalf("decoded widths = %d/%d, want %d/%d", len(row.NoPE), len(row.RoPE), l.DNoPE, l.DRoPE)
	}
	nopeTiles := l.DNoPE / l.TileSize
	wantRoPE := l.NumTiles - nopeTiles
	if len(row.NoPEScales) != nopeTiles || len(row.RoPEScales) != wantRoPE {
		t.Fatalf("scale counts = %d/%d, want %d/%d", len(row.NoPEScales), len(row.RoPEScales), nopeTiles, wantRoPE)
	}
	// Every e8m0 scale is a power of two; byte 127 (2^0) at minimum.
	for i, s := range row.NoPEScales {
		if math.IsNaN(float64(s)) || s <= 0 {
			t.Fatalf("NoPE scale[%d] = %g, want a positive power of two", i, s)
		}
	}
	// fp8 e4m3 has ~3 significant bits, so a scale-of-amax reconstruction is
	// within a few percent; assert a documented 6% relative bound.
	assertV41KVApprox(t, "fp8 NoPE", nope, row.NoPE, 0.06)
	assertV41KVApprox(t, "fp8 RoPE", rope, row.RoPE, 0.06)
}

// TestV41KVCacheRowRoundTripFP4 does the same for the 288-byte fp4 record,
// exercising nibble packing and the e4m3 amax/6 scale.
func TestV41KVCacheRowRoundTripFP4(t *testing.T) {
	l, _ := V41KVCacheLayoutFor(V41KVFP4)
	nope := v41KVTestRow(l.DNoPE)
	rope := v41KVTestRow(l.DRoPE)

	record, err := l.V41KVQuantizeRow(nope, rope)
	if err != nil {
		t.Fatal(err)
	}
	if len(record) != 288 {
		t.Fatalf("fp4 record = %d bytes, want 288", len(record))
	}

	row, err := l.V41KVDequantizeRow(record)
	if err != nil {
		t.Fatal(err)
	}
	if len(row.NoPE) != l.DNoPE || len(row.RoPE) != l.DRoPE {
		t.Fatalf("decoded widths = %d/%d, want %d/%d", len(row.NoPE), len(row.RoPE), l.DNoPE, l.DRoPE)
	}
	nopeTiles := l.DNoPE / l.TileSize
	wantRoPE := l.NumTiles - nopeTiles
	if len(row.NoPEScales) != nopeTiles || len(row.RoPEScales) != wantRoPE {
		t.Fatalf("scale counts = %d/%d, want %d/%d", len(row.NoPEScales), len(row.RoPEScales), nopeTiles, wantRoPE)
	}
	// e2m1 is a 1-bit-mantissa format with an amax/6 scale; a 35% relative bound
	// reflects the coarser grid while still catching a mispacked nibble walk.
	assertV41KVApprox(t, "fp4 NoPE", nope, row.NoPE, 0.35)
	assertV41KVApprox(t, "fp4 RoPE", rope, row.RoPE, 0.35)
}

// TestV41KVCacheRowMalformedFailsClosed proves width, length, and non-finite
// inputs are refused with the typed sentinel before any record is produced.
func TestV41KVCacheRowMalformedFailsClosed(t *testing.T) {
	l, _ := V41KVCacheLayoutFor(V41KVFP8)
	goodNoPE := v41KVTestRow(l.DNoPE)
	goodRoPE := v41KVTestRow(l.DRoPE)

	if _, err := l.V41KVQuantizeRow(goodNoPE[:l.DNoPE-1], goodRoPE); !errors.Is(err, ErrV41KVCacheLayout) {
		t.Fatalf("short NoPE err = %v, want ErrV41KVCacheLayout", err)
	}
	if _, err := l.V41KVQuantizeRow(goodNoPE, goodRoPE[:l.DRoPE-1]); !errors.Is(err, ErrV41KVCacheLayout) {
		t.Fatalf("short RoPE err = %v, want ErrV41KVCacheLayout", err)
	}
	nan := v41KVTestRow(l.DNoPE)
	nan[17] = float32(math.NaN())
	if _, err := l.V41KVQuantizeRow(nan, goodRoPE); !errors.Is(err, ErrV41KVCacheLayout) {
		t.Fatalf("non-finite NoPE err = %v, want ErrV41KVCacheLayout", err)
	}
	if _, err := l.V41KVDequantizeRow(make([]byte, 527)); !errors.Is(err, ErrV41KVCacheLayout) {
		t.Fatalf("short record decode err = %v, want ErrV41KVCacheLayout", err)
	}
	if _, err := l.V41KVDequantizeRow(make([]byte, 529)); !errors.Is(err, ErrV41KVCacheLayout) {
		t.Fatalf("long record decode err = %v, want ErrV41KVCacheLayout", err)
	}
}

// TestV41KVCacheRowDecodeIsDeterministic pins byte-for-byte reproducibility of
// the quantization, so the fixture is a deterministic reproduction witness.
func TestV41KVCacheRowDecodeIsDeterministic(t *testing.T) {
	for _, format := range []V41KVQuantFormat{V41KVFP8, V41KVFP4} {
		l, _ := V41KVCacheLayoutFor(format)
		nope, rope := v41KVTestRow(l.DNoPE), v41KVTestRow(l.DRoPE)
		a, err := l.V41KVQuantizeRow(nope, rope)
		if err != nil {
			t.Fatal(err)
		}
		b, err := l.V41KVQuantizeRow(nope, rope)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a, b) {
			t.Fatalf("%s quantization is not deterministic", l.Format)
		}
	}
}

// TestV41KVCacheRowZeroRowIsExact proves an all-zero row survives the round trip
// exactly, so the zero-tile scale branches are correct rather than NaN.-
func TestV41KVCacheRowZeroRowIsExact(t *testing.T) {
	for _, format := range []V41KVQuantFormat{V41KVFP8, V41KVFP4} {
		l, _ := V41KVCacheLayoutFor(format)
		record, err := l.V41KVQuantizeRow(make([]float32, l.DNoPE), make([]float32, l.DRoPE))
		if err != nil {
			t.Fatal(err)
		}
		row, err := l.V41KVDequantizeRow(record)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range row.NoPE {
			if v != 0 {
				t.Fatalf("%s zero NoPE[%d] = %g, want 0", l.Format, i, v)
			}
		}
		for i, v := range row.RoPE {
			if v != 0 {
				t.Fatalf("%s zero RoPE[%d] = %g, want 0", l.Format, i, v)
			}
		}
	}
}

// assertV41KVApprox requires every dequantized value to be within a relative
// tolerance of the source (absolute tolerance for near-zero elements).
func assertV41KVApprox(t *testing.T, what string, want, got []float32, tol float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s length = %d, want %d", what, len(got), len(want))
	}
	for i := range want {
		diff := math.Abs(float64(got[i] - want[i]))
		bound := tol * math.Abs(float64(want[i]))
		if bound < tol {
			bound = tol
		}
		if diff > bound {
			t.Fatalf("%s[%d] = %g, want %g (diff %g > bound %g)", what, i, got[i], want[i], diff, bound)
		}
	}
}
