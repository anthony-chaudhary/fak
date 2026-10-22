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
	if len(record) != 528 { //boundarylint:ignore CHANGE_DETECTOR_TEST V4.1 fp8 KV record byte size fixed by the cache format spec
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
	if len(record) != 288 { //boundarylint:ignore CHANGE_DETECTOR_TEST V4.1 fp4 KV record byte size fixed by the cache format spec
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

// This oracle derives finite E4M3 magnitudes directly from the format fields.
// It deliberately uses neither the production decoder nor a quantizer table.
func v41E4M3FiniteMagnitudeOracle(code int) float32 {
	exponent, fraction := code>>3, code&7
	if exponent == 0 {
		return float32(math.Ldexp(float64(fraction), -9))
	}
	return float32(math.Ldexp(float64(8+fraction), exponent-10))
}

// Every finite magnitude code and every adjacent decision boundary is covered,
// including immediate F32 neighbors and both signs. This is not a claim to
// enumerate all F32 inputs or to validate the whole activation kernel.
// fak-test:runtime fast est=100ms lane=default
// Estimated only, not measured: bounded in-memory FP8 cases.
func TestV41E4M3ValueEveryFiniteBoundary(t *testing.T) {
	check := func(x float32, want byte) {
		t.Helper()
		if got := encodeE4M3Value(x); got != want {
			t.Fatalf("input bits=%08x value=%g: code=%02x want=%02x", math.Float32bits(x), x, got, want)
		}
	}
	for code := 0; code <= 0x7e; code++ {
		value := v41E4M3FiniteMagnitudeOracle(code)
		check(value, byte(code))
		check(float32(math.Copysign(float64(value), -1)), 0x80|byte(code))
		if code == 0x7e {
			continue
		}
		next := v41E4M3FiniteMagnitudeOracle(code + 1)
		midpoint := (value + next) / 2
		even := code
		if code&1 != 0 {
			even++
		}
		for _, sample := range []struct {
			value float32
			want  byte
		}{
			{math.Nextafter32(midpoint, float32(math.Inf(-1))), byte(code)},
			{midpoint, byte(even)},
			{math.Nextafter32(midpoint, float32(math.Inf(1))), byte(code + 1)},
		} {
			check(sample.value, sample.want)
			check(-sample.value, 0x80|sample.want)
		}
	}
	for _, value := range []float32{math.Nextafter32(448, float32(math.Inf(1))), 464, 512, math.MaxFloat32} {
		check(value, 0x7e)
		check(-value, 0xfe)
	}
	check(math.SmallestNonzeroFloat32, 0)
	check(-math.SmallestNonzeroFloat32, 0x80)
}

// Public consumer witness: the 448 maximum fixes the power-of-two scale at 1,
// isolating the value conversion from all scale-selection ambiguities.
// fak-test:runtime fast est=100ms lane=default
// Estimated only, not measured: bounded in-memory FP8 cases.
func TestV41KVCacheFP8TiesToEvenPayload(t *testing.T) {
	layout, _ := V41KVCacheLayoutFor(V41KVFP8)
	nope, rope := make([]float32, layout.DNoPE), make([]float32, layout.DRoPE)
	values := []float32{448, 1.1875, -1.1875, 1.0625, -1.0625, 432, -432, float32(math.Copysign(0, -1))}
	copy(nope, values)
	copy(rope, values)
	record, err := layout.V41KVQuantizeRow(nope, rope)
	if err != nil {
		t.Fatal(err)
	}
	want := []byte{0x7e, 0x3a, 0xba, 0x38, 0xb8, 0x7e, 0xfe, 0x80}
	for _, offset := range []int{0, layout.DNoPE} {
		if !bytes.Equal(record[offset:offset+len(want)], want) {
			t.Fatalf("payload at %d = %x, want %x", offset, record[offset:offset+len(want)], want)
		}
	}
	scaleStart := layout.DNoPE + layout.DRoPE
	if record[scaleStart] != 127 || record[scaleStart+layout.DNoPE/layout.TileSize] != 127 {
		t.Fatalf("witness scales are not E8M0 unit codes: %x", record[scaleStart:])
	}
}

// fak-test:runtime fast est=100ms lane=default
// Estimated only, not measured: bounded in-memory FP8 cases.
func TestV41KVCacheFP8NonfinitePolicyUnchanged(t *testing.T) {
	layout, _ := V41KVCacheLayoutFor(V41KVFP8)
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		for _, ropeBad := range []bool{false, true} {
			nope, rope := make([]float32, layout.DNoPE), make([]float32, layout.DRoPE)
			if ropeBad {
				rope[0] = value
			} else {
				nope[0] = value
			}
			record, err := layout.V41KVQuantizeRow(nope, rope)
			if !errors.Is(err, ErrV41KVCacheLayout) || record != nil {
				t.Fatalf("nonfinite input=%g rope=%t: record=%x err=%v", value, ropeBad, record, err)
			}
		}
	}
}

// Literal source-derived witness, not a captured GPU result. DeepInfer commit
// 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9, tilelang_ops/cache_common.py
// ref_cast_from_bf16/ref_cast_to_bf16 and cuda/store_kv_cache.py use BF16
// input, E4M3 RNE scale, reciprocal of the stored scale, then E2M1 RNE.
// 6.375/6 = 1.0625 ties to scale 1 (0x38), not upward to 1.125 (0x39).
// The literal signed payload ties distinguish nearest-even from downward ties.
// fak-test:runtime fast est=100ms lane=default
// Estimated only, not measured: one bounded host row plus refusal cases.
func TestV41KVCacheFP4DeepInferLiteral(t *testing.T) {
	layout, _ := V41KVCacheLayoutFor(V41KVFP4)
	nope, rope := make([]float32, layout.DNoPE), make([]float32, layout.DRoPE)
	values := []float32{6.375, -6.375, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5, -5, 0, math.Float32frombits(0x80000000)}
	copy(nope, values)
	for i, v := range values {
		rope[i] = v * 2 // distinct RoPE scale, same packed payload
	}
	// A zero tile must use scale 2^-9, not zero. A large finite BF16 tile
	// saturates both scale and payload; nearest-distance subtraction alone
	// cannot safely choose an E2M1 code at this magnitude.
	nope[32], nope[33] = 0x1p100, -0x1p100
	// The F32 convenience boundary rounds 0.749 to BF16 0.75 BEFORE casting;
	// its E2M1 tie rounds to 1.0, not 0.5. Inputs must remain unmodified.
	nope[48], nope[49] = 6.375, 0.749
	record, err := layout.V41KVQuantizeRow(nope, rope)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]byte, 288)
	packed := []byte{0xf7, 0xa2, 0xa2, 0xc4, 0xc4, 0xe6, 0xe6, 0x80}
	copy(want, packed)
	copy(want[224:], packed)
	want[16], want[24] = 0xf7, 0x27
	for i := 256; i < len(want); i++ {
		want[i] = 0x01
	}
	want[256], want[258], want[259], want[284] = 0x38, 0x7e, 0x38, 0x40
	if !bytes.Equal(record, want) {
		t.Fatalf("FP4 record = %x, want literal %x", record, want)
	}
	if nope[49] != float32(0.749) || math.Float32bits(nope[15]) != 0x80000000 {
		t.Fatal("quantization modified caller-owned input")
	}
	row, err := layout.V41KVDequantizeRow(record)
	if err != nil {
		t.Fatal(err)
	}
	decoded := []float32{6, -6, 1, -1, 1, -1, 2, -2, 2, -2, 4, -4, 4, -4, 0, math.Float32frombits(0x80000000)}
	for i, v := range decoded {
		if math.Float32bits(row.NoPE[i]) != math.Float32bits(v) || math.Float32bits(row.RoPE[i]) != math.Float32bits(v*2) {
			t.Fatalf("decoded value %d = %g/%g, want %g/%g", i, row.NoPE[i], row.RoPE[i], v, v*2)
		}
	}
	if row.NoPEScales[1] != 0x1p-9 || row.NoPE[16] != 0 || row.NoPE[32] != 2688 || row.NoPE[33] != -2688 || row.NoPE[49] != 1 {
		t.Fatalf("zero/saturated/BF16 boundary decode = %g/%g/%g/%g/%g", row.NoPEScales[1], row.NoPE[16], row.NoPE[32], row.NoPE[33], row.NoPE[49])
	}
	for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), math.MaxFloat32} {
		for _, part := range [][]float32{nope, rope} {
			old := part[0]
			part[0] = bad
			record, err := layout.V41KVQuantizeRow(nope, rope)
			part[0] = old
			if record != nil || !errors.Is(err, ErrV41KVCacheLayout) {
				t.Fatalf("nonfinite/BF16 overflow %g: record=%x err=%v", bad, record, err)
			}
		}
	}
}
