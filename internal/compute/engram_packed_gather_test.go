package compute

import (
	"errors"
	"math"
	"testing"
)

// engramFixtureBytes builds a deterministic packed table fixture: dim 256,
// every E2M1 code cycling through each row, and an E8M0 scale set spanning
// deep underflow (byte 80 = 2^-47) to overflow (byte 254 = 2^127).
func engramFixtureBytes(t *testing.T, rows int) (data, scales []byte) {
	t.Helper()
	const dim = 256
	data = make([]byte, rows*dim/2)
	scales = make([]byte, rows*dim/32)
	scaleSet := []byte{127, 120, 135, 100, 200, 80, 128, 254}
	for r := 0; r < rows; r++ {
		for b := 0; b < dim/32; b++ {
			scales[r*dim/32+b] = scaleSet[(r+b)%len(scaleSet)]
		}
		for g := 0; g < dim/2; g++ {
			lo := byte((g*7 + r*3) % 16)
			hi := byte((g*11 + r*5 + 1) % 16)
			data[r*dim/2+g] = lo | hi<<4
		}
	}
	return data, scales
}

func TestEngramPackedGatherMatchesExpandedReferenceBitForBit(t *testing.T) {
	// fak-test:runtime fast est=5ms lane=default
	const dim = 256
	data, scales := engramFixtureBytes(t, 4)
	table, err := NewPackedEngramTable(dim, 4, data, scales)
	if err != nil {
		t.Fatalf("NewPackedEngramTable: %v", err)
	}
	ids := []int32{0, 3, 1, 1, 2, 0, 3}
	got, err := table.Gather(ids)
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	ref := table.ExpandToE4M3()
	want, err := GatherE4M3Rows(ref, scales, dim, ids)
	if err != nil {
		t.Fatalf("GatherE4M3Rows: %v", err)
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("row id %d col %d: packed bits %08x != expanded bits %08x",
				ids[i], i%dim, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}

	// Closed known-value vectors on an explicit single row: columns 0..15 hold
	// E2M1 codes 0..15 in order (even column in the low nibble, odd in the
	// high), everything else is zero.
	explicit := func(scale byte) []float32 {
		row := make([]byte, dim/2)
		sc := make([]byte, dim/32)
		for c := 0; c < 8; c++ {
			row[c] = byte(2*c) | byte(2*c+1)<<4
		}
		for b := range sc {
			sc[b] = scale
		}
		tbl, err := NewPackedEngramTable(dim, 1, row, sc)
		if err != nil {
			t.Fatalf("NewPackedEngramTable(scale %d): %v", scale, err)
		}
		out, err := tbl.Gather([]int32{0})
		if err != nil {
			t.Fatalf("Gather(scale %d): %v", scale, err)
		}
		return out
	}
	checkCols := func(name string, out []float32, want []float32) {
		t.Helper()
		for c, w := range want {
			if math.Float32bits(out[c]) != math.Float32bits(w) {
				t.Fatalf("%s col %d: got bits %08x (%v), want bits %08x (%v)",
					name, c, math.Float32bits(out[c]), out[c], math.Float32bits(w), w)
			}
		}
	}
	neg := func(v float32) float32 { return math.Float32frombits(math.Float32bits(v) | 0x80000000) }
	ld := func(mant, exp int) float32 { return float32(math.Ldexp(float64(mant), exp)) }
	inf := func(s int) float32 { return float32(math.Inf(s)) }

	// Scale 127 (2^0): the raw E2M1 magnitudes, signs included. -0 at col 8
	// must keep its sign bit, matching the upstream signed-zero guarantee.
	checkCols("scale 127", explicit(127)[:16], []float32{
		0, 0.5, 1, 1.5, 2, 3, 4, 6,
		math.Float32frombits(0x80000000), neg(0.5), neg(1), neg(1.5), neg(2), neg(3), neg(4), neg(6),
	})
	// Scale 128 (2^1): one step up.
	checkCols("scale 128", explicit(128)[:16], []float32{
		0, 1, 2, 3, 4, 6, 8, 12,
		math.Float32frombits(0x80000000), neg(1), neg(2), neg(3), neg(4), neg(6), neg(8), neg(12),
	})
	// Scale 126 (2^-1): one step down; 0.5 lands on 0.25, exact in bf16.
	checkCols("scale 126", explicit(126)[:16], []float32{
		0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3,
		math.Float32frombits(0x80000000), neg(0.25), neg(0.5), neg(0.75), neg(1), neg(1.5), neg(2), neg(3),
	})
	// Scale 0 (2^-127): the smallest E8M0. 0.5 becomes the exact bf16
	// subnormal 2^-128; 6 stays the exact subnormal 3*2^-126.
	checkCols("scale 0", explicit(0)[:16], []float32{
		0, ld(1, -128), ld(1, -127), ld(3, -128),
		ld(1, -126), ld(3, -127), ld(1, -125), ld(3, -126),
		math.Float32frombits(0x80000000), neg(ld(1, -128)), neg(ld(1, -127)),
		neg(ld(3, -128)), neg(ld(1, -126)), neg(ld(3, -127)),
		neg(ld(1, -125)), neg(ld(3, -126)),
	})
	// Scale 254 (2^127): beyond the bf16 normal range the product saturates
	// to +/-Inf exactly where the exponent field runs out.
	checkCols("scale 254", explicit(254)[:16], []float32{
		0, ld(1, 126), ld(1, 127), ld(3, 126),
		inf(1), inf(1), inf(1), inf(1),
		math.Float32frombits(0x80000000), neg(ld(1, 126)), neg(ld(1, 127)),
		neg(ld(3, 126)), inf(-1), inf(-1), inf(-1), inf(-1),
	})

	// The expanded copy itself is pinned to the upstream e4m3 encoding:
	// magnitude codes 0, 1, 7 are 0x00, 0x30, 0x4C; the sign bit is 0x80.
	row := make([]byte, dim/2)
	for c := 0; c < 8; c++ {
		row[c] = byte(2*c) | byte(2*c+1)<<4
	}
	sc := make([]byte, dim/32)
	tbl, err := NewPackedEngramTable(dim, 1, row, sc)
	if err != nil {
		t.Fatalf("NewPackedEngramTable(e4m3 pin): %v", err)
	}
	e4 := tbl.ExpandToE4M3()
	for c, want := range []byte{0x00, 0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c,
		0x80, 0xb0, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcc} {
		if e4[c] != want {
			t.Fatalf("ExpandToE4M3 col %d: got 0x%02x, want 0x%02x", c, e4[c], want)
		}
	}
}

func TestEngramPackedGatherNoExpansionAllocation(t *testing.T) {
	// fak-test:runtime fast est=5ms lane=default
	smallData, smallScales := engramFixtureBytes(t, 4)
	bigData, bigScales := engramFixtureBytes(t, 400)
	small, err := NewPackedEngramTable(256, 4, smallData, smallScales)
	if err != nil {
		t.Fatalf("NewPackedEngramTable(small): %v", err)
	}
	big, err := NewPackedEngramTable(256, 400, bigData, bigScales)
	if err != nil {
		t.Fatalf("NewPackedEngramTable(big): %v", err)
	}
	idsSmall := []int32{0, 1, 2, 3}
	idsBig := []int32{0, 399}
	prealloc := make([]float32, len(idsSmall)*small.Dim())

	if n := testing.AllocsPerRun(20, func() {
		_ = small.GatherInto(prealloc, idsSmall)
	}); n != 0 {
		t.Fatalf("GatherInto allocated %v times; the packed gather must read the packed rows only", n)
	}
	// One allocation: the output slice. Identical for a 100x larger table, so
	// no per-table expansion buffer is ever materialized.
	if n := testing.AllocsPerRun(20, func() {
		_, _ = small.Gather(idsSmall)
	}); n != 1 {
		t.Fatalf("Gather allocated %v times, want exactly 1 (the output slice)", n)
	}
	if n := testing.AllocsPerRun(20, func() {
		_, _ = big.Gather(idsBig)
	}); n != 1 {
		t.Fatalf("Gather on the large table allocated %v times, want exactly 1", n)
	}
	// Contrast: the widened reference path materializes the full e4m3 copy,
	// which is the memory traffic the packed path removes.
	if n := testing.AllocsPerRun(5, func() {
		ref := small.ExpandToE4M3()
		_, _ = GatherE4M3Rows(ref, smallScales, small.Dim(), idsSmall)
	}); n < 2 {
		t.Fatalf("expanded reference allocated %v times, want at least 2 (table copy + output)", n)
	}
}

func TestEngramPackedGatherSentinelErrors(t *testing.T) {
	// fak-test:runtime fast est=1ms lane=default
	data, scales := engramFixtureBytes(t, 2)
	if _, err := NewPackedEngramTable(255, 2, data, scales); !errors.Is(err, ErrDimNotMultipleOf32) {
		t.Fatalf("dim 255: got %v, want ErrDimNotMultipleOf32", err)
	}
	if _, err := NewPackedEngramTable(256, 2, data[:10], scales); !errors.Is(err, ErrShapeMismatch) {
		t.Fatalf("short payload: got %v, want ErrShapeMismatch", err)
	}
	badScales := make([]byte, 3*8)
	badScales[len(badScales)-1] = 0xff
	if _, err := NewPackedEngramTable(256, 3, make([]byte, 3*128), badScales); !errors.Is(err, ErrInvalidScale) {
		t.Fatalf("scale 0xFF: got %v, want ErrInvalidScale", err)
	}
	table, err := NewPackedEngramTable(256, 2, data, scales)
	if err != nil {
		t.Fatalf("NewPackedEngramTable: %v", err)
	}
	if _, err := table.Gather([]int32{-1}); !errors.Is(err, ErrIDOutOfRange) {
		t.Fatalf("id -1: got %v, want ErrIDOutOfRange", err)
	}
	if _, err := table.Gather([]int32{2}); !errors.Is(err, ErrIDOutOfRange) {
		t.Fatalf("id 2 of 2 rows: got %v, want ErrIDOutOfRange", err)
	}
	if err := table.GatherInto(make([]float32, 255), []int32{0}); !errors.Is(err, ErrOutputLength) {
		t.Fatalf("short out: got %v, want ErrOutputLength", err)
	}
	reserved := append([]byte{}, table.ExpandToE4M3()...)
	reserved[5] = 0x7f
	if _, err := GatherE4M3Rows(reserved, scales, 256, []int32{0}); !errors.Is(err, ErrE4M3ReservedByte) {
		t.Fatalf("reserved e4m3 byte: got %v, want ErrE4M3ReservedByte", err)
	}
}

func TestEngramPackedGatherHalvesTableRowBytes(t *testing.T) {
	// fak-test:runtime fast est=1ms lane=default
	data, scales := engramFixtureBytes(t, 1)
	table, err := NewPackedEngramTable(256, 1, data, scales)
	if err != nil {
		t.Fatalf("NewPackedEngramTable: %v", err)
	}
	if got, want := table.PackedBytesPerRow(), 128; got != want {
		t.Fatalf("PackedBytesPerRow = %d, want %d", got, want)
	}
	if got, want := table.ExpandedBytesPerRow(), 256; got != want {
		t.Fatalf("ExpandedBytesPerRow = %d, want %d", got, want)
	}
	// The packed payload holds half the bytes of the widened e4m3 row: the
	// lookup-table storage reduction the borrow targets (software evidence
	// only, no hardware or whole-model claim).
	if table.PackedBytesPerRow()*2 != table.ExpandedBytesPerRow() {
		t.Fatalf("packed row %d bytes is not half of expanded %d bytes",
			table.PackedBytesPerRow(), table.ExpandedBytesPerRow())
	}
}
