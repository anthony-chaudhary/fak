package v41kv

import (
	"testing"
)

// TestV41KVCacheRecordFieldOffsets pins the published record field order at
// absolute byte offsets: NoPE payload, then RoPE payload, then the scale row
// with NoPE tile scales before RoPE tile scales. The round-trip tests only
// prove the quantizer and dequantizer agree with each other, so a consistent
// mis-addressing on both sides would pass them; this witness fixes the layout
// against independently derived bytes instead.
//
// Each case marks the head of the NoPE and RoPE parts with one value chosen so
// its tile scale is the identity: fp8 amax 448 keeps the e8m0 scale at byte
// 127 (2^0), fp4 amax 6 makes the e4m3 scale exactly 1.0 (0x38). The marker
// payload bytes are then raw element codes: 448 -> e4m3 0x7e, 6 -> e2m1 nibble
// 0x7 packed low-first. fp8 zero tiles encode clean 0x00 payload bytes; fp4
// zero-tile nibbles ride the out-of-contract 0/0 scale path, so they are
// checked through the decoder (every unmarked element must come back exactly
// zero) rather than pinned as raw bytes.
// fak-test:runtime fast est=50ms lane=default
// Estimated only, not measured: bounded in-memory KV record cases.
func TestV41KVCacheRecordFieldOffsets(t *testing.T) {
	// fp8: 448 NoPE + 64 RoPE payload bytes, then 16 e8m0 scale bytes.
	fp8, ok := V41KVCacheLayoutFor(V41KVFP8)
	if !ok {
		t.Fatal("fp8 layout is not in the published set")
	}
	nope := make([]float32, fp8.DNoPE)
	rope := make([]float32, fp8.DRoPE)
	nope[0], rope[0] = 448, 448
	record, err := fp8.V41KVQuantizeRow(nope, rope)
	if err != nil {
		t.Fatal(err)
	}
	v41KVAssertByte(t, "fp8 NoPE marker", record, 0, 0x7e)
	v41KVAssertByte(t, "fp8 RoPE marker", record, fp8.DNoPE, 0x7e)
	v41KVAssertZeroRange(t, "fp8 unmarked NoPE payload", record, 1, fp8.DNoPE)
	v41KVAssertZeroRange(t, "fp8 unmarked RoPE payload", record, fp8.DNoPE+1, fp8.DNoPE+fp8.DRoPE)
	// Every tile carries amax <= 448, so all 16 e8m0 scale bytes are the unit
	// code 127, across the NoPE/RoPE tile boundary alike.
	for i := 0; i < fp8.NumTiles; i++ {
		v41KVAssertByte(t, "fp8 tile scale", record, fp8.DNoPE+fp8.DRoPE+i, 127)
	}

	// fp4: 224 packed NoPE + 32 packed RoPE payload bytes, then 32 e4m3 scale
	// bytes (28 NoPE tiles at tile 16, then 4 RoPE tiles).
	fp4, ok := V41KVCacheLayoutFor(V41KVFP4)
	if !ok {
		t.Fatal("fp4 layout is not in the published set")
	}
	nope = make([]float32, fp4.DNoPE)
	rope = make([]float32, fp4.DRoPE)
	nope[0], rope[0] = 6, 6
	record, err = fp4.V41KVQuantizeRow(nope, rope)
	if err != nil {
		t.Fatal(err)
	}
	packedNoPE := fp4.DNoPE / 2
	packedRoPE := fp4.DRoPE / 2
	v41KVAssertByte(t, "fp4 NoPE marker", record, 0, 0x07)
	v41KVAssertByte(t, "fp4 RoPE marker", record, packedNoPE, 0x07)
	scaleStart := packedNoPE + packedRoPE
	nopeTiles := fp4.DNoPE / fp4.TileSize
	v41KVAssertByte(t, "fp4 NoPE tile 0 scale", record, scaleStart, 0x38)
	v41KVAssertByte(t, "fp4 RoPE tile 0 scale", record, scaleStart+nopeTiles, 0x38)
	v41KVAssertZeroRange(t, "fp4 unmarked NoPE scales", record, scaleStart+1, scaleStart+nopeTiles)
	v41KVAssertZeroRange(t, "fp4 unmarked RoPE scales", record, scaleStart+nopeTiles+1, fp4.BytesPerToken)
	row, err := fp4.V41KVDequantizeRow(record)
	if err != nil {
		t.Fatal(err)
	}
	if row.NoPE[0] != 6 || row.RoPE[0] != 6 {
		t.Fatalf("fp4 markers = %g/%g, want 6/6", row.NoPE[0], row.RoPE[0])
	}
	for i := 1; i < len(row.NoPE); i++ {
		if row.NoPE[i] != 0 {
			t.Fatalf("fp4 NoPE[%d] = %g, want 0", i, row.NoPE[i])
		}
	}
	for i := 1; i < len(row.RoPE); i++ {
		if row.RoPE[i] != 0 {
			t.Fatalf("fp4 RoPE[%d] = %g, want 0", i, row.RoPE[i])
		}
	}
}

// v41KVAssertByte requires one record byte to hold the pinned code.
func v41KVAssertByte(t *testing.T, what string, record []byte, off, want int) {
	t.Helper()
	if off < 0 || off >= len(record) {
		t.Fatalf("%s: offset %d out of bounds (%d bytes)", what, off, len(record))
	}
	if got := record[off]; got != byte(want) {
		t.Fatalf("%s: record[%d] = %#04x, want %#04x", what, off, got, want)
	}
}

// v41KVAssertZeroRange requires a half-open payload or scale span to be all
// zero, catching a marker that leaked into the wrong field.
func v41KVAssertZeroRange(t *testing.T, what string, record []byte, from, to int) {
	t.Helper()
	if from < 0 || to > len(record) {
		t.Fatalf("%s: span [%d,%d) exceeds record length %d", what, from, to, len(record))
	}
	for i := from; i < to; i++ {
		if record[i] != 0 {
			t.Fatalf("%s: record[%d] = %#04x, want 0", what, i, record[i])
		}
	}
}
