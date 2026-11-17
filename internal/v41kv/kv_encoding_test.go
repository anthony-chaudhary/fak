package v41kv

import (
	"errors"
	"testing"
)

// TestKVWireEncodingClosedSet pins the closed encoding set and its scale
// groups: raw has no group, fp4-e4m3 groups 16, fp4-e8m0 groups 32, and any
// encoding outside the set fails closed on both the sizing and the payload
// contract.
func TestKVWireEncodingClosedSet(t *testing.T) {
	if g, ok := V41KVWireRaw.Group(); ok || g != 0 {
		t.Fatalf("raw group = (%d, %v), want (0, false)", g, ok)
	}
	if g, ok := V41KVWireFp4E4m3.Group(); !ok || g != 16 {
		t.Fatalf("fp4-e4m3 group = (%d, %v), want (16, true)", g, ok)
	}
	if g, ok := V41KVWireFp4E8m0.Group(); !ok || g != 32 {
		t.Fatalf("fp4-e8m0 group = (%d, %v), want (32, true)", g, ok)
	}
	if _, ok := V41KVWireEncoding(99).Group(); ok {
		t.Fatal("unlisted encoding resolved to a group")
	}
	if _, err := (V41KVWireSegment{DecodedBytes: 32, Encoding: V41KVWireEncoding(99)}).StoredBytes(); !errors.Is(err, ErrV41KVWireEncoding) {
		t.Fatalf("unlisted encoding StoredBytes err = %v, want ErrV41KVWireEncoding", err)
	}
	if err := (V41KVWireSegment{DecodedBytes: 32, Encoding: V41KVWireEncoding(99)}).ValidatePayload(make([]byte, 9)); !errors.Is(err, ErrV41KVWireEncoding) {
		t.Fatalf("unlisted encoding ValidatePayload err = %v, want ErrV41KVWireEncoding", err)
	}
}

// TestKVWireEncodingStoredBytes pins the closed sizing contract: raw stores its
// decoded bytes as-is; an fp4 form stores decoded/4 payload bytes plus
// decoded/(2*group) scale bytes; a decoded count that does not hold a whole
// number of scale groups is refused. The fp4-e4m3 charge at the V4.1 width must
// equal the 288 B/token geometry this package already pins.
func TestKVWireEncodingStoredBytes(t *testing.T) {
	rawCases := []struct {
		decoded, stored int
	}{
		{0, 0},
		{1024, 1024},
		{1 << 30, 1 << 30},
	}
	for _, c := range rawCases {
		got, err := V41KVWireSegmentRaw(c.decoded).StoredBytes()
		if err != nil {
			t.Fatalf("raw(%d) StoredBytes err = %v", c.decoded, err)
		}
		if got != c.stored {
			t.Fatalf("raw(%d) stored = %d, want %d", c.decoded, got, c.stored)
		}
	}

	fp4e4m3Cases := []struct {
		decoded int
		stored  int
	}{
		{0, 0},
		{32, 9},
		{1024, 288},
	}
	for _, c := range fp4e4m3Cases {
		seg := V41KVWireSegment{DecodedBytes: c.decoded, Encoding: V41KVWireFp4E4m3}
		got, err := seg.StoredBytes()
		if err != nil {
			t.Fatalf("fp4-e4m3(%d) StoredBytes err = %v", c.decoded, err)
		}
		if got != c.stored {
			t.Fatalf("fp4-e4m3(%d) stored = %d, want %d", c.decoded, got, c.stored)
		}
	}

	fp4e8m0Cases := []struct {
		decoded int
		stored  int
	}{
		{0, 0},
		{64, 17},
		{256, 68},
	}
	for _, c := range fp4e8m0Cases {
		seg := V41KVWireSegment{DecodedBytes: c.decoded, Encoding: V41KVWireFp4E8m0}
		got, err := seg.StoredBytes()
		if err != nil {
			t.Fatalf("fp4-e8m0(%d) StoredBytes err = %v", c.decoded, err)
		}
		if got != c.stored {
			t.Fatalf("fp4-e8m0(%d) stored = %d, want %d", c.decoded, got, c.stored)
		}
	}

	// The fp4-e4m3 charge at the V4.1 width (512 values = 1024 decoded bytes,
	// group 16) must reproduce the pinned V4.1 fp4 record size.
	pinned, ok := V41KVCacheLayoutFor(V41KVFP4)
	if !ok {
		t.Fatal("fp4 layout is not in the published set")
	}
	wire, err := V41KVWireSegment{DecodedBytes: 2 * (pinned.DNoPE + pinned.DRoPE), Encoding: V41KVWireFp4E4m3}.StoredBytes()
	if err != nil {
		t.Fatalf("wire fp4-e4m3 at V4.1 width err = %v", err)
	}
	if wire != pinned.BytesPerToken {
		t.Fatalf("wire fp4-e4m3 at V4.1 width = %d B/token, pinned layout = %d", wire, pinned.BytesPerToken)
	}

	// Unaligned decoded counts and negative counts fail closed.
	for _, seg := range []V41KVWireSegment{
		{DecodedBytes: 1023, Encoding: V41KVWireFp4E4m3},
		{DecodedBytes: 31, Encoding: V41KVWireFp4E4m3},
		{DecodedBytes: 255, Encoding: V41KVWireFp4E8m0},
		{DecodedBytes: 63, Encoding: V41KVWireFp4E8m0},
		{DecodedBytes: -1, Encoding: V41KVWireRaw},
		{DecodedBytes: -32, Encoding: V41KVWireFp4E4m3},
	} {
		if _, err := seg.StoredBytes(); !errors.Is(err, ErrV41KVWireEncoding) {
			t.Fatalf("%+v StoredBytes err = %v, want ErrV41KVWireEncoding", seg, err)
		}
	}
}

// TestKVWireEncodingScaleValidator pins the closed scale-code ranges: e4m3
// scales must sit in 1..=126 (0 is the zero scale, 127 the NaN code, 128 and
// up negative codes) and e8m0 scales in 1..=254 (255 is the NaN byte). A
// payload whose length disagrees with the sizing contract is refused before
// any scale is read.
func TestKVWireEncodingScaleValidator(t *testing.T) {
	// fp4-e4m3, one group: 32 decoded bytes -> 8 payload + 1 scale.
	e4m3 := V41KVWireSegment{DecodedBytes: 32, Encoding: V41KVWireFp4E4m3}
	e4m3Scale := func(scale byte) []byte {
		p := make([]byte, 9)
		p[8] = scale
		return p
	}
	for _, scale := range []byte{0, 127, 128, 255} {
		if err := e4m3.ValidatePayload(e4m3Scale(scale)); !errors.Is(err, ErrV41KVWireEncoding) {
			t.Fatalf("e4m3 scale %d err = %v, want ErrV41KVWireEncoding", scale, err)
		}
	}
	for _, scale := range []byte{1, 126} {
		if err := e4m3.ValidatePayload(e4m3Scale(scale)); err != nil {
			t.Fatalf("e4m3 scale %d err = %v, want nil", scale, err)
		}
	}

	// fp4-e8m0, one group: 64 decoded bytes -> 16 payload + 1 scale.
	e8m0 := V41KVWireSegment{DecodedBytes: 64, Encoding: V41KVWireFp4E8m0}
	e8m0Scale := func(scale byte) []byte {
		p := make([]byte, 17)
		p[16] = scale
		return p
	}
	for _, scale := range []byte{0, 255} {
		if err := e8m0.ValidatePayload(e8m0Scale(scale)); !errors.Is(err, ErrV41KVWireEncoding) {
			t.Fatalf("e8m0 scale %d err = %v, want ErrV41KVWireEncoding", scale, err)
		}
	}
	for _, scale := range []byte{1, 127, 128, 254} {
		if err := e8m0.ValidatePayload(e8m0Scale(scale)); err != nil {
			t.Fatalf("e8m0 scale %d err = %v, want nil", scale, err)
		}
	}

	// Length mismatches fail closed for every encoding.
	for _, seg := range []V41KVWireSegment{
		{DecodedBytes: 32, Encoding: V41KVWireFp4E4m3},
		{DecodedBytes: 64, Encoding: V41KVWireFp4E8m0},
		{DecodedBytes: 1024, Encoding: V41KVWireRaw},
	} {
		want, err := seg.StoredBytes()
		if err != nil {
			t.Fatalf("%+v StoredBytes err = %v", seg, err)
		}
		if err := seg.ValidatePayload(make([]byte, want-1)); !errors.Is(err, ErrV41KVWireEncoding) {
			t.Fatalf("%+v short payload err = %v, want ErrV41KVWireEncoding", seg, err)
		}
		if err := seg.ValidatePayload(make([]byte, want+1)); !errors.Is(err, ErrV41KVWireEncoding) {
			t.Fatalf("%+v long payload err = %v, want ErrV41KVWireEncoding", seg, err)
		}
	}

	// Raw carries no scale row: an exact-length payload validates regardless of
	// content, and the empty segment validates empty.
	raw := V41KVWireSegmentRaw(32)
	rawPayload := make([]byte, 32)
	rawPayload[0], rawPayload[1], rawPayload[2], rawPayload[3] = 0, 0xff, 0x7f, 0x80
	if err := raw.ValidatePayload(rawPayload); err != nil {
		t.Fatalf("raw exact-length payload err = %v, want nil", err)
	}
	if err := (V41KVWireSegment{DecodedBytes: 0, Encoding: V41KVWireFp4E4m3}).ValidatePayload(nil); err != nil {
		t.Fatalf("empty fp4-e4m3 segment err = %v, want nil", err)
	}
	if err := (V41KVWireSegment{DecodedBytes: 0, Encoding: V41KVWireFp4E8m0}).ValidatePayload(nil); err != nil {
		t.Fatalf("empty fp4-e8m0 segment err = %v, want nil", err)
	}
}

// TestKVWireEncodingBF16OverflowBound pins the finite-bf16 overflow bound on
// e8m0 scales at or above 252: every fp4 value the scale can produce must stay
// at or below the largest finite bf16 magnitude. The boundary is exact in
// f32: at scale 252 the largest magnitude (6) still fits; at 253 magnitudes 4
// and 6 overflow; at 254 only magnitudes up to 1.5 fit.
func TestKVWireEncodingBF16OverflowBound(t *testing.T) {
	seg := V41KVWireSegment{DecodedBytes: 64, Encoding: V41KVWireFp4E8m0}
	payload := func(first byte, scale byte) []byte {
		p := make([]byte, 17)
		p[0] = first
		p[16] = scale
		return p
	}
	cases := []struct {
		first, scale byte
		wantErr      bool
	}{
		{0x77, 252, false}, // 6 x 2^125 fits finite bf16
		{0x77, 253, true},  // 6 x 2^126 overflows
		{0x66, 253, true},  // 4 x 2^126 = 2^128 overflows f32
		{0x55, 253, false}, // 3 x 2^126 fits
		{0x44, 254, true},  // 2 x 2^127 = 2^128 overflows f32
		{0x33, 254, false}, // 1.5 x 2^127 fits
		{0x77, 254, true},  // 6 x 2^127 overflows
	}
	for _, c := range cases {
		err := seg.ValidatePayload(payload(c.first, c.scale))
		if c.wantErr && !errors.Is(err, ErrV41KVWireEncoding) {
			t.Fatalf("first 0x%02x scale %d err = %v, want ErrV41KVWireEncoding", c.first, c.scale, err)
		}
		if !c.wantErr && err != nil {
			t.Fatalf("first 0x%02x scale %d err = %v, want nil", c.first, c.scale, err)
		}
	}
}
