package ggufload

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

// TestMXFP4Repack is the anthony-chaudhary/fak#14456 witness: table-driven
// accept/refuse cases for every RepackMXFP4 invariant group. Refusals are
// matched with errors.Is against the closed sentinel set, never message prose.
func TestMXFP4Repack(t *testing.T) {
	// Sequential source pattern: byte j holds element 2j = j (low) and element
	// 2j+1 = j+1 (high), so element i carries value (i+1)/2 mod 16 (all 16
	// E2M1 codes are valid). Expectations derive from that source rule -
	// element k in the low nibble, element k+16 in the high - never from the
	// repack arithmetic.
	single := func() (codes, scales []byte) {
		codes = make([]byte, 16)
		for j := range codes {
			codes[j] = byte(j) | byte(j+1)<<4
		}
		return codes, []byte{0x7F}
	}

	t.Run("valid single block", func(t *testing.T) {
		codes, scales := single()
		got, err := RepackMXFP4(codes, scales)
		if err != nil {
			t.Fatalf("RepackMXFP4 valid input: %v", err)
		}
		if len(got) != blockMXFP4Bytes {
			t.Fatalf("repacked length = %d, want %d", len(got), blockMXFP4Bytes)
		}
		elem := func(i int) int {
			return (i + 1) / 2 & 0xF
		}
		want := make([]byte, blockMXFP4Bytes)
		want[0] = 0x7F
		for k := 0; k < qkMXFP4/2; k++ {
			want[1+k] = byte(elem(k)) | byte(elem(k+qkMXFP4/2))<<4
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("block byte %d = %#x, want %#x", i, got[i], want[i])
			}
		}
	})

	t.Run("empty input repacks empty", func(t *testing.T) {
		got, err := RepackMXFP4(nil, nil)
		if err != nil {
			t.Fatalf("RepackMXFP4 empty: %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("empty repack = %d bytes, want 0", len(got))
		}
	})

	refuse := []struct {
		name   string
		codes  int
		scales int
		want   error
	}{
		{"misaligned 15 codes", 15, 0, ErrMXFP4MisalignedSource},
		{"misaligned 17 codes", 17, 1, ErrMXFP4MisalignedSource},
		{"misaligned 31 codes", 31, 1, ErrMXFP4MisalignedSource},
		{"scale count short", 32, 1, ErrMXFP4ScaleShapeMismatch},
		{"scale count long", 16, 2, ErrMXFP4ScaleShapeMismatch},
		{"nonfinite scale first block", 32, -1, ErrMXFP4NonfiniteScale},
		{"nonfinite scale second block", 32, -2, ErrMXFP4NonfiniteScale},
	}
	for _, tc := range refuse {
		t.Run(tc.name, func(t *testing.T) {
			codes := make([]byte, tc.codes)
			for i := range codes {
				codes[i] = byte(i)
			}
			var scales []byte
			switch tc.scales {
			case -1:
				scales = []byte{0xFF, 0x7F}
			case -2:
				scales = []byte{0x7F, 0xFF}
			default:
				scales = make([]byte, tc.scales)
			}
			_, err := RepackMXFP4(codes, scales)
			if !errors.Is(err, tc.want) {
				t.Fatalf("RepackMXFP4 error = %v, want sentinel %v", err, tc.want)
			}
		})
	}

	t.Run("alignment checked before scale shape", func(t *testing.T) {
		_, err := RepackMXFP4(make([]byte, 15), nil)
		if !errors.Is(err, ErrMXFP4MisalignedSource) {
			t.Fatalf("error = %v, want %v", err, ErrMXFP4MisalignedSource)
		}
	})
}

// TestMXFP4RepackRoundTrip proves decode(repack(source)) == source through the
// existing dequant path: random codes and finite scales, repacked, decoded by
// dequantMXFP4, compared bit-exactly against the source expanded through the
// same E2M1/E8M0 tables.
func TestMXFP4RepackRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(14456))
	const blocks = 64
	codes := make([]byte, blocks*16)
	scales := make([]byte, blocks)
	for i := range codes {
		codes[i] = byte(rng.Intn(256))
	}
	for i := range scales {
		scales[i] = byte(rng.Intn(255)) // 0..254, never the nonfinite 255
	}
	got, err := RepackMXFP4(codes, scales)
	if err != nil {
		t.Fatalf("RepackMXFP4: %v", err)
	}
	if len(got) != blocks*blockMXFP4Bytes {
		t.Fatalf("repacked length = %d, want %d", len(got), blocks*blockMXFP4Bytes)
	}
	out := make([]float32, blocks*qkMXFP4)
	dequantMXFP4(out, got)
	for b := 0; b < blocks; b++ {
		d := e8m0ToF32Half(scales[b])
		for j := 0; j < qkMXFP4/2; j++ {
			wantLow := kvaluesMXFP4[codes[b*16+j]&0x0f] * d // element 2j
			wantHi := kvaluesMXFP4[codes[b*16+j]>>4] * d    // element 2j+1
			if out[b*qkMXFP4+2*j] != wantLow {
				t.Fatalf("block %d element %d: dequant=%#x want=%#x", b, 2*j, math.Float32bits(out[b*qkMXFP4+2*j]), math.Float32bits(wantLow))
			}
			if out[b*qkMXFP4+2*j+1] != wantHi {
				t.Fatalf("block %d element %d: dequant=%#x want=%#x", b, 2*j+1, math.Float32bits(out[b*qkMXFP4+2*j+1]), math.Float32bits(wantHi))
			}
		}
	}
}

// TestMXFP4RepackNibbleRouting pins the deterministic golden: source byte j
// carries element 2j = j (low) and element 2j+1 = 15-j (high); the expected
// GGUF payload byte k is derived from that source rule - element k in the low
// nibble, element k+16 in the high nibble - never from the repack arithmetic.
func TestMXFP4RepackNibbleRouting(t *testing.T) {
	codes := make([]byte, 16)
	for j := range codes {
		codes[j] = byte(j) | byte(15-j)<<4
	}
	got, err := RepackMXFP4(codes, []byte{0x80})
	if err != nil {
		t.Fatalf("RepackMXFP4: %v", err)
	}
	if got[0] != 0x80 {
		t.Fatalf("scale byte = %#x, want 0x80 (unchanged)", got[0])
	}
	elem := func(i int) int { // source layout: even i = i/2, odd i = 15-(i-1)/2
		if i%2 == 0 {
			return i / 2
		}
		return 15 - (i-1)/2
	}
	for k := 0; k < qkMXFP4/2; k++ {
		want := byte(elem(k)) | byte(elem(k+qkMXFP4/2))<<4
		if got[1+k] != want {
			t.Fatalf("payload byte %d = %#x, want %#x", k, got[1+k], want)
		}
	}
}

// TestMXFP4RepackF16Gate covers the F16-only-if-exactly-representable
// invariant: accepted values survive a binary16 round trip bit-exactly,
// rejected values do not.
func TestMXFP4RepackF16Gate(t *testing.T) {
	accept := []float32{0, -0, 1, -1, 0.5, 2, 3, 6, 0.1875, 65504, float32(math.Inf(1))}
	reject := []float32{0.1, 65536, 1e-30, float32(math.NaN()), 65504 + 0.0078125}
	for _, v := range accept {
		if !F16ExactlyRepresentable([]float32{v}) {
			t.Errorf("F16ExactlyRepresentable(%v) = false, want true", v)
		}
	}
	for _, v := range reject {
		if F16ExactlyRepresentable([]float32{v}) {
			t.Errorf("F16ExactlyRepresentable(%v) = true, want false", v)
		}
	}
	if F16ExactlyRepresentable(append(accept, 0.1)) {
		t.Error("one inexact value must poison the whole slice")
	}
	if !F16ExactlyRepresentable(nil) {
		t.Error("empty slice is vacuously exactly representable")
	}
}

// TestMXFP4RepackSpotCheckRecipe pins the upstream validator recipe
// {0, (layer*17+41)%count, count-1} with deduplication.
func TestMXFP4RepackSpotCheckRecipe(t *testing.T) {
	cases := []struct {
		layer, count int
		want         []int
	}{
		{0, 1, []int{0}},
		{0, 2, []int{0, 1}},
		{7, 2, []int{0, 1}},
		{0, 64, []int{0, 41, 63}},
		{1, 64, []int{0, 58, 63}},
		{2, 64, []int{0, 11, 63}},
		{5, 64, []int{0, 62, 63}},
		{39, 64, []int{0, 63}}, // mid collapses to 0: deduplicated
		{-1, 64, nil},
		{0, 0, nil},
	}
	for _, tc := range cases {
		got := SpotCheckExpertIndices(tc.layer, tc.count)
		if len(got) != len(tc.want) {
			t.Fatalf("SpotCheckExpertIndices(%d,%d) = %v, want %v", tc.layer, tc.count, got, tc.want)
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Fatalf("SpotCheckExpertIndices(%d,%d)[%d] = %d, want %d", tc.layer, tc.count, i, got[i], tc.want[i])
			}
		}
	}
}
