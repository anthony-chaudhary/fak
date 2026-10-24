package compute

import (
	"bytes"
	"errors"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestNVFP4ScaleLayoutUpstreamIndexGoldens(t *testing.T) {
	const (
		rows      = 129
		scaleCols = 5
	)
	linear := make([]byte, rows*scaleCols)
	for i := range linear {
		linear[i] = byte(i%251 + 1)
	}

	swizzled, err := SwizzleNVFP4Scales(rows, scaleCols, linear)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(swizzled), 256*8; got != want {
		t.Fatalf("swizzled length = %d, want %d", got, want)
	}

	goldens := []struct {
		row, col int
		offset   int
	}{
		{0, 0, 0},
		{1, 0, 16},
		{31, 3, 499},
		{32, 0, 4},
		{127, 3, 511},
		{0, 4, 512},
		{128, 0, 1024},
		{128, 4, 1536},
	}
	for _, golden := range goldens {
		want := linear[golden.row*scaleCols+golden.col]
		if got := swizzled[golden.offset]; got != want {
			t.Errorf("scale[%d,%d] at offset %d = %d, want %d", golden.row, golden.col, golden.offset, got, want)
		}
	}

	for row := 0; row < 256; row++ {
		for col := 0; col < 8; col++ {
			if row < rows && col < scaleCols {
				continue
			}
			if got := swizzled[nvfp4ScaleGoldenOffset(row, col, 2)]; got != 0 {
				t.Fatalf("padding[%d,%d] = %d, want zero", row, col, got)
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNVFP4ScaleLayoutRoundTripAndValidation(t *testing.T) {
	const (
		rows      = 257
		scaleCols = 9
	)
	linear := make([]byte, rows*scaleCols)
	for i := range linear {
		linear[i] = byte((i*29 + 7) % 256)
	}
	first := linear[0]

	swizzled, err := SwizzleNVFP4Scales(rows, scaleCols, linear)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(swizzled), 384*12; got != want {
		t.Fatalf("swizzled length = %d, want %d", got, want)
	}
	linear[0]++
	if got := swizzled[0]; got != first {
		t.Fatalf("swizzled aliases input: first byte = %d, want %d", got, first)
	}
	linear[0] = first

	roundTrip, err := UnswizzleNVFP4Scales(rows, scaleCols, swizzled)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(roundTrip, linear) {
		t.Fatal("round trip differs from linear scales")
	}
	swizzled[0]++
	if got := roundTrip[0]; got != first {
		t.Fatalf("linear result aliases swizzled input: first byte = %d, want %d", got, first)
	}
	swizzled[0]--

	nonCanonical := bytes.Clone(swizzled)
	nonCanonical[nvfp4ScaleGoldenOffset(rows, 0, 3)] = 1
	if _, err := UnswizzleNVFP4Scales(rows, scaleCols, nonCanonical); !errors.Is(err, ErrNVFP4Scale) {
		t.Fatalf("non-zero row padding error = %v, want ErrNVFP4Scale", err)
	}
	nonCanonical = bytes.Clone(swizzled)
	nonCanonical[nvfp4ScaleGoldenOffset(0, scaleCols, 3)] = 1
	if _, err := UnswizzleNVFP4Scales(rows, scaleCols, nonCanonical); !errors.Is(err, ErrNVFP4Scale) {
		t.Fatalf("non-zero column padding error = %v, want ErrNVFP4Scale", err)
	}

	for _, tc := range []struct {
		name            string
		rows, scaleCols int
		linear          []byte
		want            error
	}{
		{"zero rows", 0, 1, nil, ErrNVFP4Shape},
		{"zero columns", 1, 0, nil, ErrNVFP4Shape},
		{"linear length", 1, 1, nil, ErrNVFP4Scale},
		{"element overflow", int(^uint(0) >> 1), 2, nil, ErrNVFP4Shape},
		{"row padding overflow", int(^uint(0) >> 1), 1, nil, ErrNVFP4Shape},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := SwizzleNVFP4Scales(tc.rows, tc.scaleCols, tc.linear); !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
		})
	}
	if _, err := UnswizzleNVFP4Scales(1, 1, make([]byte, 511)); !errors.Is(err, ErrNVFP4Scale) {
		t.Fatalf("short swizzled error = %v, want ErrNVFP4Scale", err)
	}
}

// nvfp4ScaleGoldenOffset deliberately duplicates the pinned upstream formula,
// keeping the layout golden independent of the production index helper.
func nvfp4ScaleGoldenOffset(row, col, tileCols int) int {
	return ((row/128)*tileCols+col/4)*512 + (row%32)*16 + ((row%128)/32)*4 + col%4
}
