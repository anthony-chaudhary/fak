package model

import (
	"math"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
// Runtime estimate is unmeasured. The existing Flash official-pair loader tests
// exercise the production caller; this witness pins edge-tile conversion and
// the finite-pair refusal against the preserved scalar decoder.
func TestV4DenseFP8TiledQuantizeMatchesReference(t *testing.T) {
	const rows, cols = 129, 160
	shape := []int{rows, cols}
	weights := make([]byte, rows*cols)
	for i := range weights {
		weights[i] = byte(i % 256)
		if weights[i]&0x7f == 0x7f {
			weights[i]--
		}
	}
	// Includes the smallest E8M0 scale, a sub-unit scale and the largest
	// power-of-two scale for which every finite E4M3 product remains finite.
	scales := []byte{0, 119, 127, 246}
	expanded, err := decodeV4DenseFP8("oracle", shape, weights, scales)
	if err != nil {
		t.Fatal(err)
	}
	want := quantizeQ8(expanded, rows, cols)
	got, err := quantizeV4DenseFP8Q8("tiled", shape, weights, scales)
	if err != nil {
		t.Fatal(err)
	}
	if got.out != want.out || got.in != want.in || got.nblk != want.nblk {
		t.Fatal("Q8 geometry changed")
	}
	for i, q := range want.q {
		if got.q[i] != q {
			t.Fatalf("Q8 code[%d] = %d, want %d", i, got.q[i], q)
		}
	}
	for i, d := range want.d {
		if math.Float32bits(got.d[i]) != math.Float32bits(d) {
			t.Fatalf("Q8 scale[%d] = %08x, want %08x", i, math.Float32bits(got.d[i]), math.Float32bits(d))
		}
	}

	for _, tc := range []struct {
		name      string
		weight    byte
		scale     byte
		wantError string
	}{
		{"positive weight NaN", 0x7f, 127, "E4M3 NaN"},
		{"negative weight NaN", 0xff, 127, "E4M3 NaN"},
		{"scale NaN", 0x38, 0xff, "E8M0 NaN"},
		{"weight refusal precedes scale", 0x7f, 0xff, "E4M3 NaN"},
		{"finite operands overflow", 0x7e, 254, "non-finite"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w := append([]byte(nil), weights...)
			s := append([]byte(nil), scales...)
			w[len(w)-1], s[len(s)-1] = tc.weight, tc.scale
			_, refErr := decodeV4DenseFP8("bad", shape, w, s)
			q, err := quantizeV4DenseFP8Q8("bad", shape, w, s)
			if refErr == nil || err == nil || !strings.Contains(err.Error(), tc.wantError) || err.Error() != refErr.Error() || q != nil {
				t.Fatalf("refusal = %v (Q8=%v), reference = %v", err, q != nil, refErr)
			}
		})
	}

	t.Run("malformed geometry", func(t *testing.T) {
		for _, tc := range []struct {
			shape   []int
			weights []byte
			scales  []byte
		}{
			{nil, nil, nil},
			{[]int{0, cols}, nil, nil},
			{shape, weights[:len(weights)-1], scales},
			{shape, weights, scales[:len(scales)-1]},
			{[]int{1, 33}, make([]byte, 33), []byte{127}},
			{[]int{int(^uint(0) >> 1), cols}, nil, nil},
		} {
			if q, err := quantizeV4DenseFP8Q8("bad", tc.shape, tc.weights, tc.scales); err == nil || q != nil {
				t.Fatalf("accepted malformed shape %v", tc.shape)
			}
		}
	})
}
