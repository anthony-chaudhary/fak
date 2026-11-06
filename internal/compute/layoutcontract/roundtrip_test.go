package layoutcontract

import (
	"bytes"
	"math/rand"
	"testing"
)

// roundTripShapes lists, per family, the representative geometries its
// registered transform pair must invert: exact tiles, ragged tails, short row
// groups and the identity width. Every entry carrying a Transform is driven
// through the SAME uniform convention below, so a newly registered pair
// inherits the witness without new test code — and an entry that registers a
// pair but no shapes fails loudly instead of skipping silently.
var roundTripShapes = map[string][]Shape{
	"Q4_K": {
		{Rows: 5, Cols: 256, Width: 1}, // identity width, one super-block per row
		{Rows: 5, Cols: 512, Width: 4}, // two super-blocks, tail group of 1 row
		{Rows: 3, Cols: 256, Width: 8}, // short tail group (3 rows mod 8)
		{Rows: 8, Cols: 768, Width: 8}, // full 8-row groups, three super-blocks
	},
	"NVFP4": {
		{Rows: 1, Cols: 1},     // smallest legal payload, fully padded tile
		{Rows: 128, Cols: 4},   // exactly one 128x4 tile, no padding
		{Rows: 130, Cols: 7},   // ragged tail row and column tiles
		{Rows: 256, Cols: 100}, // two row tiles, ragged columns
	},
}

// payloadBytes sizes one deterministic pseudo-random payload for the family's
// transform input: Q4_K takes the raw row-major GGUF payload, NVFP4 takes the
// linear block-scale array the swizzle permutes.
func payloadBytes(family string, s Shape) []byte {
	var n int
	switch family {
	case "Q4_K":
		n = s.Rows * (s.Cols / 256) * 144
	case "NVFP4":
		n = s.Rows * s.Cols
	default:
		return nil
	}
	r := rand.New(rand.NewSource(int64(len(family)*1000 + n)))
	b := make([]byte, n)
	r.Read(b)
	return b
}

// fak-test:runtime fast est=150ms lane=default
func TestRegisteredTransformRoundTrip(t *testing.T) {
	drove := 0
	for _, c := range All() {
		if !c.Transform.Present() {
			continue
		}
		shapes := roundTripShapes[c.Family]
		if len(shapes) == 0 {
			t.Fatalf("%s registers a transform pair but no round-trip shapes", c.Family)
		}
		resolved, err := Get(c.Family)
		if err != nil {
			t.Fatalf("Get(%q) refused a registered transform family: %v", c.Family, err)
		}
		lookups := []struct {
			name      string
			transform Transform
		}{
			{"Get", resolved.Transform},
			{"Must", Must(c.Family).Transform},
		}
		for _, lookup := range lookups {
			if !lookup.transform.Present() {
				t.Fatalf("%s(%q) returned an incomplete transform pair", lookup.name, c.Family)
			}
		}
		for _, s := range shapes {
			raw := payloadBytes(c.Family, s)
			if raw == nil {
				t.Fatalf("%s: no payload sizing for shape %v", c.Family, s)
			}
			fwd, err := c.Transform.Forward(s, raw)
			if err != nil {
				t.Fatalf("%s: Forward(%v) failed: %v", c.Family, s, err)
			}
			back, err := c.Transform.Inverse(s, fwd)
			if err != nil {
				t.Fatalf("%s: Inverse(Forward(%v)) failed: %v", c.Family, s, err)
			}
			if !bytes.Equal(back, raw) {
				t.Fatalf("%s: Inverse o Forward is not the byte identity at %v (%d bytes)", c.Family, s, len(raw))
			}
			again, err := c.Transform.Forward(s, back)
			if err != nil {
				t.Fatalf("%s: Forward(Inverse(Forward(%v))) failed: %v", c.Family, s, err)
			}
			if !bytes.Equal(again, fwd) {
				t.Fatalf("%s: Forward is not stable under a round trip at %v", c.Family, s)
			}
			// Lookups must preserve transform behavior, not just metadata and
			// non-nil function fields.
			for _, lookup := range lookups {
				gotForward, err := lookup.transform.Forward(s, raw)
				if err != nil || !bytes.Equal(gotForward, fwd) {
					t.Fatalf("%s(%q): Forward(%v) differs from All: err=%v", lookup.name, c.Family, s, err)
				}
				gotInverse, err := lookup.transform.Inverse(s, fwd)
				if err != nil || !bytes.Equal(gotInverse, back) {
					t.Fatalf("%s(%q): Inverse(%v) differs from All: err=%v", lookup.name, c.Family, s, err)
				}
			}
			drove++
		}
	}
	if drove == 0 {
		t.Fatal("no registered transform pair was driven; the round-trip witness is vacuous")
	}
}
