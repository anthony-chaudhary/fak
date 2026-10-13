package compute

import (
	"encoding/binary"
	"os"
	"strings"
	"testing"
)

// CPU byte-layout and source-binding witness only. It does not compile GLSL,
// execute a driver, or establish physical scalar/panel floating-point parity.
// fak-test:runtime fast est=20ms lane=default
func TestQ6KDecodeFinalScaleWord(t *testing.T) {
	source, err := os.ReadFile("shaders/q6k_matmul.comp")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	for _, required := range []string{
		"const uint ROW_WORDS = 53u;",
		"uint start = ((row0 + r) * blocks + sb) * 210u;",
		"RawS[r * ROW_WORDS + lid] = raw[(start >> 2u) + lid];",
		"uint lead = ((row * blocks + sb) * 210u) & 3u;",
		"float d = unpackHalf2x16((RawS[rowBase + 52u] >> (8u * lead)) & 0xffffu).x;",
		"float d = halfAt(base + 208u);",
	} {
		if !strings.Contains(text, required) {
			t.Fatalf("Q6_K packed-scale contract changed: missing %q", required)
		}
	}
	if strings.Contains(text, "blockWord(rowBase, lead, 52u)") {
		t.Fatal("final half-scale must not read the next shared word")
	}
	// A 210-byte block alternates a 0/2-byte lead in aligned uint storage.
	// Its final half at bytes 208..209 lies wholly in staged word 52 in both
	// cases. Exactly 53 words model a final live row with no readable neighbor.
	for _, lead := range []uint{0, 2} {
		var raw [53 * 4]byte
		for i := range raw {
			raw[i] = byte(i*71 + 19)
		}
		for half := uint32(0); half <= 0xffff; half++ {
			binary.LittleEndian.PutUint16(raw[lead+208:], uint16(half))
			var staged [53]uint32
			for i := range staged {
				staged[i] = binary.LittleEndian.Uint32(raw[4*i:])
			}
			got := (staged[52] >> (8 * lead)) & 0xffff
			want := uint32(binary.LittleEndian.Uint16(raw[lead+208:]))
			if got != want {
				t.Fatalf("lead=%d half=%#04x got=%#04x", lead, half, got)
			}
		}
	}
}
