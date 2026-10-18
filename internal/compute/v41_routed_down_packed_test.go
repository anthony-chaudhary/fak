package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// fak-test:runtime fast est=50ms lane=default
// This binds the production integer decoder to an independent byte/coordinate
// oracle. It does not execute GLSL or prove driver register allocation or speed.
func TestV41RoutedDownPacked(t *testing.T) {
	t.Parallel()
	rawSource, err := os.ReadFile("shaders/v41_routed_down_sum.comp")
	if err != nil {
		t.Fatal(err)
	}
	source := strings.ReplaceAll(string(rawSource), "\r\n", "\n")
	if err := v41CheckDownPackedSource(source); err != nil {
		t.Fatal(err)
	}
	for _, change := range [][2]string{
		{"q3kPackedLead + offset", "offset"},
		{"word < 28u", "word < 27u"},
		{"q3kPackedWords[27]", "q3kPackedWords[28]"},
		{"uint qBase  = 32u;", "uint qBase  = base + 32u;"},
		{"acc = acc + uintBitsToFloat(rounded);", "acc = uintBitsToFloat(rounded) + acc;"},
		{"if (row == 0u) {", "if (row < 64u) {"},
		{"if (row == 0u) {", "if (row == 1u) {"},
		{"status[4u * p + 3u] = outDim;", "status[4u * p + 2u] = outDim;"},
	} {
		mutant := strings.Replace(source, change[0], change[1], 1)
		if mutant == source || v41CheckDownPackedSource(mutant) == nil {
			t.Fatalf("source witness missed mutation %q", change[0])
		}
	}

	seenLead := [2]bool{}
	for _, rows := range []int{1, 3, 65} {
		for _, blocks := range []int{1, 2, 3, 9} { // Flash has nine blocks/row.
			payload := rows * blocks * 110
			storage := make([]byte, (payload+3)&^3)
			for i := range storage {
				storage[i] = byte((i*73 + i/17*29 + rows*11 + blocks) & 255)
			}
			for b := 0; b < rows*blocks; b++ {
				// Finite positive/negative/zero/subnormal scales exercise signed
				// coefficient bits without making a NaN-equality test vacuous.
				scales := [...]uint16{0, 0x8000, 1, 0x8001, 0x2800, 0xb800, 0x7bff}
				binary.LittleEndian.PutUint16(storage[b*110+108:], scales[b%len(scales)])
			}
			for i := payload; i < len(storage); i++ {
				storage[i] = 0xa5 // Poison the descriptor-visible tail padding.
			}
			for row := 0; row < rows; row++ {
				gotRow := make([]float32, blocks*256)
				for sb := 0; sb < blocks; sb++ {
					base := (row*blocks + sb) * 110
					lead := base & 3
					seenLead[lead/2] = true
					var bank [28]uint32
					for word := range bank {
						at := (base/4 + word) * 4
						if at+4 > len(storage) {
							t.Fatalf("row=%d block=%d word=%d overreads final descriptor", row, sb, word)
						}
						bank[word] = binary.LittleEndian.Uint32(storage[at : at+4])
					}
					byteAt := func(offset int) byte {
						at := lead + offset
						return byte(bank[at/4] >> uint((at&3)*8))
					}
					for offset := 0; offset < 110; offset++ {
						if got := byteAt(offset); got != storage[base+offset] {
							t.Fatalf("lead=%d byte=%d got=%x want=%x", lead, offset, got, storage[base+offset])
						}
					}
					// The final f16 fits inside word27 at both alignments; no
					// generic unaligned u32 access may request word28 here.
					dbits := uint16(bank[27] >> uint(lead*8))
					if dbits != binary.LittleEndian.Uint16(storage[base+108:base+110]) {
						t.Fatal("final half consumed padding or wrong alignment")
					}
					d := math.Float32frombits(kquantbits.F16BitsToF32Bits(dbits))
					var scaleBytes [12]byte
					for i := range scaleBytes {
						scaleBytes[i] = byteAt(96 + i)
					}
					scales := unpackQ3KScales(scaleBytes[:]) // Unchanged decoder, separately source-bound.
					want := make([]float32, 256)
					q3kDequantSuperBlock(want, storage[base:base+110])
					for i := 0; i < 256; i++ {
						// Coordinate oracle, independent of the shader's n/j/l nest.
						half, group, lane := i/128, (i%128)/32, i%32
						code := int((byteAt(32+half*32+lane) >> uint(group*2)) & 3)
						if byteAt(lane)&(1<<uint(half*4+group)) == 0 {
							code -= 4
						}
						dl := d * float32(scales[half*8+group*2+lane/16]-32)
						got := dl * float32(code)
						if math.Float32bits(got) != math.Float32bits(want[i]) {
							t.Fatalf("lead=%d coefficient=%d got=%08x want=%08x", lead, i, math.Float32bits(got), math.Float32bits(want[i]))
						}
						gotRow[sb*256+i] = got
					}
				}
				x := make([]float32, blocks*256)
				var gotSum float32
				for i := range x {
					x[i] = float32((i*7)%19-9) / 16
					gotSum += gotRow[i] * x[i]
				}
				start := row * blocks * 110
				wantSum := q3kRowDot(storage[start:start+blocks*110], x, make([]float32, 256))
				if math.Float32bits(gotSum) != math.Float32bits(wantSum) {
					t.Fatalf("row=%d blocks=%d serial accumulation differs", row, blocks)
				}
			}
		}
	}
	if !seenLead[0] || !seenLead[1] {
		t.Fatal("fixture missed alternating packed alignment")
	}
}

func v41CheckDownPackedSource(source string) error {
	canonical := func(s string) string {
		return strings.Join(strings.Fields(strings.ReplaceAll(s, "\\\n", "\n")), " ")
	}
	start, end := strings.Index(source, "#define Q3K_SLOT"), strings.Index(source, "        uint qi = 0u;")
	if start < 0 || end <= start || canonical(source[start:end]) != canonical(v41DownPackedPrefix) {
		return fmt.Errorf("down packed integer decoder or pre-load order changed")
	}
	if strings.Count(source, "uint q3kPackedWords[28];") != 1 || strings.Count(source, "uint q3kPackedLead;") != 1 ||
		strings.Contains(source, "shared uint q3kPacked") {
		return fmt.Errorf("down bank must be invocation-private with exact extent")
	}
	main := strings.Index(source, "void main()")
	if main < 0 || canonical(source[main:]) != canonical(v41DownPackedMain) {
		return fmt.Errorf("down publication, failure, pick order or output suffix changed")
	}
	return nil
}

const v41DownPackedPrefix = `#define Q3K_SLOT(ROW, BYTE, RAW)
uint BYTE(uint offset) {
    uint at = q3kPackedLead + offset;
    return (q3kPackedWords[at >> 2u] >> ((at & 3u) * 8u)) & 255u;
}
float ROW(uint row, uint blocks, uint midBase) {
    float sum = 0.0;
    for (uint sb = 0u; sb < blocks; ++sb) {
        uint base = (row * blocks + sb) * 110u;
        q3kPackedLead = base & 3u;
        for (uint word = 0u; word < 28u; ++word) q3kPackedWords[word] = RAW[(base >> 2u) + word];
        float d = unpackHalf2x16((q3kPackedWords[27] >> (q3kPackedLead * 8u)) & 0xffffu).x;
        uint hmBase = 0u;
        uint qBase  = 32u;
        uint scBase = 96u;
        uint xBlock = midBase + sb * 256u;
        int scales[16];
        q3kScales(
            BYTE(scBase) | (BYTE(scBase + 1u) << 8u) | (BYTE(scBase + 2u) << 16u) | (BYTE(scBase + 3u) << 24u),
            BYTE(scBase + 4u) | (BYTE(scBase + 5u) << 8u) | (BYTE(scBase + 6u) << 16u) | (BYTE(scBase + 7u) << 24u),
            BYTE(scBase + 8u) | (BYTE(scBase + 9u) << 8u) | (BYTE(scBase + 10u) << 16u) | (BYTE(scBase + 11u) << 24u),
            scales);`

const v41DownPackedMain = `void main() {
    uint row = gl_GlobalInvocationID.x;
    uint outDim = uint(pc.outDim);
    if (row >= outDim) return;
    uint picks = pc.picks;
    if (row == 0u) {
        for (uint p = 0u; p < picks; ++p) {
            status[4u * p] = 0x42463143u;
            status[4u * p + 1u] = pc.site;
            status[4u * p + 3u] = outDim;
        }
    }
    uint blocks = uint(pc.midDim) / 256u;
    precise float acc = 0.0;
    for (uint j = 0u; j < picks; ++j) {
        uint p = ids[2u * picks + j];
        uint slot = ids[p];
        uint midBase = p * uint(pc.midDim);
        float d;
        switch (slot) {
        case 0u: d = q3kRow0(row, blocks, midBase); break;
        case 1u: d = q3kRow1(row, blocks, midBase); break;
        case 2u: d = q3kRow2(row, blocks, midBase); break;
        case 3u: d = q3kRow3(row, blocks, midBase); break;
        case 4u: d = q3kRow4(row, blocks, midBase); break;
        case 5u: d = q3kRow5(row, blocks, midBase); break;
        case 6u: d = q3kRow6(row, blocks, midBase); break;
        default: d = q3kRow7(row, blocks, midBase); break;
        }
        uint bits = floatBitsToUint(d);
        uint rounded = roundBF16(bits);
        if ((bits & 0x7f800000u) == 0x7f800000u ||
            (rounded & 0x7f800000u) == 0x7f800000u) {
            atomicMin(status[4u * p + 2u], row);
        }
        acc = acc + uintBitsToFloat(rounded);
    }
    sumOut[row] = acc;
}`
