package compute

import (
	"encoding/binary"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// fak-test:runtime fast est=50ms
// CPU indexing, scalar-order and decode-sharing witness only. Native shader
// compilation, device parity, occupancy and bandwidth remain unqualified.
func TestV41Q3KTokenPanelContract(t *testing.T) {
	const out, in = 65, 768 // A partial workgroup and an odd number of Q3 blocks.
	raw := make([]byte, out*(in/256)*110)
	for i := range raw {
		raw[i] = byte(i*73 + i/7 + 19)
	}
	for off := 0; off < len(raw); off += 110 {
		binary.LittleEndian.PutUint16(raw[off+108:], 0x1800)
	}
	for _, rows := range []int{2, 4, 8} {
		plan, err := PlanV41Q3KTokenPanel(out, in, rows)
		if err != nil || plan.PayloadBytes != len(raw) || plan.WeightBytes != len(raw)+2 || plan.InputBytes != rows*in*4 || plan.OutputBytes != rows*out*4 || plan.WorkgroupsX != 2 {
			t.Fatalf("rows=%d plan=%+v err=%v", rows, plan, err)
		}
		x := make([]float32, rows*in)
		for token := 0; token < rows; token++ {
			for col := 0; col < in; col++ {
				x[token*in+col] = float32((col*3+token*11)%31-15) / 32
			}
		}
		got, decodes := v41Q3KTokenPanelScalar(raw, x, out, in, rows)
		if decodes != out*in {
			t.Fatalf("rows=%d decodes=%d, want token-independent %d", rows, decodes, out*in)
		}
		for token := 0; token < rows; token++ {
			for row := 0; row < out; row++ {
				weight := raw[row*(in/256)*110 : (row+1)*(in/256)*110]
				// Existing independent packed decoder and scalar dot retain the
				// same per-output accumulation order; no token shares an accumulator.
				want := q3kRowDot(weight, x[token*in:(token+1)*in], make([]float32, 256))
				if math.Float32bits(got[token*out+row]) != math.Float32bits(want) {
					t.Fatalf("rows=%d token=%d output=%d got=%g want=%g", rows, token, row, got[token*out+row], want)
				}
			}
		}
	}
	for _, rows := range []int{2, 4, 8} {
		if plan, err := PlanV41Q3KTokenPanel(5120, 2304, rows); err != nil || plan.WeightBytes != 5068800 {
			t.Fatalf("published down geometry rows=%d plan=%+v err=%v", rows, plan, err)
		}
	}
	for _, dims := range [][3]int{{65, 768, 1}, {65, 768, 3}, {65, 768, 9}, {0, 256, 2}, {65, 767, 2}, {1 << 30, 256, 8}, {65, 1 << 30, 8}} {
		if _, err := PlanV41Q3KTokenPanel(dims[0], dims[1], dims[2]); err == nil {
			t.Fatalf("invalid geometry %v admitted", dims)
		}
	}
	shader, err := os.ReadFile("shaders/v41_q3k_token_panel.comp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(shader)
	decoded := strings.Index(source, "float decoded = dl * float(code);")
	if decoded < 0 || !strings.Contains(source[decoded:], "for (uint token = 0u; token < tokens; ++token)") || strings.Count(source, "int code = int(") != 1 || !strings.Contains(source, "layout(local_size_x = 64)") {
		t.Fatal("shader must decode once before its per-token accumulator loop")
	}
}

func v41Q3KTokenPanelScalar(raw []byte, x []float32, out, in, rows int) ([]float32, int) {
	y := make([]float32, rows*out)
	decodes := 0
	for row := 0; row < out; row++ {
		for sb := 0; sb < in/256; sb++ {
			base := (row*(in/256) + sb) * 110
			d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[base+108:])))
			scales := unpackQ3KScales(raw[base+96 : base+108])
			qi, si, mask := 0, 0, byte(1)
			for half := 0; half < 256; half += 128 {
				shift := uint(0)
				for j := 0; j < 4; j++ {
					for sub := 0; sub < 2; sub++ {
						dl := d * float32(scales[si]-32)
						si++
						for l := 0; l < 16; l++ {
							lane := sub*16 + l
							code := int((raw[base+32+qi+lane] >> shift) & 3)
							if raw[base+lane]&mask == 0 {
								code -= 4
							}
							decoded := dl * float32(code)
							decodes++
							col := sb*256 + half + j*32 + lane
							for token := 0; token < rows; token++ {
								y[token*out+row] += decoded * x[token*in+col]
							}
						}
					}
					shift += 2
					mask <<= 1
				}
				qi += 32
			}
		}
	}
	return y, decodes
}
