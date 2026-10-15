package compute

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kquantbits"
)

// fak-test:runtime fast est=50ms
// This software witness proves indexing, independent row arithmetic and a
// token-independent number of decoded quant quartets in the scalar model of
// the shader. It does not claim Vulkan compilation, device parity or traffic.
func TestV41Q2KTokenPanelContract(t *testing.T) {
	const out, in = 5, 512 // Includes the final odd output row and several lane groups.
	raw := make([]byte, out*(in/256)*84)
	for i := range raw {
		raw[i] = byte(i*73 + i/7 + 19)
	}
	for off := 0; off < len(raw); off += 84 {
		binary.LittleEndian.PutUint16(raw[off+80:], 0x2000) // 1/128
		binary.LittleEndian.PutUint16(raw[off+82:], 0x1c00) // 1/256
	}
	for _, rows := range []int{2, 4, 8} {
		plan, err := PlanV41Q2KTokenPanel(out, in, rows)
		if err != nil || plan.WeightBytes != len(raw) || plan.InputBytes != rows*in*4 || plan.OutputBytes != rows*out*4 || plan.WorkgroupsX != 3 {
			t.Fatalf("rows=%d plan=%+v err=%v", rows, plan, err)
		}
		x := make([]float32, rows*in)
		for token := 0; token < rows; token++ {
			for col := 0; col < in; col++ {
				x[token*in+col] = float32((col*3+token*11)%31-15) / 32
			}
		}
		got, decodes := v41Q2KTokenPanelScalar(raw, x, out, in, rows)
		if decodes != out*(in/256)*64 {
			t.Fatalf("rows=%d decoded %d quartets, want token-independent %d", rows, decodes, out*(in/256)*64)
		}
		for token := 0; token < rows; token++ {
			want, scalarDecodes := v41Q2KTokenPanelScalar(raw, x[token*in:(token+1)*in], out, in, 1)
			if scalarDecodes != decodes {
				t.Fatal("repeated scalar arms must each pay the whole quant decode count")
			}
			for o := 0; o < out; o++ {
				if math.Float32bits(got[token*out+o]) != math.Float32bits(want[o]) {
					t.Fatalf("rows=%d token=%d output=%d differs from scalar lane tree", rows, token, o)
				}
				// Independent packed-block decoder and scalar dot pin the byte
				// layout rather than merely comparing two uses of our lane code.
				var expected float64
				block := make([]float32, 256)
				for sb := 0; sb < in/256; sb++ {
					off := (o*(in/256) + sb) * 84
					q2kDequantSuperBlock(block, raw[off:off+84])
					for col, weight := range block {
						expected += float64(weight) * float64(x[token*in+sb*256+col])
					}
				}
				if math.Abs(float64(got[token*out+o])-expected) > 1e-5*(1+math.Abs(expected)) {
					t.Fatalf("rows=%d token=%d output=%d got=%g packed oracle=%g", rows, token, o, got[token*out+o], expected)
				}
			}
		}
	}
	for _, dims := range [][3]int{{2304, 5120, 2}, {2304, 5120, 4}, {5120, 2304, 8}} {
		if _, err := PlanV41Q2KTokenPanel(dims[0], dims[1], dims[2]); err != nil {
			t.Fatalf("full expert geometry %v refused: %v", dims, err)
		}
	}
	for _, dims := range [][3]int{{5, 512, 1}, {5, 512, 3}, {5, 512, 9}, {0, 512, 2}, {5, 511, 2}, {1 << 30, 512, 8}, {5, 1 << 30, 8}} {
		if _, err := PlanV41Q2KTokenPanel(dims[0], dims[1], dims[2]); err == nil {
			t.Fatalf("invalid geometry %v admitted", dims)
		}
	}
	source, err := os.ReadFile("shaders/v41_q2k_token_panel.comp")
	if err != nil {
		t.Fatal(err)
	}
	// The paired-row contract pins both quant decodes before the token loop,
	// the guarded odd tail, and the shared activation with separate row sums.
	if err := q2TokenPanelActivationReuseSourceContract(string(source)); err != nil {
		t.Fatal(err)
	}
}

func v41Q2KTokenPanelScalar(raw []byte, x []float32, out, in, rows int) ([]float32, int) {
	y := make([]float32, rows*out)
	decodes := 0
	for o := 0; o < out; o++ {
		acc := make([][64]float32, rows)
		for tid := 0; tid < 64; tid++ {
			t16, h, s, lq := tid&15, (tid&15)>>3, ((tid&15)>>2)&1, (tid&3)*4
			for sb := tid >> 4; sb < in/256; sb += 4 {
				base := (o*(in/256) + sb) * 84
				q := binary.LittleEndian.Uint32(raw[base+16+t16*4:])
				d := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[base+80:])))
				m := math.Float32frombits(kquantbits.F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[base+82:])))
				sums := make([]float32, rows)
				for j := 0; j < 4; j++ {
					sc := raw[base+h*8+j*2+s]
					dl, ml := d*float32(sc&15), m*float32(sc>>4)
					quant := (q >> uint(2*j)) & 0x03030303
					decodes++
					for t := 0; t < rows; t++ {
						ix := t*in + sb*256 + h*128 + j*32 + s*16 + lq
						var dot, xsum float32
						for c := 0; c < 4; c++ {
							v := x[ix+c]
							dot += float32((quant>>uint(c*8))&255) * v
							xsum += v
						}
						sums[t] += dl*dot - ml*xsum
					}
				}
				for t := 0; t < rows; t++ {
					acc[t][tid] += sums[t]
				}
			}
		}
		for t := 0; t < rows; t++ {
			for stride := 32; stride > 0; stride >>= 1 {
				for tid := 0; tid < stride; tid++ {
					acc[t][tid] += acc[t][tid+stride]
				}
			}
			y[t*out+o] = acc[t][0]
		}
	}
	return y, decodes
}
