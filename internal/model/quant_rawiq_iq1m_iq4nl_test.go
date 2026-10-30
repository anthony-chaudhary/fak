package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=20ms lane=default
func TestRawIQIQ1MIQ4NLDequantMatchesModelOracle(t *testing.T) {
	for _, tc := range []struct {
		kind kQuantKind
		dt   compute.Dtype
	}{{kindIQ1M, compute.IQ1_M}, {kindIQ4NL, compute.IQ4_NL}} {
		const out, in = 3, 512
		bb := tc.kind.blockBytes()
		bw := tc.kind.blockWeights()
		raw := make([]byte, out*in/bw*bb)
		state := uint64(0x1a2b3c4d5e6f7081) ^ uint64(tc.kind)
		for i := range raw {
			state ^= state << 13
			state ^= state >> 7
			state ^= state << 17
			raw[i] = byte(state)
		}
		// Keep every f16 scale finite: IQ4_NL d is each block's first u16; IQ1_M's super-scale
		// is the top nibble of each of its four scale words (bytes 49, 51, 53, 55).
		for b := 0; b < len(raw)/bb; b++ {
			blk := raw[b*bb : (b+1)*bb]
			switch tc.kind {
			case kindIQ4NL:
				blk[0], blk[1] = 0x00, 0x30+byte(b%4)
			case kindIQ1M:
				// scaleBits 0x3400 -> nibbles (sc0>>12)=0, (sc1>>12)=0, (sc2>>12)=4, (sc3>>12)=3.
				blk[49], blk[51], blk[53], blk[55] = blk[49]&0x0f, blk[51]&0x0f, blk[53]&0x0f|0x40, blk[55]&0x0f|0x30
			default:
				t.Fatalf("no scale fixture for %s", tc.kind)
			}
		}
		got := compute.DequantRawIQ(tc.dt, out, in, raw)
		want := make([]float32, out*in)
		for b := 0; b < len(raw)/bb; b++ {
			kQuantDequantSuperBlock(want[b*bw:(b+1)*bw], raw[b*bb:(b+1)*bb], tc.kind)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s: weight %d = %g, model oracle %g", tc.kind, i, got[i], want[i])
			}
		}
	}
}
