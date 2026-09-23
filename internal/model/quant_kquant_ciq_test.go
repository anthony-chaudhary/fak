package model

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestCIQQ8_0Q4_0MatchesF32 pins the compute CIQ integer decode GEMV (quant_kquant_ciq.go,
// issue #12275) against the f32 kQuantMatRows over the SAME resident legacy ggml blocks, for
// both kindQ8_0 and kindQ4_0. The int8 CIQ path adds activation quantization, so the gate is a
// tight relative error (like the Q4_K/Q5_K/Q6_K int8 paths), not bit-equality. It also guards
// the gate: with kQuantSDOT off, kQuantMatRows must stay the byte-identical f32 path.
func TestCIQQ8_0Q4_0MatchesF32(t *testing.T) {
	const (
		out = 9   // odd, to exercise the parallel row split's tail
		in  = 512 // 8 blocks of 32 per row
	)
	for _, kind := range []kQuantKind{kindQ8_0, kindQ4_0} {
		name := "Q8_0"
		if kind == kindQ4_0 {
			name = "Q4_0"
		}
		t.Run(name, func(t *testing.T) {
			bw := kind.blockWeights()
			nblk := in / bw
			bb := kind.blockBytes()
			raw := make([]byte, out*nblk*bb)
			lcgBytes(raw, 0x123456789abcdef0)
			for o := 0; o < out; o++ {
				for b := 0; b < nblk; b++ {
					blk := raw[(o*nblk+b)*bb:]
					binary.LittleEndian.PutUint16(blk[0:], f16One) // d = 1.0 -> finite decoded weights
				}
			}
			qt := quantizeKQuantFromRaw(raw, out, in, kind)
			x := make([]float32, in)
			for i := range x {
				x[i] = float32((i*7)%23) - 11
			}

			// f32 reference (gate forced off so kQuantMatRows takes the byte-identical f32 path).
			setKQuantSDOTForTest(false)
			want := kQuantMatRows(qt, x)

			// CIQ int8 path.
			setKQuantSDOTForTest(true)
			t.Cleanup(func() { kQuantSDOTForce = 0 })
			got := kQuantMatRows(qt, x)

			var dot, ng, nw float64
			var maxRel float64
			for o := 0; o < out; o++ {
				dot += float64(got[o]) * float64(want[o])
				ng += float64(got[o]) * float64(got[o])
				nw += float64(want[o]) * float64(want[o])
				den := math.Abs(float64(want[o]))
				if den < 1 {
					den = 1
				}
				if rel := math.Abs(float64(got[o]-want[o])) / den; rel > maxRel {
					maxRel = rel
				}
			}
			cos := dot / (math.Sqrt(ng)*math.Sqrt(nw) + 1e-12)
			if cos < 0.9999 {
				t.Fatalf("CIQ %s vs f32 cosine %.6f < 0.9999 (got=%v want=%v)", name, cos, got, want)
			}
			// Q4_0 carries only 4 bits of weight precision, so its own dequant error is
			// larger than Q8_0's; the CIQ-vs-f32 relative gap is correspondingly wider.
			relGate := 0.02
			if kind == kindQ4_0 {
				relGate = 0.03
			}
			if maxRel > relGate {
				t.Fatalf("CIQ %s vs f32 max rel err %.4f > %.3f", name, maxRel, relGate)
			}
			t.Logf("%s CIQ vs f32: cosine=%.8f maxRel=%.5f", name, cos, maxRel)
		})
	}
}

// TestCIQGate pins that the CIQ kinds are gated OFF by default (the f32 reduction stays the
// conservative floor) and are switched on by the same int8 SDOT gate the K-quant paths use.
func TestCIQGate(t *testing.T) {
	t.Cleanup(func() { kQuantSDOTForce = 0 })
	kQuantSDOTForce = -1
	for _, k := range []kQuantKind{kindQ8_0, kindQ4_0, kindQ5K, kindQ6K} {
		if kQuantSDOTEnabled(k) {
			t.Fatalf("kind %v enabled with the gate forced OFF", k)
		}
	}
	kQuantSDOTForce = 1
	for _, k := range []kQuantKind{kindQ8_0, kindQ4_0, kindQ5K, kindQ6K} {
		if !kQuantSDOTEnabled(k) {
			t.Fatalf("kind %v disabled with the gate forced ON", k)
		}
	}
	if kQuantSDOTEnabled(kindIQ3XXS) {
		t.Fatalf("kindIQ3XXS must not ride the int8 SDOT gate")
	}
}
