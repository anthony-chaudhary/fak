package model

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestQ5KInt8MatchesF32 pins the int8 Q5_K decode GEMV (quant_kquant_int8.go) against the f32
// kQuantMatRows over the SAME resident super-blocks. The int8 path adds activation quantization, so
// the gate is a tight relative error (like the Q4_K int8 path), not bit-equality. It also guards the
// gate: with kQuantSDOT off, kQuantMatRows must stay the byte-identical f32 path.
func TestQ5KInt8MatchesF32(t *testing.T) {
	const (
		out = 9   // odd, to exercise the parallel row split's tail
		in  = 512 // 2 super-blocks per row
	)
	nblk := in / qkK
	bb := kindQ5K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, 0x123456789abcdef0)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*bb:]
			binary.LittleEndian.PutUint16(blk[0:], f16One) // d = 1.0
			binary.LittleEndian.PutUint16(blk[2:], 0)      // min = 0 keeps decoded weights finite
		}
	}
	qt := quantizeKQuantFromRaw(raw, out, in, kindQ5K)
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*7)%23) - 11
	}

	// f32 reference (gate forced off so kQuantMatRows takes the byte-identical f32 path).
	setKQuantSDOTForTest(false)
	want := kQuantMatRows(qt, x)
	if len(want) != out {
		t.Fatalf("ref len=%d want %d", len(want), out)
	}

	// int8 path.
	setKQuantSDOTForTest(true)
	t.Cleanup(func() { kQuantSDOTForce = 0 })
	got := kQuantMatRows(qt, x)
	if len(got) != out {
		t.Fatalf("int8 len=%d want %d", len(got), out)
	}

	// Cosine + bounded relative error: activation-quant noise only, so this is tight.
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
		t.Fatalf("int8 Q5_K vs f32 cosine %.6f < 0.9999 (got=%v want=%v)", cos, got, want)
	}
	if maxRel > 0.02 {
		t.Fatalf("int8 Q5_K vs f32 max rel err %.4f > 0.02", maxRel)
	}
	t.Logf("Q5_K int8 vs f32: cosine=%.8f maxRel=%.5f", cos, maxRel)
}

// TestQ5KInt8Gate pins the gate contract: the test force defaults to 0 (so the path is decided by
// FAK_KQ_INT8, not silently on), the gate tracks the resolved env default for Q5_K, and Q6_K is
// ALWAYS off (no Q6_K int8 kernel). It must hold whether or not FAK_KQ_INT8 is set in the run's env
// (CI runs it both ways), so it asserts against kQuantSDOTDefault rather than hardcoding false.
func TestQ5KInt8Gate(t *testing.T) {
	if kQuantSDOTForce != 0 {
		t.Fatalf("kQuantSDOTForce must default to 0 (env-decided, not force-on), got %d", kQuantSDOTForce)
	}
	if got := kQuantSDOTEnabled(kindQ5K); got != kQuantSDOTDefault {
		t.Fatalf("kQuantSDOTEnabled(Q5_K)=%v must track the FAK_KQ_INT8 env default %v", got, kQuantSDOTDefault)
	}
	if kQuantSDOTEnabled(kindQ6K) {
		t.Fatal("kQuantSDOTEnabled(Q6_K) must always be false (Q6_K int8 not implemented)")
	}
}
