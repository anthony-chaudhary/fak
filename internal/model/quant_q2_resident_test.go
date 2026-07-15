package model

import (
	"encoding/binary"
	"math"
	"testing"
)

// refDequantQ2G128T1 is the T1 reference dequant (ggufload.dequantQ2_0Scalar) re-stated
// locally in its original scalar form — f16 scale d, code j at bits 2*(j%4) of byte j/4,
// y = (int(q)-1)*d — so the resident path is checked against an INDEPENDENT statement of
// the container's semantics, not against its own per-block decode. (The model package
// cannot import ggufload — ggufload imports model — hence the local copy.)
func refDequantQ2G128T1(raw []byte, out, in int) []float32 {
	nblk := in / qBlk2G128
	w := make([]float32, out*in)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := (o*nblk + b) * q2G128BlockBytes
			d := math.Float32frombits(F16BitsToF32Bits(binary.LittleEndian.Uint16(raw[base:])))
			qs := raw[base+2 : base+q2G128BlockBytes]
			for j := 0; j < qBlk2G128; j++ {
				q := (qs[j/4] >> (2 * uint(j%4))) & 0x03
				w[o*in+b*qBlk2G128+j] = float32(int(q)-1) * d
			}
		}
	}
	return w
}

// TestQ2ResidentMatchesDequant is the #4870 done-condition witness: a GEMV against the
// packed-resident Q2_0 tensor — routed through the forward dispatch seam
// (residentMatRows → q2MatRows → q2G128MatRowsRange) — is argmax-exact and within
// |Δ|<=1e-4 of the same tensor taken through the T1 dequant-to-f32 path, at the packed
// resident footprint (34 B per 128 weights) rather than the ~15× larger f32 expansion.
func TestQ2ResidentMatchesDequant(t *testing.T) {
	const (
		out = 9   // odd, to exercise the parallel row split's tail
		in  = 512 // 4 group-128 blocks per row
	)
	nblk := in / qBlk2G128
	raw := make([]byte, out*nblk*q2G128BlockBytes)
	lcgBytes(raw, 0x2545f4914f6cdd1d)
	// Pin each block's f16 scale to a small exact power of two (varied per block so a
	// scale-indexing bug cannot cancel): random scale bytes could decode to inf/NaN, and
	// large scales would let f32 reassociation noise crowd the 1e-4 gate.
	scaleBits := []uint16{0x1C00, 0x2000, 0x1800, 0x2400} // 2^-8, 2^-7, 2^-9, 2^-6
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			binary.LittleEndian.PutUint16(raw[(o*nblk+b)*q2G128BlockBytes:], scaleBits[(o+3*b)%len(scaleBits)])
		}
	}

	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*7)%23-11) / 16
	}

	// Resident side: the raw GGUF bytes wrapped verbatim, read through the SAME dispatch
	// seam the CPU forward uses (residentMatRows on the q2w store).
	qt := wrapQ2G128FromRaw(raw, out, in)
	m := &Model{q2w: map[string]*q2Tensor{"blk.ternary.weight": qt}}
	got := m.residentMatRows("blk.ternary.weight", x, out, in)

	// Reference side: the T1 dequant-to-f32 expansion, then the dense f32 GEMV.
	want := matRows(refDequantQ2G128T1(raw, out, in), x, out, in)

	if len(got) != out {
		t.Fatalf("len(got)=%d, want %d", len(got), out)
	}
	argmax := func(y []float32) int {
		best := 0
		for i := 1; i < len(y); i++ {
			if y[i] > y[best] {
				best = i
			}
		}
		return best
	}
	var maxDiff float64
	for o := 0; o < out; o++ {
		d := math.Abs(float64(got[o]) - float64(want[o]))
		if d > maxDiff {
			maxDiff = d
		}
		if d > 1e-4 {
			t.Fatalf("row %d: resident GEMV=%v, T1 dequant ref=%v (|Δ|=%g > 1e-4)", o, got[o], want[o], d)
		}
	}
	if ga, wa := argmax(got), argmax(want); ga != wa {
		t.Fatalf("argmax mismatch: resident=%d, T1 dequant ref=%d", ga, wa)
	}
	t.Logf("max |Δ| = %g over %d rows", maxDiff, out)

	// Footprint: the resident tensor holds the GGUF bytes verbatim (34 B per 128 weights,
	// 0.266 B/weight); the dequant-to-f32 expansion is 4 B/weight = 512/34 ≈ 15.06× larger.
	packed := qt.footprintBytes()
	if packed != len(raw) {
		t.Fatalf("footprintBytes=%d, want the verbatim GGUF payload %d", packed, len(raw))
	}
	f32Bytes := out * in * 4
	if f32Bytes < 15*packed {
		t.Fatalf("f32 expansion %d B is only %.2fx the packed %d B; want >=15x", f32Bytes, float64(f32Bytes)/float64(packed), packed)
	}
	t.Logf("packed=%d B, f32=%d B (%.2fx)", packed, f32Bytes, float64(f32Bytes)/float64(packed))

	// The dequantQ2Tensor g128 arm must agree element-for-element with the T1 reference —
	// the resident dequant IS the T1 semantics, not merely close to it.
	wGot := dequantQ2Tensor(qt)
	wWant := refDequantQ2G128T1(raw, out, in)
	for i := range wWant {
		if wGot[i] != wWant[i] {
			t.Fatalf("dequant[%d]=%v, T1 ref=%v (resident block decode diverged from container semantics)", i, wGot[i], wWant[i])
		}
	}
}
