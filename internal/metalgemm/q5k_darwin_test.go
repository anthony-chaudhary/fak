//go:build darwin && arm64 && cgo

// fak-test:runtime fast est=2s

package metalgemm

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// q5kRefRow dequantizes one Q5_K row the way ggml's dequantize_row_q5_K does.
func q5kRefRow(row []byte, in int) []float64 {
	out := make([]float64, 0, in)
	for b := 0; b < in/256; b++ {
		blk := row[b*Q5KBlockBytes : (b+1)*Q5KBlockBytes]
		d := float64(halfToFloat32(binary.LittleEndian.Uint16(blk[0:])))
		dmin := float64(halfToFloat32(binary.LittleEndian.Uint16(blk[2:])))
		sc := blk[4:16]
		qh := blk[16:48]
		ql := blk[48:176]
		scaleMin := func(j int) (float64, float64) {
			if j < 4 {
				return float64(sc[j] & 63), float64(sc[j+4] & 63)
			}
			return float64((sc[j+4] & 0x0f) | ((sc[j-4] >> 6) << 4)), float64((sc[j+4] >> 4) | ((sc[j] >> 6) << 4))
		}
		u1, u2 := byte(1), byte(2)
		for j := 0; j < 4; j++ {
			s1, m1 := scaleMin(2 * j)
			s2, m2 := scaleMin(2*j + 1)
			q := ql[32*j:]
			for l := 0; l < 32; l++ {
				v := float64(q[l] & 0x0f)
				if qh[l]&u1 != 0 {
					v += 16
				}
				out = append(out, d*s1*v-dmin*m1)
			}
			for l := 0; l < 32; l++ {
				v := float64(q[l] >> 4)
				if qh[l]&u2 != 0 {
					v += 16
				}
				out = append(out, d*s2*v-dmin*m2)
			}
			u1 <<= 2
			u2 <<= 2
		}
	}
	return out
}

func halfToFloat32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3ff
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+112)<<23 | mant<<13)
}

func randomQ5KRows(rng *rand.Rand, out, in int) []byte {
	raw := make([]byte, out*(in/256)*Q5KBlockBytes)
	rng.Read(raw)
	for b := 0; b < len(raw); b += Q5KBlockBytes {
		// finite, small d/dmin (0x2000..0x23ff ~ 0.0078..0.0156) so no Inf/NaN reaches the dot
		binary.LittleEndian.PutUint16(raw[b:], 0x2000|uint16(rng.Intn(0x400)))
		binary.LittleEndian.PutUint16(raw[b+2:], 0x2000|uint16(rng.Intn(0x400)))
	}
	return raw
}

// TestQ5KGraphProjectionMatchesReference pins both Q5_K graph kernels to an independent CPU
// dequantization: q5k_mul_mv (P=1, f32 weights) within float tolerance and q5k_mul_mm (P>1,
// weights rounded to half before the MMA, like the Q4_K/Q6_K mul_mm ports) within half
// tolerance, including a row count that is not a multiple of either kernel's row tile.
func TestQ5KGraphProjectionMatchesReference(t *testing.T) {
	if !Q5KAvailable() {
		t.Skip("Q5_K Metal kernels unavailable")
	}
	const out, in = 203, 1024
	rng := rand.New(rand.NewSource(5))
	raw := randomQ5KRows(rng, out, in)
	w := UploadQ5KGoOwned(raw, out, in)
	if w == nil {
		t.Fatal("UploadQ5KGoOwned returned nil")
	}
	defer w.Release()
	rowBytes := (in / 256) * Q5KBlockBytes
	ref := make([][]float64, out)
	for o := range ref {
		ref[o] = q5kRefRow(raw[o*rowBytes:(o+1)*rowBytes], in)
	}
	for _, tc := range []struct {
		p   int
		tol float64
	}{{1, 1e-4}, {3, 2e-3}, {37, 2e-3}} {
		x := make([]float32, tc.p*in)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		g, err := BeginProjectionGraph(x, nil, nil, tc.p, in)
		if err != nil {
			t.Fatal(err)
		}
		input, err := g.Input(in)
		if err != nil {
			t.Fatal(err)
		}
		y, err := g.EncodeQ5KFrom(w, input)
		if err != nil {
			t.Fatal(err)
		}
		got, _, err := g.FinishRead(y)
		g.Free()
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != 1 || len(got[0]) != tc.p*out {
			t.Fatalf("P=%d: output shape %d", tc.p, len(got[0]))
		}
		var maxRel float64
		for p := 0; p < tc.p; p++ {
			for o := 0; o < out; o++ {
				var want, norm float64
				for k := 0; k < in; k++ {
					v := ref[o][k] * float64(x[p*in+k])
					want += v
					norm += math.Abs(v)
				}
				rel := math.Abs(float64(got[0][p*out+o])-want) / math.Max(norm, 1e-6)
				maxRel = math.Max(maxRel, rel)
			}
		}
		if maxRel > tc.tol {
			t.Fatalf("P=%d: max relative error %.3g > %.3g", tc.p, maxRel, tc.tol)
		}
	}
}

func TestQ5KGraphRefusesWrongWidth(t *testing.T) {
	if !Q5KAvailable() {
		t.Skip("Q5_K Metal kernels unavailable")
	}
	raw := randomQ5KRows(rand.New(rand.NewSource(1)), 8, 512)
	w := UploadQ5KGoOwned(raw, 8, 512)
	if w == nil {
		t.Fatal("UploadQ5KGoOwned returned nil")
	}
	defer w.Release()
	g, err := BeginProjectionGraph(make([]float32, 256), nil, nil, 1, 256)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	input, err := g.Input(256)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := g.EncodeQ5KFrom(w, input); err == nil {
		t.Fatal("width-mismatched Q5_K projection was encoded")
	}
	w.Release()
	if UploadQ5KGoOwned(raw[:10], 8, 512) != nil {
		t.Fatal("short Q5_K payload was accepted")
	}
}
