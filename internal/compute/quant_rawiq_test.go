package compute_test

// fak-test:runtime fast est=5s

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// rawIQSuperWeights is the super-block width every llama.cpp i-quant shares.
const rawIQSuperWeights = 256

// rawIQGGUFType maps a native raw i-quant dtype (by its String()) to the GGUF tensor
// type whose ggufload dequant is the independent oracle.
var rawIQGGUFType = map[string]ggufload.TensorType{
	"iq4_xs":  ggufload.TensorIQ4_XS,
	"iq3_xxs": ggufload.TensorIQ3_XXS,
	"iq3_s":   ggufload.TensorIQ3_S,
	"iq2_xxs": ggufload.TensorIQ2_XXS,
	"iq2_s":   ggufload.TensorIQ2_S,
	"iq2_xs":  ggufload.TensorIQ2_XS,
	"iq1_s":   ggufload.TensorIQ1_S,
}

// registeredRawIQ lists every dtype compute registers as a native raw i-quant, so a
// newly registered format is covered without a test edit.
func registeredRawIQ(t *testing.T) []compute.Dtype {
	t.Helper()
	var dts []compute.Dtype
	for dt := compute.Dtype(0); dt < 64; dt++ {
		if compute.IsRawIQ(dt) {
			dts = append(dts, dt)
		}
	}
	if len(dts) == 0 {
		t.Fatal("no native raw i-quant dtype is registered")
	}
	return dts
}

func rawIQBlockBytes(t *testing.T, dt compute.Dtype) int {
	t.Helper()
	bb, ok := compute.RawIQBlockBytes(dt)
	if !ok || bb <= 2 {
		t.Fatalf("RawIQBlockBytes(%v) = %d,%v; want a registered block size > 2", dt, bb, ok)
	}
	return bb
}

// f32ToF16Bits encodes a finite float32 in the IEEE half normal range (round toward
// zero). The test only feeds it small positive scales well inside that range.
func f32ToF16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int((bits>>23)&0xff) - 127 + 15
	frac := bits & 0x7fffff
	if exp <= 0 || exp >= 0x1f {
		panic("f32ToF16Bits: value outside the half normal range")
	}
	return sign | uint16(exp)<<10 | uint16(frac>>13)
}

// randomRawIQ builds out*(in/256) super-blocks of any raw i-quant format: random bytes,
// then a small finite f16 d in [0.01, 0.05) at offset 0 (every i-quant keeps d there;
// every other bit pattern is a valid encoding).
func randomRawIQ(seed int64, blockBytes, out, in int) []byte {
	rng := rand.New(rand.NewSource(seed))
	n := out * (in / rawIQSuperWeights)
	raw := make([]byte, n*blockBytes)
	rng.Read(raw)
	for b := 0; b < n; b++ {
		binary.LittleEndian.PutUint16(raw[b*blockBytes:], f32ToF16Bits(float32(0.01+rng.Float64()*0.04)))
	}
	return raw
}

// ggufRawIQDequant is the independent oracle: the GGUF loader's dequant for dt's GGUF
// tensor type (internal/ggufload gguf_dequant.go), reached through its exported
// WeightSource.TensorF32 over an in-memory single-tensor source.
func ggufRawIQDequant(t *testing.T, dt compute.Dtype, out, in int, raw []byte) []float32 {
	t.Helper()
	typ, ok := rawIQGGUFType[dt.String()]
	if !ok {
		t.Fatalf("raw i-quant dtype %v (%q) has no GGUF oracle mapping", dt, dt.String())
	}
	f := &ggufload.File{Tensors: []ggufload.TensorInfo{{
		Name: "w",
		Dims: []uint64{uint64(in), uint64(out)}, // GGUF dims: fastest-varying first
		Type: typ,
	}}}
	ws, err := ggufload.NewWeightSource(f, bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	got, _, err := ws.TensorF32("w")
	if err != nil {
		t.Fatalf("TensorF32(%v): %v", dt, err)
	}
	return got
}

// fak-test:runtime fast est=5s
func TestDequantRawIQMatchesGGUFLoaderDequant(t *testing.T) {
	if !compute.IsRawIQ(compute.IQ4_XS) {
		t.Fatal("IQ4_XS is not a registered raw i-quant dtype")
	}
	if bb, ok := compute.RawIQBlockBytes(compute.IQ4_XS); !ok || bb != 136 {
		t.Fatalf("RawIQBlockBytes(IQ4_XS) = %d,%v; want 136,true", bb, ok)
	}
	for _, dt := range registeredRawIQ(t) {
		bb := rawIQBlockBytes(t, dt)
		for _, tc := range []struct {
			name    string
			out, in int
			seed    int64
		}{
			{"1x256", 1, 256, 11},
			{"5x512", 5, 512, 12},
			{"9x2048", 9, 2048, 13},
		} {
			t.Run(dt.String()+"/"+tc.name, func(t *testing.T) {
				raw := randomRawIQ(tc.seed, bb, tc.out, tc.in)
				got := compute.DequantRawIQ(dt, tc.out, tc.in, raw)
				want := ggufRawIQDequant(t, dt, tc.out, tc.in, raw)
				if len(got) != tc.out*tc.in || len(want) != len(got) {
					t.Fatalf("len got=%d want=%d, expected %d", len(got), len(want), tc.out*tc.in)
				}
				var nonzero int
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("element %d: DequantRawIQ=%g (0x%08x), gguf=%g (0x%08x)",
							i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
					}
					if got[i] != 0 {
						nonzero++
					}
				}
				if nonzero == 0 {
					t.Fatal("dequant produced an all-zero tensor; fixture is degenerate")
				}
			})
		}
	}
}

// fak-test:runtime fast est=5s
func TestDefaultMatMulRawIQMatchesDequantDot(t *testing.T) {
	be := compute.Default()
	for _, dt := range registeredRawIQ(t) {
		bb := rawIQBlockBytes(t, dt)
		for _, tc := range []struct {
			name       string
			out, in, P int
			seed       int64
		}{
			{"3x256", 3, 256, 2, 21},
			{"7x1024", 7, 1024, 3, 22},
		} {
			t.Run(dt.String()+"/"+tc.name, func(t *testing.T) {
				rng := rand.New(rand.NewSource(tc.seed))
				raw := randomRawIQ(tc.seed, bb, tc.out, tc.in)
				deq := compute.DequantRawIQ(dt, tc.out, tc.in, raw)
				X := make([]float32, tc.P*tc.in)
				for i := range X {
					X[i] = float32(rng.Float64() - 0.5)
				}
				w := compute.NewRawIQ(be, dt, []int{tc.out, tc.in}, raw)
				if w.Dtype != dt {
					t.Fatalf("NewRawIQ dtype = %v, want %v", w.Dtype, dt)
				}

				check := func(label string, got []float32, tok int) {
					t.Helper()
					x := X[tok*tc.in : (tok+1)*tc.in]
					for o := 0; o < tc.out; o++ {
						var dot, mag float64
						for i := 0; i < tc.in; i++ {
							p := float64(deq[o*tc.in+i]) * float64(x[i])
							dot += p
							mag += math.Abs(p)
						}
						if d := math.Abs(float64(got[o]) - dot); d > 1e-5*math.Max(1, mag) {
							t.Fatalf("%s token %d row %d: got %g want %g (|delta| %g > bound %g)",
								label, tok, o, got[o], dot, d, 1e-5*math.Max(1, mag))
						}
					}
				}

				y := be.Read(be.MatMul(w, compute.NewF32(be, []int{tc.in}, X[:tc.in])))
				if len(y) != tc.out {
					t.Fatalf("MatMul len = %d, want %d", len(y), tc.out)
				}
				check("MatMul", y, 0)

				Y := be.Read(be.BatchedMatMul(w, compute.NewF32(be, []int{tc.P, tc.in}, X), tc.P))
				if len(Y) != tc.P*tc.out {
					t.Fatalf("BatchedMatMul len = %d, want %d", len(Y), tc.P*tc.out)
				}
				for p := 0; p < tc.P; p++ {
					check("BatchedMatMul", Y[p*tc.out:(p+1)*tc.out], p)
				}
			})
		}
	}
}

// fak-test:runtime fast est=5s
func TestNewRawIQRejectsMismatchedByteLength(t *testing.T) {
	for _, dt := range registeredRawIQ(t) {
		raw := randomRawIQ(31, rawIQBlockBytes(t, dt), 2, 256)
		for _, tc := range []struct {
			name  string
			shape []int
			raw   []byte
		}{
			{"short_payload", []int{2, 256}, raw[:len(raw)-1]},
			{"in_not_multiple_of_256", []int{2, 128}, raw},
			{"rank1", []int{512}, raw},
		} {
			t.Run(dt.String()+"/"+tc.name, func(t *testing.T) {
				defer func() {
					if recover() == nil {
						t.Fatal("NewRawIQ accepted an invalid shape/payload")
					}
				}()
				compute.NewRawIQ(compute.Default(), dt, tc.shape, tc.raw)
			})
		}
	}
}
