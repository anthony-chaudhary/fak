package model

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// Unwired host emulation of the DeepInfer V4.1 indexer Q/K cast: e2m1 values
// with one UE8M0 power-of-two scale per 32 bf16 values, composed with fak's
// f32 indexer score and top-k. It measures selection drift only; it is not a
// parity claim against the CUDA kernel and fak scoring stays bf16-widened f32.

var errV41IndexMXFP4 = errors.New("model: v41 index MXFP4 oracle refused")

const v41IndexMXFP4Group = 32

var _ v41IndexerScoreFunc = v41IndexMXFP4Score

// v41IndexMXFP4E2M1RNE returns the v4ExpertE2M1Values code nearest v, ties to
// the even code, saturating finite magnitudes above 6. The shared KV oracle
// reuses it; only the scale rule differs between the two casts.
func v41IndexMXFP4E2M1RNE(v float32) byte {
	var sign byte
	if math.Signbit(float64(v)) {
		sign, v = 8, -v
	}
	for code := byte(0); code < 7; code++ {
		mid := (v4ExpertE2M1Values[code] + v4ExpertE2M1Values[code+1]) / 2
		if v < mid || v == mid && code%2 == 0 {
			return sign | code
		}
		if v == mid {
			return sign | (code + 1)
		}
	}
	return sign | 7
}

// v41IndexMXFP4ScaleExp is the upstream ceil-to-power-of-two scale: the amax
// floor keeps a zero tile at UE8M0 byte 1 instead of wrapping bits-1.
func v41IndexMXFP4ScaleExp(amax float32) int {
	amax = max(amax, float32(6*0x1p-126))
	bits := math.Float32bits(float32(amax * float32(1.0/6.0)))
	return int((bits-1)>>23) + 1 - 127
}

func v41IndexMXFP4Encode(x []float32, width int) (codes, sf []byte, err error) {
	if width <= 0 || width%v41IndexMXFP4Group != 0 || len(x)%width != 0 {
		return nil, nil, fmt.Errorf("%w: width %d for %d values", errV41IndexMXFP4, width, len(x))
	}
	codes = make([]byte, len(x))
	sf = make([]byte, len(x)/v41IndexMXFP4Group)
	tile := make([]float32, v41IndexMXFP4Group)
	for g := range sf {
		var amax float32
		for i := range tile {
			raw := x[g*v41IndexMXFP4Group+i]
			tile[i] = v41RoundBF16(raw)
			if !finite32(raw) || !finite32(tile[i]) {
				return nil, nil, fmt.Errorf("%w: non-finite value %d", errV41IndexMXFP4, g*v41IndexMXFP4Group+i)
			}
			amax = max(amax, float32(math.Abs(float64(tile[i]))))
		}
		e := v41IndexMXFP4ScaleExp(amax)
		sf[g] = byte(e + 127)
		for i, v := range tile {
			codes[g*v41IndexMXFP4Group+i] = v41IndexMXFP4E2M1RNE(float32(math.Ldexp(float64(v), -e)))
		}
	}
	return codes, sf, nil
}

func v41IndexMXFP4Decode(codes, sf []byte) []float32 {
	out := make([]float32, len(codes))
	for i, code := range codes {
		out[i] = float32(math.Ldexp(float64(v4ExpertE2M1Values[code]), int(sf[i/v41IndexMXFP4Group])-127))
	}
	return out
}

func v41IndexMXFP4RoundTrip(x []float32, width int) ([]float32, error) {
	codes, sf, err := v41IndexMXFP4Encode(x, width)
	if err != nil {
		return nil, err
	}
	return v41IndexMXFP4Decode(codes, sf), nil
}

// v41IndexMXFP4Score casts Q and K and keeps the reduction weights and the
// f32 logits, so bf16 logit rounding does not mix into this measurement.
func v41IndexMXFP4Score(_ int, q, keys, weights []float32, heads, headDim, rows int) ([]float32, error) {
	qq, err := v41IndexMXFP4RoundTrip(q, headDim)
	if err != nil {
		return nil, err
	}
	kk, err := v41IndexMXFP4RoundTrip(keys, headDim)
	if err != nil {
		return nil, err
	}
	return V41IndexerScore(qq, kk, weights, heads, headDim, rows)
}

func v41IndexMXFP4AssertBits(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("value %d = %g (%#08x), want %g (%#08x)", i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func v41IndexMXFP4Tile(head ...float32) []float32 {
	tile := make([]float32, v41IndexMXFP4Group)
	copy(tile, head)
	return tile
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestV41IndexMXFP4RoundTrip(t *testing.T) {
	t.Parallel()
	negZero := float32(math.Copysign(0, -1))
	t.Run("rne_ties", func(t *testing.T) {
		in := v41IndexMXFP4Tile(6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5,
			-0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, 0.5, 3)
		want := v41IndexMXFP4Tile(6, 0, 1, 1, 2, 2, 4, 4,
			negZero, -1, -1, -2, -2, -4, -4, 0.5, 3)
		codes, sf, err := v41IndexMXFP4Encode(in, v41IndexMXFP4Group)
		if err != nil || len(sf) != 1 || sf[0] != 127 {
			t.Fatalf("sf=%v err=%v", sf, err)
		}
		if codes[0] != 7 || codes[1] != 0 || codes[8] != 8 || codes[7] != 6 || codes[14] != 14 {
			t.Fatalf("codes=%v", codes)
		}
		v41IndexMXFP4AssertBits(t, v41IndexMXFP4Decode(codes, sf), want)
	})
	t.Run("ceil_scale_amax7", func(t *testing.T) {
		codes, sf, err := v41IndexMXFP4Encode(v41IndexMXFP4Tile(7, 1, 0.75, 2.5), v41IndexMXFP4Group)
		if err != nil || sf[0] != 128 {
			t.Fatalf("sf=%v err=%v", sf, err)
		}
		v41IndexMXFP4AssertBits(t, v41IndexMXFP4Decode(codes, sf), v41IndexMXFP4Tile(8, 1, 1, 2))
	})
	t.Run("power_of_two_scale", func(t *testing.T) {
		for _, k := range []int{-100, -3, 0, 7, 100} {
			p := float32(math.Ldexp(1, k))
			in := v41IndexMXFP4Tile(6*p, -1.5*p, 0.5*p)
			codes, sf, err := v41IndexMXFP4Encode(in, v41IndexMXFP4Group)
			if err != nil || int(sf[0]) != 127+k || codes[0] != 7 {
				t.Fatalf("k=%d codes=%v sf=%v err=%v", k, codes[:3], sf, err)
			}
			v41IndexMXFP4AssertBits(t, v41IndexMXFP4Decode(codes, sf), in)
		}
		rng := rand.New(rand.NewPCG(11, 4))
		for i := 0; i < 4096; i++ {
			tile := make([]float32, v41IndexMXFP4Group)
			var amax float32
			for j := range tile {
				tile[j] = v41RoundBF16(float32(rng.NormFloat64() * math.Exp(8*rng.NormFloat64())))
				amax = max(amax, float32(math.Abs(float64(tile[j]))))
			}
			e := v41IndexMXFP4ScaleExp(amax)
			for _, v := range tile {
				if scaled := math.Abs(math.Ldexp(float64(v), -e)); scaled > 6 {
					t.Fatalf("scaled magnitude %g exceeds e2m1 max at amax=%g", scaled, amax)
				}
			}
		}
	})
	t.Run("zero_tile", func(t *testing.T) {
		codes, sf, err := v41IndexMXFP4Encode(make([]float32, v41IndexMXFP4Group), v41IndexMXFP4Group)
		if err != nil || sf[0] != 0x01 {
			t.Fatalf("sf=%v err=%v", sf, err)
		}
		for i, code := range codes {
			if code != 0 {
				t.Fatalf("code %d = %d", i, code)
			}
		}
		v41IndexMXFP4AssertBits(t, v41IndexMXFP4Decode(codes, sf), make([]float32, v41IndexMXFP4Group))
	})
	t.Run("per_group_independent", func(t *testing.T) {
		row := make([]float32, 128)
		row[0], row[1] = 6, 0.75
		row[32] = 0.75
		row[64], row[65] = 7, 0.75
		codes, sf, err := v41IndexMXFP4Encode(row, 128)
		if err != nil || len(sf) != 4 || sf[0] != 127 || sf[1] != 124 || sf[2] != 128 || sf[3] != 1 {
			t.Fatalf("sf=%v err=%v", sf, err)
		}
		out := v41IndexMXFP4Decode(codes, sf)
		if out[1] != 1 || out[32] != 0.75 || out[65] != 1 || out[64] != 8 {
			t.Fatalf("group outputs %g %g %g %g", out[1], out[32], out[64], out[65])
		}
	})
	t.Run("bf16_preround", func(t *testing.T) {
		codes, sf, err := v41IndexMXFP4Encode(v41IndexMXFP4Tile(6+6*0x1p-12, 1.5), v41IndexMXFP4Group)
		if err != nil || sf[0] != 127 {
			t.Fatalf("sf=%v err=%v", sf, err)
		}
		v41IndexMXFP4AssertBits(t, v41IndexMXFP4Decode(codes, sf), v41IndexMXFP4Tile(6, 1.5))
	})
	t.Run("element_saturation", func(t *testing.T) {
		for _, tc := range []struct {
			in   float32
			want byte
		}{{6.5, 7}, {1e30, 7}, {-7, 15}, {5.0000005, 7}, {4.9999995, 6}, {0.25000003, 1}, {negZero, 8}} {
			if got := v41IndexMXFP4E2M1RNE(tc.in); got != tc.want {
				t.Fatalf("e2m1(%g) = %d, want %d", tc.in, got, tc.want)
			}
		}
	})
	t.Run("refusals", func(t *testing.T) {
		nanPayload := math.Float32frombits(0x7fffffff)
		for _, tc := range []struct {
			name  string
			x     []float32
			width int
		}{
			{"zero width", make([]float32, 32), 0},
			{"negative width", make([]float32, 32), -32},
			{"width not group multiple", make([]float32, 96), 48},
			{"length not row multiple", make([]float32, 96), 64},
			{"nan", v41IndexMXFP4Tile(float32(math.NaN())), 32},
			{"nan payload", v41IndexMXFP4Tile(1, nanPayload), 32},
			{"positive inf", v41IndexMXFP4Tile(float32(math.Inf(1))), 32},
			{"negative inf", v41IndexMXFP4Tile(0, float32(math.Inf(-1))), 32},
			{"bf16 overflow", v41IndexMXFP4Tile(math.MaxFloat32), 32},
		} {
			if _, err := v41IndexMXFP4RoundTrip(tc.x, tc.width); !errors.Is(err, errV41IndexMXFP4) {
				t.Fatalf("%s: err=%v", tc.name, err)
			}
		}
	})
}

func v41IndexMXFP4Fixture(rng *rand.Rand, n, width int, outlier bool) []float32 {
	x := make([]float32, n)
	for i := range x {
		v := rng.NormFloat64()
		if outlier && (i%width)%v41IndexMXFP4Group == 7 {
			v *= 20
		}
		x[i] = v41RoundBF16(float32(v))
	}
	return x
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41IndexMXFP4RoundTripIdempotent(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewPCG(3019, 12640))
	x := v41IndexMXFP4Fixture(rng, 4096, 128, true)
	for i := 0; i < 512; i++ {
		x[i] = float32(float64(x[i]) * math.Exp(10*rng.NormFloat64()))
	}
	copy(x[1024:1056], make([]float32, 32))
	once, err := v41IndexMXFP4RoundTrip(x, 128)
	if err != nil {
		t.Fatal(err)
	}
	twice, err := v41IndexMXFP4RoundTrip(once, 128)
	if err != nil {
		t.Fatal(err)
	}
	v41IndexMXFP4AssertBits(t, twice, once)
	for i, v := range once {
		if v41RoundBF16(v) != v || !finite32(v) {
			t.Fatalf("value %d = %g is not a finite bf16 value", i, v)
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=150ms lane=default
func TestV41IndexMXFP4TopKOverlap(t *testing.T) {
	t.Parallel()
	const heads, headDim, rows, topK = 32, 128, 8192, 512
	for _, tc := range []struct {
		name    string
		seed    uint64
		outlier bool
	}{{"gaussian", 1, false}, {"outlier_channel", 2, true}} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rng := rand.New(rand.NewPCG(tc.seed, 3019))
			q := v41IndexMXFP4Fixture(rng, heads*headDim, headDim, tc.outlier)
			keys := v41IndexMXFP4Fixture(rng, rows*headDim, headDim, tc.outlier)
			weights := v41IndexMXFP4Fixture(rng, heads, heads, false)

			qq, err := v41IndexMXFP4RoundTrip(q, headDim)
			if err != nil {
				t.Fatal(err)
			}
			again, err := v41IndexMXFP4RoundTrip(qq, headDim)
			if err != nil {
				t.Fatal(err)
			}
			v41IndexMXFP4AssertBits(t, again, qq)

			base, err := V41IndexerScore(q, keys, weights, heads, headDim, rows)
			if err != nil {
				t.Fatal(err)
			}
			cast, err := v41IndexMXFP4Score(0, q, keys, weights, heads, headDim, rows)
			if err != nil {
				t.Fatal(err)
			}
			selected := make([][]int, 2)
			for arm, score := range [][]float32{base, cast} {
				got, err := V41SelectIndexRows(score, rows, topK, 0, nil)
				if err != nil || len(got) != topK {
					t.Fatalf("arm %d: %d rows err=%v", arm, len(got), err)
				}
				seen := make(map[int]bool, topK)
				for _, row := range got {
					if row < 0 || int(row) >= rows || seen[int(row)] {
						t.Fatalf("arm %d: row %d out of range or repeated", arm, row)
					}
					seen[int(row)] = true
					selected[arm] = append(selected[arm], int(row))
				}
			}
			overlap := 0
			inBase := make(map[int]bool, topK)
			for _, row := range selected[0] {
				inBase[row] = true
			}
			for _, row := range selected[1] {
				if inBase[row] {
					overlap++
				}
			}
			jaccard := jaccardInt(selected[0], selected[1])
			if overlap <= 0 || overlap >= topK || jaccard <= 0 || jaccard >= 1 {
				t.Fatalf("overlap=%d jaccard=%g: emulation is vacuous or disjoint", overlap, jaccard)
			}
			t.Logf("MODELED index MXFP4 emulation vs bf16-widened f32 (%s): overlap=%d/%d jaccard=%.4f", tc.name, overlap, topK, jaccard)
		})
	}
}
