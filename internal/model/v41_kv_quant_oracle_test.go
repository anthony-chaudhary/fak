package model

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// Unwired host emulation of the DeepInfer V4.1 KV cache casts applied to fak's
// projected window and compressed rows: e4m3 with one UE8M0 scale per 32
// values for window rows (and compressed rows when FP4 is off), e2m1 with one
// e4m3 scale per 16 values for compressed rows. It measures the attention
// output delta against fak's bf16 rows on a synthetic fixture; it is not
// parity with the official engine and no forward path calls it.

var errV41KVQuantOracle = errors.New("model: v41 KV quant oracle refused")

const v41KVQuantOracleFP4Group = 16

// v41KVQuantOracleE4M3RNE narrows a finite non-negative value to the nearest
// e4m3 code, ties to even, saturating at 448.
func v41KVQuantOracleE4M3RNE(v float32) byte {
	for code := byte(0); code < 0x7e; code++ {
		mid := (fp8E4M3ToF32(code) + fp8E4M3ToF32(code+1)) / 2
		if v < mid || v == mid && code%2 == 0 {
			return code
		}
		if v == mid {
			return code + 1
		}
	}
	return 0x7e
}

func v41KVQuantOracleBF16Row(row []float32, group int) ([]float32, error) {
	if len(row) == 0 || len(row)%group != 0 {
		return nil, fmt.Errorf("%w: row width %d for group %d", errV41KVQuantOracle, len(row), group)
	}
	out := make([]float32, len(row))
	for i, v := range row {
		out[i] = v41RoundBF16(v)
		if !finite32(v) || !finite32(out[i]) {
			return nil, fmt.Errorf("%w: non-finite value %d", errV41KVQuantOracle, i)
		}
	}
	return out, nil
}

func v41KVQuantOracleFP4Encode(row []float32) (codes, sf []byte, err error) {
	b, err := v41KVQuantOracleBF16Row(row, v41KVQuantOracleFP4Group)
	if err != nil {
		return nil, nil, err
	}
	codes = make([]byte, len(b))
	sf = make([]byte, len(b)/v41KVQuantOracleFP4Group)
	for g := range sf {
		tile := b[g*v41KVQuantOracleFP4Group : (g+1)*v41KVQuantOracleFP4Group]
		var amax float32
		for _, v := range tile {
			amax = max(amax, float32(math.Abs(float64(v))))
		}
		// The scale is narrowed to e4m3 first and its decoded value inverted, so
		// the codes are quantized with exactly the scale stored beside them.
		sf[g] = v41KVQuantOracleE4M3RNE(min(float32(max(amax, float32(6*0x1p-9))*float32(1.0/6.0)), 448))
		mult := float32(1) / fp8E4M3ToF32(sf[g])
		for i, v := range tile {
			codes[g*v41KVQuantOracleFP4Group+i] = v41IndexMXFP4E2M1RNE(float32(v * mult))
		}
	}
	return codes, sf, nil
}

func v41KVQuantOracleFP4Decode(codes, sf []byte) []float32 {
	out := make([]float32, len(codes))
	for i, code := range codes {
		out[i] = v41RoundBF16(float32(v4ExpertE2M1Values[code] * fp8E4M3ToF32(sf[i/v41KVQuantOracleFP4Group])))
	}
	return out
}

func v41KVQuantOracleFP4Row(row []float32) ([]float32, error) {
	codes, sf, err := v41KVQuantOracleFP4Encode(row)
	if err != nil {
		return nil, err
	}
	return v41KVQuantOracleFP4Decode(codes, sf), nil
}

func v41KVQuantOracleFP8Row(row []float32) ([]float32, error) {
	b, err := v41KVQuantOracleBF16Row(row, 32)
	if err != nil {
		return nil, err
	}
	out, err := v41DSparkActFP8(b)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", errV41KVQuantOracle, err)
	}
	return out, nil
}

func v41KVQuantOracleProfileRows(window, compressed [][]float32, compressedFP4 bool) (w, c [][]float32, err error) {
	w = make([][]float32, len(window))
	for i, row := range window {
		if w[i], err = v41KVQuantOracleFP8Row(row); err != nil {
			return nil, nil, err
		}
	}
	cast := v41KVQuantOracleFP8Row
	if compressedFP4 {
		cast = v41KVQuantOracleFP4Row
	}
	c = make([][]float32, len(compressed))
	for i, row := range compressed {
		if c[i], err = cast(row); err != nil {
			return nil, nil, err
		}
	}
	return w, c, nil
}

// v41KVQuantOracleFP8DivExp is the store-kernel scale rule, which divides by
// 448 where v41DSparkActFP8 multiplies by float32(1/448).
func v41KVQuantOracleFP8DivExp(amax float32) int {
	bits := math.Float32bits(float32(max(amax, float32(1e-4)) / 448))
	return int((bits-1)>>23) + 1 - 127
}

func v41KVQuantOracleTile(n int, head ...float32) []float32 {
	tile := make([]float32, n)
	copy(tile, head)
	return tile
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestV41KVQuantOracleFP4Tiles(t *testing.T) {
	t.Parallel()
	negZero := float32(math.Copysign(0, -1))
	for _, tc := range []struct {
		name   string
		in     []float32
		want   []float32
		sfCode byte
	}{
		{"rne_ties", []float32{6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, 0.5},
			[]float32{6, 0, 1, 1, 2, 2, 4, 4, negZero, -1, -1, -2, -2, -4, -4, 0.5}, 0x38},
		{"scale_rounds_to_nearest", v41KVQuantOracleTile(16, 6.1875, 1), v41KVQuantOracleTile(16, 6, 1), 0x38},
		{"scale_tie_to_even", v41KVQuantOracleTile(16, -6.375, 3.5), v41KVQuantOracleTile(16, -6, 4), 0x38},
		{"zero_tile", make([]float32, 16), make([]float32, 16), 0x01},
		{"clamp_floor", v41KVQuantOracleTile(16, 0x1p-8, 3*0x1p-11), v41KVQuantOracleTile(16, 0x1p-8, 0x1p-9), 0x01},
		{"clamp_ceiling", v41KVQuantOracleTile(16, 10752, 448), v41KVQuantOracleTile(16, 2688, 448), 0x7e},
	} {
		codes, sf, err := v41KVQuantOracleFP4Encode(tc.in)
		if err != nil || len(sf) != 1 || sf[0] != tc.sfCode {
			t.Fatalf("%s: sf=%v err=%v", tc.name, sf, err)
		}
		v41IndexMXFP4AssertBits(t, v41KVQuantOracleFP4Decode(codes, sf), tc.want)
	}
	for _, k := range []int{-9, -3, 0, 4, 8} {
		p := float32(math.Ldexp(1, k))
		in := v41KVQuantOracleTile(32, 6*p, 1.5*p, -0.5*p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -6*p, 3*p)
		out, err := v41KVQuantOracleFP4Row(in)
		if err != nil {
			t.Fatal(err)
		}
		v41IndexMXFP4AssertBits(t, out, in)
	}
	rng := rand.New(rand.NewPCG(5, 12640))
	row := make([]float32, 512)
	for i := range row {
		row[i] = float32(rng.NormFloat64() * math.Exp(4*rng.NormFloat64()))
	}
	out, err := v41KVQuantOracleFP4Row(row)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range out {
		if !finite32(v) || v41RoundBF16(v) != v {
			t.Fatalf("value %d = %g is not a finite bf16 value", i, v)
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=30ms lane=default
func TestV41KVQuantOracleFP8Scale(t *testing.T) {
	t.Parallel()
	check := func(v float32) {
		t.Helper()
		got, err := v41DSparkActFP8(v41KVQuantOracleTile(32, 448, v))
		if want := fp8E4M3ToF32(v41KVQuantOracleE4M3RNE(v)); err != nil || got[1] != want {
			t.Fatalf("e4m3(%g): act=%g scalar=%g err=%v", v, got[1], want, err)
		}
	}
	for code := byte(0); code < 0x7e; code++ {
		lo, hi := fp8E4M3ToF32(code), fp8E4M3ToF32(code+1)
		mid := (lo + hi) / 2
		for _, v := range []float32{lo, mid, math.Nextafter32(mid, 0), math.Nextafter32(mid, 448)} {
			check(v)
		}
	}
	rng := rand.New(rand.NewPCG(7, 3019))
	for i := 0; i < 2048; i++ {
		check(float32(448 * rng.Float64()))
	}
	if v41KVQuantOracleE4M3RNE(500) != 0x7e || v41KVQuantOracleE4M3RNE(0x1p-11) != 0 || v41KVQuantOracleE4M3RNE(3*0x1p-11) != 0x01 {
		t.Fatal("scalar e4m3 saturation or subnormal rounding drifted")
	}

	for _, tc := range []struct {
		amax float32
		exp  int
	}{
		{5e-5, -22},
		{v41RoundBF16(1e-6), -22},
	} {
		if got := v41KVQuantOracleFP8DivExp(tc.amax); got != tc.exp {
			t.Fatalf("floor amax=%g exp=%d want %d", tc.amax, got, tc.exp)
		}
	}
	for _, k := range []int{-21, -8, 0, 5, 20} {
		boundary := float32(math.Ldexp(448, k))
		below, above := boundary, boundary
		for step := 0; step < 2; step++ {
			below, above = math.Nextafter32(below, 0), math.Nextafter32(above, float32(math.Inf(1)))
			for _, tc := range []struct {
				amax float32
				exp  int
			}{{boundary, k}, {below, k}, {above, k + 1}} {
				e := v41KVQuantOracleFP8DivExp(tc.amax)
				if e != tc.exp {
					t.Fatalf("division rule amax=%g exp=%d want %d", tc.amax, e, tc.exp)
				}
				probe := v41KVQuantOracleTile(32, tc.amax, float32(math.Ldexp(3, e-10)), float32(math.Ldexp(1, e-9)))
				got, err := v41DSparkActFP8(probe)
				if err != nil {
					t.Fatal(err)
				}
				lead := v41RoundBF16(float32(math.Ldexp(float64(fp8E4M3ToF32(v41KVQuantOracleE4M3RNE(float32(math.Ldexp(float64(tc.amax), -e))))), e)))
				want := v41KVQuantOracleTile(32, lead, float32(math.Ldexp(1, e-8)), float32(math.Ldexp(1, e-9)))
				v41IndexMXFP4AssertBits(t, got, want)
			}
		}
	}
	out, err := v41KVQuantOracleFP8Row(v41KVQuantOracleTile(32, 1e-6, 1e-9))
	if err != nil {
		t.Fatal(err)
	}
	v41IndexMXFP4AssertBits(t, out, v41KVQuantOracleTile(32, 0x1p-20, 0x1p-30))
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41KVQuantOracleRefusals(t *testing.T) {
	t.Parallel()
	good := make([]float32, 512)
	for _, tc := range []struct {
		name               string
		window, compressed [][]float32
		fp4                bool
	}{
		{"fp4 width", [][]float32{good}, [][]float32{make([]float32, 500)}, true},
		{"window width", [][]float32{make([]float32, 16)}, nil, true},
		{"fp8 compressed width", nil, [][]float32{make([]float32, 48)}, false},
		{"empty row", [][]float32{good}, [][]float32{{}}, true},
		{"nan", [][]float32{v41KVQuantOracleTile(512, float32(math.NaN()))}, nil, true},
		{"inf", nil, [][]float32{v41KVQuantOracleTile(512, 0, float32(math.Inf(-1)))}, true},
		{"bf16 overflow", nil, [][]float32{v41KVQuantOracleTile(512, math.MaxFloat32)}, false},
	} {
		if _, _, err := v41KVQuantOracleProfileRows(tc.window, tc.compressed, tc.fp4); !errors.Is(err, errV41KVQuantOracle) {
			t.Fatalf("%s: err=%v", tc.name, err)
		}
	}
}

type v41KVQuantOracleFixture struct {
	heads, headDim, pos, ratio int
	q, sink                    []float32
	window, compressed         [][]float32
	ids                        []int32
	scale                      float32
}

func newV41KVQuantOracleFixture(seed uint64) v41KVQuantOracleFixture {
	rng := rand.New(rand.NewPCG(seed, 12640))
	f := v41KVQuantOracleFixture{heads: 8, headDim: 512, pos: 63, ratio: 4, scale: float32(1 / math.Sqrt(512))}
	vec := func(n int) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = v41RoundBF16(float32(rng.NormFloat64()))
		}
		return out
	}
	f.q, f.sink = vec(f.heads*f.headDim), vec(f.heads)
	for i := 0; i <= f.pos; i++ {
		f.window = append(f.window, vec(f.headDim))
	}
	for i := 0; i < 16; i++ {
		f.compressed = append(f.compressed, vec(f.headDim))
		f.ids = append(f.ids, int32(i))
	}
	return f
}

func (f v41KVQuantOracleFixture) attend(t *testing.T, window, compressed [][]float32) []float32 {
	t.Helper()
	out, err := v41CombinedAttention(0, f.pos, f.ratio, f.heads, f.headDim, f.q, f.sink, window, compressed, f.ids, f.scale, nil)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func v41KVQuantOracleRelL2(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(num / den)
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=30ms lane=default
func TestV41KVQuantOracleGridRowsAttendExactly(t *testing.T) {
	t.Parallel()
	f := newV41KVQuantOracleFixture(41)
	rng := rand.New(rand.NewPCG(43, 3019))
	gridFP8 := func(rows [][]float32) [][]float32 {
		out := make([][]float32, len(rows))
		for i, row := range rows {
			var err error
			if out[i], err = v41KVQuantOracleFP8Row(row); err != nil {
				t.Fatal(err)
			}
		}
		return out
	}
	window, compressedFP8 := gridFP8(f.window), gridFP8(f.compressed)
	compressedFP4 := make([][]float32, len(f.compressed))
	for i := range compressedFP4 {
		row := make([]float32, f.headDim)
		for g := 0; g < f.headDim; g += v41KVQuantOracleFP4Group {
			sf := fp8E4M3ToF32(byte(40 + rng.IntN(60)))
			for j := 0; j < v41KVQuantOracleFP4Group; j++ {
				row[g+j] = v4ExpertE2M1Values[rng.IntN(16)] * sf
			}
			row[g+rng.IntN(v41KVQuantOracleFP4Group)] = v4ExpertE2M1Values[7+8*rng.IntN(2)] * sf
		}
		compressedFP4[i] = row
	}
	for _, tc := range []struct {
		name       string
		compressed [][]float32
		fp4        bool
	}{{"fp4_compressed", compressedFP4, true}, {"fp8_compressed", compressedFP8, false}} {
		w, c, err := v41KVQuantOracleProfileRows(window, tc.compressed, tc.fp4)
		if err != nil {
			t.Fatal(err)
		}
		for i := range w {
			v41IndexMXFP4AssertBits(t, w[i], window[i])
		}
		for i := range c {
			v41IndexMXFP4AssertBits(t, c[i], tc.compressed[i])
		}
		v41IndexMXFP4AssertBits(t, f.attend(t, w, c), f.attend(t, window, tc.compressed))
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=20ms lane=default
func TestV41KVQuantOracleAttentionDelta(t *testing.T) {
	t.Parallel()
	f := newV41KVQuantOracleFixture(64)
	base := f.attend(t, f.window, f.compressed)
	v41IndexMXFP4AssertBits(t, f.attend(t, f.window, f.compressed), base)
	delta := make(map[bool]float64, 2)
	for _, fp4 := range []bool{false, true} {
		w, c, err := v41KVQuantOracleProfileRows(f.window, f.compressed, fp4)
		if err != nil {
			t.Fatal(err)
		}
		out := f.attend(t, w, c)
		for i, v := range out {
			if !finite32(v) {
				t.Fatalf("fp4=%t output %d is non-finite", fp4, i)
			}
		}
		delta[fp4] = v41KVQuantOracleRelL2(out, base)
	}
	if !(delta[false] > 0 && delta[false] < delta[true] && delta[true] < 0.5) {
		t.Fatalf("relative L2 fp8-compressed=%g fp4-compressed=%g outside 0 < fp8 < fp4 < 0.5", delta[false], delta[true])
	}
	t.Logf("SW-VERIFIED emulated DeepInfer KV cast vs fak bf16 rows: rel_l2 window_fp8+compressed_fp8=%.5f window_fp8+compressed_fp4=%.5f", delta[false], delta[true])
}
