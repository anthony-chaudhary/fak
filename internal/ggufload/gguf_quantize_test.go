package ggufload

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestKQuantReferenceGolden(t *testing.T) {
	var doc struct {
		HalfInputs   []uint32 `json:"half_input_bits"`
		HalfExpected []uint16 `json:"half_expected_bits"`
		Pin          string   `json:"upstream_pin"`
		Fixtures     []struct {
			Name  string     `json:"name"`
			Input []uint32   `json:"input_bits"`
			Q4    string     `json:"q4_k_hex"`
			Q5    string     `json:"q5_k_hex"`
			Q6    string     `json:"q6_k_hex"`
			RMSE  [3]float64 `json:"reference_rmse"`
		} `json:"fixtures"`
	}
	b, err := os.ReadFile("testdata/kquant_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	if doc.Pin != "83078fec0db82d6b5a00d9599062c38c39145755" || len(doc.Fixtures) < 9 {
		t.Fatal("incomplete pinned C reference cohort")
	}
	if len(doc.HalfInputs) != 16 || len(doc.HalfExpected) != 16 {
		t.Fatal("missing C half-boundary oracle")
	}
	for i, bits := range doc.HalfInputs {
		if got := kQuantHalf(math.Float32frombits(bits)); got != doc.HalfExpected[i] {
			t.Fatalf("half(%08x)=%04x, C=%04x", bits, got, doc.HalfExpected[i])
		}
	}
	for _, f := range doc.Fixtures {
		t.Run(f.Name, func(t *testing.T) {
			row := make([]float32, len(f.Input))
			for i, bits := range f.Input {
				row[i] = math.Float32frombits(bits)
			}
			for i, typ := range []TensorType{TensorQ4_K, TensorQ5_K, TensorQ6_K} {
				want := []string{f.Q4, f.Q5, f.Q6}[i]
				got, err := quantizeKRow(row, typ)
				if err != nil {
					t.Fatal(err)
				}
				if hex.EncodeToString(got) != want {
					t.Fatalf("%s encoded bytes differ from pinned C oracle", typ)
				}
				decoded, err := dequantF32(TensorInfo{Name: f.Name, Dims: []uint64{uint64(len(row))}, Type: typ}, got)
				if err != nil {
					t.Fatal(err)
				}
				var sum float64
				for j, v := range decoded {
					d := float64(v) - float64(row[j])
					sum += d * d
				}
				rmse := math.Sqrt(sum / float64(len(row)))
				if math.IsNaN(rmse) || rmse > f.RMSE[i]+math.Max(1e-30, f.RMSE[i]*1e-6) {
					t.Fatalf("%s RMSE %g exceeds reference-derived bound %g", typ, rmse, f.RMSE[i])
				}
			}
		})
	}
	for _, v := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), math.MaxFloat32} {
		row := make([]float32, 256)
		row[7] = v
		for _, typ := range []TensorType{TensorQ4_K, TensorQ5_K, TensorQ6_K} {
			if _, err := quantizeKRow(row, typ); err == nil {
				t.Fatalf("%s accepted unsupported value %g", typ, v)
			}
		}
	}
	if _, err := quantizeKRow(make([]float32, 255), TensorQ4_K); err == nil {
		t.Fatal("accepted incomplete block")
	}
}

// The kqRef* functions are a line-by-line transcription of the pinned C
// reference (ggml/src/ggml-quants.c: nearest_int, make_qkx3_quants,
// make_qp_quants, get_scale_min_k4 and the quant_weights branch of
// quantize_row_q4_K_impl) in plain float32 arithmetic. They share no code with
// the encoder under test and act as its oracle.

func kqRefNearestInt(fval float32) int {
	val := float32(fval + 12582912.0)
	i := int32(math.Float32bits(val))
	return int((i & 0x007fffff) - 0x00400000)
}

func kqRefHalf(f float32) uint16 {
	var sign uint16
	if math.Signbit(float64(f)) {
		sign = 0x8000
	}
	a := math.Abs(float64(f))
	if a == 0 {
		return sign
	}
	frac, exp := math.Frexp(a)
	e := exp - 1
	if e < -14 {
		return sign | uint16(math.RoundToEven(math.Ldexp(a, 24)))
	}
	mant := int(math.RoundToEven((frac*2 - 1) * 1024))
	if mant == 1024 {
		mant = 0
		e++
	}
	if e > 15 {
		return sign | 0x7c00
	}
	return sign | uint16(e+15)<<10 | uint16(mant)
}

func kqRefHalfToFloat(h uint16) float32 {
	e := int(h>>10) & 31
	v := float64(h & 1023)
	if e == 0 {
		v = math.Ldexp(v, -24)
	} else {
		v = math.Ldexp(v+1024, e-25)
	}
	if h&0x8000 != 0 {
		v = -v
	}
	return float32(v)
}

func kqRefQKX3(n, nmax int, x, weights []float32, L, Laux []uint8, rmin, rdelta float32, nstep int) (float32, float32) {
	min := x[0]
	max := x[0]
	sum_w := weights[0]
	sum_x := float32(sum_w * x[0])
	for i := 1; i < n; i++ {
		if x[i] < min {
			min = x[i]
		}
		if x[i] > max {
			max = x[i]
		}
		w := weights[i]
		sum_w = float32(sum_w + w)
		sum_x = float32(sum_x + float32(w*x[i]))
	}
	if min > 0 {
		min = 0
	}
	if max <= min {
		for i := 0; i < n; i++ {
			L[i] = 0
		}
		return 0, -min
	}
	iscale := float32(float32(nmax) / float32(max-min))
	scale := float32(1 / iscale)
	var best_mad float32
	for i := 0; i < n; i++ {
		l := kqRefNearestInt(float32(iscale * float32(x[i]-min)))
		if l > nmax {
			l = nmax
		}
		if l < 0 {
			l = 0
		}
		L[i] = uint8(l)
		diff := float32(float32(float32(scale*float32(L[i]))+min) - x[i])
		diff = float32(diff * diff)
		best_mad = float32(best_mad + float32(weights[i]*diff))
	}
	if nstep < 1 {
		return scale, -min
	}
	for is := 0; is <= nstep; is++ {
		iscale = float32(float32(float32(rmin+float32(rdelta*float32(is)))+float32(nmax)) / float32(max-min))
		var sum_l, sum_l2, sum_xl float32
		for i := 0; i < n; i++ {
			l := kqRefNearestInt(float32(iscale * float32(x[i]-min)))
			if l > nmax {
				l = nmax
			}
			if l < 0 {
				l = 0
			}
			Laux[i] = uint8(l)
			w := weights[i]
			sum_l = float32(sum_l + float32(w*float32(l)))
			sum_l2 = float32(sum_l2 + float32(float32(w*float32(l))*float32(l)))
			sum_xl = float32(sum_xl + float32(float32(w*float32(l))*x[i]))
		}
		D := float32(float32(sum_w*sum_l2) - float32(sum_l*sum_l))
		if D > 0 {
			this_scale := float32(float32(float32(sum_w*sum_xl)-float32(sum_x*sum_l)) / D)
			this_min := float32(float32(float32(sum_l2*sum_x)-float32(sum_l*sum_xl)) / D)
			if this_min > 0 {
				this_min = 0
				this_scale = float32(sum_xl / sum_l2)
			}
			var mad float32
			for i := 0; i < n; i++ {
				diff := float32(float32(float32(this_scale*float32(Laux[i]))+this_min) - x[i])
				diff = float32(diff * diff)
				mad = float32(mad + float32(weights[i]*diff))
			}
			if mad < best_mad {
				for i := 0; i < n; i++ {
					L[i] = Laux[i]
				}
				best_mad = mad
				scale = this_scale
				min = this_min
			}
		}
	}
	return scale, -min
}

func kqRefQP(n, nmax int, x []float32, L []uint8, quant_weights []float32) float32 {
	var max float32
	for i := 0; i < n; i++ {
		if !(max > x[i]) {
			max = x[i]
		}
	}
	if max < 1e-15 {
		for i := 0; i < n; i++ {
			L[i] = 0
		}
		return 0
	}
	iscale := float32(float32(nmax) / max)
	for i := 0; i < n; i++ {
		L[i] = uint8(kqRefNearestInt(float32(iscale * x[i])))
	}
	scale := float32(1 / iscale)
	var best_mse float32
	for i := 0; i < n; i++ {
		diff := float32(x[i] - float32(scale*float32(L[i])))
		w := quant_weights[i]
		best_mse = float32(best_mse + float32(float32(w*diff)*diff))
	}
	for is := -4; is <= 4; is++ {
		if is == 0 {
			continue
		}
		iscale_is := float32(float32(float32(float32(0.1)*float32(is))+float32(nmax)) / max)
		scale_is := float32(1 / iscale_is)
		var mse float32
		for i := 0; i < n; i++ {
			l := kqRefNearestInt(float32(iscale_is * x[i]))
			if l > nmax {
				l = nmax
			}
			diff := float32(x[i] - float32(scale_is*float32(l)))
			w := quant_weights[i]
			mse = float32(mse + float32(float32(w*diff)*diff))
		}
		if mse < best_mse {
			best_mse = mse
			iscale = iscale_is
		}
	}
	var sumlx, suml2 float32
	for i := 0; i < n; i++ {
		l := kqRefNearestInt(float32(iscale * x[i]))
		if l > nmax {
			l = nmax
		}
		L[i] = uint8(l)
		w := quant_weights[i]
		sumlx = float32(sumlx + float32(float32(w*x[i])*float32(l)))
		suml2 = float32(suml2 + float32(float32(w*float32(l))*float32(l)))
	}
	for itry := 0; itry < 5; itry++ {
		n_changed := 0
		for i := 0; i < n; i++ {
			w := quant_weights[i]
			slx := float32(sumlx - float32(float32(w*x[i])*float32(L[i])))
			sl2 := float32(suml2 - float32(float32(w*float32(L[i]))*float32(L[i])))
			if slx > 0 && sl2 > 0 {
				new_l := kqRefNearestInt(float32(float32(x[i]*sl2) / slx))
				if new_l > nmax {
					new_l = nmax
				}
				if new_l != int(L[i]) {
					slx = float32(slx + float32(float32(w*x[i])*float32(new_l)))
					sl2 = float32(sl2 + float32(float32(w*float32(new_l))*float32(new_l)))
					if float32(float32(slx*slx)*suml2) > float32(float32(sumlx*sumlx)*sl2) {
						L[i] = uint8(new_l)
						sumlx = slx
						suml2 = sl2
						n_changed++
					}
				}
			}
		}
		if n_changed == 0 {
			break
		}
	}
	if suml2 > 0 {
		return float32(sumlx / suml2)
	}
	return 0
}

func kqRefQ4KImpl(x, quant_weights []float32) []byte {
	nb := len(x) / 256
	out := make([]byte, nb*144)
	var L [256]uint8
	var Laux [32]uint8
	var Ls, Lm [8]uint8
	var weights [32]float32
	var sw, mins, scales [8]float32
	for i := 0; i < nb; i++ {
		y := out[i*144 : (i+1)*144]
		yscales := y[4:16]
		var sum_x2 float32
		for l := 0; l < 256; l++ {
			sum_x2 = float32(sum_x2 + float32(x[l]*x[l]))
		}
		sigma2 := float32(float32(2*sum_x2) / 256)
		for j := 0; j < 8; j++ {
			qw := quant_weights[256*i+32*j:]
			for l := 0; l < 32; l++ {
				xx := float32(x[32*j+l] * x[32*j+l])
				weights[l] = float32(qw[l] * float32(math.Sqrt(float64(float32(sigma2+xx)))))
			}
			var sumw float32
			for l := 0; l < 32; l++ {
				sumw = float32(sumw + weights[l])
			}
			sw[j] = sumw
			scales[j], mins[j] = kqRefQKX3(32, 15, x[32*j:], weights[:], L[32*j:], Laux[:], -0.9, 0.05, 36)
		}
		d_block := kqRefQP(8, 63, scales[:], Ls[:], sw[:])
		m_block := kqRefQP(8, 63, mins[:], Lm[:], sw[:])
		for j := 0; j < 8; j++ {
			ls, lm := Ls[j], Lm[j]
			if j < 4 {
				yscales[j] = ls
				yscales[j+4] = lm
			} else {
				yscales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4)
				yscales[j-4] |= (ls >> 4) << 6
				yscales[j-0] |= (lm >> 4) << 6
			}
		}
		yd, ydmin := kqRefHalf(d_block), kqRefHalf(m_block)
		y[0], y[1] = byte(yd), byte(yd>>8)
		y[2], y[3] = byte(ydmin), byte(ydmin>>8)
		for j := 0; j < 8; j++ {
			var sc, m uint8
			if j < 4 {
				sc, m = yscales[j]&63, yscales[j+4]&63
			} else {
				sc = (yscales[j+4] & 0xF) | ((yscales[j-4] >> 6) << 4)
				m = (yscales[j+4] >> 4) | ((yscales[j-0] >> 6) << 4)
			}
			d := float32(kqRefHalfToFloat(yd) * float32(sc))
			if d == 0 {
				continue
			}
			dm := float32(kqRefHalfToFloat(ydmin) * float32(m))
			for ii := 0; ii < 32; ii++ {
				l := kqRefNearestInt(float32(float32(x[32*j+ii]+dm) / d))
				if l > 15 {
					l = 15
				}
				if l < 0 {
					l = 0
				}
				L[32*j+ii] = uint8(l)
			}
		}
		q := y[16:]
		for j := 0; j < 256; j += 64 {
			for l := 0; l < 32; l++ {
				q[l] = L[j+l] | (L[j+l+32] << 4)
			}
			q = q[32:]
		}
		x = x[256:]
	}
	return out
}

func kqSeededRow(seed int64, n int) []float32 {
	rng := rand.New(rand.NewSource(seed))
	row := make([]float32, n)
	for i := range row {
		row[i] = float32(rng.NormFloat64() * 0.05)
		if rng.Intn(40) == 0 {
			row[i] *= 6
		}
	}
	return row
}

// A skewed importance matrix: a sparse set of columns dominates the rest.
func kqSkewedImatrix(seed int64, n int) []float32 {
	rng := rand.New(rand.NewSource(seed))
	im := make([]float32, n)
	for i := range im {
		im[i] = float32(0.001 + 0.01*rng.Float64())
		if rng.Intn(8) == 0 {
			im[i] = float32(50 + 200*rng.Float64())
		}
	}
	return im
}

func kqWeightedSquaredError(t *testing.T, row, imatrix []float32, encoded []byte) float64 {
	t.Helper()
	decoded, err := dequantF32(TensorInfo{Name: "w", Dims: []uint64{uint64(len(row))}, Type: TensorQ4_K}, encoded)
	if err != nil {
		t.Fatal(err)
	}
	var sum float64
	for i, v := range decoded {
		d := float64(v) - float64(row[i])
		sum += float64(imatrix[i]) * d * d
	}
	return sum
}

// fak-test:runtime fast est=50ms lane=default
func TestQuantizeQ4KImatrixWeightedLowersWeightedError(t *testing.T) {
	t.Run("nil imatrix is the reference encoder", func(t *testing.T) {
		for seed := int64(1); seed <= 4; seed++ {
			row := kqSeededRow(seed, 512)
			want, err := quantizeKRow(row, TensorQ4_K)
			if err != nil {
				t.Fatal(err)
			}
			got, err := QuantizeQ4KWeighted(row, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("seed %d: nil imatrix bytes differ from the reference Q4_K encoder", seed)
			}
		}
	})

	t.Run("skewed imatrix lowers importance-weighted error", func(t *testing.T) {
		for seed := int64(1); seed <= 8; seed++ {
			row := kqSeededRow(seed, 1024)
			im := kqSkewedImatrix(seed+100, len(row))
			plain, err := QuantizeQ4KWeighted(row, nil)
			if err != nil {
				t.Fatal(err)
			}
			weighted, err := QuantizeQ4KWeighted(row, im)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Equal(plain, weighted) {
				t.Fatalf("seed %d: weighted encoding is identical to unweighted", seed)
			}
			ePlain := kqWeightedSquaredError(t, row, im, plain)
			eWeighted := kqWeightedSquaredError(t, row, im, weighted)
			if !(eWeighted < ePlain) {
				t.Fatalf("seed %d: weighted error %g is not below unweighted %g", seed, eWeighted, ePlain)
			}
		}
	})

	t.Run("bytes match the upstream transcription", func(t *testing.T) {
		cases := map[string]func() ([]float32, []float32){
			"skewed": func() ([]float32, []float32) {
				row := kqSeededRow(11, 768)
				return row, kqSkewedImatrix(12, len(row))
			},
			"uniform": func() ([]float32, []float32) {
				row := kqSeededRow(13, 256)
				im := make([]float32, len(row))
				for i := range im {
					im[i] = 1
				}
				return row, im
			},
			"zero importance group and zero value group": func() ([]float32, []float32) {
				row := kqSeededRow(14, 512)
				im := kqSkewedImatrix(15, len(row))
				clear(im[32:64])
				clear(row[96:128])
				clear(row[256:512])
				return row, im
			},
			"positive-only and wide range": func() ([]float32, []float32) {
				row := kqSeededRow(16, 512)
				for i := range row[:256] {
					row[i] = float32(math.Abs(float64(row[i]))) + 0.5
				}
				for i := range row[256:] {
					row[256+i] *= 4000
				}
				return row, kqSkewedImatrix(17, len(row))
			},
		}
		for name, build := range cases {
			row, im := build()
			got, err := QuantizeQ4KWeighted(row, im)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			want := kqRefQ4KImpl(row, im)
			if !bytes.Equal(got, want) {
				t.Fatalf("%s: weighted bytes differ from the upstream transcription\n got %x\nwant %x", name, got, want)
			}
		}
	})

	t.Run("qkx3 and qp hand vectors", func(t *testing.T) {
		ones := []float32{1, 1, 1, 1}
		for _, c := range []struct {
			x          []float32
			nmax       int
			scale, min float32
			codes      []uint8
		}{
			{[]float32{0, 1, 2, 3}, 3, 1, 0, []uint8{0, 1, 2, 3}},
			{[]float32{0, 4, 8, 60}, 15, 4, 0, []uint8{0, 1, 2, 15}},
			{[]float32{-8, 0, 8, 52}, 15, 4, 8, []uint8{0, 2, 4, 15}},
			{[]float32{-1, -1, -1, -1}, 15, 0, 1, []uint8{0, 0, 0, 0}},
			{[]float32{0, 0, 0, 0}, 15, 0, 0, []uint8{0, 0, 0, 0}},
		} {
			m := &kQuantMath{}
			codes := []uint8{9, 9, 9, 9}
			scale, minimum := m.qkx3(c.x, ones, codes, c.nmax, -0.9, 0.05, 36)
			if m.err != nil {
				t.Fatal(m.err)
			}
			if scale != c.scale || minimum != c.min || !bytes.Equal(codes, c.codes) {
				t.Fatalf("qkx3(%v)=(%g,%g,%v), want (%g,%g,%v)", c.x, scale, minimum, codes, c.scale, c.min, c.codes)
			}
		}
		m := &kQuantMath{}
		codes := []uint8{9, 9, 9, 9}
		if scale := m.qp([]float32{0, 4, 8, 252}, codes, ones, 63); m.err != nil || scale != 4 || !bytes.Equal(codes, []uint8{0, 1, 2, 63}) {
			t.Fatalf("qp=(%g,%v) err=%v, want (4,[0 1 2 63])", scale, codes, m.err)
		}
		if scale := m.qp([]float32{0, 0, 0, 0}, codes, ones, 63); scale != 0 || !bytes.Equal(codes, []uint8{0, 0, 0, 0}) {
			t.Fatalf("qp(zero)=(%g,%v), want (0,[0 0 0 0])", scale, codes)
		}
	})

	t.Run("refuses malformed importance input", func(t *testing.T) {
		row := kqSeededRow(21, 256)
		if _, err := QuantizeQ4KWeighted(row, make([]float32, 255)); err == nil {
			t.Fatal("accepted an importance matrix of the wrong length")
		}
		for _, bad := range []float32{-1, float32(math.NaN()), float32(math.Inf(1))} {
			im := make([]float32, 256)
			im[3] = bad
			if _, err := QuantizeQ4KWeighted(row, im); err == nil {
				t.Fatalf("accepted importance value %g", bad)
			}
		}
		if _, err := QuantizeQ4KWeighted(make([]float32, 255), make([]float32, 255)); err == nil {
			t.Fatal("accepted an incomplete block")
		}
	})
}
