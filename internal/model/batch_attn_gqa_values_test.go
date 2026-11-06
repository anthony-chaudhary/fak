package model

import (
	"fmt"
	"math"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestSplitKGQAValuesScalarBits(t *testing.T) {
	special := []uint32{0, 0x80000000, 0x3f800000, 0xbf800000, 0x00800000, 1,
		0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
		0x7fc00001, 0x7fc12345, 0xffc23456, 0x7f800001}
	for _, width := range []int{1, 7, 8, 15, 16, 33, 64, 128, 256} {
		for mode := 0; mode < 2; mode++ {
			old, got := make([]float32, 3*width), make([]float32, 3*width)
			x := make([]float32, width)
			for i := range old {
				if mode == 0 {
					old[i] = float32(math.Sin(float64(i)*0.17)) * 7
				} else {
					old[i] = math.Float32frombits(special[i%len(special)])
				}
			}
			copy(got, old)
			for step := 0; step < 19; step++ {
				var a [3]float32
				for h := range a {
					if mode == 0 {
						a[h] = float32(math.Cos(float64(step*3+h) * 0.21))
					} else {
						a[h] = math.Float32frombits(special[(step*3+h)%len(special)])
					}
				}
				for i := range x {
					if mode == 0 {
						x[i] = float32(math.Sin(float64(step*width+i) * 0.31))
					} else {
						x[i] = math.Float32frombits(special[(i+step)%len(special)])
					}
				}
				for h := 0; h < 3; h++ {
					saxpy(old[h*width:(h+1)*width], x, a[h])
				}
				saxpy3scalar(got[:width], got[width:2*width], got[2*width:], x, a[0], a[1], a[2])
				assertGQAScalarArithmetic(t, fmt.Sprintf("helper width=%d mode=%d step=%d", width, mode, step), got, old)
			}
		}
	}
}

func assertGQAScalarArithmetic(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) || cap(got) != cap(want) {
		t.Fatalf("%s len/cap got=%d/%d want=%d/%d", label, len(got), cap(got), len(want), cap(want))
	}
	for i := range want {
		if !sameGQAArithmeticResult(got[i], want[i]) {
			t.Fatalf("%s[%d] bits got=%08x want=%08x (%g vs %g)", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]), got[i], want[i])
		}
	}
}

func assertGQAValueBits(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) || cap(got) != cap(want) {
		t.Fatalf("%s len/cap got=%d/%d want=%d/%d", label, len(got), cap(got), len(want), cap(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d] bits got=%08x want=%08x (%g vs %g)", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]), got[i], want[i])
		}
	}
}

func sameGQAArithmeticResult(got, want float32) bool {
	// Register allocation can select either NaN operand's payload for a
	// commutative operation, including under race instrumentation. Every
	// non-NaN bit, including zero and infinity signs, remains part of parity.
	return math.Float32bits(got) == math.Float32bits(want) ||
		(math.IsNaN(float64(got)) && math.IsNaN(float64(want)))
}

// fak-test:runtime fast est=1ms lane=default
func TestGQAArithmeticResultComparison(t *testing.T) {
	for _, tc := range []struct {
		name      string
		got, want uint32
		equal     bool
	}{
		{"same finite", 0x3f800000, 0x3f800000, true},
		{"finite one ULP", 0x3f800001, 0x3f800000, false},
		{"zero sign", 0, 0x80000000, false},
		{"same infinity", 0x7f800000, 0x7f800000, true},
		{"infinity sign", 0x7f800000, 0xff800000, false},
		{"NaN payload and sign", 0x7fc00001, 0xffc23456, true},
		{"signaling NaN", 0x7f800001, 0x7fc12345, true},
		{"unexpected NaN", 0x7fc00001, 0x3f800000, false},
		{"missing NaN", 0x3f800000, 0x7fc00001, false},
		{"NaN versus infinity", 0x7fc00001, 0x7f800000, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := sameGQAArithmeticResult(math.Float32frombits(tc.got), math.Float32frombits(tc.want)); got != tc.equal {
				t.Fatalf("comparison(%08x, %08x) = %t, want %t", tc.got, tc.want, got, tc.equal)
			}
		})
	}
}

func gqaValueRows(nPos, hd, nKV int) decodeKVRows {
	r := decodeKVRows{w: hd * nKV, hd: hd, kf: make([]float32, nPos*hd*nKV), vf: make([]float32, nPos*hd*nKV)}
	for i := range r.kf {
		r.kf[i] = float32(math.Sin(float64(i)*0.113)) * 2
		r.vf[i] = float32(math.Cos(float64(i) * 0.071))
	}
	return r
}

func checkGQAValueChunk(t *testing.T, rows decodeKVRows, a decodeAttnStepArgs, nw, chunk int) {
	t.Helper()
	q := make([]float32, a.nH*a.hd)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.37)) * 3
	}
	checkGQAValueChunkQuery(t, rows, a, nw, chunk, q)
}

func checkGQAValueChunkQuery(t *testing.T, rows decodeKVRows, a decodeAttnStepArgs, nw, chunk int, q []float32) {
	t.Helper()
	nChunks := (a.nPos - a.lo + chunk - 1) / chunk
	old, got := make([]float32, len(q)), make([]float32, len(q))
	oldSC, gotSC := &splitKDecodeScratch{}, &splitKDecodeScratch{}
	var oldScores, gotScores []float32
	for repeat := 0; repeat < 3; repeat++ {
		clear(old)
		clear(got)
		oldScores = splitKGQAValuesBaseline(old, q, oldScores, rows, a, oldSC, nw, chunk, nChunks)
		gotScores = attnDecodeSplitK(got, q, gotScores, rows, a, gotSC, nw, chunk, nChunks)
		assertGQAValueBits(t, "output", got, old)
		assertGQAValueBits(t, "scores", gotScores, oldScores)
		assertGQAValueBits(t, "partial", gotSC.part, oldSC.part)
		assertGQAValueBits(t, "max", gotSC.mx, oldSC.mx)
		assertGQAValueBits(t, "sum", gotSC.sum, oldSC.sum)
		assertGQAValueBits(t, "row scratch", gotSC.rows, oldSC.rows)
	}
}

// fak-test:runtime fast est=300ms lane=default
func TestSplitKGQAValuesChunkBits(t *testing.T) {
	for _, grp := range []int{1, 2, 3, 4, 7, 8} {
		for _, hd := range []int{7, 33, 64} {
			for _, nPos := range []int{1, 65, 257, 1025} {
				t.Run(fmt.Sprintf("g%d-hd%d-n%d", grp, hd, nPos), func(t *testing.T) {
					lo := 0
					if nPos > 257 {
						lo = 37
					}
					a := decodeAttnStepArgs{lo: lo, nPos: nPos, nH: grp * 2, grp: grp, hd: hd, scale: float32(1 / math.Sqrt(float64(hd)))}
					if nPos == 65 || nPos == 1025 {
						a.softcap = 4
						a.sinks = make([]float32, a.nH)
						for h := range a.sinks {
							a.sinks[h] = float32(h%5) - 1
						}
					}
					checkGQAValueChunk(t, gqaValueRows(nPos, hd, 2), a, 1, 64)
				})
			}
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestSplitKGQAValuesExceptionalBits(t *testing.T) {
	inf, nan := float32(math.Inf(1)), math.Float32frombits(0x7fc12345)
	cases := []struct {
		name         string
		score, value []float32
		sink         []float32
		cap          float32
	}{
		{"finite-overflow", []float32{0, 0, 100}, []float32{math.MaxFloat32, math.MaxFloat32, 0}, nil, 0},
		{"opposing-overflow", []float32{0, 0, 0, 100}, []float32{math.MaxFloat32, math.MaxFloat32, -math.MaxFloat32, 0}, nil, 0},
		{"positive-inf", []float32{0, inf, 1}, []float32{1, 2, 3}, nil, 0},
		{"negative-inf", []float32{-inf, -inf, 1}, []float32{1, 2, 3}, nil, 0},
		{"nan-score", []float32{0, nan, 1}, []float32{1, 2, 3}, nil, 0},
		{"nan-value", []float32{0, 1, 2}, []float32{1, nan, 3}, nil, 0},
		{"zero-times-inf", []float32{-inf, 1, 2}, []float32{inf, 2, 3}, nil, 0},
		{"inf-sink", []float32{0, 1, 2}, []float32{1, 2, 3}, []float32{inf, -inf, nan}, 0},
		{"nan-softcap", []float32{0, 1, 2}, []float32{1, 2, 3}, nil, nan},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			for _, grp := range []int{3, 4, 7, 8} {
				// hd=1 makes the exceptional score/value fixture transparent.
				rows := decodeKVRows{w: 1, hd: 1, kf: tc.score, vf: tc.value}
				a := decodeAttnStepArgs{nPos: len(tc.score), nH: grp, grp: grp, hd: 1, scale: 1, sinks: tc.sink, softcap: tc.cap}
				q := make([]float32, grp)
				for i := range q {
					q[i] = 1
				}
				checkGQAValueChunkQuery(t, rows, a, 1, len(tc.score), q)
				checkGQAValueChunkQuery(t, rows, a, 1, 2, q)
			}
		})
	}
}

// Uses real production KVCache Q8 decoding and the real parallel unit scheduler.
// This integration test is deliberately not included in the serial F32 component.
// fak-test:runtime medium est=5s lane=default
func TestBlockStepSplitKGQAValuesQ8ParallelBits(t *testing.T) {
	for _, grp := range []int{3, 4, 7, 8} {
		cfg := Config{HiddenSize: 2 * grp * 32, NumLayers: 1, NumHeads: 2 * grp, NumKVHeads: 2, HeadDim: 32}
		for _, prec := range []KVPrecision{KVPrecisionFP32, KVPrecisionQ8_0} {
			cache := splitKTestCache(cfg, prec, 1000, 0x1652)
			a := decodeAttnStepArgs{lo: 37, nPos: 1000, nH: cfg.NumHeads, grp: grp, hd: 32, scale: float32(1 / math.Sqrt(32)), softcap: 4, sinks: []float32{-1, 0, 1}, cfg: &cfg}
			for _, nw := range []int{1, 3, 8} {
				checkGQAValueChunk(t, newDecodeKVRows(cache, 0, 64, 32), a, nw, 64)
			}
		}
	}
}

// This checks the real attnDecodeStep consumer and its existing split-K selection.
// It does not substitute a new route or test-only dispatch flag.
// fak-test:runtime medium est=2s lane=default
func TestBlockStepSplitKGQAValuesRouteBits(t *testing.T) {
	pinSplitK(t, true, 32)
	const nPos, hd, grp, nw = 1025, 64, 3, 8
	cfg := Config{HiddenSize: 384, NumLayers: 1, NumHeads: 6, NumKVHeads: 2, HeadDim: hd}
	a := decodeAttnStepArgs{lo: 37, nPos: nPos, nH: 6, grp: grp, hd: hd, scale: 0.125, softcap: 4, sinks: []float32{-1, 0, 1}, cfg: &cfg, nw: nw}
	span, nKV := a.nPos-a.lo, a.nH/a.grp
	chunksPerHead := (nw*decodeSplitKOversub + nKV - 1) / nKV
	chunk := max(decodeSplitKMinChunk, (span+chunksPerHead-1)/chunksPerHead)
	nChunks := (span + chunk - 1) / chunk
	if nChunks <= 1 || a.nH*span*a.hd < parThreshold {
		t.Fatal("fixture must enter the existing split-K route")
	}
	q := make([]float32, a.nH*hd)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.37)) * 3
	}
	for _, prec := range []KVPrecision{KVPrecisionFP32, KVPrecisionQ8_0} {
		cache := splitKTestCache(cfg, prec, nPos, 0x1652)
		rows := newDecodeKVRows(cache, 0, nKV*hd, hd)
		old, got := make([]float32, len(q)), make([]float32, len(q))
		oldSC, gotSC := &splitKDecodeScratch{}, &splitKDecodeScratch{}
		oldScores := splitKGQAValuesBaseline(old, q, nil, rows, a, oldSC, nw, chunk, nChunks)
		gotScores := attnDecodeStep(got, q, nil, rows, a, gotSC)
		if len(gotSC.part) == 0 {
			t.Fatal("actual split-K route not observed")
		}
		assertGQAValueBits(t, "route output", got, old)
		assertGQAValueBits(t, "route scores", gotScores, oldScores)
		assertGQAValueBits(t, "route partial", gotSC.part, oldSC.part)
		assertGQAValueBits(t, "route max", gotSC.mx, oldSC.mx)
		assertGQAValueBits(t, "route sum", gotSC.sum, oldSC.sum)
		assertGQAValueBits(t, "route row scratch", gotSC.rows, oldSC.rows)
	}
}

func BenchmarkSplitKGQAValues(b *testing.B) {
	const nPos, hd, nKV = 2049, 64, 2
	for _, grp := range []int{3, 7} {
		b.Run(fmt.Sprintf("grp%d", grp), func(b *testing.B) {
			nH := nKV * grp
			rows := gqaValueRows(nPos, hd, nKV)
			q := make([]float32, nH*hd)
			for i := range q {
				q[i] = float32(math.Sin(float64(i)*0.37)) * 3
			}
			chunk := (nPos + 7) / 8
			chunks := (nPos + chunk - 1) / chunk
			for _, candidate := range []bool{false, true} {
				name := "materialized"
				kernel := splitKGQAValuesBaseline
				if candidate {
					name, kernel = "gqa-values", attnDecodeSplitK
				}
				b.Run(name, func(b *testing.B) {
					a := decodeAttnStepArgs{nPos: nPos, nH: nH, grp: grp, hd: hd, scale: float32(1 / math.Sqrt(hd))}
					out, sc := make([]float32, nH*hd), &splitKDecodeScratch{}
					scores := kernel(out, q, nil, rows, a, sc, 1, chunk, chunks)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						clear(out)
						scores = kernel(out, q, scores, rows, a, sc, 1, chunk, chunks)
					}
					b.StopTimer()
					b.ReportMetric(float64(cap(scores)*4), "score-B")
					b.ReportMetric(float64((cap(sc.part)+cap(sc.mx)+cap(sc.sum))*4), "partial-B")
				})
			}
		})
	}
}

// Exact pre-change materialized body, renamed only. Fak base e519243109923b13
// resolves batch_attn.go to Git blob 63cdb8bf414410e3b701d6695ae8b19a742a9308.
func splitKGQAValuesBaseline(attnOut, q, scores []float32, rows decodeKVRows, a decodeAttnStepArgs, sc *splitKDecodeScratch, nw, chunk, nChunks int) []float32 {
	nKV, grp, hd := a.nH/a.grp, a.grp, a.hd
	units := nKV * nChunks
	scores = grow(scores, units*grp*chunk)
	sc.part = grow(sc.part, units*grp*hd)
	sc.mx = grow(sc.mx, units*grp)
	sc.sum = grow(sc.sum, units*grp)
	if rows.q8 {
		sc.rows = grow(sc.rows, units*2*hd)
	}
	unit := func(u int) {
		kvh, c := u/nChunks, u%nChunks
		j0 := a.lo + c*chunk
		j1 := j0 + chunk
		if j1 > a.nPos {
			j1 = a.nPos
		}
		n := j1 - j0
		var kbuf, vbuf []float32
		if rows.q8 {
			kbuf = sc.rows[u*2*hd : u*2*hd+hd]
			vbuf = sc.rows[u*2*hd+hd : (u+1)*2*hd]
		}
		base := u * grp * chunk
		for j := j0; j < j1; j++ {
			kh := rows.head(j, kvh, true, kbuf)
			for g := 0; g < grp; g++ {
				h := kvh*grp + g
				scores[base+g*chunk+j-j0] = fdot(q[h*hd:(h+1)*hd], kh) * a.scale
			}
		}
		for g := 0; g < grp; g++ {
			s := scores[base+g*chunk : base+g*chunk+n]
			softcapInPlace(s, a.softcap)
			mx := s[0]
			for _, v := range s {
				if v > mx {
					mx = v
				}
			}
			var sum float32
			for i, v := range s {
				e := float32(math.Exp(float64(v - mx)))
				s[i] = e
				sum += e
			}
			sc.mx[u*grp+g], sc.sum[u*grp+g] = mx, sum
			clear(sc.part[(u*grp+g)*hd : (u*grp+g+1)*hd])
		}
		for j := j0; j < j1; j++ {
			vh := rows.head(j, kvh, false, vbuf)
			for g := 0; g < grp; g++ {
				saxpy(sc.part[(u*grp+g)*hd:(u*grp+g+1)*hd], vh, scores[base+g*chunk+j-j0])
			}
		}
	}
	runDecodeUnits(units, nw, unit)

	// Log-sum-exp merge: softmax over the whole span is exp(s-M)/D with M the global
	// max and D = sum_c exp(m_c-M)*sum_c (+ exp(sink-M) for a sink layer), so each
	// chunk's unnormalized partial is scaled by exp(m_c-M)/D and summed.
	for h := 0; h < a.nH; h++ {
		kvh, g := h/grp, h%grp
		mxAll := float32(math.Inf(-1))
		sink, hasSink := sinkFor(a.sinks, h)
		if hasSink {
			mxAll = sink
		}
		for c := 0; c < nChunks; c++ {
			if m := sc.mx[(kvh*nChunks+c)*grp+g]; m > mxAll {
				mxAll = m
			}
		}
		var den float32
		if hasSink {
			den = float32(math.Exp(float64(sink - mxAll)))
		}
		for c := 0; c < nChunks; c++ {
			i := (kvh*nChunks+c)*grp + g
			den += float32(math.Exp(float64(sc.mx[i]-mxAll))) * sc.sum[i]
		}
		out := attnOut[h*hd : (h+1)*hd]
		for c := 0; c < nChunks; c++ {
			i := (kvh*nChunks+c)*grp + g
			wc := float32(math.Exp(float64(sc.mx[i]-mxAll))) / den
			saxpy(out, sc.part[i*hd:(i+1)*hd], wc)
		}
	}
	return scores
}
