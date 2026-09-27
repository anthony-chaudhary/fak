//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
	"time"
)

// Witnesses for the packed Q8_0 KV consumer in the Qwen3.8 graph attention
// (#12981): FullAttentionQ8 (host-owned packed prefix) and FullAttentionDeviceQ8
// (persistent DeviceKVQ8 store).
//
//   - codec:  the device quantizer emits the host cache's exact Q8_0 bytes.
//   - oracle: attention over the packed rows matches a float64 oracle that attends
//     over the same dequantized rows, so the kernel consumes the layout correctly.
//   - f32:    against the F32 entry, KRaw/KPost are bit-identical and the attention
//     output stays inside a bounded quantization error.
//   - walk:   a persistent packed store reproduces the host-packed path bit for bit
//     across a prefill panel plus decode steps, at a non-zero layer slice.
//   - greedy: a synthetic prefill+decode loop past the split-KV threshold emits the
//     same greedy tokens through the Q8 cache as through the F32 cache.

// requireKVQ8 skips only when Metal itself is absent. On a Metal device a Q8
// library that fails to compile is a defect, so it fails instead of skipping.
func requireKVQ8(tb testing.TB) {
	tb.Helper()
	if !Available() {
		tb.Skip("Metal unavailable")
	}
	if !KVQ8Available() {
		tb.Fatal("Metal is available but the packed Q8 KV pipelines did not compile")
	}
}

// refQuantizeKVQ8_0 mirrors internal/model.QuantizeKVQ8_0 statement for statement:
// symmetric block-32, scale = maxAbs/127 in f32, code = math.Round(x*(1/scale)) with
// the reciprocal in float64. metalgemm cannot import internal/model (model imports
// metalgemm), so the device codec is pinned to this mirror.
func refQuantizeKVQ8_0(src []float32) ([]int8, []float32) {
	blocks := (len(src) + KVQ8BlockSize - 1) / KVQ8BlockSize
	scales, codes := make([]float32, blocks), make([]int8, len(src))
	for b := 0; b < blocks; b++ {
		lo, hi := b*KVQ8BlockSize, min((b+1)*KVQ8BlockSize, len(src))
		var maxAbs float32
		for _, v := range src[lo:hi] {
			abs := v
			if abs < 0 {
				abs = -abs
			}
			if abs > maxAbs {
				maxAbs = abs
			}
		}
		if maxAbs == 0 {
			continue
		}
		scale := maxAbs / 127.0
		scales[b] = scale
		invScale := 1.0 / float64(scale)
		for i := lo; i < hi; i++ {
			codes[i] = int8(max(-128, min(127, int(math.Round(float64(src[i])*invScale)))))
		}
	}
	return codes, scales
}

func dequantKVQ8(codes []int8, scales []float32) []float32 {
	out := make([]float32, len(codes))
	for i, c := range codes {
		out[i] = float32(c) * scales[i/KVQ8BlockSize]
	}
	return out
}

func packKVQ8(k, v []float32) KVQ8Rows {
	kc, ks := refQuantizeKVQ8_0(k)
	vc, vs := refQuantizeKVQ8_0(v)
	return KVQ8Rows{KCodes: kc, KScales: ks, VCodes: vc, VScales: vs}
}

func appendKVQ8(dst, src KVQ8Rows) KVQ8Rows {
	return KVQ8Rows{
		KCodes: append(dst.KCodes, src.KCodes...), KScales: append(dst.KScales, src.KScales...),
		VCodes: append(dst.VCodes, src.VCodes...), VScales: append(dst.VScales, src.VScales...),
	}
}

// kvq8AdversarialPanel builds n values (n a multiple of 32) whose blocks stress the
// codec: elements 0-2 ulps from a .5 tie of x/scale (where an uncorrected f32
// quotient rounds the wrong way), Gaussian blocks, an all-zero block, a constant
// block and negative-max blocks. It reports the indices of EXACT ties, which the
// host's float64 reciprocal may break either way.
func kvq8AdversarialPanel(n int, seed int64) ([]float32, map[int]bool) {
	rng := rand.New(rand.NewSource(seed))
	x, ties := make([]float32, n), map[int]bool{}
	for b := 0; b < n/KVQ8BlockSize; b++ {
		blk := x[b*KVQ8BlockSize : (b+1)*KVQ8BlockSize]
		switch b % 8 {
		case 0, 1, 2, 3:
			amax := float32(0.001 + rng.Float64()*40)
			s := amax / 127.0
			if rng.Intn(2) == 0 {
				amax = -amax
			}
			blk[0] = amax
			for i := 1; i < len(blk); i++ {
				k := rng.Intn(254) - 127
				tie := (float64(k) + 0.5) * float64(s)
				v := float32(tie)
				for step := rng.Intn(5) - 2; step != 0; {
					if step > 0 {
						v, step = math.Nextafter32(v, float32(math.Inf(1))), step-1
					} else {
						v, step = math.Nextafter32(v, float32(math.Inf(-1))), step+1
					}
				}
				if float64(v) == tie {
					ties[b*KVQ8BlockSize+i] = true
				}
				blk[i] = v
			}
		case 4:
			if b%16 == 4 {
				continue // all-zero block: scale 0, codes 0
			}
			for i := range blk {
				blk[i] = -1.25
			}
		default:
			sigma := 0.01 + rng.Float64()*5
			for i := range blk {
				blk[i] = float32(rng.NormFloat64() * sigma)
			}
			if b%8 == 7 {
				blk[rng.Intn(len(blk))] = float32(-8 * sigma)
			}
		}
	}
	return x, ties
}

// assertKVQ8Bytes fails on any scale or code that differs from the host codec,
// except codes at exact .5 ties, whose differences it counts and returns.
func assertKVQ8Bytes(t *testing.T, name string, gotCodes []int8, gotScales []float32, wantCodes []int8, wantScales []float32, ties map[int]bool) int {
	t.Helper()
	if len(gotCodes) != len(wantCodes) || len(gotScales) != len(wantScales) {
		t.Fatalf("%s: packed shape codes=%d/%d scales=%d/%d", name, len(gotCodes), len(wantCodes), len(gotScales), len(wantScales))
	}
	for i := range wantScales {
		if math.Float32bits(gotScales[i]) != math.Float32bits(wantScales[i]) {
			t.Fatalf("%s: scale[%d]=%g want %g (not bit-identical)", name, i, gotScales[i], wantScales[i])
		}
	}
	tieBreaks := 0
	for i := range wantCodes {
		if gotCodes[i] == wantCodes[i] {
			continue
		}
		if !ties[i] {
			t.Fatalf("%s: code[%d]=%d want %d (scale %g)", name, i, gotCodes[i], wantCodes[i], wantScales[i/KVQ8BlockSize])
		}
		tieBreaks++
	}
	return tieBreaks
}

// qwenQ8Layer is one synthetic Qwen3.8 full-attention layer: Q4_K Q+gate, K and V
// projections, per-head Q/K norms, and the geometry the graph entries take.
type qwenQ8Layer struct {
	nH, nKV, hd, rotary, input int
	qgate, kw, vw              *Q4KWeight
	qnorm, knorm               []float32
	scale, eps                 float32
	receipt                    GraphReceipt // the last encode's terminal receipt
}

func newQwenQ8Layer(tb testing.TB, nH, nKV, hd, rotary, input int, seed uint64) *qwenQ8Layer {
	tb.Helper()
	qwidth, kvwidth := nH*hd, nKV*hd
	l := &qwenQ8Layer{nH: nH, nKV: nKV, hd: hd, rotary: rotary, input: input,
		qgate: UploadQ4K(q4kTestRaw(2*qwidth, input, seed+1), 2*qwidth, input),
		kw:    UploadQ4K(q4kTestRaw(kvwidth, input, seed+2), kvwidth, input),
		vw:    UploadQ4K(q4kTestRaw(kvwidth, input, seed+3), kvwidth, input),
		qnorm: make([]float32, hd), knorm: make([]float32, hd),
		scale: float32(1 / math.Sqrt(float64(hd))), eps: 1e-6,
	}
	if l.qgate == nil || l.kw == nil || l.vw == nil {
		tb.Fatal("Q8 attention layer Q4_K upload")
	}
	for i := range l.qnorm {
		l.qnorm[i] = 0.87 + float32(i%7)*0.05
		l.knorm[i] = 0.91 + float32(i%5)*0.04
	}
	return l
}

func (l *qwenQ8Layer) rope(base, rows int) (cosv, sinv []float32) {
	half := l.rotary / 2
	cosv, sinv = make([]float32, rows*half), make([]float32, rows*half)
	for row := 0; row < rows; row++ {
		for dim := 0; dim < half; dim++ {
			angle := float64((base+row+1)*(dim+1)) * 0.003
			cosv[row*half+dim], sinv[row*half+dim] = float32(math.Cos(angle)), float32(math.Sin(angle))
		}
	}
	return cosv, sinv
}

type qwenQ8Mode int

const (
	qwenQ8ModeF32 qwenQ8Mode = iota
	qwenQ8ModeHost
	qwenQ8ModeDevice
)

// qwenQ8Prefix is the attended history an encode sees: the F32 rows for the F32
// entry, the packed rows for the host-packed entry, or a store slice for the device
// entry (whose layer must already hold `base` rows).
type qwenQ8Prefix struct {
	k, v   []float32
	packed KVQ8Rows
	store  *DeviceKVQ8
	layer  int
}

// encode runs one panel of `rows` activations at positions [base, base+rows) and
// returns the terminal readback: 0=q, 1=gate, 2=k, 3=v, 4=out, 5=KRaw, 6=KPost, 7=V
// and, for the packed modes, 8=KCodes, 9=KScales, 10=VCodes, 11=VScales.
func (l *qwenQ8Layer) encode(t *testing.T, x []float32, rows, base int, prefix qwenQ8Prefix, mode qwenQ8Mode) [][]float32 {
	t.Helper()
	g, err := BeginProjectionGraph(x, nil, nil, rows, l.input)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	qg, err := g.EncodeQ4K(l.qgate)
	if err != nil {
		t.Fatal(err)
	}
	k, err := g.EncodeQ4K(l.kw)
	if err != nil {
		t.Fatal(err)
	}
	v, err := g.EncodeQ4K(l.vw)
	if err != nil {
		t.Fatal(err)
	}
	q, gate, err := g.SplitGatedQ(qg, l.nH*l.hd, l.hd)
	if err != nil {
		t.Fatal(err)
	}
	cosv, sinv := l.rope(base, rows)
	var att Qwen35GraphAttentionQ8Result
	switch mode {
	case qwenQ8ModeF32:
		att.Qwen35GraphAttentionResult, err = g.FullAttention(q, k, v, gate, l.qnorm, l.knorm, cosv, sinv, prefix.k, prefix.v, base, l.nH, l.nKV, l.hd, l.rotary, l.scale, l.eps, true, true)
	case qwenQ8ModeHost:
		att, err = g.FullAttentionQ8(q, k, v, gate, l.qnorm, l.knorm, cosv, sinv, prefix.packed, base, l.nH, l.nKV, l.hd, l.rotary, l.scale, l.eps, true, true)
	case qwenQ8ModeDevice:
		att, err = g.FullAttentionDeviceQ8(q, k, v, gate, prefix.store, prefix.layer, l.qnorm, l.knorm, cosv, sinv, base, l.nH, l.nKV, l.hd, l.rotary, l.scale, l.eps, true, true)
	}
	if err != nil {
		t.Fatalf("mode %d attention encode: %v", mode, err)
	}
	results := []*GraphResult{q, gate, k, v, att.Output, att.KRaw, att.KPost, att.V}
	if mode != qwenQ8ModeF32 {
		results = append(results, att.KCodes, att.KScales, att.VCodes, att.VScales)
	}
	out, receipt, err := g.FinishRead(results...)
	if err != nil {
		t.Fatal(err)
	}
	l.receipt = receipt
	return out
}

func q8Slots(out [][]float32) KVQ8Rows {
	return KVQ8Rows{
		KCodes: append([]int8(nil), KVQ8Codes(out[8])...), KScales: append([]float32(nil), out[9]...),
		VCodes: append([]int8(nil), KVQ8Codes(out[10])...), VScales: append([]float32(nil), out[11]...),
	}
}

// qwenQ8AttentionCPU is the float64 oracle for the packed path. Q is normalized and
// rotated exactly as qwenOrderedAttentionCPU does; every attended K/V row, prefix
// and panel alike, is the given (dequantized) row.
func qwenQ8AttentionCPU(q, gate, qnorm, cosv, sinv, allK, allV []float32, rows, base, nH, nKV, hd, rotary int, scale, eps float32) []float32 {
	qpost := qwenOrderedNormalizeCPU(q, qnorm, rows, nH, hd, eps, true)
	half := rotary / 2
	for row := 0; row < rows; row++ {
		for head := 0; head < nH; head++ {
			off := (row*nH + head) * hd
			for dim := 0; dim < half; dim++ {
				x, y := qpost[off+dim], qpost[off+half+dim]
				c, s := cosv[row*half+dim], sinv[row*half+dim]
				qpost[off+dim], qpost[off+half+dim] = x*c-y*s, x*s+y*c
			}
		}
	}
	out := make([]float32, rows*nH*hd)
	for row := 0; row < rows; row++ {
		for head := 0; head < nH; head++ {
			kvHead, upto := head/(nH/nKV), base+row+1
			scores := make([]float64, upto)
			maxScore := math.Inf(-1)
			for token := 0; token < upto; token++ {
				var dot float64
				for dim := 0; dim < hd; dim++ {
					dot += float64(qpost[(row*nH+head)*hd+dim]) * float64(allK[(token*nKV+kvHead)*hd+dim])
				}
				scores[token] = dot * float64(scale)
				maxScore = math.Max(maxScore, scores[token])
			}
			var denom float64
			for token := range scores {
				scores[token] = math.Exp(scores[token] - maxScore)
				denom += scores[token]
			}
			for dim := 0; dim < hd; dim++ {
				var sum float64
				for token := 0; token < upto; token++ {
					sum += scores[token] * float64(allV[(token*nKV+kvHead)*hd+dim])
				}
				index := (row*nH+head)*hd + dim
				out[index] = float32(sum/denom) / (1 + float32(math.Exp(-float64(gate[index]))))
			}
		}
	}
	return out
}

func maxAbs(values []float32) float64 {
	var m float64
	for _, v := range values {
		m = math.Max(m, math.Abs(float64(v)))
	}
	return m
}

func maxAbsDelta(a, b []float32) float64 {
	var m float64
	for i := range a {
		m = math.Max(m, math.Abs(float64(a[i])-float64(b[i])))
	}
	return m
}

func assertFloatBits(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]=%g want %g (not bit-identical)", name, i, got[i], want[i])
		}
	}
}

func assertKVQ8RowsEqual(t *testing.T, name string, got, want KVQ8Rows) {
	t.Helper()
	assertKVQ8Bytes(t, name+" K", got.KCodes, got.KScales, want.KCodes, want.KScales, nil)
	assertKVQ8Bytes(t, name+" V", got.VCodes, got.VScales, want.VCodes, want.VScales, nil)
}

// TestKVQ8QuantizeMatchesHostCodec pins the device quantizer to the host cache's
// codec byte for byte. The graph input doubles as Q, gate, K and V (nH=nKV=1), so V
// is the adversarial panel itself and K is its normalized, rotated image.
func TestKVQ8QuantizeMatchesHostCodec(t *testing.T) {
	requireKVQ8(t)
	const hd, rows, rotary = 256, 64, 64
	x, ties := kvq8AdversarialPanel(rows*hd, 20260925)
	g, err := BeginProjectionGraph(x, nil, nil, rows, hd)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	in, err := g.Input(hd)
	if err != nil {
		t.Fatal(err)
	}
	l := &qwenQ8Layer{hd: hd, rotary: rotary}
	cosv, sinv := l.rope(0, rows)
	norm := make([]float32, hd)
	for i := range norm {
		norm[i] = 0.5 + float32(i%9)*0.1
	}
	att, err := g.FullAttentionQ8(in, in, in, in, norm, norm, cosv, sinv, KVQ8Rows{}, 0, 1, 1, hd, rotary, 1/16.0, 1e-6, false, true)
	if err != nil {
		t.Fatal(err)
	}
	out, _, err := g.FinishRead(att.KPost, att.V, att.KCodes, att.KScales, att.VCodes, att.VScales)
	if err != nil {
		t.Fatal(err)
	}
	assertFloatBits(t, "V passthrough", out[1], x)
	wantVc, wantVs := refQuantizeKVQ8_0(x)
	tieBreaks := assertKVQ8Bytes(t, "adversarial V", KVQ8Codes(out[4]), out[5], wantVc, wantVs, ties)
	wantKc, wantKs := refQuantizeKVQ8_0(out[0])
	assertKVQ8Bytes(t, "post-RoPE K", KVQ8Codes(out[2]), out[3], wantKc, wantKs, nil)
	t.Logf("codec parity over %d elements; %d exact .5 ties, %d broken the other way by the host's float64 reciprocal", len(x), len(ties), tieBreaks)
}

// TestProjectionGraphQwenQ8AttentionParity is the #12981 packed-consumer parity
// witness on decode- and panel-shaped encodes, including hd=256 (the Qwen3.8 head)
// and a split-KV decode past 2048 tokens.
func TestProjectionGraphQwenQ8AttentionParity(t *testing.T) {
	requireKVQ8(t)
	t.Setenv("FAK_QWEN35_ATTN_SPLIT", "1") // pin split-KV decode against an ambient kill switch
	defer ResetQ4K()
	const input, nH, nKV = 256, 4, 2
	for _, tc := range []struct {
		name                   string
		rows, hd, rotary, base int
	}{
		{name: "decode_hd32", rows: 1, hd: 32, rotary: 16, base: 4},
		{name: "panel32_hd64", rows: 32, hd: 64, rotary: 32, base: 8},
		{name: "panel8_hd256", rows: 8, hd: 256, rotary: 64, base: 40},
		{name: "decode_hd256_split", rows: 1, hd: 256, rotary: 64, base: 2100},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l := newQwenQ8Layer(t, nH, nKV, tc.hd, tc.rotary, input, uint64(1298100+tc.hd+tc.base))
			kvwidth := nKV * tc.hd
			rng := rand.New(rand.NewSource(int64(1298110 + tc.base)))
			prefixK, prefixV := make([]float32, tc.base*kvwidth), make([]float32, tc.base*kvwidth)
			for i := range prefixK {
				prefixK[i] = float32(rng.NormFloat64() * 0.6)
				prefixV[i] = float32(rng.NormFloat64() * 0.4)
			}
			packed := packKVQ8(prefixK, prefixV)
			x := q4kTestVector(tc.rows*input, int64(1298120+tc.rows+tc.base))

			f32 := l.encode(t, x, tc.rows, tc.base, qwenQ8Prefix{k: prefixK, v: prefixV}, qwenQ8ModeF32)
			q8 := l.encode(t, x, tc.rows, tc.base, qwenQ8Prefix{packed: packed}, qwenQ8ModeHost)

			// Projections, KRaw and KPost come from the same kernels: bit-identical.
			for i, name := range []string{"q", "gate", "k", "v"} {
				assertFloatBits(t, name, q8[i], f32[i])
			}
			assertFloatBits(t, "KRaw", q8[5], f32[5])
			assertFloatBits(t, "KPost", q8[6], f32[6])
			assertFloatBits(t, "V", q8[7], f32[7])

			// The panel's packed rows are the host codec's bytes of its KPost/V rows.
			panel := q8Slots(q8)
			assertKVQ8RowsEqual(t, "panel", panel, packKVQ8(q8[6], q8[7]))

			// Oracle: float64 attention over the dequantized packed prefix+panel.
			all := appendKVQ8(packed, panel)
			allK, allV := dequantKVQ8(all.KCodes, all.KScales), dequantKVQ8(all.VCodes, all.VScales)
			cosv, sinv := l.rope(tc.base, tc.rows)
			want := qwenQ8AttentionCPU(q8[0], q8[1], l.qnorm, cosv, sinv, allK, allV, tc.rows, tc.base, nH, nKV, tc.hd, tc.rotary, l.scale, l.eps)
			for i := range want {
				if d := math.Abs(float64(q8[4][i] - want[i])); math.IsNaN(float64(q8[4][i])) || d > 8e-4+1e-5*math.Abs(float64(want[i])) {
					t.Fatalf("Q8 attention[%d]=%g oracle %g delta %g", i, q8[4][i], want[i], d)
				}
			}

			// Bounded quantization error against the F32 entry over the unpacked
			// prefix. Q8_0 keeps each element within scale/2 = maxAbs/254 of its
			// block; the attention output must stay inside 2% of its own range.
			delta, span := maxAbsDelta(q8[4], f32[4]), maxAbs(f32[4])
			t.Logf("Q8 vs F32 attention: max|delta|=%.3g max|out|=%.3g rel=%.3g", delta, span, delta/span)
			if delta > 0.02*span {
				t.Fatalf("Q8 attention drifts %.3g from F32 (max|out| %.3g): exceeds the 2%% bound", delta, span)
			}
		})
	}
}

// TestProjectionGraphQwenDeviceKVQ8Walk proves the persistent packed store. Two
// walks run against the host-packed path bit for bit: one from an empty layer-1
// slice (prefill panel, then decode steps), and one on layer 2 seeded with a
// 2040-row host prefix, whose 16-row panel attends over stored rows and whose
// decode steps cross 2048 tokens into split-KV decode. Each store slice must equal
// the rows the host path accumulated, and layer 0 must stay untouched.
func TestProjectionGraphQwenDeviceKVQ8Walk(t *testing.T) {
	requireKVQ8(t)
	t.Setenv("FAK_QWEN35_ATTN_SPLIT", "1") // pin split-KV decode against an ambient kill switch
	defer ResetQ4K()
	const input, nH, nKV, hd, rotary, layers, tokens, seeded = 256, 4, 2, 64, 32, 3, 2080, 2040
	kvwidth := nKV * hd
	l := newQwenQ8Layer(t, nH, nKV, hd, rotary, input, 1298200)
	store := NewDeviceKVQ8(layers, tokens, kvwidth)
	if store == nil {
		t.Fatal("NewDeviceKVQ8 returned nil")
	}
	defer store.Close()
	if got, want := store.ResidentBytes(), int64(2*layers*tokens*(kvwidth+kvwidth/KVQ8BlockSize*4)); got != want {
		t.Fatalf("ResidentBytes=%d want %d", got, want)
	}
	prefix := packKVQ8(q4kTestVector(seeded*kvwidth, 1298205), q4kTestVector(seeded*kvwidth, 1298206))
	if err := store.WriteRows(2, 0, prefix); err != nil {
		t.Fatal(err)
	}
	for _, walk := range []struct {
		layer int
		host  KVQ8Rows
		panel []int
	}{
		{layer: 1, panel: []int{16, 1, 1, 1, 1, 1, 1, 1, 1}},
		{layer: 2, host: prefix, panel: []int{16, 1, 1, 1, 1, 1, 1, 1, 1}},
	} {
		host, base := walk.host, walk.host.Rows(kvwidth)
		for step, rows := range walk.panel {
			x := q4kTestVector(rows*input, int64(1298210+10*walk.layer+step))
			hostOut := l.encode(t, x, rows, base, qwenQ8Prefix{packed: host}, qwenQ8ModeHost)
			devOut := l.encode(t, x, rows, base, qwenQ8Prefix{store: store, layer: walk.layer}, qwenQ8ModeDevice)
			for i, name := range []string{"q", "gate", "k", "v", "out", "KRaw", "KPost", "V", "KCodes", "KScales", "VCodes", "VScales"} {
				assertFloatBits(t, fmt.Sprintf("layer %d step %d %s", walk.layer, step, name), devOut[i], hostOut[i])
			}
			host = appendKVQ8(host, q8Slots(hostOut))
			base += rows
		}
		got, err := store.ReadRows(walk.layer, 0, base)
		if err != nil {
			t.Fatal(err)
		}
		assertKVQ8RowsEqual(t, fmt.Sprintf("layer %d store vs host", walk.layer), got, host)
	}
	untouched, err := store.ReadRows(0, 0, tokens)
	if err != nil {
		t.Fatal(err)
	}
	for i, c := range untouched.KCodes {
		if c != 0 || untouched.VCodes[i] != 0 {
			t.Fatalf("layer 0 code[%d] written by a layer-1/2 walk", i)
		}
	}
	const layer = 1

	// Seeding: rows written from the host read back verbatim, and the walk's
	// capacity and geometry are enforced.
	seed := packKVQ8(q4kTestVector(3*kvwidth, 1298230), q4kTestVector(3*kvwidth, 1298231))
	if err := store.WriteRows(0, tokens-3, seed); err != nil {
		t.Fatal(err)
	}
	back, err := store.ReadRows(0, tokens-3, 3)
	if err != nil {
		t.Fatal(err)
	}
	assertKVQ8RowsEqual(t, "seed round trip", back, seed)
	if err := store.WriteRows(0, tokens-2, seed); err == nil {
		t.Fatal("WriteRows past capacity succeeded")
	}
	if _, err := store.ReadRows(layers, 0, 1); err == nil {
		t.Fatal("ReadRows of a missing layer succeeded")
	}
	g, err := BeginProjectionGraph(q4kTestVector(input, 1298240), nil, nil, 1, input)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Free()
	qg, _ := g.EncodeQ4K(l.qgate)
	k, _ := g.EncodeQ4K(l.kw)
	v, _ := g.EncodeQ4K(l.vw)
	q, gate, err := g.SplitGatedQ(qg, nH*hd, hd)
	if err != nil {
		t.Fatal(err)
	}
	cosv, sinv := l.rope(tokens, 1)
	if _, err := g.FullAttentionDeviceQ8(q, k, v, gate, store, layer, l.qnorm, l.knorm, cosv, sinv, tokens, nH, nKV, hd, rotary, l.scale, l.eps, true, true); err == nil {
		t.Fatal("FullAttentionDeviceQ8 appended past the store capacity")
	}
}

// TestProjectionGraphQwenQ8GreedyTokensMatchF32 runs a synthetic single-layer
// language model (token embedding -> Qwen3.8 full attention -> vocabulary head)
// through a 2064-token prefill in 128-row panels and 24 greedy decode steps past
// the split-KV threshold, once over the F32 cache and once over the packed Q8 cache
// the host accumulates from the device's own packed rows. The greedy tokens must be
// identical.
func TestProjectionGraphQwenQ8GreedyTokensMatchF32(t *testing.T) {
	requireKVQ8(t)
	t.Setenv("FAK_QWEN35_ATTN_SPLIT", "1") // pin split-KV decode against an ambient kill switch
	defer ResetQ4K()
	const input, nH, nKV, hd, rotary, vocab, prompt, decode, panel = 256, 4, 2, 256, 64, 64, 2064, 24, 128
	qwidth := nH * hd
	l := newQwenQ8Layer(t, nH, nKV, hd, rotary, input, 1298300)
	embed := q4kTestVector(vocab*input, 1298301)
	head := q4kTestVector(vocab*qwidth, 1298302)
	rng := rand.New(rand.NewSource(1298303))
	tokens := make([]int, prompt)
	for i := range tokens {
		tokens[i] = rng.Intn(vocab)
	}
	logitsOf := func(att []float32) []float64 {
		logits := make([]float64, vocab)
		for tok := range logits {
			for i, a := range att {
				logits[tok] += float64(head[tok*qwidth+i]) * float64(a)
			}
		}
		return logits
	}
	argmax := func(logits []float64) (int, float64) {
		best, second := 0, math.Inf(-1)
		for i, v := range logits {
			if v > logits[best] {
				best, second = i, logits[best]
			} else if i != best && v > second {
				second = v
			}
		}
		return best, logits[best] - second
	}
	embedRows := func(ids []int) []float32 {
		x := make([]float32, 0, len(ids)*input)
		for _, id := range ids {
			x = append(x, embed[id*input:(id+1)*input]...)
		}
		return x
	}

	type run struct {
		k, v   []float32
		packed KVQ8Rows
		out    []int
		logits [][]float64
	}
	generate := func(mode qwenQ8Mode) *run {
		r := &run{}
		step := func(ids []int, base int) []float32 {
			out := l.encode(t, embedRows(ids), len(ids), base, qwenQ8Prefix{k: r.k, v: r.v, packed: r.packed}, mode)
			if mode == qwenQ8ModeF32 {
				r.k, r.v = append(r.k, out[6]...), append(r.v, out[7]...)
			} else {
				r.packed = appendKVQ8(r.packed, q8Slots(out))
			}
			return out[4][(len(ids)-1)*qwidth:]
		}
		var last []float32
		for base := 0; base < prompt; base += panel {
			last = step(tokens[base:min(base+panel, prompt)], base)
		}
		for i := 0; i < decode; i++ {
			logits := logitsOf(last)
			next, _ := argmax(logits)
			r.out, r.logits = append(r.out, next), append(r.logits, logits)
			last = step([]int{next}, prompt+i)
		}
		return r
	}
	start := time.Now()
	f32, q8 := generate(qwenQ8ModeF32), generate(qwenQ8ModeHost)
	// Per step, a Q8 logit drift below half the F32 top-2 margin GUARANTEES the same
	// argmax; report how much of that guarantee the fixture keeps, and how large the
	// drift is against the logit spread, so a pass is read with its margin.
	minRatio, maxRel := math.Inf(1), 0.0
	unguaranteed := 0
	for i := range f32.logits {
		_, margin := argmax(f32.logits[i])
		var drift, lo, hi float64
		lo, hi = math.Inf(1), math.Inf(-1)
		for j, v := range f32.logits[i] {
			drift = math.Max(drift, math.Abs(v-q8.logits[i][j]))
			lo, hi = math.Min(lo, v), math.Max(hi, v)
		}
		ratio := margin / (2 * drift)
		if ratio <= 1 {
			unguaranteed++
		}
		minRatio, maxRel = math.Min(minRatio, ratio), math.Max(maxRel, drift/(hi-lo))
	}
	t.Logf("greedy %d tokens after a %d-token prompt in %v: max logit drift %.3g%% of the logit range; min margin/(2*drift)=%.3g, %d/%d steps below the guaranteed-identity margin", decode, prompt, time.Since(start).Round(time.Millisecond), 100*maxRel, minRatio, unguaranteed, decode)
	t.Logf("F32 tokens %v", f32.out)
	for i := range f32.out {
		if f32.out[i] != q8.out[i] {
			t.Fatalf("greedy token %d: Q8 %d, F32 %d (F32 %v, Q8 %v)", i, q8.out[i], f32.out[i], f32.out, q8.out)
		}
	}
	if got, want := len(q8.packed.KCodes), (prompt+decode)*nKV*hd; got != want {
		t.Fatalf("packed cache holds %d codes, want %d", got, want)
	}
	f32Bytes := int64(len(f32.k)+len(f32.v)) * 4
	q8Bytes := int64(len(q8.packed.KCodes)+len(q8.packed.VCodes)) + int64(len(q8.packed.KScales)+len(q8.packed.VScales))*4
	t.Logf("attended K+V bytes at %d tokens: F32 %d, Q8 %d (%.3fx)", prompt+decode, f32Bytes, q8Bytes, float64(f32Bytes)/float64(q8Bytes))
}

// TestKVQ8ResidencyArithmetic pins KVQ8RowBytes and KVQ8Rows.Rows to the issue's
// 20k Qwen3.8 arithmetic (ResidentBytes is checked against a live store by the walk
// test): 16 full-attention layers, 20480 tokens,
// kvWidth 1024. K+V drop from 2.50 GiB (F32) to 0.70 GiB packed; with the f32 KRaw
// row the host keeps, the token KV is 1.953 GiB against 3.75 GiB.
func TestKVQ8ResidencyArithmetic(t *testing.T) {
	const layers, tokens, kvWidth = 16, 20480, 1024
	if got := KVQ8RowBytes(kvWidth); got != 1152 {
		t.Fatalf("KVQ8RowBytes(1024)=%d want 1152", got)
	}
	packedKV := int64(2 * layers * tokens * KVQ8RowBytes(kvWidth))
	kraw := int64(layers * tokens * kvWidth * 4)
	if packedKV+kraw != 2_097_152_000 {
		t.Fatalf("mixed Q8 token KV=%d B want 2097152000", packedKV+kraw)
	}
	if f32 := int64(3 * layers * tokens * kvWidth * 4); f32 != 4_026_531_840 {
		t.Fatalf("F32 token KV=%d B want 4026531840", f32)
	}
	if n := (KVQ8Rows{KCodes: make([]int8, 2*kvWidth), VCodes: make([]int8, 2*kvWidth), KScales: make([]float32, 64), VScales: make([]float32, 64)}).Rows(kvWidth); n != 2 {
		t.Fatalf("Rows=%d want 2", n)
	}
	if n := (KVQ8Rows{KCodes: make([]int8, kvWidth), VCodes: make([]int8, kvWidth), KScales: make([]float32, 31), VScales: make([]float32, 32)}).Rows(kvWidth); n != -1 {
		t.Fatalf("Rows of a torn run=%d want -1", n)
	}
}
