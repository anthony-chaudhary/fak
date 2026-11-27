//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

func ink2FP8CosineAndMaxErr(want, got []float32) (cos, maxErr float64) {
	var dot, nw, ng float64
	for i := range want {
		w, g := float64(want[i]), float64(got[i])
		if math.IsNaN(g) || math.IsInf(g, 0) {
			return -1, math.Inf(1)
		}
		dot += w * g
		nw += w * w
		ng += g * g
		if d := math.Abs(g - w); d > maxErr {
			maxErr = d
		}
	}
	if nw == 0 && ng == 0 {
		return 1, maxErr
	}
	if nw == 0 || ng == 0 {
		return 0, maxErr
	}
	return dot / (math.Sqrt(nw) * math.Sqrt(ng)), maxErr
}

// fak-test:runtime fast est=5s lane=default
func TestCUDAInk2FP8KVDecodeMatchesFP32(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	var _ Inkling2FP8KVDevice = cb

	p := Ink2AttnParams{
		Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16,
		QKClip: 3, RelClip: 2, Eps: 1e-6,
		Global: true, Alpha: 0.1, Floor: 64,
		Scale: float32(1 / math.Sqrt(128)),
	}
	kvDim := p.KVHeads * p.HeadDim
	const (
		capTok  = 256
		extent  = 128
		numToks = 48
	)
	r := rand.New(rand.NewSource(20261126))
	alloc := func(n int) Tensor {
		tt, err := cb.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { cb.Free(tt) })
		return tt
	}
	write := func(tt Tensor, v []float32) {
		if err := cb.Ink2Write(Ink2At(tt, 0), v); err != nil {
			t.Fatal(err)
		}
	}
	randVec := func(n int, sd, mean float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(mean + sd*r.NormFloat64())
		}
		return v
	}

	qclip, rclip := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.RelLen)
	relBias := alloc(extent * p.RelLen)
	write(qclip, randVec(p.Heads*p.HeadDim, 0.1, 1))
	write(rclip, randVec(p.Heads*p.RelLen, 0.1, 1))
	write(relBias, randVec(extent*p.RelLen, 0.1, 0))

	kcF32, vcF32 := alloc(capTok*kvDim), alloc(capTok*kvDim)
	kcFP8 := alloc(Ink2FP8KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	vcFP8 := alloc(Ink2FP8KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	ksFP8 := alloc(Ink2FP8KVScaleFloats(capTok, p.KVHeads))
	vsFP8 := alloc(Ink2FP8KVScaleFloats(capTok, p.KVHeads))
	ovf := alloc(1)

	qBuf, rBuf := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.RelLen)
	kBuf, vBuf := alloc(kvDim), alloc(kvDim)
	sinksBuf := alloc(p.Heads)
	scoresBuf := alloc(2 * p.Heads * capTok)
	outF32, outFP8 := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.HeadDim)
	meta := alloc(1)

	var lastSinks []float32
	for pos := 0; pos < numToks; pos++ {
		qv := randVec(p.Heads*p.HeadDim, 1.0, 0)
		rv := randVec(p.Heads*p.RelLen, 1.0, 0)
		kv := randVec(kvDim, 0.5, 0)
		vv := randVec(kvDim, 1.0, 0)
		sv := randVec(p.Heads, 0.5, 0)
		write(meta, []float32{math.Float32frombits(uint32(pos))})
		write(kBuf, kv)
		write(vBuf, vv)

		write(qBuf, qv)
		write(rBuf, rv)
		if err := cb.Ink2QKVPost(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
			Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcF32, 0), Ink2At(vcF32, 0), Ink2At(meta, 0), p, capTok, nil); err != nil {
			t.Fatalf("pos %d Ink2QKVPost: %v", pos, err)
		}

		write(qBuf, qv)
		write(rBuf, rv)
		if err := cb.Ink2QKVPostFP8(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
			Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcFP8, 0), Ink2At(vcFP8, 0), Ink2At(ksFP8, 0), Ink2At(vsFP8, 0),
			Ink2At(ovf, 0), Ink2At(meta, 0), p, capTok, nil); err != nil {
			t.Fatalf("pos %d Ink2QKVPostFP8: %v", pos, err)
		}
		if pos == numToks-1 {
			lastSinks = sv
		}
	}
	// Re-read the post-clipped Q and R of the final token for attend comparison.
	lastQ, err := cb.Ink2Read(Ink2At(qBuf, 0), p.Heads*p.HeadDim)
	if err != nil {
		t.Fatal(err)
	}
	lastR, err := cb.Ink2Read(Ink2At(rBuf, 0), p.Heads*p.RelLen)
	if err != nil {
		t.Fatal(err)
	}
	write(sinksBuf, lastSinks)

	flags, err := cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] != 0 {
		t.Fatalf("unexpected FP8 overflow flag on finite inputs: %d", flags[0])
	}

	modes := []struct {
		name  string
		split bool
		exact bool
	}{
		{"split-exact", true, true},
		{"split-tree", true, false},
		{"reference", false, true},
	}
	prevSplit := ink2SelectAttendSplit(true)
	prevExact := ink2SelectAttendExact(true)
	t.Cleanup(func() {
		ink2SelectAttendSplit(prevSplit)
		ink2SelectAttendExact(prevExact)
	})

	for _, m := range modes {
		ink2SelectAttendSplit(m.split)
		ink2SelectAttendExact(m.exact)

		write(qBuf, lastQ)
		write(rBuf, lastR)
		if err := cb.Ink2Attend(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(sinksBuf, 0), Ink2At(relBias, 0),
			Ink2At(kcF32, 0), Ink2At(vcF32, 0), Ink2At(scoresBuf, 0), Ink2At(outF32, 0), Ink2At(meta, 0),
			p, extent, 0, capTok); err != nil {
			t.Fatalf("%s Ink2Attend: %v", m.name, err)
		}
		want, err := cb.Ink2Read(Ink2At(outF32, 0), p.Heads*p.HeadDim)
		if err != nil {
			t.Fatal(err)
		}

		write(qBuf, lastQ)
		write(rBuf, lastR)
		if err := cb.Ink2AttendFP8(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(sinksBuf, 0), Ink2At(relBias, 0),
			Ink2At(kcFP8, 0), Ink2At(vcFP8, 0), Ink2At(ksFP8, 0), Ink2At(vsFP8, 0), Ink2At(scoresBuf, 0),
			Ink2At(outFP8, 0), Ink2At(meta, 0), p, extent, 0, capTok); err != nil {
			t.Fatalf("%s Ink2AttendFP8: %v", m.name, err)
		}
		got, err := cb.Ink2Read(Ink2At(outFP8, 0), p.Heads*p.HeadDim)
		if err != nil {
			t.Fatal(err)
		}

		cos, maxErr := ink2FP8CosineAndMaxErr(want, got)
		t.Logf("decode %s: cosine=%.6f max_abs_err=%.6f", m.name, cos, maxErr)
		if cos < 0.999 || maxErr > 0.05 {
			t.Fatalf("decode %s accuracy out of bounds: cosine=%.6f max_abs_err=%.6f", m.name, cos, maxErr)
		}
	}
}

// fak-test:runtime fast est=5s lane=default
func TestCUDAInk2FP8KVPrefillMatchesFP32(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	if err := cb.Ink2PFP8KVReady(); err != nil {
		t.Fatalf("Ink2PFP8KVReady: %v", err)
	}

	p := Ink2AttnParams{
		Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16,
		QKClip: 3, RelClip: 2, Eps: 1e-6,
		Global: true, Alpha: 0.1, Floor: 64,
		Scale: float32(1 / math.Sqrt(128)),
	}
	kvDim := p.KVHeads * p.HeadDim
	const (
		capTok = 512
		extent = 256
	)
	r := rand.New(rand.NewSource(20261127))
	alloc := func(n int) Tensor {
		tt, err := cb.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { cb.Free(tt) })
		return tt
	}
	write := func(tt Tensor, v []float32) {
		if err := cb.Ink2Write(Ink2At(tt, 0), v); err != nil {
			t.Fatal(err)
		}
	}
	randVec := func(n int, sd, mean float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(mean + sd*r.NormFloat64())
		}
		return v
	}

	qclip, rclip := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.RelLen)
	relBias := alloc(extent * p.RelLen)
	write(qclip, randVec(p.Heads*p.HeadDim, 0.1, 1))
	write(rclip, randVec(p.Heads*p.RelLen, 0.1, 1))
	write(relBias, randVec(extent*p.RelLen, 0.1, 0))

	kcF32, vcF32 := alloc(capTok*kvDim), alloc(capTok*kvDim)
	kcFP8 := alloc(Ink2FP8KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	vcFP8 := alloc(Ink2FP8KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	ksFP8 := alloc(Ink2FP8KVScaleFloats(capTok, p.KVHeads))
	vsFP8 := alloc(Ink2FP8KVScaleFloats(capTok, p.KVHeads))
	ovf := alloc(1)

	chunks := []struct{ pos0, T int }{{0, 128}, {128, 64}}
	for ci, ch := range chunks {
		qv := randVec(ch.T*p.Heads*p.HeadDim, 1.0, 0)
		rv := randVec(ch.T*p.Heads*p.RelLen, 1.0, 0)
		kv := randVec(ch.T*kvDim, 0.5, 0)
		vv := randVec(ch.T*kvDim, 1.0, 0)
		sv := randVec(ch.T*p.Heads, 0.5, 0)

		qF32, rF32 := alloc(len(qv)), alloc(len(rv))
		qFP8, rFP8 := alloc(len(qv)), alloc(len(rv))
		kBuf, vBuf, sBuf := alloc(len(kv)), alloc(len(vv)), alloc(len(sv))
		oF32, oFP8 := alloc(len(qv)), alloc(len(qv))
		write(qF32, qv)
		write(rF32, rv)
		write(qFP8, qv)
		write(rFP8, rv)
		write(kBuf, kv)
		write(vBuf, vv)
		write(sBuf, sv)

		if err := cb.Ink2PQKVPost(Ink2At(qF32, 0), Ink2At(rF32, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
			Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcF32, 0), Ink2At(vcF32, 0), p, capTok, ch.pos0, ch.T); err != nil {
			t.Fatalf("chunk %d Ink2PQKVPost: %v", ci, err)
		}
		if err := cb.Ink2PAttend(Ink2At(qF32, 0), Ink2At(rF32, 0), Ink2At(sBuf, 0), Ink2At(relBias, 0),
			Ink2At(kcF32, 0), Ink2At(vcF32, 0), Ink2At(oF32, 0), p, extent, 0, capTok, ch.pos0, ch.T); err != nil {
			t.Fatalf("chunk %d Ink2PAttend: %v", ci, err)
		}

		if err := cb.Ink2PQKVPostFP8(Ink2At(qFP8, 0), Ink2At(rFP8, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
			Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcFP8, 0), Ink2At(vcFP8, 0), Ink2At(ksFP8, 0), Ink2At(vsFP8, 0),
			Ink2At(ovf, 0), p, capTok, ch.pos0, ch.T); err != nil {
			t.Fatalf("chunk %d Ink2PQKVPostFP8: %v", ci, err)
		}
		if err := cb.Ink2PAttendFP8(Ink2At(qFP8, 0), Ink2At(rFP8, 0), Ink2At(sBuf, 0), Ink2At(relBias, 0),
			Ink2At(kcFP8, 0), Ink2At(vcFP8, 0), Ink2At(ksFP8, 0), Ink2At(vsFP8, 0), Ink2At(oFP8, 0),
			p, extent, 0, capTok, ch.pos0, ch.T); err != nil {
			t.Fatalf("chunk %d Ink2PAttendFP8: %v", ci, err)
		}

		want, err := cb.Ink2Read(Ink2At(oF32, 0), len(qv))
		if err != nil {
			t.Fatal(err)
		}
		got, err := cb.Ink2Read(Ink2At(oFP8, 0), len(qv))
		if err != nil {
			t.Fatal(err)
		}
		cos, maxErr := ink2FP8CosineAndMaxErr(want, got)
		t.Logf("prefill chunk %d (pos0=%d T=%d): cosine=%.6f max_abs_err=%.6f", ci, ch.pos0, ch.T, cos, maxErr)
		// E4M3FN has 3 mantissa bits (up to 6.25% rel err per element); at pos=0 (single-key
		// attention) with RMS-normalized V (|V| up to ~2.5), single-element abs err can reach ~0.10
		// while cosine stays > 0.9995 across all 393k elements.
		if cos < 0.999 || maxErr > 0.12 {
			t.Fatalf("prefill chunk %d accuracy out of bounds: cosine=%.6f max_abs_err=%.6f", ci, cos, maxErr)
		}
	}

	flags, err := cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] != 0 {
		t.Fatalf("unexpected FP8 overflow flag during prefill: %d", flags[0])
	}

	// Shape off the tensor-core tile must be refused.
	bad := p
	bad.HeadDim = 64
	tt := alloc(64)
	if err := cb.Ink2PAttendFP8(Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0),
		Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0),
		bad, 16, 0, 16, 0, 1); err == nil {
		t.Fatal("fp8 prefill attend accepted unsupported headDim=64")
	}
}

// fak-test:runtime fast est=5s lane=default
func TestCUDAInk2FP8KVZeroAndOverflow(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	p := Ink2AttnParams{
		Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16,
		QKClip: 3, RelClip: 2, Eps: 1e-6,
		Global: true, Alpha: 0.1, Floor: 64,
		Scale: float32(1 / math.Sqrt(128)),
	}
	kvDim := p.KVHeads * p.HeadDim
	const capTok = 16

	alloc := func(n int) Tensor {
		tt, err := cb.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { cb.Free(tt) })
		return tt
	}
	write := func(tt Tensor, v []float32) {
		if err := cb.Ink2Write(Ink2At(tt, 0), v); err != nil {
			t.Fatal(err)
		}
	}

	qBuf, rBuf := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.RelLen)
	kBuf, vBuf := alloc(kvDim), alloc(kvDim)
	qclip, rclip := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.RelLen)
	kcFP8 := alloc(Ink2FP8KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	vcFP8 := alloc(Ink2FP8KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	ksFP8 := alloc(Ink2FP8KVScaleFloats(capTok, p.KVHeads))
	vsFP8 := alloc(Ink2FP8KVScaleFloats(capTok, p.KVHeads))
	ovf := alloc(1)
	meta := alloc(1)

	onesQ := make([]float32, p.Heads*p.HeadDim)
	for i := range onesQ {
		onesQ[i] = 1
	}
	onesR := make([]float32, p.Heads*p.RelLen)
	for i := range onesR {
		onesR[i] = 1
	}
	write(qclip, onesQ)
	write(rclip, onesR)
	write(meta, []float32{math.Float32frombits(0)})

	// 1. All-zero K and V must produce scale=0 and overflow=0.
	if err := cb.Ink2QKVPostFP8(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
		Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcFP8, 0), Ink2At(vcFP8, 0), Ink2At(ksFP8, 0), Ink2At(vsFP8, 0),
		Ink2At(ovf, 0), Ink2At(meta, 0), p, capTok, nil); err != nil {
		t.Fatalf("zero vector Ink2QKVPostFP8: %v", err)
	}
	flags, err := cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] != 0 {
		t.Fatalf("zero vector triggered overflow flag: %d", flags[0])
	}
	ks, err := cb.Ink2Read(Ink2At(ksFP8, 0), p.KVHeads)
	if err != nil {
		t.Fatal(err)
	}
	vs, err := cb.Ink2Read(Ink2At(vsFP8, 0), p.KVHeads)
	if err != nil {
		t.Fatal(err)
	}
	for h := 0; h < p.KVHeads; h++ {
		if ks[h] != 0 || vs[h] != 0 {
			t.Fatalf("zero vector head %d non-zero scale: ks=%v vs=%v", h, ks[h], vs[h])
		}
	}

	// 2. Non-finite K element (+Inf) in decode must set overflow != 0 and zero that row's scale.
	badK := make([]float32, kvDim)
	badK[3*p.HeadDim+7] = float32(math.Inf(1))
	write(kBuf, badK)
	if err := cb.Ink2QKVPostFP8(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
		Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcFP8, 0), Ink2At(vcFP8, 0), Ink2At(ksFP8, 0), Ink2At(vsFP8, 0),
		Ink2At(ovf, 0), Ink2At(meta, 0), p, capTok, nil); err != nil {
		t.Fatalf("inf K Ink2QKVPostFP8: %v", err)
	}
	flags, err = cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] == 0 {
		t.Fatal("non-finite K element did not set overflow flag")
	}

	// 3. Non-finite V element (NaN) in prefill must set overflow != 0.
	ovf2 := alloc(1)
	badV := make([]float32, kvDim)
	badV[1*p.HeadDim+5] = float32(math.NaN())
	write(kBuf, make([]float32, kvDim))
	write(vBuf, badV)
	if err := cb.Ink2PQKVPostFP8(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
		Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcFP8, 0), Ink2At(vcFP8, 0), Ink2At(ksFP8, 0), Ink2At(vsFP8, 0),
		Ink2At(ovf2, 0), p, capTok, 0, 1); err != nil {
		t.Fatalf("nan V Ink2PQKVPostFP8: %v", err)
	}
	flags, err = cb.Ink2ReadInts(Ink2At(ovf2, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] == 0 {
		t.Fatal("non-finite V element in prefill did not set overflow flag")
	}
}
