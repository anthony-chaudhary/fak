//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

func ink2NVFP4DecodeE2M1(nib uint8) float32 {
	lut := [8]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}
	v := lut[nib&0x7]
	if nib&0x8 != 0 {
		return -v
	}
	return v
}

func ink2NVFP4DecodeE4M3FN(b uint8) float32 {
	sign := uint32(b&0x80) << 24
	e := uint32((b >> 3) & 0xF)
	m := uint32(b & 0x7)
	if e == 0 {
		v := float32(m) * (1.0 / 512.0)
		if sign != 0 {
			return -v
		}
		return v
	}
	return math.Float32frombits(sign | ((e + 120) << 23) | (m << 20))
}

func ink2NVFP4DequantToF32(codesWords, scaleWords []float32, capTok, kvHeads, headDim int) []float32 {
	codeBytes := make([]uint8, len(codesWords)*4)
	for i, f := range codesWords {
		u := math.Float32bits(f)
		codeBytes[4*i+0] = uint8(u)
		codeBytes[4*i+1] = uint8(u >> 8)
		codeBytes[4*i+2] = uint8(u >> 16)
		codeBytes[4*i+3] = uint8(u >> 24)
	}
	scaleBytes := make([]uint8, len(scaleWords)*4)
	for i, f := range scaleWords {
		u := math.Float32bits(f)
		scaleBytes[4*i+0] = uint8(u)
		scaleBytes[4*i+1] = uint8(u >> 8)
		scaleBytes[4*i+2] = uint8(u >> 16)
		scaleBytes[4*i+3] = uint8(u >> 24)
	}
	out := make([]float32, capTok*kvHeads*headDim)
	hBytes := headDim / 2
	hBlks := headDim / 16
	for p := 0; p < capTok; p++ {
		for h := 0; h < kvHeads; h++ {
			for d := 0; d < headDim; d++ {
				cb := codeBytes[(p*kvHeads+h)*hBytes+d/2]
				nib := cb & 0x0f
				if d&1 != 0 {
					nib = cb >> 4
				}
				sb := scaleBytes[(p*kvHeads+h)*hBlks+d/16]
				out[(p*kvHeads+h)*headDim+d] = ink2NVFP4DecodeE2M1(nib) * ink2NVFP4DecodeE4M3FN(sb)
			}
		}
	}
	return out
}

// fak-test:runtime fast est=5s lane=default
func TestCUDAInk2NVFP4KVDecodeMatchesRef(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	var _ Inkling2NVFP4KVDevice = cb

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
	r := rand.New(rand.NewSource(20261128))
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
	kcDeq, vcDeq := alloc(capTok*kvDim), alloc(capTok*kvDim)
	kcNV4 := alloc(Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	vcNV4 := alloc(Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	ksNV4 := alloc(Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
	vsNV4 := alloc(Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
	ovf := alloc(1)

	qBuf, rBuf := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.RelLen)
	kBuf, vBuf := alloc(kvDim), alloc(kvDim)
	sinksBuf := alloc(p.Heads)
	scoresBuf := alloc(2 * p.Heads * capTok)
	outF32, outDeq, outNV4 := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.HeadDim)
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
		if err := cb.Ink2QKVPostNVFP4(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
			Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcNV4, 0), Ink2At(vcNV4, 0), Ink2At(ksNV4, 0), Ink2At(vsNV4, 0),
			Ink2At(ovf, 0), Ink2At(meta, 0), p, capTok, nil); err != nil {
			t.Fatalf("pos %d Ink2QKVPostNVFP4: %v", pos, err)
		}
		if pos == numToks-1 {
			lastSinks = sv
		}
	}
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
		t.Fatalf("unexpected NVFP4 overflow flag on finite inputs: %d", flags[0])
	}

	// Upload host-dequantized NVFP4 caches into kcDeq/vcDeq to verify exact kernel decode parity.
	kcWords, err := cb.Ink2Read(Ink2At(kcNV4, 0), Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	if err != nil {
		t.Fatal(err)
	}
	vcWords, err := cb.Ink2Read(Ink2At(vcNV4, 0), Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	if err != nil {
		t.Fatal(err)
	}
	ksWords, err := cb.Ink2Read(Ink2At(ksNV4, 0), Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
	if err != nil {
		t.Fatal(err)
	}
	vsWords, err := cb.Ink2Read(Ink2At(vsNV4, 0), Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
	if err != nil {
		t.Fatal(err)
	}
	write(kcDeq, ink2NVFP4DequantToF32(kcWords, ksWords, capTok, p.KVHeads, p.HeadDim))
	write(vcDeq, ink2NVFP4DequantToF32(vcWords, vsWords, capTok, p.KVHeads, p.HeadDim))

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
			t.Fatalf("%s Ink2Attend(fp32): %v", m.name, err)
		}
		wantF32, err := cb.Ink2Read(Ink2At(outF32, 0), p.Heads*p.HeadDim)
		if err != nil {
			t.Fatal(err)
		}

		write(qBuf, lastQ)
		write(rBuf, lastR)
		if err := cb.Ink2Attend(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(sinksBuf, 0), Ink2At(relBias, 0),
			Ink2At(kcDeq, 0), Ink2At(vcDeq, 0), Ink2At(scoresBuf, 0), Ink2At(outDeq, 0), Ink2At(meta, 0),
			p, extent, 0, capTok); err != nil {
			t.Fatalf("%s Ink2Attend(deq): %v", m.name, err)
		}
		wantDeq, err := cb.Ink2Read(Ink2At(outDeq, 0), p.Heads*p.HeadDim)
		if err != nil {
			t.Fatal(err)
		}

		write(qBuf, lastQ)
		write(rBuf, lastR)
		if err := cb.Ink2AttendNVFP4(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(sinksBuf, 0), Ink2At(relBias, 0),
			Ink2At(kcNV4, 0), Ink2At(vcNV4, 0), Ink2At(ksNV4, 0), Ink2At(vsNV4, 0), Ink2At(scoresBuf, 0),
			Ink2At(outNV4, 0), Ink2At(meta, 0), p, extent, 0, capTok); err != nil {
			t.Fatalf("%s Ink2AttendNVFP4: %v", m.name, err)
		}
		got, err := cb.Ink2Read(Ink2At(outNV4, 0), p.Heads*p.HeadDim)
		if err != nil {
			t.Fatal(err)
		}

		cosDeq, maxErrDeq := ink2FP8CosineAndMaxErr(wantDeq, got)
		cosF32, maxErrF32 := ink2FP8CosineAndMaxErr(wantF32, got)
		t.Logf("decode %s: vs_scalar_deq(cos=%.7f max_err=%.7f) vs_fp32(cos=%.6f max_err=%.6f)",
			m.name, cosDeq, maxErrDeq, cosF32, maxErrF32)
		if cosDeq < 0.999999 || maxErrDeq > 1e-4 {
			t.Fatalf("decode %s diverged from scalar dequant reference: cos=%.7f max_err=%.7f", m.name, cosDeq, maxErrDeq)
		}
		if cosF32 < 0.99 || maxErrF32 > 0.20 {
			t.Fatalf("decode %s accuracy vs fp32 out of bounds: cos=%.6f max_err=%.6f", m.name, cosF32, maxErrF32)
		}
	}
}

// fak-test:runtime fast est=5s lane=default
func TestCUDAInk2NVFP4KVPrefillMatchesFP32(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	if err := cb.Ink2PNVFP4KVReady(); err != nil {
		t.Fatalf("Ink2PNVFP4KVReady: %v", err)
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
	r := rand.New(rand.NewSource(20261129))
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
	kcNV4 := alloc(Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	vcNV4 := alloc(Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	ksNV4 := alloc(Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
	vsNV4 := alloc(Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
	ovf := alloc(1)

	chunks := []struct{ pos0, T int }{{0, 128}, {128, 64}}
	for ci, ch := range chunks {
		qv := randVec(ch.T*p.Heads*p.HeadDim, 1.0, 0)
		rv := randVec(ch.T*p.Heads*p.RelLen, 1.0, 0)
		kv := randVec(ch.T*kvDim, 0.5, 0)
		vv := randVec(ch.T*kvDim, 1.0, 0)
		sv := randVec(ch.T*p.Heads, 0.5, 0)

		qF32, rF32 := alloc(len(qv)), alloc(len(rv))
		qNV4, rNV4 := alloc(len(qv)), alloc(len(rv))
		kBuf, vBuf, sBuf := alloc(len(kv)), alloc(len(vv)), alloc(len(sv))
		oF32, oNV4 := alloc(len(qv)), alloc(len(qv))
		write(qF32, qv)
		write(rF32, rv)
		write(qNV4, qv)
		write(rNV4, rv)
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

		if err := cb.Ink2PQKVPostNVFP4(Ink2At(qNV4, 0), Ink2At(rNV4, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
			Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcNV4, 0), Ink2At(vcNV4, 0), Ink2At(ksNV4, 0), Ink2At(vsNV4, 0),
			Ink2At(ovf, 0), p, capTok, ch.pos0, ch.T); err != nil {
			t.Fatalf("chunk %d Ink2PQKVPostNVFP4: %v", ci, err)
		}
		if err := cb.Ink2PAttendNVFP4(Ink2At(qNV4, 0), Ink2At(rNV4, 0), Ink2At(sBuf, 0), Ink2At(relBias, 0),
			Ink2At(kcNV4, 0), Ink2At(vcNV4, 0), Ink2At(ksNV4, 0), Ink2At(vsNV4, 0), Ink2At(oNV4, 0),
			p, extent, 0, capTok, ch.pos0, ch.T); err != nil {
			t.Fatalf("chunk %d Ink2PAttendNVFP4: %v", ci, err)
		}

		want, err := cb.Ink2Read(Ink2At(oF32, 0), len(qv))
		if err != nil {
			t.Fatal(err)
		}
		got, err := cb.Ink2Read(Ink2At(oNV4, 0), len(qv))
		if err != nil {
			t.Fatal(err)
		}
		cos, maxErr := ink2FP8CosineAndMaxErr(want, got)
		t.Logf("prefill chunk %d (pos0=%d T=%d): cosine=%.6f max_abs_err=%.6f", ci, ch.pos0, ch.T, cos, maxErr)
		if cos < 0.99 || maxErr > 0.60 {
			t.Fatalf("prefill chunk %d accuracy out of bounds: cosine=%.6f max_abs_err=%.6f", ci, cos, maxErr)
		}
	}

	flags, err := cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] != 0 {
		t.Fatalf("unexpected NVFP4 overflow flag during prefill: %d", flags[0])
	}

	// Shape off the tensor-core tile must be refused.
	bad := p
	bad.HeadDim = 64
	tt := alloc(64)
	if err := cb.Ink2PAttendNVFP4(Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0),
		Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0),
		bad, 16, 0, 16, 0, 1); err == nil {
		t.Fatal("nvfp4 prefill attend accepted unsupported headDim=64")
	}
}

// fak-test:runtime fast est=5s lane=default
func TestCUDAInk2NVFP4KVZeroAndOverflow(t *testing.T) {
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
	kcNV4 := alloc(Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	vcNV4 := alloc(Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	ksNV4 := alloc(Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
	vsNV4 := alloc(Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
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

	// 1. All-zero K and V -> overflow=0 and all block scales == 0.
	if err := cb.Ink2QKVPostNVFP4(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
		Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcNV4, 0), Ink2At(vcNV4, 0), Ink2At(ksNV4, 0), Ink2At(vsNV4, 0),
		Ink2At(ovf, 0), Ink2At(meta, 0), p, capTok, nil); err != nil {
		t.Fatalf("zero vector Ink2QKVPostNVFP4: %v", err)
	}
	flags, err := cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] != 0 {
		t.Fatalf("zero vector triggered overflow flag: %d", flags[0])
	}
	ksWords, err := cb.Ink2Read(Ink2At(ksNV4, 0), Ink2NVFP4KVScaleFloats(1, p.KVHeads, p.HeadDim))
	if err != nil {
		t.Fatal(err)
	}
	for i, f := range ksWords {
		if math.Float32bits(f) != 0 {
			t.Fatalf("zero vector non-zero kscale word[%d]=0x%08x", i, math.Float32bits(f))
		}
	}

	// 2. Non-finite K element (+Inf) in decode -> overflow != 0.
	badK := make([]float32, kvDim)
	badK[3*p.HeadDim+7] = float32(math.Inf(1))
	write(kBuf, badK)
	if err := cb.Ink2QKVPostNVFP4(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
		Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcNV4, 0), Ink2At(vcNV4, 0), Ink2At(ksNV4, 0), Ink2At(vsNV4, 0),
		Ink2At(ovf, 0), Ink2At(meta, 0), p, capTok, nil); err != nil {
		t.Fatalf("inf K Ink2QKVPostNVFP4: %v", err)
	}
	flags, err = cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] == 0 {
		t.Fatal("non-finite K element did not set overflow flag")
	}

	// 3. E4M3FN block-scale overflow (amax16 / 6.0 > 448.0 => amax16 > 2688.0) -> overflow != 0.
	ovfRange := alloc(1)
	pBig := p
	pBig.QKClip = 10000
	bigK := make([]float32, kvDim)
	bigK[2*p.HeadDim+19] = 3000.0
	write(kBuf, bigK)
	if err := cb.Ink2QKVPostNVFP4(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
		Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcNV4, 0), Ink2At(vcNV4, 0), Ink2At(ksNV4, 0), Ink2At(vsNV4, 0),
		Ink2At(ovfRange, 0), Ink2At(meta, 0), pBig, capTok, nil); err != nil {
		t.Fatalf("range-overflow K Ink2QKVPostNVFP4: %v", err)
	}
	flags, err = cb.Ink2ReadInts(Ink2At(ovfRange, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] == 0 {
		t.Fatal("amax16=3000 (> 6*448=2688) did not set overflow flag")
	}

	// 4. Non-finite V element (NaN) in prefill -> overflow != 0.
	ovf2 := alloc(1)
	badV := make([]float32, kvDim)
	badV[1*p.HeadDim+5] = float32(math.NaN())
	write(kBuf, make([]float32, kvDim))
	write(vBuf, badV)
	if err := cb.Ink2PQKVPostNVFP4(Ink2At(qBuf, 0), Ink2At(rBuf, 0), Ink2At(kBuf, 0), Ink2At(vBuf, 0),
		Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcNV4, 0), Ink2At(vcNV4, 0), Ink2At(ksNV4, 0), Ink2At(vsNV4, 0),
		Ink2At(ovf2, 0), p, capTok, 0, 1); err != nil {
		t.Fatalf("nan V Ink2PQKVPostNVFP4: %v", err)
	}
	flags, err = cb.Ink2ReadInts(Ink2At(ovf2, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] == 0 {
		t.Fatal("non-finite V element in prefill did not set overflow flag")
	}
}
