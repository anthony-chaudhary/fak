//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

func ink2AsymFP8DequantToF32(codesWords, scaleWords []float32, capTok, kvHeads, headDim int) []float32 {
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
	hBlks := headDim / 16
	for p := 0; p < capTok; p++ {
		for h := 0; h < kvHeads; h++ {
			for d := 0; d < headDim; d++ {
				cb := codeBytes[(p*kvHeads+h)*headDim+d]
				sb := scaleBytes[(p*kvHeads+h)*hBlks+d/16]
				out[(p*kvHeads+h)*headDim+d] = float32(int8(cb)) * (ink2NVFP4DecodeE4M3FN(sb) / 127.0)
			}
		}
	}
	return out
}

func ink2AsymFWHT16InPlace(x []float32) {
	for base := 0; base+16 <= len(x); base += 16 {
		b := x[base : base+16]
		for step := 1; step < 16; step <<= 1 {
			for i := 0; i < 16; i += 2 * step {
				for j := 0; j < step; j++ {
					u, v := b[i+j], b[i+j+step]
					b[i+j], b[i+j+step] = u+v, u-v
				}
			}
		}
		for i := 0; i < 16; i++ {
			b[i] *= 0.25
		}
	}
}

func ink2AsymCosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		da, db := float64(a[i]), float64(b[i])
		dot += da * db
		na += da * da
		nb += db * db
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// fak-test:runtime fast est=200ms lane=default
func TestCUDAInk2AsymmetricKV(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	var _ Inkling2AsymKVDevice = cb

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
		numToks = 32
	)
	rng := rand.New(rand.NewSource(20261129))
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
			v[i] = float32(mean + sd*rng.NormFloat64())
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
	kcAsym := alloc(Ink2FP8KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	vcAsym := alloc(Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
	ksAsym := alloc(Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
	vsAsym := alloc(Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
	ovf := alloc(1)

	scoresRef, scoresAsym := alloc(2*p.Heads*capTok), alloc(2*p.Heads*capTok)
	outRef, outAsym := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.HeadDim)
	sinks := alloc(p.Heads)
	write(sinks, randVec(p.Heads, 0.5, 0))
	meta := alloc(1)

	qStep := alloc(p.Heads * p.HeadDim)
	rStep := alloc(p.Heads * p.RelLen)
	qStep2 := alloc(p.Heads * p.HeadDim)
	rStep2 := alloc(p.Heads * p.RelLen)
	kStep := alloc(kvDim)
	vStep := alloc(kvDim)

	t.Run("DecodeParity", func(t *testing.T) {
		for tok := 0; tok < numToks; tok++ {
			write(meta, []float32{math.Float32frombits(uint32(tok))})
			qv := randVec(p.Heads*p.HeadDim, 1, 0)
			rv := randVec(p.Heads*p.RelLen, 1, 0)
			kv := randVec(kvDim, 1, 0)
			vv := randVec(kvDim, 1, 0)
			write(qStep, qv)
			write(rStep, rv)
			write(qStep2, qv)
			write(rStep2, rv)
			write(kStep, kv)
			write(vStep, vv)

			if err := cb.Ink2QKVPost(Ink2At(qStep, 0), Ink2At(rStep, 0), Ink2At(kStep, 0), Ink2At(vStep, 0),
				Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcF32, 0), Ink2At(vcF32, 0),
				Ink2At(meta, 0), p, capTok, nil); err != nil {
				t.Fatalf("tok %d Ink2QKVPost: %v", tok, err)
			}
			if err := cb.Ink2QKVPostAsym(Ink2At(qStep2, 0), Ink2At(rStep2, 0), Ink2At(kStep, 0), Ink2At(vStep, 0),
				Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcAsym, 0), Ink2At(vcAsym, 0), Ink2At(ksAsym, 0), Ink2At(vsAsym, 0),
				Ink2At(ovf, 0), Ink2At(meta, 0), p, capTok, nil); err != nil {
				t.Fatalf("tok %d Ink2QKVPostAsym: %v", tok, err)
			}
		}

		flags, err := cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
		if err != nil {
			t.Fatal(err)
		}
		if flags[0] != 0 {
			t.Fatalf("unexpected asym KV overflow flag: %d", flags[0])
		}

		kcWords, err := cb.Ink2Read(Ink2At(kcAsym, 0), Ink2FP8KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		vcWords, err := cb.Ink2Read(Ink2At(vcAsym, 0), Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		ksWords, err := cb.Ink2Read(Ink2At(ksAsym, 0), Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		vsWords, err := cb.Ink2Read(Ink2At(vsAsym, 0), Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		kDeq := ink2AsymFP8DequantToF32(kcWords, ksWords, capTok, p.KVHeads, p.HeadDim)
		vDeq := ink2NVFP4DequantToF32(vcWords, vsWords, capTok, p.KVHeads, p.HeadDim)
		ink2AsymFWHT16InPlace(vDeq)
		write(kcDeq, kDeq)
		write(vcDeq, vDeq)

		write(meta, []float32{math.Float32frombits(uint32(numToks - 1))})
		if err := cb.Ink2Attend(Ink2At(qStep, 0), Ink2At(rStep, 0), Ink2At(sinks, 0), Ink2At(relBias, 0),
			Ink2At(kcDeq, 0), Ink2At(vcDeq, 0), Ink2At(scoresRef, 0), Ink2At(outRef, 0),
			Ink2At(meta, 0), p, extent, 0, capTok); err != nil {
			t.Fatalf("ref Ink2Attend: %v", err)
		}
		if err := cb.Ink2AttendAsym(Ink2At(qStep2, 0), Ink2At(rStep2, 0), Ink2At(sinks, 0), Ink2At(relBias, 0),
			Ink2At(kcAsym, 0), Ink2At(vcAsym, 0), Ink2At(ksAsym, 0), Ink2At(vsAsym, 0),
			Ink2At(scoresAsym, 0), Ink2At(outAsym, 0), Ink2At(meta, 0), p, extent, 0, capTok); err != nil {
			t.Fatalf("Ink2AttendAsym: %v", err)
		}

		gotRef, err := cb.Ink2Read(Ink2At(outRef, 0), p.Heads*p.HeadDim)
		if err != nil {
			t.Fatal(err)
		}
		gotAsym, err := cb.Ink2Read(Ink2At(outAsym, 0), p.Heads*p.HeadDim)
		if err != nil {
			t.Fatal(err)
		}
		cos := ink2AsymCosine(gotRef, gotAsym)
		if cos < 0.999999 {
			t.Fatalf("decode asym attend vs scalar dequant ref cosine = %.9f, want >= 0.999999", cos)
		}
	})

	t.Run("PrefillParity", func(t *testing.T) {
		const T = 32
		qP := alloc(T * p.Heads * p.HeadDim)
		rP := alloc(T * p.Heads * p.RelLen)
		qP2 := alloc(T * p.Heads * p.HeadDim)
		rP2 := alloc(T * p.Heads * p.RelLen)
		kP := alloc(T * kvDim)
		vP := alloc(T * kvDim)
		sinksP := alloc(T * p.Heads)
		outPRef := alloc(T * p.Heads * p.HeadDim)
		outPAsym := alloc(T * p.Heads * p.HeadDim)

		qv := randVec(T*p.Heads*p.HeadDim, 1, 0)
		rv := randVec(T*p.Heads*p.RelLen, 1, 0)
		kv := randVec(T*kvDim, 1, 0)
		vv := randVec(T*kvDim, 1, 0)
		sv := randVec(T*p.Heads, 0.5, 0)
		write(qP, qv)
		write(rP, rv)
		write(qP2, qv)
		write(rP2, rv)
		write(kP, kv)
		write(vP, vv)
		write(sinksP, sv)
		write(ovf, []float32{0})

		if err := cb.Ink2PQKVPostAsym(Ink2At(qP2, 0), Ink2At(rP2, 0), Ink2At(kP, 0), Ink2At(vP, 0),
			Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcAsym, 0), Ink2At(vcAsym, 0), Ink2At(ksAsym, 0), Ink2At(vsAsym, 0),
			Ink2At(ovf, 0), p, capTok, 0, T); err != nil {
			t.Fatalf("Ink2PQKVPostAsym: %v", err)
		}
		if err := cb.Ink2PQKVPost(Ink2At(qP, 0), Ink2At(rP, 0), Ink2At(kP, 0), Ink2At(vP, 0),
			Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcF32, 0), Ink2At(vcF32, 0), p, capTok, 0, T); err != nil {
			t.Fatalf("Ink2PQKVPost: %v", err)
		}

		kcWords, err := cb.Ink2Read(Ink2At(kcAsym, 0), Ink2FP8KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		vcWords, err := cb.Ink2Read(Ink2At(vcAsym, 0), Ink2NVFP4KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		ksWords, err := cb.Ink2Read(Ink2At(ksAsym, 0), Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		vsWords, err := cb.Ink2Read(Ink2At(vsAsym, 0), Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		write(kcDeq, ink2AsymFP8DequantToF32(kcWords, ksWords, capTok, p.KVHeads, p.HeadDim))
		vDeq := ink2NVFP4DequantToF32(vcWords, vsWords, capTok, p.KVHeads, p.HeadDim)
		ink2AsymFWHT16InPlace(vDeq)
		write(vcDeq, vDeq)

		if err := cb.Ink2PAttend(Ink2At(qP, 0), Ink2At(rP, 0), Ink2At(sinksP, 0), Ink2At(relBias, 0),
			Ink2At(kcDeq, 0), Ink2At(vcDeq, 0), Ink2At(outPRef, 0), p, extent, 0, capTok, 0, T); err != nil {
			t.Fatalf("ref Ink2PAttend: %v", err)
		}
		if err := cb.Ink2PAttendAsym(Ink2At(qP2, 0), Ink2At(rP2, 0), Ink2At(sinksP, 0), Ink2At(relBias, 0),
			Ink2At(kcAsym, 0), Ink2At(vcAsym, 0), Ink2At(ksAsym, 0), Ink2At(vsAsym, 0),
			Ink2At(outPAsym, 0), p, extent, 0, capTok, 0, T); err != nil {
			t.Fatalf("Ink2PAttendAsym: %v", err)
		}

		gotRef, err := cb.Ink2Read(Ink2At(outPRef, 0), T*p.Heads*p.HeadDim)
		if err != nil {
			t.Fatal(err)
		}
		gotAsym, err := cb.Ink2Read(Ink2At(outPAsym, 0), T*p.Heads*p.HeadDim)
		if err != nil {
			t.Fatal(err)
		}
		cos := ink2AsymCosine(gotRef, gotAsym)
		if cos < 0.999999 {
			t.Fatalf("prefill asym attend vs scalar dequant ref cosine = %.9f, want >= 0.999999", cos)
		}
	})

	t.Run("ZeroInfNaNAndOverflow", func(t *testing.T) {
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

		checkOverflow := func(label string, kData, vData []float32, wantFlag bool) {
			t.Helper()
			write(ovf, []float32{0})
			write(kStep, kData)
			write(vStep, vData)
			if err := cb.Ink2QKVPostAsym(Ink2At(qStep, 0), Ink2At(rStep, 0), Ink2At(kStep, 0), Ink2At(vStep, 0),
				Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcAsym, 0), Ink2At(vcAsym, 0), Ink2At(ksAsym, 0), Ink2At(vsAsym, 0),
				Ink2At(ovf, 0), Ink2At(meta, 0), p, capTok, nil); err != nil {
				t.Fatalf("%s Ink2QKVPostAsym: %v", label, err)
			}
			flags, err := cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
			if err != nil {
				t.Fatal(err)
			}
			if wantFlag && flags[0] == 0 {
				t.Fatalf("%s: expected overflow != 0, got 0", label)
			}
			if !wantFlag && flags[0] != 0 {
				t.Fatalf("%s: expected overflow == 0, got %d", label, flags[0])
			}
		}

		zeros := make([]float32, kvDim)
		checkOverflow("zero", zeros, zeros, false)

		ksWords, err := cb.Ink2Read(Ink2At(ksAsym, 0), Ink2NVFP4KVScaleFloats(1, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		for i, f := range ksWords {
			if math.Float32bits(f) != 0 {
				t.Fatalf("zero vector non-zero kscale word[%d]=0x%08x", i, math.Float32bits(f))
			}
		}

		infK := make([]float32, kvDim)
		infK[3*p.HeadDim+7] = float32(math.Inf(1))
		checkOverflow("k_inf", infK, zeros, true)

		nanV := make([]float32, kvDim)
		nanV[2*p.HeadDim+5] = float32(math.NaN())
		checkOverflow("v_nan", zeros, nanV, true)

		// K block-16 FP8 scale overflow: post-clip element > 448*448 = 200704.
		ovfK := make([]float32, kvDim)
		ovfK[0] = 300000.0
		pLargeClip := p
		pLargeClip.QKClip = 1e6
		write(ovf, []float32{0})
		write(kStep, ovfK)
		write(vStep, zeros)
		if err := cb.Ink2QKVPostAsym(Ink2At(qStep, 0), Ink2At(rStep, 0), Ink2At(kStep, 0), Ink2At(vStep, 0),
			Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcAsym, 0), Ink2At(vcAsym, 0), Ink2At(ksAsym, 0), Ink2At(vsAsym, 0),
			Ink2At(ovf, 0), Ink2At(meta, 0), pLargeClip, capTok, nil); err != nil {
			t.Fatalf("k_scale_overflow Ink2QKVPostAsym: %v", err)
		}
		flags, err := cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
		if err != nil {
			t.Fatal(err)
		}
		if flags[0] == 0 {
			t.Fatalf("k_scale_overflow: expected overflow != 0, got 0")
		}
	})

	t.Run("SmallMagnitudeNoUnderflow", func(t *testing.T) {
		// Regression test: previously e4m3fn(amax16/448) underflowed to 0 for amax16 < 0.4375.
		kSmall := make([]float32, kvDim)
		vSmall := make([]float32, kvDim)
		for i := range kSmall {
			switch (i / 16) % 3 {
			case 0:
				kSmall[i] = 0.05 * float32((i%16)+1) / 16.0
			case 1:
				kSmall[i] = -0.20 * float32((i%16)+1) / 16.0
			default:
				kSmall[i] = 0.40 * float32((i%16)+1) / 16.0
			}
			vSmall[i] = kSmall[i]
		}
		write(ovf, []float32{0})
		write(kStep, kSmall)
		write(vStep, vSmall)
		write(meta, []float32{math.Float32frombits(0)})
		if err := cb.Ink2QKVPostAsym(Ink2At(qStep, 0), Ink2At(rStep, 0), Ink2At(kStep, 0), Ink2At(vStep, 0),
			Ink2At(qclip, 0), Ink2At(rclip, 0), Ink2At(kcAsym, 0), Ink2At(vcAsym, 0), Ink2At(ksAsym, 0), Ink2At(vsAsym, 0),
			Ink2At(ovf, 0), Ink2At(meta, 0), p, capTok, nil); err != nil {
			t.Fatalf("Ink2QKVPostAsym small: %v", err)
		}
		kcWords, err := cb.Ink2Read(Ink2At(kcAsym, 0), Ink2FP8KVCodeFloats(capTok, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		ksWords, err := cb.Ink2Read(Ink2At(ksAsym, 0), Ink2NVFP4KVScaleFloats(capTok, p.KVHeads, p.HeadDim))
		if err != nil {
			t.Fatal(err)
		}
		kDeq := ink2AsymFP8DequantToF32(kcWords, ksWords, capTok, p.KVHeads, p.HeadDim)
		var maxAbs float32
		for _, v := range kDeq[:kvDim] {
			if a := float32(math.Abs(float64(v))); a > maxAbs {
				maxAbs = a
			}
		}
		if maxAbs == 0 {
			t.Fatalf("small-magnitude K block (amax <= 0.40) underflowed to zero")
		}
	})
}
