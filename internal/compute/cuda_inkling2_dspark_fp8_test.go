//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

// fak-test:runtime fast est=5s lane=default
func TestCUDAInk2DSparkFP8KVMatchesFP32(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	var _ Inkling2DraftDevice = cb
	var _ Inkling2DraftFP8Device = cb

	const (
		heads     = 16
		kvHeads   = 8
		dim       = 128
		blockRows = 8
		capTok    = 48
		eps       = float32(1e-6)
		theta     = 10000.0
		ctxWrites = 60 // wraps past capTok to exercise ring wrap-around
	)
	kvDim := kvHeads * dim
	qDim := heads * dim
	rng := rand.New(rand.NewSource(20261127))

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
	intRef := func(v int) []float32 {
		return []float32{math.Float32frombits(uint32(int32(v)))}
	}

	kNormCtx := alloc(dim)
	write(kNormCtx, randVec(dim, 0.2, 1.0))

	kcF32, vcF32 := alloc(capTok*kvDim), alloc(capTok*kvDim)
	kcFP8 := alloc(Ink2FP8KVCodeFloats(capTok, kvHeads, dim))
	vcFP8 := alloc(Ink2FP8KVCodeFloats(capTok, kvHeads, dim))
	ksFP8 := alloc(Ink2FP8KVScaleFloats(capTok, kvHeads))
	vsFP8 := alloc(Ink2FP8KVScaleFloats(capTok, kvHeads))
	ovf := alloc(1)
	posDev := alloc(1)

	chunks := []struct {
		pos0, rows int
	}{
		{0, 32},
		{32, ctxWrites - 32},
	}
	for _, ch := range chunks {
		write(posDev, intRef(ch.pos0))
		kIn := alloc(ch.rows * kvDim)
		vIn := alloc(ch.rows * kvDim)
		write(kIn, randVec(ch.rows*kvDim, 1.5, 0))
		write(vIn, randVec(ch.rows*kvDim, 1.5, 0))

		if err := cb.Ink2DQK(Ink2At(kIn, 0), Ink2At(kNormCtx, 0), Ink2At(kcF32, 0), Ink2At(posDev, 0),
			ch.rows, kvHeads, dim, capTok, true, eps, theta); err != nil {
			t.Fatalf("pos0=%d Ink2DQK: %v", ch.pos0, err)
		}
		if err := cb.Ink2DRingStore(Ink2At(vIn, 0), Ink2At(vcF32, 0), Ink2At(posDev, 0),
			ch.rows, kvDim, capTok); err != nil {
			t.Fatalf("pos0=%d Ink2DRingStore: %v", ch.pos0, err)
		}

		if err := cb.Ink2DQKFP8(Ink2At(kIn, 0), Ink2At(kNormCtx, 0), Ink2At(kcFP8, 0), Ink2At(ksFP8, 0),
			Ink2At(ovf, 0), Ink2At(posDev, 0), ch.rows, kvHeads, dim, capTok, true, eps, theta); err != nil {
			t.Fatalf("pos0=%d Ink2DQKFP8: %v", ch.pos0, err)
		}
		if err := cb.Ink2DRingStoreFP8(Ink2At(vIn, 0), Ink2At(vcFP8, 0), Ink2At(vsFP8, 0),
			Ink2At(ovf, 0), Ink2At(posDev, 0), ch.rows, kvHeads, dim, capTok); err != nil {
			t.Fatalf("pos0=%d Ink2DRingStoreFP8: %v", ch.pos0, err)
		}
	}

	flags, err := cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] != 0 {
		t.Fatalf("finite context writes triggered overflow=%d", flags[0])
	}

	write(posDev, intRef(ctxWrites))
	qBuf := alloc(blockRows * qDim)
	kbBuf := alloc(blockRows * kvDim)
	vbBuf := alloc(blockRows * kvDim)
	write(qBuf, randVec(blockRows*qDim, 1.0, 0))
	write(kbBuf, randVec(blockRows*kvDim, 1.0, 0))
	write(vbBuf, randVec(blockRows*kvDim, 1.0, 0))

	outF32 := alloc(blockRows * qDim)
	outFP8 := alloc(blockRows * qDim)

	for _, tc := range []struct {
		name   string
		window int
	}{
		{name: "global", window: 0},
		{name: "local_swa", window: 24},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := cb.Ink2DAttend(Ink2At(qBuf, 0), Ink2At(kbBuf, 0), Ink2At(vbBuf, 0),
				Ink2At(kcF32, 0), Ink2At(vcF32, 0), Ink2At(outF32, 0), Ink2At(posDev, 0),
				blockRows, heads, kvHeads, dim, tc.window, capTok); err != nil {
				t.Fatalf("Ink2DAttend: %v", err)
			}
			if err := cb.Ink2DAttendFP8(Ink2At(qBuf, 0), Ink2At(kbBuf, 0), Ink2At(vbBuf, 0),
				Ink2At(kcFP8, 0), Ink2At(vcFP8, 0), Ink2At(ksFP8, 0), Ink2At(vsFP8, 0),
				Ink2At(outFP8, 0), Ink2At(posDev, 0),
				blockRows, heads, kvHeads, dim, tc.window, capTok); err != nil {
				t.Fatalf("Ink2DAttendFP8: %v", err)
			}
			gotF32, err := cb.Ink2Read(Ink2At(outF32, 0), blockRows*qDim)
			if err != nil {
				t.Fatal(err)
			}
			gotFP8, err := cb.Ink2Read(Ink2At(outFP8, 0), blockRows*qDim)
			if err != nil {
				t.Fatal(err)
			}
			cos, maxErr := ink2FP8CosineAndMaxErr(gotF32, gotFP8)
			t.Logf("%s: cos=%.6f maxErr=%.6f", tc.name, cos, maxErr)
			if cos < 0.999 {
				t.Fatalf("%s: FP8 vs FP32 cosine=%.6f < 0.999 (maxErr=%.6f)", tc.name, cos, maxErr)
			}
		})
	}
}

// fak-test:runtime fast est=2s lane=default
func TestCUDAInk2DSparkFP8KVZeroAndOverflow(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)

	const (
		kvHeads = 8
		dim     = 128
		capTok  = 8
		eps     = float32(1e-6)
		theta   = 10000.0
	)
	kvDim := kvHeads * dim
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

	kNorm := alloc(dim)
	ones := make([]float32, dim)
	for i := range ones {
		ones[i] = 1
	}
	write(kNorm, ones)

	kBuf, vBuf := alloc(kvDim), alloc(kvDim)
	kcFP8 := alloc(Ink2FP8KVCodeFloats(capTok, kvHeads, dim))
	vcFP8 := alloc(Ink2FP8KVCodeFloats(capTok, kvHeads, dim))
	ksFP8 := alloc(Ink2FP8KVScaleFloats(capTok, kvHeads))
	vsFP8 := alloc(Ink2FP8KVScaleFloats(capTok, kvHeads))
	ovf := alloc(1)
	posDev := alloc(1)

	// 1. All-zero K/V must emit scale=0 and keep overflow=0 (also verify zero Ink2Ref overflow works).
	if err := cb.Ink2DQKFP8(Ink2At(kBuf, 0), Ink2At(kNorm, 0), Ink2At(kcFP8, 0), Ink2At(ksFP8, 0),
		Ink2Ref{}, Ink2At(posDev, 0), 1, kvHeads, dim, capTok, true, eps, theta); err != nil {
		t.Fatalf("zero K nil-overflow Ink2DQKFP8: %v", err)
	}
	if err := cb.Ink2DQKFP8(Ink2At(kBuf, 0), Ink2At(kNorm, 0), Ink2At(kcFP8, 0), Ink2At(ksFP8, 0),
		Ink2At(ovf, 0), Ink2At(posDev, 0), 1, kvHeads, dim, capTok, true, eps, theta); err != nil {
		t.Fatalf("zero K Ink2DQKFP8: %v", err)
	}
	if err := cb.Ink2DRingStoreFP8(Ink2At(vBuf, 0), Ink2At(vcFP8, 0), Ink2At(vsFP8, 0),
		Ink2At(ovf, 0), Ink2At(posDev, 0), 1, kvHeads, dim, capTok); err != nil {
		t.Fatalf("zero V Ink2DRingStoreFP8: %v", err)
	}
	flags, err := cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] != 0 {
		t.Fatalf("zero vectors triggered overflow=%d", flags[0])
	}
	ks, err := cb.Ink2Read(Ink2At(ksFP8, 0), kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	vs, err := cb.Ink2Read(Ink2At(vsFP8, 0), kvHeads)
	if err != nil {
		t.Fatal(err)
	}
	for h := 0; h < kvHeads; h++ {
		if ks[h] != 0 || vs[h] != 0 {
			t.Fatalf("head %d zero vector non-zero scale: ks=%v vs=%v", h, ks[h], vs[h])
		}
	}

	// 2. +Inf in K sets overflow != 0 and zeros that head's scale.
	badK := make([]float32, kvDim)
	badK[3*dim+7] = float32(math.Inf(1))
	write(kBuf, badK)
	if err := cb.Ink2DQKFP8(Ink2At(kBuf, 0), Ink2At(kNorm, 0), Ink2At(kcFP8, 0), Ink2At(ksFP8, 0),
		Ink2At(ovf, 0), Ink2At(posDev, 0), 1, kvHeads, dim, capTok, true, eps, theta); err != nil {
		t.Fatalf("inf K Ink2DQKFP8: %v", err)
	}
	flags, err = cb.Ink2ReadInts(Ink2At(ovf, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] == 0 {
		t.Fatal("+Inf K did not set overflow flag")
	}

	// 3. NaN in V sets overflow != 0.
	ovf2 := alloc(1)
	badV := make([]float32, kvDim)
	badV[2*dim+11] = float32(math.NaN())
	write(vBuf, badV)
	if err := cb.Ink2DRingStoreFP8(Ink2At(vBuf, 0), Ink2At(vcFP8, 0), Ink2At(vsFP8, 0),
		Ink2At(ovf2, 0), Ink2At(posDev, 0), 1, kvHeads, dim, capTok); err != nil {
		t.Fatalf("nan V Ink2DRingStoreFP8: %v", err)
	}
	flags, err = cb.Ink2ReadInts(Ink2At(ovf2, 0), 1)
	if err != nil {
		t.Fatal(err)
	}
	if flags[0] == 0 {
		t.Fatal("NaN V did not set overflow flag")
	}
}
