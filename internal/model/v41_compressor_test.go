package model

import (
	"math"
	"testing"
)

// The expected values below independently transcribe the pinned reference's
// per-dimension softmax pooling. The raw-pool boundary begins after wkv/wgate
// projection; pushNormalized extends it through the checkpoint's learned norm.
func TestV41CompressorProjectedPooling(t *testing.T) {
	c, err := newV41CompressorPool(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if got, emitted, err := c.push(0, []float32{1, 10}, []float32{0, 2}); err != nil || emitted || got != nil {
		t.Fatalf("partial group = (%v,%v,%v), want (nil,false,nil)", got, emitted, err)
	}
	got, emitted, err := c.push(1, []float32{3, 20}, []float32{float32(math.Log(3)), 0})
	if err != nil || !emitted {
		t.Fatalf("completed group = (%v,%v,%v), want emitted", got, emitted, err)
	}
	w0 := float32(math.Exp(float64(-float32(math.Log(3)))))
	w1 := float32(1)
	d0 := w0 + w1
	v0 := 1*(w0/d0) + 3*(w1/d0)
	w1d0 := float32(1)
	w1d1 := float32(math.Exp(-2))
	d1 := w1d0 + w1d1
	v1 := 10*(w1d0/d1) + 20*(w1d1/d1)
	want := []float32{v0, v1}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("pooled[%d]=%g want exact FP32 %g", i, got[i], want[i])
		}
	}
}

func TestV41CompressorGroupingIsDeterministicAndChecksBoundaries(t *testing.T) {
	rows := [][]float32{{1, 3}, {2, 5}, {7, 11}, {13, 17}}
	scores := [][]float32{{0, 1}, {2, 0}, {-1, 1}, {0, 3}}
	run := func() ([][]float32, error) {
		c, err := newV41CompressorPool(2, 2)
		if err != nil {
			return nil, err
		}
		var out [][]float32
		for pos := range rows {
			latent, emitted, err := c.push(pos, rows[pos], scores[pos])
			if err != nil {
				return nil, err
			}
			if emitted {
				out = append(out, latent)
			}
		}
		return out, nil
	}
	a, err := run()
	if err != nil {
		t.Fatal(err)
	}
	b, err := run()
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 2 || len(b) != 2 {
		t.Fatalf("emitted groups a=%d b=%d, want 2", len(a), len(b))
	}
	for group := range a {
		for dim := range a[group] {
			if a[group][dim] != b[group][dim] {
				t.Fatalf("replay drift at group=%d dim=%d: %g != %g", group, dim, a[group][dim], b[group][dim])
			}
		}
	}

	direct, err := newV41CompressorPool(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	got, emitted, err := direct.push(0, []float32{3, 4}, []float32{99, -99})
	if err != nil || !emitted || got[0] != 3 || got[1] != 4 {
		t.Fatalf("ratio-1 direct path = (%v,%v,%v)", got, emitted, err)
	}
	if _, _, err := direct.push(2, []float32{3, 4}, []float32{0, 0}); err == nil {
		t.Fatal("non-contiguous position accepted")
	}
	if _, err := newV41CompressorPool(0, 2); err == nil {
		t.Fatal("zero ratio accepted")
	}
	if _, err := newV41CompressorPool(int(^uint(0)>>1), 2); err == nil {
		t.Fatal("overflowing geometry accepted")
	}
}

func TestV41CompressorNormalizesWeightsBeforeWeightedSum(t *testing.T) {
	c, err := newV41CompressorPool(2, 1)
	if err != nil {
		t.Fatal(err)
	}
	_, _, _ = c.push(0, []float32{1e20}, []float32{0})
	got, emitted, err := c.push(1, []float32{-1e20}, []float32{-1})
	if err != nil || !emitted {
		t.Fatalf("emitted=%v err=%v", emitted, err)
	}
	e := float32(math.Exp(-1))
	want := float32(1e20)*(1/(1+e)) + float32(-1e20)*(e/(1+e))
	if got[0] != want {
		t.Fatalf("pooled=%g want normalized-before-multiply FP32 %g", got[0], want)
	}
}

func TestV41CompressorLearnedRMSNormAndBF16Boundaries(t *testing.T) {
	c, err := newV41CompressorPool(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	weight := []float32{1.25, 0.75}
	if _, emitted, err := c.pushNormalized(0, []float32{1.003, 2.007}, []float32{0, 0}, weight, 1e-6); err != nil || emitted {
		t.Fatalf("partial normalized group emitted=%v err=%v", emitted, err)
	}
	got, emitted, err := c.pushNormalized(1, []float32{3.011, 4.015}, []float32{0, 0}, weight, 1e-6)
	if err != nil || !emitted {
		t.Fatalf("normalized group emitted=%v err=%v", emitted, err)
	}

	// Independent oracle: equal scores average each dimension, then the source
	// casts to BF16 before FP32 RMSNorm, multiplies the learned weight, and casts
	// the result to BF16. Constants are the widened IEEE BF16 results.
	want := []float32{0.9765625, 0.8828125}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("normalized[%d]=%g want exact BF16 %g", i, got[i], want[i])
		}
	}

	bad, err := newV41CompressorPool(1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := bad.pushNormalized(0, []float32{1, 2}, nil, []float32{1}, 1e-6); err == nil {
		t.Fatal("malformed learned norm weight accepted")
	}
}
