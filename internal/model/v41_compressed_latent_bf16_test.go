package model

import (
	"math"
	"testing"
)

// The normalized latent is widened BF16. The reference's apply_rotary_emb
// writes only its rotated tail view back into that dtype before publication.
// This pins that copyback alone; it does not exercise reference FP4 quantization.
// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1ms lane=default
func TestV41CompressedLatentRoPEBF16Copyback(t *testing.T) {
	t.Parallel()
	m := &Model{Cfg: Config{HeadDim: 4, IndexHeadDim: 4, QKRopeHeadDim: 2, RopeTheta: 10000}}
	// A non-BF16 prefix sentinel detects accidental whole-row copyback.
	prefix := math.Float32frombits(0x3fa01234)
	latent := []float32{prefix, -2.5, 1, 1}
	key := append([]float32(nil), latent...)
	if err := m.v41CompressedPublicationRoPE(0, 1, latent, key); err != nil {
		t.Fatal(err)
	}
	// cos(1)-sin(1) rounds to -77/256; cos(1)+sin(1) to 177/128.
	// Fixed exact BF16 values make an omitted copyback observable without
	// deriving expected output through a production rotary or rounding helper.
	want := []float32{prefix, -2.5, -0.30078125, 1.3828125}
	c, s := float32(math.Cos(1)), float32(math.Sin(1))
	wantKey := []float32{prefix, -2.5, c - s, c + s}
	for i := range latent {
		if math.Float32bits(latent[i]) != math.Float32bits(want[i]) {
			t.Fatalf("latent[%d] = %g, want BF16 copyback %g", i, latent[i], want[i])
		}
		if math.Float32bits(key[i]) != math.Float32bits(wantKey[i]) {
			t.Fatalf("index key[%d] = %g, want unchanged F32 rotation %g", i, key[i], wantKey[i])
		}
	}
}
