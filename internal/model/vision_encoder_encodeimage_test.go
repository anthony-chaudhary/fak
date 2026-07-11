package model

// EncodeImage conformance (#4030). The oracle tests in vision_encoder_test.go drive the
// unexported forwardPixels core directly; this file exercises the PUBLIC VisionEncoder
// seam end to end — decode bytes -> preprocess -> ViT forward -> VisionEmbedding — and
// then feeds that embedding through the exact admission gate the decoder splice consumes
// (admitVisionEmbedding, multimodal.go:254-292). It pins the DoD output contract that the
// forwardPixels tests do not reach: EncodeImage's rows are ADMITTED at width == HiddenSize
// and their count stays within MaxEmbeddingTokens, i.e. they are ready to splice into the
// prompt sequence without further scaling.

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
)

// synthPNG builds a deterministic w x h RGB image and encodes it as PNG bytes, so
// EncodeImage's image.Decode (which needs a registered codec — supplied by the
// image/png import above) has a real payload to decode.
func synthPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{
				R: uint8((x * 7) % 256),
				G: uint8((y * 5) % 256),
				B: uint8((x + y) % 256),
				A: 255,
			})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png.Encode: %v", err)
	}
	return buf.Bytes()
}

// TestVisionEncoderEncodeImageSplicesAtHiddenSize is the end-to-end splice contract: a
// real image goes through the public EncodeImage method, and the resulting VisionEmbedding
// must pass admitVisionEmbedding at exactly Config.HiddenSize width and within the token
// cap — the property multimodal.go relies on when it appends the vectors as decoder rows.
func TestVisionEncoderEncodeImageSplicesAtHiddenSize(t *testing.T) {
	const H = 8
	fx := visionFixture{
		hidden: H, heads: 2, layers: 1, patch: 2, merge: 2,
		temporal: 1, ffn: 16, decoderHidden: H, twoLayerProj: false,
	}
	tower := buildVisionTower(t, fx)
	enc, err := newVisionEncoder(tower, H)
	if err != nil {
		t.Fatalf("newVisionEncoder: %v", err)
	}

	// A 16x12 image is not a clean multiple of patch*merge (=4) on its own, so this also
	// exercises preprocess's resize-to-grid before the forward.
	imgW, imgH := 16, 12
	img := MultimodalImage{
		MediaType: "image/png",
		Bytes:     synthPNG(t, imgW, imgH),
		Width:     imgW,
		Height:    imgH,
	}

	emb, err := enc.EncodeImage(img)
	if err != nil {
		t.Fatalf("EncodeImage: %v", err)
	}
	if len(emb.Vectors) == 0 {
		t.Fatal("EncodeImage returned zero vectors")
	}
	if !finite(emb.Vectors) {
		t.Fatal("EncodeImage produced a non-finite vector")
	}
	for i, v := range emb.Vectors {
		if len(v) != H {
			t.Fatalf("vector %d width = %d, want HiddenSize %d", i, len(v), H)
		}
	}

	// The splice seam: the encoder output must be admitted as-is at decoder hidden width.
	policy := MultimodalPolicy{Mode: MultimodalModeQuarantine}.withDefaults()
	verdict := MultimodalVerdict{Decision: MultimodalAllow, Mode: policy.Mode}
	if err := admitVisionEmbedding(&emb, policy, &verdict, H); err != nil {
		t.Fatalf("admitVisionEmbedding rejected EncodeImage output: %v (reason %q)", err, verdict.Reason)
	}
	if verdict.Decision != MultimodalAllow {
		t.Fatalf("verdict.Decision = %q, want allow", verdict.Decision)
	}
	if verdict.EmbeddingTokens != len(emb.Vectors) {
		t.Fatalf("verdict.EmbeddingTokens = %d, want %d", verdict.EmbeddingTokens, len(emb.Vectors))
	}
	if verdict.EmbeddingTokens > policy.MaxEmbeddingTokens {
		t.Fatalf("EmbeddingTokens %d exceeds MaxEmbeddingTokens %d", verdict.EmbeddingTokens, policy.MaxEmbeddingTokens)
	}
}

// TestVisionEncoderEncodeImageEmptyBytesRejected pins the fail-closed guard: EncodeImage
// must not attempt a decode on an empty payload (it is the seam's first contract check).
func TestVisionEncoderEncodeImageEmptyBytesRejected(t *testing.T) {
	const H = 8
	fx := visionFixture{
		hidden: H, heads: 2, layers: 1, patch: 2, merge: 2,
		temporal: 1, ffn: 16, decoderHidden: H, twoLayerProj: false,
	}
	enc, err := newVisionEncoder(buildVisionTower(t, fx), H)
	if err != nil {
		t.Fatalf("newVisionEncoder: %v", err)
	}
	if _, err := enc.EncodeImage(MultimodalImage{MediaType: "image/png"}); err == nil {
		t.Fatal("EncodeImage on empty bytes: want error, got nil")
	}
}
