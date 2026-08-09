package model

// The end-to-end image+text contract (#4875): raw image bytes and a template-shaped prompt
// go in through ONE model call, and the decoder sees the encoder's rows at the placeholder
// positions. vision_splice_test.go pins the reconciliation over PRECOMPUTED embeddings;
// this file pins the seam that actually drives the tower, so EncodeImage / NewVisionEncoder
// / ForwardMultimodal finally have a caller and the placeholder count is reconciled against
// the encoder's own NumImageTokens rather than against a hand-written number.

// The PNG codec EncodeImage's image.Decode needs is registered by
// vision_encoder_encodeimage_test.go's image/png import, in this same package — which is
// itself the layering point: internal/model never imports a concrete codec, the caller
// that owns the wire does.

import (
	"errors"
	"strings"
	"testing"
)

// visionPromptModel is a synthetic decoder with a real (tiny) vision tower attached, i.e.
// the shape a --mmproj load produces: Model.Vision populated, projector out == HiddenSize.
func visionPromptModel(t *testing.T) *Model {
	t.Helper()
	m := multimodalTestModel()
	m.Vision = buildVisionTower(t, visionFixture{
		hidden: m.Cfg.HiddenSize, heads: 2, layers: 1, patch: 2, merge: 2,
		temporal: 1, ffn: 16, decoderHidden: m.Cfg.HiddenSize, twoLayerProj: false,
	})
	return m
}

// TestMultimodalForwardImagePromptSplicesTowerRowsAtPlaceholder is the done condition of
// the wiring: ONE <|image_pad|> in the template expands to exactly the tower's
// NumImageTokens rows for that image, those rows land at the placeholder's position, and
// the forward's sequence length is the expanded prompt length. The expected count is read
// from the encoder, not hardcoded, so this is a reconciliation and not a restatement.
func TestMultimodalForwardImagePromptSplicesTowerRowsAtPlaceholder(t *testing.T) {
	m := visionPromptModel(t)
	const pad = 7
	const imgW, imgH = 16, 12

	enc, err := newVisionEncoder(m.Vision, m.Cfg.HiddenSize)
	if err != nil {
		t.Fatalf("newVisionEncoder: %v", err)
	}
	wantTokens := enc.NumImageTokens(imgW, imgH)
	if wantTokens <= 0 {
		t.Fatalf("fixture yields %d image tokens for %dx%d; the test needs a non-empty image", wantTokens, imgW, imgH)
	}

	img := MultimodalImage{MediaType: "image/png", Bytes: synthPNG(t, imgW, imgH), Width: imgW, Height: imgH}
	ids := []int{1, 2, pad, 3} // template form: exactly one placeholder for the one image
	policy := MultimodalPolicy{Mode: MultimodalModeQuarantine}

	req, _, err := m.EncodeImagePrompt(ids, pad, []MultimodalImage{img}, policy)
	if err != nil {
		t.Fatalf("EncodeImagePrompt: %v", err)
	}
	if len(req.Parts) != 3 {
		t.Fatalf("parts = %d, want 3 (text | image | text)", len(req.Parts))
	}
	if req.Parts[1].Image == nil {
		t.Fatalf("part 1 = %+v, want the image part at the placeholder position", req.Parts[1])
	}
	if got := len(req.Parts[1].Image.Vectors); got != wantTokens {
		t.Fatalf("spliced image rows = %d, want NumImageTokens(%d,%d) = %d", got, imgW, imgH, wantTokens)
	}

	act, verdict, err := m.ForwardImagePrompt(ids, pad, []MultimodalImage{img}, policy)
	if err != nil {
		t.Fatalf("ForwardImagePrompt: %v", err)
	}
	// 4 template ids - 1 placeholder + wantTokens image rows.
	if want := len(ids) - 1 + wantTokens; act.Seq != want {
		t.Fatalf("forward seq = %d, want %d", act.Seq, want)
	}
	if verdict.Decision != MultimodalAllow || verdict.Images != 1 || verdict.EmbeddingTokens != wantTokens {
		t.Fatalf("verdict = %+v, want allow with 1 image and %d embedding tokens", verdict, wantTokens)
	}
}

// TestMultimodalForwardImagePromptAdmitsBeforeAnyPixelWork pins the ordering the gate is
// worth having: an image that fails the quarantine opt-in or a byte/pixel bound is refused
// WITHOUT decoding or running the tower. The witness is a model with no vision tower at
// all — reaching the encoder would fail with "no vision tower retained", so a governance
// error proves the check ran first.
func TestMultimodalForwardImagePromptAdmitsBeforeAnyPixelWork(t *testing.T) {
	m := multimodalTestModel() // deliberately no Vision tower
	const pad = 7
	ids := []int{1, pad, 2}
	img := MultimodalImage{MediaType: "image/png", Bytes: synthPNG(t, 16, 12), Width: 16, Height: 12}

	// Quarantine opt-in missing: refused before encode.
	_, verdict, err := m.ForwardImagePrompt(ids, pad, []MultimodalImage{img}, MultimodalPolicy{})
	if !errors.Is(err, ErrMultimodalQuarantined) {
		t.Fatalf("mode disabled: error = %v, want ErrMultimodalQuarantined (a tower error means pixels ran first)", err)
	}
	if verdict.Decision != MultimodalQuarantine || verdict.QuarantineID == "" {
		t.Fatalf("verdict = %+v, want a quarantine decision carrying the held-image id", verdict)
	}

	// Byte budget exceeded: same story, and the bounds keep ForwardMultimodal's precedence
	// (a bound denies even when the mode would otherwise quarantine).
	big := MultimodalPolicy{Mode: MultimodalModeQuarantine, MaxImageBytes: 4}
	_, verdict, err = m.ForwardImagePrompt(ids, pad, []MultimodalImage{img}, big)
	if !errors.Is(err, ErrMultimodalDenied) {
		t.Fatalf("byte budget: error = %v, want ErrMultimodalDenied before encode", err)
	}
	if verdict.Decision != MultimodalDeny || !strings.Contains(verdict.Reason, "bytes exceed") {
		t.Fatalf("verdict = %+v, want a byte-budget denial", verdict)
	}
}

// TestMultimodalForwardImagePromptTextOnlyNeedsNoTower proves wiring vision in cannot
// perturb — or even burden — a text-only model: no images means no encoder is built (this
// model has no tower) and the activations are bit-identical to the plain text forward.
func TestMultimodalForwardImagePromptTextOnlyNeedsNoTower(t *testing.T) {
	m := multimodalTestModel()
	ids := []int{1, 2, 3}

	for _, padID := range []int{7, -1} {
		got, verdict, err := m.ForwardImagePrompt(ids, padID, nil, MultimodalPolicy{})
		if err != nil {
			t.Fatalf("padID %d: ForwardImagePrompt: %v", padID, err)
		}
		if verdict.Images != 0 || verdict.EmbeddingTokens != 0 {
			t.Fatalf("padID %d: verdict = %+v, want no image accounting", padID, verdict)
		}
		assertActivationsBitsEqual(t, got, m.Forward(ids))
	}
}

// TestMultimodalForwardImagePromptRefusesPlaceholderCountMismatch covers the two ways a
// caller can get the template wrong: a placeholder count that disagrees with the image
// count, and a stream that was ALREADY expanded (which must not be expanded again). Both
// are typed ErrVisionSplice refusals, never a silent mis-splice.
func TestMultimodalForwardImagePromptRefusesPlaceholderCountMismatch(t *testing.T) {
	m := visionPromptModel(t)
	const pad = 7
	policy := MultimodalPolicy{Mode: MultimodalModeQuarantine}
	img := MultimodalImage{MediaType: "image/png", Bytes: synthPNG(t, 16, 12), Width: 16, Height: 12}

	// Two placeholders, one image.
	if _, _, err := m.EncodeImagePrompt([]int{pad, 1, pad}, pad, []MultimodalImage{img}, policy); !errors.Is(err, ErrVisionSplice) {
		t.Fatalf("two placeholders / one image: error = %v, want ErrVisionSplice", err)
	}
	// An image with no placeholder to land at.
	if _, _, err := m.EncodeImagePrompt([]int{1, 2}, pad, []MultimodalImage{img}, policy); !errors.Is(err, ErrVisionSplice) {
		t.Fatalf("no placeholder: error = %v, want ErrVisionSplice", err)
	}
	// Images supplied against a vocab that has no placeholder at all.
	if _, _, err := m.EncodeImagePrompt([]int{1, 2}, -1, []MultimodalImage{img}, policy); !errors.Is(err, ErrVisionSplice) {
		t.Fatalf("no placeholder in vocab: error = %v, want ErrVisionSplice", err)
	}
}
