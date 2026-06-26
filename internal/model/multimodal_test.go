package model

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestForwardMultimodalTextOnlyMatchesForward(t *testing.T) {
	m := multimodalTestModel()
	ids := []int{1, 2, 3}

	got, verdict, err := m.ForwardMultimodal(MultimodalRequest{
		Parts: []MultimodalPart{{TokenIDs: ids}},
	})
	if err != nil {
		t.Fatalf("ForwardMultimodal text-only: %v", err)
	}
	if verdict.Decision != MultimodalAllow || verdict.Images != 0 || verdict.EmbeddingTokens != 0 {
		t.Fatalf("verdict = %+v, want text-only allow with no image accounting", verdict)
	}
	want := m.Forward(ids)
	assertActivationsBitsEqual(t, got, want)
}

func TestForwardMultimodalDefaultQuarantinesImages(t *testing.T) {
	m := multimodalTestModel()
	_, verdict, err := m.ForwardMultimodal(MultimodalRequest{
		Parts: []MultimodalPart{
			{TokenIDs: []int{1}},
			{Image: validVisionEmbedding(m, 1)},
		},
	})
	if !errors.Is(err, ErrMultimodalQuarantined) {
		t.Fatalf("error = %v, want ErrMultimodalQuarantined", err)
	}
	if verdict.Decision != MultimodalQuarantine {
		t.Fatalf("decision = %q, want quarantine (verdict=%+v)", verdict.Decision, verdict)
	}
	if verdict.Images != 1 || verdict.EmbeddingTokens != 1 || verdict.ImageBytes == 0 {
		t.Fatalf("image accounting = %+v, want one image accounted before hold", verdict)
	}
	if !strings.HasPrefix(verdict.QuarantineID, "vision-sha256:") {
		t.Fatalf("QuarantineID = %q, want vision-sha256 digest", verdict.QuarantineID)
	}
}

func TestForwardMultimodalQuarantineIDBindsEmbeddingBits(t *testing.T) {
	m := multimodalTestModel()
	a := validVisionEmbedding(m, 1)
	b := validVisionEmbedding(m, 1)
	a.Image.Bytes = nil
	b.Image.Bytes = nil
	b.Vectors[0][0] += 0.25

	_, va, err := m.ForwardMultimodal(MultimodalRequest{Parts: []MultimodalPart{{Image: a}}})
	if !errors.Is(err, ErrMultimodalQuarantined) {
		t.Fatalf("first error = %v, want ErrMultimodalQuarantined", err)
	}
	_, vb, err := m.ForwardMultimodal(MultimodalRequest{Parts: []MultimodalPart{{Image: b}}})
	if !errors.Is(err, ErrMultimodalQuarantined) {
		t.Fatalf("second error = %v, want ErrMultimodalQuarantined", err)
	}
	if va.QuarantineID == "" || va.QuarantineID == vb.QuarantineID {
		t.Fatalf("quarantine ids = %q and %q, want embedding-sensitive distinct ids", va.QuarantineID, vb.QuarantineID)
	}
}

func TestForwardMultimodalQuarantineModeAllowsBoundedEmbeddings(t *testing.T) {
	m := multimodalTestModel()
	img := validVisionEmbedding(m, 2)
	orig := append([]float32(nil), img.Vectors[0]...)
	act, verdict, err := m.ForwardMultimodal(MultimodalRequest{
		Policy: MultimodalPolicy{Mode: MultimodalModeQuarantine},
		Parts: []MultimodalPart{
			{TokenIDs: []int{1}},
			{Image: img},
			{TokenIDs: []int{2}},
		},
	})
	if err != nil {
		t.Fatalf("ForwardMultimodal quarantine mode: %v", err)
	}
	if verdict.Decision != MultimodalAllow || verdict.Images != 1 || verdict.EmbeddingTokens != 2 {
		t.Fatalf("verdict = %+v, want allow with one image and two embedding tokens", verdict)
	}
	if act.Seq != 4 || len(act.Logits) != 4 || len(act.Logits[3]) != m.Cfg.VocabSize {
		t.Fatalf("bad activation shape: seq=%d logits=%d last=%d", act.Seq, len(act.Logits), len(act.Logits[3]))
	}
	for i, got := range img.Vectors[0] {
		if math.Float32bits(got) != math.Float32bits(orig[i]) {
			t.Fatalf("source image embedding mutated at %d: got %v want %v", i, got, orig[i])
		}
	}
	for i, v := range act.Logits[3] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit[%d] not finite: %v", i, v)
		}
	}
}

func TestForwardMultimodalGovernanceLimits(t *testing.T) {
	m := multimodalTestModel()
	tests := []struct {
		name   string
		image  *VisionEmbedding
		policy MultimodalPolicy
	}{
		{
			name:   "bytes",
			image:  withImageBytes(validVisionEmbedding(m, 1), make([]byte, 11)),
			policy: MultimodalPolicy{Mode: MultimodalModeQuarantine, MaxImageBytes: 10},
		},
		{
			name:   "pixels",
			image:  withImageSize(validVisionEmbedding(m, 1), 100, 100),
			policy: MultimodalPolicy{Mode: MultimodalModeQuarantine, MaxImagePixels: 99},
		},
		{
			name:   "embedding tokens",
			image:  validVisionEmbedding(m, 2),
			policy: MultimodalPolicy{Mode: MultimodalModeQuarantine, MaxEmbeddingTokens: 1},
		},
		{
			name:   "media type",
			image:  withMediaType(validVisionEmbedding(m, 1), "text/plain"),
			policy: MultimodalPolicy{Mode: MultimodalModeQuarantine},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, verdict, err := m.ForwardMultimodal(MultimodalRequest{
				Policy: tt.policy,
				Parts:  []MultimodalPart{{Image: tt.image}},
			})
			if !errors.Is(err, ErrMultimodalDenied) {
				t.Fatalf("error = %v, want ErrMultimodalDenied", err)
			}
			if verdict.Decision != MultimodalDeny {
				t.Fatalf("decision = %q, want deny (verdict=%+v)", verdict.Decision, verdict)
			}
		})
	}
}

func TestForwardMultimodalRejectsWrongEmbeddingWidth(t *testing.T) {
	m := multimodalTestModel()
	img := validVisionEmbedding(m, 1)
	img.Vectors[0] = img.Vectors[0][:m.Cfg.HiddenSize-1]

	_, verdict, err := m.ForwardMultimodal(MultimodalRequest{
		Policy: MultimodalPolicy{Mode: MultimodalModeQuarantine},
		Parts:  []MultimodalPart{{Image: img}},
	})
	if !errors.Is(err, ErrMultimodalDenied) {
		t.Fatalf("error = %v, want ErrMultimodalDenied", err)
	}
	if verdict.Decision != MultimodalDeny || !strings.Contains(verdict.Reason, "width") {
		t.Fatalf("verdict = %+v, want width denial", verdict)
	}
}

func TestForwardMultimodalRejectsInvalidPolicy(t *testing.T) {
	m := multimodalTestModel()
	tests := []struct {
		name   string
		policy MultimodalPolicy
	}{
		{name: "unknown mode", policy: MultimodalPolicy{Mode: MultimodalMode("on")}},
		{name: "negative max images", policy: MultimodalPolicy{MaxImages: -1}},
		{name: "negative max bytes", policy: MultimodalPolicy{MaxImageBytes: -1}},
		{name: "negative max pixels", policy: MultimodalPolicy{MaxImagePixels: -1}},
		{name: "negative max embedding tokens", policy: MultimodalPolicy{MaxEmbeddingTokens: -1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, verdict, err := m.ForwardMultimodal(MultimodalRequest{
				Policy: tt.policy,
				Parts:  []MultimodalPart{{TokenIDs: []int{1}}},
			})
			if !errors.Is(err, ErrMultimodalDenied) {
				t.Fatalf("error = %v, want ErrMultimodalDenied", err)
			}
			if verdict.Decision != MultimodalDeny {
				t.Fatalf("decision = %q, want deny (verdict=%+v)", verdict.Decision, verdict)
			}
		})
	}
}

func TestForwardMultimodalRejectsEmptyPrompt(t *testing.T) {
	m := multimodalTestModel()
	_, verdict, err := m.ForwardMultimodal(MultimodalRequest{})
	if !errors.Is(err, ErrMultimodalDenied) {
		t.Fatalf("error = %v, want ErrMultimodalDenied", err)
	}
	if verdict.Decision != MultimodalDeny || !strings.Contains(verdict.Reason, "empty prompt") {
		t.Fatalf("verdict = %+v, want empty-prompt denial", verdict)
	}
}

func TestForwardMultimodalRejectsTokenOutOfVocab(t *testing.T) {
	m := multimodalTestModel()
	_, verdict, err := m.ForwardMultimodal(MultimodalRequest{
		Parts: []MultimodalPart{{TokenIDs: []int{m.Cfg.VocabSize}}},
	})
	if !errors.Is(err, ErrMultimodalDenied) {
		t.Fatalf("error = %v, want ErrMultimodalDenied", err)
	}
	if verdict.Decision != MultimodalDeny || !strings.Contains(verdict.Reason, "vocab") {
		t.Fatalf("verdict = %+v, want out-of-vocab denial", verdict)
	}
}

func TestForwardMultimodalRejectsMixedOrEmptyPart(t *testing.T) {
	m := multimodalTestModel()
	tests := []struct {
		name string
		part MultimodalPart
	}{
		{name: "both text and image", part: MultimodalPart{TokenIDs: []int{1}, Image: validVisionEmbedding(m, 1)}},
		{name: "neither text nor image", part: MultimodalPart{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, verdict, err := m.ForwardMultimodal(MultimodalRequest{
				Policy: MultimodalPolicy{Mode: MultimodalModeQuarantine},
				Parts:  []MultimodalPart{tt.part},
			})
			if !errors.Is(err, ErrMultimodalDenied) {
				t.Fatalf("error = %v, want ErrMultimodalDenied", err)
			}
			if verdict.Decision != MultimodalDeny || !strings.Contains(verdict.Reason, "exactly one") {
				t.Fatalf("verdict = %+v, want exactly-one-of denial", verdict)
			}
		})
	}
}

func TestForwardMultimodalRejectsTooManyImages(t *testing.T) {
	m := multimodalTestModel()
	parts := make([]MultimodalPart, 3)
	for i := range parts {
		parts[i] = MultimodalPart{Image: validVisionEmbedding(m, 1)}
	}
	_, verdict, err := m.ForwardMultimodal(MultimodalRequest{
		Policy: MultimodalPolicy{Mode: MultimodalModeQuarantine, MaxImages: 2},
		Parts:  parts,
	})
	if !errors.Is(err, ErrMultimodalDenied) {
		t.Fatalf("error = %v, want ErrMultimodalDenied", err)
	}
	if verdict.Decision != MultimodalDeny || !strings.Contains(verdict.Reason, "too many images") {
		t.Fatalf("verdict = %+v, want too-many-images denial", verdict)
	}
}

func TestForwardMultimodalRejectsEmptyEmbedding(t *testing.T) {
	m := multimodalTestModel()
	img := validVisionEmbedding(m, 1)
	img.Vectors = nil
	_, verdict, err := m.ForwardMultimodal(MultimodalRequest{
		Policy: MultimodalPolicy{Mode: MultimodalModeQuarantine},
		Parts:  []MultimodalPart{{Image: img}},
	})
	if !errors.Is(err, ErrMultimodalDenied) {
		t.Fatalf("error = %v, want ErrMultimodalDenied", err)
	}
	if verdict.Decision != MultimodalDeny || !strings.Contains(verdict.Reason, "empty") {
		t.Fatalf("verdict = %+v, want empty-embedding denial", verdict)
	}
}

func multimodalTestModel() *Model {
	return NewSynthetic(Config{
		HiddenSize:       8,
		NumLayers:        1,
		NumHeads:         2,
		NumKVHeads:       1,
		HeadDim:          4,
		IntermediateSize: 16,
		VocabSize:        16,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       -1,
		ModelType:        "llama",
	})
}

func validVisionEmbedding(m *Model, n int) *VisionEmbedding {
	vecs := make([][]float32, n)
	for i := range vecs {
		vecs[i] = make([]float32, m.Cfg.HiddenSize)
		for j := range vecs[i] {
			vecs[i][j] = float32(i+1) * 0.01 * float32(j+1)
		}
	}
	return &VisionEmbedding{
		Image: MultimodalImage{
			MediaType: "image/png",
			Bytes:     []byte{0x89, 'P', 'N', 'G'},
			Width:     32,
			Height:    24,
		},
		Vectors: vecs,
	}
}

func withImageBytes(img *VisionEmbedding, b []byte) *VisionEmbedding {
	cp := *img
	cp.Image.Bytes = b
	return &cp
}

func withImageSize(img *VisionEmbedding, w, h int) *VisionEmbedding {
	cp := *img
	cp.Image.Width = w
	cp.Image.Height = h
	return &cp
}

func withMediaType(img *VisionEmbedding, media string) *VisionEmbedding {
	cp := *img
	cp.Image.MediaType = media
	return &cp
}

func assertActivationsBitsEqual(t *testing.T, got, want *Activations) {
	t.Helper()
	if got.Seq != want.Seq || len(got.Hidden) != len(want.Hidden) || len(got.Logits) != len(want.Logits) {
		t.Fatalf("activation shape got seq=%d hidden=%d logits=%d want seq=%d hidden=%d logits=%d",
			got.Seq, len(got.Hidden), len(got.Logits), want.Seq, len(want.Hidden), len(want.Logits))
	}
	for l := range got.Hidden {
		if len(got.Hidden[l]) != len(want.Hidden[l]) {
			t.Fatalf("hidden[%d] len=%d want %d", l, len(got.Hidden[l]), len(want.Hidden[l]))
		}
		for i := range got.Hidden[l] {
			if math.Float32bits(got.Hidden[l][i]) != math.Float32bits(want.Hidden[l][i]) {
				t.Fatalf("hidden[%d][%d]=%v want %v", l, i, got.Hidden[l][i], want.Hidden[l][i])
			}
		}
	}
	for tpos := range got.Logits {
		if len(got.Logits[tpos]) != len(want.Logits[tpos]) {
			t.Fatalf("logits[%d] len=%d want %d", tpos, len(got.Logits[tpos]), len(want.Logits[tpos]))
		}
		for i := range got.Logits[tpos] {
			if math.Float32bits(got.Logits[tpos][i]) != math.Float32bits(want.Logits[tpos][i]) {
				t.Fatalf("logits[%d][%d]=%v want %v", tpos, i, got.Logits[tpos][i], want.Logits[tpos][i])
			}
		}
	}
}
