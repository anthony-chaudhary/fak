package model

import (
	"errors"
	"math"
	"strings"
	"testing"
)

// TestMultimodalSpliceImagePadsPlacesEncoderRowsAtPlaceholders is the seam's done
// condition: a template prompt carrying ONE <|image_pad|> per image expands to the
// encoder's token count, splices into ordered parts, and reaches the decoder as a row
// sequence whose length is the expanded prompt length with each image's rows bit-exact at
// its own placeholder offsets — in prompt order, not merely in the right quantity.
func TestMultimodalSpliceImagePadsPlacesEncoderRowsAtPlaceholders(t *testing.T) {
	m := multimodalTestModel()
	const pad = 7

	imgA := validVisionEmbedding(m, 2)
	imgB := offsetVisionEmbedding(validVisionEmbedding(m, 3), 0.5)

	template := []int{1, 2, pad, 3, pad, 4}
	counts := []int{len(imgA.Vectors), len(imgB.Vectors)}

	expanded, err := ExpandImagePads(template, pad, counts)
	if err != nil {
		t.Fatalf("ExpandImagePads: %v", err)
	}
	// 6 template ids - 2 placeholders + (2 + 3) image rows = 9.
	if len(expanded) != 9 {
		t.Fatalf("expanded length = %d, want 9 (got %v)", len(expanded), expanded)
	}

	req, err := SpliceImagePads(expanded, pad, []*VisionEmbedding{imgA, imgB}, MultimodalPolicy{Mode: MultimodalModeQuarantine})
	if err != nil {
		t.Fatalf("SpliceImagePads: %v", err)
	}
	// text[1,2] | imgA | text[3] | imgB | text[4]
	if len(req.Parts) != 5 {
		t.Fatalf("parts = %d, want 5 (text/image/text/image/text)", len(req.Parts))
	}
	if req.Parts[1].Image != imgA || req.Parts[3].Image != imgB {
		t.Fatalf("image parts out of prompt order: got %p,%p want %p,%p",
			req.Parts[1].Image, req.Parts[3].Image, imgA, imgB)
	}

	rows, verdict, err := m.prepareMultimodalRows(req)
	if err != nil {
		t.Fatalf("prepareMultimodalRows: %v", err)
	}
	if len(rows) != len(expanded) {
		t.Fatalf("spliced rows = %d, want %d (one row per expanded prompt position)", len(rows), len(expanded))
	}
	if verdict.Decision != MultimodalAllow || verdict.Images != 2 || verdict.EmbeddingTokens != 5 {
		t.Fatalf("verdict = %+v, want allow with 2 images and 5 embedding tokens", verdict)
	}

	// imgA occupies expanded positions 2..3, imgB positions 5..7.
	assertRowsBitEqual(t, rows, 2, imgA.Vectors, "imgA")
	assertRowsBitEqual(t, rows, 5, imgB.Vectors, "imgB")

	act, _, err := m.ForwardMultimodal(req)
	if err != nil {
		t.Fatalf("ForwardMultimodal: %v", err)
	}
	if act.Seq != len(expanded) {
		t.Fatalf("forward seq = %d, want %d", act.Seq, len(expanded))
	}
}

// TestMultimodalSpliceImagePadsRejectsTokenCountMismatch pins the reconciliation itself:
// a placeholder run that does not match the encoder's row count for that image is a typed
// error naming both counts — never a silent truncation or pad.
func TestMultimodalSpliceImagePadsRejectsTokenCountMismatch(t *testing.T) {
	m := multimodalTestModel()
	const pad = 7

	// Prompt expanded for a 2-token image, but the encoder yielded 3 rows.
	ids := []int{1, pad, pad, 2}
	img := validVisionEmbedding(m, 3)

	_, err := SpliceImagePads(ids, pad, []*VisionEmbedding{img}, MultimodalPolicy{Mode: MultimodalModeQuarantine})
	if !errors.Is(err, ErrVisionSplice) {
		t.Fatalf("error = %v, want ErrVisionSplice", err)
	}
	if !strings.Contains(err.Error(), "2 placeholder") || !strings.Contains(err.Error(), "3 image token") {
		t.Fatalf("error %q must name both the placeholder count and the encoder token count", err)
	}
}

// TestMultimodalSpliceImagePadsRejectsImageCountMismatch covers the other axis: the number
// of placeholder RUNS must equal the number of encoded images, in both directions.
func TestMultimodalSpliceImagePadsRejectsImageCountMismatch(t *testing.T) {
	m := multimodalTestModel()
	const pad = 7
	policy := MultimodalPolicy{Mode: MultimodalModeQuarantine}

	// Two runs, one image.
	twoRuns := []int{pad, 1, pad}
	if _, err := SpliceImagePads(twoRuns, pad, []*VisionEmbedding{validVisionEmbedding(m, 1)}, policy); !errors.Is(err, ErrVisionSplice) {
		t.Fatalf("two runs / one image: error = %v, want ErrVisionSplice", err)
	}

	// One run, two images.
	oneRun := []int{1, pad, 2}
	imgs := []*VisionEmbedding{validVisionEmbedding(m, 1), validVisionEmbedding(m, 1)}
	if _, err := SpliceImagePads(oneRun, pad, imgs, policy); !errors.Is(err, ErrVisionSplice) {
		t.Fatalf("one run / two images: error = %v, want ErrVisionSplice", err)
	}

	// Images supplied against a vocab with no placeholder at all.
	if _, err := SpliceImagePads([]int{1, 2}, -1, imgs[:1], policy); !errors.Is(err, ErrVisionSplice) {
		t.Fatalf("no placeholder in vocab: error = %v, want ErrVisionSplice", err)
	}
}

// TestMultimodalExpandImagePadsExpandsPerImageTokenCount pins the template-side half: one
// placeholder per image becomes NumImageTokens copies, the caller's ids are never mutated,
// and a prompt/image count disagreement is refused before any splice is attempted.
func TestMultimodalExpandImagePadsExpandsPerImageTokenCount(t *testing.T) {
	const pad = 7
	ids := []int{1, pad, 2, pad, 3}
	orig := append([]int(nil), ids...)

	got, err := ExpandImagePads(ids, pad, []int{2, 3})
	if err != nil {
		t.Fatalf("ExpandImagePads: %v", err)
	}
	want := []int{1, pad, pad, 2, pad, pad, pad, 3}
	if len(got) != len(want) {
		t.Fatalf("expanded = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("expanded = %v, want %v", got, want)
		}
	}
	for i := range orig {
		if ids[i] != orig[i] {
			t.Fatalf("ExpandImagePads mutated the caller's ids at %d: %v want %v", i, ids, orig)
		}
	}

	if _, err := ExpandImagePads(ids, pad, []int{2}); !errors.Is(err, ErrVisionSplice) {
		t.Fatalf("count mismatch: error = %v, want ErrVisionSplice", err)
	}
	if _, err := ExpandImagePads(ids, pad, []int{2, 0}); !errors.Is(err, ErrVisionSplice) {
		t.Fatalf("zero-token image: error = %v, want ErrVisionSplice", err)
	}
}

// TestMultimodalSpliceImagePadsTextOnlyMatchesForward proves the seam is inert for text:
// a prompt with no placeholders and no images forwards bit-for-bit identically to the
// plain text path, so wiring vision in cannot perturb a text-only model.
func TestMultimodalSpliceImagePadsTextOnlyMatchesForward(t *testing.T) {
	m := multimodalTestModel()
	ids := []int{1, 2, 3}

	for _, padID := range []int{7, -1} {
		req, err := SpliceImagePads(ids, padID, nil, MultimodalPolicy{})
		if err != nil {
			t.Fatalf("padID %d: SpliceImagePads: %v", padID, err)
		}
		if len(req.Parts) != 1 || req.Parts[0].Image != nil || len(req.Parts[0].TokenIDs) != 3 {
			t.Fatalf("padID %d: parts = %+v, want a single 3-token text part", padID, req.Parts)
		}
		got, verdict, err := m.ForwardMultimodal(req)
		if err != nil {
			t.Fatalf("padID %d: ForwardMultimodal: %v", padID, err)
		}
		if verdict.Images != 0 || verdict.EmbeddingTokens != 0 {
			t.Fatalf("padID %d: verdict = %+v, want no image accounting", padID, verdict)
		}
		assertActivationsBitsEqual(t, got, m.Forward(ids))
	}
}

// offsetVisionEmbedding shifts every component so two embeddings built by
// validVisionEmbedding are distinguishable — which is what makes an ORDER assertion real.
func offsetVisionEmbedding(e *VisionEmbedding, d float32) *VisionEmbedding {
	for _, v := range e.Vectors {
		for i := range v {
			v[i] += d
		}
	}
	return e
}

func assertRowsBitEqual(t *testing.T, rows [][]float32, at int, want [][]float32, label string) {
	t.Helper()
	for i, w := range want {
		got := rows[at+i]
		if len(got) != len(w) {
			t.Fatalf("%s row %d width = %d, want %d", label, i, len(got), len(w))
		}
		for j := range w {
			if math.Float32bits(got[j]) != math.Float32bits(w[j]) {
				t.Fatalf("%s row %d component %d = %v, want %v (image rows must splice unchanged)", label, i, j, got[j], w[j])
			}
		}
	}
}
