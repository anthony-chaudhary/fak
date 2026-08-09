package model

// The companion-mmproj join (#4875). vision_prompt_test.go drives the image+text path on a
// model whose tower is already resident; this file pins how a SEPARATELY-loaded tower — the
// mmproj GGUF half, which ggufload.OpenMMProj/WeightSource.VisionTower() produces — becomes
// resident in the first place, and what attach refuses.
//
// The property under test is that attach and the encoder cannot disagree: attach admits a
// tower only by building the encoder, so "attached" and "encodable" are the same predicate.

import (
	"strings"
	"testing"
)

// attachableTower is a tiny but complete tower whose projector width matches m's decoder —
// the shape a matching mmproj produces.
func attachableTower(t *testing.T, m *Model) *VisionTower {
	t.Helper()
	return buildVisionTower(t, visionFixture{
		hidden: m.Cfg.HiddenSize, heads: 2, layers: 1, patch: 2, merge: 2,
		temporal: 1, ffn: 16, decoderHidden: m.Cfg.HiddenSize, twoLayerProj: false,
	})
}

// TestMultimodalAttachVisionTowerMakesTheImagePathReachable is the seam's reason to exist:
// a freshly loaded text model refuses to encode an image, and attaching the companion tower
// — with no RetainVision involved — is what turns that refusal into a working forward.
func TestMultimodalAttachVisionTowerMakesTheImagePathReachable(t *testing.T) {
	m := multimodalTestModel()
	if m.Vision != nil {
		t.Fatalf("a text-only model must start with no vision tower")
	}
	if _, err := m.NewVisionEncoder(); err == nil {
		t.Fatalf("NewVisionEncoder on a text-only model: want an error, got nil")
	}

	tower := attachableTower(t, m)
	if err := m.AttachVisionTower(tower); err != nil {
		t.Fatalf("AttachVisionTower: %v", err)
	}
	if m.Vision != tower {
		t.Fatalf("attach must share the tower, not copy it")
	}

	// The production entry point now builds, and the full image+text path runs through it.
	if _, err := m.NewVisionEncoder(); err != nil {
		t.Fatalf("NewVisionEncoder after attach: %v", err)
	}
	// The exported interface carries only EncodeImage, so read the expected row count from
	// the same concrete constructor attach validated with.
	enc, err := newVisionEncoder(m.Vision, m.Cfg.HiddenSize)
	if err != nil {
		t.Fatalf("newVisionEncoder after attach: %v", err)
	}
	const imgW, imgH = 16, 12
	want := enc.NumImageTokens(imgW, imgH)
	if want <= 0 {
		t.Fatalf("fixture yields %d image tokens; the test needs a non-empty image", want)
	}

	const pad = 7
	img := MultimodalImage{MediaType: "image/png", Bytes: synthPNG(t, imgW, imgH), Width: imgW, Height: imgH}
	req, _, err := m.EncodeImagePrompt([]int{1, 2, pad, 3}, pad, []MultimodalImage{img}, MultimodalPolicy{Mode: MultimodalModeQuarantine})
	if err != nil {
		t.Fatalf("EncodeImagePrompt after attach: %v", err)
	}
	var got int
	for _, p := range req.Parts {
		if p.Image != nil {
			got += len(p.Image.Vectors)
		}
	}
	if got != want {
		t.Fatalf("spliced %d image row(s), want NumImageTokens=%d", got, want)
	}
}

// TestMultimodalAttachVisionTowerRefusesMismatchedProjector is the load-time-vs-request-time
// property: a tower whose projector width does not match the decoder is refused BY ATTACH,
// carrying the encoder's own diagnosis, instead of being accepted and failing later at the
// first image.
func TestMultimodalAttachVisionTowerRefusesMismatchedProjector(t *testing.T) {
	m := multimodalTestModel()
	// decoderHidden one wider than the decoder: mm.* projects to the wrong width.
	bad := buildVisionTower(t, visionFixture{
		hidden: m.Cfg.HiddenSize, heads: 2, layers: 1, patch: 2, merge: 2,
		temporal: 1, ffn: 16, decoderHidden: m.Cfg.HiddenSize + 1, twoLayerProj: false,
	})
	err := m.AttachVisionTower(bad)
	if err == nil {
		t.Fatalf("AttachVisionTower with a mismatched projector: want an error, got nil")
	}
	if !strings.Contains(err.Error(), "would not splice") {
		t.Fatalf("attach error must carry the encoder's diagnosis, got: %v", err)
	}
	if m.Vision != nil {
		t.Fatalf("a refused attach must leave Model.Vision nil, got a tower")
	}
	// The refusal and the encoder agree, which is the point of validating by construction.
	if _, encErr := newVisionEncoder(bad, m.Cfg.HiddenSize); encErr == nil {
		t.Fatalf("newVisionEncoder accepted a tower attach refused — the two must not disagree")
	}
}

// TestMultimodalAttachVisionTowerIsOneShot pins that a second attach is refused rather than
// silently re-pointing the tower a server is already encoding against.
func TestMultimodalAttachVisionTowerIsOneShot(t *testing.T) {
	m := multimodalTestModel()
	first := attachableTower(t, m)
	if err := m.AttachVisionTower(first); err != nil {
		t.Fatalf("first AttachVisionTower: %v", err)
	}
	err := m.AttachVisionTower(attachableTower(t, m))
	if err == nil {
		t.Fatalf("second AttachVisionTower: want an error, got nil")
	}
	if !strings.Contains(err.Error(), "already carries") {
		t.Fatalf("second attach must name the conflict, got: %v", err)
	}
	if m.Vision != first {
		t.Fatalf("a refused second attach must leave the FIRST tower in place")
	}
}

// TestMultimodalAttachVisionTowerRejectsNil keeps the nil case a named error rather than a
// nil-pointer panic or a silently-detached model.
func TestMultimodalAttachVisionTowerRejectsNil(t *testing.T) {
	m := multimodalTestModel()
	if err := m.AttachVisionTower(nil); err == nil {
		t.Fatalf("AttachVisionTower(nil): want an error, got nil")
	}
	if m.Vision != nil {
		t.Fatalf("a refused attach must leave Model.Vision nil")
	}
}
