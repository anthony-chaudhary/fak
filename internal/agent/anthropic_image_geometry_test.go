package agent

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"testing"
)

// encodeTestImage renders a w-by-h image in the named container using the standard library's own
// encoders, so the geometry sniffer is proven against real encoder output rather than a hand-rolled
// fixture that could encode my own misreading of the format.
func encodeTestImage(t *testing.T, format string, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	// A flat fill keeps JPEG's entropy-coded payload small; the frame header is what matters.
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: 0x40, A: 0xFF})
		}
	}
	var buf bytes.Buffer
	var err error
	switch format {
	case "png":
		err = png.Encode(&buf, img)
	case "jpeg":
		err = jpeg.Encode(&buf, img, nil)
	case "gif":
		err = gif.Encode(&buf, img, nil)
	default:
		t.Fatalf("unknown test image format %q", format)
	}
	if err != nil {
		t.Fatalf("encode %s %dx%d: %v", format, w, h, err)
	}
	return buf.Bytes()
}

// realImageBlock builds an Anthropic image content block carrying a genuinely encoded image.
func realImageBlock(t *testing.T, format string, w, h int) json.RawMessage {
	t.Helper()
	raw := encodeTestImage(t, format, w, h)
	blk, err := json.Marshal(map[string]any{
		"type": "image",
		"source": map[string]any{
			"type":       "base64",
			"media_type": "image/" + format,
			"data":       base64.StdEncoding.EncodeToString(raw),
		},
	})
	if err != nil {
		t.Fatalf("marshal image block: %v", err)
	}
	return blk
}

// wantGeometricCost is the cost the issue's formula prescribes: ~(w*h)/750, capped at the flat
// ceiling, and never zero for a real image.
func wantGeometricCost(w, h int) int {
	c := (w * h) / imageTokenPixelsPerToken
	if c >= imageTokenCost {
		return imageTokenCost
	}
	if c < 1 {
		return 1
	}
	return c
}

// TestImageBlockTokenCostPricesByGeometry is the core of #5165: a thumbnail and a full-page
// screenshot must no longer cost the same. Before the fix both were charged the flat 1600.
func TestImageBlockTokenCostPricesByGeometry(t *testing.T) {
	for _, format := range []string{"png", "jpeg", "gif"} {
		t.Run(format, func(t *testing.T) {
			thumb := imageBlockTokenCost(realImageBlock(t, format, 64, 64))
			if want := wantGeometricCost(64, 64); thumb != want {
				t.Fatalf("64x64 %s thumbnail cost %d tokens; want %d — it is not being priced by geometry", format, thumb, want)
			}
			shot := imageBlockTokenCost(realImageBlock(t, format, 800, 600))
			if want := wantGeometricCost(800, 600); shot != want {
				t.Fatalf("800x600 %s screenshot cost %d tokens; want %d", format, shot, want)
			}
			if thumb >= shot {
				t.Fatalf("thumbnail (%d) is not cheaper than the screenshot (%d) — geometry is being ignored", thumb, shot)
			}
			if thumb >= imageTokenCost {
				t.Fatalf("thumbnail still charged the flat ceiling (%d); the whole point of #5165 is that it should not be", thumb)
			}
		})
	}
}

// TestImageBlockTokenCostCapsAtCeiling pins the cap: a large image prices above the ceiling by the
// raw formula and must be clamped to imageTokenCost, never charged more than the provider's max.
func TestImageBlockTokenCostCapsAtCeiling(t *testing.T) {
	// 1512x982 (a full retina page) is 1512*982/750 ≈ 1979 by the raw formula — above the ceiling.
	got := imageBlockTokenCost(realImageBlock(t, "png", 1512, 982))
	if got != imageTokenCost {
		t.Fatalf("oversized image cost %d tokens; want the ceiling %d", got, imageTokenCost)
	}
}

// TestImageBlockTokenCostFallsBackToFlatCeiling pins the safety property the issue insists on: when
// the geometry is NOT recoverable the image is charged the flat ceiling exactly as before, so the
// estimate can never under-count an image fak cannot measure.
func TestImageBlockTokenCostFallsBackToFlatCeiling(t *testing.T) {
	cases := []struct {
		name string
		blk  any
	}{
		{"opaque base64 that is not an image", map[string]any{
			"type":   "image",
			"source": map[string]any{"type": "base64", "media_type": "image/png", "data": bigBase64(4096)},
		}},
		{"url source carrying no dimensions", map[string]any{
			"type":   "image",
			"source": map[string]any{"type": "url", "url": "https://example.test/a.png"},
		}},
		{"truncated png header", map[string]any{
			"type":   "image",
			"source": map[string]any{"type": "base64", "media_type": "image/png", "data": base64.StdEncoding.EncodeToString([]byte("\x89PNG\r\n\x1a\n"))},
		}},
		{"absurd dimensions from a corrupt header", map[string]any{
			"type":   "image",
			"source": map[string]any{"type": "base64", "width": maxPlausibleImageDim + 1, "height": 10},
		}},
		{"zero dimensions", map[string]any{
			"type":   "image",
			"source": map[string]any{"type": "base64", "width": 0, "height": 0},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(tc.blk)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if got := imageBlockTokenCost(raw); got != imageTokenCost {
				t.Fatalf("unmeasurable image cost %d tokens; want the flat ceiling %d — the fallback floor is gone", got, imageTokenCost)
			}
		})
	}
}

// TestImageBlockTokenCostUsesExplicitDimensions pins the issue's first-listed source of geometry:
// when the block or its source states width/height, that is exact and needs no decode — and it is
// the ONLY geometry a URL source can offer.
func TestImageBlockTokenCostUsesExplicitDimensions(t *testing.T) {
	raw, err := json.Marshal(map[string]any{
		"type":   "image",
		"source": map[string]any{"type": "url", "url": "https://example.test/a.png", "width": 300, "height": 200},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if got, want := imageBlockTokenCost(raw), wantGeometricCost(300, 200); got != want {
		t.Fatalf("explicit-dimension image cost %d tokens; want %d", got, want)
	}
}

// TestImageBlockTokenCostWebP covers the one container the standard library cannot encode. The
// header is built by hand from the VP8X spec: "RIFF", size, "WEBP", "VP8X", chunk size, flags and
// reserved, then 24-bit little-endian (width-1) and (height-1).
func TestImageBlockTokenCostWebP(t *testing.T) {
	const w, h = 640, 480
	body := make([]byte, 30)
	copy(body[0:], "RIFF")
	copy(body[8:], "WEBP")
	copy(body[12:], "VP8X")
	body[16] = 10 // VP8X chunk payload size
	cw, ch := w-1, h-1
	body[24], body[25], body[26] = byte(cw), byte(cw>>8), byte(cw>>16)
	body[27], body[28], body[29] = byte(ch), byte(ch>>8), byte(ch>>16)

	raw, err := json.Marshal(map[string]any{
		"type":   "image",
		"source": map[string]any{"type": "base64", "media_type": "image/webp", "data": base64.StdEncoding.EncodeToString(body)},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if got, want := imageBlockTokenCost(raw), wantGeometricCost(w, h); got != want {
		t.Fatalf("webp image cost %d tokens; want %d", got, want)
	}
}

// TestEstimateElementTokensPricesThumbnailBelowCeiling is the byte-level half wired end to end: a
// compaction-path element carrying only a thumbnail must now estimate far below the flat ceiling,
// which is what stops fak shedding turns it never needed to shed in a thumbnail-heavy session.
func TestEstimateElementTokensPricesThumbnailBelowCeiling(t *testing.T) {
	el, err := json.Marshal(map[string]any{
		"role":    "user",
		"content": []json.RawMessage{realImageBlock(t, "png", 64, 64)},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	got := estimateElementTokens(json.RawMessage(el))
	if got >= imageTokenCost {
		t.Fatalf("thumbnail element estimated at %d tokens; the flat ceiling %d is still being charged", got, imageTokenCost)
	}
	if got < 0 {
		t.Fatalf("estimate went negative: %d", got)
	}
}

// TestEstimateAnthropicTokensPricesThumbnailByGeometry is the decoded half wired end to end: the
// count_tokens estimator must charge the same geometric cost the compaction path does, so the two
// estimators still agree on what a picture costs.
func TestEstimateAnthropicTokensPricesThumbnailByGeometry(t *testing.T) {
	body, err := json.Marshal(map[string]any{
		"model":      "claude-sonnet-4-6",
		"max_tokens": 1024,
		"messages": []map[string]any{
			{"role": "user", "content": []json.RawMessage{realImageBlock(t, "png", 64, 64)}},
		},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req, err := DecodeAnthropicMessagesRequest(body)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	got := EstimateAnthropicTokens(req)
	if want := wantGeometricCost(64, 64); got != want {
		t.Fatalf("image-only thumbnail request estimated at %d tokens; want exactly the geometric cost %d", got, want)
	}
}

// TestContentImageWeightPricesMixedTurnPerImage pins that a turn mixing a thumbnail and a large
// screenshot charges them DIFFERENTLY — the exact conflation the flat constant caused. It also
// covers the tool_result nesting, since a screenshot usually arrives from a tool.
func TestContentImageWeightPricesMixedTurnPerImage(t *testing.T) {
	content, err := json.Marshal([]any{
		realImageBlock(t, "png", 64, 64),
		map[string]any{
			"type":        "tool_result",
			"tool_use_id": "toolu_1",
			"content":     []json.RawMessage{realImageBlock(t, "png", 1512, 982)},
		},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	imgs, _, imgTokens := contentImageWeight(json.RawMessage(content))
	if imgs != 2 {
		t.Fatalf("counted %d images; want 2 (the tool_result nesting must be walked)", imgs)
	}
	want := wantGeometricCost(64, 64) + wantGeometricCost(1512, 982)
	if imgTokens != want {
		t.Fatalf("mixed turn charged %d tokens; want %d (thumbnail + capped screenshot priced separately)", imgTokens, want)
	}
	if imgTokens == 2*imageTokenCost {
		t.Fatalf("mixed turn charged 2x the flat ceiling — both images are still priced identically")
	}
}
