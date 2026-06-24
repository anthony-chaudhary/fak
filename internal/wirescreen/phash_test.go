package wirescreen

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// phash_test.go covers the rung-3 perceptual-hash frame-dedup proposer (issue #571) at
// the unit level: the hash is stable for identical frames, separates perceptually
// different frames, the digester dedups a re-send and declines a new frame, and the
// image-block decoder handles the wire shapes a screenshot tool emits. The load-bearing
// reversibility witness (through ctxmmu.MMU.Admit) lives in internal/ctxmmu.

// phashTestPNG renders a small grayscale PNG of a distinct structural variant so two
// different variants hash far apart (the low-frequency DCT block differs strongly).
func phashTestPNG(variant int) []byte {
	const dim = 64
	img := image.NewGray(image.Rect(0, 0, dim, dim))
	for y := 0; y < dim; y++ {
		for x := 0; x < dim; x++ {
			switch variant {
			case 0: // horizontal edge: bright top, dark bottom
				if y < dim/2 {
					img.SetGray(x, y, color.Gray{Y: 240})
				} else {
					img.SetGray(x, y, color.Gray{Y: 16})
				}
			case 1: // vertical edge: bright left, dark right
				if x < dim/2 {
					img.SetGray(x, y, color.Gray{Y: 240})
				} else {
					img.SetGray(x, y, color.Gray{Y: 16})
				}
			case 2: // 8x8 checkerboard (high-frequency content)
				if (x/8+y/8)%2 == 0 {
					img.SetGray(x, y, color.Gray{Y: 240})
				} else {
					img.SetGray(x, y, color.Gray{Y: 16})
				}
			}
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic(err)
	}
	return buf.Bytes()
}

// TestPerceptualHashStableAndSeparating proves the pHash is deterministic (identical
// frames hash equal) and discriminates perceptually different frames well beyond the
// dedup threshold — the property the dedup pointer and the one-sided fall-through rely on.
func TestPerceptualHashStableAndSeparating(t *testing.T) {
	a := phashTestPNG(0)
	b := phashTestPNG(1) // rotated edge
	c := phashTestPNG(2) // checkerboard

	imgA, _ := tryDecodeImage(a)
	h0, ok := perceptualHash(imgA)
	if !ok {
		t.Fatalf("perceptualHash of a real image declined")
	}
	// A second decode of the same PNG hashes identically (determinism).
	imgA2, _ := tryDecodeImage(phashTestPNG(0))
	h0b, _ := perceptualHash(imgA2)
	if h0 != h0b {
		t.Fatalf("identical frames hashed differently: %064b vs %064b", h0, h0b)
	}
	// Perceptually different frames hash well beyond the dedup threshold.
	for name, pngb := range map[string][]byte{"v-edge": b, "checker": c} {
		img, _ := tryDecodeImage(pngb)
		hh, _ := perceptualHash(img)
		if d := hamming(h0, hh); d <= phashDedupThreshold {
			t.Errorf("variant %s is only %d bits from variant 0 (threshold %d) — pHash fails to separate distinct frames", name, d, phashDedupThreshold)
		}
	}
}

// TestPerceptualHashDeclinesTooSmall proves a zero-size image declines (no hash), so the
// digester never crashes on a malformed/empty decode.
func TestPerceptualHashDeclinesTooSmall(t *testing.T) {
	if _, ok := perceptualHash(image.NewRGBA(image.Rect(0, 0, 0, 0))); ok {
		t.Fatalf("a zero-size image must decline")
	}
}

// TestPhashDigesterDedupsIdenticalAndDeclinesNew proves the proposer's contract: a new
// frame is remembered but not summarized (the full frame is strictly better the first
// time); a re-send of a seen frame collapses to "unchanged, see frame#k"; a perceptually
// different frame declines (one-sided, never a new refusal); a non-image declines.
func TestPhashDigesterDedupsIdenticalAndDeclinesNew(t *testing.T) {
	resetPhashStoreForTest()
	defer resetPhashStoreForTest()
	d := phashDigester{}
	ctx := context.Background()

	// Non-image body declines.
	if _, ok := d.Summarize(ctx, []byte(`{"rows":[1,2,3]}`), "screenshot"); ok {
		t.Errorf("non-image body must decline (ok=false)")
	}

	// A brand-new frame is remembered but NOT summarized.
	bodyA := phashTestPNG(0)
	if _, ok := d.Summarize(ctx, bodyA, "screenshot"); ok {
		t.Errorf("a first-seen frame must decline (it is new), got a summary")
	}
	// The SAME frame re-sent collapses to a dedup pointer naming frame #1.
	got, ok := d.Summarize(ctx, bodyA, "screenshot")
	if !ok {
		t.Fatalf("re-send of a seen frame must dedup, got ok=false")
	}
	if got != "unchanged, see frame#1" {
		t.Errorf("dedup pointer = %q, want %q", got, "unchanged, see frame#1")
	}

	// A perceptually DIFFERENT frame is new (declines) — strictly one-sided.
	bodyB := phashTestPNG(2)
	if _, ok := d.Summarize(ctx, bodyB, "screenshot"); ok {
		t.Errorf("a perceptually different frame must decline (new), got a summary")
	}
	// The original frame dedups to frame #1 again, not frame #2.
	got, ok = d.Summarize(ctx, bodyA, "screenshot")
	if !ok || got != "unchanged, see frame#1" {
		t.Errorf("re-send dedup = (%q,%v), want (\"unchanged, see frame#1\",true)", got, ok)
	}
}

// TestDedupsCounterIncrements proves the observability counter advances exactly once per
// dedup collapse and not on a decline (new frame / non-image).
func TestDedupsCounterIncrements(t *testing.T) {
	resetPhashStoreForTest()
	defer resetPhashStoreForTest()
	d := phashDigester{}
	ctx := context.Background()
	body := phashTestPNG(1)

	before := Dedups()
	if _, ok := d.Summarize(ctx, body, "screenshot"); ok {
		t.Fatalf("first-seen frame must not be a dedup")
	}
	if Dedups() != before {
		t.Errorf("Dedups() incremented on a new frame: %d -> %d", before, Dedups())
	}
	if _, ok := d.Summarize(ctx, body, "screenshot"); !ok {
		t.Fatalf("re-send must dedup")
	}
	if Dedups() != before+1 {
		t.Errorf("Dedups() = %d, want %d after one collapse", Dedups(), before+1)
	}
}

// TestDecodeImageBlockHandlesWireShapes proves the decoder accepts the shapes a screenshot
// tool actually emits: raw PNG/JPEG bytes, a base64 encoding, and an Anthropic/OpenAI JSON
// image content block; and declines plain text.
func TestDecodeImageBlockHandlesWireShapes(t *testing.T) {
	raw := phashTestPNG(0)

	if decodeImageBlock(raw) == nil {
		t.Errorf("raw PNG bytes must decode")
	}
	if decodeImageBlock([]byte(base64.StdEncoding.EncodeToString(raw))) == nil {
		t.Errorf("base64-encoded PNG must decode")
	}
	// Anthropic image block.
	anthropic := []byte(`[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` +
		base64.StdEncoding.EncodeToString(raw) + `"}}]`)
	if decodeImageBlock(anthropic) == nil {
		t.Errorf("Anthropic image content block must decode")
	}
	// OpenAI image_url with a data URI.
	openai := []byte(`[{"type":"image_url","image_url":{"url":"data:image/png;base64,` +
		base64.StdEncoding.EncodeToString(raw) + `"}}]`)
	if decodeImageBlock(openai) == nil {
		t.Errorf("OpenAI image_url data-URI block must decode")
	}
	// Plain text declines (not an image).
	if decodeImageBlock([]byte(`the quick brown fox`)) != nil {
		t.Errorf("plain text must not decode as an image")
	}
}

// TestPhashScreenBridgesToScreenDigest proves the exported screen seam maps a dedup hit to
// a ScreenDigest advisory carrying the dedup pointer, and is inert (ScreenAllow) otherwise.
func TestPhashScreenBridgesToScreenDigest(t *testing.T) {
	resetPhashStoreForTest()
	defer resetPhashStoreForTest()
	ctx := context.Background()
	body := phashTestPNG(0)

	scr := PhashScreen()
	if scr == nil {
		t.Fatalf("PhashScreen() returned nil")
	}
	// First sight: inert (new frame, no advisory).
	if adv := scr.ScreenResult(ctx, nil, body); adv.Disposition != abi.ScreenAllow {
		t.Errorf("first sight: want ScreenAllow (no opinion), got disposition %d", adv.Disposition)
	}
	// Re-send: ScreenDigest carrying the dedup pointer.
	adv := scr.ScreenResult(ctx, nil, body)
	if adv.Disposition != abi.ScreenDigest {
		t.Fatalf("re-send: want ScreenDigest, got disposition %d", adv.Disposition)
	}
	if adv.Digest != "unchanged, see frame#1" {
		t.Errorf("ScreenDigest carries %q, want %q", adv.Digest, "unchanged, see frame#1")
	}
	if adv.By == "" {
		t.Errorf("advisory carries no By (observability)")
	}
}
