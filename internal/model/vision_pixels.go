package model

// vision_pixels.go — the decode/preprocess wrapper that turns MultimodalImage.Bytes
// into the normalized pixel plane forwardPixels consumes (#4030).
//
// Layering (multimodal.go:94 — "concrete image decoders/encoders stay outside this
// text-forward core"): this file imports only the GENERIC image interface package, never
// a concrete image/png or image/jpeg decoder. It calls image.Decode, which dispatches to
// whatever formats the PROCESS has registered; the caller that owns the wire (serve /
// #4032, or a test) does the blank `import _ "image/png"` to register them. So the model
// package stays codec-free while EncodeImage is still a complete function once a decoder
// is registered upstream.
//
// The forward itself (patch-embed .. projector) is the witnessed core in
// vision_encoder.go. Everything here — the aspect-preserving resize to a patch*merge
// grid and the CLIP mean/std normalization — is the REFERENCE preprocessing: it is
// deterministic and unit-tested for shape/grid, but its exact policy (interpolation
// kernel, pixel budget, normalization stats) only matters for bitwise parity against a
// real mmproj, which is asset-gated. The constants are named and isolated so a real
// fixture can pin them without touching the forward.

import (
	"bytes"
	"fmt"
	"image"
	"math"
)

// CLIP/Qwen2-VL image normalization: per-channel (RGB) mean/std applied after scaling
// pixels to [0,1]. These are the OpenAI-CLIP defaults the Qwen2-VL processor uses.
var (
	visionPixelMean = [visionChannels]float32{0.48145466, 0.4578275, 0.40821073}
	visionPixelStd  = [visionChannels]float32{0.26862954, 0.26130258, 0.27577711}
)

// defaultVisionMaxEdge bounds the longer edge of the resized grid when the tower does
// not declare an image size, keeping the token count finite for a large upload.
const defaultVisionMaxEdge = 1288

// EncodeImage decodes an image's bytes, resizes to a patch*merge-aligned grid, normalizes,
// and runs the ViT forward, returning one decoder-hidden-width vector per merged patch.
// It errors when the bytes are empty or when no decoder for the payload is registered in
// the process (the caller that owns the wire registers image/png, image/jpeg, ...).
func (e *visionEncoder) EncodeImage(img MultimodalImage) (VisionEmbedding, error) {
	if len(img.Bytes) == 0 {
		return VisionEmbedding{}, fmt.Errorf("model: vision encoder: image has no bytes to decode (media %q)", img.MediaType)
	}
	src, _, err := image.Decode(bytes.NewReader(img.Bytes))
	if err != nil {
		return VisionEmbedding{}, fmt.Errorf("model: vision encoder: decode %q: %w (caller must register the codec, e.g. import _ \"image/png\")", img.MediaType, err)
	}
	px, rows, cols, err := e.preprocess(src)
	if err != nil {
		return VisionEmbedding{}, err
	}
	vecs, err := e.forwardPixels(px, rows, cols)
	if err != nil {
		return VisionEmbedding{}, err
	}
	return VisionEmbedding{Image: img, Vectors: vecs}, nil
}

// preprocess resizes src (aspect-preserving, nearest-neighbor) to a grid whose edges are
// multiples of patch*merge and bounded by the tower's image size, then returns the
// normalized CHW pixel plane and the patch grid (rows, cols).
func (e *visionEncoder) preprocess(src image.Image) (px []float32, gridRows, gridCols int, err error) {
	b := src.Bounds()
	w0, h0 := b.Dx(), b.Dy()
	if w0 <= 0 || h0 <= 0 {
		return nil, 0, 0, fmt.Errorf("model: vision encoder: decoded image has empty bounds %dx%d", w0, h0)
	}
	unit := e.patch * e.merge
	maxEdge := e.tower.Cfg.ImageSize
	if maxEdge < unit {
		maxEdge = defaultVisionMaxEdge
	}
	maxEdge -= maxEdge % unit // floor to a whole number of grid units
	if maxEdge < unit {
		maxEdge = unit
	}

	scale := 1.0
	if w0 > maxEdge || h0 > maxEdge {
		if w0 >= h0 {
			scale = float64(maxEdge) / float64(w0)
		} else {
			scale = float64(maxEdge) / float64(h0)
		}
	}
	newW := roundToUnit(int(math.Round(float64(w0)*scale)), unit)
	newH := roundToUnit(int(math.Round(float64(h0)*scale)), unit)

	px = make([]float32, visionChannels*newH*newW)
	plane := newH * newW
	for y := 0; y < newH; y++ {
		sy := b.Min.Y + y*h0/newH
		for x := 0; x < newW; x++ {
			sx := b.Min.X + x*w0/newW
			r16, g16, b16, _ := src.At(sx, sy).RGBA() // 16-bit, alpha-premultiplied
			ch := [visionChannels]uint32{r16, g16, b16}
			for c := 0; c < visionChannels; c++ {
				v := float32(ch[c]>>8) / 255.0
				px[c*plane+y*newW+x] = (v - visionPixelMean[c]) / visionPixelStd[c]
			}
		}
	}
	return px, newH / e.patch, newW / e.patch, nil
}

// roundToUnit rounds v to the nearest positive multiple of unit (minimum one unit).
func roundToUnit(v, unit int) int {
	if unit <= 0 {
		return v
	}
	r := ((v + unit/2) / unit) * unit
	if r < unit {
		r = unit
	}
	return r
}
