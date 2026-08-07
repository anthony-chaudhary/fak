package model

// vision_splice.go — the <|image_pad|> placeholder seam (#4875, the model half of #4032).
//
// Both sides of this seam already shipped, and each deliberately stopped at its own
// boundary: the tokenizer resolves the placeholder id from the loaded vocab and locates
// every occurrence (ImagePadSlots, #4031, which "computes no count and expands nothing"),
// the vision encoder turns pixels into one decoder-hidden-width vector per merged patch
// (EncodeImage / NumImageTokens, #4030), and ForwardMultimodal governs and forwards an
// ordered part list (multimodal.go). What neither owned — and the reason ForwardMultimodal
// had no production caller — is the join: turning ONE prompt id stream plus its encoded
// images into that ordered part list, and reconciling the placeholder count against the
// encoder's token count so an image cannot silently occupy the wrong number of positions.
//
// Layering: internal/model must not import internal/tokenizer (constraint.go:229), so the
// placeholder id arrives as a parameter — the caller reads it from its own vocab via
// tokenizer.ImagePadID(). Nothing here hardcodes a Qwen id, so any VLM vocab that marks
// image positions with a single repeated placeholder is served by the same seam.
//
// Run boundaries: one MAXIMAL run of padID is one image. Qwen-VL chat templates wrap each
// image as <|vision_start|> <|image_pad|>... <|vision_end|>, so consecutive images are
// always separated by their span markers and each gets its own run. Two images whose pads
// abut with no intervening token would read as a single run — that is reported as a token
// count mismatch rather than silently mis-splicing.

import (
	"errors"
	"fmt"
)

// ErrVisionSplice is the typed failure of placeholder<->image reconciliation: the prompt
// and the encoded images disagree about how many image positions exist, or about how many
// rows belong at one of them. It is deliberately distinct from ErrMultimodalDenied — this
// is a caller wiring/preprocessing fault, not a governance verdict.
var ErrVisionSplice = errors.New("model: vision splice")

// ExpandImagePads rewrites the ONE placeholder a chat template emits per image into the
// NumImageTokens copies that image actually occupies. counts[i] is the row count of the
// i'th image in prompt order — in practice len(VisionEmbedding.Vectors), which is exactly
// what the encoder's NumImageTokens(w,h) predicts for that image's dimensions.
//
// This is the step a reference VLM processor performs before the model ever sees the
// stream; fak keeps it explicit and separately witnessable. A prompt that is ALREADY
// expanded must not be passed through here — it would expand a second time; splice such a
// stream directly (SpliceImagePads verifies the counts either way).
//
// The returned slice is always a fresh allocation, so the caller's ids are never aliased
// or mutated.
func ExpandImagePads(ids []int, padID int, counts []int) ([]int, error) {
	if padID < 0 {
		if len(counts) > 0 {
			return nil, fmt.Errorf("%w: %d image(s) supplied but this vocab has no image placeholder", ErrVisionSplice, len(counts))
		}
		return append([]int(nil), ids...), nil
	}
	pads := 0
	for _, id := range ids {
		if id == padID {
			pads++
		}
	}
	if pads != len(counts) {
		return nil, fmt.Errorf("%w: prompt carries %d image placeholder(s) but %d image(s) were encoded", ErrVisionSplice, pads, len(counts))
	}
	if pads == 0 {
		return append([]int(nil), ids...), nil
	}
	total := len(ids) - pads
	for i, c := range counts {
		if c <= 0 {
			return nil, fmt.Errorf("%w: image %d expands to %d row(s); the encoder yields none for these dimensions", ErrVisionSplice, i, c)
		}
		total += c
	}
	out := make([]int, 0, total)
	slot := 0
	for _, id := range ids {
		if id != padID {
			out = append(out, id)
			continue
		}
		for j := 0; j < counts[slot]; j++ {
			out = append(out, padID)
		}
		slot++
	}
	return out, nil
}

// SpliceImagePads builds the MultimodalRequest for an expanded prompt: every maximal run
// of padID becomes one image part carrying that image's encoder rows, and the id spans
// between the runs become text parts, in stream order. images is in prompt order.
//
// The reconciliation this seam exists for is the run-length check: run i must be exactly
// len(images[i].Vectors) placeholders long. A mismatch is an error naming both counts,
// never a truncation or a pad — so a prompt whose template under- or over-expanded can
// never reach the decoder with images landing at the wrong positions.
//
// The returned request carries policy verbatim; images remain governed by ForwardMultimodal
// (mode, byte/pixel/embedding-token limits, content-type axis) exactly as before. Text
// spans are copied, so the caller's ids are never aliased; image parts intentionally share
// the caller's VisionEmbedding, matching the existing zero-copy part contract.
func SpliceImagePads(ids []int, padID int, images []*VisionEmbedding, policy MultimodalPolicy) (MultimodalRequest, error) {
	req := MultimodalRequest{Policy: policy}
	if padID < 0 {
		if len(images) > 0 {
			return MultimodalRequest{}, fmt.Errorf("%w: %d image(s) supplied but this vocab has no image placeholder", ErrVisionSplice, len(images))
		}
		if len(ids) > 0 {
			req.Parts = []MultimodalPart{{TokenIDs: append([]int(nil), ids...)}}
		}
		return req, nil
	}

	var parts []MultimodalPart
	textStart, i, slot := 0, 0, 0
	for i < len(ids) {
		if ids[i] != padID {
			i++
			continue
		}
		if i > textStart {
			parts = append(parts, MultimodalPart{TokenIDs: append([]int(nil), ids[textStart:i]...)})
		}
		run := 0
		for i < len(ids) && ids[i] == padID {
			run++
			i++
		}
		if slot >= len(images) {
			return MultimodalRequest{}, fmt.Errorf("%w: prompt carries more image placeholder runs than the %d image(s) supplied", ErrVisionSplice, len(images))
		}
		img := images[slot]
		if img == nil {
			return MultimodalRequest{}, fmt.Errorf("%w: image %d is nil", ErrVisionSplice, slot)
		}
		if got := len(img.Vectors); got != run {
			return MultimodalRequest{}, fmt.Errorf("%w: image %d occupies %d placeholder(s) but the encoder yielded %d image token(s)", ErrVisionSplice, slot, run, got)
		}
		parts = append(parts, MultimodalPart{Image: img})
		slot++
		textStart = i
	}
	if len(ids) > textStart {
		parts = append(parts, MultimodalPart{TokenIDs: append([]int(nil), ids[textStart:]...)})
	}
	if slot != len(images) {
		return MultimodalRequest{}, fmt.Errorf("%w: %d image(s) supplied but the prompt carries %d image placeholder run(s)", ErrVisionSplice, len(images), slot)
	}
	req.Parts = parts
	return req, nil
}
