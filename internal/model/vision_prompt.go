package model

// vision_prompt.go — the production entry point for an image+text prompt (#4875).
//
// Every piece of the vision path already shipped and each stopped at its own boundary:
// the tower forward turns pixels into decoder-width rows (EncodeImage / NumImageTokens,
// #4030), the tokenizer locates the placeholders (ImagePadSlots, #4031), vision_splice.go
// reconciles placeholder runs against those row counts, and ForwardMultimodal governs and
// forwards an ordered part list (multimodal.go). What no file owned is the CHOREOGRAPHY —
// build the encoder, encode each image, collect the row counts, expand the template's one
// placeholder per image, splice, forward — which is why #4875 reports that EncodeImage,
// NewVisionEncoder and ForwardMultimodal have zero production callers. A serve path had to
// re-derive all six steps at every call site, so none did.
//
// This file is that choreography, once, on the Model: a template-shaped prompt plus raw
// image bytes in, a governed *Activations out. The remaining serve wiring (#4032, the
// gateway leaf) becomes an adapter that supplies ids/padID/images, not a reimplementation
// of the vision path.
//
// Admission ordering is the property worth naming. The image count, media type, dimension
// and byte bounds — and the quarantine opt-in — are evaluated BEFORE a single pixel is
// decoded or any ViT block runs, in exactly the precedence ForwardMultimodal uses (bounds
// first, then mode), by calling the same admitImageMetadata half of the same gate. An
// oversized or unadmitted image therefore costs a metadata check, never a decode plus a
// full tower forward, and it cannot earn one verdict here and a different one there.
//
// Layering is unchanged: nothing here imports a tokenizer or a concrete image codec. The
// placeholder id arrives as a parameter (internal/model must not import internal/tokenizer,
// constraint.go:229) and image.Decode dispatches to whatever formats the process
// registered (vision_pixels.go), so the caller that owns the wire keeps both dependencies.

import "fmt"

// EncodeImagePrompt turns a TEMPLATE-shaped prompt plus its raw images into the
// MultimodalRequest the decoder consumes: it admits the images under policy, encodes each
// one with this model's retained vision tower, expands the single placeholder the chat
// template emitted per image into that image's own token count, and splices the encoder
// rows at those positions in prompt order.
//
// ids is the template stream — ONE padID per image, as a Qwen-VL-style
// <|vision_start|><|image_pad|><|vision_end|> wrapper produces. A stream that is ALREADY
// expanded must not be passed here; it is refused by name (the placeholder count will not
// equal the image count) rather than expanded a second time. Such a caller splices
// directly with SpliceImagePads.
//
// images is in prompt order. A text-only call (no images) needs no vision tower at all, so
// a text-only model keeps working through this entry point unchanged.
//
// The returned verdict is the pre-encode accounting (images and bytes admitted); the
// embedding-token accounting is completed by ForwardMultimodal when the request is
// forwarded.
func (m *Model) EncodeImagePrompt(ids []int, padID int, images []MultimodalImage, policy MultimodalPolicy) (MultimodalRequest, MultimodalVerdict, error) {
	verdict, err := admitImagesBeforeEncode(images, policy)
	if err != nil {
		return MultimodalRequest{}, verdict, err
	}
	if len(images) == 0 {
		if policy.TemplateAssociation == VisionTemplateAssociationDeepInfer {
			req, err := associateDeepInferImagePads(ids, padID, nil, policy)
			return req, verdict, err
		}
		req, err := SpliceImagePads(ids, padID, nil, policy)
		return req, verdict, err
	}
	if err := m.Cfg.RefuseDeepSeekV41Vision(); err != nil {
		verdict.Decision = MultimodalDeny
		verdict.Reason = err.Error()
		return MultimodalRequest{}, verdict, err
	}

	enc, err := m.NewVisionEncoder()
	if err != nil {
		return MultimodalRequest{}, verdict, err
	}
	embeds := make([]*VisionEmbedding, len(images))
	counts := make([]int, len(images))
	for i, img := range images {
		emb, err := enc.EncodeImage(img)
		if err != nil {
			return MultimodalRequest{}, verdict, fmt.Errorf("model: image %d: %w", i, err)
		}
		row := emb
		embeds[i] = &row
		counts[i] = len(row.Vectors)
	}

	if policy.TemplateAssociation == VisionTemplateAssociationDeepInfer {
		req, err := associateDeepInferImagePads(ids, padID, embeds, policy)
		if err != nil {
			return MultimodalRequest{}, verdict, err
		}
		return req, verdict, nil
	}

	expanded, err := ExpandImagePads(ids, padID, counts)
	if err != nil {
		return MultimodalRequest{}, verdict, err
	}
	req, err := SpliceImagePads(expanded, padID, embeds, policy)
	if err != nil {
		return MultimodalRequest{}, verdict, err
	}
	return req, verdict, nil
}

// associateDeepInferImagePads implements DeepInfer's template association rule: image i
// replaces placeholder i, while surplus placeholders remain ordinary text tokens. More
// images than placeholders is still a typed wiring error. Every text span is copied so
// neither a size-only/no-image request nor a surplus-placeholder request aliases ids.
func associateDeepInferImagePads(ids []int, padID int, images []*VisionEmbedding, policy MultimodalPolicy) (MultimodalRequest, error) {
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

	parts := make([]MultimodalPart, 0, 2*len(images)+1)
	textStart, slot := 0, 0
	for i, id := range ids {
		if id != padID || slot >= len(images) {
			continue
		}
		if i > textStart {
			parts = append(parts, MultimodalPart{TokenIDs: append([]int(nil), ids[textStart:i]...)})
		}
		if images[slot] == nil {
			return MultimodalRequest{}, fmt.Errorf("%w: image %d is nil", ErrVisionSplice, slot)
		}
		parts = append(parts, MultimodalPart{Image: images[slot]})
		slot++
		textStart = i + 1
	}
	if slot != len(images) {
		return MultimodalRequest{}, fmt.Errorf("%w: %d image(s) supplied but the prompt carries %d associated image placeholder(s)", ErrVisionSplice, len(images), slot)
	}
	if textStart < len(ids) {
		parts = append(parts, MultimodalPart{TokenIDs: append([]int(nil), ids[textStart:]...)})
	}
	req.Parts = parts
	return req, nil
}

// ForwardImagePrompt is the one-call image+text forward: EncodeImagePrompt followed by
// ForwardMultimodal. The returned verdict is the forward's own — the complete accounting,
// including the embedding-token count — except on a pre-encode refusal, where it is the
// admission verdict that stopped the request before any pixel work.
func (m *Model) ForwardImagePrompt(ids []int, padID int, images []MultimodalImage, policy MultimodalPolicy) (*Activations, MultimodalVerdict, error) {
	req, verdict, err := m.EncodeImagePrompt(ids, padID, images, policy)
	if err != nil {
		return nil, verdict, err
	}
	return m.ForwardMultimodal(req)
}

// admitImagesBeforeEncode applies the pixel-free half of the image gate to a raw image
// list, in ForwardMultimodal's precedence: the per-image bounds first, then the quarantine
// opt-in. It exists so the expensive work (decode + ViT forward) happens only for input
// that has already been admitted — the bounds are a budget, and a budget checked after the
// spend is not a budget.
func admitImagesBeforeEncode(images []MultimodalImage, policy MultimodalPolicy) (MultimodalVerdict, error) {
	p := policy.withDefaults()
	verdict := MultimodalVerdict{Decision: MultimodalAllow, Mode: p.Mode}
	if err := p.valid(); err != nil {
		verdict.Decision = MultimodalDeny
		verdict.Reason = err.Error()
		return verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
	}
	for _, img := range images {
		if err := admitImageMetadata(img, p, &verdict); err != nil {
			return verdict, err
		}
	}
	if verdict.Images > 0 && p.Mode != MultimodalModeQuarantine {
		verdict.Decision = MultimodalQuarantine
		verdict.Reason = "multimodal input requires explicit quarantine opt-in"
		verdict.QuarantineID = unencodedQuarantineID(images)
		return verdict, fmt.Errorf("%w: %s", ErrMultimodalQuarantined, verdict.Reason)
	}
	return verdict, nil
}

// unencodedQuarantineID digests images held BEFORE encoding through the one quarantine-id
// function, with no vectors — which is exactly right here: nothing was encoded, so the id
// names the held bytes and metadata and nothing else. The same image refused after its
// tower forward carries the encoder rows in its digest too, so the two ids differ by
// construction; each names what is actually being held.
func unencodedQuarantineID(images []MultimodalImage) string {
	parts := make([]MultimodalPart, len(images))
	for i, img := range images {
		parts[i] = MultimodalPart{Image: &VisionEmbedding{Image: img}}
	}
	return multimodalQuarantineID(parts)
}
