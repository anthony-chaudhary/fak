package model

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
)

// MultimodalMode selects whether image-bearing input can become model-visible.
// The zero value is fail-closed: text-only prompts are allowed, but image input is
// quarantined until the caller explicitly opts into the quarantine rollout mode.
type MultimodalMode string

const (
	MultimodalModeDisabled   MultimodalMode = ""
	MultimodalModeQuarantine MultimodalMode = "quarantine"
)

// MultimodalDecision is the governance verdict for a multimodal request.
type MultimodalDecision string

const (
	MultimodalAllow      MultimodalDecision = "allow"
	MultimodalQuarantine MultimodalDecision = "quarantine"
	MultimodalDeny       MultimodalDecision = "deny"
)

const (
	defaultMultimodalMaxImages          = 4
	defaultMultimodalMaxImageBytes      = int64(10 << 20)
	defaultMultimodalMaxImagePixels     = int64(4096 * 4096)
	defaultMultimodalMaxEmbeddingTokens = 1024
)

var (
	ErrMultimodalQuarantined = errors.New("model: multimodal input quarantined")
	ErrMultimodalDenied      = errors.New("model: multimodal input denied")
)

// MultimodalPolicy is the opt-in governance envelope for image-bearing prompts.
// Mode must be MultimodalModeQuarantine before image embeddings can be released to
// the model; the raw image bytes never enter the decoder path.
type MultimodalPolicy struct {
	Mode               MultimodalMode
	MaxImages          int
	MaxImageBytes      int64
	MaxImagePixels     int64
	MaxEmbeddingTokens int
}

// MultimodalImage is the bounded image metadata used for governance. Bytes are
// optional when a caller supplies precomputed embeddings from a trusted external
// vision encoder, but width/height and media type are still required for admission.
type MultimodalImage struct {
	MediaType string
	Bytes     []byte
	Width     int
	Height    int
}

// VisionEmbedding is the VLM seam: a CLIP/LLaVA-style image encoder supplies one
// or more hidden-size vectors that are spliced into the prompt sequence.
type VisionEmbedding struct {
	Image   MultimodalImage
	Vectors [][]float32
}

// VisionEncoder is the integration point for a CLIP-like image tower. The model
// package governs and consumes hidden-size vectors; concrete image decoders/encoders
// stay outside this text-forward core.
type VisionEncoder interface {
	EncodeImage(MultimodalImage) (VisionEmbedding, error)
}

// MultimodalPart is one ordered prompt part. A part is either text token ids or
// one precomputed image embedding, never both.
type MultimodalPart struct {
	TokenIDs []int
	Image    *VisionEmbedding
}

// MultimodalRequest is an ordered text+image prompt plus its admission policy.
type MultimodalRequest struct {
	Parts  []MultimodalPart
	Policy MultimodalPolicy
}

// MultimodalVerdict records the governance decision and bounded accounting for a
// request. QuarantineID is a digest pointer to the held image bytes/metadata.
type MultimodalVerdict struct {
	Decision        MultimodalDecision
	Mode            MultimodalMode
	Reason          string
	QuarantineID    string
	Images          int
	ImageBytes      int64
	EmbeddingTokens int
}

// ForwardMultimodal runs a governed text+image prompt. Text-only requests work
// under the zero policy; image-bearing requests require Mode=quarantine and must
// pass the image byte/pixel/embedding-token limits.
func (m *Model) ForwardMultimodal(req MultimodalRequest) (*Activations, MultimodalVerdict, error) {
	rows, verdict, err := m.prepareMultimodalRows(req)
	if err != nil {
		return nil, verdict, err
	}
	return m.forwardHiddenRows(rows), verdict, nil
}

func (m *Model) prepareMultimodalRows(req MultimodalRequest) ([][]float32, MultimodalVerdict, error) {
	policy := req.Policy.withDefaults()
	verdict := MultimodalVerdict{Decision: MultimodalAllow, Mode: policy.Mode}
	if err := policy.valid(); err != nil {
		verdict.Decision = MultimodalDeny
		verdict.Reason = err.Error()
		return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
	}
	if len(req.Parts) == 0 {
		verdict.Decision = MultimodalDeny
		verdict.Reason = "empty prompt"
		return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
	}

	H := m.Cfg.HiddenSize
	if H <= 0 {
		verdict.Decision = MultimodalDeny
		verdict.Reason = "hidden size is unset"
		return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
	}
	embed := m.embedRows()
	rows := make([][]float32, 0, len(req.Parts))
	for i, part := range req.Parts {
		hasText := len(part.TokenIDs) > 0
		hasImage := part.Image != nil
		if hasText == hasImage {
			verdict.Decision = MultimodalDeny
			verdict.Reason = fmt.Sprintf("part %d must contain exactly one of token ids or image embedding", i)
			return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
		}
		if hasText {
			for _, id := range part.TokenIDs {
				if id < 0 || id >= m.Cfg.VocabSize {
					verdict.Decision = MultimodalDeny
					verdict.Reason = fmt.Sprintf("token id %d outside vocab size %d", id, m.Cfg.VocabSize)
					return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
				}
				row := append([]float32(nil), embed[id*H:(id+1)*H]...)
				scaleEmbedInPlace(row, m.Cfg)
				rows = append(rows, row)
			}
			continue
		}
		if err := admitVisionEmbedding(part.Image, policy, &verdict, H); err != nil {
			return nil, verdict, err
		}
		for _, vec := range part.Image.Vectors {
			rows = append(rows, append([]float32(nil), vec...))
		}
	}
	if len(rows) == 0 {
		verdict.Decision = MultimodalDeny
		verdict.Reason = "empty prompt"
		return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalDenied, verdict.Reason)
	}
	if verdict.Images > 0 && policy.Mode != MultimodalModeQuarantine {
		verdict.Decision = MultimodalQuarantine
		verdict.Reason = "multimodal input requires explicit quarantine opt-in"
		verdict.QuarantineID = multimodalQuarantineID(req.Parts)
		return nil, verdict, fmt.Errorf("%w: %s", ErrMultimodalQuarantined, verdict.Reason)
	}
	return rows, verdict, nil
}

func (p MultimodalPolicy) withDefaults() MultimodalPolicy {
	if p.MaxImages == 0 {
		p.MaxImages = defaultMultimodalMaxImages
	}
	if p.MaxImageBytes == 0 {
		p.MaxImageBytes = defaultMultimodalMaxImageBytes
	}
	if p.MaxImagePixels == 0 {
		p.MaxImagePixels = defaultMultimodalMaxImagePixels
	}
	if p.MaxEmbeddingTokens == 0 {
		p.MaxEmbeddingTokens = defaultMultimodalMaxEmbeddingTokens
	}
	return p
}

func (p MultimodalPolicy) valid() error {
	switch p.Mode {
	case MultimodalModeDisabled, MultimodalModeQuarantine:
	default:
		return fmt.Errorf("unknown multimodal mode %q", p.Mode)
	}
	if p.MaxImages <= 0 {
		return fmt.Errorf("max images must be positive")
	}
	if p.MaxImageBytes <= 0 {
		return fmt.Errorf("max image bytes must be positive")
	}
	if p.MaxImagePixels <= 0 {
		return fmt.Errorf("max image pixels must be positive")
	}
	if p.MaxEmbeddingTokens <= 0 {
		return fmt.Errorf("max embedding tokens must be positive")
	}
	return nil
}

func admitVisionEmbedding(img *VisionEmbedding, policy MultimodalPolicy, verdict *MultimodalVerdict, hidden int) error {
	verdict.Images++
	if verdict.Images > policy.MaxImages {
		return denyMultimodal(verdict, "too many images")
	}
	meta := img.Image
	media := strings.ToLower(strings.TrimSpace(meta.MediaType))
	if !strings.HasPrefix(media, "image/") {
		return denyMultimodal(verdict, "media type must be image/*")
	}
	if meta.Width <= 0 || meta.Height <= 0 {
		return denyMultimodal(verdict, "image dimensions must be positive")
	}
	w, h := int64(meta.Width), int64(meta.Height)
	if w > policy.MaxImagePixels/h || w*h > policy.MaxImagePixels {
		return denyMultimodal(verdict, "image pixel count exceeds limit")
	}
	nbytes := int64(len(meta.Bytes))
	verdict.ImageBytes += nbytes
	if nbytes > policy.MaxImageBytes || verdict.ImageBytes > policy.MaxImageBytes {
		return denyMultimodal(verdict, "image bytes exceed limit")
	}
	if len(img.Vectors) == 0 {
		return denyMultimodal(verdict, "image embedding is empty")
	}
	verdict.EmbeddingTokens += len(img.Vectors)
	if verdict.EmbeddingTokens > policy.MaxEmbeddingTokens {
		return denyMultimodal(verdict, "image embedding token count exceeds limit")
	}
	for i, vec := range img.Vectors {
		if len(vec) != hidden {
			return denyMultimodal(verdict, fmt.Sprintf("image embedding vector %d has width %d, want %d", i, len(vec), hidden))
		}
	}
	return nil
}

func denyMultimodal(verdict *MultimodalVerdict, reason string) error {
	verdict.Decision = MultimodalDeny
	verdict.Reason = reason
	return fmt.Errorf("%w: %s", ErrMultimodalDenied, reason)
}

func multimodalQuarantineID(parts []MultimodalPart) string {
	h := sha256.New()
	var word [4]byte
	for _, part := range parts {
		if part.Image == nil {
			continue
		}
		img := part.Image.Image
		fmt.Fprintf(h, "%s\x00%d\x00%d\x00%d\x00%d\x00", strings.ToLower(strings.TrimSpace(img.MediaType)), img.Width, img.Height, len(img.Bytes), len(part.Image.Vectors))
		h.Write(img.Bytes)
		for _, vec := range part.Image.Vectors {
			fmt.Fprintf(h, "%d\x00", len(vec))
			for _, v := range vec {
				binary.LittleEndian.PutUint32(word[:], math.Float32bits(v))
				h.Write(word[:])
			}
		}
	}
	return "vision-sha256:" + hex.EncodeToString(h.Sum(nil))
}
