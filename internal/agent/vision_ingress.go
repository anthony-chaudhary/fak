package agent

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

var (
	ErrNativeVisionIngress            = errors.New("native vision ingress")
	ErrNativeVisionPrefillUnsupported = errors.New("native vision embedding-row prefill is not implemented")
)

const maxNativeVisionImageBytes = 10 << 20

// VisionImageInput is the non-wire image carrier retained from an OpenAI
// image_url content part. Proxy marshaling deliberately ignores it.
type VisionImageInput struct {
	URL string
}

func visionContent(raw json.RawMessage) (string, []VisionImageInput) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '[' {
		return "", nil
	}
	var parts []json.RawMessage
	if json.Unmarshal(raw, &parts) != nil {
		return "", nil
	}
	var rendered []string
	var images []VisionImageInput
	for _, rawPart := range parts {
		var part struct {
			Type     string          `json:"type"`
			ImageURL json.RawMessage `json:"image_url"`
		}
		if json.Unmarshal(rawPart, &part) != nil {
			continue
		}
		if part.Type == "image_url" {
			images = append(images, VisionImageInput{URL: imageURLValue(part.ImageURL)})
			rendered = append(rendered, tokenizer.VisionStartToken+tokenizer.ImagePadToken+tokenizer.VisionEndToken)
			continue
		}
		if text, err := contentPartText(rawPart); err == nil && text != "" {
			rendered = append(rendered, text)
		}
	}
	if len(images) == 0 {
		return "", nil
	}
	return strings.Join(rendered, "\n"), images
}

func imageURLValue(raw json.RawMessage) string {
	var direct string
	if json.Unmarshal(raw, &direct) == nil {
		return direct
	}
	var object struct {
		URL string `json:"url"`
	}
	_ = json.Unmarshal(raw, &object)
	return object.URL
}

func decodeVisionDataURL(rawURL string) (model.MultimodalImage, error) {
	if !strings.HasPrefix(rawURL, "data:") {
		return model.MultimodalImage{}, fmt.Errorf("%w: image_url must be an inline data URL", ErrNativeVisionIngress)
	}
	meta, payload, ok := strings.Cut(strings.TrimPrefix(rawURL, "data:"), ",")
	if !ok || !strings.HasSuffix(strings.ToLower(meta), ";base64") {
		return model.MultimodalImage{}, fmt.Errorf("%w: image data URL must use base64", ErrNativeVisionIngress)
	}
	mediaType := strings.ToLower(strings.TrimSpace(strings.TrimSuffix(meta, ";base64")))
	if mediaType == "" || strings.Contains(mediaType, ";") {
		return model.MultimodalImage{}, fmt.Errorf("%w: invalid image media type", ErrNativeVisionIngress)
	}
	if len(payload) > ((maxNativeVisionImageBytes+2)/3)*4 {
		return model.MultimodalImage{}, fmt.Errorf("%w: image payload exceeds %d bytes", ErrNativeVisionIngress, maxNativeVisionImageBytes)
	}
	data, err := base64.StdEncoding.Strict().DecodeString(payload)
	if err != nil || len(data) == 0 {
		return model.MultimodalImage{}, fmt.Errorf("%w: invalid base64 image payload", ErrNativeVisionIngress)
	}
	if len(data) > maxNativeVisionImageBytes {
		return model.MultimodalImage{}, fmt.Errorf("%w: image payload exceeds %d bytes", ErrNativeVisionIngress, maxNativeVisionImageBytes)
	}
	w, h, ok := sniffImageDimensions(data)
	if !ok {
		return model.MultimodalImage{}, fmt.Errorf("%w: unsupported or malformed image container", ErrNativeVisionIngress)
	}
	return model.MultimodalImage{MediaType: mediaType, Bytes: data, Width: w, Height: h}, nil
}

func (p *InKernelPlanner) visionTemplateAssociation() model.VisionTemplateAssociation {
	if p != nil && p.m != nil && p.m.Cfg.IsDeepSeekV41() && p.deepSeekV41ChatProfile == DeepSeekV411017GGUFTextProfile {
		return model.VisionTemplateAssociationDeepInfer
	}
	return model.VisionTemplateAssociationStrict
}

func prepareVisionMessages(messages []Message) ([]Message, []model.MultimodalImage, error) {
	var images []model.MultimodalImage
	var out []Message
	for i := range messages {
		if len(messages[i].VisionImages) == 0 {
			continue
		}
		if out == nil {
			out = append([]Message(nil), messages...)
		}
		out[i].Content = messages[i].VisionPrompt
		for _, input := range messages[i].VisionImages {
			image, err := decodeVisionDataURL(input.URL)
			if err != nil {
				return nil, nil, err
			}
			images = append(images, image)
		}
	}
	if out == nil {
		return messages, nil, nil
	}
	return out, images, nil
}

func visionEmbeddingTokens(req model.MultimodalRequest) int {
	var n int
	for _, part := range req.Parts {
		if part.Image != nil {
			n += len(part.Image.Vectors)
		}
	}
	return n
}

func visionSequenceTokens(req model.MultimodalRequest) int {
	var n int
	for _, part := range req.Parts {
		n += len(part.TokenIDs)
		if part.Image != nil {
			n += len(part.Image.Vectors)
		}
	}
	return n
}

func (p *InKernelPlanner) encodeVisionPrompt(ids []int, images []model.MultimodalImage, sp SampleParams) (*model.MultimodalRequest, *PromptMultimodalEncoding, error) {
	if len(images) == 0 {
		return nil, nil, nil
	}
	padID, ok := p.tok.ImagePadID()
	if !ok {
		padID = -1
	}
	association := p.visionTemplateAssociation()
	policy := model.MultimodalPolicy{Mode: sp.MultimodalMode, TemplateAssociation: association}
	req, verdict, err := p.m.EncodeImagePrompt(ids, padID, images, policy)
	if err != nil {
		return nil, nil, err
	}
	return &req, &PromptMultimodalEncoding{
		Decision: verdict.Decision, TemplateAssociation: association,
		Images: verdict.Images, ImageBytes: verdict.ImageBytes,
		EmbeddingTokens: visionEmbeddingTokens(req),
		TemplateTokens:  len(ids), SequenceTokens: visionSequenceTokens(req),
	}, nil
}
