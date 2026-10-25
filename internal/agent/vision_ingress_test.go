package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

const visionIngressPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

func visionIngressTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	tok, err := tokenizer.FromGGML(
		[]string{"a", "aa", tokenizer.VisionStartToken, tokenizer.ImagePadToken, tokenizer.VisionEndToken},
		[]string{"a a"}, []int32{1, 1, 3, 3, 3}, "qwen2",
	)
	if err != nil {
		t.Fatalf("FromGGML: %v", err)
	}
	return tok
}

// fak-test:runtime fast est=20ms lane=default
func TestMessageStructuredImageVisionIngressPreservesProxyProjection(t *testing.T) {
	raw := `{"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + visionIngressPNG + `"}},{"type":"text","text":"after"}]}`
	var message Message
	if err := json.Unmarshal([]byte(raw), &message); err != nil {
		t.Fatalf("UnmarshalJSON: %v", err)
	}
	if message.Content != "before\nafter" {
		t.Fatalf("proxy Content = %q, want historical text-only projection", message.Content)
	}
	wantPrompt := "before\n" + tokenizer.VisionStartToken + tokenizer.ImagePadToken + tokenizer.VisionEndToken + "\nafter"
	if message.VisionPrompt != wantPrompt || len(message.VisionImages) != 1 {
		t.Fatalf("vision carrier = (%q, %+v), want ordered prompt and one image", message.VisionPrompt, message.VisionImages)
	}
	wire, err := json.Marshal(message)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if strings.Contains(string(wire), "base64") || strings.Contains(string(wire), tokenizer.ImagePadToken) {
		t.Fatalf("proxy wire leaked native vision carrier: %s", wire)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestVisionIngressRequiresInlineDecodableImage(t *testing.T) {
	if _, err := decodeVisionDataURL("https://example.invalid/image.png"); !errors.Is(err, ErrNativeVisionIngress) {
		t.Fatalf("remote URL error = %v, want ErrNativeVisionIngress", err)
	}
	image, err := decodeVisionDataURL("data:image/png;base64," + visionIngressPNG)
	if err != nil {
		t.Fatalf("decodeVisionDataURL: %v", err)
	}
	if image.MediaType != "image/png" || image.Width != 1 || image.Height != 1 || len(image.Bytes) == 0 {
		t.Fatalf("decoded image = %+v, want bounded 1x1 PNG", image)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestVisionIngressModelIdentitySelectsDeepInferAndKeepsV41Refusal(t *testing.T) {
	cfg := tinyV41Cfg()
	cfg.ImageTokenID = 1
	p := &InKernelPlanner{
		m: model.NewSynthetic(cfg), tok: visionIngressTokenizer(t),
		deepSeekV41ChatProfile: DeepSeekV411017GGUFTextProfile,
	}
	if got := p.visionTemplateAssociation(); got != model.VisionTemplateAssociationDeepInfer {
		t.Fatalf("association = %q, want DeepInfer for admitted V4.1 template identity", got)
	}
	image, err := decodeVisionDataURL("data:image/png;base64," + visionIngressPNG)
	if err != nil {
		t.Fatal(err)
	}
	padID, _ := p.tok.ImagePadID()
	if _, _, err := p.encodeVisionPrompt([]int{0, padID}, []model.MultimodalImage{image}, SampleParams{}); !errors.Is(err, model.ErrMultimodalQuarantined) {
		t.Fatalf("default admission error = %v, want ErrMultimodalQuarantined", err)
	}
	if _, _, err := p.encodeVisionPrompt([]int{0, padID}, []model.MultimodalImage{image}, SampleParams{MultimodalMode: model.MultimodalModeQuarantine}); !errors.Is(err, model.ErrV41VisionUnsupported) {
		t.Fatalf("opted-in V4.1 error = %v, want ErrV41VisionUnsupported", err)
	}

	strict := &InKernelPlanner{m: model.NewSynthetic(tinyCfg()), tok: p.tok}
	if got := strict.visionTemplateAssociation(); got != model.VisionTemplateAssociationStrict {
		t.Fatalf("default association = %q, want strict", got)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestVisionIngressPreparePromptReachesGovernedModel(t *testing.T) {
	raw := `{"role":"user","content":[{"type":"text","text":"inspect"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + visionIngressPNG + `"}}]}`
	var message Message
	if err := json.Unmarshal([]byte(raw), &message); err != nil {
		t.Fatal(err)
	}
	p := &InKernelPlanner{m: model.NewSynthetic(tinyCfg()), tok: loadProbeTok(t)}
	if _, err := p.preparePrompt(context.Background(), []Message{message}, nil, SampleParams{}); !errors.Is(err, model.ErrMultimodalQuarantined) {
		t.Fatalf("preparePrompt error = %v, want governed default quarantine before tower/prefill", err)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestVisionIngressExpandedRowsDriveContextAccounting(t *testing.T) {
	req := model.MultimodalRequest{Parts: []model.MultimodalPart{
		{TokenIDs: []int{1, 2}},
		{Image: &model.VisionEmbedding{Vectors: [][]float32{{1}, {2}, {3}}}},
		{TokenIDs: []int{4}},
	}}
	if got := visionSequenceTokens(req); got != 6 {
		t.Fatalf("sequence tokens = %d, want 6 expanded rows", got)
	}
	prepared := preparedPrompt{ids: []int{1, 9, 4}, multimodalEncoding: &PromptMultimodalEncoding{SequenceTokens: 6}}
	if got := prepared.promptTokens(); got != 6 {
		t.Fatalf("context tokens = %d, want expanded sequence count 6", got)
	}
}
