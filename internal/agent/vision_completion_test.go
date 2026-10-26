package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// fak-test:runtime fast est=250ms
func TestVisionCompletionOrdinaryAndStreamConsumeExpandedRows(t *testing.T) {
	p := visionCompletionPlanner(t)
	message := visionCompletionMessage(t)
	answerID, err := p.tok.Encode("a")
	if err != nil || len(answerID) != 1 {
		t.Fatalf("encode forced answer: ids=%v err=%v", answerID, err)
	}
	opts := []SampleOpt{
		WithMultimodalMode(string(model.MultimodalModeQuarantine)),
		WithMaxTokens(1),
		WithLogitBias(map[int]float64{answerID[0]: 100}),
	}
	encoded, err := p.EncodePrompt(context.Background(), []Message{message}, nil, opts...)
	if err != nil {
		t.Fatalf("EncodePrompt: %v", err)
	}
	if encoded.Multimodal == nil || encoded.Multimodal.EmbeddingTokens != 1 || encoded.Multimodal.SequenceTokens != encoded.PromptTokens {
		t.Fatalf("encoding = %+v, want one admitted image row in sequence accounting", encoded)
	}

	ordinary, err := p.Complete(context.Background(), []Message{message}, nil, opts...)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if ordinary.Message.Content != "a" || ordinary.Usage.PromptTokens != encoded.PromptTokens || ordinary.Usage.PromptTokensDetails.CachedTokens != 0 {
		t.Fatalf("ordinary completion = %+v, want forced answer and expanded prompt accounting", ordinary)
	}

	var streamed strings.Builder
	stream, err := p.CompleteStream(context.Background(), func(piece string) error {
		streamed.WriteString(piece)
		return nil
	}, []Message{message}, nil, append(opts, WithPerTokenStream(false))...)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if stream.Message.Content != "a" || streamed.String() != "a" || stream.Usage.PromptTokens != encoded.PromptTokens || stream.Usage.PromptTokensDetails.CachedTokens != 0 {
		t.Fatalf("stream completion/content = %+v/%q, want forced answer and expanded prompt accounting", stream, streamed.String())
	}
}

// fak-test:runtime fast est=100ms
func TestVisionCompletionRefusesDeviceBeforeBackendWork(t *testing.T) {
	p := visionCompletionPlanner(t)
	backend := &countingBackend{}
	p.backend = backend
	_, err := p.Complete(context.Background(), []Message{visionCompletionMessage(t)}, nil,
		WithMultimodalMode(string(model.MultimodalModeQuarantine)), WithMaxTokens(1))
	if !errors.Is(err, model.ErrMultimodalDenied) {
		t.Fatalf("Complete error = %v, want ErrMultimodalDenied", err)
	}
	backend.mu.Lock()
	defer backend.mu.Unlock()
	if backend.mat != 0 || backend.batched != 0 {
		t.Fatalf("backend work = mat:%d batched:%d, want refusal before device work", backend.mat, backend.batched)
	}
}

func visionCompletionMessage(t *testing.T) Message {
	t.Helper()
	raw := `{"role":"user","content":[{"type":"text","text":"inspect"},{"type":"image_url","image_url":{"url":"data:image/png;base64,` + visionIngressPNG + `"}}]}`
	var message Message
	if err := json.Unmarshal([]byte(raw), &message); err != nil {
		t.Fatalf("unmarshal message: %v", err)
	}
	return message
}

func visionCompletionPlanner(t *testing.T) *InKernelPlanner {
	t.Helper()
	tok := visionCompletionTokenizer(t)
	cfg := tinyCfg()
	cfg.VocabSize = 262
	cfg.EOSTokenID = -1
	m := model.NewSynthetic(cfg)
	if err := m.AttachVisionTower(visionCompletionTower(t, cfg.HiddenSize)); err != nil {
		t.Fatalf("AttachVisionTower: %v", err)
	}
	return &InKernelPlanner{m: m, tok: tok, modelID: "vision-synthetic", maxNew: 1, tree: radixkv.New(0)}
}

func visionCompletionTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal([]byte(buildByteVocab()), &doc); err != nil {
		t.Fatalf("decode byte tokenizer fixture: %v", err)
	}
	modelDoc := doc["model"].(map[string]any)
	vocab := modelDoc["vocab"].(map[string]any)
	added := doc["added_tokens"].([]any)
	for i, token := range []string{tokenizer.VisionStartToken, tokenizer.ImagePadToken, tokenizer.VisionEndToken} {
		id := 259 + i
		vocab[token] = id
		added = append(added, map[string]any{"id": id, "content": token, "special": true})
	}
	doc["added_tokens"] = added
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal vision tokenizer fixture: %v", err)
	}
	tok, err := tokenizer.ParseJSON(raw)
	if err != nil {
		t.Fatalf("ParseJSON vision tokenizer: %v", err)
	}
	return tok
}

func visionCompletionTower(t *testing.T, hidden int) *model.VisionTower {
	t.Helper()
	ffn := hidden
	tensor := func(name string, shape ...int) model.NamedTensorF32 {
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = float32((i%7)-3) * 0.01
		}
		return model.NamedTensorF32{Name: name, Shape: shape, Data: data}
	}
	tensors := []model.NamedTensorF32{
		tensor("v.patch_embd.weight", hidden, 3),
		tensor("v.blk.0.attn_q.weight", hidden, hidden),
		tensor("v.blk.0.attn_k.weight", hidden, hidden),
		tensor("v.blk.0.attn_v.weight", hidden, hidden),
		tensor("v.blk.0.attn_out.weight", hidden, hidden),
		tensor("v.blk.0.ln1.weight", hidden),
		tensor("v.blk.0.ln2.weight", hidden),
		tensor("v.blk.0.ffn_up.weight", ffn, hidden),
		tensor("v.blk.0.ffn_down.weight", hidden, ffn),
		tensor("mm.0.weight", hidden, hidden),
	}
	tower, err := model.NewVisionTower(model.VisionConfig{
		HiddenSize: hidden, NumLayers: 1, NumHeads: 4, FFNLength: ffn,
		PatchSize: 1, MergeSize: 1, ProjOutDim: hidden, LNEps: 1e-5,
	}, tensors)
	if err != nil {
		t.Fatalf("NewVisionTower: %v", err)
	}
	return tower
}
