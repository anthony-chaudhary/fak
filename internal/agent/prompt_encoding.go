package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// PromptEncoding is the exact prompt representation consumed by native context
// admission. TokenIDs is owned by the result and is safe for the caller to mutate.
type PromptEncoding struct {
	ModelID              string `json:"model_id"`
	RendererID           string `json:"renderer_id"`
	TokenizerID          string `json:"tokenizer_id"`
	TokenIDs             []int  `json:"token_ids"`
	PromptTokens         int    `json:"prompt_tokens"`
	ContextWindowTokens  int    `json:"context_window_tokens"`
	ReservedOutputTokens int    `json:"reserved_output_tokens"`
	RenderedSHA256       string `json:"rendered_sha256"`
}

type preparedPrompt struct {
	messages         []Message
	tools            []ToolDef
	rendered         string
	ids              []int
	maxNew           int
	nativeCompaction *NativeCompactionObservation
}

// EncodePrompt applies the same request options, prompt shrink, renderer, and
// tokenizer as Complete without running prefill or decode.
func (p *InKernelPlanner) EncodePrompt(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (PromptEncoding, error) {
	if p == nil || p.m == nil || p.tok == nil {
		return PromptEncoding{}, errors.New("native prompt encoding requires a model and tokenizer")
	}
	if strings.TrimSpace(p.modelID) == "" {
		return PromptEncoding{}, errors.New("native prompt encoding requires a model identity")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return PromptEncoding{}, err
	}
	tokenizerID := p.tok.Identity()
	if tokenizerID == "" {
		return PromptEncoding{}, errors.New("native prompt encoding requires a tokenizer identity")
	}
	sp := applySampleOpts(opts...)
	prepared, err := p.preparePrompt(ctx, messages, tools, sp, opts...)
	if err != nil {
		return PromptEncoding{}, err
	}
	if err := p.refuseContextLength(len(prepared.ids), prepared.maxNew); err != nil {
		return PromptEncoding{}, err
	}
	sum := sha256.Sum256([]byte(prepared.rendered))
	return PromptEncoding{
		ModelID:              p.modelID,
		RendererID:           inKernelPromptRendererID(p.m.Cfg, sp),
		TokenizerID:          tokenizerID,
		TokenIDs:             append([]int(nil), prepared.ids...),
		PromptTokens:         len(prepared.ids),
		ContextWindowTokens:  p.ContextWindow(),
		ReservedOutputTokens: prepared.maxNew,
		RenderedSHA256:       hex.EncodeToString(sum[:]),
	}, nil
}

func (p *InKernelPlanner) preparePrompt(ctx context.Context, messages []Message, tools []ToolDef, sp SampleParams, opts ...SampleOpt) (preparedPrompt, error) {
	if err := ctx.Err(); err != nil {
		return preparedPrompt{}, err
	}
	messages, tools, shrink := p.ApplyPromptShrink(ctx, messages, tools, opts...)
	rendered := renderInKernelChatMLRequest(messages, tools, p.m.Cfg, sp.ResponseFormat, sp.ToolChoice, sp)
	ids, err := p.tok.Encode(rendered)
	if err != nil {
		return preparedPrompt{}, err
	}
	maxNew := p.maxNew
	if sp.MaxTokens != nil && *sp.MaxTokens > 0 {
		maxNew = *sp.MaxTokens
	}
	return preparedPrompt{messages: messages, tools: tools, rendered: rendered, ids: ids, maxNew: maxNew, nativeCompaction: shrink.nativeCompaction}, nil
}

func inKernelPromptRendererID(cfg model.Config, sp SampleParams) string {
	thinking := "thinking"
	if inKernelSuppressQwenThinking(cfg, sp) {
		thinking = "no-thinking"
	}
	if inKernelUsesOrnithQwen35Template(cfg) {
		return "ornith-qwen35-chatml-v1:" + thinking
	}
	return "qwen-chatml-v1:" + thinking
}
