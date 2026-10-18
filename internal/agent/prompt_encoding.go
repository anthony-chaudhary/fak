package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
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
	// sharedBoundary is the token length of the rendered system/tools block when it
	// is an exact prefix of ids (0 otherwise). Prefill admits a complete snapshot
	// there so sibling requests can restore it on a recurrent hybrid.
	sharedBoundary int
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
		RendererID:           p.promptRendererID(sp),
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
	if err := p.admitDeepSeekV41ChatProfile(messages, sp); err != nil {
		return preparedPrompt{}, err
	}
	messages, tools, shrink := p.ApplyPromptShrink(ctx, messages, tools, opts...)
	raw := inKernelRawV41Prompt(p.m.Cfg, sp)
	rendered := inKernelRenderPrompt(messages, tools, p.m.Cfg, sp)
	ids, err := p.tok.Encode(rendered)
	if err != nil {
		return preparedPrompt{}, err
	}
	if raw && p.m.Cfg.BOSTokenID != nil {
		ids = append([]int{*p.m.Cfg.BOSTokenID}, ids...)
	}
	maxNew := p.maxNew
	if sp.MaxTokens != nil && *sp.MaxTokens > 0 {
		maxNew = *sp.MaxTokens
	}
	return preparedPrompt{
		messages: messages, tools: tools, rendered: rendered, ids: ids, maxNew: maxNew,
		nativeCompaction: shrink.nativeCompaction,
		sharedBoundary:   p.sharedPrefixBoundaryTokens(rendered, ids),
	}, nil
}

// inKernelRawV41Prompt reports a raw-completion request on DeepSeek V4.1: the text
// is encoded as is and prefixed with the GGUF's BOS, matching the reference fork's
// raw completion instead of the ChatML template other families use.
func inKernelRawV41Prompt(cfg model.Config, sp SampleParams) bool {
	return sp.RawPrompt && cfg.IsDeepSeekV41()
}

func inKernelPromptRendererID(cfg model.Config, sp SampleParams) string {
	if inKernelRawV41Prompt(cfg, sp) {
		return "deepseek41-raw-bos-v1"
	}
	if cfg.IsDeepSeekV41() {
		return deepSeekV41ChatRendererID
	}
	thinking := "thinking"
	if inKernelSuppressQwenThinking(cfg, sp) {
		thinking = "no-thinking"
	}
	if inKernelUsesOrnithQwen35Template(cfg) {
		return "ornith-qwen35-chatml-v1:" + thinking
	}
	return "qwen-chatml-v1:" + thinking
}

// DeepSeekV411017GGUFTextProfile selects only the pinned converter's text-only,
// non-thinking contract. It does not select official Python encoding or admit
// native numerical, vision, or draft-model execution.
const DeepSeekV411017GGUFTextProfile = "deepseek41-1017-q4km-gguf-text-v1"

// Pins mrfakename/DeepSeek-V4.1-Flash-1017-GGUF at
// 1df54e7d1b1e442051ad393ce871870a929f243e, ordered Q4_K_M shards 1..10.
// The shard digest uses the loader receipt's path-independent canonical encoding.
const (
	deepSeekV411017ShardSetSHA256 = "a353ea6e868ab8b91325da4952bec73c52824ac6287dc87cbc1b15e5861ab8c2"
	deepSeekV411017TemplateSHA256 = "7bd4fbdf0c06cd7398ddb6913de6e4a5839b9d06569e74e5b2ec246b375ca9c9"
)

// ErrDeepSeekV41ChatProfile identifies refusal of an explicit prompt contract.
var ErrDeepSeekV41ChatProfile = errors.New("native DeepSeek V4.1 chat profile not admitted")

func (p *InKernelPlanner) admitDeepSeekV41ChatProfile(messages []Message, sp SampleParams) error {
	if p.deepSeekV41ChatProfile == "" {
		return nil
	}
	if p.deepSeekV41ChatProfile != DeepSeekV411017GGUFTextProfile {
		return fmt.Errorf("%w: unknown profile %q", ErrDeepSeekV41ChatProfile, p.deepSeekV41ChatProfile)
	}
	if p.m == nil || !p.m.Cfg.IsDeepSeekV41() || !p.q4k {
		return fmt.Errorf("%w: requires the authenticated Q4_K_M target", ErrDeepSeekV41ChatProfile)
	}
	identity, known := p.m.AuthenticatedArtifactIdentity()
	if !known || identity.ShardSetSHA256 != deepSeekV411017ShardSetSHA256 || identity.ChatTemplateSHA256 != deepSeekV411017TemplateSHA256 {
		return fmt.Errorf("%w: artifact or converter template identity differs", ErrDeepSeekV41ChatProfile)
	}
	// Validate before shrink can discard the source representation. The zero
	// profile remains compatible with legacy array-to-text decoding.
	for _, message := range messages {
		if message.ContentWasStructured {
			return fmt.Errorf("%w: structured content requires a separate renderer", ErrDeepSeekV41ChatProfile)
		}
	}
	if inKernelRawV41Prompt(p.m.Cfg, sp) {
		return nil
	}
	if (sp.ReasoningEffort != "" && sp.ReasoningEffort != EffortTierNone) ||
		(sp.ThinkingBudget != nil && *sp.ThinkingBudget != 0) || len(sp.ChatTemplateKwargs) != 0 {
		return fmt.Errorf("%w: only the non-thinking converter contract is supported", ErrDeepSeekV41ChatProfile)
	}
	for _, message := range messages {
		if message.Thinking != "" || message.ThinkingSignature != "" || len(message.RedactedThinking) != 0 || message.ReasoningContent != "" {
			return fmt.Errorf("%w: reasoning history requires a separate renderer", ErrDeepSeekV41ChatProfile)
		}
	}
	return nil
}

func (p *InKernelPlanner) promptRendererID(sp SampleParams) string {
	legacy := inKernelPromptRendererID(p.m.Cfg, sp)
	if p.deepSeekV41ChatProfile == "" || inKernelRawV41Prompt(p.m.Cfg, sp) {
		return legacy
	}
	return p.deepSeekV41ChatProfile + ":" + legacy
}
