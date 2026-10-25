package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// PromptEncoding is the prompt representation consumed by native context
// admission. For multimodal input TokenIDs is the template stream and
// Multimodal.SequenceTokens reports the expanded text+embedding row count.
// TokenIDs is owned by the result and is safe for the caller to mutate.
type PromptEncoding struct {
	ModelID              string                    `json:"model_id"`
	RendererID           string                    `json:"renderer_id"`
	TokenizerID          string                    `json:"tokenizer_id"`
	TokenIDs             []int                     `json:"token_ids"`
	PromptTokens         int                       `json:"prompt_tokens"`
	ContextWindowTokens  int                       `json:"context_window_tokens"`
	ReservedOutputTokens int                       `json:"reserved_output_tokens"`
	RenderedSHA256       string                    `json:"rendered_sha256"`
	Multimodal           *PromptMultimodalEncoding `json:"multimodal,omitempty"`
}

type PromptMultimodalEncoding struct {
	Decision            model.MultimodalDecision        `json:"decision"`
	TemplateAssociation model.VisionTemplateAssociation `json:"template_association,omitempty"`
	Images              int                             `json:"images"`
	ImageBytes          int64                           `json:"image_bytes"`
	EmbeddingTokens     int                             `json:"embedding_tokens"`
	TemplateTokens      int                             `json:"template_tokens"`
	SequenceTokens      int                             `json:"sequence_tokens"`
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
	sharedBoundary     int
	multimodal         *model.MultimodalRequest
	multimodalEncoding *PromptMultimodalEncoding
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
	sp.nativeVisionEncodeOnly = true
	prepared, err := p.preparePrompt(ctx, messages, tools, sp, opts...)
	if err != nil {
		return PromptEncoding{}, err
	}
	if err := p.refuseContextLength(prepared.promptTokens(), prepared.maxNew); err != nil {
		return PromptEncoding{}, err
	}
	sum := sha256.Sum256([]byte(prepared.rendered))
	return PromptEncoding{
		ModelID:              p.modelID,
		RendererID:           p.promptRendererID(sp),
		TokenizerID:          tokenizerID,
		TokenIDs:             append([]int(nil), prepared.ids...),
		PromptTokens:         prepared.promptTokens(),
		ContextWindowTokens:  p.ContextWindow(),
		ReservedOutputTokens: prepared.maxNew,
		RenderedSHA256:       hex.EncodeToString(sum[:]),
		Multimodal:           prepared.multimodalEncoding,
	}, nil
}

func (p preparedPrompt) promptTokens() int {
	if p.multimodalEncoding != nil {
		return p.multimodalEncoding.SequenceTokens
	}
	return len(p.ids)
}

func (p *InKernelPlanner) preparePrompt(ctx context.Context, messages []Message, tools []ToolDef, sp SampleParams, opts ...SampleOpt) (preparedPrompt, error) {
	if err := ctx.Err(); err != nil {
		return preparedPrompt{}, err
	}
	if err := p.admitDeepSeekV41ChatProfile(messages, sp); err != nil {
		return preparedPrompt{}, err
	}
	messages, tools, shrink := p.ApplyPromptShrink(ctx, messages, tools, opts...)
	renderMessages, images, err := prepareVisionMessages(messages)
	if err != nil {
		return preparedPrompt{}, err
	}
	raw := inKernelRawV41Prompt(p.m.Cfg, sp)
	var rendered string
	var ids []int
	if deepSeekV41SegmentedPrompt(p.m.Cfg, sp) {
		segs := inKernelRenderDeepSeekV41ChatSegments(renderMessages, tools, sp)
		rendered = joinPromptSegments(segs)
		ids, err = p.tok.EncodeSegments(segs)
	} else {
		rendered = inKernelRenderPrompt(renderMessages, tools, p.m.Cfg, sp)
		ids, err = p.tok.Encode(rendered)
	}
	if err != nil {
		return preparedPrompt{}, err
	}
	if raw && p.m.Cfg.BOSTokenID != nil {
		ids = append([]int{*p.m.Cfg.BOSTokenID}, ids...)
	}
	var multimodal *model.MultimodalRequest
	var multimodalEncoding *PromptMultimodalEncoding
	if len(images) > 0 {
		multimodal, multimodalEncoding, err = p.encodeVisionPrompt(ids, images, sp)
		if err != nil {
			return preparedPrompt{}, err
		}
		if !sp.nativeVisionEncodeOnly {
			return preparedPrompt{}, ErrNativeVisionPrefillUnsupported
		}
	}
	maxNew := p.maxNew
	if sp.MaxTokens != nil && *sp.MaxTokens > 0 {
		maxNew = *sp.MaxTokens
	}
	return preparedPrompt{
		messages: messages, tools: tools, rendered: rendered, ids: ids, maxNew: maxNew,
		nativeCompaction: shrink.nativeCompaction,
		sharedBoundary:   p.sharedPrefixBoundaryTokens(rendered, ids),
		multimodal:       multimodal, multimodalEncoding: multimodalEncoding,
	}, nil
}

// inKernelRawV41Prompt reports a raw-completion request on DeepSeek V4.1: the text
// is encoded as is and prefixed with the GGUF's BOS, matching the reference fork's
// raw completion instead of the ChatML template other families use.
func inKernelRawV41Prompt(cfg model.Config, sp SampleParams) bool {
	return sp.RawPrompt && cfg.IsDeepSeekV41()
}

// DeepSeekV41SegmentedPromptEnv opts V4.1 chat prompts into segmented encoding:
// client text can no longer spell control tokens. Off keeps legacy prompt ids.
const DeepSeekV41SegmentedPromptEnv = "FAK_DEEPSEEK41_SEGMENTED_PROMPT"

const deepSeekV41SegmentedRendererSuffix = "+segmented-v1"

func deepSeekV41SegmentedPrompt(cfg model.Config, sp SampleParams) bool {
	return cfg.IsDeepSeekV41() && !inKernelRawV41Prompt(cfg, sp) && os.Getenv(DeepSeekV41SegmentedPromptEnv) == "1"
}

func joinPromptSegments(segs []tokenizer.Segment) string {
	n := 0
	for _, seg := range segs {
		n += len(seg.Text)
	}
	var b strings.Builder
	b.Grow(n)
	for _, seg := range segs {
		b.WriteString(seg.Text)
	}
	return b.String()
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
	if p.m != nil && p.m.Cfg.IsDeepSeekV41() && !inKernelRawV41Prompt(p.m.Cfg, sp) {
		if (sp.ReasoningEffort != "" && sp.ReasoningEffort != EffortTierNone) ||
			(sp.ThinkingBudget != nil && *sp.ThinkingBudget != 0) || len(sp.ChatTemplateKwargs) != 0 {
			return fmt.Errorf("%w: only the non-thinking renderer is supported", ErrDeepSeekV41ChatProfile)
		}
		for _, message := range messages {
			if message.Thinking != "" || message.ThinkingSignature != "" || len(message.RedactedThinking) != 0 || message.ReasoningContent != "" {
				return fmt.Errorf("%w: reasoning history requires a separate renderer", ErrDeepSeekV41ChatProfile)
			}
		}
	}
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
		visionStructured := len(message.VisionImages) > 0 && sp.MultimodalMode == model.MultimodalModeQuarantine
		if message.ContentWasStructured && !visionStructured {
			return fmt.Errorf("%w: structured content requires a separate renderer", ErrDeepSeekV41ChatProfile)
		}
	}
	if inKernelRawV41Prompt(p.m.Cfg, sp) {
		return nil
	}
	return nil
}

func (p *InKernelPlanner) promptRendererID(sp SampleParams) string {
	legacy := inKernelPromptRendererID(p.m.Cfg, sp)
	if deepSeekV41SegmentedPrompt(p.m.Cfg, sp) {
		legacy += deepSeekV41SegmentedRendererSuffix
	}
	if p.deepSeekV41ChatProfile == "" || inKernelRawV41Prompt(p.m.Cfg, sp) {
		return legacy
	}
	return p.deepSeekV41ChatProfile + ":" + legacy
}
