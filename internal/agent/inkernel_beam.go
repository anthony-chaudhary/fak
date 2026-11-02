package agent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/beam"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
)

var ErrNativeBeamSearchUnsupported = errors.New("native beam search unsupported")

// NativeBeamSearchParams explicitly selects bounded raw-logit CPU beam search.
// Nil is the historical sampler. Temperature/top-p/penalties are not part of
// this mode; omitted LengthPenalty means 1, and an explicit 0 means no penalty.
type NativeBeamSearchParams struct {
	Width         int      `json:"width"`
	LengthPenalty *float64 `json:"length_penalty,omitempty"`
}

type NativeBeamSearchSequence struct {
	Message           Message `json:"message"`
	TokenIDs          []int   `json:"token_ids"`
	GeneratedTokens   int     `json:"generated_tokens"`
	CumulativeLogprob float64 `json:"cumulative_logprob"`
	Score             float64 `json:"score"`
	FinishReason      string  `json:"finish_reason"`
}

// NativeBeamSearchReceipt is trusted internal search state, NOT a public wire
// response. Raw messages and token IDs have not passed gateway adjudication.
// Public callers must use the gateway's separate non-content projection.
type NativeBeamSearchReceipt struct {
	Schema           string                     `json:"schema"`
	Width            int                        `json:"width"`
	LengthPenalty    float64                    `json:"length_penalty"`
	Sequences        []NativeBeamSearchSequence `json:"sequences"`
	ForwardCalls     int                        `json:"forward_calls"`
	ReplayedTokens   int                        `json:"replayed_tokens"`
	ReplayTokenBound int                        `json:"replay_token_bound"`
	SessionOwnership string                     `json:"session_ownership"`
}

// WithNativeBeamSearch captures the caller's options by value. The typed
// carrier is never forwarded to an HTTP provider.
func WithNativeBeamSearch(params *NativeBeamSearchParams) SampleOpt {
	if params == nil {
		return func(*SampleParams) {}
	}
	captured := *params
	if params.LengthPenalty != nil {
		penalty := *params.LengthPenalty
		captured.LengthPenalty = &penalty
	}
	return func(sp *SampleParams) {
		copied := captured
		if captured.LengthPenalty != nil {
			penalty := *captured.LengthPenalty
			copied.LengthPenalty = &penalty
		}
		sp.NativeBeamSearch = &copied
	}
}

type NativeBeamSearchPlanner interface {
	Planner
	NativeBeamSearchSupported() bool
}

func (p *InKernelPlanner) NativeBeamSearchSupported() bool {
	return p != nil && p.m != nil && p.tok != nil && p.backend == nil && !p.metal &&
		!p.requireDeviceExecution && p.deviceOnlyBypass == "" && p.denseGPULayers == 0 &&
		!p.m.Cfg.IsMoE() && !p.m.Cfg.IsDeepSeekV41() && p.m.ExpertParallelRanks() <= 1 && !p.m.IsExpertParallelRankLocal() && p.dflash2 == nil &&
		p.speculativeEngine == nil && !p.vulkanMTP && p.vulkanMTPDraftFactory == nil &&
		p.MetalMTPCoordinator() == nil && !p.coalescesQwenDecode() && !p.v41EngramLookAhead &&
		!p.qwenQ4KPrefillChunkTarget() && p.kvPrecisionErr == nil
}

func ValidateNativeBeamSearchParams(params *NativeBeamSearchParams) error {
	if params == nil {
		return nil
	}
	penalty := 1.0
	if params.LengthPenalty != nil {
		penalty = *params.LengthPenalty
	}
	if params.Width < 1 || params.Width > beam.MaxSearchWidth || math.IsNaN(penalty) || math.IsInf(penalty, 0) || penalty < 0 || penalty > 2 {
		return fmt.Errorf("%w: width must be 1..8 and length penalty finite in [0,2]", ErrNativeBeamSearchUnsupported)
	}
	return nil
}

func nativeBeamRequestSupported(messages []Message, tools []ToolDef, sp SampleParams) error {
	if err := ValidateNativeBeamSearchParams(sp.NativeBeamSearch); err != nil {
		return err
	}
	if len(tools) > 0 || len(sp.ToolChoice) > 0 || len(sp.ResponseFormat) > 0 || len(sp.GuidedDecode) > 0 ||
		len(sp.Stop) > 0 || len(sp.LogitBias) > 0 || sp.TopP != nil || sp.TopK != nil || sp.MinP != nil || sp.Seed != nil ||
		sp.FrequencyPenalty != nil || sp.PresencePenalty != nil || sp.SpeculativeNMax != nil ||
		sp.DecodeTokenObserver != nil || sp.DecodeTrace || sp.NativeDecodeTokenIDs || sp.NativeDecodeTopLogprobs != 0 ||
		sp.NativeInferenceReceipt || sp.nativeVisionEncodeOnly || sp.MultimodalMode != "" || sp.PerTokenStream != nil || sp.RawPrompt || len(sp.RawRequestBody) > 0 || len(sp.ChatTemplateKwargs) > 0 ||
		sp.ThinkingBudget != nil || (sp.ReasoningEffort != "" && sp.ReasoningEffort != EffortTierNone) ||
		(sp.Temperature != nil && *sp.Temperature != 0) {
		return fmt.Errorf("%w: requires an unmodified nonstreaming text request without tools, sampling modifiers or constraints", ErrNativeBeamSearchUnsupported)
	}
	for _, m := range messages {
		if (m.Role != RoleSystem && m.Role != RoleUser && m.Role != RoleAssistant) || m.ContentWasStructured ||
			len(m.VisionImages) > 0 || m.VisionPrompt != "" || len(m.ToolCalls) > 0 || m.FunctionCall != nil ||
			m.ToolCallID != "" || m.Thinking != "" || m.ThinkingSignature != "" || len(m.RedactedThinking) > 0 || m.ReasoningContent != "" {
			return fmt.Errorf("%w: text-only non-tool history required", ErrNativeBeamSearchUnsupported)
		}
	}
	return nil
}

// admitBeamWorkspace adds the search-owned upper bound to the SAME armed host
// budget as ordinary sessions. It does not substitute for model/session admission.
// Unknown usage retains that budget's existing policy; the logical search bounds
// are enforced independently. This is not an RSS or process-global memory claim.
func (p *InKernelPlanner) admitBeamWorkspace(expected *hostMemoryBudget) (func(), error) {
	p.hostBudgetMu.Lock()
	defer p.hostBudgetMu.Unlock()
	b := p.hostBudget
	if b != expected {
		return nil, fmt.Errorf("%w: host admission configuration changed", ErrNativeBeamSearchUnsupported)
	}
	if b == nil {
		return func() {}, nil
	}
	avail, known := b.available()
	if known && avail < beam.SearchWorkspaceBudgetBytes {
		return nil, &InKernelCapacityError{Want: beam.SearchWorkspaceBudgetBytes, Avail: max(avail, 0), AvailSigned: avail, Scope: compute.MemoryScopeHost, Site: "native-beam-workspace"}
	}
	if b.reserved > math.MaxInt64-beam.SearchWorkspaceBudgetBytes {
		return nil, fmt.Errorf("%w: workspace reservation overflow", ErrNativeBeamSearchUnsupported)
	}
	b.reserved += beam.SearchWorkspaceBudgetBytes
	released := false
	return func() {
		p.hostBudgetMu.Lock()
		defer p.hostBudgetMu.Unlock()
		if !released {
			released = true
			b.reserved -= beam.SearchWorkspaceBudgetBytes
		}
	}, nil
}

func (p *InKernelPlanner) completeNativeBeamSearch(ctx context.Context, messages []Message, tools []ToolDef, sp SampleParams, opts ...SampleOpt) (*Completion, error) {
	if !p.NativeBeamSearchSupported() {
		return nil, fmt.Errorf("%w: requires the plain dense CPU planner without speculative or device execution", ErrNativeBeamSearchUnsupported)
	}
	if err := nativeBeamRequestSupported(messages, tools, sp); err != nil {
		return nil, err
	}
	maxNew := p.maxNew
	if sp.MaxTokens != nil {
		maxNew = *sp.MaxTokens
	}
	if maxNew < 1 || maxNew > beam.MaxSearchSteps || p.m.Cfg.VocabSize < 1 || p.m.Cfg.VocabSize > beam.MaxSearchVocab {
		return nil, fmt.Errorf("%w: output-token or vocabulary bound", ErrNativeBeamSearchUnsupported)
	}
	if len(offeredToolsFor(ctx, tools).names) != 0 {
		return nil, fmt.Errorf("%w: offered tool context", ErrNativeBeamSearchUnsupported)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return recoverDeviceCall(func() (comp *Completion, err error) {
		prepared, err := p.preparePrompt(ctx, messages, tools, sp, opts...)
		if err != nil {
			return nil, err
		}
		if prepared.multimodal != nil || prepared.multimodalEncoding != nil || len(prepared.tools) != 0 || strings.HasSuffix(prepared.rendered, qwenThinkAssistantSeed) {
			return nil, fmt.Errorf("%w: prepared prompt requires unsupported modality, tools or reasoning mode", ErrNativeBeamSearchUnsupported)
		}
		if err := p.refuseContextLength(len(prepared.ids), prepared.maxNew); err != nil {
			return nil, err
		}
		cfg := beam.SearchConfig{Width: sp.NativeBeamSearch.Width, MaxTokens: prepared.maxNew, VocabSize: p.m.Cfg.VocabSize, LengthPenalty: 1}
		if sp.NativeBeamSearch.LengthPenalty != nil {
			cfg.LengthPenalty = *sp.NativeBeamSearch.LengthPenalty
		}
		for id, stop := range StopIDs(p.tok, p.m.Cfg) {
			if stop {
				cfg.EOS = append(cfg.EOS, id)
			}
		}
		if _, err := beam.ValidateSearch(prepared.ids, cfg); err != nil {
			return nil, err
		}
		// One session at a time, sized for the LONGEST replay rather than the original
		// prompt. We never admit a prefix snapshot or retain a cache between beams.
		longest := len(prepared.ids) + prepared.maxNew
		sessionBytes, _, _ := p.hostRequestDemand(longest, 0)
		if sessionBytes <= 0 || sessionBytes > int64(1<<30)-beam.SearchWorkspaceBudgetBytes {
			return nil, fmt.Errorf("%w: bounded session plan unavailable or above 1 GiB with search workspace", ErrNativeBeamSearchUnsupported)
		}
		if err := p.checkExecutionDeadline(ctx, len(prepared.ids), 0, prepared.maxNew); err != nil {
			return nil, err
		}
		p.hostBudgetMu.Lock()
		expectedBudget := p.hostBudget
		p.hostBudgetMu.Unlock()
		ctx, releaseHost, err := p.admitHostMemory(ctx, longest, 0)
		if err != nil {
			return nil, err
		}
		defer releaseHost()
		releaseWorkspace, err := p.admitBeamWorkspace(expectedBudget)
		if err != nil {
			return nil, err
		}
		defer releaseWorkspace()
		defer enginestep.Default.RequestStart()()
		result, err := beam.Search(ctx, prepared.ids, cfg, p.nativeBeamNextLogits)
		if err != nil {
			return nil, err
		}
		receipt := &NativeBeamSearchReceipt{Schema: "fak.native-beam-search/1", Width: cfg.Width, LengthPenalty: cfg.LengthPenalty, ForwardCalls: result.ForwardCalls, ReplayedTokens: result.ReplayedTokens, ReplayTokenBound: result.ReplayTokenBound, SessionOwnership: "serial-fresh-owned-cpu-replay"}
		for _, row := range result.Sequences {
			text, err := p.tok.Decode(row.TokenIDs)
			if err != nil {
				return nil, err
			}
			// Beam search does not select tool-call actions. Refuse the entire result
			// rather than put an unadjudicated alternative in the response extension.
			if hasTextToolCandidate(text) {
				return nil, fmt.Errorf("%w: generated tool-call text", ErrNativeBeamSearchUnsupported)
			}
			reasoning, content := splitNativeBudgetReasoning(text, false)
			// Match the ordinary plain-text projection, including a dangling close marker.
			for strings.Contains(content, thinkClose) {
				idx := strings.Index(content, thinkClose)
				residual := strings.TrimSpace(content[:idx])
				if strings.HasPrefix(strings.ToLower(residual), thinkOpen) {
					residual = residual[len(thinkOpen):]
				}
				residual = strings.TrimSpace(residual)
				if residual != "" {
					if reasoning != "" {
						reasoning = strings.TrimSpace(reasoning + "\n" + residual)
					} else {
						reasoning = residual
					}
				}
				content = strings.TrimSpace(content[idx+len(thinkClose):])
			}
			projected := normalizeInKernelToolCalls(&Completion{Message: Message{Role: RoleAssistant, Content: content, ReasoningContent: reasoning}, FinishReason: row.FinishReason}, offeredToolsFor(ctx, tools), sp.ToolChoice, tools, messages)
			if projected == nil || projected.ToolCallsDropped || len(projected.Message.ToolCalls) > 0 || projected.Message.FunctionCall != nil {
				return nil, fmt.Errorf("%w: alternative requires tool adjudication", ErrNativeBeamSearchUnsupported)
			}
			msg := projected.Message
			receipt.Sequences = append(receipt.Sequences, NativeBeamSearchSequence{Message: msg, TokenIDs: append([]int(nil), row.TokenIDs...), GeneratedTokens: row.GeneratedTokens, CumulativeLogprob: row.CumulativeLogprob, Score: row.Score, FinishReason: row.FinishReason})
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		best := receipt.Sequences[0]
		observeNativeCompaction(ctx, prepared.nativeCompaction)
		// Ordinary usage describes only the selected visible sequence. Exploration
		// and repeated prompt work are reported separately and never presented as TPS.
		gen := len(best.TokenIDs)
		return &Completion{Model: p.modelID, Message: best.Message, FinishReason: best.FinishReason,
			Usage:            Usage{PromptTokens: len(prepared.ids), CompletionTokens: gen, TotalTokens: len(prepared.ids) + gen},
			NativeBeamSearch: receipt}, nil
	})
}

func (p *InKernelPlanner) nativeBeamNextLogits(ctx context.Context, ids []int) (logits []float32, err error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := p.checkExecutionDeadline(ctx, len(ids), 0, 1); err != nil {
		return nil, err
	}
	s := p.m.NewSession()
	defer finishInKernelCleanup(&err, s.Close)
	p.configureNativeSession(s)
	logits, err = s.PrefillContext(ctx, ids)
	if err != nil {
		return nil, err
	}
	// Own the row across session teardown, including unusual future Close paths.
	return append([]float32(nil), logits...), nil
}
