package agent

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type inkling2RunPlanner struct {
	name  string
	model *ggufload.Inkling2LoopModel
	// device is the CUDA backend a device-bound model decodes on; nil on the host.
	device compute.Backend
	tok    *tokenizer.Tokenizer
	eos    int
}

// NewInkling2RunPlanner binds the interactive text runner to its checkpoint.
func NewInkling2RunPlanner(name string, m *ggufload.Inkling2LoopModel, tok *tokenizer.Tokenizer, eos int) Planner {
	return &inkling2RunPlanner{name: name, model: m, tok: tok, eos: eos}
}

// NewInkling2DeviceRunPlanner binds the runner to a model whose weights live on be.
func NewInkling2DeviceRunPlanner(name string, m *ggufload.Inkling2LoopModel, be compute.Backend, tok *tokenizer.Tokenizer, eos int) Planner {
	return &inkling2RunPlanner{name: name, model: m, device: be, tok: tok, eos: eos}
}

func (p *inkling2RunPlanner) Model() string { return p.name }

// The text-only profile of the checkpoint template. Keep operator content ordinary,
// including text that happens to spell a model control token.
func inkling2RunPrompt(messages []Message, sp SampleParams) ([]tokenizer.Segment, error) {
	if sp.RawPrompt {
		if len(messages) != 1 || messages[0].Role != RoleUser {
			return nil, fmt.Errorf("Inkling 2 raw completion needs a single user prompt")
		}
		return []tokenizer.Segment{{Text: messages[0].Content}}, nil
	}
	level, ok := map[string]string{"": "0.9", "none": "0", "minimal": "0.1", "low": "0.2", "medium": "0.7", "high": "0.9", "xhigh": "0.99", "max": "0.99"}[sp.ReasoningEffort]
	if !ok {
		return nil, fmt.Errorf("Inkling 2: unsupported effort %q", sp.ReasoningEffort)
	}
	var out []tokenizer.Segment
	control := func(s string) { out = append(out, tokenizer.Segment{Text: s, Trusted: true}) }
	message := func(role, channel, content string) {
		control(role + channel)
		out = append(out, tokenizer.Segment{Text: content})
		control("<|end_message|>")
	}
	needEffort := true
	for _, m := range messages {
		if needEffort && m.Role != RoleSystem {
			message("<|message_system|>", "<|content_text|>", "Thinking effort level: "+level)
			needEffort = false
		}
		switch m.Role {
		case RoleSystem:
			message("<|message_system|>", "<|content_text|>", m.Content)
		case RoleUser:
			message("<|message_user|>", "<|content_text|>", m.Content)
		case RoleAssistant:
			if m.ReasoningContent != "" {
				message("<|message_model|>", "<|content_thinking|>", m.ReasoningContent)
			}
			if m.Content != "" || m.ReasoningContent == "" {
				message("<|message_model|>", "<|content_text|>", m.Content)
			}
			control("<|content_model_end_sampling|>")
		default:
			return nil, fmt.Errorf("Inkling 2 run: unsupported role %q", m.Role)
		}
	}
	return out, nil
}

func inkling2RunReply(tok *tokenizer.Tokenizer, ids []int, raw bool) (Message, error) {
	msg := Message{Role: RoleAssistant}
	if raw {
		var err error
		msg.Content, err = tok.Decode(ids)
		return msg, err
	}
	modelID, _ := tok.SpecialID("<|message_model|>")
	thinkingID, _ := tok.SpecialID("<|content_thinking|>")
	textID, _ := tok.SpecialID("<|content_text|>")
	endID, _ := tok.SpecialID("<|end_message|>")
	var visible, reasoning []int
	thinking := false
	for _, id := range ids {
		switch id {
		case modelID, endID:
			thinking = false
		case thinkingID:
			thinking = true
		case textID:
			thinking = false
		default:
			if thinking {
				reasoning = append(reasoning, id)
			} else {
				visible = append(visible, id)
			}
		}
	}
	var err error
	msg.Content, err = tok.Decode(visible)
	if err == nil {
		msg.ReasoningContent, err = tok.Decode(reasoning)
	}
	return msg, err
}

func inkling2Param[T int | float64](v *T, fallback T) T {
	if v != nil {
		return *v
	}
	return fallback
}

func inkling2RunEffectiveTemperature(sp SampleParams) float64 {
	if sp.Temperature != nil {
		return *sp.Temperature
	}
	if sp.InheritUpstreamTemperature {
		return defaultInheritedTemperature
	}
	return 0
}

func inkling2RunUseDeviceGreedy(hasDevice bool, temp, freq, pres float64, sp SampleParams) bool {
	return hasDevice && temp == 0 && freq == 0 && pres == 0 && len(sp.LogitBias) == 0
}

func (p *inkling2RunPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	sp := applySampleOpts(opts...)
	if len(tools) > 0 || sp.ThinkingBudget != nil {
		return nil, fmt.Errorf("Inkling 2 run supports text and reasoning effort")
	}
	if err := validateNativeSamplingParams(sp); err != nil {
		return nil, err
	}
	segments, err := inkling2RunPrompt(messages, sp)
	if err != nil {
		return nil, err
	}
	ids, err := p.tok.EncodeSegments(segments)
	if err != nil {
		return nil, err
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("Inkling 2: prompt is empty")
	}
	maxTokens := inkling2Param(sp.MaxTokens, 512)
	if maxTokens < 1 || maxTokens > math.MaxInt-len(ids) {
		return nil, fmt.Errorf("Inkling 2: invalid token limit %d", maxTokens)
	}
	temp, topP, topK := inkling2RunEffectiveTemperature(sp), inkling2Param(sp.TopP, 0), inkling2Param(sp.TopK, 0)
	freq, pres := inkling2Param(sp.FrequencyPenalty, 0), inkling2Param(sp.PresencePenalty, 0)
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if inkling2RunUseDeviceGreedy(p.device != nil, temp, freq, pres, sp) {
		return p.deviceGreedy(ctx, sp, ids, maxTokens)
	}
	h, err := model.NewInkling2LoopHost(p.model.HP, p.model.Weights)
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		if id < 0 || id >= p.model.HP.VocabUnpadded {
			return nil, fmt.Errorf("Inkling 2: prompt token %d outside vocabulary", id)
		}
	}
	prefillPrompt := func() ([]float32, error) { return ggufload.Inkling2LoopPrefill(ctx, h, ids) }
	step := func(id int) ([]float32, error) { return ggufload.Inkling2LoopStep(h, id) }
	if r, ok := p.device.(interface{ Recycle() }); ok {
		// Device intermediates are read back per token, so recycle at each
		// token boundary instead of holding a whole batched prefill.
		prefillPrompt = func() (logits []float32, err error) {
			for i, id := range ids {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
				if logits, err = ggufload.Inkling2LoopPrefillToken(h, id, i == len(ids)-1); err != nil {
					return nil, err
				}
				r.Recycle()
			}
			return logits, nil
		}
		step = func(id int) ([]float32, error) {
			defer r.Recycle()
			return ggufload.Inkling2LoopStep(h, id)
		}
	}
	start := time.Now()
	logits, err := prefillPrompt()
	if err != nil {
		return nil, err
	}
	prefill := time.Since(start)
	start = time.Now()
	var counts []int32
	if freq != 0 || pres != 0 {
		counts = make([]int32, len(logits))
	}
	seed := time.Now().UnixNano()
	if sp.Seed != nil {
		seed = *sp.Seed
	}
	rng := rand.New(rand.NewSource(seed))
	var ws samplerWorkspace
	var cb rollingTokenCycleBreaker
	var generated []int
	steps, sampled := 0, 0
	reason := "length"
	for sampled < maxTokens {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		id := sampleLogitsWithPenaltyWS(logits, temp, topP, topK, model.LogitBias(sp.LogitBias), freq, pres, counts, rng, &ws)
		if ws.nonFinite {
			return nil, ErrNonFiniteLogits
		}
		sampled++
		if id == p.eos {
			reason = "stop"
			break
		}
		generated = append(generated, id)
		if cb.Observe(id) {
			reason = "stop"
			break
		}
		if sampled == maxTokens {
			break
		}
		if counts != nil {
			counts[id]++
		}
		logits, err = step(id)
		if err != nil {
			return nil, err
		}
		steps++
	}
	return p.result(sp, generated, len(ids), sampled, steps, reason, prefill, time.Since(start))
}

// deviceGreedy runs the resident and speculative device decoders, which stop
// at EOS like the sampled loop: the reply excludes it and it counts as sampled.
func (p *inkling2RunPlanner) deviceGreedy(ctx context.Context, sp SampleParams, ids []int, maxTokens int) (*Completion, error) {
	out, st, err := ggufload.Inkling2LoopGreedyUntilContext(ctx, p.model, p.device, ids, maxTokens, p.eos)
	if err != nil {
		return nil, err
	}
	reason, sampled := "length", len(out)
	if sampled < maxTokens {
		reason, sampled = "stop", sampled+1
	}
	return p.result(sp, out, len(ids), sampled, st.DecodeSteps, reason, st.Prefill, st.Decode)
}

func (p *inkling2RunPlanner) result(sp SampleParams, generated []int, prompt, sampled, steps int, reason string, prefill, decode time.Duration) (*Completion, error) {
	msg, err := inkling2RunReply(p.tok, generated, sp.RawPrompt)
	if err != nil {
		return nil, err
	}
	t := NewTimings(prompt, 0, steps, prefill.Seconds(), decode.Seconds())
	comp := &Completion{Model: p.name, Message: msg, FinishReason: reason, Usage: Usage{PromptTokens: prompt, CompletionTokens: sampled, TotalTokens: prompt + sampled}, Timings: t}
	populateReasoningTokensUsage(comp, p.tok)
	return comp, nil
}
