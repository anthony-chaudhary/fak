package agent

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// Inkling2LoopDefaultMaxTokens is the completion budget of a request that sets none.
// Agent harnesses usually send no max_tokens and a single tool call can carry a whole
// file, so the budget is large; the end-of-turn stop and the context clamp bound it.
const Inkling2LoopDefaultMaxTokens = 8192

// Inkling2LoopGeneration is what one Inkling2LoopGenerator.Generate call did.
type Inkling2LoopGeneration struct {
	Generated int
	Stopped   bool // a stop id or emit ended the turn before maxNew
	// CachedPromptTokens leading prompt tokens were served from a cache kept
	// from an earlier request; prefill computed only the rest.
	CachedPromptTokens            int
	PrefillSeconds, DecodeSeconds float64
	// Width is how many requests shared the forward that decoded this one; 0 is unknown.
	Width int
}

// inkling2LoopTimings is the turn's llama.cpp timings plus the width it decoded in.
func inkling2LoopTimings(promptTokens int, res Inkling2LoopGeneration) *Timings {
	t := NewTimings(promptTokens, res.CachedPromptTokens, res.Generated, res.PrefillSeconds, res.DecodeSeconds)
	if res.Width > 0 {
		t.Inkling2Width, t.Inkling2Mode = res.Width, "scalar"
		if res.Width > 1 {
			t.Inkling2Mode = "batch"
		}
	}
	return t
}

// Inkling2LoopGenerator decodes one request: it prefills prompt, possibly reusing
// a cached prefix with output identical to a fresh cache, then hands each
// generated id to emit until emit returns false, a stop id is generated (it is
// not emitted), or maxNew ids were emitted.
type Inkling2LoopGenerator interface {
	Generate(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, emit func(id int) bool) (Inkling2LoopGeneration, error)
}

// inkling2LoopGPUOwner is implemented by a generator whose own executor serializes
// every GPU call; the planner admits concurrent turns to such a generator so they
// can share a cohort.
type inkling2LoopGPUOwner interface {
	SerializesGPUAccess() bool
}

// Inkling2LoopPlanner serves chat turns on the dedicated inkling2_loop forward,
// which has no *model.Model and so cannot ride InKernelPlanner. Rendering, the
// stop set, output splitting and tool-call lifting are the shared in-kernel ones.
type Inkling2LoopPlanner struct {
	gen           Inkling2LoopGenerator
	tok           *tokenizer.Tokenizer
	cfg           model.Config
	modelID       string
	stops         map[int]bool
	contextTokens int
	mu            sync.Mutex
}

var _ StreamingPlanner = (*Inkling2LoopPlanner)(nil)

// NewInkling2LoopPlanner binds gen to the request surface. contextTokens > 0 caps
// prompt plus completion tokens.
func NewInkling2LoopPlanner(gen Inkling2LoopGenerator, tok *tokenizer.Tokenizer, modelID string, cfg model.Config, contextTokens int) (*Inkling2LoopPlanner, error) {
	if gen == nil || tok == nil {
		return nil, errors.New("inkling2_loop planner: generator and tokenizer are required")
	}
	if !cfg.IsInkling2Loop() {
		return nil, fmt.Errorf("inkling2_loop planner: model_type %q is not inkling2_loop", cfg.ModelType)
	}
	stops := StopIDs(tok, cfg)
	if _, ok := tok.SpecialID(Inkling2EndOfTurnMarker); !ok {
		return nil, fmt.Errorf("inkling2_loop planner: tokenizer has no %s special token", Inkling2EndOfTurnMarker)
	}
	return &Inkling2LoopPlanner{gen: gen, tok: tok, cfg: cfg, modelID: modelID, stops: stops, contextTokens: contextTokens}, nil
}

func (p *Inkling2LoopPlanner) generate(ctx context.Context, prompt []int, maxNew int, stops map[int]bool, emit func(int) bool) (Inkling2LoopGeneration, error) {
	if owner, ok := p.gen.(inkling2LoopGPUOwner); ok && owner.SerializesGPUAccess() {
		return p.gen.Generate(ctx, prompt, maxNew, stops, emit)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.gen.Generate(ctx, prompt, maxNew, stops, emit)
}

func (p *Inkling2LoopPlanner) Model() string { return p.modelID }

func (p *Inkling2LoopPlanner) StreamingSupported() bool { return true }

// CompleteStream delivers the content once the turn is decoded, like the default
// in-kernel stream: tool-call text must not reach the client before adjudication.
func (p *Inkling2LoopPlanner) CompleteStream(ctx context.Context, sink StreamSink, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	comp, err := p.Complete(ctx, messages, tools, opts...)
	if err != nil || sink == nil || comp == nil || comp.Message.Content == "" {
		return comp, err
	}
	if err := sink(comp.Message.Content); err != nil {
		return nil, err
	}
	return comp, nil
}

// Complete decodes greedily; the device forward selects the argmax on device, so
// temperature, top_p and top_k do not apply.
func (p *Inkling2LoopPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	sp := applySampleOpts(opts...)
	offered := offeredToolsFor(ctx, tools)
	ids, err := p.tok.Encode(inKernelRenderPrompt(messages, tools, p.cfg, sp))
	if err != nil {
		return nil, fmt.Errorf("inkling2_loop planner: encode prompt: %w", err)
	}
	if len(ids) == 0 {
		return nil, errors.New("inkling2_loop planner: empty prompt")
	}
	maxNew := Inkling2LoopDefaultMaxTokens
	if sp.MaxTokens != nil && *sp.MaxTokens > 0 {
		maxNew = *sp.MaxTokens
	}
	if p.contextTokens > 0 {
		if len(ids) >= p.contextTokens {
			return nil, fmt.Errorf("inkling2_loop planner: prompt of %d tokens leaves no room in the %d-token context", len(ids), p.contextTokens)
		}
		maxNew = min(maxNew, p.contextTokens-len(ids))
	}

	var out []int
	stopText := -1
	emit := func(id int) bool {
		out = append(out, id)
		if len(sp.Stop) == 0 {
			return true
		}
		text, err := p.tok.Decode(out)
		if err != nil {
			return true
		}
		for _, s := range sp.Stop {
			if s != "" && strings.Contains(text, s) {
				stopText = len(out)
				return false
			}
		}
		return true
	}
	res, err := p.generate(ctx, ids, maxNew, p.stops, emit)
	if err != nil {
		return nil, err
	}
	text, err := p.tok.Decode(out)
	if err != nil {
		return nil, fmt.Errorf("inkling2_loop planner: decode output: %w", err)
	}
	if stopText >= 0 {
		for _, s := range sp.Stop {
			if i := strings.Index(text, s); s != "" && i >= 0 {
				text = text[:i]
			}
		}
	}
	reasoning, content := SplitInkling2Output(text)
	finish := "length"
	if res.Stopped {
		finish = "stop"
	}
	comp := &Completion{
		Model:        p.modelID,
		Message:      Message{Role: "assistant", Content: content, ReasoningContent: reasoning},
		FinishReason: finish,
		Usage:        Usage{PromptTokens: len(ids), CompletionTokens: res.Generated, TotalTokens: len(ids) + res.Generated},
		Timings:      inkling2LoopTimings(len(ids), res),
	}
	if res.CachedPromptTokens > 0 {
		comp.Usage.PromptTokensDetails = &UsageTokenDetails{CachedTokens: res.CachedPromptTokens}
	}
	return normalizeInKernelToolCalls(comp, offered, sp.ToolChoice, tools, messages), nil
}
