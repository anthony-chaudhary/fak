package agent

import (
	"bytes"
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

// inkling2StopScanner watches generated ids for the first token at which the
// decoded output contains a stop string, and yields the output cut at the stops.
//
// Each token is decoded once and its bytes appended, so the scan is linear in the
// output. A new match must end inside the newest piece, so only the last
// longest-stop bytes before it need searching.
type inkling2StopScanner struct {
	decode  func(ids []int) (string, error)
	stops   []string
	longest int
	text    []byte
	err     error
	stopped bool
}

func newInkling2StopScanner(decode func(ids []int) (string, error), stops []string) *inkling2StopScanner {
	s := &inkling2StopScanner{decode: decode, stops: stops}
	for _, stop := range stops {
		s.longest = max(s.longest, len(stop))
	}
	return s
}

// push records id and reports whether generation should continue.
func (s *inkling2StopScanner) push(id int) bool {
	if s.err != nil {
		return true
	}
	piece, err := s.decode([]int{id})
	if err != nil {
		s.err = err
		return true
	}
	before := len(s.text)
	s.text = append(s.text, piece...)
	if s.longest == 0 {
		return true
	}
	window := s.text[max(0, before-s.longest+1):]
	for _, stop := range s.stops {
		if stop != "" && bytes.Contains(window, []byte(stop)) {
			s.stopped = true
			return false
		}
	}
	return true
}

// finish returns the generated text, cut at each stop in order when a stop ended generation.
func (s *inkling2StopScanner) finish() (string, error) {
	if s.err != nil {
		return "", fmt.Errorf("inkling2_loop planner: decode output: %w", s.err)
	}
	text := string(s.text)
	if s.stopped {
		for _, stop := range s.stops {
			if i := strings.Index(text, stop); stop != "" && i >= 0 {
				text = text[:i]
			}
		}
	}
	return text, nil
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

	scan := newInkling2StopScanner(p.tok.Decode, sp.Stop)
	res, err := p.generate(ctx, ids, maxNew, p.stops, scan.push)
	if err != nil {
		return nil, err
	}
	text, err := scan.finish()
	if err != nil {
		return nil, err
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
