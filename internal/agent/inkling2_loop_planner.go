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
	// FirstEmitSeconds is request receipt to the first emitted id, 0 when no id was emitted
	// or the engine does not stamp it.
	FirstEmitSeconds float64
	// Width is how many requests shared the forward that decoded this one; 0 is unknown.
	Width int
}

// inkling2LoopTimings is the turn's llama.cpp timings plus the width it decoded in.
func inkling2LoopTimings(promptTokens int, res Inkling2LoopGeneration) *Timings {
	t := NewTimings(promptTokens, res.CachedPromptTokens, res.Generated, res.PrefillSeconds, res.DecodeSeconds)
	if res.FirstEmitSeconds > 0 {
		t.FirstEmitMS = res.FirstEmitSeconds * 1e3
	}
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
	// release, when set, receives once per token the new run of output no later
	// stop can cut (possibly empty), in order, and returns false to end generation.
	release  func(raw []byte) bool
	released int
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
	if s.longest > 0 {
		window := s.text[max(0, before-s.longest+1):]
		for _, stop := range s.stops {
			if stop != "" && bytes.Contains(window, []byte(stop)) {
				s.stopped = true
				return false
			}
		}
	}
	return s.releaseSafe()
}

// releaseSafe hands on the output before the longest suffix that could still
// grow into a stop. A stop not yet matched must end in a later piece, so it
// starts inside that suffix and finish never cuts released bytes.
func (s *inkling2StopScanner) releaseSafe() bool {
	if s.release == nil {
		return true
	}
	end := len(s.text)
	for _, stop := range s.stops {
		for n := min(len(stop)-1, len(s.text)); n > 0 && len(s.text)-n < end; n-- {
			if bytes.HasSuffix(s.text, []byte(stop[:n])) {
				end = len(s.text) - n
				break
			}
		}
	}
	if end <= s.released {
		return s.release(nil)
	}
	raw := s.text[s.released:end]
	s.released = end
	return s.release(raw)
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

// CompleteStream streams content while the turn decodes: each delta is text that
// the finished turn's content provably starts with, so tool-call text never reaches
// the client before adjudication. Reasoning streams the same way to the reasoning
// sink on ctx, as a prefix of the finished turn's reasoning. The returned Completion
// is the one Complete builds, and the deltas concatenate to its content. A forced
// tool choice may rewrite content, so that turn is delivered once decoded. Text
// streamed before a decode error stays delivered; the error follows it. Tokens that
// deliver nothing to either sink report held activity so the caller can keep the
// stream warm: the first one before any delivery, then one per
// inkling2HeldActivityEvery in a row.
func (p *Inkling2LoopPlanner) CompleteStream(ctx context.Context, sink StreamSink, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	sp := applySampleOpts(opts...)
	if sink == nil {
		return p.complete(ctx, messages, tools, sp, nil)
	}
	held := newInkling2HeldActivity(streamHeldActivity(ctx))
	if inKernelEffectiveToolName(sp.ToolChoice, tools) != "" {
		var release func([]byte) bool
		if held != nil {
			release = func([]byte) bool { held.token(false); return true }
		}
		comp, err := p.complete(ctx, messages, tools, sp, release)
		if err != nil || comp == nil || comp.Message.Content == "" {
			return comp, err
		}
		if err := sink(comp.Message.Content); err != nil {
			return nil, err
		}
		return comp, nil
	}
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	projector := newInkling2ContentStream(len(offeredToolsFor(ctx, tools).names) > 0)
	reasoningSink := streamReasoningSink(ctx)
	pump := startInkling2StreamPump(sink, reasoningSink, cancel)
	defer pump.stop()
	comp, err := p.complete(streamCtx, messages, tools, sp, func(raw []byte) bool {
		content, reasoning := projector.feed(raw)
		if held != nil {
			held.token(content != "" || (reasoningSink != nil && reasoning != ""))
		}
		live := pump.put(content, false)
		if reasoningSink != nil {
			live = pump.put(reasoning, true) && live
		}
		return live
	})
	if sinkErr := pump.stop(); sinkErr != nil {
		return nil, sinkErr
	}
	if err != nil {
		return nil, err
	}
	streamed := projector.emitted.String()
	if !strings.HasPrefix(comp.Message.Content, streamed) {
		return nil, fmt.Errorf("inkling2_loop planner: streamed content is not a prefix of the final content")
	}
	if reasoningSink != nil && !strings.HasPrefix(comp.Message.ReasoningContent, projector.thought.String()) {
		return nil, fmt.Errorf("inkling2_loop planner: streamed reasoning is not a prefix of the final reasoning")
	}
	if rest := comp.Message.Content[len(streamed):]; rest != "" {
		if err := sink(rest); err != nil {
			return nil, err
		}
	}
	return comp, nil
}

// Complete decodes greedily; the device forward selects the argmax on device, so
// temperature, top_p and top_k do not apply.
func (p *Inkling2LoopPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	return p.complete(ctx, messages, tools, applySampleOpts(opts...), nil)
}

// complete is Complete with release receiving each run of raw output no stop can cut.
func (p *Inkling2LoopPlanner) complete(ctx context.Context, messages []Message, tools []ToolDef, sp SampleParams, release func(raw []byte) bool) (*Completion, error) {
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
			return nil, &InKernelContextLengthError{PromptTokens: len(ids), MaxNewTokens: maxNew, MaxContext: p.contextTokens}
		}
		maxNew = min(maxNew, p.contextTokens-len(ids))
	}

	scan := newInkling2StopScanner(p.tok.Decode, sp.Stop)
	scan.release = release
	res, err := p.generate(ctx, ids, maxNew, p.stops, scan.push)
	if err != nil {
		var stage *model.BackendForwardOperationError
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || IsLocalForwardError(err) || errors.As(err, &stage) {
			return nil, err
		}
		return nil, NewLocalForwardError("Inkling-2 generation", err, classifiedDeviceOOM(err))
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

// inkling2HeldActivityEvery bounds held reports on a long held span. Each report
// takes a slot in the caller's bounded event queue, so a report per held token
// would halve its room for content when held and content tokens alternate.
const inkling2HeldActivityEvery = 32

type inkling2HeldActivity struct {
	report   func()
	streamed bool
	run      int
}

func newInkling2HeldActivity(report func()) *inkling2HeldActivity {
	if report == nil {
		return nil
	}
	return &inkling2HeldActivity{report: report}
}

func (h *inkling2HeldActivity) token(content bool) {
	if content {
		h.streamed, h.run = true, 0
		return
	}
	h.run++
	if (!h.streamed && h.run == 1) || h.run%inkling2HeldActivityEvery == 0 {
		h.report()
	}
}
