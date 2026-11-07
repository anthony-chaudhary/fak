// Ported from modular/modular max/python/max/pipelines/kv_cache/checkpoint/{_checkpointer,_triggers,_heuristic/_at_event}.py@88ab8913af6b9298fe914e0f20d87a49e4172d0d (Apache-2.0 WITH LLVM-exception; Copyright (c) 2026 Modular Inc.). Direct port to fak-owned Go.

package model

// checkpoint_policy.go - dependency-free, engine-agnostic KV-cache checkpoint
// policy layer: lifecycle-event triggers plus heuristics that choose which
// token positions to persist. A later leaf wires it into the serve path beside
// gdn_align_cache.go's AlignStateManager.Step, which today snapshots every
// block boundary unconditionally.

import (
	"errors"
	"sort"
)

// RequestID identifies one request whose cache the policy may checkpoint.
type RequestID string

// EventKind is a point in a request's lifecycle.
type EventKind int

const (
	// EndOfPrompt marks where a prompt ends.
	EndOfPrompt EventKind = iota
	// Release marks where a request releases its cache.
	Release
)

// Event is something that happened to a request at a token position.
type Event struct {
	RequestID RequestID
	Kind      EventKind
	Position  int // the exact token index the event happened at
}

// Checkpoint is a token position of a request to persist the cache at.
type Checkpoint struct {
	RequestID RequestID
	Position  int // the exact token index to checkpoint at
}

// CandidateQuery answers the context's candidate queries: the last token
// position at or before position that the request can be checkpointed at now,
// or false if there is none. The engine injects it.
type CandidateQuery func(requestID RequestID, position int) (int, bool)

// Context is the context needed by checkpoint heuristics.
type Context struct {
	query      CandidateQuery
	events     map[EventKind][]Event
	candidates map[candidateKey]candidateResult
}

type candidateKey struct {
	requestID RequestID
	position  int
}

type candidateResult struct {
	position int
	found    bool
}

// NewContext initializes the context with the query that answers its
// candidates.
func NewContext(query CandidateQuery) *Context {
	return &Context{
		query:      query,
		events:     map[EventKind][]Event{},
		candidates: map[candidateKey]candidateResult{},
	}
}

// Events returns the events of kind reported since the last step.
func (c *Context) Events(kind EventKind) []Event {
	return c.events[kind]
}

// CandidateAtOrBefore returns the last checkpointable position at or before
// position, memoized per step so the engine query runs once per distinct ask.
func (c *Context) CandidateAtOrBefore(requestID RequestID, position int) (int, bool) {
	key := candidateKey{requestID, position}
	if r, ok := c.candidates[key]; ok {
		return r.position, r.found
	}
	p, found := c.query(requestID, position)
	c.candidates[key] = candidateResult{position: p, found: found}
	return p, found
}

// Record records an event.
func (c *Context) Record(event Event) {
	c.events[event.Kind] = append(c.events[event.Kind], event)
}

// Reset forgets the step's events and query results.
func (c *Context) Reset() {
	c.events = map[EventKind][]Event{}
	c.candidates = map[candidateKey]candidateResult{}
}

// Heuristic chooses checkpoints from the events recorded for a step.
// Instances must be comparable (pointer receivers) so Triggers can dedupe a
// heuristic subscribed to several kinds into one run per step.
type Heuristic interface {
	// Subscribe runs this heuristic on its own event kinds.
	Subscribe(triggers *Triggers)
	// Run returns the checkpoints this heuristic chose for the step's events.
	Run(ctx *Context) []Checkpoint
}

// Triggers decides when each subscribed heuristic's callback runs.
type Triggers struct {
	byKind map[EventKind][]Heuristic
}

// NewTriggers returns an empty trigger table.
func NewTriggers() *Triggers {
	return &Triggers{byKind: map[EventKind][]Heuristic{}}
}

// OnEvent runs h at the next step after any event of kinds.
func (t *Triggers) OnEvent(h Heuristic, kinds ...EventKind) {
	for _, kind := range kinds {
		t.byKind[kind] = append(t.byKind[kind], h)
	}
}

// Handles reports whether any heuristic runs on kind.
func (t *Triggers) Handles(kind EventKind) bool {
	return len(t.byKind[kind]) > 0
}

// Triggered returns the distinct heuristics any of kinds triggers, in
// first-seen order, so a heuristic subscribed to several pending kinds runs
// once.
func (t *Triggers) Triggered(kinds []EventKind) []Heuristic {
	var out []Heuristic
	for _, kind := range kinds {
		for _, h := range t.byKind[kind] {
			dup := false
			for _, seen := range out {
				if seen == h {
					dup = true
					break
				}
			}
			if !dup {
				out = append(out, h)
			}
		}
	}
	return out
}

// Checkpointer collects lifecycle events and runs heuristics.
type Checkpointer struct {
	ctx      *Context
	triggers *Triggers
	pending  map[EventKind]bool
}

// Register subscribes each heuristic and returns this checkpointer.
func (cp *Checkpointer) Register(heuristics ...Heuristic) *Checkpointer {
	for _, h := range heuristics {
		h.Subscribe(cp.triggers)
	}
	return cp
}

// Handles reports whether any heuristic runs on events of kind.
func (cp *Checkpointer) Handles(kind EventKind) bool {
	return cp.triggers.Handles(kind)
}

// Record records an event for the next checkpoint step. Call it before the
// request's cache can change, for example before releasing it. Events no
// heuristic handles are dropped.
func (cp *Checkpointer) Record(event Event) {
	if !cp.Handles(event.Kind) {
		return
	}
	cp.ctx.Record(event)
	cp.pending[event.Kind] = true
}

// Step runs each callback the step's events triggered, once, and returns the
// distinct checkpoints chosen.
func (cp *Checkpointer) Step() []Checkpoint {
	var checkpoints []Checkpoint
	seen := map[Checkpoint]bool{}
	add := func(more ...Checkpoint) {
		for _, c := range more {
			if !seen[c] {
				seen[c] = true
				checkpoints = append(checkpoints, c)
			}
		}
	}
	for _, h := range cp.triggers.Triggered(cp.pendingKinds()) {
		add(h.Run(cp.ctx)...)
	}
	cp.pending = map[EventKind]bool{}
	cp.ctx.Reset()
	return checkpoints
}

// pendingKinds lists the step's pending event kinds in stable order.
func (cp *Checkpointer) pendingKinds() []EventKind {
	kinds := make([]EventKind, 0, len(cp.pending))
	for kind := range cp.pending {
		kinds = append(kinds, kind)
	}
	sort.Slice(kinds, func(i, j int) bool { return kinds[i] < kinds[j] })
	return kinds
}

// ErrMissingCandidateQuery is returned by Build when an enabled checkpointer
// lacks the candidate_at_or_before query its context needs.
var ErrMissingCandidateQuery = errors.New("checkpoint: a checkpointer needs a candidate_at_or_before query")

// CheckpointerFactory builds a checkpointer once every query its context needs
// is provided.
type CheckpointerFactory struct {
	heuristics          []Heuristic
	candidateAtOrBefore CandidateQuery
	disabled            bool
}

// NewCheckpointerFactory returns a factory to configure and build a
// checkpointer.
func NewCheckpointerFactory() *CheckpointerFactory {
	return &CheckpointerFactory{}
}

// WithHeuristics adds heuristics for the checkpointer to run.
func (f *CheckpointerFactory) WithHeuristics(heuristics ...Heuristic) *CheckpointerFactory {
	f.heuristics = append(f.heuristics, heuristics...)
	return f
}

// WithCandidateAtOrBefore sets the query that answers the context's
// candidates.
func (f *CheckpointerFactory) WithCandidateAtOrBefore(query CandidateQuery) *CheckpointerFactory {
	f.candidateAtOrBefore = query
	return f
}

// Disabled makes the checkpointer drop every event.
func (f *CheckpointerFactory) Disabled() *CheckpointerFactory {
	f.disabled = true
	return f
}

// Build returns the checkpointer. It fails closed with
// ErrMissingCandidateQuery when the checkpointer is enabled and lacks a
// candidate query.
func (f *CheckpointerFactory) Build() (*Checkpointer, error) {
	if f.disabled {
		alwaysNone := func(RequestID, int) (int, bool) { return 0, false }
		return newCheckpointer(NewContext(alwaysNone)), nil
	}
	if f.candidateAtOrBefore == nil {
		return nil, ErrMissingCandidateQuery
	}
	cp := newCheckpointer(NewContext(f.candidateAtOrBefore))
	return cp.Register(f.heuristics...), nil
}

func newCheckpointer(ctx *Context) *Checkpointer {
	return &Checkpointer{ctx: ctx, triggers: NewTriggers(), pending: map[EventKind]bool{}}
}

// AtEventHeuristic checkpoints requests at the events of the kinds it is
// given. Each event's request is checkpointed at the last candidate at or
// before each position targets returns for the event, by default just the
// event's own position.
type AtEventHeuristic struct {
	kinds   []EventKind
	targets func(Event) []int
}

// NewAtEventHeuristic builds an AtEventHeuristic. A nil targets defaults to
// the event's own position.
func NewAtEventHeuristic(kinds []EventKind, targets func(Event) []int) *AtEventHeuristic {
	if targets == nil {
		targets = func(event Event) []int { return []int{event.Position} }
	}
	ks := make([]EventKind, len(kinds))
	copy(ks, kinds)
	return &AtEventHeuristic{kinds: ks, targets: targets}
}

// Subscribe runs this heuristic on its event kinds.
func (h *AtEventHeuristic) Subscribe(triggers *Triggers) {
	triggers.OnEvent(h, h.kinds...)
}

// Run returns a checkpoint per target that has a candidate at or before it.
func (h *AtEventHeuristic) Run(ctx *Context) []Checkpoint {
	var checkpoints []Checkpoint
	for _, kind := range h.kinds {
		for _, event := range ctx.Events(kind) {
			for _, target := range h.targets(event) {
				if position, ok := ctx.CandidateAtOrBefore(event.RequestID, target); ok {
					checkpoints = append(checkpoints, Checkpoint{RequestID: event.RequestID, Position: position})
				}
			}
		}
	}
	return checkpoints
}

// EndOfPromptHeuristic checkpoints where a prompt ends and where a replay of
// it resumes. A replay of the same prompt recomputes its last token to sample
// from, so it resumes one token short of the prompt's end.
func EndOfPromptHeuristic() *AtEventHeuristic {
	return NewAtEventHeuristic([]EventKind{EndOfPrompt}, func(event Event) []int {
		return []int{event.Position, event.Position - 1}
	})
}

// ReleaseHeuristic checkpoints a request where it releases its cache.
func ReleaseHeuristic() *AtEventHeuristic {
	return NewAtEventHeuristic([]EventKind{Release}, nil)
}
