package generationctl

// streamadapter.go is the provider-facing half of continuous generation: the
// bridge a real streaming adapter drives from its own SSE callback, so accepted
// output and speculative tool-call arguments reach the Controller as they
// arrive instead of after the turn (#6342, parent #6219).
//
// The load-bearing idea is that steering RESOLUTION is measured, not declared.
// An adapter says how fine a boundary it believes it can act on; the bridge
// records how the provider actually chunked the turn and refuses to let the two
// diverge silently. That matters because the two are not the same across real
// providers: every OpenAI-compatible endpoint recorded under testdata/captures
// delivers a tool call's whole argument object in ONE chunk, so the earliest
// point such an adapter can stop the call is the tool-call boundary — not the
// delta boundary its streaming shape suggests. A non-streaming adapter is the
// same claim taken further: its only boundary is between requests.
//
// The safety invariant the bridge exists to hold: arguments that are still
// speculative never become an effect. A tool call becomes admissible only at
// ToolCallBoundary, and only when the epoch is still open AND the provider
// sealed the block AND the accumulated arguments parse as complete JSON. A
// stream cut mid-arguments, or a redirect fired by a rule that matched those
// arguments, both leave the call inadmissible with a named reason.

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/streamrules"
)

// Resolution is the finest boundary at which an adapter can act on a steering
// directive. It is a property of the transport plus the provider's chunking,
// which is why the bridge measures it per turn.
type Resolution string

const (
	// ResolutionUnknown is the zero value: nothing was observed yet.
	ResolutionUnknown Resolution = "unknown"
	// ResolutionRequest can only steer between whole requests. Every
	// non-streaming adapter is here, and must say so.
	ResolutionRequest Resolution = "request"
	// ResolutionToolCall can steer between content blocks / tool calls, but a
	// single call's arguments arrive as one indivisible chunk.
	ResolutionToolCall Resolution = "tool-call"
	// ResolutionDelta can steer inside a content block, mid-arguments.
	ResolutionDelta Resolution = "delta"
)

// fineness orders resolutions so an overclaim is a comparison, not a table.
func (r Resolution) fineness() int {
	switch r {
	case ResolutionRequest:
		return 1
	case ResolutionToolCall:
		return 2
	case ResolutionDelta:
		return 3
	default:
		return 0
	}
}

// Adapter is what a provider adapter declares about itself before a turn. The
// declaration is the claim; the bridge's report is the check on it.
type Adapter struct {
	// Name identifies the adapter, e.g. "gateway/anthropic-passthrough".
	Name string `json:"name"`
	// Wire is the protocol shape, e.g. "anthropic-messages-sse".
	Wire string `json:"wire"`
	// Streaming is false for a buffered adapter that sees the whole turn at once.
	Streaming bool `json:"streaming"`
	// Declared is the resolution the adapter claims. A buffered adapter that
	// declares anything finer than ResolutionRequest is refused at Open.
	Declared Resolution `json:"declared_resolution"`
}

// Closed-vocabulary outcomes for a tool call at its safe boundary.
const (
	ToolAdmitted            = "ADMITTED"
	ToolRefusedRedirected   = "EPOCH_REDIRECTED"
	ToolRefusedIncomplete   = "ARGUMENTS_INCOMPLETE"
	ToolRefusedUnknownCall  = "UNKNOWN_TOOL_CALL"
	ToolRefusedAlreadyTaken = "BOUNDARY_ALREADY_TAKEN"
)

// ToolBoundary is the verdict on one proposed tool call at the adapter's next
// safe boundary. Arguments are populated only when Admit is true, so a caller
// cannot accidentally dispatch a refused call's speculative bytes.
type ToolBoundary struct {
	CallID    string `json:"call_id"`
	Tool      string `json:"tool"`
	Admit     bool   `json:"admit"`
	Reason    string `json:"reason"`
	Arguments string `json:"arguments,omitempty"`
	// Fragments is how many ordered argument chunks the provider sent for this
	// call. 1 means the arguments were not steerable below the call boundary.
	Fragments int `json:"fragments"`
}

// Report is the per-turn steering record an adapter emits. It is the artifact
// that makes "this adapter really did live steering" checkable by someone who
// did not watch the stream.
type Report struct {
	Adapter    Adapter `json:"adapter"`
	Trajectory string  `json:"trajectory_id"`
	Epoch      uint64  `json:"epoch"`

	ObservedText     Resolution `json:"observed_text_resolution"`
	ObservedToolArgs Resolution `json:"observed_tool_args_resolution"`
	// Effective is the resolution that actually bounds this turn's effects:
	// the tool-argument resolution when the turn proposed a tool, else the text
	// resolution. It is what Declared is judged against.
	Effective Resolution `json:"effective_resolution"`
	Verdict   string     `json:"verdict"`

	TextDeltas       int `json:"text_deltas"`
	ThinkingDeltas   int `json:"thinking_deltas"`
	ToolCalls        int `json:"tool_calls"`
	ToolArgFragments int `json:"tool_arg_fragments"`

	Boundaries []ToolBoundary `json:"boundaries,omitempty"`
	Steering   *Transition    `json:"steering,omitempty"`
	Checkpoint Checkpoint     `json:"checkpoint"`
}

// Closed-vocabulary resolution verdicts.
const (
	// ResolutionHonest means the adapter claimed no finer a boundary than the
	// provider actually gave it.
	ResolutionHonest = "RESOLUTION_HONEST"
	// ResolutionOverclaimed means the adapter claimed live steering it cannot
	// perform on this turn — the case #6342 exists to stop being silent.
	ResolutionOverclaimed = "RESOLUTION_OVERCLAIMED"
	// ResolutionUnavailable means the turn produced nothing to measure.
	ResolutionUnavailable = "RESOLUTION_UNAVAILABLE"
)

// ErrEpochClosed is returned when an adapter keeps feeding a bridge after a
// steering point closed the epoch. It is the signal to stop reading upstream
// and cancel at the adapter's next safe boundary.
var ErrEpochClosed = errors.New("generationctl: generation epoch is closed")

type toolStream struct {
	callID    string
	name      string
	path      string
	args      strings.Builder
	fragments int
	sealed    bool
	settled   bool
}

// StreamBridge feeds one provider turn into a Controller. It is not safe for
// concurrent use: a provider stream is ordered, and the bridge preserves that
// order deliberately.
type StreamBridge struct {
	adapter Adapter
	ctl     *Controller

	textDeltas     int
	thinkingDeltas int

	tools     map[string]*toolStream
	toolOrder []string
	byIndex   map[int]string

	steering   *Transition
	boundaries []ToolBoundary
}

// Open starts a generation epoch for a streaming adapter and returns the bridge
// its callback drives.
func Open(a Adapter, trajectoryID, owner string, compute Compute, rules []streamrules.Rule) (*StreamBridge, error) {
	if err := validateAdapter(a); err != nil {
		return nil, err
	}
	ctl, err := New(trajectoryID, owner, compute, rules)
	if err != nil {
		return nil, err
	}
	return newBridge(a, ctl), nil
}

// Reopen resumes a trajectory from a checkpoint under a (possibly different)
// adapter, owner, and compute placement, and returns the bridge for the next
// epoch. This is the restart half of the done condition: the accepted prefix
// crosses the boundary, the speculative bytes do not.
func Reopen(a Adapter, cp Checkpoint, owner string, compute Compute, rules []streamrules.Rule) (*StreamBridge, error) {
	if err := validateAdapter(a); err != nil {
		return nil, err
	}
	ctl, err := Resume(cp, owner, compute, rules)
	if err != nil {
		return nil, err
	}
	return newBridge(a, ctl), nil
}

func newBridge(a Adapter, ctl *Controller) *StreamBridge {
	return &StreamBridge{
		adapter: a,
		ctl:     ctl,
		tools:   map[string]*toolStream{},
		byIndex: map[int]string{},
	}
}

func validateAdapter(a Adapter) error {
	if strings.TrimSpace(a.Name) == "" {
		return errors.New("generationctl: adapter name is required")
	}
	switch a.Declared {
	case ResolutionRequest, ResolutionToolCall, ResolutionDelta:
	default:
		return fmt.Errorf("generationctl: adapter %q must declare a steering resolution", a.Name)
	}
	// A buffered adapter cannot see inside a turn, so it may not claim it can.
	// Refusing here is what keeps "non-streaming" from quietly reading as live.
	if !a.Streaming && a.Declared != ResolutionRequest {
		return fmt.Errorf("generationctl: non-streaming adapter %q declared %q; only %q is available between requests",
			a.Name, a.Declared, ResolutionRequest)
	}
	return nil
}

// Controller exposes the underlying epoch state for callers that need it.
func (b *StreamBridge) Controller() *Controller { return b.ctl }

// Epoch is the epoch this bridge is feeding.
func (b *StreamBridge) Epoch() Epoch { return b.ctl.Epoch() }

// Cancelled reports whether a steering point closed the epoch. An adapter polls
// it after every fed delta and stops at its next safe boundary once it is true.
func (b *StreamBridge) Cancelled() bool { return b.steering != nil }

// Steering is the transition that closed the epoch, or nil.
func (b *StreamBridge) Steering() *Transition { return b.steering }

// Checkpoint is the committed prefix at the current boundary.
func (b *StreamBridge) Checkpoint() Checkpoint { return b.ctl.Checkpoint() }

// Text feeds one accepted output delta. The bytes join the durable prefix and
// are checked against text-scope rules in the same step.
func (b *StreamBridge) Text(delta string) (Transition, error) {
	return b.feedDelta(delta, &b.textDeltas, func(delta string) (Transition, error) {
		return b.ctl.ObserveText(streamrules.StreamKey{Scope: streamrules.ScopeText}, delta)
	})
}

// Thinking feeds one reasoning delta. It is steerable but never accepted.
func (b *StreamBridge) Thinking(delta string) (Transition, error) {
	return b.feedDelta(delta, &b.thinkingDeltas, func(delta string) (Transition, error) {
		return b.ctl.ObserveThinking(streamrules.StreamKey{Scope: streamrules.ScopeThinking}, delta)
	})
}

// feedDelta applies the provider bridge's common boundary semantics before handing a
// non-empty delta to its scope-specific controller method. Counting happens only after
// the epoch-open and non-empty gates, exactly once per observed provider fragment.
func (b *StreamBridge) feedDelta(delta string, count *int, observe func(string) (Transition, error)) (Transition, error) {
	if b.steering != nil {
		return Transition{}, ErrEpochClosed
	}
	if delta == "" {
		return Transition{Directive: Directive{Kind: Continue}}, nil
	}
	(*count)++
	tr, err := observe(delta)
	if err != nil {
		return Transition{}, err
	}
	return b.settle(tr), nil
}

// ToolCallStart registers a proposed tool call. Calling it twice for the same
// id is a no-op, so an adapter whose wire repeats the header per fragment (the
// OpenAI-compatible shape) needs no de-duplication of its own.
func (b *StreamBridge) ToolCallStart(callID, tool, path string) error {
	if b.steering != nil {
		return ErrEpochClosed
	}
	if strings.TrimSpace(callID) == "" {
		return errors.New("generationctl: tool call id is required")
	}
	if ts, ok := b.tools[callID]; ok {
		if tool != "" {
			ts.name = tool
		}
		if path != "" {
			ts.path = path
		}
		return nil
	}
	b.tools[callID] = &toolStream{callID: callID, name: tool, path: path}
	b.toolOrder = append(b.toolOrder, callID)
	return nil
}

// BindIndex associates a provider's content-block or tool-call index with a
// call id, because most wires send the id once and the index on every fragment.
func (b *StreamBridge) BindIndex(index int, callID string) { b.byIndex[index] = callID }

// CallIDForIndex resolves a provider index to a registered call id.
func (b *StreamBridge) CallIDForIndex(index int) (string, bool) {
	id, ok := b.byIndex[index]
	return id, ok
}

// ToolArgs feeds one ordered fragment of a tool call's arguments. The bytes are
// speculative: they are checked against the rules and never accepted.
func (b *StreamBridge) ToolArgs(callID, delta string) (Transition, error) {
	if b.steering != nil {
		return Transition{}, ErrEpochClosed
	}
	ts, ok := b.tools[callID]
	if !ok {
		return Transition{}, fmt.Errorf("generationctl: unregistered tool call %q", callID)
	}
	if ts.sealed {
		return Transition{}, fmt.Errorf("generationctl: tool call %q already sealed", callID)
	}
	if delta == "" {
		return Transition{Directive: Directive{Kind: Continue}}, nil
	}
	ts.args.WriteString(delta)
	ts.fragments++
	tr, err := b.ctl.ObserveToolDelta(streamrules.StreamKey{
		ToolCallID: callID,
		ToolName:   ts.name,
		Path:       ts.path,
		Scope:      streamrules.ScopeNamedTool,
	}, delta)
	if err != nil {
		return Transition{}, err
	}
	return b.settle(tr), nil
}

// SealToolCall marks the provider as having finished this call's arguments. An
// unsealed call is never admissible, which is what makes a stream truncated
// mid-arguments fail closed instead of dispatching a prefix.
func (b *StreamBridge) SealToolCall(callID string) error {
	ts, ok := b.tools[callID]
	if !ok {
		return fmt.Errorf("generationctl: unregistered tool call %q", callID)
	}
	ts.sealed = true
	return nil
}

// SealAll seals every registered call. Adapters whose wire has one terminal
// frame for the whole message (finish_reason, message_delta) call this there.
func (b *StreamBridge) SealAll() {
	for _, id := range b.toolOrder {
		b.tools[id].sealed = true
	}
}

// ToolCallBoundary is the adapter's next safe boundary for one proposed call:
// the single place a speculative tool call may become an effect. It answers
// once per call; a second ask is refused so a caller cannot re-roll a verdict.
func (b *StreamBridge) ToolCallBoundary(callID string) ToolBoundary {
	ts, ok := b.tools[callID]
	if !ok {
		out := ToolBoundary{CallID: callID, Reason: ToolRefusedUnknownCall}
		b.boundaries = append(b.boundaries, out)
		return out
	}
	out := ToolBoundary{CallID: callID, Tool: ts.name, Fragments: ts.fragments}
	switch {
	case ts.settled:
		out.Reason = ToolRefusedAlreadyTaken
	case b.steering != nil:
		// The epoch was redirected. Whatever these arguments were about to do,
		// they do not happen: this is the cancel-at-the-next-safe-boundary step.
		out.Reason = ToolRefusedRedirected
	case !ts.sealed || !json.Valid([]byte(ts.args.String())):
		out.Reason = ToolRefusedIncomplete
	default:
		out.Admit = true
		out.Reason = ToolAdmitted
		out.Arguments = ts.args.String()
	}
	ts.settled = true
	b.boundaries = append(b.boundaries, out)
	return out
}

// Boundaries takes the verdict for every proposed call, in arrival order. It is
// the batch form an adapter uses at its terminal frame.
func (b *StreamBridge) Boundaries() []ToolBoundary {
	out := make([]ToolBoundary, 0, len(b.toolOrder))
	for _, id := range b.toolOrder {
		if b.tools[id].settled {
			continue
		}
		out = append(out, b.ToolCallBoundary(id))
	}
	return out
}

// Steer applies an external live directive (an operator redirect, a fork, a
// stop) at the current point, closing the epoch at a checkpoint.
func (b *StreamBridge) Steer(d Directive) (Transition, error) {
	if b.steering != nil {
		return Transition{}, ErrEpochClosed
	}
	tr, err := b.ctl.Steer(d)
	if err != nil {
		return Transition{}, err
	}
	return b.settle(tr), nil
}

// settle latches the first transition that closed the epoch.
func (b *StreamBridge) settle(tr Transition) Transition {
	if tr.Checkpoint != nil && b.steering == nil {
		latched := tr
		b.steering = &latched
	}
	return tr
}

// observedText is the measured text-side resolution.
func (b *StreamBridge) observedText() Resolution {
	if !b.adapter.Streaming {
		return ResolutionRequest
	}
	if b.textDeltas+b.thinkingDeltas > 1 {
		return ResolutionDelta
	}
	if b.textDeltas+b.thinkingDeltas == 1 {
		return ResolutionToolCall
	}
	return ResolutionUnknown
}

// observedToolArgs is the measured tool-argument resolution: fine only when
// some call's arguments actually arrived in more than one ordered fragment.
func (b *StreamBridge) observedToolArgs() Resolution {
	if len(b.toolOrder) == 0 {
		return ResolutionUnknown
	}
	if !b.adapter.Streaming {
		return ResolutionRequest
	}
	for _, id := range b.toolOrder {
		if b.tools[id].fragments > 1 {
			return ResolutionDelta
		}
	}
	return ResolutionToolCall
}

// Report folds the turn into the steering record, taking any outstanding tool
// boundaries first so a report can never claim a call was left unjudged.
func (b *StreamBridge) Report() Report {
	b.Boundaries()

	text, tool := b.observedText(), b.observedToolArgs()
	effective := text
	if tool != ResolutionUnknown {
		effective = tool
	}

	verdict := ResolutionUnavailable
	switch {
	case effective == ResolutionUnknown:
	case b.adapter.Declared.fineness() > effective.fineness():
		verdict = ResolutionOverclaimed
	default:
		verdict = ResolutionHonest
	}

	frags := 0
	for _, id := range b.toolOrder {
		frags += b.tools[id].fragments
	}

	return Report{
		Adapter:          b.adapter,
		Trajectory:       b.ctl.Epoch().TrajectoryID,
		Epoch:            b.ctl.Epoch().Number,
		ObservedText:     text,
		ObservedToolArgs: tool,
		Effective:        effective,
		Verdict:          verdict,
		TextDeltas:       b.textDeltas,
		ThinkingDeltas:   b.thinkingDeltas,
		ToolCalls:        len(b.toolOrder),
		ToolArgFragments: frags,
		Boundaries:       append([]ToolBoundary(nil), b.boundaries...),
		Steering:         b.steering,
		Checkpoint:       b.ctl.Checkpoint(),
	}
}
