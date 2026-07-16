package executionroute

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

// ---------------------------------------------------------------------------
// THE COMPATIBILITY DESCRIPTOR — what a session is, durably, across a move.
// ---------------------------------------------------------------------------

// DescriptorVersion is the current descriptor schema version. It is bumped
// only when a field's MEANING changes; adding an optional field is additive and
// does not bump it. A descriptor is refused when it declares a version this build
// cannot interpret (see SessionDescriptor.Validate), which is the whole point of
// versioning it: a session persisted by a newer build must not be silently
// mis-read by an older one that would ignore fields it has never heard of.
const DescriptorVersion = 1

// CompatAxis is one dimension along which two sessions may agree or differ. The
// set is CLOSED and each axis names a DISTINCT portability property, so a
// compatibility verdict can be explained per-axis instead of collapsing into one
// opaque "portable" boolean. Adding an axis is a new constant plus a comparison
// arm, never a free-text field.
type CompatAxis string

const (
	// AxisHarness is the agent harness that owns the session (claude / codex / ...).
	AxisHarness CompatAxis = "harness"
	// AxisWire is the provider wire protocol the session's turns were spoken over.
	AxisWire CompatAxis = "wire"
	// AxisModelFamily is the model lineage that produced the session's turns.
	AxisModelFamily CompatAxis = "model_family"
	// AxisToolProtocol is the encoding of tool calls and their results.
	AxisToolProtocol CompatAxis = "tool_protocol"
	// AxisTranscriptFormat is the on-the-wire shape of the conversation transcript.
	AxisTranscriptFormat CompatAxis = "transcript_format"
)

// StateKind is a CLOSED set of session state elements a move may need to carry.
// Each kind is BOUND to the axes it cannot survive a change of (see BoundAxes) —
// that binding, not a caller's assertion, is what decides translatability.
type StateKind string

const (
	// StateSystemPrompt is the system instruction: plain text, bound to nothing,
	// so it survives every move.
	StateSystemPrompt StateKind = "system_prompt"
	// StateMessages is the conversation transcript. Its serialization is the
	// context format, so it must be re-encoded when that format changes.
	StateMessages StateKind = "messages"
	// StateToolCalls is the tool-call/result pairs. Their encoding follows both the
	// tool protocol and the surrounding transcript format.
	StateToolCalls StateKind = "tool_calls"
	// StateThinking is extended-thinking / reasoning content. It is emitted (and
	// often signed or encrypted) by one model family and is not re-interpretable by
	// another, nor outside the transcript format that framed it.
	StateThinking StateKind = "thinking"
	// StateProviderKV is provider-side prefix cache. It is keyed by the exact
	// wire and model that minted it and lives on the provider's side of the
	// boundary, so it can never be re-pointed by translating local bytes.
	StateProviderKV StateKind = "provider_kv"
)

// BoundAxes reports the axes a state kind cannot survive a change of. A move that
// changes ANY bound axis leaves that state untranslatable; when such state is
// required, the move is refused. A kind bound to no axis (a system prompt is just
// text) travels everywhere. Order is stable so explanations are diffable.
func (k StateKind) BoundAxes() []CompatAxis {
	switch k {
	case StateSystemPrompt:
		return nil
	case StateMessages:
		return []CompatAxis{AxisTranscriptFormat}
	case StateToolCalls:
		return []CompatAxis{AxisToolProtocol, AxisTranscriptFormat}
	case StateThinking:
		return []CompatAxis{AxisModelFamily, AxisTranscriptFormat}
	case StateProviderKV:
		return []CompatAxis{AxisWire, AxisModelFamily}
	}
	return nil
}

func knownState(k StateKind) bool {
	switch k {
	case StateSystemPrompt, StateMessages, StateToolCalls, StateThinking, StateProviderKV:
		return true
	}
	return false
}

// SessionDescriptor is the versioned, durable record of what a session IS along
// every compatibility axis, plus the state a move of it must preserve. It is the
// evidence that replaces a caller's `portable: true` assertion: eligibility is
// COMPUTED by comparing two descriptors, so a caller cannot assert a session into
// a harness that could never read it.
type SessionDescriptor struct {
	// Version is the descriptor schema version; see DescriptorVersion.
	Version int `json:"version"`
	// ID is the session this describes ("" for a target envelope, which names no
	// existing session — it is the shape a session would be moved INTO).
	ID string `json:"id,omitempty"`

	Harness       string              `json:"harness,omitempty"`
	Wire          harnessprofile.Wire `json:"wire,omitempty"`
	ModelFamily   string              `json:"model_family,omitempty"`
	ToolProtocol  string              `json:"tool_protocol,omitempty"`
	TranscriptFormat string              `json:"transcript_format,omitempty"`

	// RequiredState names the state that MUST survive the move for it to be worth
	// making. State absent here is best-effort: it may be dropped without refusing.
	// A source that requires nothing can always be moved (there is nothing to lose).
	RequiredState []StateKind `json:"required_state,omitempty"`
}

// Validate reports whether the descriptor is interpretable by this build: it must
// carry a version this build knows, and every required state must name a known
// kind. It fails CLOSED on an unknown version — an unversioned (zero) descriptor
// is refused rather than assumed to be v1, and a future version is refused rather
// than read with fields silently ignored.
func (d SessionDescriptor) Validate() error {
	if d.Version == 0 {
		return fmt.Errorf("execution route: session descriptor is unversioned; set Version to %d", DescriptorVersion)
	}
	if d.Version > DescriptorVersion {
		return fmt.Errorf("execution route: session descriptor version %d is newer than this build understands (%d)", d.Version, DescriptorVersion)
	}
	seen := make(map[StateKind]bool, len(d.RequiredState))
	for _, s := range d.RequiredState {
		if !knownState(s) {
			return fmt.Errorf("execution route: unknown required state %q", s)
		}
		if seen[s] {
			return fmt.Errorf("execution route: required state %q is declared more than once", s)
		}
		seen[s] = true
	}
	return nil
}

// axisValue reads the descriptor's value on one axis, so comparison is table-
// driven rather than five hand-written field compares.
func (d SessionDescriptor) axisValue(a CompatAxis) string {
	switch a {
	case AxisHarness:
		return d.Harness
	case AxisWire:
		return string(d.Wire)
	case AxisModelFamily:
		return d.ModelFamily
	case AxisToolProtocol:
		return d.ToolProtocol
	case AxisTranscriptFormat:
		return d.TranscriptFormat
	}
	return ""
}

// compatAxes is the stable comparison order; it is also the explanation order.
var compatAxes = []CompatAxis{AxisHarness, AxisWire, AxisModelFamily, AxisToolProtocol, AxisTranscriptFormat}

// ---------------------------------------------------------------------------
// THE VERDICT — field-by-field, then a decision derived from it.
// ---------------------------------------------------------------------------

// AxisComparison is ONE axis's verdict, carrying both sides' values and why they
// were read as matching or differing. Every axis is reported — including the ones
// that match — so the explanation is complete rather than a list of complaints.
type AxisComparison struct {
	Axis   CompatAxis `json:"axis"`
	Source string     `json:"source,omitempty"`
	Target string     `json:"target,omitempty"`
	Match  bool       `json:"match"`
	Reason string     `json:"reason"`
}

// StateBlock records one state kind that cannot be carried across the move, and
// the axis whose change stranded it. Required marks whether that loss is fatal
// (the state was declared required) or merely dropped.
type StateBlock struct {
	State    StateKind  `json:"state"`
	Axis     CompatAxis `json:"axis"`
	Required bool       `json:"required"`
	Reason   string     `json:"reason"`
}

// CompatVerdict is the resolved portability of a move. The set is closed.
type CompatVerdict string

const (
	// CompatIdentical means every axis matches: the session can be resumed in place.
	CompatIdentical CompatVerdict = "identical"
	// CompatTranslatable means some axis differs, but no REQUIRED state is bound to
	// a differing axis: the session can be forked with translation.
	CompatTranslatable CompatVerdict = "translatable"
	// CompatIncompatible means a REQUIRED state is bound to a differing axis: the
	// state cannot be translated, so the move is refused.
	CompatIncompatible CompatVerdict = "incompatible"
)

// CompatResult is the inspectable record of one eligibility computation: the
// per-axis comparison, every stranded state, the resolved verdict, and the
// lifecycle Action that follows from it. Refused distinguishes a REFUSED move
// (state existed but could not be carried) from an ordinary fresh start.
type CompatResult struct {
	Verdict CompatVerdict `json:"verdict"`
	Action  SessionAction `json:"action"`
	ID      string        `json:"id,omitempty"`
	// Refused is true when a fork was denied because required state could not be
	// translated. The Action is then SessionStart: the caller may still begin fresh,
	// but it must not pretend the prior state came along.
	Refused bool `json:"refused,omitempty"`
	// Axes is the field-by-field comparison, in stable axis order, always complete.
	Axes []AxisComparison `json:"axes"`
	// Untranslatable lists every state stranded by a differing axis, required or not.
	Untranslatable []StateBlock `json:"untranslatable,omitempty"`
	Reason         string       `json:"reason"`
}

// Differs reports whether the named axis was read as different between the two
// descriptors.
func (d CompatResult) Differs(a CompatAxis) bool {
	for _, c := range d.Axes {
		if c.Axis == a {
			return !c.Match
		}
	}
	return false
}

// Blocked reports whether the named state was stranded, and whether that loss was
// fatal to the move.
func (d CompatResult) Blocked(k StateKind) (StateBlock, bool) {
	for _, b := range d.Untranslatable {
		if b.State == k {
			return b, true
		}
	}
	return StateBlock{}, false
}

// ---------------------------------------------------------------------------
// ELIGIBILITY — compute the move from two envelopes, never from an assertion.
// ---------------------------------------------------------------------------

// RouteCompat computes resume/fork/start eligibility by comparing the SOURCE
// session's descriptor against the TARGET execution envelope, and explains the
// result axis by axis.
//
// The rule is mechanical, in three steps:
//
//  1. Compare all five axes. If every axis matches, the target envelope is the one
//     that minted the session: resume it in place, nothing needs translating.
//  2. Otherwise, some axis changed. Strand every state kind BOUND to a changed axis
//     (StateKind.BoundAxes) — that state cannot be carried, by construction.
//  3. If any stranded state was declared REQUIRED, the move is refused: forking
//     would silently drop state the caller said it could not lose. Otherwise the
//     fork is eligible, carrying what translates and dropping what does not.
//
// A refusal returns Action SessionStart with Refused set, not an error: an
// incompatible move is a routing DECISION with a reason, not a malformed input.
// Errors are reserved for descriptors this build cannot interpret at all.
//
// It is pure and deterministic: no I/O, no goroutines, stable ordering throughout.
func RouteCompat(source, target SessionDescriptor) (CompatResult, error) {
	if err := source.Validate(); err != nil {
		return CompatResult{}, fmt.Errorf("source: %w", err)
	}
	if err := target.Validate(); err != nil {
		return CompatResult{}, fmt.Errorf("target: %w", err)
	}
	if source.ID == "" {
		return CompatResult{}, fmt.Errorf("execution route: source session descriptor has no ID; there is no session to move")
	}

	dec := CompatResult{ID: source.ID, Axes: make([]AxisComparison, 0, len(compatAxes))}
	changed := make(map[CompatAxis]bool, len(compatAxes))
	for _, axis := range compatAxes {
		src, tgt := source.axisValue(axis), target.axisValue(axis)
		cmp := AxisComparison{Axis: axis, Source: src, Target: tgt, Match: src == tgt}
		switch {
		case cmp.Match:
			cmp.Reason = fmt.Sprintf("%s matches (%s)", axis, orUnset(src))
		default:
			cmp.Reason = fmt.Sprintf("%s differs: source %s, target %s", axis, orUnset(src), orUnset(tgt))
			changed[axis] = true
		}
		dec.Axes = append(dec.Axes, cmp)
	}

	if len(changed) == 0 {
		dec.Verdict = CompatIdentical
		dec.Action = SessionResume
		dec.Reason = "every compatibility axis matches; the session resumes in its own envelope"
		return dec, nil
	}

	required := make(map[StateKind]bool, len(source.RequiredState))
	for _, s := range source.RequiredState {
		required[s] = true
	}
	// Walk the closed state set in declared order (not map order) so the stranded
	// list is stable across runs.
	fatal := make([]StateBlock, 0, len(source.RequiredState))
	for _, kind := range []StateKind{StateSystemPrompt, StateMessages, StateToolCalls, StateThinking, StateProviderKV} {
		for _, axis := range kind.BoundAxes() {
			if !changed[axis] {
				continue
			}
			block := StateBlock{
				State:    kind,
				Axis:     axis,
				Required: required[kind],
			}
			if block.Required {
				block.Reason = fmt.Sprintf("required state %s is bound to %s, which changed; it cannot be translated", kind, axis)
				fatal = append(fatal, block)
			} else {
				block.Reason = fmt.Sprintf("state %s is bound to %s, which changed; it is dropped (not required)", kind, axis)
			}
			dec.Untranslatable = append(dec.Untranslatable, block)
			break // the first changed bound axis strands it; one reason per state.
		}
	}

	if len(fatal) > 0 {
		dec.Verdict = CompatIncompatible
		dec.Action = SessionStart
		dec.Refused = true
		dec.Reason = fmt.Sprintf("fork refused: %s", fatal[0].Reason)
		return dec, nil
	}

	dec.Verdict = CompatTranslatable
	dec.Action = SessionFork
	dec.Reason = "axes differ but no required state is bound to a changed axis; the session forks with translation"
	return dec, nil
}

func orUnset(s string) string {
	if s == "" {
		return "(unset)"
	}
	return s
}
