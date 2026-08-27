// Package inputtrigger classifies WHAT SHAPE OF INPUT triggered an admitted turn —
// once, at ingress — into a small closed vocabulary a routing policy can match on.
//
// THE GAP IT FILLS. "Is this turn a tool result coming back, a user typing, or a
// prefilled assistant continuation?" is a fact about the turn's ENVELOPE, and every
// consumer that wants it today re-derives it from the raw prompt: one seam peeks at the
// last role, another sniffs for a tool_call_id, a third greps the text. Each derivation
// is a second taxonomy that drifts from the others, and a drifting taxonomy is how a
// route silently changes meaning. So the shape is observed ONE time, by this classifier,
// and carried as a typed value (modelroute.Subject.InputTrigger) that policy READS.
// Route policy may branch on the enum; it must never re-parse the prompt to re-decide.
//
// A TRIGGER IS A ROUTING HINT, NEVER AN AUTHORIZATION FACT. It may select a model, a
// provider, or a cache route. It may NOT stand in for adjudication: nothing here decides
// whether a tool call is permitted, whether a payload may leave the box, or which
// capability floor applies — those remain the adjudicator's and the tier policy's, over
// their own inputs. The turn's messages are attacker-influenced (a model can emit an
// assistant message that LOOKS like a tool result), so a trigger buys a route, never a
// permission. modelroute.ClassOf / PolicyFor deliberately do not read this field.
//
// FAIL-CONSERVATIVE, NEVER OPTIMISTIC. Classify answers Other for anything it cannot
// name with certainty: an empty turn, a turn carrying a role outside the closed set, a
// trailing tool message with no tool_call_id linking it to a call, a trailing assistant
// message with nothing to continue. Other routes to whatever the manifest's general
// rules or fail-closed default say, so an unrecognizable turn can never TALK ITS WAY
// INTO the cheap tool-result lane by being malformed.
//
// The package is pure (stdlib only) and imports nothing internal, so the gateway, the
// model-route spine, and the agent loop can all share the one taxonomy.
package inputtrigger

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
	"unicode/utf8"
)

// Trigger is the closed, typed vocabulary of turn-input shapes. It is CLOSED
// deliberately: a new shape is a new constant plus a Classify arm plus a test, never a
// free-text string a manifest invents. The zero value is the empty Trigger — "not
// classified" — which is distinct from Other ("classified, and it is nothing we name").
type Trigger string

const (
	// Other is every input the classifier will not name: empty, mixed, unknown, or
	// structurally incomplete. It is the conservative sink, not an error.
	Other Trigger = "other"
	// SystemOnly is a turn whose messages are ALL system messages — a bare policy /
	// instruction load with no user or tool content behind it.
	SystemOnly Trigger = "system_only"
	// UserMessage is a turn whose newest message is a user message: a human (or a
	// caller acting as one) speaking.
	UserMessage Trigger = "user_message"
	// AssistantPrefill is a turn whose newest message is a non-empty assistant
	// message: partial assistant text the model is being asked to CONTINUE.
	AssistantPrefill Trigger = "assistant_prefill"
	// ToolResult is a turn whose newest message is a tool result linked to the call it
	// answers — the agent-loop continuation shape.
	ToolResult Trigger = "tool_result"
)

// Known reports whether t is one of the closed Trigger set. The empty Trigger
// (unclassified) is NOT known: a policy that wants "any" leaves its field unset rather
// than naming the empty value.
func Known(t Trigger) bool {
	switch t {
	case Other, SystemOnly, UserMessage, AssistantPrefill, ToolResult:
		return true
	}
	return false
}

// Role is the author of one admitted message, over the closed set the classifier
// recognizes. Anything else (a provider's bespoke "developer" role, a typo, an empty
// role) is unrecognized and drives the whole turn to Other.
type Role string

const (
	// RoleSystem is an instruction / policy message.
	RoleSystem Role = "system"
	// RoleUser is a user (or caller-as-user) message.
	RoleUser Role = "user"
	// RoleAssistant is a model message.
	RoleAssistant Role = "assistant"
	// RoleTool is a tool result returned to the model.
	RoleTool Role = "tool"
)

// ParseRole normalizes a wire role token (trimmed, lower-cased) against the closed set.
// It is the ONE place a raw role string is interpreted, so an ingress seam that reads
// roles off the wire and this classifier can never disagree about what "System " means.
func ParseRole(raw string) (Role, bool) {
	switch r := Role(strings.ToLower(strings.TrimSpace(raw))); r {
	case RoleSystem, RoleUser, RoleAssistant, RoleTool:
		return r, true
	}
	return "", false
}

// Message is one message of an admitted turn, reduced to the only three facts
// classification needs: who authored it, whether it carries content, and — for a tool
// message — which call it answers. It is deliberately NOT the wire message type: the
// classifier never sees (and so can never route on) attachments, names, or metadata.
type Message struct {
	Role Role
	// Content is the message text. Only its emptiness is ever inspected — the
	// classifier reads SHAPE, never meaning, so no prompt text can steer a route.
	Content string
	// ToolCallID links a tool message to the call it answers. A tool message without
	// one is not a tool result; it is an unlinked fragment, and classifies as Other.
	ToolCallID string
}

// Classify names the shape of an admitted turn. It is pure and deterministic: the same
// turn always yields the same Trigger, so a decision can be replayed from an audit log.
//
// The precedence, in order:
//
//  1. An EMPTY turn is Other. There is no shape to name.
//  2. A turn carrying ANY unrecognized role is Other — whole-turn, not per-message. A
//     turn we only partly understand is a turn we do not understand, and guessing from
//     the part we can read is exactly the optimism this classifier refuses.
//  3. Otherwise the NEWEST (last) message decides, because it is the message that
//     triggered this turn:
//     - tool     -> ToolResult, but ONLY with a non-blank ToolCallID; an unlinked tool
//     message is Other. This is the one arm an attacker would want to
//     forge (the tool-result lane is the cheap continuation lane), so it
//     is the arm that demands the structural link to a real call.
//     - assistant-> AssistantPrefill, but ONLY with non-blank content; an empty
//     trailing assistant message has nothing to continue, so its shape is
//     ambiguous and it is Other.
//     - user     -> UserMessage. The role is explicit and unforgeable-by-shape; a blank
//     user message is still a user-shaped turn.
//     - system   -> SystemOnly only when EVERY message is a system message. A system
//     message appended AFTER user or tool content is a mixed turn, not a
//     bare instruction load, and is Other.
//
// Blank means empty or whitespace-only throughout.
func Classify(turn []Message) Trigger {
	if len(turn) == 0 {
		return Other
	}
	for _, m := range turn {
		if _, ok := ParseRole(string(m.Role)); !ok {
			return Other
		}
	}
	last := turn[len(turn)-1]
	role, _ := ParseRole(string(last.Role))
	switch role {
	case RoleTool:
		if strings.TrimSpace(last.ToolCallID) == "" {
			return Other
		}
		return ToolResult
	case RoleAssistant:
		if strings.TrimSpace(last.Content) == "" {
			return Other
		}
		return AssistantPrefill
	case RoleUser:
		return UserMessage
	case RoleSystem:
		for _, m := range turn {
			if r, _ := ParseRole(string(m.Role)); r != RoleSystem {
				return Other
			}
		}
		return SystemOnly
	}
	return Other
}

// Schema is the immutable ingress-trigger receipt schema.
const Schema = "fak.input-trigger/1"

const (
	MaxMetadataEntries  = 8
	MaxMetadataKeyBytes = 32
	MaxMetadataValBytes = 128
)

// Provenance names how the gateway learned the causal trigger. It is closed so
// route receipts have bounded cardinality and cannot become free-text log
// channels.
type Provenance string

const (
	ProvenanceExplicit             Provenance = "caller_explicit"
	ProvenanceRetry                Provenance = "retry"
	ProvenanceCompatibilityMissing Provenance = "compatibility_missing"
)

// KnownProvenance reports whether p is in the closed ingress provenance set.
func KnownProvenance(p Provenance) bool {
	switch p {
	case ProvenanceExplicit, ProvenanceRetry, ProvenanceCompatibilityMissing:
		return true
	}
	return false
}

// Explicit is the untrusted wire form accepted from a fak-aware caller.
type Explicit struct {
	Classification Trigger           `json:"classification"`
	Provenance     Provenance        `json:"provenance"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

// UnmarshalJSON rejects additive guesses at the explicit trust boundary.
func (e *Explicit) UnmarshalJSON(data []byte) error {
	type explicitWire Explicit
	var wire explicitWire
	if err := decodeStrict(data, &wire); err != nil {
		return fmt.Errorf("input_trigger: %w", err)
	}
	decoded := Explicit(wire)
	if err := validateExplicit(decoded); err != nil {
		return err
	}
	*e = decoded
	return nil
}

// InputTrigger is the validated immutable value created at gateway ingress. All
// fields are unexported and caller metadata is reduced to a bounded count.
type InputTrigger struct {
	classification Trigger
	provenance     Provenance
	metadataCount  int
}

// AdmitTurn classifies the turn exactly once. A nil explicit value is the
// compatibility path; an explicit classification must match the observed
// envelope, including for retry provenance.
func AdmitTurn(turn []Message, explicit *Explicit) (InputTrigger, error) {
	classification := Classify(turn)
	if explicit == nil {
		return InputTrigger{
			classification: classification,
			provenance:     ProvenanceCompatibilityMissing,
		}, nil
	}
	if err := validateExplicit(*explicit); err != nil {
		return InputTrigger{}, err
	}
	if explicit.Classification != classification {
		return InputTrigger{}, errors.New("input_trigger: explicit classification does not match ingress classification")
	}
	return InputTrigger{
		classification: classification,
		provenance:     explicit.Provenance,
		metadataCount:  len(explicit.Metadata),
	}, nil
}

func (t InputTrigger) Classification() Trigger { return t.classification }
func (t InputTrigger) Provenance() Provenance  { return t.provenance }
func (t InputTrigger) MetadataCount() int      { return t.metadataCount }
func (t InputTrigger) Validate() error {
	if !Known(t.classification) {
		return errors.New("input_trigger: unknown classification")
	}
	if !KnownProvenance(t.provenance) {
		return errors.New("input_trigger: unknown provenance")
	}
	if t.metadataCount < 0 || t.metadataCount > MaxMetadataEntries {
		return fmt.Errorf("input_trigger: metadata count %d exceeds limit %d", t.metadataCount, MaxMetadataEntries)
	}
	if t.provenance == ProvenanceCompatibilityMissing && t.metadataCount != 0 {
		return errors.New("input_trigger: compatibility-missing trigger cannot carry metadata")
	}
	return nil
}

type metadataReceipt struct {
	Entries  int  `json:"entries"`
	Redacted bool `json:"redacted"`
}

type triggerReceipt struct {
	Schema         string           `json:"schema"`
	Classification Trigger          `json:"classification"`
	Provenance     Provenance       `json:"provenance"`
	Metadata       *metadataReceipt `json:"metadata,omitempty"`
}

// MarshalJSON emits only the closed classification/provenance plus a redacted
// metadata count.
func (t InputTrigger) MarshalJSON() ([]byte, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	wire := triggerReceipt{
		Schema:         Schema,
		Classification: t.classification,
		Provenance:     t.provenance,
	}
	if t.metadataCount > 0 {
		wire.Metadata = &metadataReceipt{Entries: t.metadataCount, Redacted: true}
	}
	return json.Marshal(wire)
}

// Parse validates the bounded receipt form during replay.
func Parse(data []byte) (InputTrigger, error) {
	var wire triggerReceipt
	if err := decodeStrict(data, &wire); err != nil {
		return InputTrigger{}, fmt.Errorf("input_trigger: parse receipt: %w", err)
	}
	if wire.Schema != Schema {
		return InputTrigger{}, errors.New("input_trigger: unexpected receipt schema")
	}
	t := InputTrigger{classification: wire.Classification, provenance: wire.Provenance}
	if wire.Metadata != nil {
		if wire.Metadata.Entries <= 0 || !wire.Metadata.Redacted {
			return InputTrigger{}, errors.New("input_trigger: receipt metadata must be a positive redacted count")
		}
		t.metadataCount = wire.Metadata.Entries
	}
	if err := t.Validate(); err != nil {
		return InputTrigger{}, err
	}
	return t, nil
}

func validateExplicit(e Explicit) error {
	if !Known(e.Classification) {
		return errors.New("input_trigger: unknown explicit classification")
	}
	switch e.Provenance {
	case ProvenanceExplicit, ProvenanceRetry:
	default:
		return errors.New("input_trigger: invalid explicit provenance")
	}
	return validateMetadata(e.Metadata)
}

func validateMetadata(metadata map[string]string) error {
	if len(metadata) > MaxMetadataEntries {
		return fmt.Errorf("input_trigger: metadata has %d entries, limit is %d", len(metadata), MaxMetadataEntries)
	}
	keys := make([]string, 0, len(metadata))
	for key := range metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for i, key := range keys {
		value := metadata[key]
		if key == "" || !utf8.ValidString(key) || len(key) > MaxMetadataKeyBytes {
			return fmt.Errorf("input_trigger: metadata key %d is empty, invalid UTF-8, or exceeds %d bytes", i+1, MaxMetadataKeyBytes)
		}
		if !utf8.ValidString(value) || len(value) > MaxMetadataValBytes {
			return fmt.Errorf("input_trigger: metadata value %d is invalid UTF-8 or exceeds %d bytes", i+1, MaxMetadataValBytes)
		}
	}
	return nil
}

func decodeStrict(data []byte, target any) error {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(target); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("multiple JSON values are not allowed")
		}
		return err
	}
	return nil
}
