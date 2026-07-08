package modelroute

// The WITNESSED MODEL-SWITCH path (#2934, epic #2908 — Hermes-parity).
//
// Hermes' `hermes model` switches provider/model with an asserted "no code
// changes, no lock-in" claim. The claim is ASSERTED; nothing proves the switch
// preserved the conversation's cached prefix or its semantic (tool-call) behavior
// — a provider switch can silently bust the KV-prefix cache or change the
// tool-call DIALECT, and the user only learns from the bill or a broken turn.
//
// fak turns that unproven claim into a WITNESS. fak owns the conversation prefix
// (the token sequence, not the provider's opaque KV tensors) and adjudicates
// tool-call emission across dialects (#470 text-embedded tool-call parsing). So on
// a model/provider switch, this file records — deterministically, with no I/O —
// exactly what carried over:
//
//   - PREFIX-REUSE STATUS: preserved (the owned prefix is reused as-is because the
//     target endpoint keys the prefix the same way) or intentionally RESET (the
//     prefix is re-encoded because the tokenizer / prompt format changed). A reset
//     is a CORRECT outcome on a cross-tokenizer switch, NOT a bug — the witness
//     records WHICH case occurred and WHY, rather than assuming reuse is expected.
//   - TOOL-CALL DIALECT RENORMALIZATION: whether the emission dialect had to change
//     (e.g. Anthropic's structured tool_calls channel -> a local model's Hermes
//     <tool_call> text dialect). The dialect PARSING itself is #470's; this file
//     only names the before/after dialect and flags the renorm.
//   - COST DELTA: the rough $/Mtok change from before to after, reusing this
//     package's cost lens (see cost.go, Price). Positive == the switch got pricier.
//
// PURITY (lane rule): stdlib-only, pure and deterministic — the same (from, to)
// pair always yields the same witness, so it is content-addressable (Digest) and
// dos-verifiable exactly like a routing AuditRecord. The LIVE wiring that CALLS
// this on a real mid-session switch (the gateway/agent loop swapping the active
// endpoint) is additive on top, tracked separately — this is the witnessed spine.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

// ToolDialect names the tool-call EMISSION dialect an endpoint speaks. The
// text-form values mirror the #470 registry (internal/agent/toolcall_fallback.go's
// toolCallDialects) WITHOUT importing it — this leaf stays stdlib-only and only
// NAMES the dialect for the renorm witness; #470 owns the actual parsing. It is a
// CLOSED, additive set: a new dialect is a new constant here, never manifest free
// text.
type ToolDialect string

const (
	// DialectStructured is the provider's native structured tool_calls channel
	// (Anthropic, OpenAI): the call arrives as a first-class field, not text.
	DialectStructured ToolDialect = "structured"
	// DialectHermes is the Hermes/Qwen <tool_call>{...}</tool_call> TEXT dialect.
	DialectHermes ToolDialect = "hermes"
	// DialectFunctionCallTag is the <function_call>{...}</function_call> TEXT dialect.
	DialectFunctionCallTag ToolDialect = "function_call_tag"
	// DialectLlamaPython is Llama-3.1's <|python_tag|>{...} TEXT dialect.
	DialectLlamaPython ToolDialect = "llama_python_tag"
	// DialectMistral is Mistral/Mixtral's [TOOL_CALLS][...] TEXT dialect.
	DialectMistral ToolDialect = "mistral_tool_calls"
	// DialectFenced is a ```json ... ``` fenced tool-call TEXT dialect.
	DialectFenced ToolDialect = "fenced_json"
	// DialectBareJSON is a bare JSON tool-call object as the whole message.
	DialectBareJSON ToolDialect = "bare_json"
)

// knownDialect reports whether d is one of the closed ToolDialect set.
func knownDialect(d ToolDialect) bool {
	switch d {
	case DialectStructured, DialectHermes, DialectFunctionCallTag,
		DialectLlamaPython, DialectMistral, DialectFenced, DialectBareJSON:
		return true
	}
	return false
}

// Endpoint is the identity of one active serving target for a session — the facts
// a switch must reason about. It is a thin, pure descriptor: a subset of
// ResolvedModel (Model, Provider) plus the tool-call Dialect and the PrefixKey the
// owned conversation prefix is addressed by, plus a rough Price for the cost lens.
type Endpoint struct {
	// Model is the concrete engine/model id (e.g. "claude-sonnet", "qwen2.5:1.5b").
	Model string `json:"model"`
	// Provider is the serving provider ("anthropic", "openai", "ollama"); "" for a
	// local / in-kernel endpoint.
	Provider string `json:"provider,omitempty"`
	// Dialect is the tool-call emission dialect this endpoint speaks.
	Dialect ToolDialect `json:"dialect"`
	// PrefixKey is the identity fak keys its OWNED conversation prefix on — the
	// tokenizer + prompt-format pair, NOT the provider's opaque per-model KV
	// tensors. Two endpoints reuse the prefix across a switch iff their PrefixKey is
	// equal and non-empty; a differing (or empty) key means the prefix is
	// re-encoded. It is deliberately NOT the model id: a same-family retier
	// (sonnet->haiku) shares a key and preserves; a cross-tokenizer switch does not.
	PrefixKey string `json:"prefix_key"`
	// Price is the rough $/Mtok cost of this endpoint, for the cost-delta lens.
	Price Price `json:"price"`
}

// PrefixStatus is the closed outcome of the prefix-reuse question on a switch.
type PrefixStatus string

const (
	// PrefixPreserved means the owned conversation prefix carried over unchanged —
	// the target endpoint keys the prefix the same way, so it is reused as-is.
	PrefixPreserved PrefixStatus = "preserved"
	// PrefixReset means the prefix was re-encoded for the target endpoint. This is a
	// CORRECT outcome on a cross-tokenizer / cross-format switch, not a cache bust.
	PrefixReset PrefixStatus = "reset"
)

// SwitchWitness is the emitted, content-addressable record of one model/provider
// switch — the proof that turns Hermes' asserted "no lock-in" into a checked fact.
// It carries the three things a bare switch claim never proves: the prefix-reuse
// status (with the REASON, so a reset is legible as intentional), the tool-call
// dialect renormalization, and the measured cost delta. Pure over (from, to).
type SwitchWitness struct {
	From Endpoint `json:"from"`
	To   Endpoint `json:"to"`

	// Prefix is preserved | reset; PrefixReason explains WHY (the load-bearing
	// half: it distinguishes an intentional reset from a silent cache bust).
	Prefix       PrefixStatus `json:"prefix"`
	PrefixReason string       `json:"prefix_reason"`

	// DialectFrom / DialectTo are the before/after emission dialects; Renormalized
	// is true iff they differ (the tool-call output had to be re-normalized).
	DialectFrom  ToolDialect `json:"dialect_from"`
	DialectTo    ToolDialect `json:"dialect_to"`
	Renormalized bool        `json:"dialect_renormalized"`

	// CostDeltaIn / CostDeltaOut are the $/Mtok change (to - from) per direction;
	// positive == the switch got pricier. Rough, from the cost lens — never a bill.
	CostDeltaIn  float64 `json:"cost_delta_in"`
	CostDeltaOut float64 `json:"cost_delta_out"`
}

// WitnessSwitch records what a switch from one endpoint to another preserved. Pure
// and deterministic: the same (from, to) always yields the same witness, so it is
// content-addressable and replayable. It NEVER fails — a witness of a
// misconfigured switch (e.g. an unknown dialect) is still a truthful record of
// that switch; validation of the endpoints is the caller's concern (see
// Endpoint.Validate) and must not silence the witness.
func WitnessSwitch(from, to Endpoint) SwitchWitness {
	w := SwitchWitness{
		From:         from,
		To:           to,
		DialectFrom:  from.Dialect,
		DialectTo:    to.Dialect,
		Renormalized: from.Dialect != to.Dialect,
		CostDeltaIn:  to.Price.In - from.Price.In,
		CostDeltaOut: to.Price.Out - from.Price.Out,
	}
	w.Prefix, w.PrefixReason = prefixOutcome(from, to)
	return w
}

// prefixOutcome decides whether the owned prefix survives the switch and explains
// why. Reuse requires a shared, non-empty PrefixKey; every other case is a reset,
// with a reason that names it as intentional (differing keys) or as an
// unaddressable prefix (an endpoint declared no key), never as a bug.
func prefixOutcome(from, to Endpoint) (PrefixStatus, string) {
	switch {
	case from.PrefixKey != "" && from.PrefixKey == to.PrefixKey:
		return PrefixPreserved, fmt.Sprintf("same prefix key %q — owned conversation prefix reused across the switch", to.PrefixKey)
	case from.PrefixKey == "" || to.PrefixKey == "":
		return PrefixReset, "an endpoint declares no prefix key — prefix re-encoded from scratch (cannot prove reuse)"
	default:
		return PrefixReset, fmt.Sprintf("prefix key changed (%q -> %q) — prefix re-encoded for the new tokenizer/format (intentional reset, not a cache bust)", from.PrefixKey, to.PrefixKey)
	}
}

// PrefixReused reports whether the owned prefix carried over the switch.
func (w SwitchWitness) PrefixReused() bool { return w.Prefix == PrefixPreserved }

// Validate reports the first way an Endpoint is unfit to witness a switch: an empty
// model or an unknown tool dialect. WitnessSwitch itself never fails (a witness of a
// bad switch is still true); a caller that wants to REFUSE a misconfigured switch
// calls this on both endpoints first.
func (e Endpoint) Validate() error {
	if e.Model == "" {
		return fmt.Errorf("modelroute: switch endpoint has an empty model")
	}
	if !knownDialect(e.Dialect) {
		return fmt.Errorf("modelroute: switch endpoint %q has unknown tool dialect %q", e.Model, e.Dialect)
	}
	return nil
}

// Digest content-addresses a SwitchWitness: a stable "sha256:"+hex hash over the
// whole witness. Same switch -> same digest, within and across processes (the
// pre-image is a canonical JSON encoding of fixed fields, no top-level map), so the
// witness is dos-verifiable exactly like a routing AuditRecord.
func (w SwitchWitness) Digest() string {
	b, err := json.Marshal(w)
	if err != nil {
		// Unreachable: every field is a stdlib-marshalable scalar/struct.
		return ""
	}
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// JSON renders the witness as canonical indented JSON, newline-terminated so it
// appends cleanly to a journal/ledger line — the emitted, auditable form.
func (w SwitchWitness) JSON() []byte {
	b, _ := json.MarshalIndent(w, "", "  ")
	return append(b, '\n')
}

// Headline renders the one-line human summary of a switch: the model move, the
// prefix outcome, the dialect renorm, and the rough output-token cost delta. ASCII
// only (matches Savings.Headline). Never a bill — the cost figure is the rough lens.
func (w SwitchWitness) Headline() string {
	prefix := "prefix RESET"
	if w.PrefixReused() {
		prefix = "prefix preserved"
	}
	dialect := fmt.Sprintf("dialect %s (unchanged)", w.DialectTo)
	if w.Renormalized {
		dialect = fmt.Sprintf("dialect renormalized %s->%s", w.DialectFrom, w.DialectTo)
	}
	var cost string
	switch {
	case w.CostDeltaOut > 0:
		cost = fmt.Sprintf("+$%s/Mtok-out (pricier)", money(w.CostDeltaOut))
	case w.CostDeltaOut < 0:
		cost = fmt.Sprintf("-$%s/Mtok-out (cheaper)", money(-w.CostDeltaOut))
	default:
		cost = "same $/Mtok-out"
	}
	return fmt.Sprintf("switch %s -> %s: %s; %s; %s (rough lens, not a bill)",
		w.From.Model, w.To.Model, prefix, dialect, cost)
}
