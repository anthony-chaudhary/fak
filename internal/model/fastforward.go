package model

// Tool-schema fast-forward (jump-forward) drafting for the speculative-decode verify
// loop (#4103, child of the tool-call speculation epic #4102). This is the DUAL of the
// constraint.go logit mask:
//
//   - GuidedByteMask answers "which single next token is schema-legal here" — a MASK that
//     constrains ONE step.
//   - FastForwardSpan answers "how far ahead is the schema DETERMINISTIC" — the maximal run
//     of bytes the tool-call FSM forces with no model choice, which can be proposed as a
//     multi-token DRAFT that VerifyForward accepts in a single target pass.
//
// A tool call is JSON pinned by a fixed envelope: `{"name":"<one declared name>","arguments":
// <value>}`. Every byte of the skeleton — and the tool name up to the point the declared set
// still shares a unique prefix — is schema-determined; only the argument VALUES are free.
// Those structural bytes are the highest-acceptance draft source that exists for tool calls
// (ToolSpec arXiv:2604.13519 pairs exactly this schema FSM with a retrieval draft for the
// values; XGrammar arXiv:2411.15100 names the context-independent / context-dependent token
// partition that makes them free), and their draft cost is a table walk, not a model forward.
//
// LOSSLESS BY CONSTRUCTION. The draft is only a PROPOSAL: VerifyForward (verify.go) re-runs
// the target over the draft ids and the caller accepts a token only when the target agrees;
// a mis-proposed token is rejected and re-decoded normally. Correctness therefore never
// depends on the draft being right — only speed does. On the deterministic span acceptance
// is 100% by construction, because a schema-masked target has exactly one legal next byte at
// every position of the span (the same byte FastForwardSpan emitted).
//
// GATED, OFF BY DEFAULT. Like GuidedByteMask this is NOT wired into any default decode.
// FastForwardDrafter.Draft returns no draft unless FAK_NATIVE_GUIDED_DECODE=1
// (GuidedDecodeEnabled), so the proposer is dormant until an operator opts in — the gen/next
// "gated / dogfood" bar. FastForwardSpan itself is an ungated pure function (the dual of
// guideddecode.AllowedNextBytes), mirroring how AllowedNextBytes is ungated while the mask
// that applies it (maskActive) is flag-gated.

import (
	"bytes"

	"github.com/anthony-chaudhary/fak/internal/guideddecode"
)

// FastForwardSpan returns the maximal run of bytes the tool-call schema FSM forces from
// prefix with NO model choice — the deterministic (jump-forward) span, the dual of the
// per-step mask. It walks the byte-FSM (guideddecode.AllowedNextBytes) and appends a byte
// for as long as EXACTLY ONE byte is admissible, stopping at the first state that
//
//   - admits two or more bytes (an enum/value BRANCH the model must resolve),
//   - is UNCONSTRAINED (nil — the arguments value region, where any byte is legal), or
//   - is a DEAD END (empty non-nil — unreachable under correct masking).
//
// Every returned byte is, at its position, the sole schema-legal continuation, so a
// schema-masked target can only reproduce the span: acceptance on it is 100% by
// construction. An already-branching or unconstrained prefix yields an empty span (draft
// nothing; decode normally). prefix is never mutated.
func FastForwardSpan(prefix []byte, schema guideddecode.ToolSchema) []byte {
	var span []byte
	cur := append([]byte(nil), prefix...)
	for {
		allowed := guideddecode.AllowedNextBytes(cur, schema)
		if len(allowed) != 1 {
			return span // branch (>=2), unconstrained (nil), or dead end (empty): stop
		}
		var b byte
		for k := range allowed {
			b = k
		}
		span = append(span, b)
		cur = append(cur, b)
	}
}

// FastForwardDrafter lifts FastForwardSpan onto a tokenizer's vocabulary: it proposes the
// schema-fixed span as DRAFT TOKEN IDS for the VerifyForward accept loop. It mirrors
// GuidedByteMask's dependency inversion (internal/model imports neither internal/grammar nor
// internal/tokenizer): the id->bytes decode of the tokens emitted so far (TokenBytes) and the
// bytes->ids encode of the drafted span (Encode) are INJECTED by a higher layer that can see
// the tokenizer.
type FastForwardDrafter struct {
	// Schema is the declared tool-name set the envelope FSM constrains to — the SAME schema
	// a paired GuidedByteMask verifies against.
	Schema guideddecode.ToolSchema
	// TokenBytes maps a token id to the exact bytes it decodes to, or nil for an id that is
	// not a decodable token. Inject tokenizer.(*Tokenizer).TokenBytes. It rebuilds the
	// envelope byte prefix from the tokens emitted so far this turn.
	TokenBytes func(id int) []byte
	// Encode maps a byte string to the token ids a greedy tokenizer would emit for it. Inject
	// the tokenizer's encoder. It segments the deterministic byte span into draft tokens.
	Encode func([]byte) []int
}

// Draft proposes the schema-deterministic continuation from the tokens emitted so far this
// turn (history — the generated ids, not the prompt) as a chain of draft token ids for
// VerifyForward. It rebuilds the envelope byte prefix from history, takes the maximal
// deterministic byte span, and segments it into tokens, keeping only whole tokens whose bytes
// lie ENTIRELY within the span and match it exactly. A token whose bytes would cross the span
// boundary — into a model-chosen region — or whose decode does not match the span (a lossy
// proposer) is dropped, so every drafted token stays schema-forced and the 100%-acceptance
// guarantee holds.
//
// Returns nil when nothing is deterministic here (an enum/value branch or the value region),
// when the native guided-decode gate is off (FAK_NATIVE_GUIDED_DECODE != 1), or when the
// receiver / TokenBytes / Encode is nil — so the drafter is a dormant no-op by default.
func (d *FastForwardDrafter) Draft(history []int) []int {
	if d == nil || d.TokenBytes == nil || d.Encode == nil || !GuidedDecodeEnabled() {
		return nil
	}
	span := FastForwardSpan(d.decodePrefix(history), d.Schema)
	if len(span) == 0 {
		return nil
	}
	ids := d.Encode(span)
	kept := make([]int, 0, len(ids))
	used := 0
	for _, id := range ids {
		tb := d.TokenBytes(id)
		if len(tb) == 0 || used+len(tb) > len(span) || !bytes.Equal(tb, span[used:used+len(tb)]) {
			break // a boundary-crossing or non-matching token: stop before it (stay schema-forced)
		}
		kept = append(kept, id)
		used += len(tb)
	}
	if len(kept) == 0 {
		return nil
	}
	return kept
}

// decodePrefix reconstructs the envelope byte prefix emitted so far this turn by decoding
// each history token through TokenBytes and concatenating — the same reconstruction
// GuidedByteMask.decodePrefix performs. Under correct masking history holds only admitted
// tokens, so the reconstruction is exact.
func (d *FastForwardDrafter) decodePrefix(history []int) []byte {
	var prefix []byte
	for _, id := range history {
		prefix = append(prefix, d.TokenBytes(id)...)
	}
	return prefix
}
