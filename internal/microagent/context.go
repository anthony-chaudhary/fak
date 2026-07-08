package microagent

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Bounded linear-history context (epic #2000 M4, issue #2004).
//
// The mini-SWE-agent invariant is "completely linear history; every step appends
// to messages" — which is what makes a per-agent transcript predictable and
// debuggable. Context is that linear list with ONE addition the invariant needs to
// scale to 1000s of in-process agents: a HARD token ceiling. Unbounded history is
// the memory blow-up risk (a runaway agent grows its transcript without bound);
// with a cap, per-agent memory is O(cap), independent of run length.
//
// The ceiling is priced with the SAME tokenizer the served gateway / guard boot
// path uses — agent.EstimateAnthropicTokens, the house ~4-char/token estimate — so
// this Context enforces its cap in the same units the model request is charged in,
// never a second tokenizer that could silently drift from the one production bills
// against (TestContextTokenizerMatchesGuardBootPath pins that equality).
//
// # Drop policy at the cap — and the M25 hand-off
//
// At the cap, Append DROPS the oldest whole messages (FIFO) until the history fits.
// Whole-turn drop is the cheapest O(cap) policy and the debuggable default: what
// remains is always a suffix of the real linear history, never a rewritten one.
// Lossy content compaction (summarize-and-fold a dropped span into a synthetic
// note) is deliberately NOT done here — that is the M25 compaction hand-off. The
// guarantee this type makes is: Tokens() <= Cap() after every Append, PROVIDED the
// newest single message individually fits the cap. A lone message larger than the
// whole cap cannot be honored by dropping older turns; Append keeps it and Overflow
// reports true. That degenerate case is exactly what M25 content-level compaction
// (truncate/summarize a single oversized turn) owns — this leaf refuses to fake it.
//
// # Deterministic serialization (M12 hibernation building block)
//
// Encode/Decode are a byte-identical round trip: Encode produces canonical JSON
// (struct fields in a fixed order, no maps), so Decode-then-Encode reproduces the
// exact bytes. That is what lets a parked/hibernated agent (M12, issue #2012,
// hibernate.go) freeze its context to disk and resume it with no state loss — a
// Hibernable agent's Freeze delegates to Context.Encode and its Thaw to Decode, and
// the HibernationStore's no-state-loss check (re-Freeze must equal the read bytes)
// holds precisely because this round trip is deterministic.
//
// # Generation intent (gen/second-next, #2004)
//
// This is an architectural OPTION behind the microagent import boundary — nothing
// in the default fak serve / guard / dispatch path constructs a Context. Closing
// evidence for the generation frame:
//
//   - Promotion evidence: TestContextBoundedTokenCapAcrossLongRun witnesses the
//     hard-ceiling invariant across a long synthetic run (history tokens never
//     exceed the cap while the run length grows without bound, so residency is
//     O(cap)); TestContextEncodeDecodeRoundTripIdentical witnesses the deterministic
//     serialize->deserialize round trip. Promote once the #2001 RunArm stepping lands
//     a real Microagent whose per-turn state IS a Context, and a footprint
//     measurement (#2033) confirms unbounded per-agent history was a binding cost.
//   - Demotion / retirement criteria: retire the cap if the footprint benchmark
//     shows a real agent's natural transcript never approaches the ceiling in a
//     bounded run (the cap then guards nothing), or if M25 compaction supersedes
//     whole-turn drop entirely (the drop policy then becomes dead code behind the
//     compactor).
//   - Invalidating assumption: the cap assumes a turn's cost is well-estimated by
//     the ~4-char/token house heuristic and that dropping OLDEST turns preserves
//     enough context for the agent to keep making progress. If a real loop needs a
//     pinned prefix (a system/goal turn that must survive eviction) or the heuristic
//     diverges from provider-billed tokens under real tool-result payloads, this
//     drop policy under-serves and the M25 compactor (pinned spans + summarization)
//     must land before a Context carries a production agent.

// DefaultContextCap is the token ceiling NewContext(0) selects. It is a usable
// starting default, not a model-specific window; callers size the cap to their
// model's context window (and to how many agents share the host's RAM).
const DefaultContextCap = 8192

// Structured refusals / edges for the context seam.
var (
	// ErrContextVersion is returned by Decode for a serialized context whose
	// version tag this build does not understand (a forward-incompatible blob).
	ErrContextVersion = errors.New("microagent: unknown context serialization version")
)

// Msg is one linear-history entry: a role and its text content. This is the whole
// mini-SWE-agent step record — every Step appends exactly one Msg; history is a
// suffix of appends, never reordered or rewritten in place. Tool calls and results
// are carried as content by the caller so the token accounting stays a single
// char-walk (matching EstimateAnthropicTokens' per-message cost).
type Msg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Context is a linear message history with a hard token ceiling. The zero value is
// NOT usable (its cap is 0, which every Append would exceed); construct one with
// NewContext. It is NOT safe for concurrent use — one Context belongs to one agent
// loop, mutated only from that agent's Step.
type Context struct {
	cap  int
	msgs []Msg
}

// NewContext builds an empty context whose history is capped at tokenCap tokens.
// A non-positive tokenCap selects DefaultContextCap.
func NewContext(tokenCap int) *Context {
	if tokenCap <= 0 {
		tokenCap = DefaultContextCap
	}
	return &Context{cap: tokenCap}
}

// Cap reports the hard token ceiling.
func (c *Context) Cap() int { return c.cap }

// Len reports how many messages are currently in the history.
func (c *Context) Len() int { return len(c.msgs) }

// Tokens reports the history's estimated token cost, priced with the guard boot
// path tokenizer (agent.EstimateAnthropicTokens) so it is charged in the same units
// the served gateway bills. After a well-sized Append this is <= Cap.
func (c *Context) Tokens() int { return estContextTokens(c.msgs) }

// Overflow reports the degenerate case the drop policy cannot honor: a single
// message that alone exceeds the whole cap (so dropping older turns can never bring
// the history under the ceiling). It is the boundary the M25 compactor owns; a
// well-sized run never trips it.
func (c *Context) Overflow() bool {
	return len(c.msgs) == 1 && estContextTokens(c.msgs) > c.cap
}

// Messages returns a defensive copy of the linear history (callers may not mutate
// the Context's backing array).
func (c *Context) Messages() []Msg {
	out := make([]Msg, len(c.msgs))
	copy(out, c.msgs)
	return out
}

// Append adds one message to the linear history, then enforces the hard token
// ceiling by the drop policy: it evicts the OLDEST messages (FIFO) until the
// history fits the cap, and returns how many it evicted. Evicted messages are
// released for GC (the retained slice is compacted), so per-agent memory stays
// O(cap) no matter how long the run is.
//
// See the package note on the drop-vs-compact boundary: when the newest single
// message alone exceeds the cap, Append keeps it (Overflow then reports true) — the
// ceiling can only be honored there by M25 content-level compaction, not by drop.
func (c *Context) Append(role, content string) (evicted int) {
	c.msgs = append(c.msgs, Msg{Role: role, Content: content})
	drop := 0
	for len(c.msgs)-drop > 1 && estContextTokens(c.msgs[drop:]) > c.cap {
		drop++
	}
	if drop > 0 {
		kept := make([]Msg, len(c.msgs)-drop)
		copy(kept, c.msgs[drop:])
		c.msgs = kept
	}
	return drop
}

// estContextTokens prices a linear history with the guard boot path tokenizer. It
// maps each Msg onto the canonical transcript vocabulary and defers to
// agent.EstimateAnthropicTokens, so the number can never drift from the estimate
// the served gateway / footprint verbs report (the reuse the issue calls for).
func estContextTokens(msgs []Msg) int {
	return agent.EstimateAnthropicTokens(&agent.AnthropicMessagesRequest{
		Messages: toAgentMessages(msgs),
	})
}

func toAgentMessages(msgs []Msg) []agent.Message {
	out := make([]agent.Message, len(msgs))
	for i, m := range msgs {
		out[i] = agent.Message{Role: m.Role, Content: m.Content}
	}
	return out
}

// contextWireVersion tags the serialized form so a future breaking change can be
// refused (ErrContextVersion) rather than silently mis-decoded.
const contextWireVersion = 1

// contextWire is the on-the-wire shape of a Context. It is a plain struct (fixed
// field order, no maps), so encoding/json emits canonical, deterministic bytes and
// Encode->Decode->Encode is byte-identical.
type contextWire struct {
	Version int   `json:"v"`
	Cap     int   `json:"cap"`
	Msgs    []Msg `json:"msgs"`
}

// Encode serializes the context to canonical JSON. Two Encode calls on an unchanged
// context return equal bytes, and Decode(Encode(c)) reproduces c — the deterministic
// round trip M12 hibernation stands on. Msgs is normalized to a non-nil slice so an
// empty context and a drained-to-empty one encode identically.
func (c *Context) Encode() ([]byte, error) {
	msgs := c.msgs
	if msgs == nil {
		msgs = []Msg{}
	}
	return json.Marshal(contextWire{Version: contextWireVersion, Cap: c.cap, Msgs: msgs})
}

// Decode restores a context from Encode's output, replacing the receiver's state.
// It refuses ErrContextVersion for a blob tagged with an unknown version. After a
// successful Decode, Encode returns the exact bytes Decode read (byte-identical
// round trip).
func (c *Context) Decode(b []byte) error {
	var w contextWire
	if err := json.Unmarshal(b, &w); err != nil {
		return err
	}
	if w.Version != contextWireVersion {
		return fmt.Errorf("%w: %d", ErrContextVersion, w.Version)
	}
	c.cap = w.Cap
	if w.Msgs == nil {
		w.Msgs = []Msg{}
	}
	c.msgs = w.Msgs
	return nil
}
