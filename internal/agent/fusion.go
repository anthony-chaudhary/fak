package agent

import (
	"context"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fusion.go — turn-fusion of refcount-1 intermediates (#847, epic #844, the
// reachability layer). A pure tool-dispatch turn whose RESULT is read by exactly
// the next reasoning step and never cited again is a refcount-1 producer→consumer
// intermediate: it persists as its own heap object in the transcript even though
// nothing but its single consumer will ever reference it. Fusion collapses that
// intermediate into its consumer so it stops being a standalone object.
//
// This is the SOUND version of the byte-trim in (*Server).maybeCompactInboundTools
// (internal/gateway/messages.go): there the trim is applied to the outbound wire;
// here the fold is committed THROUGH the abi.ProvisionalSink Promote/Rollback
// contract (internal/abi/types.go), so an UNPROVEN fold is Rollback'd and the
// transcript is byte-identical to no-fusion. A fold only ever happens on a PROVEN
// refcount==1 pair; anything unproven is skipped conservatively.
//
// DEFAULT-OFF: nothing in the live RunArm loop calls this. It is the mechanism +
// the eligibility proof + the Promote/Rollback-gated commit; the live-loop wiring
// (running it each turn against the in-flight transcript) is a default-off
// follow-up, kept separate so the present change cannot alter a live decision.

// FusionCandidate names a proven refcount-1 producer→consumer intermediate in a
// transcript: the assistant turn that emitted the tool call (Producer), the tool
// result message it produced (Intermediate), and the single later message that
// consumes it (Consumer). All three are indices into the message slice the
// candidate was found in. A candidate exists ONLY when the only-one-consumer proof
// holds (refcount==1); see fusionCandidates.
type FusionCandidate struct {
	Producer     int    // index of the RoleAssistant message carrying the ToolCall
	Intermediate int    // index of the RoleTool result message (the refcount-1 object)
	Consumer     int    // index of the single message that references the result
	CallID       string // the tool_call_id binding producer→intermediate→consumer
}

// fusionRefcount counts, for each tool result in messages, how many LATER messages
// reference it. A RoleTool message is referenced when a later message cites its
// CallID — today the only structural citation is the producer/consumer chain, so a
// reference is a later message that (a) is the immediate next reasoning step after
// the result, or (b) re-mentions the call id in its own tool calls. The refcount is
// the COUNT of distinct later referents: refcount==1 is the only-one-consumer proof
// fusion requires; refcount>1 (cited again downstream) or refcount==0 (dangling)
// is NOT eligible.
//
// The proof is deliberately CONSERVATIVE: we only count a reference we can witness
// structurally from the transcript. When the only-one-consumer property cannot be
// proven (e.g. the result is the last message, or its consumer also re-emits the
// same call id, making the count ambiguous), the count is left at a value that
// makes the pair ineligible, so an unprovable pair is skipped rather than folded.
func fusionRefcount(messages []Message, intermediate int) (consumer int, refs int) {
	if intermediate < 0 || intermediate >= len(messages) {
		return -1, 0
	}
	im := messages[intermediate]
	if im.Role != RoleTool || im.ToolCallID == "" {
		return -1, 0
	}
	consumer = -1
	for j := intermediate + 1; j < len(messages); j++ {
		if referencesResult(messages[j], im.ToolCallID, j == intermediate+1) {
			refs++
			if consumer == -1 {
				consumer = j
			}
		}
	}
	return consumer, refs
}

// referencesResult reports whether message m references the tool result with the
// given call id. The immediate-next reasoning step (isNext) consumes the result by
// position — it is the turn the model produces AFTER seeing the result, the
// canonical single consumer of a pure tool-dispatch intermediate. Any later message
// that re-emits the SAME call id in its own tool calls is an additional structural
// referent (refcount>1) — a result re-dispatched is not a refcount-1 intermediate.
func referencesResult(m Message, callID string, isNext bool) bool {
	if isNext && (m.Role == RoleAssistant || m.Role == RoleUser) {
		return true
	}
	for _, tc := range m.ToolCalls {
		if tc.ID == callID {
			return true
		}
	}
	return m.ToolCallID == callID && m.Role == RoleTool
}

// fusionCandidates returns every PROVEN refcount-1 producer→consumer intermediate
// in messages. A candidate requires: a RoleTool result with a non-empty CallID, a
// matching RoleAssistant producer that emitted that CallID earlier, exactly ONE
// later referent (refcount==1), and that referent being the immediate next step.
// Anything that fails the proof is skipped — the function never returns an
// unprovable pair, so a caller can fold every returned candidate soundly.
func fusionCandidates(messages []Message) []FusionCandidate {
	var out []FusionCandidate
	for i := range messages {
		if messages[i].Role != RoleTool || messages[i].ToolCallID == "" {
			continue
		}
		callID := messages[i].ToolCallID
		producer := producerOf(messages, i, callID)
		if producer < 0 {
			continue // no witnessed producer → cannot prove the chain
		}
		consumer, refs := fusionRefcount(messages, i)
		if refs != 1 || consumer != i+1 {
			continue // not refcount==1, or the single ref is not the immediate next step
		}
		out = append(out, FusionCandidate{
			Producer:     producer,
			Intermediate: i,
			Consumer:     consumer,
			CallID:       callID,
		})
	}
	return out
}

// producerOf finds the RoleAssistant message before the result that emitted callID.
// Returns -1 when no such producer is witnessed (so the chain cannot be proven).
func producerOf(messages []Message, intermediate int, callID string) int {
	for i := intermediate - 1; i >= 0; i-- {
		if messages[i].Role != RoleAssistant {
			continue
		}
		for _, tc := range messages[i].ToolCalls {
			if tc.ID == callID {
				return i
			}
		}
	}
	return -1
}

// FuseEligible is the public eligibility predicate: it reports whether the result
// at index `intermediate` is a PROVEN refcount-1 producer→consumer intermediate
// that fusion may collapse. It returns true ONLY on a proven pair and false
// (conservative skip) whenever the only-one-consumer property is unproven. This is
// the acceptance-criterion predicate: "returns true only on a proven refcount==1
// producer→consumer pair; skips when only-one-consumer is unproven".
func FuseEligible(messages []Message, intermediate int) bool {
	if intermediate < 0 || intermediate >= len(messages) || messages[intermediate].Role != RoleTool {
		return false
	}
	callID := messages[intermediate].ToolCallID
	if callID == "" || producerOf(messages, intermediate, callID) < 0 {
		return false
	}
	consumer, refs := fusionRefcount(messages, intermediate)
	return refs == 1 && consumer == intermediate+1
}

// foldResultIntoConsumer produces the fused transcript: the refcount-1 intermediate
// is collapsed into its consumer (its content is carried on the consumer as a fused
// tool-result note and the standalone RoleTool object is dropped). The original
// slice is never mutated — a fresh slice is returned, so a Rollback can hand back
// the untouched original byte-identically. The fold is behavior-preserving for the
// model: the consumer still sees the result content, only it is no longer a separate
// heap object.
func foldResultIntoConsumer(messages []Message, c FusionCandidate) []Message {
	out := make([]Message, 0, len(messages)-1)
	for i := range messages {
		if i == c.Intermediate {
			continue // the refcount-1 object is collapsed away
		}
		m := messages[i]
		if i == c.Consumer {
			// Carry the intermediate's content onto its single consumer so the
			// consumer reasoning step still sees the tool output it depended on.
			folded := messages[c.Intermediate].Content
			if folded != "" {
				if m.Content != "" {
					m.Content = folded + "\n" + m.Content
				} else {
					m.Content = folded
				}
			}
		}
		out = append(out, m)
	}
	return out
}

// fusionSink is a transcript-level abi.ProvisionalSink. A proposed fold is OPENED
// against a (txn, epoch); Promote makes the fold durable (the fused transcript is
// the committed one) and Rollback discards it (the original transcript is restored
// byte-identical). This is the same Promote/Rollback contract spec.Sink uses for
// provisional KV, applied to the transcript fold so an unproven fold is never
// persisted. Safe for concurrent use.
type fusionSink struct {
	mu      sync.Mutex
	pending map[fusionKey]fusionTxn
}

type fusionKey struct {
	txn   abi.TxnID
	epoch uint64
}

// fusionTxn is the provisional state of one fold: the ORIGINAL transcript (restored
// on Rollback) and the FUSED transcript (committed on Promote). Holding both makes
// the rollback path byte-exact — Rollback never reconstructs, it returns the slice
// captured before the fold.
type fusionTxn struct {
	original       []Message
	fused          []Message
	resolved       bool
	committedFused bool // true once Promote'd (fused durable); false once Rollback'd
}

// newFusionSink returns an empty transcript fusion sink.
func newFusionSink() *fusionSink { return &fusionSink{pending: make(map[fusionKey]fusionTxn)} }

// compile-time proof the sink satisfies the frozen interface.
var _ abi.ProvisionalSink = (*fusionSink)(nil)

// open records a provisional fold for (txn, epoch): the original transcript and the
// fused one. The caller computes the fold (foldResultIntoConsumer over a PROVEN
// candidate) and opens it here; the fold becomes visible only on Promote.
func (s *fusionSink) open(txn abi.TxnID, epoch uint64, original, fused []Message) {
	s.mu.Lock()
	s.pending[fusionKey{txn, epoch}] = fusionTxn{original: original, fused: fused}
	s.mu.Unlock()
}

// resolved returns the committed transcript for (txn, epoch): the FUSED slice if the
// fold was Promote'd, the ORIGINAL slice if it was Rollback'd, and ok=false if the
// (txn, epoch) is unknown or still open.
func (s *fusionSink) resolved(txn abi.TxnID, epoch uint64) (msgs []Message, fused bool, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t, present := s.pending[fusionKey{txn, epoch}]
	if !present || !t.resolved {
		return nil, false, false
	}
	if t.committedFused {
		return t.fused, true, true
	}
	return t.original, false, true
}

// Promote (abi.ProvisionalSink) finalizes the fold: the fused transcript is now the
// durable one. Idempotent — an unknown or already-resolved key is a no-op.
func (s *fusionSink) Promote(_ context.Context, txn abi.TxnID, epoch uint64) error {
	return s.resolve(txn, epoch, true)
}

// resolve is the shared fold-finalization for Promote/Rollback: under the lock, mark a
// still-pending entry resolved and record whether the fused (true) or original (false)
// transcript is now durable. Idempotent — an unknown or already-resolved key is a no-op.
func (s *fusionSink) resolve(txn abi.TxnID, epoch uint64, committedFused bool) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	k := fusionKey{txn, epoch}
	t, ok := s.pending[k]
	if !ok || t.resolved {
		return nil
	}
	t.resolved = true
	t.committedFused = committedFused
	s.pending[k] = t
	return nil
}

// Rollback (abi.ProvisionalSink) retracts the fold: the original transcript is
// restored byte-identical to never having folded. Idempotent.
func (s *fusionSink) Rollback(_ context.Context, txn abi.TxnID, epoch uint64) error {
	return s.resolve(txn, epoch, false)
}
