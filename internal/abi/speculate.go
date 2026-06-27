// speculate.go — the LIVE driver for SEAM 4 (the provisional speculation
// lifecycle reserved in types.go). It replaces the v0.1 no-op: where v0.1's MMU
// registers a sink whose Promote is a bare append and whose Rollback drops the
// scratch, this file gives the seam a REAL predicted-next-path engine — predict,
// run on slack, then commit on a match or squash on a miss — driving the frozen
// SpeculationContext / Outcome / ProvisionalSink contract end to end.
//
// It adds NOTHING to the frozen wire (types.go is untouched; the goldens do not
// move). It is a new driver file in the spine package, the same way reasons.go
// and events.go realize the closed vocabularies the frozen types.go declares.
//
// DEFAULT-OFF (the non-negotiable safety posture). A zero-value Speculator has
// Enabled=false and predicts nothing, so a kernel that never opts in behaves
// EXACTLY as v0.1: every call is ordinary (Speculative=false, Epoch=0) and the
// lifecycle never fires. Speculation is a deliberate, per-Speculator opt-in.
//
// THE LAW — DEFAULT-DENY ON EFFECTS (epic #809). A call is speculated ONLY if it
// is provably effect-free: read-only AND idempotent AND not write-shaped. Every
// mutating call is NEVER speculated — it is left for the model's authoritative
// emission. This is the CPU store-buffer discipline at the agent layer: our
// "stores" hit payments / emails / deletes that have no rollback, so the gate
// fails CLOSED on an unstamped or ambiguous call. The predicate is the SAME
// read-only/idempotent/not-destructive decision internal/vdso.Speculatable uses
// to admit a pure result to the cache; it is reimplemented here over ToolCall.Meta
// (vdso imports abi, so abi cannot import vdso) and the two must not drift — the
// shared meta keys "readOnlyHint" / "idempotentHint" / "destructive" and the
// write-shape name heuristic are the single contract.
//
// PASTE-style prediction (arXiv 2603.18897): a pattern is a tuple of
//   context signature -> predicted tool type -> a symbolic function deriving the
//   args from prior tool outputs -> an empirical success probability.
// Args a model generates freely from scratch resist speculation; only args
// DERIVABLE from what already happened are predictable, so DeriveArgs is the heart
// of the contract — a pattern that cannot derive its args declines to predict.

package abi

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
)

// SpecPattern is one PASTE-style prediction rule. It fires when Signature matches
// the live context signature, predicts a call to PredictTool, and derives that
// call's Args symbolically from the prior tool outputs via DeriveArgs (nil/ok=false
// => the args are not derivable, so the pattern declines). SuccessProb is the
// empirical hit rate that gates whether predicting is worth the slack.
type SpecPattern struct {
	// Signature is the context signature this pattern keys on (e.g. the just-
	// completed tool name, or a small folded fingerprint of recent calls). A pattern
	// fires only on an exact signature match — the predictor never guesses across
	// signatures.
	Signature string

	// PredictTool is the tool type the pattern predicts will be called next.
	PredictTool string

	// DeriveArgs is the symbolic arg-deriving function: it builds the predicted
	// call's Args from the prior tool outputs. It returns ok=false when the args
	// cannot be derived from what is known (the resist-speculation case for
	// freely-generated args), and the pattern then declines rather than guessing.
	DeriveArgs func(prior []*Result) (Ref, bool)

	// Meta is stamped onto the predicted call (the read-only / idempotent effect
	// hints the default-deny-on-effects gate reads). A pattern that predicts an
	// effect-free read sets readOnlyHint+idempotentHint here.
	Meta map[string]string

	// SuccessProb is the empirical probability the prediction is correct, in [0,1].
	// A prediction below the Speculator's MinProb is not worth the slack and is not
	// issued.
	SuccessProb float64
}

// Speculator is the live predicted-next-path engine for SEAM 4. It holds the
// PASTE-style pattern table and issues a speculative ToolCall stamped with a fresh
// epoch when a pattern matches AND the predicted call clears the default-deny-on-
// effects law. A zero-value Speculator (Enabled=false) predicts nothing, so the
// seam stays the v0.1 no-op until a caller opts in.
type Speculator struct {
	// Enabled is the DEFAULT-OFF master switch. False (the zero value) => Predict
	// always returns nil and no epoch is ever issued, reproducing v0.1 exactly.
	Enabled bool

	// MinProb is the probability floor a pattern must clear to be issued. Zero
	// admits any matching pattern; set it to require empirical confidence.
	MinProb float64

	mu       sync.Mutex
	patterns map[string][]SpecPattern // keyed by Signature
	epoch    uint64                   // monotonically issued speculation epoch ids
}

// NewSpeculator builds an ENABLED speculator with the given probability floor. The
// zero-value Speculator is the disabled (v0.1 no-op) form; this constructor is the
// opt-in. Patterns are added with Learn.
func NewSpeculator(minProb float64) *Speculator {
	return &Speculator{Enabled: true, MinProb: minProb, patterns: map[string][]SpecPattern{}}
}

// Learn registers a prediction pattern. Patterns are indexed by signature, so
// prediction is an O(1) signature lookup over a small per-signature list.
func (s *Speculator) Learn(p SpecPattern) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.patterns == nil {
		s.patterns = map[string][]SpecPattern{}
	}
	s.patterns[p.Signature] = append(s.patterns[p.Signature], p)
}

// nextEpoch issues a fresh, non-zero speculation epoch id (0 is reserved for
// "not speculative" by the frozen contract).
func (s *Speculator) nextEpoch() uint64 {
	return atomic.AddUint64(&s.epoch, 1)
}

// Predict returns the speculative ToolCall to run ahead of the model for the given
// context signature and prior tool outputs, or nil when nothing should be
// speculated. It returns nil (no speculation) when ANY of these hold:
//   - the speculator is disabled (the default-off floor),
//   - no pattern matches the signature,
//   - the best matching pattern is below MinProb,
//   - the pattern's symbolic DeriveArgs cannot derive the args (resist-speculation),
//   - the predicted call is not provably effect-free (default-deny on effects).
//
// A returned call is stamped Speculative=true with a fresh non-zero Epoch and
// branches from parentEpoch, so the frozen SpeculationContext rides it and every
// effect it produces is provisional until Promote/Rollback.
func (s *Speculator) Predict(sig string, prior []*Result, parentEpoch uint64) *ToolCall {
	if s == nil || !s.Enabled {
		return nil // default-deny: a disabled (or nil) speculator never predicts
	}
	s.mu.Lock()
	cands := s.patterns[sig]
	s.mu.Unlock()

	// Pick the highest-probability pattern that clears the floor AND can derive its
	// args. A pattern that cannot derive its args is skipped, never guessed.
	var best *SpecPattern
	var bestArgs Ref
	for i := range cands {
		p := &cands[i]
		if p.SuccessProb < s.MinProb {
			continue
		}
		if best != nil && p.SuccessProb <= best.SuccessProb {
			continue
		}
		if p.DeriveArgs == nil {
			continue
		}
		args, ok := p.DeriveArgs(prior)
		if !ok {
			continue // args not derivable from prior outputs — resist speculation
		}
		best, bestArgs = p, args
	}
	if best == nil {
		return nil
	}

	c := &ToolCall{
		Tool: best.PredictTool,
		Args: bestArgs,
		Meta: cloneMeta(best.Meta),
		Spec: SpeculationContext{
			Speculative: true,
			Epoch:       s.nextEpoch(),
			ParentEpoch: parentEpoch,
		},
	}

	// THE LAW: default-deny on effects. Only a provably effect-free call may run
	// ahead of the model. A mutating / unstamped / ambiguous call is never
	// speculated — it is left for the model's authoritative emission.
	if !specEffectFree(c) {
		return nil
	}
	return c
}

// cloneMeta copies a pattern's meta so the issued call never aliases the pattern
// table (a later Learn or a caller mutation cannot reach back into a live call).
func cloneMeta(m map[string]string) map[string]string {
	if len(m) == 0 {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// PredictionMatches reports whether the model's AUTHORITATIVE next call confirms
// the speculation: same predicted tool and byte-identical derived args (the args
// are what make a speculation reusable — a matching tool with different args is a
// MISS, because the provisional result was computed for the predicted args). A nil
// predicted or authoritative call is a miss (fail-closed: an absent prediction
// never "matches").
func PredictionMatches(predicted, authoritative *ToolCall) bool {
	if predicted == nil || authoritative == nil {
		return false
	}
	if predicted.Tool != authoritative.Tool {
		return false
	}
	return refsEqual(predicted.Args, authoritative.Args)
}

// refsEqual reports whether two Refs address the same bytes for the purpose of a
// speculation match: identical content digest when both are backend-resident, or
// identical inline bytes when both are inline. Differing kinds or digests are a
// mismatch (the provisional result was computed for the predicted args).
func refsEqual(a, b Ref) bool {
	if a.Kind != b.Kind {
		return false
	}
	switch a.Kind {
	case RefInline:
		return string(a.Inline) == string(b.Inline)
	default:
		return a.Digest != "" && a.Digest == b.Digest
	}
}

// Commit resolves a CORRECT prediction: it Promotes the speculation's provisional
// effects across every sink (making them durable) and reports OutcomeCommitted.
// This is the predict->match->commit path. Any sink error short-circuits and is
// returned with OutcomeCommitted still reported as the intended resolution's
// classification — the caller decides whether a partial promote is fatal.
func Commit(ctx context.Context, sinks []ProvisionalSink, txn TxnID, epoch uint64) (Outcome, error) {
	for _, s := range sinks {
		if err := s.Promote(ctx, txn, epoch); err != nil {
			return OutcomeCommitted, err
		}
	}
	return OutcomeCommitted, nil
}

// Squash resolves an INCORRECT prediction: it Rolls back the speculation's
// provisional effects across every sink (retracting them — "squash actually undoes
// the effect", the frozen cross-driver contract) and reports OutcomeSquashed. This
// is the predict->miss->squash path.
func Squash(ctx context.Context, sinks []ProvisionalSink, txn TxnID, epoch uint64) (Outcome, error) {
	for _, s := range sinks {
		if err := s.Rollback(ctx, txn, epoch); err != nil {
			return OutcomeSquashed, err
		}
	}
	return OutcomeSquashed, nil
}

// Resolve drives the whole lifecycle from a speculation's outcome: it compares the
// predicted call against the model's authoritative emission and either Commits
// (match) or Squashes (miss) the provisional effects across the sinks, returning
// the resolved Outcome. It is the one entrypoint a speculative dispatcher calls
// once the model's real next call is known — the predict->{match->commit |
// miss->squash} fork in a single call.
func Resolve(ctx context.Context, sinks []ProvisionalSink, txn TxnID, predicted, authoritative *ToolCall) (Outcome, error) {
	epoch := uint64(0)
	if predicted != nil {
		epoch = predicted.Spec.Epoch
	}
	if PredictionMatches(predicted, authoritative) {
		return Commit(ctx, sinks, txn, epoch)
	}
	return Squash(ctx, sinks, txn, epoch)
}

// ---------------------------------------------------------------------------
// BufferSink — a REAL ProvisionalSink (the v0.1 no-op replaced).
// ---------------------------------------------------------------------------

// BufferSink is a concrete ProvisionalSink that actually buffers provisional
// effects per epoch and either makes them durable on Promote or RETRACTS them on
// Rollback. It is the store-buffer the seam's contract describes: an effect
// produced under a speculative epoch lands here as provisional and becomes visible
// ONLY on Promote; Rollback discards it so a squash leaves no trace. v0.1's MMU
// registered a sink whose Promote was a bare append and whose Rollback merely
// dropped the scratch; this one closes the loop so "squash actually undoes the
// effect" is executable, not aspirational.
//
// It is intentionally small and self-contained (a content-free effect ledger keyed
// by epoch) so the spine package carries no driver dependency: a real MMU sink
// composes the same Promote/Rollback shape over its own paged store.
type BufferSink struct {
	mu        sync.Mutex
	provis    map[uint64][]Ref // epoch -> provisional (not-yet-committed) effects
	committed []Ref            // promoted effects, in promote order (durable)
}

// NewBufferSink builds an empty provisional-effect sink.
func NewBufferSink() *BufferSink {
	return &BufferSink{provis: map[uint64][]Ref{}}
}

// Stage records a provisional effect produced under a speculative epoch. The effect
// is NOT visible in Committed until the epoch is Promoted; a Rollback of the epoch
// discards it. This is the only way an effect enters the buffer under speculation.
func (b *BufferSink) Stage(epoch uint64, eff Ref) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.provis == nil {
		b.provis = map[uint64][]Ref{}
	}
	b.provis[epoch] = append(b.provis[epoch], eff)
}

// Promote makes an epoch's provisional effects durable (appended to Committed in
// stage order) and clears the scratch. txn is accepted for the frozen signature; a
// non-zero txn scopes a transaction the same way (this sink keys on epoch). A
// Promote of an unknown/empty epoch is a no-op — promoting nothing commits nothing.
func (b *BufferSink) Promote(_ context.Context, _ TxnID, epoch uint64) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	effs := b.provis[epoch]
	b.committed = append(b.committed, effs...)
	delete(b.provis, epoch)
	return nil
}

// Rollback RETRACTS an epoch's provisional effects: the scratch is dropped and
// nothing reaches Committed. This is the executable form of "squash actually undoes
// the effect" — after a Rollback the buffer holds no trace of the squashed branch.
func (b *BufferSink) Rollback(_ context.Context, _ TxnID, epoch uint64) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.provis, epoch)
	return nil
}

// Committed returns the durable effects in promote order (a copy; the caller may
// not mutate the sink's ledger). The forensic witness that a commit landed and a
// squash left nothing.
func (b *BufferSink) Committed() []Ref {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]Ref(nil), b.committed...)
}

// PendingEpochs reports how many epochs still hold un-resolved provisional effects
// (every staged epoch must end Promoted or Rolled back; a non-zero count after a
// run is a leaked speculation).
func (b *BufferSink) PendingEpochs() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.provis)
}

var _ ProvisionalSink = (*BufferSink)(nil)

// ---------------------------------------------------------------------------
// The default-deny-on-effects predicate (in-lane mirror of vdso.Speculatable).
// ---------------------------------------------------------------------------

// specEffectFree reports whether a call is provably safe to run ahead of the model:
// read-only AND idempotent AND not write-shaped/destructive. It is the SAME
// decision internal/vdso.Speculatable makes (vdso imports abi, so the predicate is
// mirrored here, not imported) over the shared Meta keys; the write-shape name
// heuristic is the same over-approximation the kernel's cache gate uses. Fails
// CLOSED: a nil or unstamped call is non-speculatable.
func specEffectFree(c *ToolCall) bool {
	if c == nil {
		return false
	}
	if !specMetaTrue(c, "readOnlyHint") {
		return false
	}
	if !specMetaTrue(c, "idempotentHint") {
		return false
	}
	if specMetaTrue(c, "destructive") || specWriteShaped(c.Tool) {
		return false
	}
	return true
}

func specMetaTrue(c *ToolCall, k string) bool {
	if c.Meta == nil {
		return false
	}
	return c.Meta[k] == "true"
}

// specWriteShapeNeedles mirrors internal/vdso.writeShapeNeedles — the tool-NAME
// substrings that mark a call write-shaped regardless of its hints. Kept in sync
// with vdso (the canonical list); the abi package cannot import vdso (vdso imports
// abi), so the two share the contract, not the code.
var specWriteShapeNeedles = []string{"write", "edit", "delete", "patch", "exec", "run", "book", "update", "cancel", "send"}

func specWriteShaped(tool string) bool {
	t := strings.ToLower(tool)
	for _, p := range specWriteShapeNeedles {
		if strings.Contains(t, p) {
			return true
		}
	}
	return false
}
