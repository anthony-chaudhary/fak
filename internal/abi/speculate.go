// speculate.go is the LIVE driver for SEAM 4: a predicted-next-path engine that
// predicts, runs on slack, commits on a match, squashes on a miss — driving the
// frozen SpeculationContext / Outcome / ProvisionalSink contract end to end. It
// adds nothing to the frozen wire (types.go is untouched; the goldens do not move).
//
// DEFAULT-OFF POSTURE — a zero-value Speculator (Enabled=false) predicts nothing,
// so a kernel that never opts in behaves exactly as v0.1.
//
// DEFAULT-DENY ON EFFECTS — a call is speculated only if provably effect-free:
// read-only AND idempotent AND not write-shaped. Our "stores" hit payments, emails,
// and deletes with no rollback, so the gate fails closed on an unstamped or ambiguous
// call. The predicate mirrors internal/vdso.Speculatable over the shared meta keys
// "readOnlyHint" / "idempotentHint" / "destructive" and the write-shape name heuristic
// (vdso imports abi, so abi cannot import vdso; the two must not drift).
//
// PASTE-style prediction — only args DERIVABLE from prior tool outputs are
// predictable, so a pattern that cannot derive its args declines to predict.

package abi

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
)

// SpecPattern is one PASTE-style prediction rule: it fires when Signature matches
// the live context signature, predicts PredictTool, and derives Args from prior
// tool outputs via DeriveArgs (not derivable => the pattern declines). SuccessProb gates whether predicting is worth the slack.
type SpecPattern struct {
	// Signature is the context signature this pattern keys on (e.g. the just-completed
	// tool name); a pattern fires only on an exact match — the predictor never guesses across signatures.
	Signature string

	// PredictTool is the tool type the pattern predicts will be called next.
	PredictTool string

	// DeriveArgs builds the predicted call's Args from the prior tool outputs; ok=false
	// means the args cannot be derived from what is known and the pattern declines rather than guessing.
	DeriveArgs func(prior []*Result) (Ref, bool)

	// Meta is stamped onto the predicted call — the read-only / idempotent hints the default-deny-on-effects gate reads.
	Meta map[string]string

	// SuccessProb is the empirical probability the prediction is correct, in [0,1]; below the Speculator's MinProb it is not issued.
	SuccessProb float64
}

// Speculator is the live predicted-next-path engine for SEAM 4: it holds the
// PASTE-style pattern table and issues a speculative ToolCall with a fresh epoch
// when a pattern matches AND the predicted call clears the default-deny-on-effects gate. Zero value (Enabled=false) predicts nothing.
type Speculator struct {
	// Enabled is the DEFAULT-OFF master switch: false (the zero value) => Predict always returns nil, reproducing v0.1 exactly.
	Enabled bool

	// MinProb is the probability floor a pattern must clear to be issued; zero admits any matching pattern.
	MinProb float64

	// MinTrials is the warmup floor: a pattern must accumulate this many observed outcomes
	// before Predict gates on its MEASURED hit-rate instead of the declared SuccessProb prior. 0 => defaultMinTrials.
	MinTrials int

	mu       sync.Mutex
	patterns map[string][]SpecPattern // keyed by Signature
	stats    map[string]*patternStat  // keyed by statKey(sig, tool): observed outcomes
	epoch    uint64                   // monotonically issued speculation epoch ids
}

// patternStat is the running empirical-outcome tally for one pattern: trials
// (issued-and-resolved speculations) and hits (those the model's authoritative call confirmed). laplaceRate derives the measured α from these.
type patternStat struct {
	hits   uint64
	trials uint64
}

// defaultMinTrials is the warmup floor when Speculator.MinTrials is unset: small enough to adapt quickly, large enough that early misses cannot evict a good pattern.
const defaultMinTrials = 20

// statKey keys outcomes by (signature, predicted tool) — the pair Predict issues under and Observe resolves — so patterns sharing a signature track α independently.
func statKey(sig, tool string) string { return sig + "\x00" + tool }

// laplaceRate is the Laplace-smoothed hit-rate (hits+1)/(trials+2): a Beta(1,1) prior
// so few trials pull toward 0.5 instead of swinging to 0 or 1, converging to the raw rate as evidence accumulates.
func laplaceRate(hits, trials uint64) float64 {
	return float64(hits+1) / float64(trials+2)
}

// NewSpeculator builds an ENABLED speculator with the given probability floor — the opt-in counterpart of the zero-value (disabled, v0.1 no-op) form.
func NewSpeculator(minProb float64) *Speculator {
	return &Speculator{
		Enabled:  true,
		MinProb:  minProb,
		patterns: map[string][]SpecPattern{},
		stats:    map[string]*patternStat{},
	}
}

// warmupFloor is the effective MinTrials (defaulted). Read under s.mu.
func (s *Speculator) warmupFloor() int {
	if s.MinTrials > 0 {
		return s.MinTrials
	}
	return defaultMinTrials
}

// effectiveProbLocked is the probability Predict gates on for one pattern: the
// MEASURED Laplace-smoothed hit-rate once warm, else the declared SuccessProb prior.
// Called under s.mu.
//
// It deliberately does NOT max(declared, measured): an optimistic prior must not permanently mask a bad measured rate, or the loop could never self-correct.
func (s *Speculator) effectiveProbLocked(sig, tool string, declared float64) float64 {
	st := s.stats[statKey(sig, tool)]
	if st == nil || st.trials < uint64(s.warmupFloor()) {
		return declared // warmup: too little evidence to trust the measured rate
	}
	return laplaceRate(st.hits, st.trials)
}

// Learn registers a prediction pattern; patterns are indexed by signature, so prediction is an O(1) lookup over a small per-signature list.
func (s *Speculator) Learn(p SpecPattern) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.patterns == nil {
		s.patterns = map[string][]SpecPattern{}
	}
	s.patterns[p.Signature] = append(s.patterns[p.Signature], p)
}

// nextEpoch issues a fresh, non-zero speculation epoch id (0 is reserved for "not speculative" by the frozen contract).
func (s *Speculator) nextEpoch() uint64 {
	return atomic.AddUint64(&s.epoch, 1)
}

// Predict returns the speculative ToolCall to run ahead of the model for the given
// context signature and prior tool outputs, or nil when nothing should be speculated:
// the speculator is disabled, no pattern matches, the best match is below MinProb,
// DeriveArgs cannot derive the args, or the predicted call is not provably effect-free.
//
// A returned call is stamped Speculative=true with a fresh non-zero Epoch branching from parentEpoch, so every effect it produces is provisional until Promote/Rollback.
func (s *Speculator) Predict(sig string, prior []*Result, parentEpoch uint64) *ToolCall {
	if s == nil || !s.Enabled {
		return nil // default-deny: a disabled (or nil) speculator never predicts
	}
	// Snapshot candidates and their effective probabilities under the lock, then release
	// it before calling the user's DeriveArgs — that callback must never run under s.mu (it could be slow or re-enter the speculator).
	s.mu.Lock()
	cands := s.patterns[sig]
	eff := make([]float64, len(cands))
	for i := range cands {
		eff[i] = s.effectiveProbLocked(sig, cands[i].PredictTool, cands[i].SuccessProb)
	}
	s.mu.Unlock()

	// Pick the highest-probability pattern that clears the floor AND can derive its args; a pattern that cannot derive its args is skipped, never guessed.
	var best *SpecPattern
	var bestArgs Ref
	var bestProb float64
	for i := range cands {
		p := &cands[i]
		prob := eff[i]
		if prob < s.MinProb {
			continue
		}
		if best != nil && prob <= bestProb {
			continue
		}
		if p.DeriveArgs == nil {
			continue
		}
		args, ok := p.DeriveArgs(prior)
		if !ok {
			continue // args not derivable from prior outputs — resist speculation
		}
		best, bestArgs, bestProb = p, args, prob
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

	// Default-deny on effects: only a provably effect-free call may run ahead of the
	// model; mutating/unstamped/ambiguous calls are left for the model's authoritative emission.
	if !specEffectFree(c) {
		return nil
	}
	return c
}

// cloneMeta copies a pattern's meta so the issued call never aliases the pattern table — a later Learn or caller mutation cannot reach back into a live call.
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

// Observe feeds a resolved speculation's ground-truth outcome back into the
// pattern's empirical hit-rate — the loop-closing counterpart to Predict. sig is
// the issue-time signature; predicted is the call Predict returned; authoritative
// is the model's real next call. The trial is attributed to (sig, predicted.Tool)
// and counts a hit exactly when PredictionMatches — the SAME verdict Resolve uses to commit/squash, so the measured rate can never drift from the commit/squash ledger.
func (s *Speculator) Observe(sig string, predicted, authoritative *ToolCall) {
	if s == nil || !s.Enabled || predicted == nil {
		return
	}
	key := statKey(sig, predicted.Tool)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stats == nil {
		s.stats = map[string]*patternStat{}
	}
	st := s.stats[key]
	if st == nil {
		st = &patternStat{}
		s.stats[key] = st
	}
	st.trials++
	if PredictionMatches(predicted, authoritative) {
		st.hits++
	}
}

// MeasuredProb reports the Laplace-smoothed empirical success probability for the
// pattern keyed by (sig, tool) and whether it has cleared the warmup floor. While
// warm is false Predict still gates on the declared SuccessProb prior; unobserved patterns report (0, false).
func (s *Speculator) MeasuredProb(sig, tool string) (prob float64, warm bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := s.stats[statKey(sig, tool)]
	if st == nil {
		return 0, false
	}
	return laplaceRate(st.hits, st.trials), st.trials >= uint64(s.warmupFloor())
}

// PredictionMatches reports whether the model's AUTHORITATIVE next call confirms
// the speculation: same tool and byte-identical derived args — a matching tool with
// different args is a MISS, because the provisional result was computed for the
// predicted args. A nil predicted or authoritative call is a miss: an absent prediction never "matches".
func PredictionMatches(predicted, authoritative *ToolCall) bool {
	if predicted == nil || authoritative == nil {
		return false
	}
	if predicted.Tool != authoritative.Tool {
		return false
	}
	return refsEqual(predicted.Args, authoritative.Args)
}

// refsEqual reports whether two Refs address the same bytes for a speculation match:
// identical digest when both are backend-resident, identical inline bytes when both are inline; differing kinds or digests are a mismatch.
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

// Commit resolves a CORRECT prediction (the predict->match->commit path): it Promotes
// the speculation's provisional effects across every sink and reports OutcomeCommitted.
// Any sink error short-circuits and is returned with OutcomeCommitted still reported — the caller decides whether a partial promote is fatal.
func Commit(ctx context.Context, sinks []ProvisionalSink, txn TxnID, epoch uint64) (Outcome, error) {
	for _, s := range sinks {
		if err := s.Promote(ctx, txn, epoch); err != nil {
			return OutcomeCommitted, err
		}
	}
	return OutcomeCommitted, nil
}

// Squash resolves an INCORRECT prediction (the predict->miss->squash path): it Rolls
// back the speculation's provisional effects across every sink — "squash actually undoes the effect" — and reports OutcomeSquashed.
func Squash(ctx context.Context, sinks []ProvisionalSink, txn TxnID, epoch uint64) (Outcome, error) {
	for _, s := range sinks {
		if err := s.Rollback(ctx, txn, epoch); err != nil {
			return OutcomeSquashed, err
		}
	}
	return OutcomeSquashed, nil
}

// Resolve drives the whole lifecycle from a speculation's outcome: it compares the
// predicted call against the model's authoritative emission and either Commits (match)
// or Squashes (miss) the provisional effects across the sinks. It is the one entrypoint
// a speculative dispatcher calls once the model's real next call is known.
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

// ResolveAndObserve is the loop-closing entrypoint a speculative dispatcher calls once
// the model's authoritative next call is known: it feeds the outcome into the pattern's
// measured hit-rate (Observe) and drives the provisional lifecycle (Resolve) in one call,
// so the measured α and the commit/squash ledger advance in lockstep and a caller cannot
// forget to pair them. sig is the issue-time signature.
func (s *Speculator) ResolveAndObserve(ctx context.Context, sinks []ProvisionalSink, txn TxnID, sig string, predicted, authoritative *ToolCall) (Outcome, error) {
	s.Observe(sig, predicted, authoritative)
	return Resolve(ctx, sinks, txn, predicted, authoritative)
}

// BufferSink is a concrete ProvisionalSink that buffers provisional effects per epoch:
// an effect produced under a speculative epoch lands here as provisional and becomes
// visible ONLY on Promote; Rollback discards it so a squash leaves no trace. It is
// intentionally small and self-contained (a content-free effect ledger keyed by epoch)
// so the spine package carries no driver dependency — a real MMU sink composes the same Promote/Rollback shape over its own paged store.
type BufferSink struct {
	mu        sync.Mutex
	provis    map[uint64][]Ref // epoch -> provisional (not-yet-committed) effects
	committed []Ref            // promoted effects, in promote order (durable)
	rollbacks uint64
}

// NewBufferSink builds an empty provisional-effect sink.
func NewBufferSink() *BufferSink {
	return &BufferSink{provis: map[uint64][]Ref{}}
}

// Stage records a provisional effect produced under a speculative epoch; it is NOT
// visible in Committed until the epoch is Promoted, and a Rollback discards it. This is the only way an effect enters the buffer under speculation.
func (b *BufferSink) Stage(epoch uint64, eff Ref) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.provis == nil {
		b.provis = map[uint64][]Ref{}
	}
	b.provis[epoch] = append(b.provis[epoch], eff)
}

// Promote makes an epoch's provisional effects durable (appended to Committed in stage
// order) and clears the scratch; txn is accepted for the frozen signature (this sink keys on epoch). An unknown/empty epoch is a no-op.
func (b *BufferSink) Promote(_ context.Context, _ TxnID, epoch uint64) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	effs := b.provis[epoch]
	b.committed = append(b.committed, effs...)
	delete(b.provis, epoch)
	return nil
}

// Rollback RETRACTS an epoch's provisional effects: the scratch is dropped, nothing reaches Committed, and the buffer holds no trace of the squashed branch.
func (b *BufferSink) Rollback(_ context.Context, _ TxnID, epoch uint64) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.provis, epoch)
	b.rollbacks++
	return nil
}

// Rollbacks reports how many provisional epochs this sink discarded.
func (b *BufferSink) Rollbacks() uint64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.rollbacks
}

// Committed returns the durable effects in promote order (a copy; the caller may not mutate the sink's ledger) — the forensic witness that a commit landed and a squash left nothing.
func (b *BufferSink) Committed() []Ref {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]Ref(nil), b.committed...)
}

// PendingEpochs reports how many epochs still hold un-resolved provisional effects; a non-zero count after a run is a leaked speculation.
func (b *BufferSink) PendingEpochs() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.provis)
}

var _ ProvisionalSink = (*BufferSink)(nil)

// specEffectFree reports whether a call is provably safe to run ahead of the model:
// read-only AND idempotent AND not write-shaped/destructive. It mirrors
// internal/vdso.Speculatable over the shared Meta keys (vdso imports abi, so the predicate is mirrored, not imported). A nil or unstamped call is non-speculatable.
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

// specWriteShapeNeedles mirrors internal/vdso.writeShapeNeedles — the tool-name substrings
// that mark a call write-shaped regardless of its hints. The abi package cannot import vdso (vdso imports abi), so the two share the list, not the code.
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

// IsWriteShaped is the EXPORTED write-shape check — the same name-substring over-approximation
// specWriteShaped / vdso.writeShapeNeedles use, surfaced so the before-consumption write barrier
// in internal/agent (#1319) can ask without re-deriving the needle list. A write-shaped call is never speculated.
func IsWriteShaped(tool string) bool { return specWriteShaped(tool) }
