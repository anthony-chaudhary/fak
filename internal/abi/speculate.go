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

	// MinTrials is the empirical-α WARMUP floor: a pattern must accumulate at least
	// this many observed outcomes (Observe) before Predict gates on its MEASURED
	// hit-rate instead of the declared SuccessProb prior. Below the floor the small
	// sample is not yet trustworthy, so the declared prior governs. 0 =>
	// defaultMinTrials.
	MinTrials int

	mu       sync.Mutex
	patterns map[string][]SpecPattern // keyed by Signature
	stats    map[string]*patternStat  // keyed by statKey(sig, tool): observed outcomes
	epoch    uint64                   // monotonically issued speculation epoch ids
}

// patternStat is the running empirical-outcome tally for one pattern: how many
// speculations were issued-and-resolved (trials) and how many the model's
// authoritative call confirmed (hits). The MEASURED success probability α is
// derived from these by laplaceRate, closing the loop the declared SuccessProb
// prior only opens.
type patternStat struct {
	hits   uint64
	trials uint64
}

// defaultMinTrials is the warmup floor used when Speculator.MinTrials is unset: a
// pattern gates on its declared prior until this many outcomes are observed, then
// switches to the measured rate. Small enough to adapt quickly, large enough that a
// handful of unlucky early misses cannot evict a genuinely good pattern.
const defaultMinTrials = 20

// statKey is the per-pattern stats key. Outcomes are attributed by (signature,
// predicted tool) — the same pair Predict issues under and Observe resolves — so a
// pattern's measured α is tracked independently of any other pattern sharing its
// signature.
func statKey(sig, tool string) string { return sig + "\x00" + tool }

// laplaceRate is the Laplace (add-one) smoothed hit-rate (hits+1)/(trials+2): a
// Beta(1,1) uniform prior so a pattern with few trials is pulled toward 0.5 rather
// than swinging to a hard 0 or 1 on a single outcome, and converges to the raw
// hits/trials as evidence accumulates.
func laplaceRate(hits, trials uint64) float64 {
	return float64(hits+1) / float64(trials+2)
}

// NewSpeculator builds an ENABLED speculator with the given probability floor. The
// zero-value Speculator is the disabled (v0.1 no-op) form; this constructor is the
// opt-in. Patterns are added with Learn.
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

// effectiveProbLocked is the probability Predict actually gates on for one pattern:
// the MEASURED Laplace-smoothed hit-rate once the pattern has cleared the warmup
// floor, else the declared SuccessProb prior. Called under s.mu.
//
// It deliberately does NOT max(declared, measured): an optimistic declared prior
// must NOT be able to permanently mask a bad measured rate, or the loop could never
// self-correct — a pattern the model keeps refuting has to fall below the floor and
// stop being issued. Symmetrically, once warm the measured rate can PROMOTE a
// pattern whose conservative declared prior sat below MinProb.
func (s *Speculator) effectiveProbLocked(sig, tool string, declared float64) float64 {
	st := s.stats[statKey(sig, tool)]
	if st == nil || st.trials < uint64(s.warmupFloor()) {
		return declared // warmup: too little evidence to trust the measured rate
	}
	return laplaceRate(st.hits, st.trials)
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
	// Snapshot the candidates AND their effective (measured-or-declared) probability
	// under the lock, then release it before calling the user's DeriveArgs — that
	// callback must never run under s.mu (it could be slow or re-enter the
	// speculator). eff[i] is the α Predict gates on: the measured rate once pattern i
	// is warm, else its declared prior.
	s.mu.Lock()
	cands := s.patterns[sig]
	eff := make([]float64, len(cands))
	for i := range cands {
		eff[i] = s.effectiveProbLocked(sig, cands[i].PredictTool, cands[i].SuccessProb)
	}
	s.mu.Unlock()

	// Pick the highest-probability pattern that clears the floor AND can derive its
	// args, ranking on the effective (empirically-corrected) probability. A pattern
	// that cannot derive its args is skipped, never guessed.
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

// Observe feeds a resolved speculation's ground-truth outcome back into the
// pattern's empirical hit-rate — the loop-closing counterpart to Predict. Where
// Predict ISSUES a speculation off the declared prior, Observe RECORDS whether the
// model's authoritative call confirmed it, so the α that gates future predictions
// is MEASURED on real traffic. sig is the context signature the speculation was
// issued under; predicted is the call Predict returned; authoritative is the
// model's real next call. The trial is attributed to the pattern keyed by (sig,
// predicted.Tool) and counts a hit exactly when PredictionMatches — the SAME
// verdict Resolve uses to commit/squash, so the measured rate can never drift from
// the commit/squash ledger.
//
// A nil/disabled speculator, or a nil predicted call, observes nothing: the
// default-off invariant means a kernel that never speculates measures nothing.
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

// MeasuredProb reports the Laplace-smoothed empirical success probability observed
// for the pattern keyed by (sig, tool), and whether it has cleared the warmup floor
// (MinTrials). While warm is false the sample is too small to trust and Predict
// still gates on the pattern's declared SuccessProb prior; once warm is true Predict
// gates on this measured value. An unobserved pattern reports (0, false).
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

// ResolveAndObserve is the loop-closing entrypoint a speculative dispatcher calls
// once the model's authoritative next call is known: it both feeds the outcome back
// into the pattern's measured hit-rate (Observe) and drives the provisional
// lifecycle (Resolve) — commit on a match, squash on a miss. Bundling them means a
// caller that holds the live signature closes the empirical-α loop with one call
// and cannot forget to pair an Observe with a Resolve; the measured α and the
// commit/squash ledger therefore advance in lockstep. sig is the context signature
// the speculation was issued under; the (predicted, authoritative, sinks, txn)
// arguments are exactly Resolve's.
func (s *Speculator) ResolveAndObserve(ctx context.Context, sinks []ProvisionalSink, txn TxnID, sig string, predicted, authoritative *ToolCall) (Outcome, error) {
	s.Observe(sig, predicted, authoritative)
	return Resolve(ctx, sinks, txn, predicted, authoritative)
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
	rollbacks uint64
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
	b.rollbacks++
	return nil
}

// Rollbacks reports how many provisional epochs this sink discarded.
func (b *BufferSink) Rollbacks() uint64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.rollbacks
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

// IsWriteShaped is the EXPORTED write-shape check — the same name-substring
// over-approximation specWriteShaped / vdso.writeShapeNeedles use, surfaced so a
// consumer (the before-consumption write barrier in internal/agent, #1319) can ask "is
// this call write-shaped?" without re-deriving the needle list. A write-shaped call is
// never speculated AND, under the barrier, is never committed behind an unconfirmed
// speculative read.
func IsWriteShaped(tool string) bool { return specWriteShaped(tool) }
