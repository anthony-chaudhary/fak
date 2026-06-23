package spec

import (
	"context"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// The reserved OpsSpec ops (abi.OpsSpec = [64,96)). Names per ARCHITECTURE.md: a
// speculative turn COMMITs (promotes) the accepted branch and SQUASHes (rolls back)
// the rejected one. Each op fans its method across every registered ProvisionalSink.
const (
	// OpSpecCommit promotes a resolved speculation/transaction: its provisional
	// effects become durable. For the KV sink this is a finalize no-op (the accepted
	// positions already live in the cache); the Outcome is OutcomeCommitted.
	OpSpecCommit abi.OpCode = 64
	// OpSpecSquash rolls back a resolved speculation/transaction: its provisional
	// effects are retracted. For the KV sink this evicts the provisional span
	// bit-exactly; the Outcome is OutcomeSquashed.
	OpSpecSquash abi.OpCode = 65
)

// The two branches of one speculative round, keyed under the same TxnID by epoch: the
// accepted prefix (promoted) and the rejected suffix (squashed). A caller may use any
// non-zero epochs; these are the ones SpeculativeGreedy uses.
const (
	// EpochAccept tags the accepted-prefix branch of a speculative round.
	EpochAccept uint64 = 1
	// EpochReject tags the rejected-suffix branch of a speculative round.
	EpochReject uint64 = 2
)

// span is the provisional KV region of one open speculation: positions [from, from+n)
// of cache were appended speculatively under one (txn, epoch) and are provisional
// until the kernel resolves them with Promote or Rollback.
type span struct {
	cache *model.KVCache
	from  int
	n     int
}

// key identifies one open speculation. A TxnID scopes a transaction; the epoch
// distinguishes branches within it (e.g. the accepted vs the rejected span of a round).
type key struct {
	txn   abi.TxnID
	epoch uint64
}

// Sink implements abi.ProvisionalSink for speculative / transactional KV. It records
// the provisional KV span produced under each open (txn, epoch) with Open, and when
// the kernel resolves it either Promotes it (the accepted branch — the KV already
// lives in the cache, so promotion just finalizes the bookkeeping) or Rolls it back
// (the rejected branch — removed from the KV cache bit-exactly via model.KVCache.Evict,
// byte-identical to never having drafted it). It is the first registrant of the frozen
// abi.ProvisionalSink seam (registered, gated, by Install).
//
// A Sink is safe for concurrent use; the per-(txn,epoch) bookkeeping is mutex-guarded.
// The KV mutation in Rollback runs while holding no lock on the cache itself — callers
// must not resolve two speculations that touch the SAME cache concurrently (a single
// decode lane never does: the lane is serial by construction, polymodel.Schedule).
type Sink struct {
	mu      sync.Mutex
	pending map[key]span
}

// NewSink returns an empty Sink. Use Install to also register it on the kernel.
func NewSink() *Sink { return &Sink{pending: make(map[key]span)} }

// Compile-time proof that Sink satisfies the frozen interface.
var _ abi.ProvisionalSink = (*Sink)(nil)

// Open records that positions [from, from+n) of cache are the provisional KV of the
// speculation (txn, epoch). The caller appends those positions (a drafted span) BEFORE
// Open; the kernel later resolves them via Promote or Rollback (directly or through
// OpSpecCommit / OpSpecSquash). A non-positive n or nil cache is ignored, so a
// fully-accepted round (no rejected span) or a fully-rejected one (no accepted span)
// needs no special-casing at the call site.
func (s *Sink) Open(txn abi.TxnID, epoch uint64, cache *model.KVCache, from, n int) {
	if cache == nil || n <= 0 || from < 0 {
		return
	}
	s.mu.Lock()
	s.pending[key{txn, epoch}] = span{cache: cache, from: from, n: n}
	s.mu.Unlock()
}

// Promote (abi.ProvisionalSink) finalizes the speculation: its provisional KV is now
// durable. The accepted positions already live in the cache, so promotion only drops
// the bookkeeping. Idempotent — an unknown or already-resolved (txn, epoch) is a no-op.
func (s *Sink) Promote(_ context.Context, txn abi.TxnID, epoch uint64) error {
	s.mu.Lock()
	delete(s.pending, key{txn, epoch})
	s.mu.Unlock()
	return nil
}

// Rollback (abi.ProvisionalSink) retracts the speculation: its provisional KV span is
// removed from the cache with the bit-exact model.KVCache.Evict — byte-identical to
// never having drafted it. Idempotent — an unknown or already-resolved (txn, epoch) is
// a no-op. The Evict runs after the bookkeeping is dropped, so a second Rollback of the
// same key cannot double-evict.
func (s *Sink) Rollback(_ context.Context, txn abi.TxnID, epoch uint64) error {
	s.mu.Lock()
	sp, ok := s.pending[key{txn, epoch}]
	delete(s.pending, key{txn, epoch})
	s.mu.Unlock()
	if !ok {
		return nil
	}
	sp.cache.Evict(sp.from, sp.n)
	return nil
}

// OpenCount reports the number of unresolved speculations (open provisional spans). A
// drained lane leaves this at 0; a non-zero value after a decode loop means a
// speculation was never resolved (a leak the witness suite asserts against).
func (s *Sink) OpenCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.pending)
}

// op is the abi.Op for one reserved OpsSpec code. commit selects Promote vs Rollback;
// each Invoke fans the chosen resolution across every registered ProvisionalSink for
// the call's (Txn, Spec.Epoch), so the speculation lifecycle is driven through the
// frozen kernel op table rather than a private path.
type op struct {
	code   abi.OpCode
	commit bool
}

func (o op) Code() abi.OpCode { return o.code }

// Invoke resolves the call's (Txn, Spec.Epoch) across every registered sink. It does
// not consult the Kernel (resolution is sink-local), so it is safe to invoke with a
// nil kernel in a unit test. The returned Verdict is always Allow — resolving a
// provisional lifecycle is a kernel-internal bookkeeping op, not an adjudicated tool call.
func (o op) Invoke(ctx context.Context, _ abi.Kernel, c *abi.ToolCall) (*abi.Result, abi.Verdict) {
	var firstErr error
	for _, s := range abi.ProvisionalSinks() {
		var err error
		if o.commit {
			err = s.Promote(ctx, c.Txn, c.Spec.Epoch)
		} else {
			err = s.Rollback(ctx, c.Txn, c.Spec.Epoch)
		}
		if err != nil && firstErr == nil {
			firstErr = err
		}
	}
	out, status := abi.OutcomeSquashed, abi.StatusOK
	if o.commit {
		out = abi.OutcomeCommitted
	}
	if firstErr != nil {
		status = abi.StatusError
	}
	return &abi.Result{Call: c, Status: status, Outcome: out}, abi.Verdict{Kind: abi.VerdictAllow}
}

// Install registers the speculation seam — a fresh Sink (the first abi.ProvisionalSink
// registrant) and the reserved OpSpecCommit / OpSpecSquash ops — and returns the Sink
// so the caller can Open provisional spans against it. It is the LIVE-PATH wiring, so
// it is gated on polymodel.Enabled(): when the poly-model lane is off (the default),
// Install is a no-op and returns nil, leaving the global ABI registries untouched.
// This is the second of the two safety layers (the first is keeping this leaf out of
// the defconfig so the kernel never links it). RegisterOp panics on a duplicate
// OpCode, so Install must not run twice without an intervening abi.ResetForTest — which
// is exactly the isolation the witness suite uses.
func Install() *Sink {
	if !polymodel.Enabled() {
		return nil
	}
	s := NewSink()
	abi.RegisterProvisionalSink(s)
	abi.RegisterOp(op{code: OpSpecCommit, commit: true})
	abi.RegisterOp(op{code: OpSpecSquash, commit: false})
	return s
}

// Drafter proposes speculative tokens for the active decoder — the "idle co-resident
// models become the speculation ensemble" idea (polymodel.PickDrafter selects one in
// production), or the target's own cheaper quant. It is never required to be correct: a
// wrong draft is simply squashed, costing nothing but a bit-exact rollback. Draft
// returns up to k proposed token ids continuing the committed context; Commit advances
// the drafter's own state by the tokens the target actually committed this round, so
// its next Draft continues from the true context (a model-backed drafter rolls back its
// own speculative KV in Commit).
type Drafter interface {
	Draft(k int) []int
	Commit(committed []int)
}

// SpeculativeGreedy runs n tokens of greedy speculative decoding on target, drafting up
// to k tokens per round from drafter, and resolves every round THROUGH the Sink: the
// accepted prefix is promoted (durable) and the rejected suffix is squashed (the
// bit-exact model.KVCache.Evict). It returns the committed token ids plus draft/accept/
// rollback counts. Because greedy speculation is provably lossless, the output is
// token-identical to plain greedy decode of target — a property that holds ONLY if the
// squash is bit-exact, which is exactly what the witness asserts.
//
// target must be a fresh, un-prefilled session (SpeculativeGreedy prefills it). The
// verify here is sequential target Steps (correctness-faithful); the single-pass
// batched verify that turns acceptance into throughput is rung #533.
func SpeculativeGreedy(ctx context.Context, sink *Sink, target *model.Session, prompt []int, n, k int, drafter Drafter) (out []int, drafted, accepted, rolledBack int) {
	tl := target.Prefill(prompt)
	out = make([]int, 0, n)
	var txn abi.TxnID

	for len(out) < n {
		from := target.Cache.Len()

		// 1. The drafter proposes up to k tokens (clamped); k<=0 or an empty draft
		//    degrades to plain greedy (verify advances by the correction alone). The
		//    clamp bounds an over-eager drafter for EVERY k (a negative k clamps to 0,
		//    so it cannot run an unbounded draft).
		limit := k
		if limit < 0 {
			limit = 0
		}
		drafts := drafter.Draft(k)
		if len(drafts) > limit {
			drafts = drafts[:limit]
		}
		kk := len(drafts)
		drafted += kk

		// 2. The target verifies: its argmax at the current position (from tl) plus its
		//    argmax after each drafted token fed sequentially → kk+1 argmaxes. Each Step
		//    appends one provisional position to the target cache (positions [from, from+kk)).
		targetArgmax := make([]int, 0, kk+1)
		targetArgmax = append(targetArgmax, argmax(tl))
		for j := 0; j < kk; j++ {
			targetArgmax = append(targetArgmax, argmax(target.Step(drafts[j])))
		}

		// 3. polymodel.AcceptGreedy picks the accepted prefix and the keep/evict split.
		res := polymodel.AcceptGreedy(drafts, targetArgmax)
		accepted += res.Accepted
		txn++

		// 4. Resolve the round through the ProvisionalSink seam. The accepted prefix
		//    [from, from+Accepted) is promoted (durable); the rejected suffix
		//    [from+Accepted, from+kk) is squashed (bit-exact Evict). Open ignores a
		//    zero-length branch, so full-accept / full-reject need no special case.
		sink.Open(txn, EpochAccept, target.Cache, from, res.Accepted)
		_ = sink.Promote(ctx, txn, EpochAccept)
		sink.Open(txn, EpochReject, target.Cache, from+res.Accepted, res.EvictKV)
		_ = sink.Rollback(ctx, txn, EpochReject)
		rolledBack += res.EvictKV

		// 5. Emit the committed tokens (accepted drafts + the correction/bonus) and
		//    advance the target by the correction so the next round shares the context.
		//    The drafter is told the same committed tokens so it stays aligned.
		correction := targetArgmax[res.Accepted]
		committed := make([]int, 0, res.Accepted+1)
		for j := 0; j < res.Accepted && len(out) < n; j++ {
			out = append(out, drafts[j])
			committed = append(committed, drafts[j])
		}
		if len(out) < n {
			out = append(out, correction)
			committed = append(committed, correction)
			tl = target.Step(correction)
		}
		drafter.Commit(committed)
	}
	return out, drafted, accepted, rolledBack
}

// argmax returns the index of the maximum logit (ties resolve to the lowest index),
// matching the greedy decode rule the losslessness proof compares against.
func argmax(v []float32) int {
	bi, bv := 0, float32(0)
	for i, x := range v {
		if i == 0 || x > bv {
			bv, bi = x, i
		}
	}
	return bi
}
