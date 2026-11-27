package agent

// next_turn_prime.go - fak-private#3728: live next-turn KV prime at tool dispatch.
//
// When a turn ends with tool_calls the next request's prefix (history plus the
// assistant message, up to the tool-result slot) is byte-determined while the tool
// still runs. A NextTurnPrimeRunner renders that prefix, asks the existing prewarm
// admission gate (compute.DecidePrewarmAdmission) whether a warm lands hot, and on
// WarmNow materializes the KV state through the existing InKernelPlanner.PrimeCacheState
// in a goroutine. It never prefills by hand and never touches the radix tree directly.
//
// Bounds: a non-blocking semaphore (default 4, TryAcquire semantics, never queues), a
// timeout (default 60s) that starts when the turn ends, cancellation through the loop
// context and Close, and a skip when the cache already covers the prefix. The option is
// off by default: a nil runner leaves the loop unchanged.

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const (
	// DefaultNextTurnPrimeConcurrency is the no-queue semaphore width.
	DefaultNextTurnPrimeConcurrency = 4
	// DefaultNextTurnPrimeTimeout bounds one prime, measured from the end of the turn.
	DefaultNextTurnPrimeTimeout = 60 * time.Second
)

// NextTurnPrimeConfig tunes a runner; zero values select the defaults above.
type NextTurnPrimeConfig struct {
	MaxConcurrent int
	Timeout       time.Duration
}

// NextTurnPrimeStats is the observable counter set (the semaphore-skip count is
// SkippedSaturated).
type NextTurnPrimeStats struct {
	Started          int64
	Primed           int64
	Failed           int64
	SkippedSaturated int64
	SkippedCovered   int64
	SkippedAdmission int64
}

type nextTurnPrimeFunc func(ctx context.Context, ids []int) (usable bool, err error)

// nextTurnPrefixCover is the optional resident-prefix probe (InKernelPlanner has it).
type nextTurnPrefixCover interface {
	cachedPrefixLen(ids []int) int
}

type nextTurnEncodeFunc func(ctx context.Context, messages []Message, tools []ToolDef) ([]int, error)

// NextTurnPrimeRunner owns the bounded prime goroutines for one or more loops.
type NextTurnPrimeRunner struct {
	prime   nextTurnPrimeFunc
	cover   nextTurnPrefixCover
	encode  nextTurnEncodeFunc
	sem     chan struct{}
	timeout time.Duration
	base    context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup

	started, primed, failed       atomic.Int64
	skipSat, skipCovered, skipAdm atomic.Int64

	// onResult, when set, observes every finished prime (tests and ops taps).
	onResult func(usable bool, err error)
}

// NewNextTurnPrimeRunner builds a runner over a native planner.
func NewNextTurnPrimeRunner(p *InKernelPlanner, cfg NextTurnPrimeConfig) *NextTurnPrimeRunner {
	if p == nil {
		return nil
	}
	prime := func(ctx context.Context, ids []int) (bool, error) {
		rec, err := p.PrimeCacheState(ctx, ids)
		return rec.Usable(), err
	}
	enc := func(ctx context.Context, msgs []Message, tools []ToolDef) ([]int, error) {
		e, err := p.EncodePrompt(ctx, msgs, tools)
		if err != nil {
			return nil, err
		}
		return e.TokenIDs, nil
	}
	return newNextTurnPrimeRunner(prime, p, enc, cfg)
}

func newNextTurnPrimeRunner(prime nextTurnPrimeFunc, cover nextTurnPrefixCover, enc nextTurnEncodeFunc, cfg NextTurnPrimeConfig) *NextTurnPrimeRunner {
	if cfg.MaxConcurrent <= 0 {
		cfg.MaxConcurrent = DefaultNextTurnPrimeConcurrency
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultNextTurnPrimeTimeout
	}
	base, cancel := context.WithCancel(context.Background())
	return &NextTurnPrimeRunner{
		prime: prime, cover: cover, encode: enc,
		sem: make(chan struct{}, cfg.MaxConcurrent), timeout: cfg.Timeout,
		base: base, cancel: cancel,
	}
}

// Stats returns a snapshot of the counters.
func (r *NextTurnPrimeRunner) Stats() NextTurnPrimeStats {
	if r == nil {
		return NextTurnPrimeStats{}
	}
	return NextTurnPrimeStats{
		Started: r.started.Load(), Primed: r.primed.Load(), Failed: r.failed.Load(),
		SkippedSaturated: r.skipSat.Load(), SkippedCovered: r.skipCovered.Load(),
		SkippedAdmission: r.skipAdm.Load(),
	}
}

// Close cancels in-flight primes and waits for them to return. Nil-safe.
func (r *NextTurnPrimeRunner) Close() {
	if r == nil {
		return
	}
	r.cancel()
	r.wg.Wait()
}

// wait blocks until every started prime has returned (tests).
func (r *NextTurnPrimeRunner) wait() { r.wg.Wait() }

// Trigger starts a bounded prime for the prefix messages+tools describe. It returns
// immediately; false means no goroutine was started (nil runner or saturated).
func (r *NextTurnPrimeRunner) Trigger(ctx context.Context, messages []Message, tools []ToolDef) bool {
	if r == nil || r.prime == nil || r.encode == nil {
		return false
	}
	select {
	case r.sem <- struct{}{}:
	default:
		r.skipSat.Add(1)
		return false
	}
	msgs := append([]Message(nil), messages...)
	if ctx == nil {
		ctx = context.Background()
	}
	// The timeout clock starts now (end of turn); the prime dies with the loop ctx,
	// with Close, or at the deadline, whichever is first.
	pctx, cancel := context.WithTimeout(ctx, r.timeout)
	stop := context.AfterFunc(r.base, cancel)
	r.started.Add(1)
	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		defer func() { <-r.sem }()
		defer cancel()
		defer stop()
		r.run(pctx, msgs, tools)
	}()
	return true
}

func (r *NextTurnPrimeRunner) run(ctx context.Context, msgs []Message, tools []ToolDef) {
	ids, err := r.encode(ctx, msgs, tools)
	if err != nil || len(ids) == 0 {
		r.failed.Add(1)
		return
	}
	if r.cover != nil && r.cover.cachedPrefixLen(ids) >= len(ids) {
		r.skipCovered.Add(1)
		return
	}
	window := int(r.timeout / time.Millisecond)
	dec := compute.DecidePrewarmAdmission(compute.PrewarmCandidate{
		PrefixByteKnown:   true, // history + assistant turn is fixed at the tool-call boundary
		WarmPoolFree:      true, // PrimeCacheState admits through the production tree path
		ToolLatencyMillis: window,
		ResidencyMillis:   window,
		PrefixTokens:      len(ids),
		Reason:            "tool-dispatch:next-turn-prime",
	})
	if dec.Verdict != compute.WarmNow {
		r.skipAdm.Add(1)
		return
	}
	usable, perr := r.prime(ctx, ids)
	if perr != nil || !usable {
		r.failed.Add(1)
	} else {
		r.primed.Add(1)
	}
	if r.onResult != nil {
		r.onResult(usable, perr)
	}
}

// primeNextTurn is the loop-side hook: called once a turn's assistant message (already
// appended to messages) carries tool_calls. Nil runner => no-op.
func (c *runConfig) primeNextTurn(ctx context.Context, messages []Message, tools []ToolDef) {
	if c == nil || c.nextTurnPrime == nil {
		return
	}
	c.nextTurnPrime.Trigger(ctx, messages, tools)
}
