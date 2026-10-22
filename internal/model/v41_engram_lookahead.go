package model

import (
	"context"
	"fmt"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
)

// This is an explicit producer contract, not unforgeable factory provenance.
// A producer declares immutable geometry/bytes, concurrent-safe row and encoded
// reads, and an upper bound on its own staging bytes per requested row. Lifetime
// retention is separate and must succeed before any background read starts.
// The requested row width must match the producer: otherwise an adapter could
// silently allocate covering rows outside the declared per-row staging bound.
type v41EngramLookAheadSource interface {
	V41EngramLookAheadReadPolicy(rowBytes int) (stagingBytesPerRow int, eligible bool)
}

func v41EngramForwardLookAheadPolicy(source any, rowBytes int) (int, bool) {
	if policy, ok := source.(v41EngramLookAheadSource); ok {
		return policy.V41EngramLookAheadReadPolicy(rowBytes)
	}
	return 0, false
}

func (s *v41PackedEngramRowSource) V41EngramLookAheadReadPolicy(rowBytes int) (int, bool) {
	if s == nil || rowBytes != s.rowBytes || s.rowBytes <= 0 || s.shard.Offset%int64(s.rowBytes) != 0 {
		return 0, false
	}
	return v41EngramForwardLookAheadPolicy(s.reader, rowBytes)
}

func (s *v41VerifiedEngramRowSource) V41EngramLookAheadReadPolicy(rowBytes int) (int, bool) {
	if s == nil || rowBytes != s.rowBytes {
		return 0, false
	}
	return v41EngramForwardLookAheadPolicy(s.inner, rowBytes)
}

func (s *v41EngramQ2KRowSource) V41EngramLookAheadReadPolicy(rowBytes int) (int, bool) {
	if s == nil || rowBytes != v41EngramQ2KEncodedRowBytes {
		return 0, false
	}
	// EncodedForm only constructs this source from verified, row-aligned F32
	// geometry. The original provider indexes 1024-byte decoded rows even
	// though this source returns their 84-byte Q2 blocks.
	return v41EngramForwardLookAheadPolicy(s.enc, V41EngramF32RowBytes)
}

const (
	v41EngramLookAheadMaxRows       = 256
	v41EngramLookAheadBudget        = 1 << 20
	v41EngramLookAheadMetadata      = 64 << 10
	v41EngramLookAheadCacheMetadata = 64 << 10
)

// One process-wide job, including its ready panel and synchronous consumption.
// The value is a conservative logical allocation charge, NOT an RSS guarantee.
// Map and position-slice cardinalities are each bounded by MaxRows. Their fixed
// allowance does not claim exact accounting of Go's allocator or GC retention.
var v41EngramLookAheadCharge atomic.Int64

// No scheduler calls this seam yet. Its caller owns one transaction and must
// consume synchronously or close the future on every exit. The callback must not
// retain panel slices or call close/the backing source's Close on its own lease.
// cancel is concurrent-safe; close joins both gathering and an active callback.
// Transaction mutation remains serial: the caller must close/join this future
// before tx.close or releasing model/transaction state. This is not a lock for
// tx.closed or the model, and does not authorize concurrent transaction mutation.
type v41EngramLookAhead = interface {
	wait()
	consume(*v41StepTransaction, int, []uint32, func(*v41EngramGatheredRows) error) error
	cancel()
	close()
}

type v41EngramLookAheadJob struct {
	m               *Model
	tx              *v41StepTransaction
	layer           int
	geometry        v41EngramGeometry
	ids             []uint32
	ctx             context.Context
	closing         <-chan struct{}
	release         func()
	charge          int64
	done            chan struct{}
	cancelled       atomic.Bool
	mu              sync.Mutex
	claimed, closed bool
	consumerDone    chan struct{}
	retire          sync.Once
	rows            *v41EngramGatheredRows
	err             error
	panicValue      any
}

func v41EngramLookAheadErr(layer int, detail string) error {
	return v41StageErr(v41StageEngram, layer, fmt.Errorf("%w: Engram look-ahead %s", ErrV41NativeUnsupported, detail))
}

func v41EngramLookAheadCost(cache *V41EngramRowCache, count, staging int) (int64, bool) {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	if count < 1 || count > v41EngramLookAheadMaxRows || staging < 0 || staging > v41EngramLookAheadBudget || cache.opts.PrefetchRows > 0 {
		return 0, false
	}
	width := cache.opts.RowBytes
	if width != v41EngramQ2KEncodedRowBytes && width != V41EngramPackedRowBytes && width != V41EngramF32RowBytes {
		return 0, false
	}
	// This first envelope admits only small, immutable-geometry caches. The
	// ordinary loader uses four rows. A larger pre-existing cache can otherwise
	// resize its maps/order on one insertion, independently of this batch size.
	inner := cache.inner
	if cache.opts.BudgetBytes > int64(v41EngramLookAheadMaxRows*width) || inner == nil || inner.budgetBytes != cache.opts.BudgetBytes ||
		len(inner.entries) > v41EngramLookAheadMaxRows || len(inner.cached) != len(inner.entries) || len(inner.order) != len(inner.entries) || cap(inner.order) > 2*v41EngramLookAheadMaxRows {
		return 0, false
	}
	for _, payload := range inner.entries {
		if len(payload) != width {
			return 0, false
		}
	}
	// Panel + possible new cache payload copies; output slice headers, copied
	// IDs, unique/missing arrays; all source inputs and producer staging across
	// the gather, not only its largest coalesced span. Coalescing joins only
	// consecutive requested misses; with prefetch off it cannot amplify gaps.
	// The metadata allowance covers map entries, position slices, and fixed job
	// state. Count/width/staging caps make every multiplication representable.
	// A separate logical allowance covers growth of the two bounded cache maps
	// and order slice. Existing residency is separately cache-budgeted; transient
	// new payload copies are charged above. Neither allowance is an RSS claim.
	charge := int64(v41EngramLookAheadMetadata+v41EngramLookAheadCacheMetadata) + int64(count)*(3*int64(width)+int64(staging)+int64(3*strconv.IntSize/8+12))
	return charge, charge <= v41EngramLookAheadBudget
}

// Unsupported sources, disabled opt-in and occupied credit remain outside the
// async path (nil, nil). Once an eligible source is selected, a failed hold is a
// lifetime error, never permission to read without retention. Speculative cache
// and I/O effects may precede consumption; they are not synchronous-I/O parity.
func (m *Model) startV41EngramLookAhead(ctx context.Context, tx *v41StepTransaction, enabled bool, layer int, ids []uint32, eps float32) (v41EngramLookAhead, error) {
	if !enabled {
		return nil, nil
	}
	if ctx == nil || tx == nil || tx.closed {
		return nil, v41EngramLookAheadErr(layer, "requires an open transaction and context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	g, err := m.v41EngramInjectionGeometry(layer, 1)
	if err != nil {
		return nil, err
	}
	if g.cacheIdx < 0 || g.cacheIdx >= len(g.stage.caches) {
		return nil, v41EngramLookAheadErr(layer, "cache index is out of bounds")
	}
	cache := g.stage.caches[g.cacheIdx]
	if len(ids) != g.cols {
		return nil, v41EngramLookAheadErr(layer, "row count does not match geometry")
	}
	if len(ids) < 1 || len(ids) > v41EngramLookAheadMaxRows {
		return nil, v41EngramLookAheadErr(layer, "row count exceeds the admitted envelope")
	}
	if err := v41EngramGateParamsErr(layer, eps, g.qNorm, g.kNorm); err != nil {
		return nil, err
	}
	if err := cache.validateGatherGeometry(); err != nil {
		return nil, err
	}
	for _, id := range ids {
		if uint64(id) >= uint64(cache.opts.TableRows) {
			return nil, v41EngramLookAheadErr(layer, "row identifier is out of bounds")
		}
	}
	staging, eligible := v41EngramForwardLookAheadPolicy(cache.src, cache.opts.RowBytes)
	if !eligible {
		return nil, nil
	}
	charge, admitted := v41EngramLookAheadCost(cache, len(ids), staging)
	if !admitted {
		return nil, v41EngramLookAheadErr(layer, "allocation envelope is unsupported")
	}
	if !v41EngramLookAheadCharge.CompareAndSwap(0, charge) {
		return nil, nil
	}
	started := false
	var release func()
	defer func() {
		if !started {
			defer v41EngramLookAheadCharge.Store(0)
			if release != nil {
				release()
			}
		}
	}()
	owner, ok := cache.src.(v41EngramSourceReadOwner)
	if !ok {
		return nil, v41EngramLookAheadErr(layer, "eligible source has no lifetime owner")
	}
	var closing <-chan struct{}
	release, closing, ok = owner.V41EngramHoldSourceReads()
	if !ok || release == nil || closing == nil {
		return nil, v41EngramLookAheadErr(layer, "source retention refused")
	}
	f := &v41EngramLookAheadJob{m: m, tx: tx, layer: layer, geometry: g,
		ids: slices.Clone(ids), ctx: ctx, closing: closing, release: release, charge: charge, done: make(chan struct{})}
	if err := f.stopError(); err != nil {
		return nil, err
	}
	go f.gather()
	started = true
	return f, nil
}

func (f *v41EngramLookAheadJob) stopError() error {
	if err := f.ctx.Err(); err != nil {
		return err
	}
	if f.cancelled.Load() {
		return context.Canceled
	}
	select {
	case <-f.closing:
		return v41EngramLookAheadErr(f.layer, "source is closing")
	default:
		return nil
	}
}

func (f *v41EngramLookAheadJob) gather() {
	returned := false
	defer func() {
		f.panicValue = recover()
		if !returned && f.panicValue == nil {
			f.err = v41EngramLookAheadErr(f.layer, "gather exited without completion")
		}
		close(f.done)
	}()
	if f.err = f.stopError(); f.err == nil {
		// Gate metadata was validated synchronously. Do not revisit model weight
		// slices on the worker: its lease covers the row source, not arbitrary
		// model tensor owners. The actual cache/gather operation stays shared.
		cache := f.geometry.stage.caches[f.geometry.cacheIdx]
		staging, eligible := v41EngramForwardLookAheadPolicy(cache.src, cache.opts.RowBytes)
		charge, admitted := v41EngramLookAheadCost(cache, len(f.ids), staging)
		if !eligible || !admitted || charge != f.charge {
			f.err = v41EngramLookAheadErr(f.layer, "cache or source envelope changed before reading")
			returned = true
			return
		}
		// The immutable options and fixed-width insertion invariant retain the
		// bounded geometry across this revalidation/gather lock handoff.
		rows, encoded, err := cache.gatherKnownRowsOwned(f.ids, true)
		if err != nil {
			f.err = v41StageErr(v41StageEngram, f.layer, fmt.Errorf("%w: %w", ErrV41NativeUnsupported, err))
		} else {
			f.rows = &v41EngramGatheredRows{layer: f.layer, stage: f.geometry.stage, cacheIdx: f.geometry.cacheIdx, rows: rows, dim: f.geometry.dim, encoded: encoded}
		}
	}
	returned = true
}

func (f *v41EngramLookAheadJob) cancel() { f.cancelled.Store(true) }

// wait observes completion only; errors remain pending until consumption and
// the ready panel remains charged. It never releases the owner or credit.
func (f *v41EngramLookAheadJob) wait() { <-f.done }

func (f *v41EngramLookAheadJob) consume(tx *v41StepTransaction, layer int, ids []uint32, apply func(*v41EngramGatheredRows) error) (err error) {
	f.mu.Lock()
	if f.closed || f.claimed {
		f.mu.Unlock()
		return v41EngramLookAheadErr(layer, "future is already claimed or closed")
	}
	f.claimed = true
	f.consumerDone = make(chan struct{})
	consumerDone := f.consumerDone
	f.mu.Unlock()
	defer func() { close(consumerDone); f.releaseJoined() }()
	<-f.done
	if err := f.stopError(); err != nil {
		return err
	}
	if tx != f.tx || tx.closed || layer != f.layer || !slices.Equal(ids, f.ids) || f.m.v41EngramStageFor() != f.geometry.stage || apply == nil {
		return v41EngramLookAheadErr(layer, "consumption does not match the captured transaction, stage and rows")
	}
	if f.panicValue != nil {
		panic(f.panicValue)
	}
	if f.err != nil {
		return f.err
	}
	if err = apply(f.rows); err != nil {
		return err
	}
	return f.stopError()
}

func (f *v41EngramLookAheadJob) close() {
	f.cancel()
	f.mu.Lock()
	f.closed = true
	consumerDone := f.consumerDone
	f.mu.Unlock()
	<-f.done
	if consumerDone != nil {
		<-consumerDone
	}
	f.releaseJoined()
}

func (f *v41EngramLookAheadJob) releaseJoined() {
	f.retire.Do(func() {
		// No cache/source/transaction mutex is held while joining or releasing.
		<-f.done
		f.rows, f.ids = nil, nil
		release := f.release
		f.release, f.m, f.tx = nil, nil, nil
		f.geometry = v41EngramGeometry{}
		defer v41EngramLookAheadCharge.Store(0)
		release()
	})
}
