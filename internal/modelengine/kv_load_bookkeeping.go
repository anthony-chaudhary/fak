package modelengine

import (
	"errors"
	"fmt"
	"sync"
)

// Adapted from vLLM PR #65831 "[Perf][Core] Reduce per-block CPU cost of KV
// block allocation and caching" (vllm-project/vllm, merged at revision
// 518c624a1c731cfc6c73369ced6351cb03754ddf, Apache-2.0; the pinned sources are
// the v1 core block pool and KV cache manager modules): host-side bookkeeping
// for an async KV load that starts, is promoted to running, and finishes used
// to grow linearly with prompt length because each scheduler step re-derived
// pool-wide aggregates by re-walking the whole pool; a slow schedule() on one
// rank then held up the per-step all-reduce in DP deployments. The fix keeps
// cached aggregates and applies batched O(1) delta updates so N load
// transitions cost O(N) total pool touches. Re-expressed in idiomatic Go; no
// upstream code copied. Scheduler integration point: nativesched.go.

// kvLoadPhase is the lifecycle phase of one request's async KV load.
type kvLoadPhase int

const (
	// kvLoadPhaseLoading marks a started load whose blocks are reserved but
	// not yet part of the running batch.
	kvLoadPhaseLoading kvLoadPhase = iota
	// kvLoadPhaseRunning marks a load promoted into the running batch.
	kvLoadPhaseRunning
	// kvLoadPhaseFinished marks a completed load whose blocks stay held until
	// release.
	kvLoadPhaseFinished
)

var (
	// ErrKVLoadInvalidPoolSize is returned when the pool size is non-positive.
	ErrKVLoadInvalidPoolSize = errors.New("modelengine: kv load pool size must be positive")
	// ErrKVLoadInvalidBlocks is returned when a load requests a non-positive block count.
	ErrKVLoadInvalidBlocks = errors.New("modelengine: kv load block count must be positive")
	// ErrKVLoadNotFound is returned when an operation references a request with no active load state.
	ErrKVLoadNotFound = errors.New("modelengine: kv load request not found")
	// ErrKVLoadAlreadyActive is returned when starting a load for a request that already has one.
	ErrKVLoadAlreadyActive = errors.New("modelengine: kv load already active for request")
	// ErrKVLoadWrongPhase is returned when a transition is requested from an unexpected phase.
	ErrKVLoadWrongPhase = errors.New("modelengine: kv load transition from unexpected phase")
	// ErrKVLoadCapacityExceeded is returned when a start would exceed the pool's free slots.
	ErrKVLoadCapacityExceeded = errors.New("modelengine: kv load capacity exceeded")
)

// KVLoadAggregates is the cached, schedule-time view of all tracked KV loads.
// Every field is a running sum maintained incrementally by the bookkeeper, so
// reading it performs zero individual pool-slot accesses.
type KVLoadAggregates struct {
	LoadingRequests  int
	RunningRequests  int
	FinishedRequests int
	LoadingBlocks    int
	RunningBlocks    int
	FinishedBlocks   int
	FreeBlocks       int
}

// kvLoadRequest is the per-request load state kept in amortized-constant
// structures: one map entry plus the slice of owned pool slots.
type kvLoadRequest struct {
	id     string
	phase  kvLoadPhase
	blocks []int
}

// KVLoadBookkeeper keeps per-request async KV-load state (start, promote to
// running, finish, release) in amortized-constant structures with cached
// aggregate updates. A pool touch is one individual pool-slot access:
// allocating a slot from the free set, returning one to it, or visiting a
// slot during the consistency audit. StartLoad and ReleaseLoad perform one
// touch per slot they move; PromoteLoad, FinishLoad, Aggregates, and
// PoolTouches perform none. The contract pinned by the tests: N load
// transitions cost O(N) total pool touches, independent of pool size, and
// per-step aggregate reads never walk the pool.
type KVLoadBookkeeper struct {
	mu sync.Mutex

	poolSize int
	free     []int // LIFO free pool slots
	reqs     map[string]*kvLoadRequest

	// Cached running sums (the borrowed "batched aggregate updates"): each
	// transition applies its delta once; schedule-time reads take these as-is.
	loadingRequests  int
	runningRequests  int
	finishedRequests int
	loadingBlocks    int
	runningBlocks    int
	finishedBlocks   int

	// poolTouches counts individual pool-slot accesses since construction.
	poolTouches int64
}

// NewKVLoadBookkeeper creates a bookkeeper over a virtual pool of poolSize
// KV-cache blocks.
func NewKVLoadBookkeeper(poolSize int) (*KVLoadBookkeeper, error) {
	if poolSize <= 0 {
		return nil, ErrKVLoadInvalidPoolSize
	}
	free := make([]int, 0, poolSize)
	for i := poolSize - 1; i >= 0; i-- {
		free = append(free, i)
	}
	return &KVLoadBookkeeper{
		poolSize: poolSize,
		free:     free,
		reqs:     make(map[string]*kvLoadRequest),
	}, nil
}

// StartLoad begins an async KV load for reqID spanning blocks pool slots. It
// performs exactly blocks pool touches and updates the cached aggregates in
// O(1).
func (b *KVLoadBookkeeper) StartLoad(reqID string, blocks int) error {
	if blocks <= 0 {
		return ErrKVLoadInvalidBlocks
	}
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.reqs[reqID]; exists {
		return ErrKVLoadAlreadyActive
	}
	if blocks > len(b.free) {
		return ErrKVLoadCapacityExceeded
	}
	owned := make([]int, 0, blocks)
	for i := 0; i < blocks; i++ {
		last := len(b.free) - 1
		owned = append(owned, b.free[last])
		b.free = b.free[:last]
		b.poolTouches++
	}
	b.reqs[reqID] = &kvLoadRequest{id: reqID, phase: kvLoadPhaseLoading, blocks: owned}
	b.loadingRequests++
	b.loadingBlocks += blocks
	return nil
}

// PromoteLoad moves a started load into the running phase. O(1); zero pool
// touches.
func (b *KVLoadBookkeeper) PromoteLoad(reqID string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.transitionLocked(reqID, kvLoadPhaseLoading, kvLoadPhaseRunning)
}

// FinishLoad marks a running load as complete. O(1); zero pool touches.
func (b *KVLoadBookkeeper) FinishLoad(reqID string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.transitionLocked(reqID, kvLoadPhaseRunning, kvLoadPhaseFinished)
}

// transitionLocked moves reqID from wantPhase to nextPhase, applying the
// aggregate deltas once. Caller holds b.mu.
func (b *KVLoadBookkeeper) transitionLocked(reqID string, wantPhase, nextPhase kvLoadPhase) error {
	req, exists := b.reqs[reqID]
	if !exists {
		return ErrKVLoadNotFound
	}
	if req.phase != wantPhase {
		return ErrKVLoadWrongPhase
	}
	b.movePhaseLocked(wantPhase, -1, -len(req.blocks))
	req.phase = nextPhase
	b.movePhaseLocked(nextPhase, 1, len(req.blocks))
	return nil
}

// movePhaseLocked applies a signed request/block delta to the running sums of
// phase. Caller holds b.mu.
func (b *KVLoadBookkeeper) movePhaseLocked(phase kvLoadPhase, signedRequests, signedBlocks int) {
	switch phase {
	case kvLoadPhaseLoading:
		b.loadingRequests += signedRequests
		b.loadingBlocks += signedBlocks
	case kvLoadPhaseRunning:
		b.runningRequests += signedRequests
		b.runningBlocks += signedBlocks
	case kvLoadPhaseFinished:
		b.finishedRequests += signedRequests
		b.finishedBlocks += signedBlocks
	}
}

// ReleaseLoad frees a request's load state and returns its slots to the pool.
// It performs one pool touch per returned slot.
func (b *KVLoadBookkeeper) ReleaseLoad(reqID string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	req, exists := b.reqs[reqID]
	if !exists {
		return ErrKVLoadNotFound
	}
	b.movePhaseLocked(req.phase, -1, -len(req.blocks))
	b.free = append(b.free, req.blocks...)
	b.poolTouches += int64(len(req.blocks))
	delete(b.reqs, reqID)
	return nil
}

// Aggregates returns the cached schedule-time view. Zero pool touches.
func (b *KVLoadBookkeeper) Aggregates() KVLoadAggregates {
	b.mu.Lock()
	defer b.mu.Unlock()
	return KVLoadAggregates{
		LoadingRequests:  b.loadingRequests,
		RunningRequests:  b.runningRequests,
		FinishedRequests: b.finishedRequests,
		LoadingBlocks:    b.loadingBlocks,
		RunningBlocks:    b.runningBlocks,
		FinishedBlocks:   b.finishedBlocks,
		FreeBlocks:       len(b.free),
	}
}

// PoolTouches reports the total individual pool-slot accesses since
// construction.
func (b *KVLoadBookkeeper) PoolTouches() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.poolTouches
}

// VerifyConsistency audits the bookkeeping against the pool: it visits every
// pool slot (poolSize touches) and verifies that free plus owned equals the
// pool size, that no slot is owned by two requests, and that the cached
// aggregates equal a recomputation from per-request state. This is the
// deliberately slow path; the contract is that schedule() never needs it.
func (b *KVLoadBookkeeper) VerifyConsistency() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	owner := make(map[int]string, len(b.reqs))
	var owned int
	for reqID, req := range b.reqs {
		for _, slot := range req.blocks {
			if prev, dup := owner[slot]; dup {
				return fmt.Errorf("modelengine: kv load slot %d owned by both %q and %q", slot, prev, reqID)
			}
			owner[slot] = reqID
			owned++
		}
	}
	if owned+len(b.free) != b.poolSize {
		return fmt.Errorf("modelengine: kv load accounting mismatch: owned (%d) + free (%d) != pool (%d)",
			owned, len(b.free), b.poolSize)
	}
	var loading, running, finished int
	for _, req := range b.reqs {
		switch req.phase {
		case kvLoadPhaseLoading:
			loading += len(req.blocks)
		case kvLoadPhaseRunning:
			running += len(req.blocks)
		case kvLoadPhaseFinished:
			finished += len(req.blocks)
		}
	}
	if loading != b.loadingBlocks || running != b.runningBlocks || finished != b.finishedBlocks {
		return fmt.Errorf("modelengine: kv load cached aggregates stale: got (loading %d, running %d, finished %d), want (loading %d, running %d, finished %d)",
			b.loadingBlocks, b.runningBlocks, b.finishedBlocks, loading, running, finished)
	}
	b.poolTouches += int64(b.poolSize)
	return nil
}
