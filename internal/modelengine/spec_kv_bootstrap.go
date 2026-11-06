package modelengine

import (
	"errors"
	"sync"
)

// Source: sgl-project/sglang PR #38978 "Reduce decode bootstrap latency with
// request-owned speculative KV", python/sglang/srt, revision
// 3c6a6c8068658b88e40bcaa5a0bf48b9a94ebdba (Apache-2.0):
// https://github.com/sgl-project/sglang/pull/38978
//
// Adapted for Fak as idiomatic Go (no upstream code blocks copied): only the
// request-ownership rule is ported. Upstream, a disaggregated-prefill request
// whose decode target has not finished bootstrapping ("pending_bootstrap")
// used to release its computed KV back to the shared pool and re-queue, paying
// re-prefill. The PR instead lets the request own that speculative KV from
// admission, bounds the speculative allocation per request, publishes the
// prefix incrementally while the decode side is still pending, and releases
// through a single abort path without inserting into the prefix cache
// (is_insert=False). On the fak side this type owns just that ownership
// ledger: per-request bounded reservation against a shared paged pool,
// hold-while-bootstrap-waits (pool pressure grants zero instead of forcing a
// release-and-requeue), and exactly-once release on abort or handoff. Wiring
// the ledger into the P/D admission path (native_pd.go) is a follow-on.

var (
	// ErrSpecKVUnknownRequest is the closed contract for operating on a
	// request id that was never admitted, an empty id, or an id already
	// handed off by Complete. Callers match it with errors.Is; the prose is
	// not part of the API.
	ErrSpecKVUnknownRequest = errors.New("modelengine: spec-kv request not admitted")

	// ErrSpecKVAborted is the closed contract for reserving speculative KV
	// after the request's abort path already released everything it owned,
	// or for completing a request that aborted first.
	ErrSpecKVAborted = errors.New("modelengine: spec-kv request aborted")

	// ErrSpecKVDuplicateAdmission is the closed contract for admitting an id
	// that is already registered live or in its terminal aborted state.
	ErrSpecKVDuplicateAdmission = errors.New("modelengine: spec-kv request already admitted")
)

// specKVRequest is one admitted request's ownership record. Aborted requests
// keep their record so the terminal state stays observable; completed ones
// are removed so the identity can be re-admitted.
type specKVRequest struct {
	owned   int
	aborted bool
}

// SpecKVBootstrap is the request-ownership ledger for speculative KV blocks
// held while a request waits on decode bootstrap. Blocks are drawn from one
// shared paged pool of PoolBlocks() size; each live request may own at most
// PerRequestCap() blocks until it terminates via Abort or Complete. The
// invariant Free() + sum(Owned over live requests) == PoolBlocks() holds
// after every operation.
type SpecKVBootstrap struct {
	mu            sync.Mutex
	poolBlocks    int
	free          int
	perRequestCap int
	reqs          map[string]*specKVRequest
}

// NewSpecKVBootstrap creates the ledger over poolBlocks shared blocks with a
// per-request speculative bound of perRequestCap. Both must be positive.
func NewSpecKVBootstrap(poolBlocks, perRequestCap int) (*SpecKVBootstrap, error) {
	if poolBlocks <= 0 || perRequestCap <= 0 {
		return nil, errors.New("modelengine: spec-kv pool blocks and per-request bound must be positive")
	}
	return &SpecKVBootstrap{
		poolBlocks:    poolBlocks,
		free:          poolBlocks,
		perRequestCap: perRequestCap,
		reqs:          make(map[string]*specKVRequest),
	}, nil
}

// PoolBlocks reports the shared pool size.
func (b *SpecKVBootstrap) PoolBlocks() int { return b.poolBlocks }

// PerRequestCap reports the per-request speculative bound.
func (b *SpecKVBootstrap) PerRequestCap() int { return b.perRequestCap }

// Free reports the blocks no request currently owns.
func (b *SpecKVBootstrap) Free() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.free
}

// Admit registers a request at admission time, before it reserves any
// speculative KV. An empty id is unusable and reports
// ErrSpecKVUnknownRequest.
func (b *SpecKVBootstrap) Admit(id string) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if id == "" {
		return ErrSpecKVUnknownRequest
	}
	if _, exists := b.reqs[id]; exists {
		return ErrSpecKVDuplicateAdmission
	}
	b.reqs[id] = &specKVRequest{}
	return nil
}

// Owned reports the blocks id currently holds.
func (b *SpecKVBootstrap) Owned(id string) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	r, err := b.lookupLocked(id)
	if err != nil {
		return 0, err
	}
	return r.owned, nil
}

// Reserve draws up to want blocks for id, bounded by the per-request cap and
// the pool's free blocks. Under pool pressure it grants less — possibly zero
// — instead of failing: the request keeps what it owns and waits, which is
// the point of request ownership (no release-and-requeue churn). A negative
// want is treated as zero.
func (b *SpecKVBootstrap) Reserve(id string, want int) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	r, err := b.lookupLocked(id)
	if err != nil {
		return 0, err
	}
	if r.aborted {
		return 0, ErrSpecKVAborted
	}
	if want < 0 {
		want = 0
	}
	if headroom := b.perRequestCap - r.owned; want > headroom {
		want = headroom
	}
	if want > b.free {
		want = b.free
	}
	r.owned += want
	b.free -= want
	return want, nil
}

// Abort releases every block id still owns, exactly once, without publishing
// them into any shared prefix cache (upstream is_insert=False). A second
// abort finds nothing left and reports zero. The id stays registered in its
// terminal aborted state: Reserve and Complete report ErrSpecKVAborted and
// re-admission reports ErrSpecKVDuplicateAdmission.
func (b *SpecKVBootstrap) Abort(id string) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	r, err := b.lookupLocked(id)
	if err != nil {
		return 0, err
	}
	if r.aborted {
		return 0, nil
	}
	released := r.owned
	r.owned = 0
	r.aborted = true
	b.free += released
	return released, nil
}

// Complete hands off every block id still owns to the decode side and removes
// the request from the ledger; the id may be re-admitted afterwards.
// Completing an aborted request reports ErrSpecKVAborted.
func (b *SpecKVBootstrap) Complete(id string) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	r, err := b.lookupLocked(id)
	if err != nil {
		return 0, err
	}
	if r.aborted {
		return 0, ErrSpecKVAborted
	}
	released := r.owned
	delete(b.reqs, id)
	b.free += released
	return released, nil
}

// lookupLocked resolves a live-or-aborted record. Callers hold b.mu.
func (b *SpecKVBootstrap) lookupLocked(id string) (*specKVRequest, error) {
	if id == "" {
		return nil, ErrSpecKVUnknownRequest
	}
	r, ok := b.reqs[id]
	if !ok {
		return nil, ErrSpecKVUnknownRequest
	}
	return r, nil
}
