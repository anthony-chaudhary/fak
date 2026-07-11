package ctxmmu

// pinreaper.go — the TIME dimension of the held-quarantine ledger bound (issue #3385).
//
// evictExcessLocked (mmu.go) bounds the ledger by COUNT: once len(held) exceeds
// maxHeld the OLDEST pin is force-unpinned. That frees a leaked pin only when
// newer quarantines push it out — a crashed holder on a QUIET gate keeps its CAS
// pin forever. This file adds the missing time bound: every held pin carries a
// last-touch keepalive stamp (set on hold, RESET on repin/Clear/PageIn), and
// ReapExpiredPins force-unpins any pin idle past a TTL, counting it in
// forcedUnpins — a metric distinct from the count-cap evicted counter, so a
// time-leak is tellable from cap pressure. Clean-room technique borrow from
// LMCache's pin monitor (lmcache/v1/pin_monitor.py:91-107 @ aaf7c0d3, Apache-2.0):
// TTL expiry + reset-on-repin, original Go.
//
// Deterministic by construction (the repo's injected-now reaper pattern —
// cachemeta's nowMillis folds, leaseref's Reap(ctx, now)): the reap DECISION
// never reads a wall clock — the caller injects nowUnixMillis, and last-touch is
// stored as the millis handed to touchLocked. There is NO background goroutine;
// callers drive the sweep, exactly like leaseref's crashed-holder reaper.

import (
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// holdNowMillis is the stamp the PRODUCTION hold/keepalive sites inject into
// touchLocked (quarantineResult, digestToPointer, Clear, PageIn). The wall clock
// is read once here, at the stamping boundary; the reap decision itself never
// calls it — ReapExpiredPins compares stamps only against its injected now.
func holdNowMillis() int64 { return time.Now().UnixMilli() }

// touchLocked stamps (or re-stamps) id's keepalive to nowUnixMillis. The caller
// holds m.mu and has ensured id is (being) held. The map is lazily initialized so
// a zero-value MMU stays usable; lastTouch stays ⊆ held because every removal
// path (evictExcessLocked, ReapExpiredPins) deletes the stamp with the entry.
func (m *MMU) touchLocked(id string, nowUnixMillis int64) {
	if m.lastTouch == nil {
		m.lastTouch = map[string]int64{}
	}
	m.lastTouch[id] = nowUnixMillis
}

// TouchPin is the re-pin (keepalive) entry: it resets id's TTL countdown to run
// from nowUnixMillis, so an actively-used pin is never reaped (reset-on-repin).
// It reports whether id is currently held; touching an unheld or already-reaped
// id is a no-op false — a touch never resurrects an entry. nowUnixMillis is
// injected unix millis, so deterministic tests drive the clock themselves.
func (m *MMU) TouchPin(id string, nowUnixMillis int64) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.held[id]; !ok {
		return false
	}
	m.touchLocked(id, nowUnixMillis)
	return true
}

// ReapExpiredPins is the leaked-pin reaper: it force-unpins every held pin whose
// keepalive is idle past the TTL — (nowUnixMillis - lastTouch) > ttlMillis — the
// holder that never Cleared/PageIn'd/re-pinned, or died. Each expired entry has
// its CAS pin released (abi.UnpinResolved, the same release evictExcessLocked
// performs), is dropped from held/cleared/lastTouch together (cleared stays ⊆
// held), and is counted in forcedUnpins — DISTINCT from the count-cap evicted
// counter. Returns how many pins were reaped. Like an eviction, a reaped id's
// later PageIn refuses like an unknown id — fail-closed, never a bytes leak.
//
// Semantics:
//   - ttlMillis <= 0 disables the sweep (returns 0): a misconfigured TTL must
//     fail safe, never mass-unpin the ledger.
//   - a held id with NO stamp (a zero-value MMU literal that bypassed both the
//     constructors and the hold path) is ADOPTED at nowUnixMillis instead of
//     reaped — it earns a full TTL from first observation, so the reaper never
//     frees a pin without a witnessed idle interval.
//   - a reaped id keeps its (now stale) slot in the FIFO order slice;
//     evictExcessLocked already skips ids absent from held, so the count bound
//     and the time bound compose without double-unpinning.
//   - idempotent at a fixed now: a second sweep with the same arguments reaps
//     nothing, because the expired entries are already gone.
//
// NO background goroutine — callers drive the reap, and the decision never reads
// a wall clock (nowUnixMillis is injected; mirror of leaseref.Store.Reap).
func (m *MMU) ReapExpiredPins(nowUnixMillis int64, ttlMillis int64) (reaped int) {
	if ttlMillis <= 0 {
		return 0
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for id, h := range m.held {
		stamp, ok := m.lastTouch[id]
		if !ok {
			m.touchLocked(id, nowUnixMillis) // adopt: full TTL from first observation
			continue
		}
		if nowUnixMillis-stamp <= ttlMillis {
			continue // within TTL — the keepalive is holding it live
		}
		abi.UnpinResolved(h) // release the CAS pin taken in quarantineResult/digestToPointer
		delete(m.held, id)
		delete(m.cleared, id)
		delete(m.lastTouch, id)
		atomic.AddInt64(&m.forcedUnpins, 1)
		reaped++
	}
	return reaped
}

// ForcedUnpins reports the lifetime count of held pins force-unpinned by the TTL
// reaper — the leaked-pin metric, distinct from Evicted() (the count-cap bound):
// ForcedUnpins climbing on a quiet gate means holders are leaking pins by TIME,
// not that the ledger is under cap pressure.
func (m *MMU) ForcedUnpins() int64 { return atomic.LoadInt64(&m.forcedUnpins) }
