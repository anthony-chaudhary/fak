// Package idempotency gives retryable mutating tool ops a keyed, time-windowed
// dedup so a retried call after a hang/timeout is a safe no-op that returns the
// ORIGINAL result instead of double-applying (#2093, part of epic #2063).
//
// The problem: after a tool hang an agent often retries, but a non-idempotent op
// (create a GitHub issue, push, append to a ledger) can double-apply if the first
// attempt actually landed before the hang signal reached the caller. The caller
// cannot tell, so it either risks a duplicate or wastes a turn checking.
//
// The fix: derive an idempotency key from the op label + a caller-supplied token
// and record each applied op in an append-only JSONL ledger. A repeated key seen
// within the dedup window replays the recorded result WITHOUT re-running the op;
// a fresh key (a genuinely new op) proceeds and is recorded. The ledger is on
// disk, so a post-hang retry that arrives in a FRESH PROCESS still dedupes.
//
// This is the pure core (window logic + ledger). The thin CLI shell that turns it
// into an executor a real op can ride is cmd/fak/idempotency.go (`fak
// idempotency`). It is off the hot path; the kernel-request-path wiring
// (abi.ToolCall.Meta["idempotency_key"] consulted in the executor) is a separate,
// larger change tracked as follow-on work.
//
// Concurrency: two CONCURRENT same-key applies across separate processes are
// serialized, not just the sequential post-hang retry. This matters wherever a
// comms channel fans one logical op out to more than one worker — the dgx-bridge
// double-dispatch race (two workers pick up the same nonce and both run the
// expensive op because neither has recorded yet), and the same readback lesson
// internal/slackoutbox learned as "concurrent drainers lose the tail". Do holds a
// cross-process advisory lock (internal/flock) on <ledger>.lock across its whole
// check-and-apply, and RE-READS the ledger under that lock before deciding: the
// in-memory view is a snapshot from Open, so without the reload the loser of the
// race would still miss and double-apply. The lock is per-LEDGER (coarse), so
// distinct keys on one ledger serialize too — fine for an off-hot-path store;
// per-key locking is a deliberate non-goal. Contention waits up to Store.LockWait
// and then returns ErrBusy (retryable) rather than risking a double-apply.
package idempotency

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// DefaultWindow is the dedup horizon: a key applied within this span replays; an
// older record is treated as expired so a reused token eventually proceeds again.
// A post-hang retry arrives seconds-to-minutes later, so a generous default still
// catches it while never pinning a token forever.
const DefaultWindow = 24 * time.Hour

// DefaultLockWait is how long Do polls for the per-ledger cross-process lock before
// giving up with ErrBusy. It is generous because the winner of a race holds the lock
// across its whole apply() — a slow op (e.g. a remote witness run) legitimately makes
// a concurrent caller wait that long before it replays the recorded result. A caller
// wrapping an even longer apply raises it via Store.LockWait.
const DefaultLockWait = 2 * time.Minute

// ErrBusy is returned by Do when the per-ledger cross-process lock could not be
// acquired within Store.LockWait. It is a distinct, retryable condition (test with
// errors.Is) — the op did NOT apply and nothing was recorded, so the caller may retry
// rather than risk a double-apply by proceeding unlocked.
var ErrBusy = errors.New("idempotency: another process holds the ledger lock")

// lockPoll is the gap between flock.TryLock attempts while waiting for the ledger
// lock. flock is non-blocking by design (see internal/flock), so a blocking
// acquire IS this poll — the same idiom loopmgr.withLedgerLock uses.
const lockPoll = 25 * time.Millisecond

// Record is one applied op: the derived key, the op label, the captured original
// result, and when it was applied (unix nanoseconds). One JSON object per ledger
// line.
type Record struct {
	Key       string `json:"key"`
	Op        string `json:"op"`
	Result    string `json:"result"`
	AppliedAt int64  `json:"applied_at"`
}

// Key derives a stable idempotency key from an op label and a caller-supplied
// token. The same (op, token) always yields the same key, so a retried op that
// reuses its token dedupes; a different token (or op) yields a fresh key and
// proceeds. The NUL separator keeps ("ab","c") distinct from ("a","bc").
func Key(op, token string) string {
	sum := sha256.Sum256([]byte(op + "\x00" + token))
	return hex.EncodeToString(sum[:])
}

// Store is a keyed, time-windowed dedup backed by an append-only JSONL ledger so
// dedup survives across process restarts — a post-hang retry is usually a fresh
// process. The zero value is not usable; construct with Open (or OpenClock).
type Store struct {
	path   string
	window time.Duration
	now    func() time.Time

	// LockWait bounds how long Do waits for the cross-process ledger lock before
	// returning ErrBusy. Zero uses DefaultLockWait. Set it after Open when the op
	// being wrapped is slower than that default: the race winner holds the lock for
	// the whole duration of its apply, so a loser must be willing to wait that long
	// to be able to replay instead of double-applying.
	LockWait time.Duration

	mu   sync.Mutex
	recs map[string]Record // key -> latest applied record
}

// Open loads (or begins) the ledger at path with the given dedup window. A
// missing ledger is an empty history, not an error. A window <= 0 uses
// DefaultWindow.
func Open(path string, window time.Duration) (*Store, error) {
	return OpenClock(path, window, time.Now)
}

// OpenClock is Open with an injectable clock, for deterministic window tests.
func OpenClock(path string, window time.Duration, now func() time.Time) (*Store, error) {
	if window <= 0 {
		window = DefaultWindow
	}
	if now == nil {
		now = time.Now
	}
	s := &Store{path: path, window: window, now: now, recs: map[string]Record{}}
	if err := s.loadLocked(); err != nil {
		return nil, err
	}
	return s, nil
}

// loadLocked folds the on-disk ledger into recs. It MERGES rather than resets, so
// a caller that already holds records keeps them and only gains what landed since;
// keeping the latest record per key means a re-applied (expired-then-reused) key
// reflects its most recent apply, not the first. A missing ledger is an empty
// history, not an error.
//
// Open calls it to seed the snapshot. Do calls it again under the cross-process
// lock — that second read is what lets the loser of a race see the winner's
// just-appended record instead of missing and double-applying. Callers hold s.mu.
func (s *Store) loadLocked() error {
	b, err := os.ReadFile(s.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("idempotency: read ledger %s: %w", s.path, err)
	}
	for _, r := range jsonlledger.Parse[Record](string(b), nil) {
		if r.Key == "" {
			continue
		}
		if prev, ok := s.recs[r.Key]; !ok || r.AppliedAt >= prev.AppliedAt {
			s.recs[r.Key] = r
		}
	}
	return nil
}

// withLock runs fn while holding an exclusive cross-process advisory lock on
// <ledger>.lock, so concurrent Do calls in separate processes serialize instead of
// both applying. flock.TryLock is non-blocking, so this polls until the lock frees
// or LockWait elapses (then ErrBusy) — the loopmgr.withLedgerLock idiom. The lock
// fd is closed on return, which releases the OS lock (and the OS releases it if
// this process dies mid-apply, so a crash cannot wedge the ledger forever).
//
// The poll deadline is real wall-clock time, NOT s.now: the injected clock exists
// to make window expiry deterministic, and a fixed test clock would never pass a
// deadline derived from it. fn's error is returned verbatim so an apply error
// reaches the caller unwrapped.
func (s *Store) withLock(fn func() error) error {
	wait := s.LockWait
	if wait <= 0 {
		wait = DefaultLockWait
	}
	if dir := filepath.Dir(s.path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("idempotency: mkdir %s: %w", dir, err)
		}
	}
	lockPath := s.path + ".lock"
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		return fmt.Errorf("idempotency: open lock %s: %w", lockPath, err)
	}
	defer f.Close()

	deadline := time.Now().Add(wait)
	for {
		lerr := flock.TryLock(f)
		if lerr == nil {
			break
		}
		if !errors.Is(lerr, flock.ErrLockBusy) {
			return fmt.Errorf("idempotency: lock %s: %w", lockPath, lerr)
		}
		if time.Now().After(deadline) {
			return ErrBusy
		}
		time.Sleep(lockPoll)
	}
	defer func() { _ = flock.Unlock(f) }()
	return fn()
}

// Lookup returns the recorded result for key if it was applied within the window,
// else (zero, false). An expired record reports a miss so the op proceeds again.
func (s *Store) Lookup(key string) (Record, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lookupLocked(key)
}

func (s *Store) lookupLocked(key string) (Record, bool) {
	r, ok := s.recs[key]
	if !ok {
		return Record{}, false
	}
	if s.now().Sub(time.Unix(0, r.AppliedAt)) > s.window {
		return Record{}, false
	}
	return r, true
}

// Do runs apply for a fresh or expired key and records its result, or replays the
// recorded result WITHOUT calling apply for a key already applied within the
// window (replayed=true). This is the executor-dedup a retried op rides: the first
// attempt applies and records; a post-hang retry with the same key is a safe
// no-op that returns the original result, so a non-idempotent op never
// double-applies.
//
// An apply error is returned and nothing is recorded, so a failed op stays
// retryable (a failure is not a landed effect to dedupe).
//
// The whole check-and-apply runs under a cross-process lock on <ledger>.lock, and
// re-reads the ledger once inside it, so two CONCURRENT same-key applies in
// separate processes collapse to one: the winner applies and records, the loser
// blocks, then sees the fresh record and replays it. Without the reload the loser
// would consult its Open-time snapshot, miss, and double-apply. If the lock cannot
// be taken within LockWait, Do returns ErrBusy having applied nothing.
func (s *Store) Do(key, op string, apply func() (string, error)) (result string, replayed bool, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var res string
	var rep bool
	if err := s.withLock(func() error {
		// Re-read under the lock: a concurrent process may have recorded this key
		// since Open, and that record is the one that makes this a replay.
		if err := s.loadLocked(); err != nil {
			return err
		}
		if r, ok := s.lookupLocked(key); ok {
			res, rep = r.Result, true
			return nil
		}
		out, err := apply()
		if err != nil {
			return err
		}
		rec := Record{Key: key, Op: op, Result: out, AppliedAt: s.now().UnixNano()}
		if err := s.appendLocked(rec); err != nil {
			return err
		}
		s.recs[key] = rec
		res, rep = out, false
		return nil
	}); err != nil {
		return "", false, err
	}
	return res, rep, nil
}

func (s *Store) appendLocked(rec Record) error {
	line, err := json.Marshal(rec)
	if err != nil {
		return fmt.Errorf("idempotency: encode record: %w", err)
	}
	if dir := filepath.Dir(s.path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("idempotency: mkdir %s: %w", dir, err)
		}
	}
	f, err := os.OpenFile(s.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("idempotency: open ledger %s: %w", s.path, err)
	}
	defer f.Close()
	if _, err := f.Write(append(line, '\n')); err != nil {
		return fmt.Errorf("idempotency: append ledger %s: %w", s.path, err)
	}
	return nil
}
