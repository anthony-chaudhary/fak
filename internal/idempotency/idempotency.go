// Package idempotency gives mutating tool ops a durable keyed state machine that
// replays proven results and blocks ambiguous outcomes until read-back resolves
// whether the effect landed (#2093, #8284, part of epic #2063).
//
// The problem: after a tool hang an agent often retries, but a non-idempotent op
// (create a GitHub issue, push, append to a ledger) can double-apply if the first
// attempt actually landed before the hang signal reached the caller. The caller
// cannot tell, so it either risks a duplicate or wastes a turn checking.
//
// The fix: derive an idempotency key from the op label + a caller-supplied token,
// fsync a PENDING intent before apply, and append the outcome to a JSONL ledger.
// A response-loss error becomes UNKNOWN_APPLIED and never expires into a retry;
// operation-specific read-back must prove APPLIED or ABSENT first. The ledger is
// on disk, so this fail-closed behavior survives a fresh process.
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

// DefaultWindow is the dedup horizon for proven APPLIED results. Ambiguous PENDING
// and UNKNOWN_APPLIED records never expire into automatic reapplication.
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

// ErrUnknownApplied means apply may have landed and the same key is blocked until
// Resolve proves the effect applied or absent.
var ErrUnknownApplied = errors.New("idempotency: outcome UNKNOWN_APPLIED; resolution required")

// ErrNotFound means no durable intent or outcome exists for a key.
var ErrNotFound = errors.New("idempotency: key not found")

// ErrIdentityConflict is returned when a retry token is reused with a different request identity.
var ErrIdentityConflict = errors.New("idempotency: retry token bound to different request identity")

// ErrLegacyBindingRequired is returned when an unbound legacy record exists and cannot be reused in bound mode.
var ErrLegacyBindingRequired = errors.New("idempotency: legacy unbound record cannot be reused in bound mode")

// lockPoll is the gap between flock.TryLock attempts while waiting for the ledger
// lock. flock is non-blocking by design (see internal/flock), so a blocking
// acquire IS this poll — the same idiom loopmgr.withLedgerLock uses.
const lockPoll = 25 * time.Millisecond

// State is one durable stage in a key's append-only state machine.
type State string

const (
	StatePending        State = "PENDING"
	StateUnknownApplied State = "UNKNOWN_APPLIED"
	StateApplied        State = "APPLIED"
	StateProvenAbsent   State = "PROVEN_ABSENT"
)

// Resolution is an operation-specific read-back verdict.
type Resolution string

const (
	ResolutionApplied Resolution = "APPLIED"
	ResolutionAbsent  Resolution = "ABSENT"
	ResolutionUnknown Resolution = "UNKNOWN"
)

// Readback resolves an ambiguous intent using operation-specific state. APPLIED
// carries the original result to replay; ABSENT permits one new apply; UNKNOWN
// keeps the key blocked.
type Readback func(Record) (Resolution, string, error)

// RequestIdentity binds an idempotency key to immutable semantic request properties.
type RequestIdentity struct {
	Operation     string `json:"operation,omitempty"`
	Resource      string `json:"resource,omitempty"`
	Principal     string `json:"principal,omitempty"`
	PayloadDigest string `json:"payload_digest,omitempty"`
}

// IsZero reports whether all request identity fields are empty.
func (r RequestIdentity) IsZero() bool {
	return r.Operation == "" && r.Resource == "" && r.Principal == "" && r.PayloadDigest == ""
}

// Digest returns a deterministic SHA-256 hex digest of the request identity fields.
func (r RequestIdentity) Digest() string {
	sum := sha256.Sum256([]byte(r.Operation + "\x00" + r.Resource + "\x00" + r.Principal + "\x00" + r.PayloadDigest))
	return hex.EncodeToString(sum[:])
}

// Record is one state transition. AppliedAt is retained for compatibility with
// success-only ledgers written before states existed; a missing State on read is
// interpreted as APPLIED.
type Record struct {
	Key       string           `json:"key"`
	Op        string           `json:"op"`
	State     State            `json:"state,omitempty"`
	Result    string           `json:"result,omitempty"`
	Identity  *RequestIdentity `json:"identity,omitempty"`
	AppliedAt int64            `json:"applied_at,omitempty"`
	UpdatedAt int64            `json:"updated_at,omitempty"`
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
	recs map[string]Record // key -> latest durable state

	// appendRecord is a package-test fault seam for post-apply ledger failures.
	// Production stores leave it nil and use appendLocked.
	appendRecord func(Record) error
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
		if r.State == "" {
			r.State = StateApplied
		}
		if r.UpdatedAt == 0 {
			r.UpdatedAt = r.AppliedAt
		}
		if prev, ok := s.recs[r.Key]; !ok || recordTime(r) >= recordTime(prev) {
			s.recs[r.Key] = r
		}
	}
	return nil
}

func recordTime(r Record) int64 {
	if r.UpdatedAt != 0 {
		return r.UpdatedAt
	}
	return r.AppliedAt
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

// Lookup returns a proven result if key is APPLIED within the dedup window.
func (s *Store) Lookup(key string) (Record, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lookupLocked(key)
}

func (s *Store) lookupLocked(key string) (Record, bool) {
	r, ok := s.recs[key]
	if !ok || r.State != StateApplied {
		return Record{}, false
	}
	if s.now().Sub(time.Unix(0, r.AppliedAt)) > s.window {
		return Record{}, false
	}
	return r, true
}

// Status reloads the ledger under the cross-process lock and returns the latest
// durable state. Unlike Lookup, status does not hide expired or ambiguous rows.
func (s *Store) Status(key string) (Record, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var rec Record
	var ok bool
	err := s.withLock(func() error {
		if err := s.loadLocked(); err != nil {
			return err
		}
		rec, ok = s.recs[key]
		return nil
	})
	return rec, ok, err
}

// DoBound executes apply under the cross-process lock, binding key to ident.
// If an existing record exists for key:
//   - if it lacks an identity (or has a zero identity), it returns ErrLegacyBindingRequired;
//   - if its identity does not match ident, it returns ErrIdentityConflict without calling apply();
//   - if its identity matches, it replays a fresh APPLIED result, blocks on ambiguous states, or allows PROVEN_ABSENT.
func (s *Store) DoBound(key string, ident RequestIdentity, apply func() (string, error)) (result string, replayed bool, err error) {
	return s.do(key, ident.Operation, ident, apply)
}

// DoWithIdentity executes apply with an explicit op label and request identity.
func (s *Store) DoWithIdentity(key, op string, ident RequestIdentity, apply func() (string, error)) (result string, replayed bool, err error) {
	if op == "" {
		op = ident.Operation
	}
	return s.do(key, op, ident, apply)
}

// Do fsyncs PENDING before apply. A proven APPLIED result within the window is
// replayed; PROVEN_ABSENT permits one new apply; PENDING or UNKNOWN_APPLIED fails
// closed with ErrUnknownApplied. Any apply error, or a failure to record a
// successful apply, transitions to UNKNOWN_APPLIED before returning.
//
// The whole check-and-apply runs under a cross-process lock on <ledger>.lock, and
// re-reads the ledger once inside it, so two CONCURRENT same-key applies in
// separate processes collapse to one: the winner applies and records, the loser
// blocks, then sees the fresh record and replays it. Without the reload the loser
// would consult its Open-time snapshot, miss, and double-apply. If the lock cannot
// be taken within LockWait, Do returns ErrBusy having applied nothing.
func (s *Store) Do(key, op string, apply func() (string, error)) (result string, replayed bool, err error) {
	return s.do(key, op, RequestIdentity{}, apply)
}

func (s *Store) do(key, op string, ident RequestIdentity, apply func() (string, error)) (result string, replayed bool, err error) {
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
		bound := !ident.IsZero()
		if r, ok := s.recs[key]; ok {
			if bound {
				if r.Identity == nil || r.Identity.IsZero() {
					return ErrLegacyBindingRequired
				}
				if *r.Identity != ident {
					return ErrIdentityConflict
				}
			}
			switch r.State {
			case StateApplied:
				if applied, fresh := s.lookupLocked(key); fresh {
					res, rep = applied.Result, true
					return nil
				}
			case StatePending, StateUnknownApplied:
				return unknownAppliedError(key, nil)
			case StateProvenAbsent:
				// Operation-specific read-back proved retry is safe.
			default:
				return fmt.Errorf("idempotency: key %s has invalid state %q", key, r.State)
			}
		}

		var identPtr *RequestIdentity
		if bound {
			identCopy := ident
			identPtr = &identCopy
		}

		now := s.now().UnixNano()
		intent := Record{Key: key, Op: op, State: StatePending, Identity: identPtr, UpdatedAt: now}
		if err := s.appendAndRememberLocked(intent); err != nil {
			return err
		}
		out, err := apply()
		if err != nil {
			unknown := Record{Key: key, Op: op, State: StateUnknownApplied, Identity: identPtr, UpdatedAt: s.now().UnixNano()}
			persistErr := s.appendAndRememberLocked(unknown)
			return unknownAppliedError(key, errors.Join(err, persistErr))
		}
		appliedAt := s.now().UnixNano()
		rec := Record{
			Key: key, Op: op, State: StateApplied, Result: out,
			Identity:  identPtr,
			AppliedAt: appliedAt, UpdatedAt: appliedAt,
		}
		if err := s.appendAndRememberLocked(rec); err != nil {
			unknown := Record{Key: key, Op: op, State: StateUnknownApplied, Identity: identPtr, UpdatedAt: s.now().UnixNano()}
			persistErr := s.appendAndRememberLocked(unknown)
			return unknownAppliedError(key, errors.Join(err, persistErr))
		}
		res, rep = out, false
		return nil
	}); err != nil {
		return "", false, err
	}
	return res, rep, nil
}

// Resolve records an operation-specific read-back verdict for an ambiguous key.
// APPLIED makes the supplied result replayable; ABSENT permits one new apply;
// UNKNOWN and read-back failures retain the fail-closed state.
func (s *Store) Resolve(key string, readback Readback) (Record, error) {
	if readback == nil {
		return Record{}, errors.New("idempotency: nil read-back resolver")
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	var resolved Record
	err := s.withLock(func() error {
		if err := s.loadLocked(); err != nil {
			return err
		}
		current, ok := s.recs[key]
		if !ok {
			return fmt.Errorf("%w: %s", ErrNotFound, key)
		}
		if current.State == StateApplied || current.State == StateProvenAbsent {
			resolved = current
			return nil
		}
		if current.State != StatePending && current.State != StateUnknownApplied {
			return fmt.Errorf("idempotency: key %s has invalid state %q", key, current.State)
		}

		verdict, result, readErr := readback(current)
		if readErr != nil {
			return s.retainUnknownLocked(current, readErr)
		}
		now := s.now().UnixNano()
		switch verdict {
		case ResolutionApplied:
			resolved = Record{
				Key: key, Op: current.Op, State: StateApplied, Result: result,
				Identity:  current.Identity,
				AppliedAt: now, UpdatedAt: now,
			}
		case ResolutionAbsent:
			resolved = Record{Key: key, Op: current.Op, State: StateProvenAbsent, Identity: current.Identity, UpdatedAt: now}
		case ResolutionUnknown:
			return s.retainUnknownLocked(current, nil)
		default:
			return s.retainUnknownLocked(current, fmt.Errorf("idempotency: invalid resolution %q", verdict))
		}
		if err := s.appendAndRememberLocked(resolved); err != nil {
			return s.retainUnknownLocked(current, err)
		}
		return nil
	})
	return resolved, err
}

func (s *Store) retainUnknownLocked(current Record, cause error) error {
	unknown := Record{
		Key: current.Key, Op: current.Op, State: StateUnknownApplied,
		Identity:  current.Identity,
		UpdatedAt: s.now().UnixNano(),
	}
	persistErr := s.appendAndRememberLocked(unknown)
	return unknownAppliedError(current.Key, errors.Join(cause, persistErr))
}

func unknownAppliedError(key string, cause error) error {
	if cause == nil {
		return fmt.Errorf("%w for key %s", ErrUnknownApplied, key)
	}
	return fmt.Errorf("%w for key %s: %w", ErrUnknownApplied, key, cause)
}

func (s *Store) appendAndRememberLocked(rec Record) error {
	appendRecord := s.appendRecord
	if appendRecord == nil {
		appendRecord = s.appendLocked
	}
	if err := appendRecord(rec); err != nil {
		return err
	}
	s.recs[rec.Key] = rec
	return nil
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
	if err := f.Sync(); err != nil {
		return fmt.Errorf("idempotency: fsync ledger %s: %w", s.path, err)
	}
	return nil
}
