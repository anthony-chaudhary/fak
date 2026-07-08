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
// Scope note: dedup here assumes the retry is SEQUENTIAL after the first attempt
// has recorded (the hang case: attempt 1 lands, its signal is lost, attempt 2
// retries later). Two concurrent same-key applies across separate processes are
// not serialized by this store — that needs a cross-process lock (internal/flock)
// and is deliberately out of scope for the minimal mechanism.
package idempotency

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

// DefaultWindow is the dedup horizon: a key applied within this span replays; an
// older record is treated as expired so a reused token eventually proceeds again.
// A post-hang retry arrives seconds-to-minutes later, so a generous default still
// catches it while never pinning a token forever.
const DefaultWindow = 24 * time.Hour

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
	b, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return s, nil
		}
		return nil, fmt.Errorf("idempotency: read ledger %s: %w", path, err)
	}
	// Keep the latest record per key so a re-applied (expired-then-reused) key
	// reflects its most recent apply, not the first.
	for _, r := range jsonlledger.Parse[Record](string(b), nil) {
		if r.Key == "" {
			continue
		}
		if prev, ok := s.recs[r.Key]; !ok || r.AppliedAt >= prev.AppliedAt {
			s.recs[r.Key] = r
		}
	}
	return s, nil
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
func (s *Store) Do(key, op string, apply func() (string, error)) (result string, replayed bool, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if r, ok := s.lookupLocked(key); ok {
		return r.Result, true, nil
	}
	res, err := apply()
	if err != nil {
		return "", false, err
	}
	rec := Record{Key: key, Op: op, Result: res, AppliedAt: s.now().UnixNano()}
	if err := s.appendLocked(rec); err != nil {
		return "", false, err
	}
	s.recs[key] = rec
	return res, false, nil
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
