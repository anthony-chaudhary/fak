// Package goalpark persists long provider Retry-After waits outside a worker's
// active context budget and arbitrates exactly-once resume after the reset.
package goalpark

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const Schema = "fak.goal-park.v1"
const LongWaitFloor = time.Hour

// MaxWait caps how long ONE park may wall its account off a goal. A provider
// Retry-After is untrusted input — an oversized, mis-scaled or malformed value
// (seconds vs milliseconds, a far-future HTTP-date) parks for months — and
// #4805's park has no other expiry, so an unclamped parked_until is the
// difference between "a wait" and "a permanent wall". 24h sits above every real
// subscription reset window (the longest, the weekly cap, resolves well inside a
// day) and far below the multi-day walls actually observed in the field, so a
// legitimate long 429 is unchanged and only an absurd one is clipped.
const MaxWait = 24 * time.Hour

var (
	ErrNotDue   = errors.New("goalpark: earliest legal resume time has not arrived")
	ErrClaimed  = errors.New("goalpark: resume already claimed")
	ErrNotFound = errors.New("goalpark: parked goal not found")
)

type Record struct {
	Schema      string   `json:"schema"`
	Goal        string   `json:"goal"`
	Lane        string   `json:"lane,omitempty"`
	Reason      string   `json:"reason"`
	ParkedUntil int64    `json:"parked_until"`
	ParkedAt    int64    `json:"parked_at"`
	Account     string   `json:"account,omitempty"`
	Pool        string   `json:"pool,omitempty"`
	Lease       string   `json:"lease,omitempty"`
	Witness     string   `json:"witness_requirement,omitempty"`
	Command     []string `json:"command"`
	ClaimedAt   int64    `json:"claimed_at,omitempty"`
	ClaimedBy   string   `json:"claimed_by,omitempty"`
	NextAction  string   `json:"next_legal_action"`
}

// SameAccount is the one account-identity comparison the park uses. Identities
// arrive from env (DISPATCH_ACCOUNT) and from a JSON record, so both sides are
// trimmed; an empty identity never matches anything, including another empty one
// — see Blocks for why an unattributed park must wall nobody.
func SameAccount(a, b string) bool {
	a, b = strings.TrimSpace(a), strings.TrimSpace(b)
	return a != "" && a == b
}

// Blocks reports whether this park record should still stop `account` from
// working its goal. It is the ONE predicate every reader consults, so a park can
// never be account-blind on one code path and account-scoped on another.
//
// A park is a statement about ONE account's wall on a goal, never about the goal
// itself: account A's hour-long 429 must not stop account B from taking the same
// lane. Before this predicate existed the check was lane-scoped and
// account-blind, so a single account's 429 disabled an entire lane for DAYS
// across every account while the dispatcher kept dispatching into it. Hence:
//
//   - an EMPTY Record.Account blocks NOBODY. A record that cannot name whose wall
//     it is has no one to stop, and walling every account on an unattributed
//     record is exactly the regression above.
//   - a record naming a DIFFERENT account blocks nobody: that account's wall is
//     not this account's problem, and the caller must fall through to its own
//     account-rotation path.
//   - a record naming THIS account blocks only until parked_until, and only while
//     the resume is still unclaimed.
func (r Record) Blocks(account string, now time.Time) bool {
	if r.Schema != Schema || !SameAccount(r.Account, account) {
		return false
	}
	return r.ClaimedAt == 0 && now.Unix() < r.ParkedUntil
}

type Store struct{ Dir string }

func (s Store) path(goal string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(goal)))
	return filepath.Join(s.Dir, hex.EncodeToString(sum[:12])+".json")
}

func (s Store) Park(r Record) error {
	if strings.TrimSpace(r.Goal) == "" || r.ParkedUntil <= r.ParkedAt || len(r.Command) == 0 {
		return errors.New("goalpark: invalid park record")
	}
	r.Schema = Schema
	r.ClaimedAt = 0
	r.ClaimedBy = ""
	r.NextAction = "wait until parked_until; then supervisor atomically claims and resumes the same command"
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp, err := os.CreateTemp(s.Dir, "park-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = tmp.Write(b); err == nil {
		err = tmp.Sync()
	}
	if closeErr := tmp.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	return os.Rename(name, s.path(r.Goal))
}

// List returns every readable parked record. Malformed siblings are skipped so
// one torn/foreign file cannot hide the rest of the supervisor queue.
func (s Store) List() ([]Record, error) {
	entries, err := os.ReadDir(s.Dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	out := make([]Record, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		b, readErr := os.ReadFile(filepath.Join(s.Dir, entry.Name()))
		if readErr != nil {
			continue
		}
		var r Record
		if json.Unmarshal(b, &r) == nil && r.Schema == Schema {
			out = append(out, r)
		}
	}
	return out, nil
}

func (s Store) Load(goal string) (Record, error) {
	b, err := os.ReadFile(s.path(goal))
	if errors.Is(err, os.ErrNotExist) {
		return Record{}, ErrNotFound
	}
	if err != nil {
		return Record{}, err
	}
	var r Record
	if err = json.Unmarshal(b, &r); err != nil {
		return Record{}, err
	}
	if r.Schema != Schema {
		return Record{}, errors.New("goalpark: unsupported schema")
	}
	return r, nil
}

// ClaimDue creates an exclusive claim sidecar before updating the public record.
// Across process restarts and concurrent supervisors exactly one caller can win.
func (s Store) ClaimDue(goal, supervisor string, now time.Time) (Record, error) {
	r, err := s.Load(goal)
	if err != nil {
		return Record{}, err
	}
	if r.ClaimedAt != 0 {
		return r, ErrClaimed
	}
	if now.Unix() < r.ParkedUntil {
		return r, ErrNotDue
	}
	claim := s.path(goal) + ".claim"
	f, err := os.OpenFile(claim, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if errors.Is(err, os.ErrExist) {
		return r, ErrClaimed
	}
	if err != nil {
		return r, err
	}
	fmt.Fprintf(f, "%s %d\n", supervisor, now.Unix())
	if err = f.Close(); err != nil {
		return r, err
	}
	r.ClaimedAt = now.Unix()
	r.ClaimedBy = supervisor
	r.NextAction = "resume claimed; launch command exactly once and retain lease/witness contract"
	b, _ := json.MarshalIndent(r, "", "  ")
	b = append(b, '\n')
	if err = os.WriteFile(s.path(goal), b, 0o600); err != nil {
		return r, err
	}
	return r, nil
}

// Resolve answers the only question a supervisor actually has — "may `account`
// work `goal` right now?" — and is the seam that makes a park RETIRE instead of
// linger. It is the single entry point every guard/dispatch reader uses; going
// through Load + an ad-hoc condition is what let the account scoping drift.
//
// blocked is Record.Blocks: only THIS account's own unclaimed, unexpired park
// walls it. Everything else falls through so the caller can rotate accounts.
//
// When the wait HAS elapsed, Resolve claims the record on the spot. #4805 wrote
// "wait until parked_until; then supervisor atomically claims and resumes the
// same command" into every record's next_legal_action, but nothing in the
// product ever called ClaimDue — only the `fak goal-park claim` CLI, which no
// automation invokes — so claimed_at stayed 0 forever and a due park was never
// resumed or retired; it just accumulated in the store. The guard asking this
// question IS that supervisor: it is the process about to run the goal's command
// past the elapsed wait, so it claims the resume exactly once (ClaimDue's
// O_EXCL sidecar keeps that exactly-once across concurrent supervisors) and the
// park is done. A claim is only ever taken for a record that is ALREADY due, so
// it can never cut short another account's live wall; a foreign account's due
// park is left for that account's own supervisor, which is what keeps the
// exactly-once resume per account.
//
// Errors fail OPEN (not blocked): a missing, unreadable or malformed park record
// must never be able to wall a lane — over-parking is the failure mode this
// whole seam exists to prevent.
func (s Store) Resolve(goal, account, supervisor string, now time.Time) (Record, bool) {
	rec, err := s.Load(goal)
	if err != nil {
		return Record{}, false
	}
	if rec.Blocks(account, now) {
		return rec, true
	}
	due := rec.ClaimedAt == 0 && now.Unix() >= rec.ParkedUntil
	ours := SameAccount(rec.Account, account) || strings.TrimSpace(rec.Account) == ""
	if due && ours {
		if claimed, claimErr := s.ClaimDue(goal, supervisor, now); claimErr == nil {
			return claimed, false
		}
	}
	return rec, false
}

func ParseRetryAfter(value string, now time.Time) (time.Time, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Time{}, false
	}
	if seconds, err := strconv.ParseInt(value, 10, 64); err == nil && seconds >= 0 {
		return now.Add(time.Duration(seconds) * time.Second), true
	}
	when, err := http.ParseTime(value)
	return when, err == nil && !when.Before(now)
}

// RecordLongRetry persists only a provider 429 whose legal wait is at least one
// hour. Other 429 classes remain on their existing transient/quarantine paths.
func (s Store) RecordLongRetry(status int, h http.Header, now time.Time, r Record) (bool, error) {
	if status != http.StatusTooManyRequests {
		return false, nil
	}
	until, ok := ParseRetryAfter(h.Get("Retry-After"), now)
	if !ok || until.Sub(now) < LongWaitFloor {
		return false, nil
	}
	// Clamp before persisting, not at read time, so the bound is visible in the
	// record an operator inspects and cannot be lost by a reader that forgets it.
	if until.Sub(now) > MaxWait {
		until = now.Add(MaxWait)
	}
	r.Reason = "LONG_RETRY_AFTER"
	r.ParkedAt = now.Unix()
	r.ParkedUntil = until.Unix()
	return true, s.Park(r)
}
