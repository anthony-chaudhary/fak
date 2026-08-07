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
// difference between "a wait" and "a permanent wall".
//
// 24h is a BOUND ON THE DAMAGE ONE PARK MAY DO, not a claim that every wall fits
// inside a day — an earlier version of this comment asserted the latter ("the
// longest, the weekly cap, resolves well inside a day") and the corpus refutes
// it: of the 27 records on disk when this was measured, waits of 127.89h and
// 129.73h (both announcing 2026-08-12T14:59:59Z, a genuine weekly reset 5.4 days
// out), plus 71.71h, 39.05h and 35.23h, all announced walls LONGER than the
// clamp. So the clamp routinely retires a park BEFORE its wall actually lifts.
// That is the intended trade and it is safe in this direction: a park that
// expires early fails OPEN (the account is re-admitted and, if the wall is still
// up, the next 429 re-parks under the newly announced wait — Park re-arms the
// probe budget for exactly this), whereas an unclamped park fails CLOSED and can
// wall a seat for months on one malformed Retry-After. AdmitProbe is the other
// half: it is what keeps a genuine multi-day wall from costing a full window of
// suppressed work. Do not raise the clamp to "cover" the weekly reset — that
// re-introduces the permanent-wall failure this bound exists to prevent.
const MaxWait = 24 * time.Hour

// ProbeBudget bounds how many probe runs ONE park may admit over its whole
// wait. It is the reason a park cannot self-seal.
//
// A park suppresses exactly the runs that would produce the evidence its wall is
// gone, so a park whose only exit is its own timer can never learn that its
// condition ended early — and these conditions DO end early. The wall is one
// account's subscription cap; the provider resets it on its own schedule, an
// operator can clear the matching account cooldown, and the pool holds other
// accounts that were never walled at all. Measured over the clean current-fleet
// window (2026-08-04T00:00Z..08-07T07:00Z, 291 graded resolve units): of the 41
// units the guard tore down on the "context budget signal ignored as terminal"
// branch, 25 had ANOTHER pool account with logged successful turns within ±45
// minutes of the teardown, and in 0 of the 41 was every other account inside a
// recorded cooldown. "The whole pool is walled" — the reading under which
// declining is correct — was therefore demonstrably false most of the time this
// park stopped work.
//
// 4 is what makes probing affordable: a probe costs one worker unit, and a park
// may spend at most ProbeBudget of them however long the wall is. Spacing comes
// from the provider's own announced wait (ProbeInterval), never from an invented
// timeout.
const ProbeBudget = 4

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
	// Probes/LastProbeAt are the anti-self-seal ledger: how many probe runs this
	// park generation has already let through the wall, and when the last one was
	// admitted. Persisted (not in-memory) because the supervisors that consult a
	// park are separate processes, and reset by Park so a re-park under a freshly
	// announced wait starts with a full budget.
	Probes      int   `json:"probes,omitempty"`
	LastProbeAt int64 `json:"last_probe_at,omitempty"`
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

// ProbeInterval is the minimum spacing between two probe runs of one park: the
// provider's announced wait divided by ProbeBudget, floored at LongWaitFloor.
//
// Deriving it from the ANNOUNCED wait is what keeps the bound honest. The
// provider hands us a real number (the field's observed 429s announce waits from
// ~1h13m to the multi-day weekly reset), so a short wall gets tight probing and a
// long one gets sparse probing, with the same worst-case cost of ProbeBudget
// units either way; no fixed guess can do both.
//
// The LongWaitFloor floor is the re-hit bound. A probe that immediately walks
// back into the same wall is worse than no probe — it burns a unit and learns
// nothing — and this park only exists for waits of at least LongWaitFloor, so no
// probe may be admitted less than that after the wall was recorded. A degenerate
// or non-positive window falls back to the floor rather than probing freely.
func (r Record) ProbeInterval() time.Duration {
	wait := time.Duration(r.ParkedUntil-r.ParkedAt) * time.Second
	if iv := wait / ProbeBudget; iv > LongWaitFloor {
		return iv
	}
	return LongWaitFloor
}

// ProbeDue reports whether this park's next probe slot has opened. It is a pure
// predicate over the record — AdmitProbe is what actually spends the slot — and
// it deliberately measures from the LAST admitted probe, falling back to
// ParkedAt, so the very first probe is spaced off the moment the wall was
// recorded rather than off process start.
func (r Record) ProbeDue(now time.Time) bool {
	if r.Probes >= ProbeBudget {
		return false
	}
	since := r.LastProbeAt
	if since == 0 {
		since = r.ParkedAt
	}
	return !now.Before(time.Unix(since, 0).Add(r.ProbeInterval()))
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
	// A re-park is a NEW wall with a newly announced wait — the provider just told
	// us the old number is stale — so it starts with a full probe budget. The slot
	// sidecars are keyed by ParkedAt (probePath) precisely so this reset cannot
	// collide with the previous generation's already-spent slots and silently
	// leave the new park unprobeable.
	r.Probes = 0
	r.LastProbeAt = 0
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
// A blocked verdict is additionally offered the park's open probe slot, if any,
// so the wall cannot seal itself shut for its whole announced window — see
// AdmitProbe for the bound and for all three of this park's clearing paths.
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
		// The wall is still standing for this account — but a park that only ever
		// answers "blocked" suppresses the very run that would show the wall is
		// gone, so it can outlive its own condition indefinitely. Spend a probe
		// slot if one has opened (AdmitProbe documents the bound and the other two
		// clearing paths); the record stays live either way, so a wall that really
		// is still up costs one unit per slot rather than the whole window.
		if probed, ok := s.AdmitProbe(goal, supervisor, now); ok {
			return probed, false
		}
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

// probePath names the exclusive sidecar for one probe SLOT of one park
// GENERATION. ParkedAt is in the name because Park resets Probes to 0: without
// it a re-park would ask for slot 0 again, find the previous generation's file,
// and never be probeable — a park that re-arms itself into permanent silence,
// which is the exact failure this whole seam exists to prevent.
func (s Store) probePath(goal string, parkedAt int64, slot int) string {
	return fmt.Sprintf("%s.probe-%d-%d", s.path(goal), parkedAt, slot)
}

// AdmitProbe hands exactly ONE caller the park's currently-open probe slot: a
// single run allowed through a wall that is otherwise still standing, so the
// park can learn whether its condition survives.
//
// It is the second of this park's three clearing paths, and the only one that
// does not depend on the wall lasting exactly as long as it was announced to:
//
//  1. parked_until elapses — Resolve claims and retires the record (the timer).
//  2. a probe slot opens — one run passes through and its outcome is the
//     evidence: a run that re-hits the 429 re-parks under the newly announced
//     wait (RecordLongRetry -> Park, which re-arms the budget), a run that does
//     not simply works, which is the recovery this exists for.
//  3. MaxWait clamps the recorded window at write time, so no announced wait —
//     malformed, mis-scaled, or a genuine multi-day weekly reset — parks longer
//     than a day.
//
// Admitting a probe deliberately does NOT clear, claim, or shorten the park: the
// record stays live and keeps walling every other run on the walled account. A
// probe is a one-shot pass, not a verdict, so a wall that really is still up
// costs one unit per slot instead of the whole window.
//
// Exclusivity uses the same O_EXCL sidecar discipline as ClaimDue, so concurrent
// supervisors across processes cannot both spend one slot. Every failure path —
// unreadable record, slot already taken, unwritable store — returns false: an
// error must never manufacture a probe, because a probe costs a worker unit.
func (s Store) AdmitProbe(goal, prober string, now time.Time) (Record, bool) {
	r, err := s.Load(goal)
	if err != nil || !r.ProbeDue(now) {
		return r, false
	}
	f, err := os.OpenFile(s.probePath(goal, r.ParkedAt, r.Probes), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return r, false
	}
	fmt.Fprintf(f, "%s %d\n", prober, now.Unix())
	if err = f.Close(); err != nil {
		return r, false
	}
	r.Probes++
	r.LastProbeAt = now.Unix()
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return r, false
	}
	if err = os.WriteFile(s.path(goal), append(b, '\n'), 0o600); err != nil {
		return r, false
	}
	return r, true
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
