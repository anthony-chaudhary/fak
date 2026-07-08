package accounts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

// cooldown.go is the account usage-limit cooldown gate. LoginStatus() folds only
// static registry facts (creds present, enabled, not tombstoned); it deliberately
// knows nothing about a seat being *temporarily* spent. When a launch hits a
// usage/weekly/session cap or a transient 429, the account is logged-in and its
// credentials are valid, so LoginStatus() still returns Ready and the pool keeps
// dispatching into a wall.
//
// This file adds a small, persistent, time-injected overlay: an account that hit
// a limit is recorded with a reset_at, and CooledDown(account, now) reports
// whether it is still within that window. The login overlay (see loginObservation)
// downgrades an otherwise-Ready seat to LoginCooledDown so CanServe drops it from
// the pool until the window elapses, then it auto-restores with no manual action.
//
// The store is keyed by ACCOUNT (not seat name): a usage cap is billed to the
// upstream account, so every seat sharing that account key must cool together.

// DefaultCooldownWindow is the fallback window applied to a usage/weekly limit
// when the error text carries no explicit reset time. Sized to a rolling 5-hour
// cap's typical near-term reset; a still-limited account simply re-cools on its
// next probe, so this never over-holds an account that is actually free.
const DefaultCooldownWindow = time.Hour

// RateLimitCooldownWindow is the short window for a transient 429/overload, which
// clears far faster than a usage cap.
const RateLimitCooldownWindow = 5 * time.Minute

// CooldownKind distinguishes a self-recovering usage/weekly cap from a transient
// throttle. It mirrors the launch-layer classification (launchModelUsageLimit /
// launchModelRateLimit) so the write side maps one-to-one.
type CooldownKind string

const (
	CooldownUsageLimit CooldownKind = "usage-limit"
	CooldownRateLimit  CooldownKind = "rate-limit"
)

// CooldownEntry records one account's active cooldown. Times are UTC.
type CooldownEntry struct {
	Account  string       `json:"account"`
	Kind     CooldownKind `json:"kind"`
	Reason   string       `json:"reason,omitempty"`
	CooledAt time.Time    `json:"cooled_at"`
	ResetAt  time.Time    `json:"reset_at"`
}

// Active reports whether the entry still holds at now (i.e. now is before ResetAt).
func (e CooldownEntry) Active(now time.Time) bool { return now.Before(e.ResetAt) }

// resetAtRE pulls an explicit absolute reset time out of a limit message when the upstream
// names one ("usage limit reached; resets at 2026-07-07T15:00:00Z"). Only the RFC3339
// date+time form is trusted as absolute; a looser "resets in 42 minutes" or a bare
// wall-clock without a date is deliberately NOT matched, so a reset is never mis-parsed
// from an ambiguous local time. It is compiled once and shared by every reset writer.
var resetAtRE = regexp.MustCompile(`(?i)resets?\s+at\s+(\d{4}-\d{2}-\d{2}[tT]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?)`)

// ParseReset returns the explicit absolute reset time named in a limit message, normalized
// to UTC, or the zero time when none is confidently parseable (the caller then falls back to
// the cooldown kind's default window). It is the SINGLE reset parser both cooldown writers
// share — the launch-exit writer (cmd/fak recordLaunchCooldown) and the live-429 rehome
// writer (guardrotate.PersistCooldownForRehome) — so the two can never hold one account to
// two different reset times by drifting apart. An absolute timestamp stands on its own, so
// no `now` anchor is needed.
func ParseReset(message string) time.Time {
	m := resetAtRE.FindStringSubmatch(message)
	if m == nil {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05"} {
		if ts, err := time.Parse(layout, m[1]); err == nil {
			return ts.UTC()
		}
	}
	return time.Time{}
}

// CooldownStore is the fleet-shared, on-disk set of account cooldowns. It is a
// plain JSON map keyed by account so concurrent checkouts and the durable
// watchdogs honor the same windows. All methods are pure over the in-memory map;
// Load/Save are the only I/O.
type CooldownStore struct {
	path    string
	entries map[string]CooldownEntry
}

// CooldownStoreSchema tags the persisted file.
const CooldownStoreSchema = "fak.accounts.cooldown.v1"

type cooldownFile struct {
	Schema  string          `json:"schema"`
	Entries []CooldownEntry `json:"entries"`
}

// LoadCooldownStore reads the cooldown file at path. A missing file is not an
// error — it yields an empty store bound to that path so a later Save creates it.
func LoadCooldownStore(path string) (*CooldownStore, error) {
	s := &CooldownStore{path: path, entries: map[string]CooldownEntry{}}
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return s, nil
		}
		return s, err
	}
	if len(strings.TrimSpace(string(raw))) == 0 {
		return s, nil
	}
	var f cooldownFile
	if err := json.Unmarshal(raw, &f); err != nil {
		return s, err
	}
	for _, e := range f.Entries {
		if e.Account == "" {
			continue
		}
		s.entries[e.Account] = e
	}
	return s, nil
}

// Save writes the store atomically (temp + rename) so a concurrent reader never
// sees a half-written file. Entries are sorted by account for a stable diff.
func (s *CooldownStore) Save() error {
	if s.path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o755); err != nil {
		return err
	}
	f := cooldownFile{Schema: CooldownStoreSchema}
	for _, e := range s.entries {
		f.Entries = append(f.Entries, e)
	}
	sort.Slice(f.Entries, func(i, j int) bool { return f.Entries[i].Account < f.Entries[j].Account })
	raw, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.path)
}

// CooledDown reports the active cooldown entry for account at now, and whether one
// exists. An entry whose window has elapsed is treated as absent (and is pruned on
// the next Save via Prune).
func (s *CooldownStore) CooledDown(account string, now time.Time) (CooldownEntry, bool) {
	if account == "" {
		return CooldownEntry{}, false
	}
	e, ok := s.entries[account]
	if !ok || !e.Active(now) {
		return CooldownEntry{}, false
	}
	return e, true
}

// Cool records (or extends) a cooldown for account. The reset time is the later of
// any existing active window and the newly computed one, so a weekly cap is never
// shortened by a subsequent transient 429. When resetAt is zero, the kind's
// default window from cooledAt is used.
func (s *CooldownStore) Cool(account string, kind CooldownKind, reason string, cooledAt, resetAt time.Time) CooldownEntry {
	if resetAt.IsZero() {
		resetAt = cooledAt.Add(defaultWindowFor(kind))
	}
	if existing, ok := s.entries[account]; ok && existing.Active(cooledAt) && existing.ResetAt.After(resetAt) {
		resetAt = existing.ResetAt
	}
	e := CooldownEntry{
		Account:  account,
		Kind:     kind,
		Reason:   reason,
		CooledAt: cooledAt,
		ResetAt:  resetAt.UTC(),
	}
	s.entries[account] = e
	return e
}

// Clear removes any cooldown for account (manual override for "it's back now").
// Reports whether an entry was present.
func (s *CooldownStore) Clear(account string) bool {
	if _, ok := s.entries[account]; ok {
		delete(s.entries, account)
		return true
	}
	return false
}

// Prune drops entries whose window has elapsed at now. Reports how many were removed.
func (s *CooldownStore) Prune(now time.Time) int {
	n := 0
	for k, e := range s.entries {
		if !e.Active(now) {
			delete(s.entries, k)
			n++
		}
	}
	return n
}

// Active returns the entries still holding at now, sorted by account.
func (s *CooldownStore) Active(now time.Time) []CooldownEntry {
	var out []CooldownEntry
	for _, e := range s.entries {
		if e.Active(now) {
			out = append(out, e)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Account < out[j].Account })
	return out
}

func defaultWindowFor(kind CooldownKind) time.Duration {
	if kind == CooldownRateLimit {
		return RateLimitCooldownWindow
	}
	return DefaultCooldownWindow
}
