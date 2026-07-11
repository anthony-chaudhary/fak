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
	// Signals is the hysteretic overload latch. Each key is one independent load
	// signal and its value is that signal's clear deadline. The account remains
	// non-servable while ANY signal is active and is re-admitted only after ALL
	// signals clear. Omitted on legacy v1 rows, which are treated as one Kind signal.
	Signals map[string]time.Time `json:"signals,omitempty"`
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

	// Fleet degraded mode (#3383): an aggregate over the cooldown set. degraded
	// is true exactly while at least one account sits in the skip-set; it engages
	// on the first-down edge (set size 0->1) and restores on the last-up edge
	// (1->0). degradedSince marks the engage edge's observation time, zero when
	// not degraded. Derived state, never persisted — LoadCooldownStore re-derives
	// both from the entries it reads.
	degraded      bool
	degradedSince time.Time
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
	s.syncDegraded(earliestCooledAt(s.entries))
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
	if !ok {
		return CooldownEntry{}, false
	}
	signals, resetAt := activeCooldownSignals(e, now)
	if len(signals) == 0 {
		return CooldownEntry{}, false
	}
	e.Signals = signals
	e.ResetAt = resetAt
	return e, true
}

// Degraded reports whether the fleet is in degraded mode: at least one account
// is currently in the cooldown skip-set (#3383). Expiry is realized lazily,
// matching the entries map itself: an elapsed entry keeps the fleet degraded
// until the store observes the recovery (Prune, or the account's last signal
// clearing). Added observability only — no admission decision reads it.
func (s *CooldownStore) Degraded() bool { return s.degraded }

// DegradedSince returns when degraded mode engaged — the observation time of
// the first account down (for a loaded store, the earliest CooledAt among its
// entries) — or the zero time when the fleet is not degraded.
func (s *CooldownStore) DegradedSince() time.Time { return s.degradedSince }

// syncDegraded re-derives the fleet degraded flag from cooldown-set membership.
// It is called at every point an account enters or leaves the skip-set
// (UpdateOverload/Cool, Clear, Prune, LoadCooldownStore), so the flag can never
// disagree with the set. Only the two edges transition: 0->1 engages (recording
// engagedAt as the engaged-since marker) and 1->0 restores; any other size
// change — a second account down, one of several recovering, an already-cooled
// account re-cooling — leaves both fields untouched, so the fleet never
// re-engages or flaps inside a stable state. engagedAt is ignored on restore.
func (s *CooldownStore) syncDegraded(engagedAt time.Time) {
	switch {
	case !s.degraded && len(s.entries) > 0:
		s.degraded = true
		s.degradedSince = engagedAt.UTC()
	case s.degraded && len(s.entries) == 0:
		s.degraded = false
		s.degradedSince = time.Time{}
	}
}

// UpdateOverload folds one independent load signal into the account latch. An over-threshold
// observation sets/extends that signal. A cleared observation removes only that signal; the
// account remains latched while any sibling signal is active. changed is true ONLY when pool
// membership crosses the boundary (servable->latched or latched->servable), so a publisher can
// republish incrementally without emitting on threshold noise inside either stable state.
func (s *CooldownStore) UpdateOverload(account, signal string, kind CooldownKind, overloaded bool, reason string, observedAt, resetAt time.Time) (entry CooldownEntry, changed bool) {
	account = strings.TrimSpace(account)
	signal = strings.TrimSpace(signal)
	if account == "" || signal == "" {
		return CooldownEntry{}, false
	}
	beforeEntry, beforeLatched := s.CooledDown(account, observedAt)
	signals := beforeEntry.Signals
	if signals == nil {
		signals = map[string]time.Time{}
	}

	if !overloaded {
		delete(signals, signal)
		if len(signals) == 0 {
			delete(s.entries, account)
			s.syncDegraded(observedAt)
			return CooldownEntry{}, beforeLatched
		}
		beforeEntry.Signals = signals
		beforeEntry.ResetAt = latestSignalReset(signals)
		s.entries[account] = beforeEntry
		return beforeEntry, false
	}

	if resetAt.IsZero() {
		resetAt = observedAt.Add(defaultWindowFor(kind))
	}
	resetAt = resetAt.UTC()
	if prior, ok := signals[signal]; !ok || resetAt.After(prior) {
		signals[signal] = resetAt
	}
	cooledAt := observedAt
	if beforeLatched && !beforeEntry.CooledAt.IsZero() {
		cooledAt = beforeEntry.CooledAt
	}
	entry = CooldownEntry{
		Account:  account,
		Kind:     kind,
		Reason:   reason,
		CooledAt: cooledAt,
		ResetAt:  latestSignalReset(signals),
		Signals:  signals,
	}
	s.entries[account] = entry
	s.syncDegraded(observedAt)
	return entry, !beforeLatched
}

// Cool records (or extends) a cooldown for account. The reset time is the later of
// any existing active window and the newly computed one, so a weekly cap is never
// shortened by a subsequent transient 429. When resetAt is zero, the kind's
// default window from cooledAt is used.
func (s *CooldownStore) Cool(account string, kind CooldownKind, reason string, cooledAt, resetAt time.Time) CooldownEntry {
	e, _ := s.UpdateOverload(account, string(kind), kind, true, reason, cooledAt, resetAt)
	return e
}

// Clear removes any cooldown for account (manual override for "it's back now").
// Reports whether an entry was present.
func (s *CooldownStore) Clear(account string) bool {
	if _, ok := s.entries[account]; ok {
		delete(s.entries, account)
		s.syncDegraded(time.Time{})
		return true
	}
	return false
}

// Prune drops entries whose window has elapsed at now. Reports how many were removed.
func (s *CooldownStore) Prune(now time.Time) int {
	n := 0
	for k, e := range s.entries {
		signals, resetAt := activeCooldownSignals(e, now)
		if len(signals) == 0 {
			delete(s.entries, k)
			n++
			continue
		}
		e.Signals = signals
		e.ResetAt = resetAt
		s.entries[k] = e
	}
	s.syncDegraded(now)
	return n
}

// Active returns the entries still holding at now, sorted by account.
func (s *CooldownStore) Active(now time.Time) []CooldownEntry {
	var out []CooldownEntry
	for account := range s.entries {
		if e, ok := s.CooledDown(account, now); ok {
			out = append(out, e)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Account < out[j].Account })
	return out
}

func activeCooldownSignals(e CooldownEntry, now time.Time) (map[string]time.Time, time.Time) {
	active := map[string]time.Time{}
	if len(e.Signals) == 0 {
		if e.Active(now) {
			name := strings.TrimSpace(string(e.Kind))
			if name == "" {
				name = "legacy"
			}
			active[name] = e.ResetAt.UTC()
		}
		return active, latestSignalReset(active)
	}
	for signal, resetAt := range e.Signals {
		if strings.TrimSpace(signal) != "" && now.Before(resetAt) {
			active[signal] = resetAt.UTC()
		}
	}
	return active, latestSignalReset(active)
}

// earliestCooledAt returns the earliest CooledAt across entries (UTC), or the
// zero time when entries is empty. It reconstructs the engaged-since marker for
// a store loaded from disk: the first account down is the oldest cool still held.
func earliestCooledAt(entries map[string]CooldownEntry) time.Time {
	var earliest time.Time
	for _, e := range entries {
		if !e.CooledAt.IsZero() && (earliest.IsZero() || e.CooledAt.Before(earliest)) {
			earliest = e.CooledAt
		}
	}
	return earliest.UTC()
}

func latestSignalReset(signals map[string]time.Time) time.Time {
	var latest time.Time
	for _, resetAt := range signals {
		if resetAt.After(latest) {
			latest = resetAt
		}
	}
	return latest.UTC()
}

func defaultWindowFor(kind CooldownKind) time.Duration {
	if kind == CooldownRateLimit {
		return RateLimitCooldownWindow
	}
	return DefaultCooldownWindow
}
