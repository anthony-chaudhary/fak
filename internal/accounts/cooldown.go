package accounts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
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
// the pool until the window elapses, then it auto-restores with no manual action —
// unless the owner armed the canary exit gate (#3389, canary.go), which trades that
// timer-only restore for a witnessed round-trip.
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

// WeeklyLimitWindow is the fallback window for a limit message that names a WEEKLY cap
// but announces no window. DefaultCooldownWindow is sized for a rolling 5-hour cap, whose
// "re-cool on the next probe" self-correction is cheap; a weekly cap's real window is DAYS,
// so the same 1-hour fallback re-admits the seat ~168 times before it can possibly serve,
// and every re-admission is a doomed spawn. Six hours is the compromise: still bounded (an
// over-hold only costs idle seat time, and the account re-cools if it is genuinely free),
// but no longer a spin. #5890.
const WeeklyLimitWindow = 6 * time.Hour

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
	// Probation is derived read-time state (#3389): every signal window has
	// elapsed but the store's canary exit gate is armed, so the account is still
	// held out of the pool until CanaryExit witnesses one successful round-trip.
	// Set only on the copy CooledDown returns, never on the stored entry.
	Probation bool `json:"probation,omitempty"`
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

// ResolveReset is the reset instant a limit message implies, resolved in two tiers:
// first the absolute "resets at <RFC3339>" form (ParseReset), then a RELATIVE
// announced wait anchored to now — the shape a weekly-limit 429 actually carries
// ("weekly limit reached; announced_wait≈1h7m", "retry-after: 4020", "session
// limit; resets in 42 minutes"). It returns the zero time only when NEITHER tier is
// confidently parseable, so the caller still falls back to the cooldown kind's
// default window. This is the single reset resolver BOTH cooldown writers share
// (cmd/fak recordLaunchCooldown and guardrotate.PersistCooldownForRehome), so a
// weekly cap's announced window can never be honored by one writer and dropped by
// the other. now anchors the relative tier; an absolute reset ignores it.
//
// Why relative parsing lives here and not in ParseReset: a bare wall-clock ("resets
// at 15:00") is genuinely ambiguous (which day? which zone?) and ParseReset must
// keep refusing it. A relative wait is NOT ambiguous — now+42m is a definite instant
// — so it is safe to trust, but only ParseReset's absolute form may stand without a
// now anchor. Keeping the two tiers separate preserves ParseReset's stricter
// contract while still capturing the weekly-limit window that used to collapse to
// the 1-hour DefaultCooldownWindow and re-offer a still-capped seat early (#2610).
func ResolveReset(message string, now time.Time) time.Time {
	if at := ParseReset(message); !at.IsZero() {
		return at
	}
	if d, ok := parseRelativeWait(message); ok {
		return now.Add(d).UTC()
	}
	return time.Time{}
}

// weeklyLimitRE names a WEEKLY cap in a limit message. It is deliberately phrase-anchored —
// the word "weekly" adjacent to "limit" (including the gateway's `kind=weekly_limit` form),
// "limit … for the week", or an explicit 7-day limit — so an ordinary usage-limit line is
// never promoted to the longer weekly floor by a stray word. Anything it does not match keeps
// DefaultCooldownWindow, which is the right size for the 5-hour rolling cap.
var weeklyLimitRE = regexp.MustCompile(`(?i)weekly[\w /_-]*limit|limit[\w ]*for the week|\b7[ -]?day limit`)

// IsWeeklyLimit reports whether a limit message names a WEEKLY cap as opposed to the rolling
// 5-hour cap. The two self-recover on windows that differ by two orders of magnitude, so the
// cooldown writers must not hold them for the same fallback duration (#5890).
func IsWeeklyLimit(message string) bool { return weeklyLimitRE.MatchString(message) }

// ResolveCooldownReset is the reset instant the COOLDOWN WRITERS use: ResolveReset's two
// announced tiers (absolute, then relative), and — only when neither is present — a WEEKLY
// floor for a message that names a weekly cap. It is the single entry point both writers call
// (cmd/fak recordLaunchCooldown and guardrotate.PersistCooldownForRehome) so the weekly floor
// can never be applied by one and dropped by the other, exactly as ResolveReset already
// guarantees for the announced tiers.
//
// Why the floor lives here and not in ResolveReset: ResolveReset answers "what window did the
// upstream ANNOUNCE?", and the honest answer for an unannounced weekly cap is still "none" —
// callers that ask that question must keep getting the zero time. This function answers the
// different question "how long should we hold the seat?", where a weekly cap's known-long
// window is a better default than the rolling cap's one hour.
func ResolveCooldownReset(message string, now time.Time) time.Time {
	if at := ResolveReset(message, now); !at.IsZero() {
		return at
	}
	if IsWeeklyLimit(message) {
		return now.Add(WeeklyLimitWindow).UTC()
	}
	return time.Time{}
}

// maxRelativeWait caps a parsed relative wait so a garbled or absurd announced
// window (a parse artifact, an upstream typo) can never pin a seat out of the pool
// for longer than any real cap. 14 days comfortably clears a weekly reset.
const maxRelativeWait = 14 * 24 * time.Hour

// relativeWaitRE anchors a relative wait to an explicit retry/reset/wait cue so a
// bare number elsewhere in the message is never mistaken for one. After the cue it
// captures a compact Go-style duration (1h7m, 1h5m45s, 30s — the announced_wait
// form) OR a spaced "<n> <unit>" phrase (42 minutes, 2 hours 30 minutes). The `\b`
// keeps the standalone "wait" cue from firing inside a word like "await".
var relativeWaitRE = regexp.MustCompile(`(?i)\b(?:announced[ _]?wait|retry[ _-]?after|resets?\s+in|try\s+again\s+in|back\s+in|available\s+in|please\s+wait|wait)\s*[:=≈~]?\s*` +
	`((?:\d+(?:\.\d+)?\s*(?:hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\s*){1,3})`)

// retryAfterSecsRE matches the bare HTTP "Retry-After: <seconds>" delta form, which
// carries no unit word and so is invisible to relativeWaitRE. Tried only after the
// unit-bearing form fails, so "retry-after: 1h7m" is never misread as 1 second.
var retryAfterSecsRE = regexp.MustCompile(`(?i)\bretry[ _-]?after\s*[:=]?\s*(\d+)\b`)

// parseRelativeWait extracts a positive, cue-anchored relative wait duration from a
// limit message and reports whether one was found. It never returns a value beyond
// maxRelativeWait.
func parseRelativeWait(message string) (time.Duration, bool) {
	if m := relativeWaitRE.FindStringSubmatch(message); m != nil {
		if d, ok := normalizeWaitToDuration(m[1]); ok {
			return d, true
		}
	}
	if m := retryAfterSecsRE.FindStringSubmatch(message); m != nil {
		if n, err := strconv.Atoi(m[1]); err == nil && n > 0 {
			return capWait(time.Duration(n) * time.Second), true
		}
	}
	return 0, false
}

// waitWordReplacer rewrites unit WORDS to Go duration suffixes, longest-form first
// so "minutes" collapses to "m" rather than "min"+leftover. Compact forms already
// using h/m/s pass through untouched.
var waitWordReplacer = strings.NewReplacer(
	"hours", "h", "hour", "h", "hrs", "h", "hr", "h",
	"minutes", "m", "minute", "m", "mins", "m", "min", "m",
	"seconds", "s", "second", "s", "secs", "s", "sec", "s",
)

// normalizeWaitToDuration turns a captured wait phrase ("42 minutes", "1h 7m",
// "2 hours 30 minutes") into a Go duration, capped at maxRelativeWait. Reports false
// for an empty or unparseable phrase.
func normalizeWaitToDuration(s string) (time.Duration, bool) {
	s = waitWordReplacer.Replace(strings.ToLower(strings.TrimSpace(s)))
	s = strings.ReplaceAll(s, " ", "")
	if s == "" {
		return 0, false
	}
	d, err := time.ParseDuration(s)
	if err != nil || d <= 0 {
		return 0, false
	}
	return capWait(d), true
}

func capWait(d time.Duration) time.Duration {
	if d > maxRelativeWait {
		return maxRelativeWait
	}
	return d
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

	// canary is the probation gate's injected round-trip probe (#3389, see
	// canary.go). Nil means the gate is unarmed and cooldown exit stays
	// timer-only. Process-local, never persisted.
	canary CooldownCanary
}

// CooldownStoreSchema tags the persisted file.
const CooldownStoreSchema = "fak.accounts.cooldown.v1"

type cooldownFile struct {
	Schema  string          `json:"schema"`
	Entries []CooldownEntry `json:"entries"`
}

// shareRetryAttempts/shareRetryDelay bound the retry both ends of the store use for a
// TRANSIENT sharing failure. On Windows a peer's open handle makes an unrelated process's
// read fail (ERROR_SHARING_VIOLATION) and its publish rename fail (ERROR_ACCESS_DENIED)
// for as long as that handle lives — microseconds, since every fak toucher of this store
// reads or renames it whole and closes. Both failures are silent-but-costly on this file
// specifically: a failed read folds the gate to cooldown-blind (LoadCooldownStoreFailOpen)
// and a failed rename DROPS the cool being recorded, re-offering a capped account. ~100ms
// of total patience buys past the contention; a genuinely permanent permission error just
// pays that once and still surfaces.
const (
	shareRetryAttempts = 12
	shareRetryDelay    = 8 * time.Millisecond
)

// retryTransientShare runs op until it succeeds, reports a missing file, or the patience
// budget is spent, returning op's last error. A missing file returns immediately: absence
// is a stable answer here, not contention.
func retryTransientShare(op func() error) error {
	var err error
	for attempt := 0; attempt < shareRetryAttempts; attempt++ {
		if err = op(); err == nil || os.IsNotExist(err) {
			return err
		}
		time.Sleep(shareRetryDelay)
	}
	return err
}

// LoadCooldownStore reads the cooldown file at path. A missing file is not an
// error — it yields an empty store bound to that path so a later Save creates it.
func LoadCooldownStore(path string) (*CooldownStore, error) {
	s := &CooldownStore{path: path, entries: map[string]CooldownEntry{}}
	var raw []byte
	err := retryTransientShare(func() error {
		var readErr error
		raw, readErr = os.ReadFile(path)
		return readErr
	})
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
//
// The staging file gets a UNIQUE name per Save (#6027 root cause). This store is
// fleet-shared and written by many processes at once — every checkout, launch exit
// and watchdog tick — so a FIXED `<path>.tmp` is not a private scratch file but a
// second shared one: two savers open it concurrently, each writes its own payload
// from offset 0, and whichever renames last publishes a SPLICE of both. The payload
// length varies by a byte or two between saves all on its own (RFC3339Nano trims
// trailing zeros off the reset timestamps), so the longer writer's tail survives the
// shorter one and the published file ends in a stray `}` — valid JSON followed by a
// trailing brace, which is precisely the corruption observed in the live fleet store
// on 2026-08-09 and again on 2026-08-11. Both were sticky: the read then fails, the
// gate folds to cooldown-blind (LoadCooldownStoreFailOpen), and the write paths
// refuse to overwrite state they could not read, so nothing self-repairs.
//
// A unique staging name makes each writer's payload self-consistent; the renames
// still race, but each one publishes a WHOLE file, so last-writer-wins is the only
// loss and readers never see a spliced one.
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
	tmp, err := os.CreateTemp(filepath.Dir(s.path), ".account-cooldown-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	// 0o644, not CreateTemp's 0o600: the store is read by every fleet process and
	// the pre-#6027 os.WriteFile published it world-readable.
	if err := tmp.Chmod(0o644); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(raw); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	// Publishing is where a lost write costs the most: the caller has already decided
	// an account is capped, and a dropped rename re-offers it. Retry past a peer's
	// transient handle rather than returning an error the launch path can only log.
	return retryTransientShare(func() error { return os.Rename(name, s.path) })
}

// CooledDown reports the active cooldown entry for account at now, and whether one
// exists. An entry whose window has elapsed is treated as absent (and is pruned on
// the next Save via Prune) — unless the canary exit gate is armed (#3389, canary.go),
// in which case the elapsed entry is held in probation (Probation set) until
// CanaryExit witnesses a successful round-trip.
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
		if !s.canaryArmed() {
			return CooldownEntry{}, false
		}
		// Canary-gated exit (#3389): the window elapsing alone does not re-admit
		// the account. Hold it in probation — still cooled, stored Signals/ResetAt
		// untouched so reports can say when the window lapsed — until CanaryExit
		// witnesses one successful round-trip.
		e.Probation = true
		return e, true
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
	// The probation marker is derived read-time state; never write it back. A
	// cleared observation below is a WITNESSED healthy signal, so it may exit
	// probation the same way a canary pass does (#3389).
	beforeEntry.Probation = false
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
// On a canary-armed store (#3389) elapsed entries are in probation, not recovered, so
// they are retained until CanaryExit (or an explicit Clear) releases them.
func (s *CooldownStore) Prune(now time.Time) int {
	n := 0
	for k, e := range s.entries {
		signals, resetAt := activeCooldownSignals(e, now)
		if len(signals) == 0 {
			if s.canaryArmed() {
				// Probation hold (#3389): an elapsed entry is awaiting its canary
				// round-trip, not recovered; dropping it here would silently
				// re-admit the account on the timer alone.
				continue
			}
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
		if e.Active(now) || e.Kind == CooldownOrgAuthWall {
			name := strings.TrimSpace(string(e.Kind))
			if name == "" {
				name = "legacy"
			}
			active[name] = e.ResetAt.UTC()
		}
		return active, latestSignalReset(active)
	}
	for signal, resetAt := range e.Signals {
		if strings.TrimSpace(signal) == "" {
			continue
		}
		// An upstream ORG AUTH WALL never lapses on its own timer (#4998). A usage cap
		// self-recovers, so the window elapsing IS the recovery; an organization with
		// OAuth/subscription access disabled does not repair itself, and re-admitting it
		// because a clock ran out re-dispatches into the same terminal 403. Its deadline
		// means "a reprobe is due" (CooldownEntry.ReprobeDue), not "re-admit" — only a
		// witnessed healthy round-trip clears it (ObserveSeatHealth/ClearOrgAuthWall).
		// Sibling usage/rate signals on the same account keep expiring normally.
		if signal == string(CooldownOrgAuthWall) || now.Before(resetAt) {
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
