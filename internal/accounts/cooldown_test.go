package accounts

import (
	"path/filepath"
	"testing"
	"time"
)

func mustTime(t *testing.T, s string) time.Time {
	t.Helper()
	ts, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parse %q: %v", s, err)
	}
	return ts
}

func TestCooldownActiveBeforeResetAbsentAfter(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	s.Cool("acct-1", CooldownUsageLimit, "weekly limit", base, time.Time{})

	// Default window is 1h: active at +59m, gone at +61m.
	if _, ok := s.CooledDown("acct-1", base.Add(59*time.Minute)); !ok {
		t.Fatal("expected acct-1 cooled at +59m")
	}
	if _, ok := s.CooledDown("acct-1", base.Add(61*time.Minute)); ok {
		t.Fatal("expected acct-1 NOT cooled at +61m (window elapsed)")
	}
}

func TestCooldownRateLimitShortWindow(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	s.Cool("acct-r", CooldownRateLimit, "429", base, time.Time{})
	if _, ok := s.CooledDown("acct-r", base.Add(4*time.Minute)); !ok {
		t.Fatal("rate-limit should still hold at +4m")
	}
	if _, ok := s.CooledDown("acct-r", base.Add(6*time.Minute)); ok {
		t.Fatal("rate-limit should clear by +6m")
	}
}

func TestCooldownExplicitResetHonored(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	reset := base.Add(3 * time.Hour)
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	e := s.Cool("acct-x", CooldownUsageLimit, "resets at 15:00", base, reset)
	if !e.ResetAt.Equal(reset.UTC()) {
		t.Fatalf("explicit reset not honored: got %s want %s", e.ResetAt, reset.UTC())
	}
	if _, ok := s.CooledDown("acct-x", base.Add(2*time.Hour)); !ok {
		t.Fatal("should hold to explicit 3h reset")
	}
}

func TestCoolNeverShortensLongerWindow(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	// Weekly usage cap: long window.
	s.Cool("acct-1", CooldownUsageLimit, "weekly", base, base.Add(6*time.Hour))
	// A later transient 429 must not shrink it to 5 minutes.
	e := s.Cool("acct-1", CooldownRateLimit, "429", base.Add(time.Minute), time.Time{})
	if e.ResetAt.Before(base.Add(6 * time.Hour)) {
		t.Fatalf("transient 429 shortened the usage-cap window: reset=%s", e.ResetAt)
	}
}

func TestCooldownClearAndPrune(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	s.Cool("a", CooldownUsageLimit, "", base, base.Add(time.Hour))
	s.Cool("b", CooldownUsageLimit, "", base, base.Add(time.Hour))

	if !s.Clear("a") {
		t.Fatal("Clear(a) should report an entry was removed")
	}
	if s.Clear("a") {
		t.Fatal("second Clear(a) should report nothing removed")
	}
	// b elapsed → pruned.
	if n := s.Prune(base.Add(2 * time.Hour)); n != 1 {
		t.Fatalf("Prune should drop 1 elapsed entry, dropped %d", n)
	}
	if len(s.entries) != 0 {
		t.Fatalf("store should be empty, has %d", len(s.entries))
	}
}

func TestCooldownEmptyAccountNeverCooled(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	if _, ok := s.CooledDown("", base); ok {
		t.Fatal("empty account must never be cooled")
	}
}

func TestCooldownRoundTripsThroughDisk(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	path := filepath.Join(t.TempDir(), "nested", "cooldown.json")

	s, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("load empty: %v", err)
	}
	s.Cool("acct-1", CooldownUsageLimit, "weekly limit", base, base.Add(time.Hour))
	if err := s.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	reloaded, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	e, ok := reloaded.CooledDown("acct-1", base.Add(30*time.Minute))
	if !ok {
		t.Fatal("reloaded store lost the active cooldown")
	}
	if e.Reason != "weekly limit" || e.Kind != CooldownUsageLimit {
		t.Fatalf("reloaded entry corrupted: %+v", e)
	}
}

func TestLoadMissingFileIsEmptyNotError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "does-not-exist.json")
	s, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("missing file must not error: %v", err)
	}
	if len(s.Active(mustTime(t, "2026-07-06T12:00:00Z"))) != 0 {
		t.Fatal("missing file should yield empty store")
	}
}

func TestCooldownOverloadLatchClearsOnlyAfterEverySignalAndRepublishesOnChange(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	reg := Registry{Homes: []Home{active("latched-seat", "acct-latched", "latched@example.test")}}
	account := UUIDBucketKey("acct-latched")
	canServe := func() int { return reg.LoginReportAt(s, base.Add(time.Minute)).Summary.CanServe }
	publishes := 0
	update := func(signal string, overloaded bool) {
		_, changed := s.UpdateOverload(account, signal, CooldownRateLimit, overloaded, signal, base, base.Add(time.Hour))
		if changed {
			publishes++
		}
	}

	update("kv_used_blocks", true) // first crossing latches out and republishes
	update("kv_used_blocks", true) // same state is change-gated
	update("active_decode_blocks", true)
	update("kv_used_blocks", false) // one signal cleared; decode still holds the latch
	if _, ok := s.CooledDown(account, base.Add(time.Minute)); !ok {
		t.Fatal("account flapped back into the servable pool while decode remained overloaded")
	}
	if got := canServe(); got != 0 {
		t.Fatalf("servable pool readmitted %d seats while an overload signal remained", got)
	}
	update("kv_used_blocks", true)  // oscillation around one threshold
	update("kv_used_blocks", false) // still held by decode, so neither edge republishes
	if publishes != 1 {
		t.Fatalf("intermediate oscillation published %d times, want only the initial latch transition", publishes)
	}
	update("active_decode_blocks", false) // every signal clear: unlatch + one republish
	if _, ok := s.CooledDown(account, base.Add(time.Minute)); ok {
		t.Fatal("account stayed latched after every overload signal cleared")
	}
	if got := canServe(); got != 1 {
		t.Fatalf("servable pool has %d seats after every signal cleared, want 1", got)
	}
	if publishes != 2 {
		t.Fatalf("pool published %d times, want exactly latch + unlatch transitions", publishes)
	}
}

func TestCooldownOverloadLatchSignalsSurviveDiskRoundTrip(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	path := filepath.Join(t.TempDir(), "cooldown.json")
	s, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatal(err)
	}
	s.UpdateOverload("acct-latched", "kv", CooldownUsageLimit, true, "kv high", base, base.Add(time.Hour))
	s.UpdateOverload("acct-latched", "decode", CooldownUsageLimit, true, "decode high", base, base.Add(2*time.Hour))
	if err := s.Save(); err != nil {
		t.Fatal(err)
	}

	reloaded, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, changed := reloaded.UpdateOverload("acct-latched", "kv", CooldownUsageLimit, false, "", base.Add(time.Minute), time.Time{}); changed {
		t.Fatal("clearing one reloaded signal must not republish while another remains active")
	}
	if _, ok := reloaded.CooledDown("acct-latched", base.Add(90*time.Minute)); !ok {
		t.Fatal("reloaded decode signal did not keep the account latched")
	}
	if _, changed := reloaded.UpdateOverload("acct-latched", "decode", CooldownUsageLimit, false, "", base.Add(90*time.Minute), time.Time{}); !changed {
		t.Fatal("clearing the last reloaded signal must publish the unlatch transition")
	}
}

// TestFleetDegradedEngageOnFirstRestoreOnLast walks the full fleet degrade
// cycle (#3383): all-healthy is not degraded; the FIRST account into cooldown
// engages; a second account down and a partial recovery both leave the flag in
// its stable state; only the LAST recovery restores.
func TestFleetDegradedEngageOnFirstRestoreOnLast(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}

	// (a) empty store, all healthy: not degraded.
	if s.Degraded() {
		t.Fatal("empty store must not be degraded")
	}
	if !s.DegradedSince().IsZero() {
		t.Fatalf("healthy fleet has DegradedSince=%s, want zero", s.DegradedSince())
	}

	// (b) first account down: engage edge.
	s.Cool("acct-1", CooldownUsageLimit, "weekly", base, base.Add(time.Hour))
	if !s.Degraded() {
		t.Fatal("first account into cooldown must engage degraded mode")
	}
	if !s.DegradedSince().Equal(base) {
		t.Fatalf("DegradedSince=%s, want the first-down observation %s", s.DegradedSince(), base)
	}

	// (c) second account down: still degraded, no re-engage (marker unchanged).
	s.Cool("acct-2", CooldownUsageLimit, "weekly", base.Add(time.Minute), base.Add(time.Hour))
	if !s.Degraded() {
		t.Fatal("second account down must leave the fleet degraded")
	}
	if !s.DegradedSince().Equal(base) {
		t.Fatalf("second account down re-engaged: DegradedSince=%s, want %s", s.DegradedSince(), base)
	}

	// (d) one of two recovers: others still down, so still degraded.
	if !s.Clear("acct-1") {
		t.Fatal("Clear(acct-1) should remove an entry")
	}
	if !s.Degraded() {
		t.Fatal("fleet must stay degraded while acct-2 remains cooled")
	}

	// (e) the LAST recovers: restore edge.
	if !s.Clear("acct-2") {
		t.Fatal("Clear(acct-2) should remove an entry")
	}
	if s.Degraded() {
		t.Fatal("last recovery must restore the fleet to normal mode")
	}
	if !s.DegradedSince().IsZero() {
		t.Fatalf("restored fleet has DegradedSince=%s, want zero", s.DegradedSince())
	}
}

// TestFleetDegradedIdempotentReCool: (f) re-cooling an already-cooled account is
// not a 0->1 edge — the flag stays engaged with its original marker — and the
// signal-clear recovery path (UpdateOverload overloaded=false) drives the same
// restore edge Clear does.
func TestFleetDegradedIdempotentReCool(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}

	s.Cool("acct-1", CooldownUsageLimit, "weekly", base, base.Add(time.Hour))
	since := s.DegradedSince()
	s.Cool("acct-1", CooldownUsageLimit, "weekly again", base.Add(10*time.Minute), base.Add(2*time.Hour))
	if !s.Degraded() {
		t.Fatal("re-cooling the same account must keep the fleet degraded")
	}
	if !s.DegradedSince().Equal(since) {
		t.Fatalf("re-cool double-engaged: DegradedSince=%s, want original %s", s.DegradedSince(), since)
	}

	// Cool folds through UpdateOverload with signal = string(kind); clearing that
	// last signal empties the account's latch and restores the fleet.
	s.UpdateOverload("acct-1", string(CooldownUsageLimit), CooldownUsageLimit, false, "", base.Add(20*time.Minute), time.Time{})
	if s.Degraded() {
		t.Fatal("clearing the last signal of the last account must restore the fleet")
	}
}

// TestFleetDegradedRestoresViaPrune: an elapsed window keeps the fleet degraded
// until the store OBSERVES the recovery — the same lazy-expiry discipline the
// entries map follows — and Prune is that observation point.
func TestFleetDegradedRestoresViaPrune(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	s := &CooldownStore{entries: map[string]CooldownEntry{}}
	s.Cool("acct-1", CooldownUsageLimit, "weekly", base, base.Add(time.Hour))

	// Window elapsed but not yet observed: membership-driven flag still holds.
	if !s.Degraded() {
		t.Fatal("fleet must stay degraded until the expiry is observed")
	}
	if n := s.Prune(base.Add(2 * time.Hour)); n != 1 {
		t.Fatalf("Prune dropped %d entries, want 1", n)
	}
	if s.Degraded() {
		t.Fatal("Prune of the last cooled account must restore the fleet")
	}
}

// TestFleetDegradedSurvivesDiskRoundTrip: the flag is derived, not persisted —
// a reload re-engages from the entries on disk with the earliest CooledAt as
// the engaged-since marker, and an empty file loads not-degraded.
func TestFleetDegradedSurvivesDiskRoundTrip(t *testing.T) {
	base := mustTime(t, "2026-07-06T12:00:00Z")
	path := filepath.Join(t.TempDir(), "cooldown.json")

	s, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("load empty: %v", err)
	}
	if s.Degraded() {
		t.Fatal("missing file must load not-degraded")
	}
	s.Cool("acct-2", CooldownUsageLimit, "weekly", base.Add(time.Minute), base.Add(time.Hour))
	s.Cool("acct-1", CooldownUsageLimit, "weekly", base, base.Add(time.Hour))
	if err := s.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	reloaded, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if !reloaded.Degraded() {
		t.Fatal("reloaded store with active cooldowns must be degraded")
	}
	if !reloaded.DegradedSince().Equal(base) {
		t.Fatalf("reloaded DegradedSince=%s, want earliest CooledAt %s", reloaded.DegradedSince(), base)
	}
}

// TestParseResetAbsolute: an explicit RFC3339 reset in the message is parsed to that instant
// (UTC). ParseReset is the single source both cooldown writers share, so this pins its contract
// in one place instead of once per writer.
func TestParseResetAbsolute(t *testing.T) {
	got := ParseReset("usage limit reached; resets at 2026-07-07T15:00:00Z, try later")
	want := mustTime(t, "2026-07-07T15:00:00Z")
	if !got.Equal(want) {
		t.Fatalf("ParseReset absolute: got %s want %s", got, want)
	}
	if got.Location() != time.UTC {
		t.Fatalf("ParseReset must normalize to UTC, got %s", got.Location())
	}
}

// TestParseResetVagueOrAbsentIsZero: a date-less "in 42 minutes" phrasing and a message with no
// reset language both yield the zero time — the caller then falls back to the kind's default
// window rather than a mis-parsed wall-clock.
func TestParseResetVagueOrAbsentIsZero(t *testing.T) {
	if got := ParseReset("session limit; resets in 42 minutes"); !got.IsZero() {
		t.Fatalf("vague reset should be zero, got %s", got)
	}
	if got := ParseReset("no limit language here"); !got.IsZero() {
		t.Fatalf("absent reset should be zero, got %s", got)
	}
	// A bare wall-clock without a date must NOT be guessed as absolute.
	if got := ParseReset("weekly limit reached; resets at 15:00"); !got.IsZero() {
		t.Fatalf("date-less wall-clock should be zero, got %s", got)
	}
}
