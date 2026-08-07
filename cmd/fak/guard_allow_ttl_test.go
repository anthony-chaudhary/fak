package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// guard_allow_ttl_test.go — #5179 (epic #5170 Track D): expiring operator widenings via
// `fak guard allow --ttl`. The launch-boundary contract is: an entry whose recorded
// expiry has passed is auto-reverted (dropped from the enforced floor AND from
// `fak guard allow --list`) on the next `fak guard` launch, while a non-expired entry is
// retained. Every read funnels through loadGuardAllowOverlay, so testing that loader (and
// the real launch read, loadGuardAllowOverlayLayers) covers both surfaces at once.

func guardAllowNamePresent(list []string, name string) bool {
	for _, e := range list {
		if e == name {
			return true
		}
	}
	return false
}

// TestGuardAllowTTLExpiredDroppedNonExpiredRetained is the done-condition witness: an
// entry with an expiry in the PAST is dropped at the next launch; one with an expiry in
// the FUTURE (as `--ttl 1h` would write) is retained. The overlay is written to disk
// verbatim so the test controls the exact stamps, then read back through the loader.
func TestGuardAllowTTLExpiredDroppedNonExpiredRetained(t *testing.T) {
	path := filepath.Join(t.TempDir(), "allow.json")
	future := guardAllowExpiryStamp(time.Now().Add(time.Hour))
	body := `{
  "version": "` + guardAllowOverlayVersion + `",
  "allow": ["expired_tool", "live_tool"],
  "expiry": {
    "expired_tool": "2000-01-01T00:00:00Z",
    "live_tool": "` + future + `"
  }
}` + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if guardAllowNamePresent(got.Allow, "expired_tool") {
		t.Errorf("expired entry must be dropped at launch, still present: %v", got.Allow)
	}
	if !guardAllowNamePresent(got.Allow, "live_tool") {
		t.Errorf("non-expired entry must be retained, missing: %v", got.Allow)
	}
	if _, ok := got.Expiry["expired_tool"]; ok {
		t.Errorf("expired entry's stale stamp must be dropped, still present: %v", got.Expiry)
	}
	if got.Expiry["live_tool"] != future {
		t.Errorf("live entry's stamp must survive, got %q want %q", got.Expiry["live_tool"], future)
	}
}

// TestGuardAllowTTLListOmitsExpired is the second half of the witness: the expired entry
// no longer appears in the `fak guard allow --list` rendering (which loads then prints).
func TestGuardAllowTTLListOmitsExpired(t *testing.T) {
	path := filepath.Join(t.TempDir(), "allow.json")
	future := guardAllowExpiryStamp(time.Now().Add(time.Hour))
	body := `{"allow":["expired_tool","live_tool"],"expiry":{"expired_tool":"2000-01-01T00:00:00Z","live_tool":"` + future + `"}}` + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	ov, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	var out bytes.Buffer
	printGuardAllowOverlay(&out, path, ov)
	s := out.String()
	if strings.Contains(s, "expired_tool") {
		t.Errorf("--list must not show the expired entry:\n%s", s)
	}
	if !strings.Contains(s, "live_tool") {
		t.Errorf("--list must still show the live entry:\n%s", s)
	}
	if !strings.Contains(s, "expires") {
		t.Errorf("--list should surface the remaining TTL of the live entry:\n%s", s)
	}
}

// TestGuardAllowTTLLaunchReadDropsExpired proves the drop happens on the REAL launch read
// path (loadGuardAllowOverlayLayers, the one loadGuardCapabilityFloor folds into the
// enforced floor), not only the single-file loader. The env override makes the temp file
// the sole base layer.
func TestGuardAllowTTLLaunchReadDropsExpired(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "allow.json")
	future := guardAllowExpiryStamp(time.Now().Add(time.Hour))
	body := `{"allow":["expired_tool","live_tool"],"expiry":{"expired_tool":"2000-01-01T00:00:00Z","live_tool":"` + future + `"}}` + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(guardAllowOverlayEnv, path)

	merged, _, err := loadGuardAllowOverlayLayers()
	if err != nil {
		t.Fatalf("launch read: %v", err)
	}
	if guardAllowNamePresent(merged.Allow, "expired_tool") {
		t.Errorf("launch floor must not union the expired entry: %v", merged.Allow)
	}
	if !guardAllowNamePresent(merged.Allow, "live_tool") {
		t.Errorf("launch floor must union the live entry: %v", merged.Allow)
	}
}

// TestGuardAllowDropExpired unit-checks the classifier: a past stamp drops, a future stamp
// stays, an entry with NO stamp (permanent) stays, and an UNPARSEABLE stamp is kept
// (fail-safe — a malformed timestamp must never silently revoke a widening).
func TestGuardAllowDropExpired(t *testing.T) {
	now := time.Date(2026, 8, 3, 12, 0, 0, 0, time.UTC)
	ov := guardAllowOverlay{
		Allow:       []string{"past", "future", "permanent", "garbled"},
		AllowPrefix: []string{"past_"},
		Expiry: map[string]string{
			"past":    "2020-01-01T00:00:00Z",
			"past_":   "2020-01-01T00:00:00Z",
			"future":  "2030-01-01T00:00:00Z",
			"garbled": "not-a-timestamp",
		},
	}
	got, dropped := guardAllowDropExpired(ov, now)
	if strings.Join(dropped, ",") != "past,past_" {
		t.Errorf("dropped = %v, want [past past_] (sorted)", dropped)
	}
	if guardAllowNamePresent(got.Allow, "past") || guardAllowNamePresent(got.AllowPrefix, "past_") {
		t.Errorf("expired entries not dropped: %+v", got)
	}
	for _, keep := range []string{"future", "permanent", "garbled"} {
		if !guardAllowNamePresent(got.Allow, keep) {
			t.Errorf("%q must be retained, got %v", keep, got.Allow)
		}
	}
	if _, ok := got.Expiry["past"]; ok {
		t.Error("expired stamp must be dropped from the map")
	}
	if got.Expiry["garbled"] != "not-a-timestamp" {
		t.Error("unparseable stamp must be retained")
	}
}

// TestGuardAllowSaveRoundTripsExpiryAndPrunesOrphans: a save preserves live stamps and
// drops a stamp that no longer names any Allow / AllowPrefix entry (e.g. after a remove).
func TestGuardAllowSaveRoundTripsExpiryAndPrunesOrphans(t *testing.T) {
	path := filepath.Join(t.TempDir(), "allow.json")
	future := guardAllowExpiryStamp(time.Now().Add(time.Hour))
	in := guardAllowOverlay{
		Allow: []string{"kept"},
		Expiry: map[string]string{
			"kept":   future,
			"orphan": future, // no matching Allow/AllowPrefix entry — must be pruned
		},
	}
	if err := saveGuardAllowOverlay(path, in); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got.Expiry["kept"] != future {
		t.Errorf("live stamp not round-tripped: %v", got.Expiry)
	}
	if _, ok := got.Expiry["orphan"]; ok {
		t.Errorf("orphan stamp must be pruned on save: %v", got.Expiry)
	}
}

// guardAllowPinClock pins guardAllowNow for the duration of the test, so every TTL
// decision under test (the read-path drop, the `--list` render, the `--ttl` stamp) is
// evaluated at a chosen instant instead of the wall clock.
func guardAllowPinClock(t *testing.T, at time.Time) func(time.Time) {
	t.Helper()
	saved := guardAllowNow
	t.Cleanup(func() { guardAllowNow = saved })
	cur := at
	guardAllowNow = func() time.Time { return cur }
	return func(to time.Time) { cur = to }
}

// TestGuardAllowTTLBoundaryIsCrossedAtRead pins the load-bearing claim the earlier
// witnesses could only assert one side of: the SAME on-disk overlay is honored before its
// expiry and refused at/after it, with nothing between the two reads but the clock. That
// is what "the TTL is enforced" means operationally — the guard does not consult a
// sweeper, a mtime, or a rewrite of the file; it re-decides every read against now.
//
// The boundary is asserted as inclusive (an entry is gone AT its stamp, not one tick
// later), because an off-by-one there is the difference between a widening that ends when
// the operator was told it ends and one that outlives its window. Pinning the clock is
// what makes this checkable at all: with the wall clock the only stamps a test can use are
// so far past or future that the crossing itself is never exercised, and any attempt to
// actually wait for a window is flaky on a loaded shared runner.
func TestGuardAllowTTLBoundaryIsCrossedAtRead(t *testing.T) {
	expiresAt := time.Date(2026, 8, 3, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "allow.json")
	body := `{"allow":["temp_tool","permanent_tool"],"expiry":{"temp_tool":"` +
		guardAllowExpiryStamp(expiresAt) + `"}}` + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	setClock := guardAllowPinClock(t, expiresAt.Add(-time.Second))
	ov, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatalf("load (inside window): %v", err)
	}
	if !guardAllowNamePresent(ov.Allow, "temp_tool") {
		t.Errorf("one second BEFORE its expiry the entry must still be honored: %v", ov.Allow)
	}

	// Exactly at the stamp: the window is over. `now >= expiry` drops.
	setClock(expiresAt)
	if ov, err = loadGuardAllowOverlay(path); err != nil {
		t.Fatalf("load (at the boundary): %v", err)
	}
	if guardAllowNamePresent(ov.Allow, "temp_tool") {
		t.Errorf("AT its expiry the entry must be dropped, still honored: %v", ov.Allow)
	}

	setClock(expiresAt.Add(24 * time.Hour))
	if ov, err = loadGuardAllowOverlay(path); err != nil {
		t.Fatalf("load (past the window): %v", err)
	}
	if guardAllowNamePresent(ov.Allow, "temp_tool") {
		t.Errorf("past its expiry the entry must stay dropped: %v", ov.Allow)
	}
	if !guardAllowNamePresent(ov.Allow, "permanent_tool") {
		t.Errorf("an entry with no stamp must survive every crossing: %v", ov.Allow)
	}

	// READ-time, not a sweeper: three loads across the boundary and the file on disk is
	// still byte-for-byte what the operator wrote. A guard that rewrote the overlay to
	// enforce a TTL would revoke the widening for every OTHER reader too — including a
	// concurrent session whose own clock has not reached the window yet — and would turn a
	// read-only surface (`--list`) into a mutation.
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Errorf("expiry must be evaluated at READ time, not by rewriting the overlay:\nbefore: %s\nafter:  %s", before, after)
	}
}

// TestGuardAllowTTLZeroMeansPermanent pins the DEFAULT half of the contract: `--ttl 0`
// (i.e. the flag not passed at all) is "no expiry", never "expires immediately". A zero
// window that stamped now+0 would make every ordinary `fak guard allow <tool>` a widening
// the next launch silently revokes — the loudest possible regression, and one no other
// test in this file would catch because they all supply an explicit stamp.
//
// It also pins the promotion path the flag help promises: re-adding an entry with no
// --ttl CLEARS the stamp it carried, so a "just for now" widening becomes permanent
// again rather than keeping its old, now-invisible deadline.
func TestGuardAllowTTLZeroMeansPermanent(t *testing.T) {
	now := time.Date(2026, 8, 3, 12, 0, 0, 0, time.UTC)
	ov := guardAllowOverlay{Allow: []string{"tool_a"}}

	if stamp := guardAllowStampExpiry(&ov, []string{"tool_a"}, 0, now); stamp != "" {
		t.Errorf("--ttl 0 must record NO stamp, got %q", stamp)
	}
	if len(ov.Expiry) != 0 {
		t.Errorf("--ttl 0 must leave the expiry map empty, got %v", ov.Expiry)
	}
	// The permanence has to hold at the read path too, arbitrarily far in the future.
	if got, dropped := guardAllowDropExpired(ov, now.AddDate(100, 0, 0)); len(dropped) != 0 ||
		!guardAllowNamePresent(got.Allow, "tool_a") {
		t.Errorf("an unstamped entry must never expire; dropped=%v allow=%v", dropped, got.Allow)
	}

	// A positive window stamps exactly now+ttl off the injected clock...
	stamp := guardAllowStampExpiry(&ov, []string{"tool_a"}, 90*time.Minute, now)
	if want := guardAllowExpiryStamp(now.Add(90 * time.Minute)); stamp != want {
		t.Errorf("--ttl 90m stamped %q, want %q", stamp, want)
	}
	if ov.Expiry["tool_a"] != stamp {
		t.Errorf("the stamp on the entry (%q) must be the one reported to the operator (%q)", ov.Expiry["tool_a"], stamp)
	}

	// ...and a re-add with no --ttl promotes it back to permanent.
	if stamp := guardAllowStampExpiry(&ov, []string{"tool_a"}, 0, now); stamp != "" {
		t.Errorf("a re-add with no --ttl must report no expiry, got %q", stamp)
	}
	if _, ok := ov.Expiry["tool_a"]; ok {
		t.Errorf("a re-add with no --ttl must clear the prior stamp: %v", ov.Expiry)
	}
}
