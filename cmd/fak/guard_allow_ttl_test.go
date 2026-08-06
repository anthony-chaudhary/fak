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
