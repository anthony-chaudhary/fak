package accounts

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// corruptCooldownFixture writes the exact corruption shape observed on a maintainer box
// (#6027): a complete, valid {schema, entries} object followed by a stray closing brace,
// which json.Unmarshal rejects with "invalid character '}' after top-level value". Returns
// the store path.
func corruptCooldownFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "account-cooldown.json")
	healthy := &CooldownStore{path: path, entries: map[string]CooldownEntry{}}
	now := time.Date(2026, 8, 9, 12, 0, 0, 0, time.UTC)
	healthy.Cool(UUIDBucketKey("u-sink"), CooldownUsageLimit, "weekly limit", now, now.Add(2*time.Hour))
	if err := healthy.Save(); err != nil {
		t.Fatalf("seed healthy store: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read seeded store: %v", err)
	}
	if err := os.WriteFile(path, append(raw, "\n}\n"...), 0o644); err != nil {
		t.Fatalf("corrupt store: %v", err)
	}
	return path
}

// TestLoadCooldownStoreFailOpenWarnsOnCorruptStore is the #6027 witness: a corrupt store
// must keep the REAL admission seam (Registry.ServeAt — what `fak accounts resolve` and
// `fak accounts launch` both call) succeeding, AND must emit one operator-visible warning
// naming the store. Asserting both halves at once is the point: a fix that only warns
// would block the fleet (#4675), and a fix that only folds is the silent gate-off this
// issue files.
func TestLoadCooldownStoreFailOpenWarnsOnCorruptStore(t *testing.T) {
	path := corruptCooldownFixture(t)
	var warn bytes.Buffer

	store := LoadCooldownStoreFailOpen(path, "fak accounts launch", &warn)

	// Half one: the fold is still fail-open — nil store, no error returned to the caller.
	if store != nil {
		t.Fatalf("LoadCooldownStoreFailOpen on a corrupt store = %+v, want the nil cooldown-blind fold (#4675)", store)
	}
	// ...and the real launch/resolve seam still lands on a seat with that nil store.
	r := cooldownServeFixture()
	now := time.Date(2026, 8, 9, 13, 0, 0, 0, time.UTC)
	home, chain, entry, err := r.ServeAt("gone", store, now)
	if err != nil {
		t.Fatalf("ServeAt with a corrupt-store fold: %v — a corrupt cooldown file must never block a launch (#4675)", err)
	}
	if home.Name != "sink" || strings.Join(chain, ",") != "gone" || entry != nil {
		t.Fatalf("ServeAt = %q chain=%v entry=%+v, want sink via [gone] with nil entry (cooldown-blind resolution still serves)",
			home.Name, chain, entry)
	}

	// Half two: the operator is told the gate is off, on the launch path itself.
	got := warn.String()
	if !strings.Contains(got, path) {
		t.Fatalf("warning %q does not name the unreadable store %q", got, path)
	}
	if !strings.Contains(got, "fak accounts launch:") {
		t.Fatalf("warning %q does not name the surface that went cooldown-blind", got)
	}
	if !strings.Contains(got, "invalid character '}' after top-level value") {
		t.Fatalf("warning %q does not carry the underlying parse error", got)
	}
	if !strings.Contains(got, "gate is OFF") {
		t.Fatalf("warning %q does not say the cooldown gate is off", got)
	}
	if n := strings.Count(strings.TrimSpace(got), "\n"); n != 0 {
		t.Fatalf("warning %q spans %d extra lines; it must be one line so it survives a busy launch log", got, n+1)
	}

	// The corrupt file is left in place for the operator the warning just notified
	// (see LoadCooldownStoreFailOpen: quarantining from a READ would let a transient
	// error destroy live fleet state).
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("corrupt store was not left in place: %v", err)
	}
}

// TestLoadCooldownStoreFailOpenSilentWhenHealthy is the control: the warning is reserved
// for a store that is genuinely unreadable. A healthy store loads with its entries intact
// and still GATES (the seam walks past the cooled seat), and an absent store — a fleet
// that has simply never cooled an account — is not an error and must not cry wolf.
func TestLoadCooldownStoreFailOpenSilentWhenHealthy(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "account-cooldown.json")
	now := time.Date(2026, 8, 9, 12, 0, 0, 0, time.UTC)
	seed := &CooldownStore{path: path, entries: map[string]CooldownEntry{}}
	seed.Cool(UUIDBucketKey("u-sink"), CooldownUsageLimit, "weekly limit", now, now.Add(2*time.Hour))
	if err := seed.Save(); err != nil {
		t.Fatalf("seed store: %v", err)
	}

	var warn bytes.Buffer
	store := LoadCooldownStoreFailOpen(path, "fak accounts launch", &warn)
	if store == nil {
		t.Fatal("a healthy store folded to nil — the gate would be off with no corruption at all")
	}
	if warn.Len() != 0 {
		t.Fatalf("healthy store emitted %q, want silence", warn.String())
	}
	r := cooldownServeFixture()
	if home, _, _, err := r.ServeAt("gone", store, now.Add(time.Hour)); err != nil || home.Name != "anchor-seat" {
		t.Fatalf("ServeAt with the healthy store = %q,%v, want anchor-seat (the gate must still skip the cooled sink)", home.Name, err)
	}

	warn.Reset()
	missing := filepath.Join(dir, "nope", "account-cooldown.json")
	if store := LoadCooldownStoreFailOpen(missing, "fak accounts launch", &warn); store == nil {
		t.Fatal("an ABSENT store folded to nil; a fleet that never cooled an account is healthy, not blind")
	}
	if warn.Len() != 0 {
		t.Fatalf("absent store emitted %q, want silence", warn.String())
	}
}

// TestLoadCooldownStoreFailOpenNilWarn pins the fold's independence from the sink: a
// caller with no writer (or no surface label) still gets the fail-open nil rather than a
// panic, so wiring a new admission path can never turn a corrupt store into a crash.
func TestLoadCooldownStoreFailOpenNilWarn(t *testing.T) {
	path := corruptCooldownFixture(t)
	if store := LoadCooldownStoreFailOpen(path, "", nil); store != nil {
		t.Fatalf("LoadCooldownStoreFailOpen(nil warn) = %+v, want the nil fold", store)
	}
	var warn bytes.Buffer
	LoadCooldownStoreFailOpen(path, "   ", &warn)
	if !strings.HasPrefix(warn.String(), "fak accounts:") {
		t.Fatalf("blank surface warning = %q, want the generic `fak accounts:` label", warn.String())
	}
}
