package guardrotate

import (
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// grHome builds a live, serveable Claude seat at dir logged into account uuid — the minimal
// shape Plan/NextRotationDecision/Serve read (status active, disk identity with creds). No
// disk I/O: the plan reads these fields directly.
func grHome(name, dir, uuid string) accounts.Home {
	return accounts.Home{
		Name:     name,
		Dir:      dir,
		Status:   accounts.StatusActive,
		Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: uuid, Email: name + "@x.test"},
	}
}

func grMustTime(t *testing.T, s string) time.Time {
	t.Helper()
	ts, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("parse time %q: %v", s, err)
	}
	return ts
}

// grStore builds an in-memory cooldown store (bound to a temp path so a stray Save can't
// touch a real fleet file) with the given account keys cooled until reset.
func grStore(t *testing.T, cooledAt, resetAt time.Time, keys ...string) *accounts.CooldownStore {
	t.Helper()
	s, err := accounts.LoadCooldownStore(filepath.Join(t.TempDir(), "cd.json"))
	if err != nil {
		t.Fatalf("load cooldown store: %v", err)
	}
	for _, k := range keys {
		s.Cool(k, accounts.CooldownUsageLimit, "weekly", cooledAt, resetAt)
	}
	return s
}

// TestPlanRotatesOffCooledSeat is the load-bearing behavior: the ambient dir maps to a
// cooled account and a live alternate with open room exists -> Plan returns the alternate's
// dir and a note naming both seats. This is the "walled account must not spawn; the roomy
// one must" case from the request.
func TestPlanRotatesOffCooledSeat(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	reset := now.Add(2 * time.Hour)
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", "/home/.claude", "u-alice"),
		grHome("bob", "/home/.claude-bob", "u-bob"),
	}} // fixtures carry full in-memory identities; no .Refresh() (it would re-derive from a non-existent test disk and wipe them)
	// alice's bucket is cooled; the ambient CLAUDE_CONFIG_DIR points at alice's dir.
	store := grStore(t, now, reset, "uuid:u-alice")
	// Headroom mirrors what rotationHeadroom would produce: alice walled by cooldown, bob free.
	hr := accounts.RotationHeadroom{"uuid:u-alice": -1, "uuid:u-bob": 1.5}

	dir, note, ok := Plan(reg, store, hr, "/home/.claude", now)
	if !ok {
		t.Fatal("expected a rotation off the cooled seat, got ok=false")
	}
	if got := guardWantDir(dir); got != guardWantDir("/home/.claude-bob") {
		t.Fatalf("rotated dir = %q, want the bob seat dir", dir)
	}
	if note.From != "alice" || note.To != "bob" {
		t.Fatalf("note = %+v, want From=alice To=bob", note)
	}
	if !note.ResetAt.Equal(reset) {
		t.Fatalf("note.ResetAt = %v, want %v", note.ResetAt, reset)
	}
	if note.Headroom == nil || *note.Headroom != 1.5 {
		t.Fatalf("note.Headroom = %v, want 1.5 (bob's open-room score)", note.Headroom)
	}
}

// TestPlanCooledButAlternateOnlyUnknownFailsOpen is the room guarantee: the ambient seat is
// cooled, and an alternate exists that is NOT known-walled but has UNKNOWN headroom (0 — no
// runtime telemetry). We must NOT rotate onto it: swapping a provably-cooled seat for an
// unmeasured one that might also be capped is no improvement. Fail open, keep the current seat.
func TestPlanCooledButAlternateOnlyUnknownFailsOpen(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", "/home/.claude", "u-alice"),
		grHome("bob", "/home/.claude-bob", "u-bob"),
	}}
	store := grStore(t, now, now.Add(time.Hour), "uuid:u-alice")
	// alice walled by cooldown; bob is UNKNOWN (0) — present so headroom mode is on, but no
	// positive room signal. NextRotationDecision would return bob (not walled), but Plan must
	// reject it for lack of proven room.
	hr := accounts.RotationHeadroom{"uuid:u-alice": -1, "uuid:u-bob": 0}

	dir, _, ok := Plan(reg, store, hr, "/home/.claude", now)
	if ok {
		t.Fatalf("must NOT rotate onto an UNKNOWN-headroom seat, got ok=true dir=%q", dir)
	}
	if dir != "/home/.claude" {
		t.Fatalf("fail-open must keep the original dir, got %q", dir)
	}
}

// TestPlanPicksRoomiestAmongOfferable proves that when several alternates have real room, the
// rotation lands on the one with the MOST room (highest positive headroom), since the pool is
// ordered most-headroom-first and Plan takes the first offerable non-anchor seat.
func TestPlanPicksRoomiestAmongOfferable(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", "/home/.claude", "u-alice"),
		grHome("bob", "/home/.claude-bob", "u-bob"),
		grHome("carol", "/home/.claude-carol", "u-carol"),
	}}
	store := grStore(t, now, now.Add(time.Hour), "uuid:u-alice")
	// alice cooled; bob has a little room (1.2), carol has more (1.9). Must land on carol.
	hr := accounts.RotationHeadroom{"uuid:u-alice": -1, "uuid:u-bob": 1.2, "uuid:u-carol": 1.9}

	dir, note, ok := Plan(reg, store, hr, "/home/.claude", now)
	if !ok {
		t.Fatal("expected a rotation onto a roomy seat, got ok=false")
	}
	if note.To != "carol" {
		t.Fatalf("must rotate onto the roomiest seat, got To=%q want carol", note.To)
	}
	if guardWantDir(dir) != guardWantDir("/home/.claude-carol") {
		t.Fatalf("rotated dir = %q, want carol's dir", dir)
	}
}

// TestPlanNotCooledIsNoop is the warm common case — the ambient seat's account is not
// cooled, so Plan is a no-op and the caller keeps the original dir. Zero behavior change for
// every normal launch.
func TestPlanNotCooledIsNoop(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", "/home/.claude", "u-alice"),
		grHome("bob", "/home/.claude-bob", "u-bob"),
	}} // fixtures carry full in-memory identities; no .Refresh() (it would re-derive from a non-existent test disk and wipe them)
	store := grStore(t, now, now.Add(time.Hour)) // nothing cooled
	hr := accounts.RotationHeadroom{"uuid:u-alice": 1.9, "uuid:u-bob": 1.5}

	dir, _, ok := Plan(reg, store, hr, "/home/.claude", now)
	if ok {
		t.Fatalf("expected no rotation for a warm seat, got ok=true dir=%q", dir)
	}
	if dir != "/home/.claude" {
		t.Fatalf("no-op must keep the original dir, got %q", dir)
	}
}

// TestPlanCooledButNoAlternateFailsOpen: the ambient seat is cooled but it is the ONLY
// account bucket (no live alternate) -> NextRotationDecision returns !OK and Plan fails open
// to the original dir rather than blocking the launch.
func TestPlanCooledButNoAlternateFailsOpen(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", "/home/.claude", "u-alice"),
	}} // fixtures carry full in-memory identities; no .Refresh() (it would re-derive from a non-existent test disk and wipe them)
	store := grStore(t, now, now.Add(time.Hour), "uuid:u-alice")
	hr := accounts.RotationHeadroom{"uuid:u-alice": -1}

	dir, _, ok := Plan(reg, store, hr, "/home/.claude", now)
	if ok {
		t.Fatalf("cooled sole-bucket must fail open, got ok=true dir=%q", dir)
	}
	if dir != "/home/.claude" {
		t.Fatalf("fail-open must keep the original dir, got %q", dir)
	}
}

// TestPlanCooledAllOthersWalledFailsOpen: the ambient seat is cooled AND every other bucket
// is walled too (a fleet-wide cap) -> no live target, fail open.
func TestPlanCooledAllOthersWalledFailsOpen(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", "/home/.claude", "u-alice"),
		grHome("bob", "/home/.claude-bob", "u-bob"),
	}} // fixtures carry full in-memory identities; no .Refresh() (it would re-derive from a non-existent test disk and wipe them)
	store := grStore(t, now, now.Add(time.Hour), "uuid:u-alice", "uuid:u-bob")
	hr := accounts.RotationHeadroom{"uuid:u-alice": -1, "uuid:u-bob": -1}

	dir, _, ok := Plan(reg, store, hr, "/home/.claude", now)
	if ok {
		t.Fatalf("cooled with all-others-walled must fail open, got ok=true dir=%q", dir)
	}
	if dir != "/home/.claude" {
		t.Fatalf("fail-open must keep the original dir, got %q", dir)
	}
}

// TestPlanUnenrolledDirFailsOpen: the ambient CLAUDE_CONFIG_DIR is not an enrolled seat
// (e.g. a bespoke dir) -> not ours to rotate, keep it untouched.
func TestPlanUnenrolledDirFailsOpen(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", "/home/.claude", "u-alice"),
		grHome("bob", "/home/.claude-bob", "u-bob"),
	}} // fixtures carry full in-memory identities; no .Refresh() (it would re-derive from a non-existent test disk and wipe them)
	store := grStore(t, now, now.Add(time.Hour), "uuid:u-alice")
	hr := accounts.RotationHeadroom{"uuid:u-alice": -1, "uuid:u-bob": 1.5}

	dir, _, ok := Plan(reg, store, hr, "/home/.claude-bespoke", now)
	if ok {
		t.Fatalf("an unenrolled dir must fail open, got ok=true dir=%q", dir)
	}
	if dir != "/home/.claude-bespoke" {
		t.Fatalf("fail-open must keep the original dir, got %q", dir)
	}
}

// TestPlanNilStoreFailsOpen: a nil cooldown store (an unreadable/absent store the wrapper
// turned into nil) is a valid zero-signal input — no rotation.
func TestPlanNilStoreFailsOpen(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", "/home/.claude", "u-alice"),
	}} // fixtures carry full in-memory identities; no .Refresh() (it would re-derive from a non-existent test disk and wipe them)
	dir, _, ok := Plan(reg, nil, nil, "/home/.claude", now)
	if ok {
		t.Fatalf("nil store must fail open, got ok=true dir=%q", dir)
	}
	if dir != "/home/.claude" {
		t.Fatalf("fail-open must keep the original dir, got %q", dir)
	}
}

// TestHomeForDirMatchesNormalized: the dir match tolerates trailing-separator and (on
// Windows) case differences between the registry's stored dir and the ambient env var,
// since they can name the same directory in different spellings.
func TestHomeForDirMatchesNormalized(t *testing.T) {
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", filepath.FromSlash("/home/user/.claude"), "u-alice"),
	}}
	// Same dir, trailing separator: must still resolve to alice.
	got, ok := HomeForDir(reg, filepath.FromSlash("/home/user/.claude")+string(filepath.Separator))
	if !ok || got.Name != "alice" {
		t.Fatalf("trailing-sep dir: got (%q, %v), want (alice, true)", got.Name, ok)
	}
	if filepath.Separator == '\\' {
		// Windows: case-insensitive path equality.
		up := strings.ToUpper(filepath.FromSlash("/home/user/.claude"))
		got, ok := HomeForDir(reg, up)
		if !ok || got.Name != "alice" {
			t.Fatalf("upper-case dir on Windows: got (%q, %v), want (alice, true)", got.Name, ok)
		}
	}
}

// TestHomeForDirUnknownDir: a dir naming no enrolled seat reports ok=false.
func TestHomeForDirUnknownDir(t *testing.T) {
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", filepath.FromSlash("/home/user/.claude"), "u-alice"),
	}}
	if _, ok := HomeForDir(reg, filepath.FromSlash("/home/user/.claude-other")); ok {
		t.Fatal("unknown dir must report ok=false")
	}
}

// TestPersistCooldownForRehomeWritesUsageCap: a live 429 account-cap rehome (isAccountCap true)
// records a self-recovering usage-limit cooldown keyed on the walled account, honoring an
// explicit reset when the reason names one.
func TestPersistCooldownForRehomeWritesUsageCap(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	store := grStore(t, now, now) // empty store
	entry, wrote := PersistCooldownForRehome(store, "uuid:u-alice", "resets at 2026-07-07T17:00:00Z", "resets at 2026-07-07T17:00:00Z", true, now)
	if !wrote {
		t.Fatal("a 429 account cap must persist a cooldown")
	}
	if entry.Kind != accounts.CooldownUsageLimit {
		t.Fatalf("kind = %q, want usage-limit", entry.Kind)
	}
	want := grMustTime(t, "2026-07-07T17:00:00Z")
	if !entry.ResetAt.Equal(want) {
		t.Fatalf("reset = %v, want explicit %v", entry.ResetAt, want)
	}
	// And it is now readable back as an active cooldown.
	if _, ok := store.CooledDown("uuid:u-alice", now); !ok {
		t.Fatal("the walled account must read as cooled after persist")
	}
}

// TestPersistCooldownForRehomeDefaultWindow: with no explicit reset in the reason, the default
// usage-limit window (1h) is applied.
func TestPersistCooldownForRehomeDefaultWindow(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	store := grStore(t, now, now)
	entry, wrote := PersistCooldownForRehome(store, "uuid:u-alice", "rehomed_seat", "rehomed_seat", true, now)
	if !wrote {
		t.Fatal("expected a write")
	}
	if got := entry.ResetAt.Sub(now); got != accounts.DefaultCooldownWindow {
		t.Fatalf("default window = %v, want %v", got, accounts.DefaultCooldownWindow)
	}
}

// TestPersistCooldownForRehomeSkipsOrgWall is the correctness guard on the reason gate: a 403
// org/region/billing wall (isAccountCap false) is NOT a timed cap and must NOT be persisted —
// a default window would wrongly re-admit a durably-blocked org after it elapses.
func TestPersistCooldownForRehomeSkipsOrgWall(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	store := grStore(t, now, now)
	if _, wrote := PersistCooldownForRehome(store, "uuid:u-alice", "failover_account", "failover_account", false, now); wrote {
		t.Fatal("a 403 org-wall (isAccountCap=false) must NOT persist a cooldown")
	}
	if _, ok := store.CooledDown("uuid:u-alice", now); ok {
		t.Fatal("the org-walled account must NOT read as cooled")
	}
}

// TestPersistCooldownForRehomeEmptyAccountNoop / nil store: fail-open guards.
func TestPersistCooldownForRehomeGuards(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	store := grStore(t, now, now)
	if _, wrote := PersistCooldownForRehome(store, "", "rehomed_seat", "", true, now); wrote {
		t.Fatal("empty account must not write")
	}
	if _, wrote := PersistCooldownForRehome(nil, "uuid:u-alice", "rehomed_seat", "", true, now); wrote {
		t.Fatal("nil store must not write")
	}
}

// TestLive429RehomePersistThenNextLaunchRotatesOff is the END-TO-END proof of the automatic
// loop the user asked for: a live 429 account-cap rehome persists a cooldown for the capped
// account, and a SUBSEQUENT launch (Plan) then automatically rotates OFF that account onto a
// seat with room — without any launch-time re-detection. This is the "on 429, cool it AND the
// next spawn avoids it" behavior, proven across the write side and the read side together.
func TestLive429RehomePersistThenNextLaunchRotatesOff(t *testing.T) {
	now := grMustTime(t, "2026-07-07T12:00:00Z")
	reg := accounts.Registry{Homes: []accounts.Home{
		grHome("alice", "/home/.claude", "u-alice"),
		grHome("bob", "/home/.claude-bob", "u-bob"),
	}}

	// (1) A live 429 on alice's account rehomes and persists the cooldown.
	store := grStore(t, now, now) // start with nothing cooled
	if _, wrote := PersistCooldownForRehome(store, "uuid:u-alice", "rehomed_seat", "rehomed_seat", true, now); !wrote {
		t.Fatal("the 429 rehome must persist alice's cooldown")
	}

	// (2) The NEXT launch builds headroom that folds in the just-written cooldown (alice walled,
	// bob offerable) — exactly what rotationHeadroom + applyCooldownToHeadroom produce — and Plan
	// rotates off alice onto bob. This is the automatic avoidance a bare `fak guard` now gets.
	hr := accounts.RotationHeadroom{}
	for _, e := range store.Active(now) {
		hr[e.Account] = -1 // cooled -> walled, mirroring applyCooldownToHeadroom
	}
	hr["uuid:u-bob"] = 1.5 // bob offerable per the roster

	dir, note, ok := Plan(reg, store, hr, "/home/.claude", now)
	if !ok {
		t.Fatal("the next launch must rotate off the account the 429 just cooled")
	}
	if note.From != "alice" || note.To != "bob" {
		t.Fatalf("note = %+v, want rotate alice->bob", note)
	}
	if guardWantDir(dir) != guardWantDir("/home/.claude-bob") {
		t.Fatalf("rotated dir = %q, want bob's dir", dir)
	}
}

// guardWantDir normalizes a dir the same way Plan's match does, so the rotated-dir assertion
// is robust to the absolute/clean/case-fold canonicalization Serve's returned Dir carries on
// each platform (the test's /home/... literals resolve differently on Windows vs POSIX).
func guardWantDir(dir string) string { return NormalizeDir(dir) }
