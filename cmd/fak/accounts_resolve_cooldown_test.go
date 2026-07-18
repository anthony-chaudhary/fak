package main

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// arHome builds a live, serveable seat with a full in-memory identity (no disk I/O — the
// resolve core reads these fields directly, exactly as internal/guardrotate's fixtures do).
func arHome(name, dir, uuid string) accounts.Home {
	return accounts.Home{
		Name:     name,
		Dir:      dir,
		Status:   accounts.StatusActive,
		Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: uuid, Email: name + "@x.test"},
	}
}

// arStore builds an in-memory cooldown store (bound to a temp path so a stray Save cannot
// touch a real fleet file) with the given account keys cooled until reset.
func arStore(t *testing.T, cooledAt, resetAt time.Time, keys ...string) *accounts.CooldownStore {
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

// arFixture is the live july11→july16(blocked) repro shape from #4675: july11's tombstone
// rehomes onto july16, whose account sits inside an active usage-limit window, and the
// anchor day27 serves.
func arFixture(t *testing.T, now time.Time) (accounts.Registry, *accounts.CooldownStore) {
	t.Helper()
	reg := accounts.Registry{
		Homes: []accounts.Home{
			{Name: "july11", Status: accounts.StatusTombstoned, RehomeTo: "july16"},
			arHome("july16", filepath.FromSlash("/h/.claude-july16"), "u-16"),
			arHome("day27", filepath.FromSlash("/h/.claude-day27"), "u-27"),
		},
		Roles: map[string]string{accounts.RoleAnchor: "day27"},
	}
	return reg, arStore(t, now, now.Add(2*time.Hour), accounts.UUIDBucketKey("u-16"))
}

// TestAccountsResolveServeSkipsCooledSeat is #4675's operator-facing witness: the DEFAULT
// `fak accounts resolve` must land on the serving pool-mate (day27), not the throttled
// rehome target — the cooldown-blind Serve answered july16 here.
func TestAccountsResolveServeSkipsCooledSeat(t *testing.T) {
	now := time.Date(2026, 7, 16, 12, 0, 0, 0, time.UTC)
	reg, cd := arFixture(t, now)

	home, _, entry, err := accountsResolveServe(reg, "july11", false, cd, now)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if entry != nil {
		t.Fatalf("a serving pool-mate exists — entry must be nil, got %+v", entry)
	}
	if home.Name != "day27" {
		t.Fatalf("default resolve landed on %q, want day27 (must skip the cooled july16)", home.Name)
	}
}

// TestAccountsResolveServePinStaysCooldownBlind: --pin answers the raw static-pointer
// question ("where does july11's tombstone point?") and must keep the pure, cooldown-blind
// Resolve — the explicit escape the issue requires.
func TestAccountsResolveServePinStaysCooldownBlind(t *testing.T) {
	now := time.Date(2026, 7, 16, 12, 0, 0, 0, time.UTC)
	reg, cd := arFixture(t, now)

	home, _, entry, err := accountsResolveServe(reg, "july11", true, cd, now)
	if err != nil {
		t.Fatalf("pinned resolve: %v", err)
	}
	if entry != nil {
		t.Fatalf("the pin path cannot degrade — entry must be nil, got %+v", entry)
	}
	if home.Name != "july16" {
		t.Fatalf("--pin resolve = %q, want the raw pointer july16 (cooldown must not divert it)", home.Name)
	}
}
