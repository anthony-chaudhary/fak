package main

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestDispatchApplyAccountCooldown proves the dispatch roster overlay drops a seat whose
// upstream account holds an ACTIVE cooldown from the servable pool, while leaving
// non-cooled seats, uuid-less rows, and (with a nil store) the whole roster untouched.
// This is the guard that stops RouteAccount/AllocateWave routing onto a cooled seat that
// immediately 429s.
func TestDispatchApplyAccountCooldown(t *testing.T) {
	now := time.Date(2026, 7, 8, 18, 0, 0, 0, time.UTC)
	store, err := accounts.LoadCooldownStore(filepath.Join(t.TempDir(), "account-cooldown.json"))
	if err != nil {
		t.Fatalf("LoadCooldownStore: %v", err)
	}
	// Active cooldown: window still open at now.
	store.Cool(accounts.UUIDBucketKey("uuid-cooled"), accounts.CooldownUsageLimit, "test", now, now.Add(time.Hour))
	// Expired cooldown: window already elapsed at now — must NOT block.
	store.Cool(accounts.UUIDBucketKey("uuid-expired"), accounts.CooldownUsageLimit, "test", now.Add(-2*time.Hour), now.Add(-time.Hour))

	rows := []dispatchtick.AccountRow{
		{Tag: "cooled", AccountUUID: "uuid-cooled", Available: true},
		{Tag: "servable", AccountUUID: "uuid-ok", Available: true},
		{Tag: "expired", AccountUUID: "uuid-expired", Available: true},
		{Tag: "nouuid", AccountUUID: "", Available: true},
	}
	out := dispatchApplyAccountCooldown(rows, store, now)

	byTag := map[string]dispatchtick.AccountRow{}
	for _, r := range out {
		byTag[r.Tag] = r
	}

	cooled := byTag["cooled"]
	if cooled.Available {
		t.Errorf("cooled seat: Available = true, want false")
	}
	if cooled.CanServe == nil || *cooled.CanServe {
		t.Errorf("cooled seat: CanServe = %v, want non-nil false", cooled.CanServe)
	}
	if cooled.BlockReason == "" {
		t.Errorf("cooled seat: BlockReason empty, want a cooldown reason")
	}

	for _, tag := range []string{"servable", "expired", "nouuid"} {
		r := byTag[tag]
		if !r.Available {
			t.Errorf("%s seat: Available = false, want true (not cooled)", tag)
		}
		if r.CanServe != nil {
			t.Errorf("%s seat: CanServe = %v, want untouched (nil)", tag, *r.CanServe)
		}
	}
}

// TestDispatchApplyAccountCooldownNilStore proves the overlay fails open: a nil store
// (absent/unreadable cooldown file) leaves the roster fully servable.
func TestDispatchApplyAccountCooldownNilStore(t *testing.T) {
	now := time.Date(2026, 7, 8, 18, 0, 0, 0, time.UTC)
	rows := []dispatchtick.AccountRow{{Tag: "a", AccountUUID: "uuid-a", Available: true}}
	out := dispatchApplyAccountCooldown(rows, nil, now)
	if !out[0].Available || out[0].CanServe != nil {
		t.Errorf("nil store must leave roster untouched, got %+v", out[0])
	}
}
