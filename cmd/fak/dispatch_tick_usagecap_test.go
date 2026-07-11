package main

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestDispatchUsageCapCensus proves the advisory census correctly counts the routable
// fleet and, among it, the accounts under an ACTIVE usage-limit cooldown -- filtering by
// product, ignoring a rate-limit (not usage) cooldown and an expired one, deduping by
// account, and picking the soonest reset. This is the seam that turns the raw cooldown
// store into the account-based signal UsageCapAdvisory.Armed judges.
func TestDispatchUsageCapCensus(t *testing.T) {
	now := time.Date(2026, 7, 8, 18, 0, 0, 0, time.UTC)
	store, err := accounts.LoadCooldownStore(filepath.Join(t.TempDir(), "account-cooldown.json"))
	if err != nil {
		t.Fatalf("LoadCooldownStore: %v", err)
	}
	// Two accounts usage-capped (soonest reset is +30m), one rate-limit-capped (NOT usage,
	// must not count), one expired (window elapsed, must not count).
	store.Cool(accounts.UUIDBucketKey("uuid-cap-a"), accounts.CooldownUsageLimit, "test", now, now.Add(time.Hour))
	store.Cool(accounts.UUIDBucketKey("uuid-cap-b"), accounts.CooldownUsageLimit, "test", now, now.Add(30*time.Minute))
	store.Cool(accounts.UUIDBucketKey("uuid-rate"), accounts.CooldownRateLimit, "test", now, now.Add(time.Hour))
	store.Cool(accounts.UUIDBucketKey("uuid-expired"), accounts.CooldownUsageLimit, "test", now.Add(-2*time.Hour), now.Add(-time.Hour))

	rows := []dispatchtick.AccountRow{
		{Tag: "a", AccountUUID: "uuid-cap-a", Product: "claude", Available: true},
		{Tag: "a-dup", AccountUUID: "uuid-cap-a", Product: "claude", Available: true}, // dedup: same uuid
		{Tag: "b", AccountUUID: "uuid-cap-b", Product: "claude", Available: true},
		{Tag: "rate", AccountUUID: "uuid-rate", Product: "claude", Available: true},
		{Tag: "expired", AccountUUID: "uuid-expired", Product: "claude", Available: true},
		{Tag: "healthy", AccountUUID: "uuid-ok", Product: "claude", Available: true},
		{Tag: "other", AccountUUID: "uuid-other", Product: "codex", Available: true}, // wrong product
	}

	adv := dispatchUsageCapCensus(rows, store, "claude", now, dispatchtick.DefaultUsageCapAdvisoryMin, 2)

	// 5 unique claude accounts (a, b, rate, expired, healthy); the codex row and the dup
	// are excluded.
	if adv.Accounts != 5 {
		t.Errorf("Accounts = %d, want 5 (unique claude accounts)", adv.Accounts)
	}
	// Only the two active usage-limit accounts count.
	if adv.Capped != 2 {
		t.Errorf("Capped = %d, want 2 (usage-limit, active, deduped)", adv.Capped)
	}
	if !adv.EarliestReset.Equal(now.Add(30 * time.Minute)) {
		t.Errorf("EarliestReset = %s, want %s (soonest of the capped)", adv.EarliestReset, now.Add(30*time.Minute))
	}
	if adv.FreeSeats != 2 {
		t.Errorf("FreeSeats = %d, want 2 (carried context)", adv.FreeSeats)
	}
	// 2 capped of 5 is a minority: not armed (honest on a still-mostly-healthy fleet).
	if adv.Armed() {
		t.Errorf("Armed() = true, want false (2 of 5 capped is a minority)")
	}
}

// TestDispatchUsageCapCensusArmsWhenMajorityCapped proves the census arms once usage caps
// take over the majority of the routable fleet.
func TestDispatchUsageCapCensusArmsWhenMajorityCapped(t *testing.T) {
	now := time.Date(2026, 7, 8, 18, 0, 0, 0, time.UTC)
	store, err := accounts.LoadCooldownStore(filepath.Join(t.TempDir(), "account-cooldown.json"))
	if err != nil {
		t.Fatalf("LoadCooldownStore: %v", err)
	}
	rows := []dispatchtick.AccountRow{{Tag: "ok", AccountUUID: "uuid-ok", Product: "claude", Available: true}}
	for i, uuid := range []string{"u1", "u2", "u3", "u4"} {
		store.Cool(accounts.UUIDBucketKey(uuid), accounts.CooldownUsageLimit, "test", now, now.Add(time.Duration(i+1)*time.Hour))
		rows = append(rows, dispatchtick.AccountRow{Tag: uuid, AccountUUID: uuid, Product: "claude", Available: true})
	}
	adv := dispatchUsageCapCensus(rows, store, "claude", now, dispatchtick.DefaultUsageCapAdvisoryMin, 1)
	if adv.Capped != 4 || adv.Accounts != 5 {
		t.Fatalf("census = %d/%d, want 4/5", adv.Capped, adv.Accounts)
	}
	if !adv.Armed() {
		t.Fatalf("Armed() = false, want true (4 of 5 capped is a majority >= floor)")
	}
	note := adv.Note()
	if note["capped_accounts"] != 4 || note["total_accounts"] != 5 {
		t.Errorf("note census = %v/%v, want 4/5", note["capped_accounts"], note["total_accounts"])
	}
}

// TestDispatchUsageCapCensusNilStoreFailsOpen proves a nil store (absent/unreadable
// cooldown file) yields a zero-capped census -- the advisory never arms and dispatch is
// byte-identical to before.
func TestDispatchUsageCapCensusNilStoreFailsOpen(t *testing.T) {
	now := time.Date(2026, 7, 8, 18, 0, 0, 0, time.UTC)
	rows := []dispatchtick.AccountRow{{Tag: "a", AccountUUID: "uuid-a", Product: "claude", Available: true}}
	adv := dispatchUsageCapCensus(rows, nil, "claude", now, dispatchtick.DefaultUsageCapAdvisoryMin, 1)
	if adv.Capped != 0 || adv.Armed() {
		t.Errorf("nil store must yield 0 capped / not armed, got capped=%d armed=%v", adv.Capped, adv.Armed())
	}
}

// TestDispatchPreflightUsageCapCodexAbstains proves the codex backend (no usage-limit
// cooldown store) abstains without touching the filesystem.
func TestDispatchPreflightUsageCapCodexAbstains(t *testing.T) {
	adv := dispatchPreflightUsageCap(t.TempDir(), "codex", dispatchtick.SeatCheck{})
	if adv.Armed() || adv.Accounts != 0 {
		t.Errorf("codex must abstain, got %+v", adv)
	}
}

// TestDispatchUsageCapAdvisoryThresholdEnv proves the arming floor honors
// FAK_USAGECAP_ADVISORY_MIN and falls back to the default on empty/invalid input.
func TestDispatchUsageCapAdvisoryThresholdEnv(t *testing.T) {
	if got := dispatchUsageCapAdvisoryThreshold(); got != dispatchtick.DefaultUsageCapAdvisoryMin {
		t.Errorf("default threshold = %d, want %d", got, dispatchtick.DefaultUsageCapAdvisoryMin)
	}
	t.Setenv("FAK_USAGECAP_ADVISORY_MIN", "7")
	if got := dispatchUsageCapAdvisoryThreshold(); got != 7 {
		t.Errorf("env threshold = %d, want 7", got)
	}
	t.Setenv("FAK_USAGECAP_ADVISORY_MIN", "nonsense")
	if got := dispatchUsageCapAdvisoryThreshold(); got != dispatchtick.DefaultUsageCapAdvisoryMin {
		t.Errorf("invalid env threshold = %d, want default %d", got, dispatchtick.DefaultUsageCapAdvisoryMin)
	}
}
