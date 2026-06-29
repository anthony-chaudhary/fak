package dispatchtick

import "testing"

func accountRowsFixture() []AccountRow {
	return []AccountRow{
		{Account: ".claude-gem7", Tag: "gem7", Product: "claude", Dir: "C:/Users/u/.claude-gem7", Available: true, ModelTier: 1, Model: "opus", LiveSessions: 4, ActiveSessions: 8},
		{Account: ".claude-day26", Tag: "day26", Product: "claude", Dir: "C:/Users/u/.claude-day26", Available: true, ModelTier: 1, Model: "opus", LiveSessions: 4, ActiveSessions: 8, RouteWeight: 10},
		{Account: ".claude-busy", Tag: "busy", Product: "claude", Dir: "C:/Users/u/.claude-busy", Available: true, ModelTier: 1, Model: "opus", LiveSessions: 12, ActiveSessions: 30},
		{Account: ".claude-blocked", Tag: "blocked", Product: "claude", Dir: "C:/Users/u/.claude-blocked", Available: false, ModelTier: 1, BlockReason: "usage limit"},
		{Account: "opencode-zai", Tag: "zai", Product: "opencode", Dir: "C:/Users/u/opencode-zai", Available: true, ModelTier: 2, Model: "zai-coding-plan/glm-5.2"},
		{Account: ".claude-copy", Tag: "copy", Product: "claude", Dir: "C:/Users/u/.claude-copy", Available: true, ModelTier: 1, IdentityRole: "duplicate"},
	}
}

func TestRouteAccountPicksTierOneByLoadAndWeight(t *testing.T) {
	got := RouteAccount(AccountRouteInput{Rows: accountRowsFixture(), Product: "claude", WorkKind: "engineering"})
	if !got.OK {
		t.Fatalf("RouteAccount returned not ok: %+v", got)
	}
	if got.Account.Tag != "day26" {
		t.Fatalf("selected tag = %q, want route-weighted day26", got.Account.Tag)
	}
	if got.SelectedTier != 1 || got.FallbackUsed {
		t.Fatalf("tier/fallback = %d/%v, want 1/false", got.SelectedTier, got.FallbackUsed)
	}
	if len(got.BlockedTargetAccounts) != 1 || got.BlockedTargetAccounts[0].Tag != "blocked" {
		t.Fatalf("blocked target accounts = %+v, want blocked tier-one account", got.BlockedTargetAccounts)
	}
}

func TestRouteAccountGardeningTargetsTierTwoAndFallsBackUp(t *testing.T) {
	rows := accountRowsFixture()
	got := RouteAccount(AccountRouteInput{Rows: rows, Product: "opencode", WorkKind: "gardening"})
	if !got.OK || got.Account.Tag != "zai" || got.SelectedTier != 2 {
		t.Fatalf("opencode gardening route = %+v, want tier-2 zai", got)
	}

	for i := range rows {
		if rows[i].Product == "opencode" {
			rows[i].Available = false
		}
	}
	got = RouteAccount(AccountRouteInput{Rows: rows, Product: "claude", WorkKind: "gardening"})
	if !got.OK || got.SelectedTier != 1 || !got.FallbackUsed {
		t.Fatalf("gardening fallback route = %+v, want tier-1 fallback", got)
	}
}

func TestRouteAccountNoTierOneFallbackByDefault(t *testing.T) {
	rows := accountRowsFixture()
	for i := range rows {
		if rows[i].Product == "claude" && rows[i].ModelTier == 1 {
			rows[i].Available = false
		}
	}
	got := RouteAccount(AccountRouteInput{Rows: rows, Product: "claude", WorkKind: "engineering"})
	if got.OK || got.TargetTier != 1 {
		t.Fatalf("tier-one blocked route = %+v, want not ok target tier 1", got)
	}
}

func TestSeatPoolCountsFreeLeasedBlockedAndSkipsDuplicate(t *testing.T) {
	rows := accountRowsFixture()
	leases := []SeatLease{{Worker: "resolve-1", Tag: "gem7", Dir: "C:/Users/u/.claude-gem7"}}
	got := BuildSeatPool(rows, leases, "claude")
	if got.TotalSeats != 4 || got.FreeSeats != 2 || got.LeasedSeats != 1 || got.BlockedSeats != 1 || got.Depleted {
		t.Fatalf("seat pool = %+v, want total=4 free=2 leased=1 blocked=1 depleted=false", got)
	}
	if got.Seats[0].State != "leased" || got.Seats[0].Tag != "gem7" {
		t.Fatalf("first seat = %+v, want leased gem7", got.Seats[0])
	}
	for _, seat := range got.Seats {
		if seat.Tag == "copy" {
			t.Fatalf("duplicate identity seat was included: %+v", seat)
		}
	}
}

func TestAccountNormalizeInfersProductTagAndTier(t *testing.T) {
	claude := NormalizeAccountRow(AccountRow{Account: ".claude"})
	if claude.Product != "claude" || claude.Tag != "default" || claude.ModelTier != 1 {
		t.Fatalf("claude normalized = %+v", claude)
	}
	open := NormalizeAccountRow(AccountRow{Account: "opencode-zai2"})
	if open.Product != "opencode" || open.Tag != "zai2" || open.ModelTier != 2 {
		t.Fatalf("opencode normalized = %+v", open)
	}
}
