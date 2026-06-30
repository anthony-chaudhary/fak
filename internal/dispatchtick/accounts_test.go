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

func TestAllocateWaveGrantsDistinctPoolsAndUnderfills(t *testing.T) {
	got := AllocateWave(AccountWaveInput{Rows: accountRowsFixture(), Count: 8, Product: "claude", WorkKind: "engineering"})
	if !got.OK || got.Granted != 3 || got.Shortfall != 5 || got.DistinctPools != 3 || got.TargetTier != 1 {
		t.Fatalf("wave = %+v, want 3 granted distinct tier-1 pools and shortfall 5", got)
	}
	if got.Lanes[0].Tag != "day26" || got.Lanes[0].Rank != 0 || got.Lanes[0].Size != 3 {
		t.Fatalf("first lane = %+v, want route-weighted day26 rank 0 size 3", got.Lanes[0])
	}
	if got.WaveID == "" || got.Lanes[0].WaveID != got.WaveID {
		t.Fatalf("wave id not stamped consistently: %+v", got)
	}
}

func TestAllocateWaveCollapsesDuplicatePools(t *testing.T) {
	rows := []AccountRow{
		{Account: ".claude-a", Tag: "a", Product: "claude", Dir: "C:/a", AccountUUID: "same", Available: true, ModelTier: 1},
		{Account: ".claude-b", Tag: "b", Product: "claude", Dir: "C:/b", AccountUUID: "same", Available: true, ModelTier: 1},
		{Account: ".claude-c", Tag: "c", Product: "claude", Dir: "C:/c", Available: true, ModelTier: 1},
	}
	got := AllocateWave(AccountWaveInput{Rows: rows, Count: 3, Product: "claude", WorkKind: "engineering"})
	if got.Granted != 2 || got.Shortfall != 1 {
		t.Fatalf("wave = %+v, want duplicate UUID pool collapsed to two grants", got)
	}
	if got.Lanes[0].Pool == got.Lanes[1].Pool {
		t.Fatalf("lanes share a pool: %+v", got.Lanes)
	}
}

func TestAllocateWaveGardeningFallsBackUpAndProductFilters(t *testing.T) {
	got := AllocateWave(AccountWaveInput{Rows: accountRowsFixture(), Count: 2, Product: "claude", WorkKind: "gardening"})
	if !got.OK || got.Granted != 2 || got.Lanes[0].SelectedTier != 1 || !got.Lanes[0].FallbackUsed {
		t.Fatalf("gardening claude wave = %+v, want tier-1 fallback lanes", got)
	}
	open := AllocateWave(AccountWaveInput{Rows: accountRowsFixture(), Count: 2, Product: "opencode", WorkKind: "gardening"})
	if !open.OK || open.Granted != 1 || open.Lanes[0].Tag != "zai" || open.Lanes[0].SelectedTier != 2 {
		t.Fatalf("opencode wave = %+v, want only tier-2 zai", open)
	}
}

func TestAllocateWaveIDDeterministicAndOverrideable(t *testing.T) {
	in := AccountWaveInput{Rows: accountRowsFixture(), Count: 3, Product: "claude", WorkKind: "engineering"}
	a := AllocateWave(in)
	b := AllocateWave(in)
	if a.WaveID == "" || a.WaveID != b.WaveID {
		t.Fatalf("wave ids = %q/%q, want deterministic non-empty id", a.WaveID, b.WaveID)
	}
	in.WaveID = "wave-pinned"
	pinned := AllocateWave(in)
	if pinned.WaveID != "wave-pinned" || pinned.Lanes[0].WaveID != "wave-pinned" {
		t.Fatalf("pinned wave = %+v, want explicit wave id stamped", pinned)
	}
}

func TestRouteAccountSkipsExplicitNonWorkerKinds(t *testing.T) {
	rows := []AccountRow{
		{Account: ".claude-excluded", Tag: "excluded", Kind: "excluded", Product: "claude", Available: true, ModelTier: 1},
		{Account: ".claude-ok", Tag: "ok", Kind: "worker", Product: "claude", Available: true, ModelTier: 1},
	}
	got := RouteAccount(AccountRouteInput{Rows: rows, Product: "claude", WorkKind: "engineering"})
	if !got.OK || got.Account.Tag != "ok" {
		t.Fatalf("route with excluded row = %+v, want worker account", got)
	}
	pool := BuildSeatPool(rows, nil, "claude")
	if pool.TotalSeats != 1 || pool.Seats[0].Tag != "ok" {
		t.Fatalf("seat pool = %+v, want only worker seat", pool)
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
