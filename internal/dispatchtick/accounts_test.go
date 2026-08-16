package dispatchtick

import (
	"encoding/json"
	"strings"
	"testing"
)

func boolPtr(v bool) *bool { return &v }

func TestAccountSessionCapEnvKnob(t *testing.T) {
	claude := AccountRow{Product: "claude"}
	if got := AccountSessionCap(claude); got != DefaultClaudeSessionsPerAccount {
		t.Fatalf("default claude cap = %d, want %d", got, DefaultClaudeSessionsPerAccount)
	}

	t.Setenv(SessionsPerAccountEnv, "7")
	if got := AccountSessionCap(claude); got != DefaultClaudeSessionsPerAccount {
		t.Fatalf("FAK_SESSIONS_PER_ACCOUNT=7 claude cap = %d, want hard default %d", got, DefaultClaudeSessionsPerAccount)
	}
	if got := AccountSessionCap(AccountRow{Product: "opencode"}); got != DefaultAccountSessionsPerWorker {
		t.Fatalf("opencode cap = %d, want %d", got, DefaultAccountSessionsPerWorker)
	}

	for _, bad := range []string{"0", "-3", "notanint", "  "} {
		t.Setenv(SessionsPerAccountEnv, bad)
		if got := AccountSessionCap(claude); got != DefaultClaudeSessionsPerAccount {
			t.Fatalf("FAK_SESSIONS_PER_ACCOUNT=%q claude cap = %d, want default %d",
				bad, got, DefaultClaudeSessionsPerAccount)
		}
	}
}

func accountRowsFixture() []AccountRow {
	return []AccountRow{
		{Account: ".claude-gem7", Tag: "gem7", Product: "claude", Dir: "C:/Users/u/.claude-gem7", Available: true, ModelTier: 1, Model: "opus", LiveSessions: 2, ActiveSessions: 8},
		{Account: ".claude-day26", Tag: "day26", Product: "claude", Dir: "C:/Users/u/.claude-day26", Available: true, ModelTier: 1, Model: "opus", LiveSessions: 1, ActiveSessions: 8, RouteWeight: 10, LoginStatus: "ready", CanServe: boolPtr(true)},
		{Account: ".claude-busy", Tag: "busy", Product: "claude", Dir: "C:/Users/u/.claude-busy", Available: true, ModelTier: 1, Model: "opus", LiveSessions: 12, ActiveSessions: 30},
		{Account: ".claude-blocked", Tag: "blocked", Product: "claude", Dir: "C:/Users/u/.claude-blocked", Available: false, ModelTier: 1, BlockReason: "config directory exists but has no live credentials", LoginStatus: "needs_login", CanServe: boolPtr(false)},
		{Account: "opencode-zai", Tag: "zai", Product: "opencode", Dir: "C:/Users/u/opencode-zai", Available: true, ModelTier: 2, Model: "zai-coding-plan/glm-5.2"},
		{Account: ".claude-copy", Tag: "copy", Product: "claude", Dir: "C:/Users/u/.claude-copy", Available: true, ModelTier: 1, IdentityRole: "duplicate"},
	}
}

func TestRouteAccountEmitsRankedSeatSelectionReasons(t *testing.T) {
	t.Setenv(SessionsPerAccountEnv, "4")
	no := false
	rows := []AccountRow{
		{Account: "healthy", Tag: "healthy", Product: "claude", Kind: "worker", ModelTier: 1, Available: true, CanServe: boolPtr(true), RouteWeight: 9},
		{Account: "capped", Tag: "capped", Product: "claude", Kind: "worker", ModelTier: 1, Available: true, CanServe: boolPtr(true), LiveSessions: DefaultClaudeSessionsPerAccount},
		{Account: "dead", Tag: "dead", Product: "claude", Kind: "worker", ModelTier: 1, Available: false, CanServe: &no, LoginStatus: "auth_failed", BlockReason: "stale credential"},
		{Account: "cooled", Tag: "cooled", Product: "claude", Kind: "worker", ModelTier: 1, Available: false, CanServe: boolPtr(true), BlockReason: "account in cooldown"},
	}
	got := RouteAccount(AccountRouteInput{Rows: rows, Product: "claude", WorkKind: "implementation"})
	if !got.OK || got.SeatSelection.WinnerTag != "healthy" || len(got.SeatSelection.Candidates) != 4 {
		t.Fatalf("selection = %+v", got.SeatSelection)
	}
	reasons := map[string]string{}
	for _, c := range got.SeatSelection.Candidates {
		reasons[c.Tag] = c.SkipReason
	}
	if reasons["capped"] != "session_cap" || reasons["dead"] != "auth_failed" || reasons["cooled"] != "cooldown" {
		t.Fatalf("skip reasons = %+v", reasons)
	}
	if got.SeatSelection.Summary != "picked healthy over 3 (selected target tier)" {
		t.Fatalf("summary = %q", got.SeatSelection.Summary)
	}
}

func TestSeatSelectionNamesChosenDespiteAuthFailed(t *testing.T) {
	no := false
	res := AccountRouteResult{OK: true, Reason: "selected target tier", TargetTier: 1, Account: AccountRow{Account: "dead", Tag: "dead", ModelTier: 1, Available: true, CanServe: &no, LoginStatus: "auth_failed"}}
	got := buildSeatSelection([]AccountRow{res.Account}, 1, res)
	if got.WinnerReason != "chosen despite auth_failed" || !strings.Contains(got.Summary, "chosen despite auth_failed") {
		t.Fatalf("selection = %+v", got)
	}
}

func TestRouteAccountPicksTierOneByLoadAndWeight(t *testing.T) {
	t.Setenv(SessionsPerAccountEnv, "")
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
	if len(got.BlockedTargetAccounts) != 2 || got.BlockedTargetAccounts[0].Tag != "blocked" {
		t.Fatalf("blocked target accounts = %+v, want login-blocked account plus at-cap busy", got.BlockedTargetAccounts)
	}
	if got.BlockedTargetAccounts[0].LoginStatus != "needs_login" || got.BlockedTargetAccounts[0].CanServe == nil || *got.BlockedTargetAccounts[0].CanServe {
		t.Fatalf("blocked target readiness = %+v, want needs_login/can_serve=false", got.BlockedTargetAccounts[0])
	}
	if got.BlockedTargetAccounts[1].Tag != "busy" || !strings.Contains(got.BlockedTargetAccounts[1].BlockReason, "at session cap") {
		t.Fatalf("blocked target accounts = %+v, want busy reported at session cap", got.BlockedTargetAccounts)
	}
}

func TestRouteAccountSkipsSeatAtSessionCap(t *testing.T) {
	t.Setenv(SessionsPerAccountEnv, "")
	rows := accountRowsFixture()
	for i := range rows {
		if rows[i].Tag == "day26" {
			rows[i].LiveSessions = DefaultClaudeSessionsPerAccount
		}
	}
	got := RouteAccount(AccountRouteInput{Rows: rows, Product: "claude", WorkKind: "engineering"})
	if !got.OK || got.Account.Tag != "gem7" {
		t.Fatalf("route = %+v, want under-cap gem7 over route-weighted day26 at cap", got)
	}
	foundAtCap := false
	for _, b := range got.BlockedTargetAccounts {
		if b.Tag == "day26" && strings.Contains(b.BlockReason, "at session cap") {
			foundAtCap = true
		}
	}
	if !foundAtCap {
		t.Fatalf("blocked = %+v, want day26 reported at session cap", got.BlockedTargetAccounts)
	}
}

func TestRouteAccountRefusesWhenEverySeatAtCap(t *testing.T) {
	t.Setenv(SessionsPerAccountEnv, "")
	rows := accountRowsFixture()
	for i := range rows {
		if rows[i].Product == "claude" && rows[i].Available {
			rows[i].LiveSessions += DefaultClaudeSessionsPerAccount
		}
	}
	got := RouteAccount(AccountRouteInput{Rows: rows, Product: "claude", WorkKind: "engineering"})
	if got.OK || !strings.Contains(got.Reason, "session cap") {
		t.Fatalf("route = %+v, want refusal naming the session cap", got)
	}
	t.Setenv(SessionsPerAccountEnv, "40")
	relaxed := RouteAccount(AccountRouteInput{Rows: rows, Product: "claude", WorkKind: "engineering"})
	if !relaxed.OK {
		t.Fatalf("route with raised cap = %+v, want admitted again", relaxed)
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
	t.Setenv(SessionsPerAccountEnv, "")
	// Registry-live sessions floor each pool's load: day26 has 1, gem7 has 2, and
	// busy sits over the cap entirely, so 8 requested slots underfill to 5.
	got := AllocateWave(AccountWaveInput{Rows: accountRowsFixture(), Count: 8, Product: "claude", WorkKind: "engineering"})
	if !got.OK || got.Granted != 5 || got.Shortfall != 3 || got.DistinctPools != 2 || got.TargetTier != 1 {
		t.Fatalf("wave = %+v, want 5 granted session slots across 2 under-cap tier-1 pools", got)
	}
	if got.Lanes[0].Tag != "day26" || got.Lanes[0].Rank != 0 || got.Lanes[0].Size != 5 ||
		got.Lanes[0].SessionSlot != 2 || got.Lanes[0].SessionCap != DefaultClaudeSessionsPerAccount {
		t.Fatalf("first lane = %+v, want least-loaded day26 rank 0 size 5 slot 2/4", got.Lanes[0])
	}
	if got.Lanes[0].LoginStatus != "ready" || got.Lanes[0].CanServe == nil || !*got.Lanes[0].CanServe {
		t.Fatalf("first lane readiness = %+v, want ready/can_serve=true", got.Lanes[0])
	}
	m := got.Lanes[0].Map()
	if m["login_status"] != "ready" || m["can_serve"] != true || m["session_slot"] != 2 || m["session_cap"] != DefaultClaudeSessionsPerAccount {
		t.Fatalf("first lane map readiness = %+v, want login_status/can_serve", m)
	}
	if got.WaveID == "" || got.Lanes[0].WaveID != got.WaveID {
		t.Fatalf("wave id not stamped consistently: %+v", got)
	}
}

func TestAllocateWaveFloorsLoadAtRegistryLiveSessions(t *testing.T) {
	t.Setenv(SessionsPerAccountEnv, "")
	rows := []AccountRow{{
		Account: ".claude-seat-a", Tag: "seat-a", Product: "claude",
		Dir: "C:/seats/a", AccountUUID: "acct-a", Available: true, ModelTier: 1,
		LiveSessions: 3,
	}}
	got := AllocateWave(AccountWaveInput{Rows: rows, Count: 4, Product: "claude", WorkKind: "engineering"})
	if !got.OK || got.Granted != 1 || got.Shortfall != 3 || got.Lanes[0].SessionSlot != 4 {
		t.Fatalf("wave = %+v, want registry-live sessions to consume 3 of 4 slots", got)
	}
	// A lease held by one of those live sessions must not double-count: the load
	// is max(leases, live), never the sum.
	leases := []SeatLease{{Worker: "resolve-1", Tag: "seat-a", Dir: "C:/seats/a"}}
	again := AllocateWave(AccountWaveInput{Rows: rows, Leases: leases, Count: 4, Product: "claude", WorkKind: "engineering"})
	if !again.OK || again.Granted != 1 || again.Lanes[0].SessionSlot != 4 {
		t.Fatalf("wave with overlapping lease = %+v, want max(leases, live)=3 so one slot free", again)
	}
}

func TestAllocateWaveCapsEachDistinctPoolAtSessionBudget(t *testing.T) {
	rows := []AccountRow{
		{Account: ".claude-seat-a", Tag: "seat-a", Product: "claude", Dir: "C:/seats/a", AccountUUID: "acct-a", Available: true, ModelTier: 1},
		{Account: ".claude-seat-a-copy", Tag: "seat-a-copy", Product: "claude", Dir: "C:/seats/a-copy", AccountUUID: "acct-a", Available: true, ModelTier: 1},
		{Account: ".claude-seat-b", Tag: "seat-b", Product: "claude", Dir: "C:/seats/b", AccountUUID: "acct-b", Available: true, ModelTier: 1},
	}
	got := AllocateWave(AccountWaveInput{Rows: rows, Count: 9, Product: "claude", WorkKind: "engineering"})
	if !got.OK || got.Requested != 9 || got.Granted != 8 || got.Shortfall != 1 || got.DistinctPools != 2 {
		t.Fatalf("wave = %+v, want requested=9 granted=8 shortfall=1 distinct_pools=2", got)
	}
	byPool := map[string]int{}
	for _, lane := range got.Lanes {
		byPool[lane.Pool]++
		if lane.Tag == "seat-a-copy" {
			t.Fatalf("duplicate account row was allocated: %+v", lane)
		}
	}
	for pool, n := range byPool {
		if n > DefaultClaudeSessionsPerAccount {
			t.Fatalf("pool %q assigned %d slots, want cap %d", pool, n, DefaultClaudeSessionsPerAccount)
		}
	}
	if !strings.Contains(got.Reason, "granted 8 of 9 session slot") || !strings.Contains(got.Reason, "1 short") {
		t.Fatalf("reason = %q, want explicit capacity shortfall", got.Reason)
	}
}

func TestAllocateWaveSubtractsLiveLeaseSlots(t *testing.T) {
	rows := []AccountRow{{
		Account: ".claude-seat-a", Tag: "seat-a", Product: "claude",
		Dir: "C:/seats/a", AccountUUID: "acct-a", Available: true, ModelTier: 1,
	}}
	leases := []SeatLease{
		{Worker: "resolve-1", Tag: "seat-a", Dir: "C:/seats/a"},
		{Worker: "resolve-2", Tag: "seat-a", Dir: "C:/seats/a"},
		{Worker: "resolve-3", Tag: "seat-a", Dir: "C:/seats/a"},
	}
	got := AllocateWave(AccountWaveInput{Rows: rows, Leases: leases, Count: 4, Product: "claude", WorkKind: "engineering"})
	if !got.OK || got.Granted != 1 || got.Shortfall != 3 || got.DistinctPools != 1 {
		t.Fatalf("wave = %+v, want only the fourth session slot free", got)
	}
	if got.Lanes[0].SessionSlot != 4 || got.Lanes[0].SessionCap != DefaultClaudeSessionsPerAccount {
		t.Fatalf("lane = %+v, want slot 4/%d", got.Lanes[0], DefaultClaudeSessionsPerAccount)
	}
}

func TestAllocateWaveCollapsesDuplicatePools(t *testing.T) {
	rows := []AccountRow{
		{Account: ".claude-a", Tag: "a", Product: "claude", Dir: "C:/a", AccountUUID: "same", Available: true, ModelTier: 1},
		{Account: ".claude-b", Tag: "b", Product: "claude", Dir: "C:/b", AccountUUID: "same", Available: true, ModelTier: 1},
		{Account: ".claude-c", Tag: "c", Product: "claude", Dir: "C:/c", Available: true, ModelTier: 1},
	}
	got := AllocateWave(AccountWaveInput{Rows: rows, Count: 9, Product: "claude", WorkKind: "engineering"})
	if got.Granted != 8 || got.Shortfall != 1 || got.DistinctPools != 2 {
		t.Fatalf("wave = %+v, want duplicate UUID pool collapsed to two session-capped pools", got)
	}
	pools := map[string]bool{}
	for _, lane := range got.Lanes {
		pools[lane.Pool] = true
		if lane.Tag == "b" {
			t.Fatalf("duplicate UUID row was allocated: %+v", got.Lanes)
		}
	}
	if len(pools) != 2 {
		t.Fatalf("pools = %+v, want exactly two collapsed pools", pools)
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
	if pool.TotalSeats != DefaultClaudeSessionsPerAccount || pool.Seats[0].Tag != "ok" {
		t.Fatalf("seat pool = %+v, want only worker account's Claude session slots", pool)
	}
}

func TestSeatPoolCountsFreeLeasedBlockedAndSkipsDuplicate(t *testing.T) {
	rows := accountRowsFixture()
	leases := []SeatLease{{Worker: "resolve-1", Tag: "gem7", Dir: "C:/Users/u/.claude-gem7"}}
	got := BuildSeatPool(rows, leases, "claude")
	if got.TotalSeats != 16 || got.FreeSeats != 11 || got.LeasedSeats != 1 || got.BlockedSeats != 4 || got.Depleted {
		t.Fatalf("seat pool = %+v, want total=16 free=11 leased=1 blocked=4 depleted=false", got)
	}
	if got.Seats[0].State != "leased" || got.Seats[0].Tag != "gem7" ||
		got.Seats[0].LeasedSlots != 1 || got.Seats[0].FreeSlots != 3 || got.Seats[0].SessionCap != 4 {
		t.Fatalf("first seat = %+v, want leased gem7 with 3 free slots", got.Seats[0])
	}
	for _, seat := range got.Seats {
		if seat.Tag == "copy" {
			t.Fatalf("duplicate identity seat was included: %+v", seat)
		}
	}
}

// TestAccountWirePreservesFleetAccountsFields locks the JSON field names that
// fak dispatch tick/wave consumed from tools/fleet_accounts.py route|seats|wave
// output. The Go structs carry them as tags, so a rename compiles cleanly while
// silently breaking every consumer written against the Python wire shape.
func TestAccountWirePreservesFleetAccountsFields(t *testing.T) {
	rows := accountRowsFixture()
	leases := []SeatLease{{Worker: "resolve-1", Tag: "gem7", Dir: "C:/Users/u/.claude-gem7"}}
	raw, err := json.Marshal(BuildSeatPool(rows, leases, "claude"))
	if err != nil {
		t.Fatalf("marshal seat pool: %v", err)
	}
	seatWire := map[string]json.RawMessage{}
	if err := json.Unmarshal(raw, &seatWire); err != nil {
		t.Fatalf("unmarshal seat pool: %v", err)
	}
	for _, key := range []string{"total_seats", "free_seats", "leased_seats", "depleted", "seats"} {
		if _, ok := seatWire[key]; !ok {
			t.Fatalf("seat pool wire payload lost %q: %s", key, raw)
		}
	}
	var seats []map[string]any
	if err := json.Unmarshal(seatWire["seats"], &seats); err != nil {
		t.Fatalf("unmarshal seat rows: %v", err)
	}
	if len(seats) == 0 {
		t.Fatalf("seat pool wire payload has no seats: %s", raw)
	}
	for _, key := range []string{"tag", "model", "state"} {
		if _, ok := seats[0][key]; !ok {
			t.Fatalf("seat row wire payload lost %q: %s", key, raw)
		}
	}

	wave := AllocateWave(AccountWaveInput{Rows: rows, Count: 2, Product: "claude", WorkKind: "engineering"})
	waveWire := wave.Map()
	for _, key := range []string{"wave_id", "shortfall", "granted", "lanes", "blocked_target_accounts"} {
		if _, ok := waveWire[key]; !ok {
			t.Fatalf("wave wire payload lost %q: %+v", key, waveWire)
		}
	}
	if len(wave.Lanes) == 0 {
		t.Fatalf("wave allocated no lanes: %+v", wave)
	}
	laneWire := wave.Lanes[0].Map()
	for _, key := range []string{"tag", "config_dir", "selected_tier", "model", "wave_id"} {
		if _, ok := laneWire[key]; !ok {
			t.Fatalf("wave lane wire payload lost %q: %+v", key, laneWire)
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

func TestAccountNormalizeAppliesClaudeLoginGate(t *testing.T) {
	row := AccountRow{
		Account:     ".claude-needs",
		Tag:         "needs",
		Product:     "claude",
		Dir:         "C:/Users/u/.claude-needs",
		Available:   true,
		ModelTier:   1,
		LoginStatus: "needs_login",
		CanServe:    boolPtr(false),
	}
	norm := NormalizeAccountRow(row)
	if norm.Available || norm.BlockReason == "" {
		t.Fatalf("normalized row = %+v, want blocked with reason", norm)
	}

	route := RouteAccount(AccountRouteInput{Rows: []AccountRow{row}, Product: "claude", WorkKind: "engineering"})
	if route.OK || len(route.BlockedTargetAccounts) != 1 ||
		route.BlockedTargetAccounts[0].LoginStatus != "needs_login" ||
		route.BlockedTargetAccounts[0].CanServe == nil || *route.BlockedTargetAccounts[0].CanServe {
		t.Fatalf("route = %+v, want no route and blocked login posture", route)
	}

	wave := AllocateWave(AccountWaveInput{Rows: []AccountRow{row}, Count: 1, Product: "claude", WorkKind: "engineering"})
	if wave.OK || wave.Granted != 0 || len(wave.BlockedTargetAccounts) != 1 {
		t.Fatalf("wave = %+v, want blocked login posture and no granted lane", wave)
	}

	pool := BuildSeatPool([]AccountRow{row}, nil, "claude")
	if pool.TotalSeats != DefaultClaudeSessionsPerAccount || pool.FreeSeats != 0 ||
		pool.BlockedSeats != DefaultClaudeSessionsPerAccount || pool.Seats[0].State != "blocked" {
		t.Fatalf("seat pool = %+v, want login-blocked seat", pool)
	}
}

func TestLaneSeatShare(t *testing.T) {
	// Lanes A (x2), B (x1), C (x1), plus two leases whose lane does not resolve
	// (one empty, one whitespace-only) -> both bucketed under LaneUnattributed.
	leases := []SeatLease{
		{Worker: "w1", Tag: "a-one"},
		{Worker: "w2", Tag: "a-two"},
		{Worker: "w3", Tag: "b-one"},
		{Worker: "w4", Tag: "c-one"},
		{Worker: "w5", Tag: "orphan"}, // laneOf returns "" -> unattributed
		{Worker: "w6", Tag: "spacey"}, // laneOf returns "  " -> trimmed empty -> unattributed
	}
	laneOf := func(l SeatLease) string {
		switch l.Tag {
		case "a-one", "a-two":
			return "A"
		case "b-one":
			return "B"
		case "c-one":
			return "C"
		case "spacey":
			return "  " // whitespace must trim to empty and bucket as unattributed
		default:
			return "" // unresolved lane
		}
	}

	got := LaneSeatShare(leases, laneOf)
	want := map[string]LaneShare{
		"A":              {Count: 2, Share: 2.0 / 6.0},
		"B":              {Count: 1, Share: 1.0 / 6.0},
		"C":              {Count: 1, Share: 1.0 / 6.0},
		LaneUnattributed: {Count: 2, Share: 2.0 / 6.0},
	}
	if len(got) != len(want) {
		t.Fatalf("lane count = %d %+v, want %d lanes", len(got), got, len(want))
	}
	var sum float64
	for lane, w := range want {
		g, ok := got[lane]
		if !ok {
			t.Fatalf("missing lane %q in %+v", lane, got)
		}
		if g.Count != w.Count {
			t.Fatalf("lane %q count = %d, want %d", lane, g.Count, w.Count)
		}
		if diff := g.Share - w.Share; diff > 1e-9 || diff < -1e-9 {
			t.Fatalf("lane %q share = %v, want %v", lane, g.Share, w.Share)
		}
	}
	for _, g := range got {
		sum += g.Share
	}
	if diff := sum - 1.0; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("shares sum = %v, want 1.0", sum)
	}

	// Empty input -> empty (non-nil) map: no phantom unattributed lane, no NaN share.
	if empty := LaneSeatShare(nil, laneOf); len(empty) != 0 {
		t.Fatalf("empty input lanes = %+v, want empty map", empty)
	}

	// Nil laneOf -> every lease is unattributed.
	nilFold := LaneSeatShare([]SeatLease{{Worker: "x"}, {Worker: "y"}}, nil)
	if len(nilFold) != 1 {
		t.Fatalf("nil laneOf lanes = %+v, want only unattributed", nilFold)
	}
	if u := nilFold[LaneUnattributed]; u.Count != 2 || u.Share != 1.0 {
		t.Fatalf("nil laneOf unattributed = %+v, want count 2 share 1.0", u)
	}
}
