package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

func hrBool(b bool) *bool    { return &b }
func hrStr(s string) *string { return &s }
func hrInt(i int) *int       { return &i }

// hrNow is a fixed anchor for the reset-soonness tie-break, so a "3pm"-style reset resolves
// deterministically relative to it.
var hrNow = time.Date(2026, time.June, 30, 10, 0, 0, 0, time.UTC)

// TestHeadroomFromRoster pins the BANDED offerability tiering that turns the live runtime
// roster into the config-plane rotation signal: offerable -> (1,2], walled/throttled ->
// [-1,0), unknown -> 0, keyed by the account bucket key; non-claude and identity-less rows
// are ignored. The exact within-tier value is a separate test; here we assert the bands so a
// bucket never jumps tier.
func TestHeadroomFromRoster(t *testing.T) {
	rows := []fleetaccounts.Account{
		// Offerable claude worker -> room (band (1,2]).
		{Product: "claude", AccountUUID: hrStr("u-day30"), Available: hrBool(true), Blocked: hrBool(false)},
		// Blocked claude worker -> walled (band [-1,0)).
		{Product: "claude", AccountUUID: hrStr("u-gem7"), Available: hrBool(false), Blocked: hrBool(true)},
		// Usage-throttled claude worker -> walled ([-1,0)), even if a stale Available lingered true.
		{Product: "claude", AccountUUID: hrStr("u-cap"), Available: hrBool(true), Throttled: hrBool(true)},
		// No runtime availability signal -> unknown (0).
		{Product: "claude", AccountUUID: hrStr("u-unknown")},
		// Non-claude row -> ignored (config-plane rotation is over Claude seats).
		{Product: "opencode", AccountUUID: hrStr("u-glm"), Available: hrBool(true)},
		// No resolved identity -> ignored (nothing to key on).
		{Product: "claude", Available: hrBool(true)},
	}

	hr := headroomFromRoster(rows, hrNow)
	if len(hr) != 4 {
		t.Fatalf("headroom map = %v, want 4 entries", hr)
	}
	// Offerable band is (1,2] — strictly above the unknown/walled tiers. Walled band is [-1,0)
	// — its floor -1 is INCLUSIVE (a walled bucket with no parseable reset carries the bare
	// base, zero soonness bonus, which is exactly what u-gem7/u-cap have here).
	assertOfferable(t, "u-day30 offerable", hr["uuid:u-day30"])
	assertWalled(t, "u-gem7 walled", hr["uuid:u-gem7"])
	assertWalled(t, "u-cap throttled", hr["uuid:u-cap"])
	if got := hr["uuid:u-unknown"]; got != 0 {
		t.Fatalf("unknown bucket = %v, want 0", got)
	}
	if _, ok := hr["uuid:u-glm"]; ok {
		t.Fatal("opencode bucket must not appear in the claude rotation headroom")
	}
}

// assertOfferable fails unless v is strictly inside the offerable band (1,2], so an offerable
// bucket always outranks unknown (0) and walled (<0).
func assertOfferable(t *testing.T, name string, v float64) {
	t.Helper()
	if v <= 1 || v > 2 {
		t.Fatalf("%s score = %v, want in (1,2]", name, v)
	}
}

// assertWalled fails unless v is inside the walled band [-1,0): the floor -1 is inclusive (no
// reset bonus), the ceiling 0 is exclusive so a walled bucket never reaches unknown/offerable.
func assertWalled(t *testing.T, name string, v float64) {
	t.Helper()
	if v < -1 || v >= 0 {
		t.Fatalf("%s score = %v, want in [-1,0)", name, v)
	}
}

// TestHeadroomOfferableLeastLoadedFirst proves the within-offerable tie-break: two offerable
// buckets with different live-session counts order least-loaded-first, and both stay above 1.
func TestHeadroomOfferableLeastLoadedFirst(t *testing.T) {
	rows := []fleetaccounts.Account{
		{Product: "claude", AccountUUID: hrStr("u-busy"), Available: hrBool(true), LiveSessions: hrInt(5)},
		{Product: "claude", AccountUUID: hrStr("u-idle"), Available: hrBool(true), LiveSessions: hrInt(0)},
	}
	hr := headroomFromRoster(rows, hrNow)
	busy, idle := hr["uuid:u-busy"], hr["uuid:u-idle"]
	if !(idle > busy) {
		t.Fatalf("idle (%v) must outrank busy (%v) among offerable buckets", idle, busy)
	}
	if busy <= 1 || idle <= 1 {
		t.Fatalf("both offerable scores must stay > 1: busy=%v idle=%v", busy, idle)
	}
}

// TestHeadroomWalledSoonestResetFirst proves the within-walled tie-break: two walled buckets
// with different reset times order soonest-reset-first, and both stay below 0. Anchored at
// 10:00 UTC, "11am" is sooner than "3pm".
func TestHeadroomWalledSoonestResetFirst(t *testing.T) {
	rows := []fleetaccounts.Account{
		{Product: "claude", AccountUUID: hrStr("u-late"), Blocked: hrBool(true), Reset: hrStr("3pm")},
		{Product: "claude", AccountUUID: hrStr("u-soon"), Blocked: hrBool(true), Reset: hrStr("11am")},
	}
	hr := headroomFromRoster(rows, hrNow)
	late, soon := hr["uuid:u-late"], hr["uuid:u-soon"]
	if !(soon > late) {
		t.Fatalf("soonest-reset (%v) must outrank later reset (%v) among walled buckets", soon, late)
	}
	if late >= 0 || soon >= 0 {
		t.Fatalf("both walled scores must stay < 0: late=%v soon=%v", late, soon)
	}
}

// TestHeadroomTierBandsNeverOverlap is the load-bearing invariant: any offerable bucket
// outranks any unknown, which outranks any walled — regardless of within-tier bonuses. A
// heavily-loaded offerable bucket must still beat a soon-to-reset walled one.
func TestHeadroomTierBandsNeverOverlap(t *testing.T) {
	rows := []fleetaccounts.Account{
		{Product: "claude", AccountUUID: hrStr("u-off"), Available: hrBool(true), LiveSessions: hrInt(99)},
		{Product: "claude", AccountUUID: hrStr("u-unk")},
		{Product: "claude", AccountUUID: hrStr("u-wall"), Blocked: hrBool(true), Reset: hrStr("11am")},
	}
	hr := headroomFromRoster(rows, hrNow)
	off, unk, wall := hr["uuid:u-off"], hr["uuid:u-unk"], hr["uuid:u-wall"]
	if !(off > unk && unk > wall) {
		t.Fatalf("tier order broken: offerable=%v unknown=%v walled=%v (want offerable>unknown>walled)", off, unk, wall)
	}
}

// TestHeadroomFromRosterBucketBestScore checks that when several dirs map to ONE account
// bucket, the best score wins — a bucket has room if ANY of its dirs can be offered.
func TestHeadroomFromRosterBucketBestScore(t *testing.T) {
	rows := []fleetaccounts.Account{
		// Same bucket u-gem7: one blocked dir, one offerable dir -> the bucket has room.
		{Product: "claude", AccountUUID: hrStr("u-gem7"), Available: hrBool(false), Blocked: hrBool(true)},
		{Product: "claude", AccountUUID: hrStr("u-gem7"), Available: hrBool(true), Blocked: hrBool(false)},
	}
	hr := headroomFromRoster(rows, hrNow)
	if got := hr["uuid:u-gem7"]; got <= 1 {
		t.Fatalf("bucket best score = %v, want > 1 (any offerable dir gives the bucket room)", got)
	}
}

func TestHeadroomWeeklyLimitDominatesBucketPeerAvailability(t *testing.T) {
	rows := []fleetaccounts.Account{
		// Same bucket u-gem8: one row carries a current weekly usage cap while a stale peer row
		// still says available. Weekly caps are account-wide, so the bucket is walled.
		{
			Product: "claude", AccountUUID: hrStr("u-gem8"), Available: hrBool(false),
			Blocked: hrBool(true), BlockKind: hrStr("usage"), Throttled: hrBool(true),
			Weekly: hrStr("Jul 2, 6am (America/Los_Angeles)"),
		},
		{Product: "claude", AccountUUID: hrStr("u-gem8"), Available: hrBool(true), Blocked: hrBool(false)},
	}
	hr := headroomFromRoster(rows, hrNow)
	assertWalled(t, "active weekly cap should wall the whole bucket", hr["uuid:u-gem8"])
}

func TestHeadroomExpiredWeeklyLimitDoesNotDominateBucketPeerAvailability(t *testing.T) {
	rows := []fleetaccounts.Account{
		{
			Product: "claude", AccountUUID: hrStr("u-gem8"), Available: hrBool(false),
			Blocked: hrBool(true), BlockKind: hrStr("usage"), Throttled: hrBool(true),
			Weekly: hrStr("Jun 29, 6am (America/Los_Angeles)"),
		},
		{Product: "claude", AccountUUID: hrStr("u-gem8"), Available: hrBool(true), Blocked: hrBool(false)},
	}
	hr := headroomFromRoster(rows, hrNow)
	if got := hr["uuid:u-gem8"]; got <= 1 {
		t.Fatalf("expired weekly cap should not dominate a fresh peer; got %v, want offerable > 1", got)
	}
}

// TestHeadroomFromRosterEmptyIsNil ensures an empty/irrelevant roster yields a nil signal, so
// the pure planner falls back to stable-by-name rather than a spurious all-zero headroom order.
func TestHeadroomFromRosterEmptyIsNil(t *testing.T) {
	if hr := headroomFromRoster(nil, hrNow); hr != nil {
		t.Fatalf("nil roster -> %v, want nil signal", hr)
	}
	if hr := headroomFromRoster([]fleetaccounts.Account{{Product: "opencode", AccountUUID: hrStr("u-glm")}}, hrNow); hr != nil {
		t.Fatalf("no-claude roster -> %v, want nil signal", hr)
	}
}

// hrIsolate points every fleet path env at one temp dir so the roster resolves EMPTY and the
// cooldown store lands somewhere writable — the verb then reads no real accounts off the dev
// box and the test is hermetic. Returns the temp dir (also the cooldown store's directory).
func hrIsolate(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for _, k := range []string{"FLEET_STATE_DIR", "FLEET_USER_HOME", "FLEET_CONFIG_HOME", "FLEET_REG_DIR", "FLEET_POLICY_DIR"} {
		t.Setenv(k, dir)
	}
	t.Setenv("FLEET_POLICY_PATH", filepath.Join(dir, "accounts_policy.json"))
	return dir
}

// hrCool writes an ACTIVE usage-limit cooldown for account into the fleet-shared store that
// defaultCooldownStorePath resolves under the isolated FLEET_STATE_DIR.
func hrCool(t *testing.T, account string, now time.Time) {
	t.Helper()
	cd, err := accounts.LoadCooldownStore(defaultCooldownStorePath())
	if err != nil {
		t.Fatalf("load cooldown store: %v", err)
	}
	cd.Cool(account, accounts.CooldownUsageLimit, "weekly limit reached", now, now.Add(time.Hour))
	if err := cd.Save(); err != nil {
		t.Fatalf("save cooldown store: %v", err)
	}
}

// TestAccountsHeadroomVerbHonorsCooldown is the regression for #5853: `fak accounts headroom`
// used to call headroomFromRoster bare, skipping the cooldown override rotationHeadroom
// applies, so it printed a bucket `fak accounts cooldown` reported walled as offerable/absent.
// Both paths now fold through rosterHeadroom, so the two verbs agree.
func TestAccountsHeadroomVerbHonorsCooldown(t *testing.T) {
	hrIsolate(t)
	const bucket = "uuid:u-cooled"
	hrCool(t, bucket, time.Now().UTC())

	var stdout, stderr bytes.Buffer
	if code := runAccountsHeadroom(&stdout, &stderr, []string{"--required", "1"}); code != 0 {
		t.Fatalf("runAccountsHeadroom = %d, stderr=%s", code, stderr.String())
	}
	var payload accountsHeadroomPayload
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatalf("decode payload %q: %v", stdout.String(), err)
	}
	got, ok := payload.Headroom[bucket]
	if !ok {
		t.Fatalf("cooled bucket %q missing from headroom %v — the verb is cooldown-blind", bucket, payload.Headroom)
	}
	if got >= 0 {
		t.Fatalf("cooled bucket %q scored %v, want walled (<0) — headroom must not offer a cooled seat", bucket, got)
	}
}

// TestRosterHeadroomWallsCooledOfferableBucket pins the exact reported misread: the live
// roster still says the seat is available (an offerable +1.x read), while the durable store
// holds an active usage-limit cooldown. The cooldown is the fresher, certain signal, so the
// single fold must land the bucket in the walled band.
func TestRosterHeadroomWallsCooledOfferableBucket(t *testing.T) {
	hrIsolate(t)
	now := hrNow
	hrCool(t, "uuid:u-cooled", now)

	rows := []fleetaccounts.Account{
		{Product: "claude", AccountUUID: hrStr("u-cooled"), Available: hrBool(true), LiveSessions: hrInt(0)},
		{Product: "claude", AccountUUID: hrStr("u-open"), Available: hrBool(true), LiveSessions: hrInt(0)},
	}
	hr := rosterHeadroom(rows, now)
	assertWalled(t, "cooled bucket must be walled despite an offerable roster row", hr["uuid:u-cooled"])
	assertOfferable(t, "uncooled bucket keeps its roster read", hr["uuid:u-open"])
}

func TestBuildAccountsSeatDeficitUnderCapacity(t *testing.T) {
	t.Setenv(fleetaccounts.SessionsPerAccountEnv, "2")
	available := true
	uuid1, uuid2 := "u1", "u2"
	rows := []fleetaccounts.Account{
		{Kind: fleetaccounts.KindWorker, Product: "claude", Tag: "a", AccountUUID: &uuid1, Available: &available},
		{Kind: fleetaccounts.KindWorker, Product: "claude", Tag: "b", AccountUUID: &uuid2, Available: &available},
	}
	got := buildAccountsSeatDeficit(rows, "claude", 7)
	if got.Required != 7 || got.FreshCeiling != 2 || got.Shortfall != 5 || got.Verdict != "UNDER_CAPACITY" {
		t.Fatalf("got=%+v", got)
	}
}

func TestBuildAccountsSeatDeficitSufficient(t *testing.T) {
	t.Setenv(fleetaccounts.SessionsPerAccountEnv, "3")
	available := true
	uuid := "u1"
	got := buildAccountsSeatDeficit([]fleetaccounts.Account{{Kind: fleetaccounts.KindWorker, Product: "claude", Tag: "a", AccountUUID: &uuid, Available: &available}}, "claude", 1)
	if got.FreshCeiling != 1 || got.Shortfall != 0 || got.Verdict != "OK" {
		t.Fatalf("got=%+v", got)
	}
}

func TestAccountsRequiredDemandFromForecast(t *testing.T) {
	t.Setenv("FAK_FLEET_TARGET_IPH", "60")
	t.Setenv("FAK_FLEET_SESSION_MIN", "10")
	if got := accountsRequiredDemandFromEnv(); got != 10 {
		t.Fatalf("required=%d, want 10", got)
	}
	t.Setenv("FAK_FLEET_TARGET_IPH", "")
	if got := accountsRequiredDemandFromEnv(); got != 0 {
		t.Fatalf("unset required=%d, want 0", got)
	}
}
