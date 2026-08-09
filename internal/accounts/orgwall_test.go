package accounts

// orgwall_test.go — the behavioral witnesses for #4998.
//
// The defect these pin is not "a 403 was misread". It is that a STRONGER, witnessed
// cause (the organization is walled upstream) was displaced by a WEAKER, wrong one
// (needs_login) as soon as Claude blanked the seat's tokens — so `fak accounts doctor`
// prescribed `/login`, the one repair the original terminal 403 had already proven
// futile. Each test below is one acceptance criterion from the issue.

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// liveOrgWallBody is the exact 403 body captured from the live gateway when the seat's
// organization had OAuth disabled. It is duplicated verbatim from
// internal/agent.liveOrgOAuthDisabledBody — internal/agent is tier 4 and cannot be
// imported by this tier-1 leaf (ARCH_LAYER_VIOLATION), so the string is restated here
// and pinned by TestOrgAuthWallSignatureCoversWitnessedBodies. It is the canonical
// "deceiving error": the prose reads as if a re-login would help, and it never would.
const liveOrgWallBody = `{"type":"error","error":{"type":"permission_error",` +
	`"message":"OAuth authentication is currently not allowed for this organization."},"request_id":"[redacted]"}`

// TestOrgAuthWallSignatureCoversWitnessedBodies pins every phrasing the fleet has
// actually witnessed for an organization-scoped wall, plus the near-miss bodies that
// must NOT be read as one. orgwall.go names this test as the drift alarm between the
// three independent readers of this signature (internal/agent, internal/fleetaccounts,
// and this leaf), so a future edit to any one of them fails here instead of silently
// mis-triaging a seat.
func TestOrgAuthWallSignatureCoversWitnessedBodies(t *testing.T) {
	walled := []struct{ name, body string }{
		{"live gateway 403 (api wording)", liveOrgWallBody},
		{"org subscription disabled", `{"error":{"message":"Your organization has disabled Claude subscription access"}}`},
		{"cli banner: api key instead", "Please use an Anthropic API key instead."},
		{"cli banner: ask your admin", "Ask your admin to enable access for this organization."},
	}
	for _, tc := range walled {
		t.Run(tc.name, func(t *testing.T) {
			if !orgAuthWallRE.MatchString(tc.body) {
				t.Fatalf("witnessed org-wall body not matched by orgAuthWallRE: %q", tc.body)
			}
		})
	}

	// The near misses. Each of these is a DIFFERENT cause with a different repair, and
	// folding any of them into the wall would durably exclude a recoverable seat.
	notWalled := []struct{ name, body string }{
		{"expired credential", `{"error":{"message":"OAuth session expired and could not be refreshed"}}`},
		{"weekly usage cap", `{"error":{"message":"You have reached your weekly limit"}}`},
		{"credit balance", `{"error":{"message":"Your credit balance is too low"}}`},
		{"model entitlement", `{"error":{"message":"This model is not available for your account"}}`},
	}
	for _, tc := range notWalled {
		t.Run("not-a-wall/"+tc.name, func(t *testing.T) {
			if orgAuthWallRE.MatchString(tc.body) {
				t.Fatalf("non-org-wall body wrongly matched orgAuthWallRE: %q", tc.body)
			}
		})
	}
}

// TestClassifySeatHealthSeparatesCapFromWall is the load-bearing fence. The SAME 403
// prose covers a standing organization wall and a self-recovering rolling-window usage
// cap; only the rate-limit headers (or a named cap/reset in the text) tell them apart.
// Misfiling a cap as a wall would durably exclude an account that returns on its own,
// so the ordering in ClassifySeatHealth is pinned here rather than left to review.
func TestClassifySeatHealthFencesRelayBorrowed401(t *testing.T) {
	now := time.Date(2026, 7, 15, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name string
		body string
		want SeatHealth
	}{
		{"upstream provider block", `{"error":"blocked by upstream provider"}`, SeatHealthUnknown},
		{"upstream request block is case insensitive", `REQUEST BLOCKED BY UPSTREAM`, SeatHealthUnknown},
		{"expired credential remains login", `{"error":"OAuth session expired"}`, SeatHealthNeedsLogin},
		{"unrelated upstream text remains login", `upstream relay rejected expired token`, SeatHealthNeedsLogin},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifySeatHealth(http.StatusUnauthorized, []byte(tc.body), nil, now); got != tc.want {
				t.Fatalf("ClassifySeatHealth(401, %q) = %q, want %q", tc.body, got, tc.want)
			}
		})
	}
}

func TestClassifySeatHealthSeparatesCapFromWall(t *testing.T) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	org := []byte(liveOrgWallBody)

	overage := http.Header{}
	overage.Set("Anthropic-Ratelimit-Unified-Status", "allowed")
	overage.Set("Anthropic-Ratelimit-Unified-Overage-Status", "rejected")

	windowRejected := http.Header{}
	windowRejected.Set("Anthropic-Ratelimit-Unified-5h-Status", "rejected")

	allowed := http.Header{}
	allowed.Set("Anthropic-Ratelimit-Unified-Status", "allowed")

	cases := []struct {
		name   string
		status int
		body   []byte
		hdr    http.Header
		want   SeatHealth
	}{
		{"2xx is the only positive witness", 200, []byte(`{"ok":true}`), nil, SeatHealthReady},
		{"429 is always a cap", 429, []byte(`{}`), nil, SeatHealthUsageLimited},
		{"401 is always the credential", 401, org, nil, SeatHealthNeedsLogin},
		{"bare org 403 is the wall", 403, org, allowed, SeatHealthOrgAuthWall},
		{"org 403 + overage-rejected header is a CAP", 403, org, overage, SeatHealthUsageLimited},
		{"org 403 + window-rejected header is a CAP", 403, org, windowRejected, SeatHealthUsageLimited},
		{"403 naming a weekly limit is a CAP", 403, []byte(`{"error":{"message":"weekly limit reached"}}`), nil, SeatHealthUsageLimited},
		{"403 with an expired token is needs_login", 403, []byte(`{"error":{"message":"OAuth session expired"}}`), nil, SeatHealthNeedsLogin},
		{"unrecognized 403 stays unknown", 403, []byte(`{"error":{"message":"teapot"}}`), nil, SeatHealthUnknown},
		{"500 stays unknown", 500, []byte(`upstream exploded`), nil, SeatHealthUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifySeatHealth(tc.status, tc.body, tc.hdr, now); got != tc.want {
				t.Fatalf("ClassifySeatHealth(%d) = %q, want %q", tc.status, got, tc.want)
			}
		})
	}
}

// TestCanarySeatClassifiesLiveOrgWallFixture is acceptance criterion 1: a fixture that
// returns the known OAuth/subscription-disabled 403 classifies as org_auth_wall at the
// REAL probe seam — the actual HTTP round-trip CanarySeat performs, not a hand-called
// classifier. It also pins the credential-safety contract: the returned detail carries
// the status only, never the upstream body, and never the token.
func TestCanarySeatClassifiesLiveOrgWallFixture(t *testing.T) {
	var gotAuth, gotVersion string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotVersion = r.Header.Get("anthropic-version")
		w.WriteHeader(http.StatusForbidden)
		w.Write([]byte(liveOrgWallBody))
	}))
	defer srv.Close()

	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	health, detail, err := CanarySeat(srv.Client(), srv.URL, "tok-secret-value", "", now)
	if err != nil {
		t.Fatalf("CanarySeat: %v", err)
	}
	if health != SeatHealthOrgAuthWall {
		t.Fatalf("live org-wall fixture classified as %q, want %q", health, SeatHealthOrgAuthWall)
	}
	if detail != "http 403" {
		t.Fatalf("detail = %q, want the bare status (never the upstream body)", detail)
	}
	// Credential safety: the classified detail must not leak the body or the token.
	if strings.Contains(detail, "organization") || strings.Contains(detail, "tok-secret-value") {
		t.Fatalf("detail leaked upstream prose or credential material: %q", detail)
	}
	if gotAuth != "Bearer tok-secret-value" || gotVersion != anthropicVersion {
		t.Fatalf("canary sent auth=%q version=%q, want a bearer token and the pinned api version", gotAuth, gotVersion)
	}

	// A seat with no credential never spends a round-trip, and never files a wall.
	if h, _, err := CanarySeat(srv.Client(), srv.URL, "  ", "", now); err != nil || h != SeatHealthNeedsLogin {
		t.Fatalf("empty token = (%q, %v), want needs_login with no error", h, err)
	}
}

// walledSeatFixture builds the seat exactly as #4998 observed it AFTER the degradation:
// the config directory is present and the cached profile identity survives (so the
// canonical account key still resolves), but Claude has blanked the OAuth tokens, so the
// pure LoginStatus() fold now returns needs_login. This is the state in which doctor used
// to prescribe the futile `/login`.
func walledSeatFixture() (Registry, string) {
	seat := Home{
		Name:   "netra",
		Dir:    "/home/netra",
		Status: StatusActive,
		// HasCreds false is the whole point: the tokens are gone, the identity is not.
		Identity: Identity{Exists: true, HasCreds: false, AccountUUID: "u-netra", Email: "netra@example.test"},
	}
	return Registry{Homes: []Home{seat}}, UUIDBucketKey("u-netra")
}

// TestLoginReportAtOrgWallOutranksNeedsLogin is the core #4998 regression — acceptance
// criterion 3. A durable org wall must keep its diagnosis EVEN AFTER Claude empties the
// local tokens and the seat degrades to needs_login, and the rendered repair must never
// name `/login`.
//
// FAILING BEFORE: loginObservation applied the cooldown overlay only to an otherwise
// LoginReady seat, so a blanked seat skipped the overlay entirely and reported
// needs_login with "run /login for this CLAUDE_CONFIG_DIR".
func TestLoginReportAtOrgWallOutranksNeedsLogin(t *testing.T) {
	reg, account := walledSeatFixture()
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)

	// Control: with no wall recorded, the blanked seat legitimately IS needs_login and
	// `/login` is the right repair. Without this half the test could pass vacuously.
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	obs := reg.LoginReportAt(cd, now).Seats[0]
	if obs.Status != LoginNeedsLogin {
		t.Fatalf("control: blanked seat with no wall = %q, want needs_login", obs.Status)
	}
	if !strings.Contains(obs.NextAction, "/login") {
		t.Fatalf("control: a genuine needs_login must still prescribe /login, got %q", obs.NextAction)
	}

	// Now witness the upstream wall, exactly as the guard would on the terminal 403.
	if _, ok := cd.RecordOrgAuthWall(account, string(SeatHealthOrgAuthWall), now); !ok {
		t.Fatal("RecordOrgAuthWall did not record")
	}

	obs = reg.LoginReportAt(cd, now.Add(time.Hour)).Seats[0]
	if obs.Status != LoginOrgAuthWall {
		t.Fatalf("walled seat = %q, want %q — the weaker needs_login displaced the witnessed org wall (#4998)",
			obs.Status, LoginOrgAuthWall)
	}
	if obs.CanServe {
		t.Fatal("a walled seat must never report CanServe")
	}
	if strings.Contains(obs.NextAction, "/login") {
		t.Fatalf("doctor prescribed the futile repair for an org wall: %q", obs.NextAction)
	}
	// The repair must name the ACCOUNT-level escapes the issue enumerates.
	for _, want := range []string{"organization", "API key"} {
		if !strings.Contains(obs.Reason+" "+obs.NextAction, want) {
			t.Fatalf("org-wall repair never mentions %q: reason=%q action=%q", want, obs.Reason, obs.NextAction)
		}
	}
	// The observation is credential-safe and keeps the canonical identity binding.
	if obs.Account != account {
		t.Fatalf("observation account = %q, want the canonical key %q", obs.Account, account)
	}
}

// TestOrgAuthWallOutranksNeedsLoginForReadySeat pins the other half: a seat whose tokens
// are still present is ALSO walled, and the wall outranks a plain cooldown rendering.
func TestOrgAuthWallOutranksNeedsLoginForReadySeat(t *testing.T) {
	reg := Registry{Homes: []Home{active("netra", "u-netra", "netra@example.test")}}
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	cd.RecordOrgAuthWall(UUIDBucketKey("u-netra"), string(SeatHealthOrgAuthWall), now)

	obs := reg.LoginReportAt(cd, now).Seats[0]
	if obs.Status != LoginOrgAuthWall || obs.CanServe {
		t.Fatalf("live-token walled seat = %q canServe=%v, want org_auth_wall and no service", obs.Status, obs.CanServe)
	}
	if strings.Contains(obs.NextAction, "wait for the cooldown window") {
		t.Fatalf("an org wall must not render as a self-recovering cooldown: %q", obs.NextAction)
	}
}

// TestOrgAuthWallDoesNotOverrideDeliberateLifecycle keeps the override narrow. Tombstoned
// and disabled are deliberate OPERATOR states whose repair is more actionable than the
// wall's; only the ready/needs_login pair — the states the #4998 degradation actually
// walks between — may be overridden.
func TestOrgAuthWallDoesNotOverrideDeliberateLifecycle(t *testing.T) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	disabled := active("off", "u-off", "off@example.test")
	disabled.Enabled = boolp(false)

	reg := Registry{Homes: []Home{
		disabled,
		{Name: "old", Status: StatusTombstoned, RehomeTo: "off",
			Identity: Identity{Exists: true, HasCreds: true, AccountUUID: "u-old"}},
	}}
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	cd.RecordOrgAuthWall(UUIDBucketKey("u-off"), string(SeatHealthOrgAuthWall), now)
	cd.RecordOrgAuthWall(UUIDBucketKey("u-old"), string(SeatHealthOrgAuthWall), now)

	for _, obs := range reg.LoginReportAt(cd, now).Seats {
		if obs.Status == LoginOrgAuthWall {
			t.Fatalf("seat %q: a deliberate lifecycle state was overwritten by the org wall", obs.Name)
		}
	}
}

// TestOrgAuthWallSurvivesNewProcess is acceptance criterion 2: the typed result outlives
// the process that witnessed it and still excludes the same canonical account. The wall
// is written, saved, and re-read through a genuinely separate store — the cross-process
// path — and must still be active well past the reprobe deadline, because an org wall
// never lapses on its own timer the way a usage cap does.
func TestOrgAuthWallSurvivesNewProcess(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cooldown.json")
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	reg, account := walledSeatFixture()

	// Process 1: the guard witnesses the terminal 403 and persists the typed evidence.
	first, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("LoadCooldownStore: %v", err)
	}
	if changed := first.ObserveSeatHealth(account, SeatHealthOrgAuthWall, now); !changed {
		t.Fatal("first org-wall observation must report a state change")
	}
	if err := first.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// The persisted file must carry the typed label and NO token material or raw body.
	raw := readFileString(t, path)
	if !strings.Contains(raw, string(CooldownOrgAuthWall)) {
		t.Fatalf("persisted store lost the typed org-auth-wall kind:\n%s", raw)
	}
	if strings.Contains(raw, "OAuth authentication is currently not allowed") {
		t.Fatalf("persisted store leaked the raw upstream body:\n%s", raw)
	}

	// Process 2: a brand-new store, and a clock LONG past the reprobe window.
	later := now.Add(30 * 24 * time.Hour)
	second, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	e, ok := second.OrgAuthWall(account, later)
	if !ok {
		t.Fatal("the org wall did not survive the process boundary — a timer must never clear it")
	}
	if !e.ReprobeDue(later) {
		t.Fatal("a month-old wall must be reprobe-due")
	}
	// Exclusion, not merely a record: the seat is still dropped from the servable pool.
	obs := reg.LoginReportAt(second, later).Seats[0]
	if obs.Status != LoginOrgAuthWall || obs.CanServe {
		t.Fatalf("after reload seat = %q canServe=%v, want a durable org_auth_wall exclusion", obs.Status, obs.CanServe)
	}
	if strings.Contains(obs.NextAction, "/login") {
		t.Fatalf("reloaded wall regressed to the futile repair: %q", obs.NextAction)
	}
}

// TestSuccessfulCanaryClearsOrgWall is acceptance criterion 4: a later healthy probe —
// and ONLY a healthy probe — returns the seat to service. A fresh needs_login or a usage
// cap is not evidence the organization was repaired; treating either as a clear is
// exactly the displacement #4998 is about.
func TestSuccessfulCanaryClearsOrgWall(t *testing.T) {
	reg, account := walledSeatFixture()
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	cd.RecordOrgAuthWall(account, string(SeatHealthOrgAuthWall), now)

	// Non-witnesses must leave the wall standing.
	for _, h := range []SeatHealth{SeatHealthNeedsLogin, SeatHealthUsageLimited, SeatHealthUnknown} {
		if changed := cd.ObserveSeatHealth(account, h, now.Add(time.Hour)); changed {
			t.Fatalf("%q wrongly reported a state change — it is not evidence of an upstream repair", h)
		}
		if _, ok := cd.OrgAuthWall(account, now.Add(time.Hour)); !ok {
			t.Fatalf("%q cleared the org wall; only a healthy round-trip may", h)
		}
	}

	// The witnessed repair: one successful canary.
	repaired := now.Add(48 * time.Hour)
	if changed := cd.ObserveSeatHealth(account, SeatHealthReady, repaired); !changed {
		t.Fatal("a successful canary must report the wall clearing")
	}
	if _, ok := cd.OrgAuthWall(account, repaired); ok {
		t.Fatal("the org wall survived a witnessed healthy round-trip")
	}

	// Eligibility is restored — but this seat's tokens are still blanked, so the honest
	// status is the ordinary needs_login, with `/login` correct again.
	obs := reg.LoginReportAt(cd, repaired).Seats[0]
	if obs.Status != LoginNeedsLogin {
		t.Fatalf("after the repair seat = %q, want the ordinary needs_login back", obs.Status)
	}
	if !strings.Contains(obs.NextAction, "/login") {
		t.Fatalf("once the wall is gone /login is the correct repair again, got %q", obs.NextAction)
	}

	// A seat whose tokens are intact returns to full service.
	live := Registry{Homes: []Home{active("netra", "u-netra", "netra@example.test")}}
	if o := live.LoginReportAt(cd, repaired).Seats[0]; o.Status != LoginReady || !o.CanServe {
		t.Fatalf("a repaired org returns a credentialed seat to service, got %q canServe=%v", o.Status, o.CanServe)
	}
}

// TestClearOrgAuthWallLeavesSiblingCooldown pins that clearing a wall never re-admits a
// still-capped account: the two signals are independent, and a witnessed org repair says
// nothing about a rolling usage window.
func TestClearOrgAuthWallLeavesSiblingCooldown(t *testing.T) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	account := UUIDBucketKey("u-netra")
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	cd.Cool(account, CooldownUsageLimit, "weekly limit", now, now.Add(5*time.Hour))
	cd.RecordOrgAuthWall(account, string(SeatHealthOrgAuthWall), now)

	if !cd.ClearOrgAuthWall(account, now.Add(time.Hour)) {
		t.Fatal("ClearOrgAuthWall reported no wall to clear")
	}
	if _, ok := cd.CooledDown(account, now.Add(time.Hour)); !ok {
		t.Fatal("clearing the org wall wrongly dropped the sibling usage cooldown")
	}
	// And once that window elapses the account is free, since the wall is gone.
	if _, ok := cd.CooledDown(account, now.Add(6*time.Hour)); ok {
		t.Fatal("with the wall cleared the usage cooldown must expire on its own timer")
	}
}

// TestRecordOrgAuthWallRefusesEmptyAccount keeps a seat with no derivable identity from
// walling the empty bucket, which would exclude every other identity-less seat with it.
func TestRecordOrgAuthWallRefusesEmptyAccount(t *testing.T) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	cd := &CooldownStore{entries: map[string]CooldownEntry{}}
	if _, ok := cd.RecordOrgAuthWall("   ", "reason", now); ok {
		t.Fatal("an empty account key must never record a wall")
	}
	if cd.ObserveSeatHealth("", SeatHealthOrgAuthWall, now) {
		t.Fatal("an empty account key must never observe through to a wall")
	}
	if len(cd.entries) != 0 {
		t.Fatalf("store gained %d phantom entries", len(cd.entries))
	}
}

// readFileString reads path for an assertion, failing the test rather than the caller.
func readFileString(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	// Round-trip through the store schema so a malformed file fails loudly here.
	var f cooldownFile
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatalf("persisted store is not valid %s: %v", CooldownStoreSchema, err)
	}
	return string(raw)
}
