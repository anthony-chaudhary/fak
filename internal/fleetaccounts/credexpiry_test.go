package fleetaccounts

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// credexpiry_test.go — the #2059/#2075 witnesses. Before the credential-expiry gate,
// a seat whose OAuth token had already expired (but had no auth-blocked session in
// the registry) annotated available=true: the pool offered it and every spawn died
// STALE_CRED at the guard. These tests fail on that pre-fix fold and pass with the
// gate + the dropped-seat surfacing in place.

// addClaudeSeat scaffolds one Claude worker dir under home with the given identity
// and .credentials.json body ("" skips the credential file), returning the dir.
func addClaudeSeat(t *testing.T, home, dirName, uuid, email, credBody string) string {
	t.Helper()
	acctDir := filepath.Join(home, dirName)
	if err := os.MkdirAll(filepath.Join(acctDir, "projects"), 0o755); err != nil {
		t.Fatal(err)
	}
	identity := fmt.Sprintf(
		`{"oauthAccount":{"accountUuid":%q,"emailAddress":%q,"organizationUuid":"org-x","organizationType":"team"}}`,
		uuid, email)
	if err := os.WriteFile(filepath.Join(acctDir, ".claude.json"), []byte(identity), 0o644); err != nil {
		t.Fatal(err)
	}
	if credBody != "" {
		if err := os.WriteFile(filepath.Join(acctDir, ".credentials.json"), []byte(credBody), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return acctDir
}

func expiredCredBody(t *testing.T, until time.Duration) string {
	t.Helper()
	return fmt.Sprintf(`{"claudeAiOauth":{"expiresAt":%d}}`, time.Now().Add(until).UnixMilli())
}

// TestExpiredCredentialSeatNeedsLogin is the #2059 done-condition: an expired OAuth
// credential with no setup-token fallback and no fresh probe-OK annotates the seat
// blocked/auth needs-login — never free — and the capacity preflight counts it stale
// so a dispatcher downsizes instead of spawning a doomed worker.
func TestExpiredCredentialSeatNeedsLogin(t *testing.T) {
	home, cfg, regPath := fixture(t)
	addClaudeSeat(t, home, ".claude-stale-acct", "uuid-stale", "stale@example.com",
		expiredCredBody(t, -2*time.Hour))

	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), LoadRegistry(regPath))
	r := find(rows, ".claude-stale-acct")
	if r == nil {
		t.Fatal("stale seat not discovered")
	}
	if derefBool(r.Available) || !derefBool(r.Blocked) || derefStr(r.BlockKind) != "auth" {
		t.Fatalf("expired seat annotated available=%v blocked=%v kind=%q, want unavailable auth block",
			derefBool(r.Available), derefBool(r.Blocked), derefStr(r.BlockKind))
	}
	if derefStr(r.LoginStatus) != "needs_login" || derefBool(r.CanServe) {
		t.Fatalf("login_status/can_serve = %q/%v, want needs_login/false",
			derefStr(r.LoginStatus), derefBool(r.CanServe))
	}
	if !strings.Contains(derefStr(r.BlockReason), "credential expired") {
		t.Fatalf("block_reason = %q, want the credential-expired needs-login text", derefStr(r.BlockReason))
	}
	for _, a := range Available(rows) {
		if a.Account == ".claude-stale-acct" {
			t.Fatal("expired-credential seat was offered as available")
		}
	}
	rep := BuildCapacityPreflight(rows, "claude", 0)
	if got := byAccount(rep)[".claude-stale-acct"]; got.State != CapacityStale {
		t.Fatalf("capacity state = %+v, want stale (out of the true concurrency ceiling)", got)
	}
}

// TestExpiredCredentialSetupTokenFallbackStaysServeable: a seat with an expired
// interactive credential but a long-lived setup token still serves headless — it
// must not be tombstoned to needs-login.
func TestExpiredCredentialSetupTokenFallbackStaysServeable(t *testing.T) {
	home, cfg, _ := fixture(t)
	acctDir := addClaudeSeat(t, home, ".claude-tok-acct", "uuid-tok", "tok@example.com",
		expiredCredBody(t, -2*time.Hour))
	if err := os.WriteFile(filepath.Join(acctDir, ".oauth-token"), []byte("sk-ant-oat01-fixture\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), Registry{})
	r := find(rows, ".claude-tok-acct")
	if r == nil || !derefBool(r.Available) || derefStr(r.LoginStatus) != "ready" {
		t.Fatalf("setup-token seat = %+v, want available/ready despite the expired interactive credential", r)
	}
}

// TestFutureExpiryStaysServeable: a live (future-dated) credential is untouched by
// the gate.
func TestFutureExpiryStaysServeable(t *testing.T) {
	home, cfg, _ := fixture(t)
	addClaudeSeat(t, home, ".claude-live-acct", "uuid-live", "live@example.com",
		expiredCredBody(t, 2*time.Hour))

	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), Registry{})
	r := find(rows, ".claude-live-acct")
	if r == nil || !derefBool(r.Available) || derefBool(r.Blocked) {
		t.Fatalf("future-expiry seat = %+v, want available", r)
	}
}

// TestFreshProbeOKOverridesExpiredCredential is #2059's tombstone guard: the refresh
// token can silently re-mint an access token, so a fresh active probe that just hit
// the live account outranks the stale on-disk expiry instant.
func TestFreshProbeOKOverridesExpiredCredential(t *testing.T) {
	home, cfg, _ := fixture(t)
	addClaudeSeat(t, home, ".claude-stale-acct", "uuid-stale", "stale@example.com",
		expiredCredBody(t, -2*time.Hour))
	reg := Registry{
		GeneratedUTC: time.Now().UTC().Format("2006-01-02T15:04:05Z"),
		Sessions: []Session{{
			Account: ".claude-stale-acct", Project: "_probe", Disp: "LIVE",
			ProbeStatus: "OK",
		}},
	}

	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), reg)
	r := find(rows, ".claude-stale-acct")
	if r == nil || !derefBool(r.Available) || derefStr(r.StatusSource) != "probe" {
		t.Fatalf("probe-OK seat = available=%v source=%q, want available via the fresh probe",
			derefBool(r.Available), derefStr(r.StatusSource))
	}
}

// TestDroppedSeatSurfacedInStatus is the #2075 done-condition, half one: a simulated
// stale-credential seat is reported as a dropped seat in the status output with the
// exact re-login prompt.
func TestDroppedSeatSurfacedInStatus(t *testing.T) {
	home, cfg, regPath := fixture(t)
	acctDir := addClaudeSeat(t, home, ".claude-stale-acct", "uuid-stale", "stale@example.com",
		expiredCredBody(t, -2*time.Hour))

	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), LoadRegistry(regPath))
	dropped := DroppedSeats(rows)
	if len(dropped) != 1 || dropped[0].Tag != "stale" {
		t.Fatalf("dropped seats = %+v, want exactly the stale seat", dropped)
	}
	if !strings.Contains(dropped[0].NextAction, "claude /login") ||
		!strings.Contains(dropped[0].NextAction, acctDir) {
		t.Fatalf("next_action = %q, want the CLAUDE_CONFIG_DIR re-login prompt", dropped[0].NextAction)
	}
	out := RenderList(rows, home, "", false, "")
	if !strings.Contains(out, "NEEDS RE-LOGIN") || !strings.Contains(out, "credential expired") {
		t.Fatalf("status render missing the dropped-seat re-login section:\n%s", out)
	}
	// A usage-throttled seat (fixture gem8) heals at reset — it is NOT a dropped seat.
	for _, d := range dropped {
		if d.Tag == "gem8" {
			t.Fatal("throttled seat misreported as needing re-login")
		}
	}
}

// TestWaveFanoutReflectsDroppedSeat is the #2075 done-condition, half two: a wave
// sized against the roster grants only the live seats, and the shrink is named in
// the wave reason instead of passing silently.
func TestWaveFanoutReflectsDroppedSeat(t *testing.T) {
	root := t.TempDir()
	home := filepath.Join(root, "home")
	cfg := filepath.Join(root, "cfg")
	for _, d := range []string{home, cfg} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	addClaudeSeat(t, home, ".claude-alpha-acct", "uuid-alpha", "alpha@example.com", `{}`)
	addClaudeSeat(t, home, ".claude-stale-acct", "uuid-stale", "stale@example.com",
		expiredCredBody(t, -2*time.Hour))

	rows := AnnotatedRoster(home, cfg, DefaultPolicy(), Registry{})
	wave := AllocateWave(rows, WaveRequest{Count: 2, Product: "claude"}, DefaultPolicy())
	if wave.Granted != 1 || wave.Shortfall != 1 {
		t.Fatalf("wave granted/shortfall = %d/%d, want 1/1 (the stale seat is out of the pool)",
			wave.Granted, wave.Shortfall)
	}
	for _, lane := range wave.Lanes {
		if lane.Account == ".claude-stale-acct" {
			t.Fatal("wave allocated the expired-credential seat")
		}
	}
	if len(wave.DroppedSeats) != 1 || wave.DroppedSeats[0].Tag != "stale" {
		t.Fatalf("wave dropped_seats = %+v, want the stale seat surfaced", wave.DroppedSeats)
	}
	if !strings.Contains(wave.Reason, "re-login") {
		t.Fatalf("wave reason = %q, want the pending re-login note", wave.Reason)
	}
}
