package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// doctorTestRegistry writes a registry with an anchor seat, a creds-less seat, and a
// seat whose config dir does not exist, returning the registry path.
func doctorTestRegistry(t *testing.T, home string) string {
	t.Helper()
	anchor := mkHome(t, home, ".claude-anchor-seat", "anchor@example.test", true)
	needs := mkHome(t, home, ".claude-needs-seat", "needs@example.test", false)
	gone := filepath.Join(home, ".claude-gone-seat") // never created on disk
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"anchor-seat","dir":"` + jsonPath(anchor) + `"},` +
		`{"name":"needs-seat","dir":"` + jsonPath(needs) + `"},` +
		`{"name":"gone-seat","dir":"` + jsonPath(gone) + `"}` +
		`],"roles":{"active":"anchor-seat","anchor":"anchor-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	return regPath
}

func TestAccountsDoctorReportsClosedActions(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	home := t.TempDir()
	regPath := doctorTestRegistry(t, home)

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("doctor with actionable seats rc=%d, want 1; stderr=%s\nout=%s", rc, errb.String(), out.String())
	}
	got := out.String()
	for _, want := range []string{"fak.accounts.doctor.v1", "prune", "relogin",
		"CLAUDE_CONFIG_DIR=", "fak accounts remove --name gone-seat",
		"auto-fixable: 1", "doctor --write"} {
		if !strings.Contains(got, want) {
			t.Fatalf("doctor output missing %q:\n%s", want, got)
		}
	}
}

func TestAccountsDoctorWritePrunesMissingDir(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_JOB_ROSTER", "")
	home := t.TempDir()
	regPath := doctorTestRegistry(t, home)

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--write", "--registry", regPath, "--home", home})
	// The relogin seat still needs an operator, so the exit stays 1 — but the vanished
	// dir must have been tombstoned through the audited remove path.
	if rc != 1 {
		t.Fatalf("doctor --write rc=%d, want 1 (relogin remains); stderr=%s\nout=%s", rc, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "APPLIED") {
		t.Fatalf("doctor --write should report the applied repair:\n%s", out.String())
	}
	reg, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("registry after doctor --write does not load: %v", err)
	}
	found := false
	for _, h := range reg.Homes {
		if h.Name == "gone-seat" {
			found = true
			if h.Active() || h.RehomeTo != "anchor-seat" {
				t.Fatalf("gone-seat = status %q rehome %q, want tombstoned -> anchor-seat", h.Status, h.RehomeTo)
			}
		}
	}
	if !found {
		t.Fatalf("gone-seat missing from registry after doctor --write")
	}

	// A second doctor pass sees the tombstone as retired (action none): the repair
	// converges instead of re-firing.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home}); rc != 1 {
		t.Fatalf("second doctor rc=%d, want 1 (relogin remains)", rc)
	}
	if strings.Contains(out.String(), "prune") {
		t.Fatalf("second doctor should not re-propose the applied prune:\n%s", out.String())
	}
}

func TestAccountsDoctorCleanRegistryExitsZero(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	home := t.TempDir()
	ready := mkHome(t, home, ".claude-ready-seat", "ready@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"ready-seat","dir":"` + jsonPath(ready) + `"}` +
		`],"roles":{"active":"ready-seat","anchor":"ready-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("doctor on a clean registry rc=%d, want 0; out=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "actionable: 0") {
		t.Fatalf("clean doctor output:\n%s", out.String())
	}
}

func TestAccountsDoctorProbeLedgerOverlay(t *testing.T) {
	home := t.TempDir()
	ready := mkHome(t, home, ".claude-ready-seat", "ready@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"ready-seat","dir":"` + jsonPath(ready) + `"}` +
		`],"roles":{"active":"ready-seat","anchor":"ready-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	line := `{"ts":"` + time.Now().UTC().Format(time.RFC3339) + `","account":".claude-ready-seat","status":"LIMIT","reset":"3pm"}`
	if err := os.WriteFile(filepath.Join(rd, "probe_ledger.jsonl"), []byte(line+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("doctor with a fresh LIMIT probe rc=%d, want 1; out=%s", rc, out.String())
	}
	got := out.String()
	if !strings.Contains(got, "wait_reset") || !strings.Contains(got, "resets 3pm") {
		t.Fatalf("doctor should fold the fresh usage limit into wait_reset:\n%s", got)
	}
}

func TestAccountsDoctorAccessWallIsNotRelogin(t *testing.T) {
	// An upstream ACCESS wall (org disabled the subscription) is NOT fixable by
	// re-login: re-auth hits the same disabled account. Doctor must route it to the
	// operator-judgment action, never propose `claude /login`, so an account confused
	// out of serving isn't handed a dead-end recovery step.
	home := t.TempDir()
	ready := mkHome(t, home, ".claude-ready-seat", "ready@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"ready-seat","dir":"` + jsonPath(ready) + `"}` +
		`],"roles":{"active":"ready-seat","anchor":"ready-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	line := `{"ts":"` + time.Now().UTC().Format(time.RFC3339) + `","account":".claude-ready-seat","status":"ACCESS","block_reason":"organization has disabled Claude subscription access"}`
	if err := os.WriteFile(filepath.Join(rd, "probe_ledger.jsonl"), []byte(line+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("doctor with a fresh ACCESS probe rc=%d, want 1; out=%s", rc, out.String())
	}
	got := out.String()
	if !strings.Contains(got, "access_blocked") {
		t.Fatalf("doctor should fold a fresh ACCESS wall into access_blocked:\n%s", got)
	}
	if strings.Contains(got, "relogin") || strings.Contains(got, "CLAUDE_CONFIG_DIR=") {
		t.Fatalf("doctor must not propose re-login for an ACCESS wall:\n%s", got)
	}
}

func TestAccountsDoctorIdentityMismatchRequiresRelogin(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	home := t.TempDir()
	wrong := mkHome(t, home, ".claude-gem8-seat", "day26@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"gem8-seat","dir":"` + jsonPath(wrong) + `","chrome_profile":"Profile 9"}` +
		`],"roles":{"active":"gem8-seat","anchor":"gem8-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("doctor with identity mismatch rc=%d, want 1; stderr=%s\nout=%s", rc, errb.String(), out.String())
	}
	got := out.String()
	if !strings.Contains(got, "identity_mismatch") || !strings.Contains(got, "relogin") ||
		!strings.Contains(got, "CLAUDE_CONFIG_DIR=") {
		t.Fatalf("doctor should route identity mismatch to relogin:\n%s", got)
	}
}

// TestAccountsDoctorRecoveryWorklistClassifies pins #3580: a needs-login seat classifies
// as recoverable (action login) and a weekly usage-capped seat as hard, and the worklist
// reports the servable-seat gain from reclaiming the recoverable one.
func TestAccountsDoctorRecoveryWorklistClassifies(t *testing.T) {
	home := t.TempDir()
	ready := mkHome(t, home, ".claude-capped-seat", "capped@example.test", true)
	needs := mkHome(t, home, ".claude-needs-seat", "needs@example.test", false)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"capped-seat","dir":"` + jsonPath(ready) + `"},` +
		`{"name":"needs-seat","dir":"` + jsonPath(needs) + `"}` +
		`],"roles":{"active":"capped-seat","anchor":"capped-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	// Wire a fresh weekly usage LIMIT probe for the ready seat so it folds to wait_reset (hard).
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	line := `{"ts":"` + time.Now().UTC().Format(time.RFC3339) + `","account":".claude-capped-seat","status":"LIMIT","reset":"3pm"}`
	if err := os.WriteFile(filepath.Join(rd, "probe_ledger.jsonl"), []byte(line+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--json", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("doctor rc=%d, want 1 (actionable seats); stderr=%s\nout=%s", rc, errb.String(), out.String())
	}
	var rep acctDoctorReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("doctor --json did not parse: %v\n%s", err, out.String())
	}
	byName := map[string]doctorSeat{}
	for _, s := range rep.Seats {
		byName[s.Name] = s
	}
	if got := byName["needs-seat"].Recovery; got != recoveryRecoverable {
		t.Fatalf("needs-seat recovery = %q, want %q", got, recoveryRecoverable)
	}
	if got := byName["capped-seat"].Recovery; got != recoveryHard {
		t.Fatalf("capped-seat recovery = %q, want %q; seat=%+v", got, recoveryHard, byName["capped-seat"])
	}
	if rep.Recovery.HardWalled != 1 {
		t.Fatalf("hard_walled = %d, want 1; worklist=%+v", rep.Recovery.HardWalled, rep.Recovery)
	}
	if rep.Recovery.ServableSeatGain != 1 || len(rep.Recovery.Recoverable) != 1 {
		t.Fatalf("recoverable worklist = %+v, want one seat / gain 1", rep.Recovery)
	}
	rs := rep.Recovery.Recoverable[0]
	if rs.Name != "needs-seat" || rs.Action != "login" || rs.SeatGain != 1 {
		t.Fatalf("recoverable[0] = %+v, want needs-seat/login/gain 1", rs)
	}
}

// TestAccountsDoctorRecoveryWorklistEmptyWhenOfferable pins #3580 acceptance #3: a
// fully-offerable roster yields an empty worklist (no recoverable seats, zero gain).
func TestAccountsDoctorRecoveryWorklistEmptyWhenOfferable(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	home := t.TempDir()
	ready := mkHome(t, home, ".claude-ready-seat", "ready@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"ready-seat","dir":"` + jsonPath(ready) + `"}` +
		`],"roles":{"active":"ready-seat","anchor":"ready-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"doctor", "--json", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("doctor on clean registry rc=%d, want 0; out=%s stderr=%s", rc, out.String(), errb.String())
	}
	var rep acctDoctorReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("doctor --json did not parse: %v\n%s", err, out.String())
	}
	if len(rep.Recovery.Recoverable) != 0 || rep.Recovery.ServableSeatGain != 0 || rep.Recovery.HardWalled != 0 {
		t.Fatalf("fully-offerable worklist = %+v, want empty", rep.Recovery)
	}
}

func TestAccountsDoctorWriteHydratesCanonicalPeer(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_JOB_ROSTER", "")
	home := t.TempDir()
	src := filepath.Join(home, ".claude")
	dst := filepath.Join(home, ".claude-july4-netra")
	for _, d := range []string{src, dst} {
		if err := os.MkdirAll(filepath.Join(d, "projects", "C--work-fak"), 0o755); err != nil {
			t.Fatal(err)
		}
		body := `{"oauthAccount":{"emailAddress":"july4@example.test","accountUuid":"uuid-july4"}}`
		if err := os.WriteFile(filepath.Join(d, ".claude.json"), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	validCred := `{"claudeAiOauth":{"accessToken":"access","refreshToken":"refresh"}}`
	if err := os.WriteFile(filepath.Join(src, ".credentials.json"), []byte(validCred), 0o644); err != nil {
		t.Fatal(err)
	}
	placeholder := `{"claudeAiOauth":{"expiresAt":0,"scopes":["user:profile"],"subscriptionType":"max"}}`
	if err := os.WriteFile(filepath.Join(dst, ".credentials.json"), []byte(placeholder), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dst, ".oauth-token"), []byte("stale-token\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(src, "projects", "C--work-fak", "session-a.jsonl"), []byte("{}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"default","dir":"` + jsonPath(src) + `"},` +
		`{"name":"july4-netra","dir":"` + jsonPath(dst) + `"}` +
		`],"roles":{"active":"july4-netra","anchor":"default"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--write", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("doctor --write rc=%d, want 1 because duplicate default remains; stderr=%s\nout=%s", rc, errb.String(), out.String())
	}
	got := out.String()
	if !strings.Contains(got, "hydrate") || !strings.Contains(got, "APPLIED: hydrated july4-netra from default") {
		t.Fatalf("doctor --write should hydrate canonical peer:\n%s", got)
	}
	cred, err := os.ReadFile(filepath.Join(dst, ".credentials.json"))
	if err != nil {
		t.Fatal(err)
	}
	if string(cred) != validCred {
		t.Fatalf("target credentials not hydrated: %s", string(cred))
	}
	if _, err := os.Stat(filepath.Join(dst, ".oauth-token")); !os.IsNotExist(err) {
		t.Fatalf("target stale oauth token should be removed, stat err=%v", err)
	}
	if _, err := os.Stat(filepath.Join(dst, "projects", "C--work-fak", "session-a.jsonl")); err != nil {
		t.Fatalf("session transcript not copied: %v", err)
	}
}

// TestAccountsDoctorDurableOrgWallOutranksNeedsLogin is the doctor surface of #4998.
// Unlike TestAccountsDoctorAccessWallIsNotRelogin (a FRESH probe-ledger verdict on a
// still-credentialed seat), this wall was witnessed in an EARLIER process and persisted
// in the fleet-shared cooldown store — and Claude has since blanked the seat's tokens,
// so the config-plane fold alone says needs_login. Doctor must keep the stronger typed
// diagnosis across the process boundary and never hand the operator the futile relogin.
func TestAccountsDoctorDurableOrgWallOutranksNeedsLogin(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	home := t.TempDir()
	walled := mkHome(t, home, ".claude-wall-seat", "wall@example.test", false)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"wall-seat","dir":"` + jsonPath(walled) + `"}` +
		`],"roles":{"active":"wall-seat","anchor":"wall-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	sd := t.TempDir()
	t.Setenv("FLEET_STATE_DIR", sd)

	// Control: with no wall recorded, the blanked seat legitimately IS a relogin.
	// Without this half the test could pass vacuously off the switch arm alone.
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home}); rc != 1 {
		t.Fatalf("control doctor rc=%d, want 1; out=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "relogin") {
		t.Fatalf("control: a blanked seat with no wall must fold to relogin:\n%s", out.String())
	}

	// Process 1 witnessed the terminal 403 and persisted the typed evidence, exactly
	// as the guard's ObserveSeatHealth would; this doctor run is process 2.
	store, err := accounts.LoadCooldownStore(filepath.Join(sd, "account-cooldown.json"))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := store.RecordOrgAuthWall(accounts.UUIDBucketKey("u-wall@example.test"), "", time.Now().UTC()); !ok {
		t.Fatal("RecordOrgAuthWall did not record")
	}
	if err := store.Save(); err != nil {
		t.Fatal(err)
	}

	out.Reset()
	errb.Reset()
	rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("doctor with a durable org wall rc=%d, want 1; out=%s stderr=%s", rc, out.String(), errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "org_auth_wall") || !strings.Contains(got, "access_blocked") {
		t.Fatalf("doctor should keep the typed org-wall diagnosis and route it to access_blocked:\n%s", got)
	}
	if strings.Contains(got, "relogin") || strings.Contains(got, "CLAUDE_CONFIG_DIR=") {
		t.Fatalf("doctor must never propose re-login for a durable org wall:\n%s", got)
	}
}
