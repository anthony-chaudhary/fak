package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// rotateRegistry writes a registry with three live seats — alice, bob (rotation pool) and a
// RESERVED carol (held out of routine rotation) — with the active role on alice. It returns
// the registry path plus the three seat dirs.
func rotateRegistry(t *testing.T, home string) (regPath, aDir, bDir, cDir string) {
	t.Helper()
	aDir = mkHome(t, home, ".claude-alice-seat", "alice@example.test", true)
	bDir = mkHome(t, home, ".claude-bob-seat", "bob@example.test", true)
	cDir = mkHome(t, home, ".claude-carol-seat", "carol@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"alice-seat","dir":"` + jsonPath(aDir) + `"},` +
		`{"name":"bob-seat","dir":"` + jsonPath(bDir) + `"},` +
		`{"name":"carol-seat","dir":"` + jsonPath(cDir) + `","reserved":true}],` +
		`"roles":{"active":"alice-seat"}}`
	regPath = filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	return regPath, aDir, bDir, cDir
}

func rotateRegistryWithBrokenSeat(t *testing.T, home string) (regPath, bDir, brokenDir string) {
	t.Helper()
	aDir := mkHome(t, home, ".claude-alice-seat", "alice@example.test", true)
	bDir = mkHome(t, home, ".claude-bob-seat", "bob@example.test", true)
	brokenDir = mkHome(t, home, ".claude-broken-seat", "broken@example.test", false)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"alice-seat","dir":"` + jsonPath(aDir) + `"},` +
		`{"name":"bob-seat","dir":"` + jsonPath(bDir) + `"},` +
		`{"name":"broken-seat","dir":"` + jsonPath(brokenDir) + `"}` +
		`],"roles":{"active":"alice-seat","anchor":"alice-seat"}}`
	regPath = filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	return regPath, bDir, brokenDir
}

func TestAccountsNextRoundRobin(t *testing.T) {
	home := t.TempDir()
	regPath, _, bDir, _ := rotateRegistry(t, home)

	run := func(args ...string) (string, string, int) {
		var out, errb bytes.Buffer
		rc := runAccounts(&out, &errb, append([]string{"next", "--registry", regPath, "--home", home}, args...))
		return out.String(), errb.String(), rc
	}

	// Pool is [alice-seat, bob-seat] (carol reserved). Round-robin wraps.
	for _, c := range []struct{ after, want string }{
		{"alice-seat", "bob-seat"},
		{"bob-seat", "alice-seat"},   // wrap (carol is reserved, not in the pool)
		{"", "alice-seat"},           // no anchor -> first in rotation
		{"carol-seat", "alice-seat"}, // a reserved anchor is not in the pool -> fresh start
	} {
		out, errb, rc := run("--after", c.after)
		if rc != 0 {
			t.Fatalf("next --after %q rc=%d stderr=%s", c.after, rc, errb)
		}
		if !strings.Contains(out, "next: "+c.want) {
			t.Fatalf("next --after %q = %q, want next: %s", c.after, strings.TrimSpace(out), c.want)
		}
		if !strings.Contains(out, "login=ready can_serve=true") {
			t.Fatalf("next --after %q missing login readiness: %q", c.after, strings.TrimSpace(out))
		}
	}

	// --env prints the chosen seat's CLAUDE_CONFIG_DIR for eval/wrappers.
	out, errb, rc := run("--after", "alice-seat", "--env")
	if rc != 0 {
		t.Fatalf("next --env rc=%d stderr=%s", rc, errb)
	}
	if !strings.Contains(out, "CLAUDE_CONFIG_DIR="+bDir) {
		t.Fatalf("next --env = %q, want CLAUDE_CONFIG_DIR=%s", strings.TrimSpace(out), bDir)
	}

	// --json prints the chosen RotationSeat.
	out, _, rc = run("--after", "alice-seat", "--json")
	if rc != 0 || !strings.Contains(out, `"name": "bob-seat"`) ||
		!strings.Contains(out, `"status": "included"`) ||
		!strings.Contains(out, `"login_status": "ready"`) ||
		!strings.Contains(out, `"can_serve": true`) {
		t.Fatalf("next --json = %q", strings.TrimSpace(out))
	}
}

func TestAccountsNextReportsActionableFixes(t *testing.T) {
	home := t.TempDir()
	regPath, bDir, _ := rotateRegistryWithBrokenSeat(t, home)

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"next", "--after", "alice-seat", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("next rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"next: bob-seat",
		"account fixes: 1 seat(s) need action (relogin=1); run `fak accounts doctor`",
		"broken-seat: relogin",
		"CLAUDE_CONFIG_DIR=",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("next output missing %q:\n%s", want, got)
		}
	}

	out.Reset()
	errb.Reset()
	rc = runAccounts(&out, &errb, []string{"next", "--after", "alice-seat", "--env", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("next --env rc=%d stderr=%s", rc, errb.String())
	}
	if strings.TrimSpace(out.String()) != "CLAUDE_CONFIG_DIR="+bDir {
		t.Fatalf("next --env stdout = %q, want only CLAUDE_CONFIG_DIR=%s", strings.TrimSpace(out.String()), bDir)
	}
	if !strings.Contains(errb.String(), "account fixes: 1 seat(s) need action") ||
		!strings.Contains(errb.String(), "broken-seat: relogin") {
		t.Fatalf("next --env stderr should carry account fixes:\n%s", errb.String())
	}
}

func TestAccountsNextSingleBucketFailsLoud(t *testing.T) {
	home := t.TempDir()
	seat := mkHome(t, home, ".claude-solo-seat", "solo@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[{"name":"solo-seat","dir":"` + jsonPath(seat) + `"}],` +
		`"roles":{"active":"solo-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	// No anchor -> the sole bucket is a valid fresh start.
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"next", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("next (no anchor) rc=%d stderr=%s", rc, errb.String())
	}
	// Rotating OFF the only bucket has nowhere to go -> fail loud.
	out.Reset()
	errb.Reset()
	rc := runAccounts(&out, &errb, []string{"next", "--after", "solo-seat", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("next --after solo-seat rc=%d, want 1; stderr=%s", rc, errb.String())
	}
	if !strings.Contains(errb.String(), "only one account bucket") {
		t.Fatalf("expected single-bucket message, got %q", errb.String())
	}
}

func TestAccountsLaunchRotate(t *testing.T) {
	home := t.TempDir()
	regPath, _, bDir, _ := rotateRegistry(t, home)

	// --rotate with no --after rotates off the ACTIVE seat (alice) onto the next bucket (bob).
	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--rotate", "--dry-run", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch --rotate rc=%d stderr=%s", rc, errb.String())
	}
	got := errb.String()
	if !strings.Contains(got, `rotating off "alice-seat" -> "bob-seat"`) {
		t.Fatalf("missing rotation note:\n%s", got)
	}
	if !strings.Contains(got, `seat "bob-seat"`) ||
		!strings.Contains(got, "identity          = bob@example.test") ||
		!strings.Contains(got, "CLAUDE_CONFIG_DIR = <account-dir>") {
		t.Fatalf("rotated launch plan should target bob-seat without exposing its account dir:\n%s", got)
	}
	if strings.Contains(got, bDir) {
		t.Fatalf("rotated launch plan exposed the raw account dir %q:\n%s", bDir, got)
	}
	if !strings.Contains(got, "login             = ready (can_serve=true)") {
		t.Fatalf("rotated launch plan missing login readiness:\n%s", got)
	}

	// --rotate --after bob-seat rotates off bob onto alice (wrap).
	out.Reset()
	errb.Reset()
	rc = runAccounts(&out, &errb, []string{"launch", "--rotate", "--after", "bob-seat", "--dry-run", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch --rotate --after bob rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(errb.String(), `rotating off "bob-seat" -> "alice-seat"`) {
		t.Fatalf("expected rotate off bob -> alice:\n%s", errb.String())
	}
}

func TestAccountsLaunchRotateReportsActionableFixes(t *testing.T) {
	home := t.TempDir()
	regPath, _, _ := rotateRegistryWithBrokenSeat(t, home)

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"launch", "--rotate", "--dry-run", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("launch --rotate rc=%d stderr=%s", rc, errb.String())
	}
	got := errb.String()
	for _, want := range []string{
		`rotating off "alice-seat" -> "bob-seat"`,
		"account fixes: 1 seat(s) need action (relogin=1); run `fak accounts doctor`",
		"broken-seat: relogin",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rotated launch stderr missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(out.String(), "account fixes") {
		t.Fatalf("dry-run stdout should stay scriptable, got:\n%s", out.String())
	}
}

func TestAccountsNextAndLaunchRotateRefuseWeeklyCappedAlternative(t *testing.T) {
	home := t.TempDir()
	regPath, _, _, _ := rotateRegistry(t, home)
	regDir := filepath.Join(home, "runtime-registry")
	if err := os.MkdirAll(regDir, 0o755); err != nil {
		t.Fatal(err)
	}
	runtime := `{"generated_utc":"2026-07-06T12:00:00Z",` +
		`"throttle":{".claude-bob-seat":{"reset":"Jul 9, 6am (America/Los_Angeles)","weekly":"Jul 9, 6am (America/Los_Angeles)"}},` +
		`"auth":{},"sessions":[]}`
	if err := os.WriteFile(filepath.Join(regDir, "sessions.json"), []byte(runtime), 0o644); err != nil {
		t.Fatal(err)
	}
	cfg := filepath.Join(home, "cfg")
	if err := os.MkdirAll(cfg, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FLEET_USER_HOME", home)
	t.Setenv("FLEET_CONFIG_HOME", cfg)
	t.Setenv("FLEET_REG_DIR", regDir)
	t.Setenv("FLEET_POLICY_PATH", filepath.Join(home, "missing-policy.json"))

	var nextOut, nextErr bytes.Buffer
	if rc := runAccounts(&nextOut, &nextErr, []string{"next", "--after", "alice-seat", "--registry", regPath, "--home", home}); rc != 1 {
		t.Fatalf("next should refuse capped bob rc=%d stdout=%s stderr=%s", rc, nextOut.String(), nextErr.String())
	}
	if strings.Contains(nextOut.String(), "CLAUDE_CONFIG_DIR") || strings.Contains(nextOut.String(), "next: bob-seat") {
		t.Fatalf("next stdout should not offer capped bob:\n%s", nextOut.String())
	}

	oldRun := accountsLaunchRun
	called := false
	accountsLaunchRun = func(_, _ io.Writer, _ []string, _ []string) launchRunResult {
		called = true
		return launchRunResult{Code: 0}
	}
	t.Cleanup(func() { accountsLaunchRun = oldRun })

	var launchOut, launchErr bytes.Buffer
	if rc := runAccounts(&launchOut, &launchErr, []string{"launch", "--rotate", "--registry", regPath, "--home", home}); rc != 1 {
		t.Fatalf("launch --rotate should refuse capped bob rc=%d stdout=%s stderr=%s", rc, launchOut.String(), launchErr.String())
	}
	if called {
		t.Fatal("launch runner was called even though the only non-anchor account was weekly-capped")
	}
	for _, want := range []string{"no runtime-launchable account", "usage/weekly-capped", "bob-seat"} {
		if !strings.Contains(launchErr.String(), want) {
			t.Fatalf("launch stderr missing %q:\n%s", want, launchErr.String())
		}
	}
	if strings.Contains(launchOut.String(), "guard -- claude") {
		t.Fatalf("launch stdout should not emit a guard command for capped bob:\n%s", launchOut.String())
	}
}

// TestAccountsRotationInspect covers the full witnessed rotation dump — the pool in launch
// order, every exclusion with its reason, and the registry-drift check — plus the heal loop:
// a registry whose stored identities lag disk reports drift, `discover --write` heals it,
// and the re-run reports none.
func TestAccountsRotationInspect(t *testing.T) {
	home := t.TempDir()
	regPath, _, _, _ := rotateRegistry(t, home)

	run := func(args ...string) (string, string, int) {
		var out, errb bytes.Buffer
		rc := runAccounts(&out, &errb, append([]string{"rotation", "--registry", regPath, "--home", home}, args...))
		return out.String(), errb.String(), rc
	}

	// The fixture registry carries NO identity blocks, so every seat's stored identity
	// disagrees with disk truth — the exact rot the drift check exists to surface.
	out, errb, rc := run()
	if rc != 0 {
		t.Fatalf("rotation rc=%d stderr=%s", rc, errb)
	}
	if !strings.Contains(out, "POOL — 2 distinct account bucket(s)") {
		t.Fatalf("rotation pool header missing: %q", out)
	}
	if !strings.Contains(out, "alice-seat") || !strings.Contains(out, "bob-seat") {
		t.Fatalf("rotation pool rows missing: %q", out)
	}
	if !strings.Contains(out, "carol-seat") || !strings.Contains(out, "reserved") {
		t.Fatalf("rotation excluded row for reserved carol missing: %q", out)
	}
	if !strings.Contains(out, "registry drift: 3 seat(s)") || !strings.Contains(out, "discover --write") {
		t.Fatalf("rotation drift report missing: %q", out)
	}

	// JSON form carries the same decision as data.
	out, _, rc = run("--json")
	if rc != 0 || !strings.Contains(out, `"schema": "fak.accounts.rotation.v1"`) ||
		!strings.Contains(out, `"status": "reserved"`) ||
		!strings.Contains(out, `"registry_drift"`) {
		t.Fatalf("rotation --json = %q", strings.TrimSpace(out))
	}

	// Heal the drift the way the report says to, then the re-run is clean.
	var out2, errb2 bytes.Buffer
	if rc := runAccounts(&out2, &errb2, []string{"discover", "--write", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("discover --write rc=%d stderr=%s", rc, errb2.String())
	}
	out, errb, rc = run()
	if rc != 0 {
		t.Fatalf("rotation after heal rc=%d stderr=%s", rc, errb)
	}
	if !strings.Contains(out, "registry drift: none") {
		t.Fatalf("rotation after discover --write still drifts: %q", out)
	}
}

func TestAccountsRotationJSONReportsActionableFixes(t *testing.T) {
	home := t.TempDir()
	regPath, _, _ := rotateRegistryWithBrokenSeat(t, home)

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"rotation", "--json", "--registry", regPath, "--home", home})
	if rc != 0 {
		t.Fatalf("rotation --json rc=%d stderr=%s", rc, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		`"account_fixes"`,
		`"actionable": 1`,
		`"by_action"`,
		`"relogin": 1`,
		`"name": "broken-seat"`,
		`"action": "relogin"`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("rotation --json missing %q:\n%s", want, got)
		}
	}
}

func TestAccountsCheckNamesUDriftAndActionableFixes(t *testing.T) {
	home := t.TempDir()
	regPath, _, _ := rotateRegistryWithBrokenSeat(t, home)
	jobView := filepath.Join(home, "claude_accounts.yaml")
	if err := os.WriteFile(jobView, []byte("accounts: []\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"check", "--registry", regPath, "--home", home,
		"--dos-view", "", "--job-view", jobView,
	})
	if rc != 1 {
		t.Fatalf("check rc=%d, want drift exit 1; stderr=%s\nstdout=%s", rc, errb.String(), out.String())
	}
	got := out.String()
	for _, want := range []string{
		"DRIFT job:",
		"this is what `u` reads",
		"run `fak accounts sync`",
		"account fixes: 1 seat(s) need action",
		"broken-seat: relogin",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("check output missing %q:\n%s", want, got)
		}
	}
}
