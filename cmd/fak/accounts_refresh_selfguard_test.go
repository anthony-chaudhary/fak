package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// accounts_refresh_selfguard_test.go — witnesses that `fak accounts refresh` will not log the
// operator out of the session they are typing in without being told to, and that when it is told
// to, it names the dir it killed and the command that revives it.
//
// The incident these pin (2026-08-08, #5954): a seat enrolled from ~/.claude with --no-divorce
// still shared ONE refresh token with it. `fak accounts refresh --name <seat> --force` rotated the
// seat exactly as designed, which invalidated ~/.claude — the operator's own live session. It went
// hollow and the next four turns each returned `Login expired · Please run /login`, from a message
// that named neither the cause nor the fix.
//
// Every assertion below is a fingerprint compare over on-disk bytes: no network, no real spawn.

// sharedFamilyFixture lays down the armed state — a caller config dir (~/.claude, the session the
// operator is typing in) and a roster seat holding a byte-identical refresh token, both with an
// expiry far in the future so nothing is DUE and only --force can rotate. It returns
// (home, callerDir, seatDir); the seat discovers as "seat-netra".
func sharedFamilyFixture(t *testing.T, callerToken, seatToken string) (string, string, string) {
	t.Helper()
	t.Setenv("CLAUDE_CONFIG_DIR", "") // caller dir must resolve to <home>/.claude, not the real one

	home := t.TempDir()
	caller := filepath.Join(home, ".claude")
	seat := filepath.Join(home, ".claude-seat-netra")
	for dir, tok := range map[string]string{caller: callerToken, seat: seatToken} {
		writeFileString(t, filepath.Join(dir, ".credentials.json"), fmt.Sprintf(
			`{"claudeAiOauth":{"accessToken":"at-%s","refreshToken":%q,"expiresAt":4102444800000}}`,
			filepath.Base(dir), tok))
		writeFileString(t, filepath.Join(dir, ".claude.json"),
			`{"oauthAccount":{"emailAddress":"live@example.test","accountUuid":"uuid-live"}}`)
	}
	return home, caller, seat
}

// refreshFixtureParams points a refresh at the fixture's discovered roster. The registry path does
// not exist, so loadOrDiscover scans <home>/.claude* — the same disk truth the operator has.
func refreshFixtureParams(home string, spawn accounts.RefreshSpawn) refreshParams {
	return refreshParams{
		name:         "seat-netra",
		timeout:      5 * time.Second,
		registryPath: filepath.Join(home, "registry.json"),
		homeDir:      home,
		spawn:        spawn,
	}
}

// countingRotateSpawn stands in for a real `claude -p` refresh: it moves the dir it is pointed at
// onto a NEW token family and counts how many times it ran, so a test can prove a refusal refused
// before anything was spawned.
func countingRotateSpawn(t *testing.T, newToken string, calls *int) accounts.RefreshSpawn {
	t.Helper()
	return func(_ context.Context, cfgDir string) error {
		*calls++
		return os.WriteFile(filepath.Join(cfgDir, ".credentials.json"), []byte(fmt.Sprintf(
			`{"claudeAiOauth":{"accessToken":"at-rotated","refreshToken":%q,"expiresAt":4102444800000}}`,
			newToken)), 0o600)
	}
}

// TestAccountsRefreshForceRefusesToLogTheCallerOut is the acceptance: `--force` on a seat that
// shares a family with the caller's own config dir must REFUSE, name the session it would have
// ended, and touch nothing.
func TestAccountsRefreshForceRefusesToLogTheCallerOut(t *testing.T) {
	home, caller, seat := sharedFamilyFixture(t, "rt-shared", "rt-shared")
	family := accounts.RefreshFamilyID(seat)
	callerBefore, err := os.ReadFile(filepath.Join(caller, ".credentials.json"))
	if err != nil {
		t.Fatal(err)
	}

	calls := 0
	p := refreshFixtureParams(home, countingRotateSpawn(t, "rt-seat-own", &calls))
	p.force = true

	var out, errb bytes.Buffer
	rc := runAccountsRefresh(&out, &errb, p)
	if rc == 0 {
		t.Fatalf("a refresh that would log THIS session out must exit nonzero; got 0\nstdout=%s", out.String())
	}
	if calls != 0 {
		t.Errorf("the refusal must land BEFORE the spawn; the refresh turn ran %d time(s)", calls)
	}

	// Nothing may have moved on either side: the seat is still on the shared family, and the
	// caller's credential is byte-for-byte what it was.
	if got := accounts.RefreshFamilyID(seat); got != family {
		t.Errorf("a refused refresh must not rotate the seat: family %s -> %s", family, got)
	}
	callerAfter, err := os.ReadFile(filepath.Join(caller, ".credentials.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(callerBefore, callerAfter) {
		t.Error("a refused refresh must leave the caller's own credential untouched")
	}

	stdout := out.String()
	for _, want := range []string{"REFUSED", caller, family, refreshAckLogoutFlag, "blocked=1"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("the refusal must name %q — the session at risk and the way forward; got:\n%s", want, stdout)
		}
	}
}

// TestAccountsRefreshAcknowledgedLogoutNamesTheDirAndItsRecovery is the second half of the
// acceptance: WITH the acknowledgement the rotation proceeds, and the report states which dir it
// just invalidated and the exact command that repairs it — the thing the operator had to work out
// for themselves.
func TestAccountsRefreshAcknowledgedLogoutNamesTheDirAndItsRecovery(t *testing.T) {
	home, caller, seat := sharedFamilyFixture(t, "rt-shared", "rt-shared")
	before := accounts.RefreshFamilyID(seat)

	calls := 0
	p := refreshFixtureParams(home, countingRotateSpawn(t, "rt-seat-own", &calls))
	p.force, p.ackLogout = true, true

	var out, errb bytes.Buffer
	if rc := runAccountsRefresh(&out, &errb, p); rc != 0 {
		t.Fatalf("an acknowledged rotation is a SUCCESS: rc=%d\nstdout=%s\nstderr=%s", rc, out.String(), errb.String())
	}
	if calls != 1 {
		t.Errorf("the acknowledgement must let the refresh run exactly once; ran %d time(s)", calls)
	}
	if got := accounts.RefreshFamilyID(seat); got == before || got == "" {
		t.Fatalf("the seat should have moved onto its own family; %s -> %q", before, got)
	}
	if accounts.DetectSharedRefreshFamily(caller, seat).Shared {
		t.Error("fixture problem: the two dirs should no longer share a family after the rotation")
	}

	stdout := out.String()
	for _, want := range []string{caller, "INVALID", "CLAUDE_CONFIG_DIR=" + caller + " claude /login"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("the report must name %q — what died and the one command that fixes it; got:\n%s", want, stdout)
		}
	}
	// The caller's file is still intact here (only its TOKEN is dead), so the hollow-on-disk line
	// must not be claimed. Reporting a state the disk does not hold is the same defect in reverse.
	if strings.Contains(stdout, "HOLLOW on disk") {
		t.Errorf("the caller's credential is not hollow yet; the report must not say it is:\n%s", stdout)
	}
}

// TestAccountsRefreshAcknowledgedLogoutReportsAHollowedCallerDir covers the shape the operator
// actually witnessed: by the time the rotation settled, ~/.claude's credential had been BLANKED
// (accessToken "", refreshToken "", expiresAt 0), which is what turns every later turn into
// `Login expired`. That end state has its own line, because it is the one a running session feels.
func TestAccountsRefreshAcknowledgedLogoutReportsAHollowedCallerDir(t *testing.T) {
	home, caller, _ := sharedFamilyFixture(t, "rt-shared", "rt-shared")

	// A spawn that rotates the seat AND blanks the caller — the witnessed end state.
	spawn := func(_ context.Context, cfgDir string) error {
		if err := os.WriteFile(filepath.Join(caller, ".credentials.json"),
			[]byte(`{"claudeAiOauth":{"accessToken":"","refreshToken":"","expiresAt":0}}`), 0o600); err != nil {
			return err
		}
		return os.WriteFile(filepath.Join(cfgDir, ".credentials.json"),
			[]byte(`{"claudeAiOauth":{"accessToken":"at-rotated","refreshToken":"rt-seat-own","expiresAt":4102444800000}}`), 0o600)
	}
	p := refreshFixtureParams(home, spawn)
	p.force, p.ackLogout = true, true

	var out, errb bytes.Buffer
	if rc := runAccountsRefresh(&out, &errb, p); rc != 0 {
		t.Fatalf("rc=%d\nstdout=%s\nstderr=%s", rc, out.String(), errb.String())
	}
	if !accounts.CredentialHollow(caller) {
		t.Fatal("fixture problem: the caller's credential should be hollow by now")
	}
	stdout := out.String()
	if !strings.Contains(stdout, "HOLLOW on disk") {
		t.Errorf("a hollowed caller dir must be reported as such; got:\n%s", stdout)
	}
	if !strings.Contains(stdout, "CLAUDE_CONFIG_DIR="+caller+" claude /login") {
		t.Errorf("the recovery command must be printed for the hollowed dir; got:\n%s", stdout)
	}
}

// TestAccountsRefreshUnforcedHintDoesNotInviteASelfLogout pins defect #1 of #5954. The unforced
// run rotates nothing, so it is not refused — but its standing hint ("pass --force to prove it can
// rotate") described the destructive path as a proof, and the operator took it at its word.
func TestAccountsRefreshUnforcedHintDoesNotInviteASelfLogout(t *testing.T) {
	home, caller, seat := sharedFamilyFixture(t, "rt-shared", "rt-shared")
	family := accounts.RefreshFamilyID(seat)

	calls := 0
	p := refreshFixtureParams(home, countingRotateSpawn(t, "rt-seat-own", &calls))

	var out, errb bytes.Buffer
	if rc := runAccountsRefresh(&out, &errb, p); rc != 0 {
		t.Fatalf("a not-due seat is healthy, not a failure: rc=%d\nstdout=%s", rc, out.String())
	}
	if calls != 0 {
		t.Errorf("an unforced, not-due refresh must spawn nothing; ran %d time(s)", calls)
	}
	stdout := out.String()
	if !strings.Contains(stdout, "fresh") {
		t.Fatalf("expected the seat to be graded fresh; got:\n%s", stdout)
	}
	if strings.Contains(stdout, refreshForceHint) {
		t.Errorf("the bare --force invitation must not stand on a seat whose rotation logs this session out; got:\n%s", stdout)
	}
	for _, want := range []string{"do NOT --force it blind", caller, family, refreshAckLogoutFlag} {
		if !strings.Contains(stdout, want) {
			t.Errorf("the amended hint must name %q; got:\n%s", want, stdout)
		}
	}
}

// TestAccountsRefreshIndependentSeatIsNotBlocked is the over-refusal guard: a seat on its OWN
// family shares nothing with the caller, so --force must behave exactly as it always has. A safety
// check that fires on the healthy path would just teach operators to pass the flag by reflex.
func TestAccountsRefreshIndependentSeatIsNotBlocked(t *testing.T) {
	home, _, seat := sharedFamilyFixture(t, "rt-caller-own", "rt-seat-own")
	before := accounts.RefreshFamilyID(seat)

	calls := 0
	p := refreshFixtureParams(home, countingRotateSpawn(t, "rt-seat-rotated", &calls))
	p.force = true

	var out, errb bytes.Buffer
	if rc := runAccountsRefresh(&out, &errb, p); rc != 0 {
		t.Fatalf("an independent seat must refresh normally: rc=%d\nstdout=%s\nstderr=%s", rc, out.String(), errb.String())
	}
	if calls != 1 {
		t.Errorf("the refresh turn should have run once; ran %d time(s)", calls)
	}
	if got := accounts.RefreshFamilyID(seat); got == before {
		t.Errorf("the seat should have rotated off family %s", before)
	}
	stdout := out.String()
	for _, unwanted := range []string{"REFUSED", "blocked=1", "claude /login"} {
		if strings.Contains(stdout, unwanted) {
			t.Errorf("a seat that shares nothing with this session must not report %q; got:\n%s", unwanted, stdout)
		}
	}
}

// TestAccountsRefreshCallerDirFollowsClaudeConfigDir proves the guard protects the session that is
// ACTUALLY running, not a hardcoded ~/.claude: under a launched seat ($CLAUDE_CONFIG_DIR set) the
// dir at risk is that seat's, and ~/.claude is just another dir.
func TestAccountsRefreshCallerDirFollowsClaudeConfigDir(t *testing.T) {
	home, _, seat := sharedFamilyFixture(t, "rt-default-own", "rt-shared")
	live := filepath.Join(home, ".claude-live-netra")
	writeFileString(t, filepath.Join(live, ".credentials.json"),
		`{"claudeAiOauth":{"accessToken":"at-live","refreshToken":"rt-shared","expiresAt":4102444800000}}`)
	writeFileString(t, filepath.Join(live, ".claude.json"),
		`{"oauthAccount":{"emailAddress":"live@example.test","accountUuid":"uuid-live"}}`)
	t.Setenv("CLAUDE_CONFIG_DIR", live)

	calls := 0
	p := refreshFixtureParams(home, countingRotateSpawn(t, "rt-seat-own", &calls))
	p.force = true

	var out, errb bytes.Buffer
	if rc := runAccountsRefresh(&out, &errb, p); rc == 0 {
		t.Fatalf("rotating %s ends the $CLAUDE_CONFIG_DIR session; it must be refused\nstdout=%s", seat, out.String())
	}
	if calls != 0 {
		t.Errorf("the refusal must land before the spawn; ran %d time(s)", calls)
	}
	if !strings.Contains(out.String(), live) {
		t.Errorf("the refusal must name the $CLAUDE_CONFIG_DIR session at risk (%s); got:\n%s", live, out.String())
	}
}

// TestAccountsRefreshJSONCarriesTheLoggedOutDir: a --json consumer (the scheduled sweep) must be
// able to SEE the logout, not discover it later as a dead session.
func TestAccountsRefreshJSONCarriesTheLoggedOutDir(t *testing.T) {
	home, caller, _ := sharedFamilyFixture(t, "rt-shared", "rt-shared")
	calls := 0
	p := refreshFixtureParams(home, countingRotateSpawn(t, "rt-seat-own", &calls))
	p.force, p.ackLogout, p.asJSON = true, true, true

	var out, errb bytes.Buffer
	if rc := runAccountsRefresh(&out, &errb, p); rc != 0 {
		t.Fatalf("rc=%d\nstdout=%s\nstderr=%s", rc, out.String(), errb.String())
	}
	payload := out.String()
	if !strings.Contains(payload, `"logged_out_dir"`) {
		t.Errorf("the JSON payload must report the invalidated dir; got:\n%s", payload)
	}
	if !strings.Contains(payload, strings.ReplaceAll(caller, `\`, `\\`)) {
		t.Errorf("the JSON payload must name %s; got:\n%s", caller, payload)
	}
}

// TestDetectRefreshSelfHazardOnlyFiresOnASecondDir is the pure-fingerprint unit behind all of the
// above. The same dir must never be a hazard: refreshing the caller's OWN dir in place is the
// healthy path — the session keeps reading that file and simply finds a newer token in it.
func TestDetectRefreshSelfHazardOnlyFiresOnASecondDir(t *testing.T) {
	_, caller, seat := sharedFamilyFixture(t, "rt-shared", "rt-shared")
	other := filepath.Join(filepath.Dir(caller), ".claude-other-netra")
	writeFileString(t, filepath.Join(other, ".credentials.json"),
		`{"claudeAiOauth":{"accessToken":"at-other","refreshToken":"rt-other","expiresAt":4102444800000}}`)

	if hz := detectRefreshSelfHazard(caller, seat); !hz.Hit || hz.CallerDir != caller || hz.FamilyID == "" {
		t.Errorf("two dirs on one refresh token are a logout waiting to happen: %+v", hz)
	}
	for _, tc := range []struct {
		why         string
		caller, dst string
	}{
		{"the caller's own dir rotates in place", caller, caller},
		{"the same dir spelled differently is still the same dir", caller, caller + string(filepath.Separator)},
		{"separate families are independent logins", caller, other},
		{"an unresolvable caller dir cannot be at risk", "", seat},
		{"a dir with no credential holds no family", filepath.Join(other, "nope"), seat},
	} {
		if hz := detectRefreshSelfHazard(tc.caller, tc.dst); hz.Hit {
			t.Errorf("%s: expected no hazard, got %+v", tc.why, hz)
		}
	}
}
