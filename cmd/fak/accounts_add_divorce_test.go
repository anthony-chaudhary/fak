package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// accounts_add_divorce_test.go — CLI-level witnesses that an enroll resolves the OAuth
// token-family hazard a credential copy creates, and TELLS the operator what it cost.
//
// The incident these pin (2026-08-06): `enroll-current` copied ~/.claude's credential into a new
// seat. Both dirs then held one refresh token. The seat refreshed, the family rotated, and the
// operator's own interactive session — whose access token was still hours from expiring — started
// returning 401 with no explanation and no way back except a manual /login. The enroll had said
// nothing about any of it.

// rotatingDivorceSpawn returns a spawn that behaves like a real `claude -p` refresh: it writes a
// NEW refresh token (a new family) and a later expiry into the dir it is pointed at.
func rotatingDivorceSpawn(newRefreshToken string, expiresAtMs int64) accounts.RefreshSpawn {
	return func(_ context.Context, cfgDir string) error {
		body := fmt.Sprintf(
			`{"claudeAiOauth":{"accessToken":"at-rotated","refreshToken":%q,"expiresAt":%d}}`,
			newRefreshToken, expiresAtMs,
		)
		return os.WriteFile(filepath.Join(cfgDir, ".credentials.json"), []byte(body), 0o600)
	}
}

// enrollFixture lays down a source dir holding a live credential on family `refreshToken`, and
// returns (home, sourceDir, registryPath).
func enrollFixture(t *testing.T, refreshToken string) (string, string, string) {
	t.Helper()
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	t.Setenv("CLAUDE_CONFIG_DIR", "")

	home := t.TempDir()
	src := filepath.Join(home, ".claude")
	writeFileString(t, filepath.Join(src, ".credentials.json"), fmt.Sprintf(
		`{"claudeAiOauth":{"accessToken":"at-live","refreshToken":%q,"expiresAt":4102444800000}}`, refreshToken))
	writeFileString(t, filepath.Join(src, ".claude.json"),
		`{"oauthAccount":{"emailAddress":"live@example.test","accountUuid":"uuid-live"}}`)
	return home, src, filepath.Join(home, "registry.json")
}

// TestEnrollCurrent_DivorcesTokenFamilyByDefault is the acceptance: after enrolling from a dir that
// KEEPS RUNNING, the seat must be on its own OAuth family — never sharing the source's — so that
// neither side can silently invalidate the other later.
func TestEnrollCurrent_DivorcesTokenFamilyByDefault(t *testing.T) {
	home, src, regPath := enrollFixture(t, "rt-shared")
	hermeticProbeStub(t)

	divorceRefreshSpawn = rotatingDivorceSpawn("rt-seat-own", 4102444800000)
	t.Cleanup(func() { divorceRefreshSpawn = func(context.Context, string) error { return nil } })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"enroll-current", "--name", "seat", "--from", src,
		"--home", home, "--registry", regPath, "--job-view", "",
	})
	if rc != 0 {
		t.Fatalf("enroll-current rc=%d\nstdout=%s\nstderr=%s", rc, out.String(), errb.String())
	}

	seatDir := filepath.Join(home, ".claude-seat-netra")
	if share := accounts.DetectSharedRefreshFamily(src, seatDir); share.Shared {
		t.Fatalf("seat still shares OAuth family %s with %s — the first refresh by either side would silently 401 the other", share.FamilyID, src)
	}
	if got := accounts.RefreshFamilyID(seatDir); got == "" {
		t.Fatal("seat has no refresh token after the divorce; it cannot refresh independently")
	}

	// The consequence must be STATED. The whole defect was a logout nobody was warned about, so a
	// silent-but-correct divorce would still be a regression.
	stdout := out.String()
	if !strings.Contains(stdout, "own OAuth family") {
		t.Errorf("enroll must report that the seat moved onto its own family; got:\n%s", stdout)
	}
	if !strings.Contains(stdout, src) || !strings.Contains(stdout, "claude /login") {
		t.Errorf("enroll must name the source dir that now needs a re-login, with the command; got:\n%s", stdout)
	}
}

// TestEnrollCurrent_NoDivorceFlagWarnsInsteadOfHidingIt: the opt-out must not be a silent opt-out.
// Declining the divorce leaves the hazard armed, so the enroll owes the operator that fact.
func TestEnrollCurrent_NoDivorceFlagWarnsInsteadOfHidingIt(t *testing.T) {
	home, src, regPath := enrollFixture(t, "rt-shared")
	hermeticProbeStub(t)

	spawned := false
	divorceRefreshSpawn = func(context.Context, string) error { spawned = true; return nil }
	t.Cleanup(func() { divorceRefreshSpawn = func(context.Context, string) error { return nil } })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"enroll-current", "--name", "seat", "--from", src, "--no-divorce",
		"--home", home, "--registry", regPath, "--job-view", "",
	})
	if rc != 0 {
		t.Fatalf("enroll-current --no-divorce rc=%d\nstdout=%s\nstderr=%s", rc, out.String(), errb.String())
	}
	if spawned {
		t.Error("--no-divorce must not spawn a refresh")
	}

	seatDir := filepath.Join(home, ".claude-seat-netra")
	if !accounts.DetectSharedRefreshFamily(src, seatDir).Shared {
		t.Fatal("fixture problem: --no-divorce should leave the two dirs sharing one family")
	}
	stderr := errb.String()
	if !strings.Contains(stderr, "share OAuth token family") {
		t.Errorf("--no-divorce must WARN that the family is shared; got:\n%s", stderr)
	}
	if !strings.Contains(stderr, "invalidate the other") {
		t.Errorf("--no-divorce warning must state the consequence; got:\n%s", stderr)
	}
}

// TestEnrollCurrent_FailedDivorceWarnsAndStillEnrolls: a refresh that rotates nothing means the
// credential probably cannot refresh at all — a seat headed for a human /login. That must be a loud
// warning, but must NOT fail an enroll whose seat and registry row are already correct.
func TestEnrollCurrent_FailedDivorceWarnsAndStillEnrolls(t *testing.T) {
	home, src, regPath := enrollFixture(t, "rt-shared")
	hermeticProbeStub(t)

	divorceRefreshSpawn = func(context.Context, string) error { return nil } // exits clean, rotates nothing
	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"enroll-current", "--name", "seat", "--from", src,
		"--home", home, "--registry", regPath, "--job-view", "",
	})
	if rc != 0 {
		t.Fatalf("a failed divorce must not fail the enroll: rc=%d\nstderr=%s", rc, errb.String())
	}
	if h := findHome(t, regPath, "seat-netra"); h.Name == "" || !h.Active() {
		t.Fatalf("seat should still be enrolled active after a failed divorce: %+v", h)
	}
	stderr := errb.String()
	if !strings.Contains(stderr, "could NOT move off") {
		t.Errorf("a failed divorce must warn; got:\n%s", stderr)
	}
	if !strings.Contains(stderr, "cannot refresh") {
		t.Errorf("the warning should name the likely cause (a credential that cannot refresh); got:\n%s", stderr)
	}
}

// TestEnrollCurrent_TokenOnlySourceNeedsNoDivorce: a source whose only credential is a setup-token
// has no refresh token to share, so nothing should be spawned. An enroll must not burn a turn — or
// risk touching a credential — for a hazard that does not exist.
func TestEnrollCurrent_TokenOnlySourceNeedsNoDivorce(t *testing.T) {
	t.Setenv("FAK_JOB_ROSTER", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_ACCOUNT_SUFFIX", "-netra")
	t.Setenv("CLAUDE_CONFIG_DIR", "")
	home := t.TempDir()
	src := filepath.Join(home, ".claude")
	writeFileString(t, filepath.Join(src, ".oauth-token"), "sk-ant-oat-only\n")
	writeFileString(t, filepath.Join(src, ".claude.json"),
		`{"oauthAccount":{"emailAddress":"live@example.test","accountUuid":"uuid-live"}}`)
	hermeticProbeStub(t)

	spawned := false
	divorceRefreshSpawn = func(context.Context, string) error { spawned = true; return nil }
	t.Cleanup(func() { divorceRefreshSpawn = func(context.Context, string) error { return nil } })

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{
		"enroll-current", "--name", "seat", "--from", src,
		"--home", home, "--registry", filepath.Join(home, "registry.json"), "--job-view", "",
	})
	if rc != 0 {
		t.Fatalf("token-only enroll rc=%d\nstderr=%s", rc, errb.String())
	}
	if spawned {
		t.Error("a token-only source has no refresh token to share; the divorce must not spawn")
	}
}
