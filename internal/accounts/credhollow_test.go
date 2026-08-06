package accounts

import (
	"os"
	"path/filepath"
	"testing"
)

// credhollow_test.go — the distinction that decides whether a scheduled refresh sweep ALERTS or
// stays quiet. Witnessed 2026-08-06: a seat whose refresh grant had failed was left with both OAuth
// tokens blanked, the sweep graded it "skipped — nothing to refresh", and the scheduled task exited
// 0 with the roster silently down a seat.

// TestCredentialHollow_DistinguishesBlankedFromAbsent: a blanked credential is a seat that needs a
// human login; a dir with no session credential is an api-key/setup-token seat with nothing to do.
func TestCredentialHollow_DistinguishesBlankedFromAbsent(t *testing.T) {
	hollow := t.TempDir()
	// The exact shape observed on disk: tokens present as empty strings, expiry zeroed, the rest of
	// the block (scopes, subscriptionType) still there.
	body := `{"claudeAiOauth":{"accessToken":"","refreshToken":"","expiresAt":0,` +
		`"scopes":["user:inference"],"subscriptionType":"max"}}`
	if err := os.WriteFile(filepath.Join(hollow, ".credentials.json"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	if !CredentialHollow(hollow) {
		t.Error("a .credentials.json with both tokens blank IS hollow — the shape a failed refresh leaves")
	}

	absent := t.TempDir()
	if CredentialHollow(absent) {
		t.Error("a dir with NO session credential is absent, not hollow (an api-key/token-only seat must not alarm)")
	}

	tokenOnly := t.TempDir()
	if err := os.WriteFile(filepath.Join(tokenOnly, ".oauth-token"), []byte("sk-ant-oat-x\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if CredentialHollow(tokenOnly) {
		t.Error("a setup-token-only seat is not hollow: it has a credential, just not a refreshable one")
	}

	live := t.TempDir()
	writeFamilyCred(t, live, "rt-live", 4102444800000)
	if CredentialHollow(live) {
		t.Error("a live credential must never grade hollow")
	}

	if CredentialHollow("") {
		t.Error("an empty dir path is not hollow")
	}
}

// TestCredentialHollow_HalfBlankedIsNotHollow: only BOTH tokens empty is the failed-refresh shape. A
// credential retaining a refresh token is still revivable without a human, so it must not be graded
// as needing a login.
func TestCredentialHollow_HalfBlankedIsNotHollow(t *testing.T) {
	dir := t.TempDir()
	body := `{"claudeAiOauth":{"accessToken":"","refreshToken":"rt-still-here","expiresAt":0}}`
	if err := os.WriteFile(filepath.Join(dir, ".credentials.json"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	if CredentialHollow(dir) {
		t.Error("an expired access token with an intact refresh token is refreshable, not hollow")
	}
	if RefreshFamilyID(dir) == "" {
		t.Error("the surviving refresh token must still identify a family")
	}
}
