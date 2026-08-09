package accounts

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// credexpirynudge_test.go — witnesses for the expiry nudge, the fix for a false alarm dogfooded on
// 2026-08-06: `fak accounts refresh` graded a seat refreshed 20 minutes earlier as "stale — refresh
// token likely dead", because Claude Code correctly declines to rotate a token that is not near
// expiry. A no-op on a FRESH credential is health, not death; forcing a rotation requires making the
// token look due first.

// TestCredentialDueForRefresh_Windows: due when expired or inside the window, not due when comfortably
// ahead of it, and due when there is nothing readable to judge (never call an unreadable seat healthy).
func TestCredentialDueForRefresh_Windows(t *testing.T) {
	now := time.UnixMilli(1_700_000_000_000)

	fresh := t.TempDir()
	writeFamilyCred(t, fresh, "rt", now.Add(8*time.Hour).UnixMilli())
	if CredentialDueForRefresh(fresh, now) {
		t.Error("a credential 8h from expiry is NOT due for refresh")
	}

	soon := t.TempDir()
	writeFamilyCred(t, soon, "rt", now.Add(NudgeWindow/2).UnixMilli())
	if !CredentialDueForRefresh(soon, now) {
		t.Error("a credential inside the nudge window IS due")
	}

	expired := t.TempDir()
	writeFamilyCred(t, expired, "rt", now.Add(-time.Hour).UnixMilli())
	if !CredentialDueForRefresh(expired, now) {
		t.Error("an expired credential IS due")
	}

	if !CredentialDueForRefresh(t.TempDir(), now) {
		t.Error("a dir with no credential must be treated as due, never as healthy")
	}
}

// TestNudgeExpiryForRefresh_BackdatesThenRestoresVerbatim: the nudge must move ONLY the expiry, and
// the restore must return the original bytes exactly — otherwise a failed refresh would leave a
// perfectly valid credential looking dead.
func TestNudgeExpiryForRefresh_BackdatesThenRestoresVerbatim(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, ".credentials.json")
	// Include an unrelated field to prove a round-trip does not drop data it does not understand.
	original := `{"claudeAiOauth":{"accessToken":"at","refreshToken":"rt","expiresAt":4102444800000,"scopes":["user:inference"]},"somethingElse":{"keep":true}}`
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	now := time.UnixMilli(1_700_000_000_000)

	restore, err := NudgeExpiryForRefresh(dir, now)
	if err != nil {
		t.Fatalf("nudge: %v", err)
	}
	if CredentialDueForRefresh(dir, now) != true {
		t.Fatal("after a nudge the credential must look DUE")
	}

	// Tokens and unknown fields survive the nudge untouched.
	var doc map[string]any
	b, _ := os.ReadFile(path)
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatalf("nudged file is not valid JSON: %v", err)
	}
	block := doc["claudeAiOauth"].(map[string]any)
	if block["accessToken"] != "at" || block["refreshToken"] != "rt" {
		t.Errorf("the nudge must never touch the tokens, got %+v", block)
	}
	if block["scopes"] == nil || doc["somethingElse"] == nil {
		t.Errorf("the nudge dropped fields it does not understand: %+v", doc)
	}
	if RefreshFamilyID(dir) == "" {
		t.Error("the nudge must leave the token family intact")
	}

	if err := restore(); err != nil {
		t.Fatalf("restore: %v", err)
	}
	got, _ := os.ReadFile(path)
	if string(got) != original {
		t.Fatalf("restore must rewrite the ORIGINAL bytes verbatim:\n got: %s\nwant: %s", got, original)
	}
}

// TestNudgeExpiryForRefresh_RefusesHollowAndUnparseable: a torn or hollow credential is not something
// to rewrite — the caller's own check reports the real problem instead.
func TestNudgeExpiryForRefresh_RefusesHollowAndUnparseable(t *testing.T) {
	now := time.Now()

	hollow := t.TempDir()
	if err := os.WriteFile(filepath.Join(hollow, ".credentials.json"),
		[]byte(`{"claudeAiOauth":{"accessToken":"","refreshToken":""}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := NudgeExpiryForRefresh(hollow, now); err == nil {
		t.Error("a hollow credential must be refused, not nudged")
	}

	torn := t.TempDir()
	if err := os.WriteFile(filepath.Join(torn, ".credentials.json"), []byte(`{"claudeAiOa`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := NudgeExpiryForRefresh(torn, now); err == nil {
		t.Error("an unparseable credential must be refused, not nudged")
	}

	if _, err := NudgeExpiryForRefresh(t.TempDir(), now); err == nil {
		t.Error("a missing credential must be refused")
	}
}

// TestDivorce_NudgesAFreshCopyIntoRotating is the regression for the real-world case: the credential
// an adopt copies is nowhere near expiry, so without the nudge the refresh is a legitimate no-op and
// the two dirs stay on ONE family — the exact armed state that later logged an operator out.
func TestDivorce_NudgesAFreshCopyIntoRotating(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	farFuture := time.Now().Add(8 * time.Hour).UnixMilli()
	writeFamilyCred(t, src, "rt-shared", farFuture)
	writeFamilyCred(t, dst, "rt-shared", farFuture)

	// A realistic spawn: it rotates ONLY when it sees an expired access token, exactly like the CLI.
	spawn := func(_ context.Context, cfgDir string) error {
		if !CredentialDueForRefresh(cfgDir, time.Now()) {
			return nil // nothing to do — the behavior that made the un-nudged divorce a no-op
		}
		writeFamilyCred(t, cfgDir, "rt-rotated", time.Now().Add(8*time.Hour).UnixMilli())
		return nil
	}
	rep := DivorceRefreshFamily(context.Background(), src, dst, spawn, nil)

	if rep.Outcome != DivorceDone {
		t.Fatalf("a fresh copy must still be divorced (via the expiry nudge): got %q (%+v)", rep.Outcome, rep)
	}
	if DetectSharedRefreshFamily(src, dst).Shared {
		t.Fatal("the seat still shares its source's token family")
	}
}

// TestDivorce_RestoresExpiryWhenNudgedRefreshFails: if the nudge does not buy a rotation, the seat's
// credential must be left exactly as it was — still valid, not looking expired.
func TestDivorce_RestoresExpiryWhenNudgedRefreshFails(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	farFuture := time.Now().Add(8 * time.Hour).UnixMilli()
	writeFamilyCred(t, src, "rt-shared", farFuture)
	writeFamilyCred(t, dst, "rt-shared", farFuture)
	before, _ := os.ReadFile(filepath.Join(dst, ".credentials.json"))

	spawn := func(_ context.Context, _ string) error { return nil } // never rotates
	rep := DivorceRefreshFamily(context.Background(), src, dst, spawn, nil)

	if rep.Outcome != DivorceFailed {
		t.Fatalf("want %q, got %q", DivorceFailed, rep.Outcome)
	}
	after, _ := os.ReadFile(filepath.Join(dst, ".credentials.json"))
	if string(after) != string(before) {
		t.Fatalf("a failed divorce must leave the credential byte-identical:\n got: %s\nwant: %s", after, before)
	}
	if CredentialDueForRefresh(dst, time.Now()) {
		t.Fatal("the backdated expiry was not restored: a valid credential now looks expired")
	}
}
