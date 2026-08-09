package accounts

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// credfamily_test.go — witnesses for the OAuth token-FAMILY hazard an adopt creates and for the
// divorce that resolves it. The scenario every test here encodes is the one observed on
// 2026-08-06: a seat enrolled by copying ~/.claude's credential refreshed, and the operator's own
// interactive session — holding the byte-identical old pair, its expiresAt still hours in the
// future — was 401'd out of existence with no warning.

// writeFamilyCred writes a credential carrying a specific refresh token (the family identity) and
// expiry, so a test can arrange "these two dirs share one family" precisely.
func writeFamilyCred(t *testing.T, dir, refreshToken string, expiresAtMs int64) {
	t.Helper()
	body := fmt.Sprintf(
		`{"claudeAiOauth":{"accessToken":"sk-ant-oat01-access","refreshToken":%s,"expiresAt":%d}}`,
		mustJSON(t, refreshToken), expiresAtMs,
	)
	if err := os.WriteFile(filepath.Join(dir, ".credentials.json"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func mustJSON(t *testing.T, s string) string {
	t.Helper()
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestRefreshFamilyID_NonSecretAndStable: the fingerprint identifies a family without ever being
// the token itself — the property that lets an operator-facing log line name a family safely.
func TestRefreshFamilyID_NonSecretAndStable(t *testing.T) {
	dir := t.TempDir()
	writeFamilyCred(t, dir, "rt-secret-value", 1)

	id := RefreshFamilyID(dir)
	if id == "" {
		t.Fatal("a dir with a refresh token must have a family id")
	}
	if id == "rt-secret-value" {
		t.Fatal("the family id must not be the token itself")
	}
	if len(id) != 8 {
		t.Fatalf("family id should be a short 8-hex fingerprint, got %q", id)
	}
	if again := RefreshFamilyID(dir); again != id {
		t.Fatalf("family id must be stable: %q then %q", id, again)
	}
}

// TestRefreshFamilyID_AbsentAndHollow: no credential and a HOLLOW credential (empty-string tokens,
// the shape a credential-dead seat has) both yield no family — never a bogus shared verdict.
func TestRefreshFamilyID_AbsentAndHollow(t *testing.T) {
	empty := t.TempDir()
	if id := RefreshFamilyID(empty); id != "" {
		t.Fatalf("a dir with no credential must have no family, got %q", id)
	}

	hollow := t.TempDir()
	body := `{"claudeAiOauth":{"accessToken":"","refreshToken":"","expiresAt":0}}`
	if err := os.WriteFile(filepath.Join(hollow, ".credentials.json"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	if id := RefreshFamilyID(hollow); id != "" {
		t.Fatalf("a hollow credential must have no family, got %q", id)
	}
}

// TestDetectSharedRefreshFamily_IdenticalTokenIsShared: the armed state — the exact byte-identical
// copy `copyLoginBundle` produces.
func TestDetectSharedRefreshFamily_IdenticalTokenIsShared(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	writeFamilyCred(t, src, "rt-same", 1)
	writeFamilyCred(t, dst, "rt-same", 1)

	got := DetectSharedRefreshFamily(src, dst)
	if !got.Shared {
		t.Fatal("two dirs holding an identical refresh token ARE sharing one family")
	}
	if got.SourceID != got.TargetID || got.FamilyID != got.TargetID {
		t.Fatalf("a shared family must report one id on all three fields, got %+v", got)
	}
}

// TestDetectSharedRefreshFamily_DistinctTokensNotShared: two dirs on their own families are safe
// even when they serve the SAME ACCOUNT. A duplicate account bucket is DetectEnrollCollision's
// concern; conflating the two would make every duplicate look like a credential emergency.
func TestDetectSharedRefreshFamily_DistinctTokensNotShared(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	writeFamilyCred(t, src, "rt-one", 1)
	writeFamilyCred(t, dst, "rt-two", 1)

	got := DetectSharedRefreshFamily(src, dst)
	if got.Shared {
		t.Fatal("distinct refresh tokens are distinct families, even for one account")
	}
	if got.SourceID == got.TargetID {
		t.Fatal("distinct families must have distinct fingerprints")
	}
}

// TestDetectSharedRefreshFamily_MissingSideNeverShared: an absent or hollow side cannot share.
func TestDetectSharedRefreshFamily_MissingSideNeverShared(t *testing.T) {
	withCred, without := t.TempDir(), t.TempDir()
	writeFamilyCred(t, withCred, "rt-only", 1)

	if DetectSharedRefreshFamily(without, withCred).Shared {
		t.Fatal("a source with no credential cannot share a family")
	}
	if DetectSharedRefreshFamily(withCred, without).Shared {
		t.Fatal("a target with no credential cannot share a family")
	}
}

// TestDivorce_MovesTargetOffSharedFamily: the happy path. The refresh rotates the target onto a new
// family, which is what makes the seat independently refreshable — and the report names the source
// dir, because a successful divorce is exactly what invalidates the source's credential and the
// caller must be able to tell the operator which dir now needs a login.
func TestDivorce_MovesTargetOffSharedFamily(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	nowMs := int64(1_700_000_000_000)
	writeFamilyCred(t, src, "rt-shared", nowMs-1)
	writeFamilyCred(t, dst, "rt-shared", nowMs-1)

	spawn := func(ctx context.Context, cfgDir string) error {
		// What a real `claude -p` does: a NEW access+refresh pair with a later expiry.
		writeFamilyCred(t, cfgDir, "rt-rotated", nowMs+3_600_000)
		return nil
	}
	rep := DivorceRefreshFamily(context.Background(), src, dst, spawn, fixedNow(nowMs))

	if rep.Outcome != DivorceDone {
		t.Fatalf("want %q, got %q (%+v)", DivorceDone, rep.Outcome, rep)
	}
	if !rep.Divorced() {
		t.Fatal("a completed divorce must report Divorced()")
	}
	if rep.Before == rep.After || rep.After == "" {
		t.Fatalf("the target's family must provably CHANGE: before=%q after=%q", rep.Before, rep.After)
	}
	if rep.SourceDir != src {
		t.Fatalf("the report must name the source dir that now needs a relogin, got %q", rep.SourceDir)
	}
	if DetectSharedRefreshFamily(src, dst).Shared {
		t.Fatal("after a divorce the two dirs must no longer share a family")
	}
}

// TestDivorce_NotNeededWhenFamiliesAlreadyDiffer: no shared family means no refresh is spawned at
// all. Enrolling a seat must not burn a token turn (or risk a rotation) for nothing.
func TestDivorce_NotNeededWhenFamiliesAlreadyDiffer(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	writeFamilyCred(t, src, "rt-one", 1)
	writeFamilyCred(t, dst, "rt-two", 1)

	spawned := false
	spawn := func(ctx context.Context, cfgDir string) error {
		spawned = true
		return nil
	}
	rep := DivorceRefreshFamily(context.Background(), src, dst, spawn, fixedNow(1))

	if rep.Outcome != DivorceNotNeeded {
		t.Fatalf("want %q, got %q", DivorceNotNeeded, rep.Outcome)
	}
	if spawned {
		t.Fatal("no shared family: the divorce must not spawn a refresh")
	}
}

// TestDivorce_ExitZeroWithoutRotationIsFailure: the anti-self-report witness. A spawn that succeeds
// but leaves the shared token in place must NOT be reported as divorced — that is the signature of
// a credential that cannot refresh, i.e. a seat headed for a human /login, and smoothing it over is
// how a hollow seat gets enrolled as healthy.
func TestDivorce_ExitZeroWithoutRotationIsFailure(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	nowMs := int64(1_700_000_000_000)
	writeFamilyCred(t, src, "rt-shared", nowMs-1)
	writeFamilyCred(t, dst, "rt-shared", nowMs-1)

	spawn := func(ctx context.Context, cfgDir string) error { return nil } // exits clean, rotates nothing
	rep := DivorceRefreshFamily(context.Background(), src, dst, spawn, fixedNow(nowMs))

	if rep.Outcome != DivorceFailed {
		t.Fatalf("want %q, got %q (%+v)", DivorceFailed, rep.Outcome, rep)
	}
	if rep.Divorced() {
		t.Fatal("a failed divorce must not claim the seat owns its own family")
	}
	if !DetectSharedRefreshFamily(src, dst).Shared {
		t.Fatal("the hazard is still armed after a failed divorce; the fixture should show it")
	}
}

// TestDivorce_ClearedCredentialIsFailure: the API-key failure mode TriggerRefresh guards against —
// the spawn BLANKS the credential. That leaves the seat with no family at all, which must be a
// failure, never "the fingerprint changed, so we divorced".
func TestDivorce_ClearedCredentialIsFailure(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	nowMs := int64(1_700_000_000_000)
	writeFamilyCred(t, src, "rt-shared", nowMs-1)
	writeFamilyCred(t, dst, "rt-shared", nowMs-1)

	spawn := func(ctx context.Context, cfgDir string) error {
		return os.WriteFile(filepath.Join(cfgDir, ".credentials.json"), []byte(`{}`), 0o600)
	}
	rep := DivorceRefreshFamily(context.Background(), src, dst, spawn, fixedNow(nowMs))

	if rep.Outcome != DivorceFailed {
		t.Fatalf("a cleared credential must be %q, got %q", DivorceFailed, rep.Outcome)
	}
	if rep.After != "" {
		t.Fatalf("a cleared credential has no family, got %q", rep.After)
	}
}

// TestDivorce_SpawnErrorSurfaces: the spawn's error is carried, never swallowed.
func TestDivorce_SpawnErrorSurfaces(t *testing.T) {
	src, dst := t.TempDir(), t.TempDir()
	nowMs := int64(1_700_000_000_000)
	writeFamilyCred(t, src, "rt-shared", nowMs-1)
	writeFamilyCred(t, dst, "rt-shared", nowMs-1)

	want := fmt.Errorf("claude exploded")
	spawn := func(ctx context.Context, cfgDir string) error { return want }
	rep := DivorceRefreshFamily(context.Background(), src, dst, spawn, fixedNow(nowMs))

	if rep.Err == nil {
		t.Fatal("the spawn error must be surfaced on the report")
	}
	if rep.Outcome != DivorceFailed {
		t.Fatalf("a failed spawn leaves the hazard armed: want %q, got %q", DivorceFailed, rep.Outcome)
	}
}
