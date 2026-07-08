package accounts

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// writeCredsJSON seeds a dir's .credentials.json with a live session access token.
func writeCredsJSON(t *testing.T, dir, accessToken string) {
	t.Helper()
	doc := map[string]any{"claudeAiOauth": map[string]any{
		"accessToken":  accessToken,
		"refreshToken": "rt-" + accessToken,
	}}
	b, _ := json.MarshalIndent(doc, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, ".credentials.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
}

// writeClaudeJSONAccount seeds a dir's .claude.json oauthAccount metadata.
func writeClaudeJSONAccount(t *testing.T, dir, email, uuid string) {
	t.Helper()
	doc := map[string]any{"oauthAccount": map[string]any{
		"emailAddress": email,
		"accountUuid":  uuid,
	}}
	b, _ := json.MarshalIndent(doc, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, ".claude.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// profileServer stands in for Anthropic's OAuth profile endpoint, answering with the account
// keyed by the bearer token so a test can serve different identities per credential.
func profileServer(t *testing.T, byToken map[string]ProbedIdentity) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tok := r.Header.Get("Authorization")
		const p = "Bearer "
		if len(tok) > len(p) {
			tok = tok[len(p):]
		}
		id, ok := byToken[tok]
		if !ok {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"account": map[string]any{
			"uuid":  id.AccountUUID,
			"email": id.Email,
		}})
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestResolveCredentialIdentity_CredentialWinsOverStaleDisk is the #3216 acceptance: a dir whose
// .claude.json names account A but whose .credentials.json serves account B resolves as B, and
// the disagreement is flagged Stale.
func TestResolveCredentialIdentity_CredentialWinsOverStaleDisk(t *testing.T) {
	dir := t.TempDir()
	writeClaudeJSONAccount(t, dir, "july9@example.test", "4d04b4b1-aaaa") // stale metadata (A)
	writeCredsJSON(t, dir, "at-july11")                                   // live credential (B)
	srv := profileServer(t, map[string]ProbedIdentity{
		"at-july11": {Email: "july11@example.test", AccountUUID: "967c23c1-bbbb"},
	})
	probe := func(tok string) (ProbedIdentity, error) { return ProbeToken(nil, srv.URL, tok) }

	res := ResolveCredentialIdentity(dir, probe)

	if !res.Probed {
		t.Fatalf("expected a live probe, got Probed=false (err=%v)", res.ProbeErr)
	}
	if !res.Stale {
		t.Errorf("expected Stale=true (disk A != credential B)")
	}
	if res.Resolved.AccountUUID != "967c23c1-bbbb" || res.Resolved.Email != "july11@example.test" {
		t.Errorf("resolved identity = %q/%q, want credential july11 (967c23c1-bbbb)", res.Resolved.Email, res.Resolved.AccountUUID)
	}
	if res.Resolved.AccountKey() != "uuid:967c23c1-bbbb" {
		t.Errorf("resolved AccountKey = %q, want uuid:967c23c1-bbbb (the credential bucket)", res.Resolved.AccountKey())
	}
}

// TestResolveCredentialIdentity_AgreementNotStale: when disk metadata and the credential name the
// same account, nothing is stale and the resolved identity is unchanged.
func TestResolveCredentialIdentity_AgreementNotStale(t *testing.T) {
	dir := t.TempDir()
	writeClaudeJSONAccount(t, dir, "july11@example.test", "967c23c1-bbbb")
	writeCredsJSON(t, dir, "at-july11")
	srv := profileServer(t, map[string]ProbedIdentity{
		"at-july11": {Email: "july11@example.test", AccountUUID: "967c23c1-bbbb"},
	})
	probe := func(tok string) (ProbedIdentity, error) { return ProbeToken(nil, srv.URL, tok) }

	res := ResolveCredentialIdentity(dir, probe)
	if res.Stale {
		t.Errorf("identities agree; Stale should be false")
	}
	if res.Resolved.AccountUUID != "967c23c1-bbbb" {
		t.Errorf("resolved uuid = %q, want 967c23c1-bbbb", res.Resolved.AccountUUID)
	}
}

// TestResolveCredentialIdentity_EmptyDiskFilledNotStale: an empty disk identity is filled from the
// credential without being flagged stale (there was nothing to be wrong, only a gap).
func TestResolveCredentialIdentity_EmptyDiskFilledNotStale(t *testing.T) {
	dir := t.TempDir()
	writeCredsJSON(t, dir, "at-july11") // no .claude.json at all
	srv := profileServer(t, map[string]ProbedIdentity{
		"at-july11": {Email: "july11@example.test", AccountUUID: "967c23c1-bbbb"},
	})
	probe := func(tok string) (ProbedIdentity, error) { return ProbeToken(nil, srv.URL, tok) }

	res := ResolveCredentialIdentity(dir, probe)
	if res.Stale {
		t.Errorf("empty disk identity is a gap, not a stale mislabel")
	}
	if res.Resolved.AccountUUID != "967c23c1-bbbb" {
		t.Errorf("resolved uuid = %q, want the probed credential identity", res.Resolved.AccountUUID)
	}
}

// TestResolveCredentialIdentity_NoProberOrNoCredIsDiskOnly: with no prober, or no live credential,
// the resolution is exactly DeriveIdentity — the pure disk-only fallback.
func TestResolveCredentialIdentity_NoProberOrNoCredIsDiskOnly(t *testing.T) {
	dir := t.TempDir()
	writeClaudeJSONAccount(t, dir, "july9@example.test", "4d04b4b1-aaaa")
	writeCredsJSON(t, dir, "at-july11")

	// nil prober → disk-only
	if res := ResolveCredentialIdentity(dir, nil); res.Probed || res.Resolved.AccountUUID != "4d04b4b1-aaaa" {
		t.Errorf("nil prober should be disk-only; got Probed=%v uuid=%q", res.Probed, res.Resolved.AccountUUID)
	}

	// prober present but no credential file → disk-only, no probe
	bare := t.TempDir()
	writeClaudeJSONAccount(t, bare, "july9@example.test", "4d04b4b1-aaaa")
	probe := func(string) (ProbedIdentity, error) { return ProbedIdentity{Email: "x@y", AccountUUID: "z"}, nil }
	if res := ResolveCredentialIdentity(bare, probe); res.Probed {
		t.Errorf("no live credential to probe; Probed should be false")
	}
}

// TestResolveCredentialIdentity_ProbeErrorFallsBackToDisk: a probe transport error is non-fatal —
// Resolved falls back to disk and ProbeErr is surfaced.
func TestResolveCredentialIdentity_ProbeErrorFallsBackToDisk(t *testing.T) {
	dir := t.TempDir()
	writeClaudeJSONAccount(t, dir, "july9@example.test", "4d04b4b1-aaaa")
	writeCredsJSON(t, dir, "at-unknown") // server will 401 this token
	srv := profileServer(t, map[string]ProbedIdentity{"at-known": {Email: "e", AccountUUID: "u"}})
	probe := func(tok string) (ProbedIdentity, error) { return ProbeToken(nil, srv.URL, tok) }

	res := ResolveCredentialIdentity(dir, probe)
	if res.Probed {
		t.Errorf("a 401 probe is not a successful probe")
	}
	if res.ProbeErr == nil {
		t.Errorf("expected ProbeErr to be surfaced")
	}
	if res.Resolved.AccountUUID != "4d04b4b1-aaaa" {
		t.Errorf("resolved should fall back to disk on probe error, got %q", res.Resolved.AccountUUID)
	}
}
