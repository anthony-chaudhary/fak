package main

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// guard_codex_oauth_test.go — coverage for the Codex ChatGPT-subscription credential
// resolver (guard_codex_oauth.go), the OpenAI twin of the Anthropic OAuth resolver. The
// load-bearing invariants: the access token and the ChatGPT account id are read as a
// MATCHED PAIR from the one auth.json that holds both; the account id is found wherever the
// Codex version put it (tokens.account_id, top-level, or the id_token JWT claim); and an
// API-key / empty login yields a clear error (not a subscription credential) so the caller
// can fall back to API billing.

// makeIDToken builds a syntactically valid JWT (header.payload.signature) whose payload
// carries the ChatGPT account id under the `https://api.openai.com/auth` claim, so the
// JWT-fallback path can be exercised without a real login. The signature segment is a
// throwaway — the resolver never verifies it (the account id is a routing header, not a
// trust decision).
func makeIDToken(t *testing.T, accountID string) string {
	t.Helper()
	enc := func(s string) string { return base64.RawURLEncoding.EncodeToString([]byte(s)) }
	header := enc(`{"alg":"RS256","typ":"JWT"}`)
	payload := enc(`{"https://api.openai.com/auth":{"chatgpt_account_id":"` + accountID + `"}}`)
	return header + "." + payload + ".not-a-real-signature"
}

// parseCodexSubscriptionCredential must extract the token + account id for a ChatGPT-mode
// login, find the account id wherever it lives, infer the mode when the field is absent,
// and refuse (with the mode still reported) when there is no OAuth token.
func TestParseCodexSubscriptionCredential(t *testing.T) {
	jwtOnly := makeIDToken(t, "acct-from-jwt")

	t.Run("chatgpt mode with tokens.account_id", func(t *testing.T) {
		raw := []byte(`{
			"OPENAI_API_KEY": null,
			"auth_mode": "chatgpt",
			"tokens": {"id_token": "hdr.pl.sig", "access_token": "at-123", "refresh_token": "rt", "account_id": "acct-nested"},
			"last_refresh": "2026-06-30T00:00:00Z"
		}`)
		got, err := parseCodexSubscriptionCredential(raw, "auth.json")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.AccessToken != "at-123" {
			t.Errorf("AccessToken = %q, want at-123", got.AccessToken)
		}
		if got.AccountID != "acct-nested" {
			t.Errorf("AccountID = %q, want acct-nested (tokens.account_id wins)", got.AccountID)
		}
		if got.AuthMode != "chatgpt" {
			t.Errorf("AuthMode = %q, want chatgpt", got.AuthMode)
		}
	})

	t.Run("top-level account_id when tokens.account_id absent", func(t *testing.T) {
		raw := []byte(`{"auth_mode":"chatgpt","account_id":"acct-top","tokens":{"access_token":"at"}}`)
		got, err := parseCodexSubscriptionCredential(raw, "auth.json")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.AccountID != "acct-top" {
			t.Errorf("AccountID = %q, want acct-top (top-level fallback)", got.AccountID)
		}
	})

	t.Run("account_id decoded from id_token JWT when not stored plainly", func(t *testing.T) {
		raw := []byte(`{"auth_mode":"chatgpt","tokens":{"id_token":"` + jwtOnly + `","access_token":"at"}}`)
		got, err := parseCodexSubscriptionCredential(raw, "auth.json")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.AccountID != "acct-from-jwt" {
			t.Errorf("AccountID = %q, want acct-from-jwt (JWT claim fallback)", got.AccountID)
		}
	})

	t.Run("auth_mode inferred chatgpt when field absent but token present", func(t *testing.T) {
		raw := []byte(`{"tokens":{"access_token":"at","account_id":"a"}}`)
		got, err := parseCodexSubscriptionCredential(raw, "auth.json")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got.AuthMode != "chatgpt" {
			t.Errorf("AuthMode = %q, want inferred chatgpt", got.AuthMode)
		}
	})

	t.Run("api-key login is not a subscription credential", func(t *testing.T) {
		raw := []byte(`{"OPENAI_API_KEY":"sk-live-abc","auth_mode":"apikey","tokens":null}`)
		got, err := parseCodexSubscriptionCredential(raw, "auth.json")
		if err == nil {
			t.Fatalf("expected an error for an API-key login, got credential %+v", got)
		}
		if got.AuthMode != "apikey" {
			t.Errorf("AuthMode = %q, want apikey reported on the miss", got.AuthMode)
		}
		if got.AccessToken != "" {
			t.Errorf("AccessToken = %q, want empty on an API-key login", got.AccessToken)
		}
	})

	t.Run("empty token is refused", func(t *testing.T) {
		raw := []byte(`{"tokens":{"access_token":"   "}}`)
		if _, err := parseCodexSubscriptionCredential(raw, "auth.json"); err == nil {
			t.Error("expected an error for a whitespace-only access token")
		}
	})

	t.Run("malformed JSON errors", func(t *testing.T) {
		if _, err := parseCodexSubscriptionCredential([]byte(`{not json`), "auth.json"); err == nil {
			t.Error("expected a parse error for malformed JSON")
		}
	})
}

// codexAccountIDFromIDToken must decode the claim from a valid JWT and degrade to "" on any
// malformed input rather than panicking on a hand-edited file.
func TestCodexAccountIDFromIDToken(t *testing.T) {
	if got := codexAccountIDFromIDToken(makeIDToken(t, "acct-xyz")); got != "acct-xyz" {
		t.Errorf("valid JWT: got %q, want acct-xyz", got)
	}
	cases := map[string]string{
		"empty":          "",
		"two-parts.only": "a.b",
		"not base64 !!!": "aaa.!!!not-base64!!!.ccc",
		"claim absent":   makeIDTokenNoClaim(t),
	}
	for name, tok := range cases {
		if got := codexAccountIDFromIDToken(tok); got != "" {
			t.Errorf("%s: got %q, want empty", name, got)
		}
	}
}

func makeIDTokenNoClaim(t *testing.T) string {
	t.Helper()
	enc := func(s string) string { return base64.RawURLEncoding.EncodeToString([]byte(s)) }
	return enc(`{"alg":"none"}`) + "." + enc(`{"sub":"user-1"}`) + ".sig"
}

// readCodexSubscriptionCredential must surface a clean, actionable error for a missing
// auth.json (the common "not logged in" case) and read a real file when present.
func TestReadCodexSubscriptionCredential(t *testing.T) {
	dir := t.TempDir()
	missing := filepath.Join(dir, "auth.json")
	_, err := readCodexSubscriptionCredential(missing)
	if err == nil || !strings.Contains(err.Error(), "codex login") {
		t.Fatalf("missing file: err = %v, want a `codex login` hint", err)
	}

	path := filepath.Join(dir, "auth.json")
	if werr := os.WriteFile(path, []byte(`{"auth_mode":"chatgpt","tokens":{"access_token":"at-file","account_id":"acct-file"}}`), 0o600); werr != nil {
		t.Fatal(werr)
	}
	got, err := readCodexSubscriptionCredential(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.AccessToken != "at-file" || got.AccountID != "acct-file" {
		t.Errorf("read = %+v, want token at-file / account acct-file", got)
	}
	if got.Source != path {
		t.Errorf("Source = %q, want the file path %q", got.Source, path)
	}
}

// resolveCodexSubscriptionCredential must honor CODEX_HOME and read <home>/auth.json from
// it, so the guard resolver targets the same login `codex` itself maintains.
func TestResolveCodexSubscriptionCredentialHonorsCodexHome(t *testing.T) {
	home := t.TempDir()
	t.Setenv("CODEX_HOME", home)
	if err := os.WriteFile(filepath.Join(home, "auth.json"),
		[]byte(`{"auth_mode":"chatgpt","tokens":{"access_token":"at-env","account_id":"acct-env"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := resolveCodexSubscriptionCredential("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got.AccessToken != "at-env" || got.AccountID != "acct-env" {
		t.Errorf("resolve = %+v, want token at-env / account acct-env from CODEX_HOME", got)
	}
}
