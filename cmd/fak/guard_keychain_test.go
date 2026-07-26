package main

import (
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// stubGuardKeychain pins cmd/fak's keychain seams to a fixed credential (tok==""/exp==0
// permutations model the miss cases) for the test's duration, so the guard-side wiring
// (#5363) is testable on every GOOS without a real macOS Keychain.
func stubGuardKeychain(t *testing.T, tok string, exp int64) {
	t.Helper()
	prevSup, prevTok, prevCred := guardKeychainSupported, guardKeychainAccessToken, guardKeychainCred
	t.Cleanup(func() {
		guardKeychainSupported, guardKeychainAccessToken, guardKeychainCred = prevSup, prevTok, prevCred
	})
	guardKeychainSupported = func() bool { return true }
	guardKeychainCred = func(dir string) (accounts.KeychainCred, bool) {
		if tok == "" {
			return accounts.KeychainCred{}, false
		}
		return accounts.KeychainCred{AccessToken: tok, ExpiresAt: exp}, true
	}
	guardKeychainAccessToken = func(dir string, now time.Time) (string, bool) {
		if tok == "" {
			return "", false
		}
		if exp > 0 && exp < now.UnixMilli() {
			return "", false
		}
		return tok, true
	}
}

// TestGuardOAuthLoaderKeychainPrecedence pins the keychain rung's position in the token
// ladder: below the named env var and the credentials FILE, above the long-lived
// .oauth-token setup-token fallback.
func TestGuardOAuthLoaderKeychainPrecedence(t *testing.T) {
	dir := t.TempDir()
	stubGuardKeychain(t, "sk-ant-oat01-keychain", 0)
	now := func() time.Time { return time.UnixMilli(1_000_000) }

	// Keychain vs setup token: keychain (the active login) wins.
	if err := os.WriteFile(filepath.Join(dir, ".oauth-token"), []byte("sk-ant-oat01-setup"), 0o600); err != nil {
		t.Fatal(err)
	}
	loader, tried := guardAnthropicOAuthLoader("", dir, now, io.Discard)
	tok, src, ok := loader.LookupSource(guardAnthropicOAuthSecretKey)
	if !ok || tok != "sk-ant-oat01-keychain" || src != guardOAuthKeychainSourceName {
		t.Fatalf("LookupSource = (%q,%q,%v), want the keychain token above the setup token", tok, src, ok)
	}
	found := false
	for _, name := range tried {
		if name == guardOAuthKeychainSourceName {
			found = true
		}
	}
	if !found {
		t.Fatalf("tried list %v must name the keychain source for the not-found diagnostic", tried)
	}

	// Credentials FILE vs keychain: a present-and-live file keeps winning.
	if err := os.WriteFile(filepath.Join(dir, ".credentials.json"),
		[]byte(`{"claudeAiOauth":{"accessToken":"sk-ant-oat01-file","expiresAt":2000000}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	loader, _ = guardAnthropicOAuthLoader("", dir, now, io.Discard)
	if tok, _, ok := loader.LookupSource(guardAnthropicOAuthSecretKey); !ok || tok != "sk-ant-oat01-file" {
		t.Fatalf("got %q, want the credentials-file token to outrank the keychain", tok)
	}

	// An EXPIRED file token must fall through to the live keychain one, not block it.
	if err := os.WriteFile(filepath.Join(dir, ".credentials.json"),
		[]byte(`{"claudeAiOauth":{"accessToken":"sk-ant-oat01-stale","expiresAt":1}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	loader, _ = guardAnthropicOAuthLoader("", dir, now, io.Discard)
	if tok, _, ok := loader.LookupSource(guardAnthropicOAuthSecretKey); !ok || tok != "sk-ant-oat01-keychain" {
		t.Fatalf("got %q, want the keychain token once the file token expired", tok)
	}
}

// TestGuardOAuthLoaderKeychainUnsupported pins the non-darwin posture: no keychain rung
// in the ladder, no keychain row in the "looked in:" list — byte-for-byte the historical
// disk-only behavior.
func TestGuardOAuthLoaderKeychainUnsupported(t *testing.T) {
	prevSup := guardKeychainSupported
	t.Cleanup(func() { guardKeychainSupported = prevSup })
	guardKeychainSupported = func() bool { return false }

	_, tried := guardAnthropicOAuthLoader("", t.TempDir(), func() time.Time { return time.Unix(0, 0) }, io.Discard)
	for _, name := range tried {
		if name == guardOAuthKeychainSourceName {
			t.Fatalf("unsupported platform must not name the keychain source: %v", tried)
		}
	}
}

// TestResolveGuardUpstreamKeychainAPIKey pins the API-billing adoption (#5363): with no
// subscription token anywhere but Claude Code's saved API key in the macOS Keychain,
// resolveGuardUpstream adopts the key upstream (API billing, not pinned OAuth, no
// passthrough demotion, headless spawn permitted) — and without the key the historical
// passthrough/noTokenAnywhere posture is untouched.
func TestResolveGuardUpstreamKeychainAPIKey(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("CLAUDE_CONFIG_DIR", dir)
	const tokenEnv = "FAK_TEST_GUARD_OAUTH_KC"
	t.Setenv(tokenEnv, "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	prevAPI := guardKeychainAPIKey
	t.Cleanup(func() { guardKeychainAPIKey = prevAPI })

	guardKeychainAPIKey = func(d string) (string, bool) {
		if d != dir {
			t.Errorf("API-key probe asked about %q, want the guard config dir %q", d, dir)
		}
		return "sk-ant-api03-keychain", true
	}
	us := resolveGuardUpstream("anthropic", "claude", "", "", "", false, tokenEnv)
	if !us.keychainAPIKey || us.apiKey != "sk-ant-api03-keychain" {
		t.Fatalf("want the keychain API key adopted, got %+v", us)
	}
	if us.pinUpstream || us.oauthSource != "" {
		t.Fatalf("an adopted API key is API billing, never a pinned OAuth posture: %+v", us)
	}
	if us.passthroughFallback || us.noTokenAnywhere {
		t.Fatalf("adopted key must not demote to passthrough or refuse headless spawn: %+v", us)
	}

	guardKeychainAPIKey = func(string) (string, bool) { return "", false }
	us = resolveGuardUpstream("anthropic", "claude", "", "", "", false, tokenEnv)
	if us.keychainAPIKey || us.apiKey != "" {
		t.Fatalf("no keychain key: nothing to adopt, got %+v", us)
	}
	if !us.passthroughFallback || !us.noTokenAnywhere {
		t.Fatalf("no credential anywhere must keep the historical fail-loud posture: %+v", us)
	}
}

// TestCredExpiresAtKeychainFallback pins the headless StaleCred rung's #5363 fallback:
// with no credential FILE, the keychain answers under credExpiresAt's own contract —
// including the "no recorded expiry reads as always-fresh" convention, which the strict
// send-safe sources deliberately do not share.
func TestCredExpiresAtKeychainFallback(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")

	stubGuardKeychain(t, "tok", 1234567)
	exp, ok := credExpiresAt(credPath)
	if !ok || !exp.Equal(time.UnixMilli(1234567)) {
		t.Fatalf("credExpiresAt = (%v,%v), want the keychain expiry", exp, ok)
	}

	stubGuardKeychain(t, "tok", 0)
	if exp, ok := credExpiresAt(credPath); !ok || !exp.IsZero() {
		t.Fatalf("no recorded expiry: got (%v,%v), want (zero,true) — the always-fresh convention", exp, ok)
	}

	stubGuardKeychain(t, "", 0)
	if _, ok := credExpiresAt(credPath); ok {
		t.Fatal("no file and no keychain credential: nothing to vouch for")
	}

	// A non-credentials basename (an operator-named path) must never consult the keychain.
	stubGuardKeychain(t, "tok", 1234567)
	if _, ok := credExpiresAt(filepath.Join(dir, "some-other.json")); ok {
		t.Fatal("non-.credentials.json path must stay file-only")
	}
}

// TestReadLiveAccessTokenKeychainFallback pins the failover picker's fallback under its
// STRICT contract: positive recorded expiry, strictly after now — a keychain token with
// no expiry is not failover-eligible, exactly like its file counterpart.
func TestReadLiveAccessTokenKeychainFallback(t *testing.T) {
	dir := t.TempDir()
	now := time.UnixMilli(1_000_000)

	stubGuardKeychain(t, "tok-live", 2_000_000)
	if tok, ok := readLiveAccessToken(dir, now); !ok || tok != "tok-live" {
		t.Fatalf("got (%q,%v), want the live keychain token", tok, ok)
	}

	stubGuardKeychain(t, "tok-dead", 1)
	if _, ok := readLiveAccessToken(dir, now); ok {
		t.Fatal("expired keychain token must not be failover-eligible")
	}

	stubGuardKeychain(t, "tok-eternal", 0)
	if _, ok := readLiveAccessToken(dir, now); ok {
		t.Fatal("keychain token without positive expiry must not be failover-eligible")
	}

	// The FILE still wins when present and live.
	if err := os.WriteFile(filepath.Join(dir, ".credentials.json"),
		[]byte(`{"claudeAiOauth":{"accessToken":"tok-file","expiresAt":2000000}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	stubGuardKeychain(t, "tok-keychain", 2_000_000)
	if tok, ok := readLiveAccessToken(dir, now); !ok || tok != "tok-file" {
		t.Fatalf("got (%q,%v), want the file token to outrank the keychain", tok, ok)
	}
}
