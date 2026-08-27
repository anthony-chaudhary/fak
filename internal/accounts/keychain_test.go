package accounts

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// stubKeychain routes the darwin exec seam to an in-memory map for the test's duration
// (restoring the real seam and clearing the TTL cache on cleanup), so every keychain
// test runs identically on all GOOS — including non-darwin CI, where the real seam is
// nil and the fallback is otherwise dormant.
func stubKeychain(t *testing.T, items map[string]string) *int {
	t.Helper()
	calls := 0
	prev := claudeKeychainReadPassword
	claudeKeychainReadPassword = func(service, account string) ([]byte, error) {
		calls++
		if v, ok := items[service]; ok {
			return []byte(v), nil
		}
		return nil, errors.New("item not found")
	}
	resetClaudeKeychainCache()
	t.Cleanup(func() {
		claudeKeychainReadPassword = prev
		resetClaudeKeychainCache()
	})
	return &calls
}

func keychainServiceFor(dir string) string {
	sum := sha256.Sum256([]byte(dir))
	return claudeKeychainServiceBase + "-" + hex.EncodeToString(sum[:])[:8]
}

func keychainAPIKeyServiceFor(dir string) string {
	sum := sha256.Sum256([]byte(dir))
	return claudeKeychainAPIKeyBase + "-" + hex.EncodeToString(sum[:])[:8]
}

func TestClaudeKeychainServicesNaming(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		t.Skip("no resolvable home dir")
	}
	def := filepath.Join(home, ".claude")
	got := claudeKeychainServices(claudeKeychainServiceBase, def)
	want := []string{claudeKeychainServiceBase, keychainServiceFor(def)}
	if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("default home services = %v, want %v (unsuffixed first, own hash second)", got, want)
	}

	other := filepath.Join(string(filepath.Separator), "seats", "alpha", ".claude")
	got = claudeKeychainServices(claudeKeychainServiceBase, other)
	if len(got) != 1 || got[0] != keychainServiceFor(other) {
		t.Fatalf("non-default services = %v, want only the hash-suffixed service %q", got, keychainServiceFor(other))
	}

	if got := claudeKeychainServices(claudeKeychainServiceBase, ""); got != nil {
		t.Fatalf("empty dir must probe nothing, got %v", got)
	}
}

// TestClaudeKeychainAPIKey pins the API-billing rung (#5363): the bare `Claude Code`
// item's raw key is adopted; a JSON payload, a multi-line value, or a blank item is
// rejected rather than sent upstream as a doomed bearer.
func TestClaudeKeychainAPIKey(t *testing.T) {
	dir := t.TempDir()
	service := keychainAPIKeyServiceFor(dir)

	stubKeychain(t, map[string]string{service: "sk-ant-api03-abc123\n"})
	if key, ok := ClaudeKeychainAPIKey(dir); !ok || key != "sk-ant-api03-abc123" {
		t.Fatalf("got (%q,%v), want the trimmed raw key", key, ok)
	}

	for name, body := range map[string]string{
		"json payload": `{"mcpOAuth":{}}`,
		"multi line":   "sk-ant\nsecond-line",
		"blank":        "   \n",
	} {
		stubKeychain(t, map[string]string{service: body})
		if key, ok := ClaudeKeychainAPIKey(dir); ok {
			t.Fatalf("%s: got (%q,%v), want a miss — a mis-shaped value must not be adopted", name, key, ok)
		}
	}

	// The OAuth credential item must never satisfy the API-key probe (distinct services).
	stubKeychain(t, map[string]string{
		keychainServiceFor(dir): `{"claudeAiOauth":{"accessToken":"tok","expiresAt":9999999999999}}`,
	})
	if _, ok := ClaudeKeychainAPIKey(dir); ok {
		t.Fatal("an OAuth-credentials item must not answer the API-key probe")
	}
}

func TestParseClaudeKeychainCred(t *testing.T) {
	cases := []struct {
		name string
		body string
		ok   bool
	}{
		{"live login", `{"claudeAiOauth":{"accessToken":"sk-ant-oat01-x","refreshToken":"r","expiresAt":123}}`, true},
		{"refresh only", `{"claudeAiOauth":{"accessToken":"","refreshToken":"r"}}`, true},
		{"placeholder no tokens", `{"claudeAiOauth":{"accessToken":"","refreshToken":""}}`, false},
		{"no claudeAiOauth", `{"other":true}`, false},
		{"garbage", `not-json`, false},
	}
	for _, tc := range cases {
		if _, ok := parseClaudeKeychainCred([]byte(tc.body)); ok != tc.ok {
			t.Errorf("%s: ok=%v, want %v", tc.name, ok, tc.ok)
		}
	}
}

// TestHasClaudeCredentialsKeychainFallback pins #5363's core behavior change: a config
// home with no credential FILES reads as logged-in when the keychain holds the login,
// and a placeholder credentials file (the July-4 rule) no longer masks it. With no
// keychain seam (non-darwin, or an un-stubbed test) the disk-only answer is unchanged.
func TestHasClaudeCredentialsKeychainFallback(t *testing.T) {
	dir := t.TempDir()

	// Baseline first: with a nil seam the fallback is inert.
	prev := claudeKeychainReadPassword
	t.Cleanup(func() {
		claudeKeychainReadPassword = prev
		resetClaudeKeychainCache()
	})
	claudeKeychainReadPassword = nil
	resetClaudeKeychainCache()
	if hasClaudeCredentials(dir) {
		t.Fatal("no files, no keychain seam: must read as no creds")
	}
	claudeKeychainReadPassword = prev

	stubKeychain(t, map[string]string{
		keychainServiceFor(dir): `{"claudeAiOauth":{"accessToken":"sk-ant-oat01-live","expiresAt":9999999999999}}`,
	})
	if !hasClaudeCredentials(dir) {
		t.Fatal("keychain login present: must read as creds")
	}

	// A placeholder file (claudeAiOauth object, empty tokens) is not a login on its
	// own — but must fall through to the keychain, not veto it.
	if err := os.WriteFile(filepath.Join(dir, ".credentials.json"),
		[]byte(`{"claudeAiOauth":{"accessToken":"","refreshToken":""}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if !hasClaudeCredentials(dir) {
		t.Fatal("placeholder file beside a keychain login: must still read as creds")
	}
}

// TestExpiredKeychainOAuthNeedsLogin pins #9344's operator-visible failure: token
// presence is not liveness when Keychain records a positive expiry already in the
// past. An expired Keychain-only seat must not enter the servable pool, and its
// status must name the concrete human repair rather than attempting to mutate the
// credential or log in automatically.
func TestExpiredKeychainOAuthNeedsLogin(t *testing.T) {
	dir := t.TempDir()
	stubKeychain(t, map[string]string{
		keychainServiceFor(dir): `{"claudeAiOauth":{"accessToken":"sk-ant-oat01-expired","refreshToken":"refresh-present","expiresAt":1}}`,
	})

	id := DeriveIdentity(dir)
	if id.HasCreds {
		t.Fatal("expired Keychain OAuth: HasCreds=true, want false")
	}
	report := (Registry{Homes: []Home{{Name: "expired", Dir: dir, Identity: id}}}).LoginReport()
	if len(report.Seats) != 1 {
		t.Fatalf("seats=%d, want 1", len(report.Seats))
	}
	seat := report.Seats[0]
	if seat.Status != LoginNeedsLogin || seat.CanServe {
		t.Fatalf("expired Keychain OAuth: status=%q can_serve=%v, want needs_login false", seat.Status, seat.CanServe)
	}
	if !strings.Contains(seat.NextAction, "/login") || !strings.Contains(seat.NextAction, "CLAUDE_CONFIG_DIR") {
		t.Fatalf("expired Keychain OAuth: next_action=%q, want concrete /login action for this CLAUDE_CONFIG_DIR", seat.NextAction)
	}

	// A missing/non-positive expiry is Claude Code's explicit non-expiring
	// convention, not evidence that the credential is stale.
	noExpiryDir := t.TempDir()
	stubKeychain(t, map[string]string{
		keychainServiceFor(noExpiryDir): `{"claudeAiOauth":{"accessToken":"sk-ant-oat01-no-expiry","expiresAt":0}}`,
	})
	noExpiryID := DeriveIdentity(noExpiryDir)
	if !noExpiryID.HasCreds {
		t.Fatal("no-expiry Keychain OAuth: HasCreds=false, want true")
	}
	noExpiryReport := (Registry{Homes: []Home{{Name: "no-expiry", Dir: noExpiryDir, Identity: noExpiryID}}}).LoginReport()
	noExpirySeat := noExpiryReport.Seats[0]
	if noExpirySeat.Status != LoginReady || !noExpirySeat.CanServe {
		t.Fatalf("no-expiry Keychain OAuth: status=%q can_serve=%v, want ready true", noExpirySeat.Status, noExpirySeat.CanServe)
	}
}

func TestClaudeKeychainAccessTokenExpiry(t *testing.T) {
	dir := t.TempDir()
	now := time.UnixMilli(1_000_000)
	service := keychainServiceFor(dir)

	set := func(body string) {
		stubKeychain(t, map[string]string{service: body})
	}

	set(`{"claudeAiOauth":{"accessToken":"tok-live","expiresAt":2000000}}`)
	if tok, ok := ClaudeKeychainAccessToken(dir, now); !ok || tok != "tok-live" {
		t.Fatalf("future expiry: got (%q,%v), want the live token", tok, ok)
	}

	set(`{"claudeAiOauth":{"accessToken":"tok-dead","expiresAt":1}}`)
	if tok, ok := ClaudeKeychainAccessToken(dir, now); ok || tok != "" {
		t.Fatalf("past expiry: got (%q,%v), want a miss — an expired bearer must never be sent", tok, ok)
	}

	set(`{"claudeAiOauth":{"accessToken":"tok-eternal"}}`)
	if tok, ok := ClaudeKeychainAccessToken(dir, now); !ok || tok != "tok-eternal" {
		t.Fatalf("no recorded expiry: got (%q,%v), want the token (non-expiring convention)", tok, ok)
	}

	set(`{"claudeAiOauth":{"accessToken":"","refreshToken":"r"}}`)
	if _, ok := ClaudeKeychainAccessToken(dir, now); ok {
		t.Fatal("refresh-only credential: HasCreds may be true but there is no access token to send")
	}
}

// TestClaudeKeychainCacheTTL pins the probe economics: hits AND misses are cached per
// service inside the TTL (one exec, not one per caller), and age out after it — the
// property TriggerRefresh's before/after comparison and the park poll depend on.
func TestClaudeKeychainCacheTTL(t *testing.T) {
	dir := t.TempDir()
	calls := stubKeychain(t, map[string]string{
		keychainServiceFor(dir): `{"claudeAiOauth":{"accessToken":"tok","expiresAt":9999999999999}}`,
	})
	clock := time.Unix(0, 0)
	prevNow := claudeKeychainCacheNow
	claudeKeychainCacheNow = func() time.Time { return clock }
	t.Cleanup(func() { claudeKeychainCacheNow = prevNow })

	for i := 0; i < 3; i++ {
		if _, ok := ClaudeKeychainCred(dir); !ok {
			t.Fatal("stubbed credential must resolve")
		}
	}
	if *calls != 1 {
		t.Fatalf("3 probes inside the TTL cost %d execs, want 1", *calls)
	}

	clock = clock.Add(claudeKeychainCacheTTL + time.Second)
	if _, ok := ClaudeKeychainCred(dir); !ok {
		t.Fatal("stubbed credential must resolve after cache expiry")
	}
	if *calls != 2 {
		t.Fatalf("aged-out cache should re-exec once, got %d total execs", *calls)
	}

	// Misses are cached too — a missing item must not storm the keychain.
	missDir := t.TempDir()
	before := *calls
	for i := 0; i < 3; i++ {
		if _, ok := ClaudeKeychainCred(missDir); ok {
			t.Fatal("unknown service must miss")
		}
	}
	if *calls != before+1 {
		t.Fatalf("3 missing-item probes cost %d execs, want 1", *calls-before)
	}
}

// TestCredExpiryKeychainFallback pins the TriggerRefresh witness path on a keychain-only
// home: no credential file, expiry answered from the keychain — under credExpiry's own
// strict positive-expiry contract.
func TestCredExpiryKeychainFallback(t *testing.T) {
	dir := t.TempDir()
	credPath := filepath.Join(dir, ".credentials.json")

	stubKeychain(t, map[string]string{
		keychainServiceFor(dir): `{"claudeAiOauth":{"accessToken":"tok","expiresAt":1234567}}`,
	})
	exp, ok := credExpiry(credPath)
	if !ok || !exp.Equal(time.UnixMilli(1234567)) {
		t.Fatalf("credExpiry = (%v,%v), want the keychain expiry instant", exp, ok)
	}

	// A non-credentials path must never consult the keychain.
	if _, ok := credExpiry(filepath.Join(dir, "unrelated.json")); ok {
		t.Fatal("non-.credentials.json path must not resolve via keychain")
	}

	// No positive expiry -> no answer, per credExpiry's contract.
	stubKeychain(t, map[string]string{
		keychainServiceFor(dir): `{"claudeAiOauth":{"accessToken":"tok"}}`,
	})
	if _, ok := credExpiry(credPath); ok {
		t.Fatal("keychain token without positive expiry must not satisfy credExpiry")
	}
}
