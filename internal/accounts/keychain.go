package accounts

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

// keychain.go — the macOS login-Keychain credential source (#5363). Claude Code 2.x on
// darwin stores the live subscription OAuth credential as a generic password in the
// login Keychain, NOT in <config>/.credentials.json (the plaintext file is only its
// fallback store when the Keychain is unavailable). Every fak probe was disk-only, so a
// healthy Mac with a live `claude` login read as needs_login and `fak guard` refused to
// launch. This file is the ONE keychain reader every disk probe falls back to:
// hasClaudeCredentials (login posture), the guard OAuth loader, credExpiry
// (TriggerRefresh), and cmd/fak's credExpiresAt / readLiveAccessToken.
//
// Platform split: all naming/parsing/caching logic lives here, platform-neutral and
// unit-tested everywhere; ONLY the `security find-generic-password` exec lives in
// keychain_darwin.go behind the claudeKeychainReadPassword seam. On every other GOOS the
// seam stays nil and every probe here misses immediately — byte-for-byte the historical
// disk-only behavior.

// claudeKeychainReadPassword is the darwin exec seam: read the generic-password value for
// (service, account) from the login Keychain. nil (every non-darwin build, and any test
// that has not stubbed it) means "no keychain on this platform" — probes miss without
// side effects. keychain_darwin.go's init wires the real `security` exec; tests stub it.
var claudeKeychainReadPassword func(service, account string) ([]byte, error)

// ClaudeKeychainSupported reports whether this build can consult a keychain at all —
// i.e. the darwin exec seam is wired. Callers use it to keep diagnostics honest (the
// guard loader's "looked in:" list names the keychain only where one exists).
func ClaudeKeychainSupported() bool { return claudeKeychainReadPassword != nil }

// Claude Code keeps TWO generic-password items fak cares about, both named by its
// minified B8(): `Claude Code` + OAUTH_FILE_SUFFIX (empty on the stable build) + a
// per-item suffix + the config-dir suffix below. The `-credentials` item holds the
// subscription-OAuth JSON (the keychain analogue of .credentials.json); the bare
// `Claude Code` item holds the RAW saved API key a user on API billing entered at
// onboarding — the credential an API-account Mac actually authenticates with. The
// account attribute mirrors its J5(): $USER when it matches the conservative charset,
// else the literal fallback.
const (
	claudeKeychainServiceBase    = "Claude Code-credentials"
	claudeKeychainAPIKeyBase     = "Claude Code"
	claudeKeychainAccountDefault = "claude-code-user"
)

var claudeKeychainAccountRe = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

// claudeKeychainAccount resolves the Keychain account attribute the way Claude Code
// does: $USER, sanitized to a conservative charset, with a stable literal fallback so a
// hostile/odd username can never inject into the `security` argv.
func claudeKeychainAccount() string {
	u := strings.TrimSpace(os.Getenv("USER"))
	if u == "" || !claudeKeychainAccountRe.MatchString(u) {
		return claudeKeychainAccountDefault
	}
	return u
}

// claudeKeychainServices returns the candidate service names for one item base and a
// config home, most likely first. Claude Code keys the item by HOW the login was
// launched: a bare `claude` (no CLAUDE_CONFIG_DIR) writes the unsuffixed base service; a
// CLAUDE_CONFIG_DIR launch appends "-" + the first 8 hex chars of sha256 over the dir
// string (NFC-normalized — identical to the raw string for the ASCII paths fak handles;
// a non-ASCII dir whose NFC form differs simply misses, which degrades to the historical
// no-creds answer).
//
// fak probes a DIRECTORY, not a launch, so the mapping is best-effort: the default home
// (~/.claude) tries the unsuffixed service first, then its own hash (covering an
// explicit CLAUDE_CONFIG_DIR=~/.claude login); any other dir tries only its hash. A
// probe of a service that does not exist returns instantly with no user prompt, so the
// two-candidate default-home probe costs one extra silent miss at worst.
func claudeKeychainServices(base, dir string) []string {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return nil
	}
	hashed := base + "-" + claudeKeychainDirSuffix(dir)
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		if filepath.Clean(dir) == filepath.Join(home, ".claude") {
			return []string{base, hashed}
		}
	}
	return []string{hashed}
}

func claudeKeychainDirSuffix(dir string) string {
	sum := sha256.Sum256([]byte(dir))
	return hex.EncodeToString(sum[:])[:8]
}

// KeychainCred is the credential-safe slice of the Keychain item fak reads: the same
// claudeAiOauth shape .credentials.json carries. ExpiresAt is epoch milliseconds; <=0
// means no expiry was recorded (Claude Code's convention for a token that does not
// rotate). RefreshToken's VALUE never leaves this package — only its presence is folded
// into HasToken, so login-posture callers can see "a login exists" without holding a
// second secret.
type KeychainCred struct {
	AccessToken string
	ExpiresAt   int64
	hasRefresh  bool
}

// HasToken reports whether the credential carries any usable token material (an access
// token now, or a refresh token a `claude` run can rotate into one) — the keychain
// analogue of hasClaudeCredentials' file test.
func (c KeychainCred) HasToken() bool {
	return strings.TrimSpace(c.AccessToken) != "" || c.hasRefresh
}

// parseClaudeKeychainCred decodes the Keychain item's JSON body. ok is false when the
// body does not parse or carries no claudeAiOauth object with token material — the same
// "placeholder is not a login" rule hasClaudeCredentials applies to the plaintext file.
func parseClaudeKeychainCred(b []byte) (KeychainCred, bool) {
	var doc struct {
		ClaudeAIOauth struct {
			AccessToken  string `json:"accessToken"`
			RefreshToken string `json:"refreshToken"`
			ExpiresAt    int64  `json:"expiresAt"`
		} `json:"claudeAiOauth"`
	}
	if json.Unmarshal(b, &doc) != nil {
		return KeychainCred{}, false
	}
	cred := KeychainCred{
		AccessToken: strings.TrimSpace(doc.ClaudeAIOauth.AccessToken),
		ExpiresAt:   doc.ClaudeAIOauth.ExpiresAt,
		hasRefresh:  strings.TrimSpace(doc.ClaudeAIOauth.RefreshToken) != "",
	}
	if !cred.HasToken() {
		return KeychainCred{}, false
	}
	return cred, true
}

// claudeKeychainCacheTTL bounds how long one probe answer (hit OR miss) is reused. Short
// on purpose: TriggerRefresh compares expiry before/after a `claude -p` spawn and the
// park/rehydrate polls watch for a rotation landing, so a stale hit must age out well
// inside their observation windows; 5s also caps the `security` exec rate (and, when the
// operator clicked "Allow" instead of "Always Allow", the prompt rate) during a long
// guarded session's per-turn token re-reads.
const claudeKeychainCacheTTL = 5 * time.Second

type claudeKeychainCacheEntry struct {
	raw []byte
	ok  bool
	at  time.Time
}

var (
	claudeKeychainCacheMu sync.Mutex
	claudeKeychainCache   = map[string]claudeKeychainCacheEntry{}
	// claudeKeychainCacheNow is the cache clock seam so tests can age entries without
	// sleeping wall-clock time.
	claudeKeychainCacheNow = time.Now
)

func resetClaudeKeychainCache() {
	claudeKeychainCacheMu.Lock()
	claudeKeychainCache = map[string]claudeKeychainCacheEntry{}
	claudeKeychainCacheMu.Unlock()
}

// claudeKeychainRead probes ONE service name through the seam with the TTL cache in
// front, returning the item's raw value. Hits and misses are both cached: a miss
// re-execs `security` at most once per TTL, so registry reconciles that sweep many
// seats cannot storm the keychain. Interpretation (OAuth JSON vs raw API key) is the
// caller's — the cache is shape-blind.
func claudeKeychainRead(service string) ([]byte, bool) {
	read := claudeKeychainReadPassword
	if read == nil {
		return nil, false
	}
	now := claudeKeychainCacheNow()
	claudeKeychainCacheMu.Lock()
	if e, hit := claudeKeychainCache[service]; hit && now.Sub(e.at) < claudeKeychainCacheTTL {
		claudeKeychainCacheMu.Unlock()
		return e.raw, e.ok
	}
	claudeKeychainCacheMu.Unlock()

	b, err := read(service, claudeKeychainAccount())
	ok := err == nil
	if !ok {
		b = nil
	}
	claudeKeychainCacheMu.Lock()
	claudeKeychainCache[service] = claudeKeychainCacheEntry{raw: b, ok: ok, at: now}
	claudeKeychainCacheMu.Unlock()
	return b, ok
}

// ClaudeKeychainCred reads the Keychain OAuth credential for a config home: the
// candidate services in order, first parseable hit wins. ok=false means no keychain on
// this platform, no item, or a placeholder item with no token material —
// indistinguishable on purpose (every caller treats them all as "this fallback has
// nothing").
func ClaudeKeychainCred(dir string) (KeychainCred, bool) {
	for _, service := range claudeKeychainServices(claudeKeychainServiceBase, dir) {
		if b, ok := claudeKeychainRead(service); ok {
			if cred, ok := parseClaudeKeychainCred(b); ok {
				return cred, true
			}
		}
	}
	return KeychainCred{}, false
}

// ClaudeKeychainAPIKey reads the RAW Anthropic API key Claude Code saved at onboarding
// (the bare `Claude Code` item) for a config home. This is the credential a Mac on API
// billing actually authenticates with — there is no subscription OAuth to find on such
// a box, so `fak guard` adopts this key upstream instead (#5363). The value is accepted
// only when it is shaped like a key (one line, no whitespace, not a JSON blob): the
// same item name carried other payloads across Claude Code versions, and adopting a
// mis-shaped value would turn a clean needs-login diagnosis into an opaque upstream
// 401.
func ClaudeKeychainAPIKey(dir string) (string, bool) {
	for _, service := range claudeKeychainServices(claudeKeychainAPIKeyBase, dir) {
		b, ok := claudeKeychainRead(service)
		if !ok {
			continue
		}
		key := strings.TrimSpace(string(b))
		if key == "" || strings.HasPrefix(key, "{") || strings.ContainsAny(key, " \t\n\r") {
			continue
		}
		return key, true
	}
	return "", false
}

// ClaudeKeychainHasCreds is the login-posture answer: does the Keychain hold a live
// credential for this config home? It is hasClaudeCredentials' darwin fallback, so a
// keychain-only Mac seat reads ready instead of needs_login. A positive recorded expiry
// must still be in the future; ExpiresAt<=0 keeps Claude Code's non-expiring convention.
func ClaudeKeychainHasCreds(dir string) bool {
	cred, ok := ClaudeKeychainCred(dir)
	if !ok {
		return false
	}
	return cred.ExpiresAt <= 0 || time.UnixMilli(cred.ExpiresAt).After(time.Now())
}

// ClaudeKeychainAccessToken returns the Keychain access token when it is safe to SEND
// right now: present and not past a recorded expiry at now. A recorded-and-past expiry
// misses (an expired bearer 401s upstream — same drop rule as the guard's
// .credentials.json source); ExpiresAt<=0 is treated as non-expiring, matching Claude
// Code's own convention for tokens that do not rotate.
func ClaudeKeychainAccessToken(dir string, now time.Time) (string, bool) {
	cred, ok := ClaudeKeychainCred(dir)
	if !ok || cred.AccessToken == "" {
		return "", false
	}
	if cred.ExpiresAt > 0 && cred.ExpiresAt < now.UnixMilli() {
		return "", false
	}
	return cred.AccessToken, true
}
