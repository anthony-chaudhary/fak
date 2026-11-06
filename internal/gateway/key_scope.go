package gateway

// key_scope.go — per-key auth scopes and exact method/route grants, adapted from
// TensorFold's Apache-2.0 key-file scope auth at revision
// 9e8c368b0b4c2c7892a0c9f0f302d7c164016209:
//   - zig/src/server/auth.zig (the digest-matched key carries a COPIED label and
//     its scopes; one key may not carry two scope sets; a refused key-file reload
//     closes every file key while in-flight requests keep their copied scopes);
//   - zig/src/server/auth_scope.zig (the closed inference/read/admin scope set and
//     the exact method+route grant matrix, never suffix grants).
//
// This file is the GRANT MODEL and the FILE arm of the port: it loads into
// keyset.go's keysetEntry, which carries the per-key scope bitset, so the file
// arm and the scoped-load arm share one entry type. Scopes are authorization
// only — they bound which routes a key may reach, exactly like the
// X-Fak-Auth-Scope guard (scope_guard.go) bounds a caller-declared claim, but
// carried BY THE KEY rather than declared by the caller, so a compromised caller
// cannot widen them. Key material is never held raw past hashing and never
// logged: refusals name lines and labels, never keys.

import (
	"crypto/sha256"
	"crypto/subtle"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"unicode"
)

// Closed load/parse refusal classes for scoped key material. Callers branch on
// these via errors.Is, never on message prose.
var (
	// ErrInvalidKeyScopeLabel is a malformed key-file label (empty, over 64
	// chars, or outside letters/digits/dot/underscore/hyphen).
	ErrInvalidKeyScopeLabel = errors.New("invalid key scope label")
	// ErrInvalidKeyScopes is an unknown, repeated or empty scope member, an
	// unterminated bracket list, or admin mixed with another scope.
	ErrInvalidKeyScopes = errors.New("invalid key scopes")
	// ErrConflictingKeyScopes is one key digest carrying two different scope
	// sets in one load: a restricted line shadowed by a full copy of the same
	// key. The whole load is refused, never the conflicting line dropped.
	ErrConflictingKeyScopes = errors.New("the same api key cannot have different scopes")
	// ErrEmptyKeyFile is a key file with no usable key line.
	ErrEmptyKeyFile = errors.New("api key file contains no keys")
)

// keyScopes is the per-key authorization bitset: the closed inference/read/admin
// set. A zero value grants nothing (fail closed); fullKeyScopes is what a key
// WITHOUT a scope list keeps.
type keyScopes uint8

const (
	keyScopeInference keyScopes = 1 << iota // mutating inference routes, POST only
	keyScopeRead                            // observation-plane reads, GET only
	keyScopeAdmin                           // every route, every verb
)

// fullKeyScopes is the grant a key without a scope list keeps: every route
// (TensorFold auth_scope.zig `pub const full: Scopes = .{ .admin = true }`).
// Admin is the FULL value, not a fourth additive privilege.
const fullKeyScopes = keyScopeAdmin

// parseKeyScopeLabel parses one key-file label: `name`, `name[inference]`,
// `name[read]`, `name[inference,read]` or `name[admin]`. A bare name keeps
// fullKeyScopes. An unknown, repeated or empty scope member, admin mixed with
// another scope, or a malformed label is refused — the caller must refuse the
// WHOLE load, never silently drop the line (an unknown scope that normalized to
// read would be a silent widening, the same failure class the X-Fak-Auth-Scope
// parser fails closed on).
func parseKeyScopeLabel(raw string) (string, keyScopes, error) {
	bracket := strings.IndexByte(raw, '[')
	if bracket < 0 {
		if !validKeyScopeLabel(raw) {
			return "", 0, fmt.Errorf("%w: %q", ErrInvalidKeyScopeLabel, raw)
		}
		return raw, fullKeyScopes, nil
	}
	name := raw[:bracket]
	if !validKeyScopeLabel(name) {
		return "", 0, fmt.Errorf("%w: %q", ErrInvalidKeyScopeLabel, raw)
	}
	if len(raw) < bracket+3 || raw[len(raw)-1] != ']' {
		return "", 0, fmt.Errorf("%w: %q", ErrInvalidKeyScopes, raw)
	}
	var scopes keyScopes
	for _, member := range strings.Split(raw[bracket+1:len(raw)-1], ",") {
		member = strings.Trim(member, " \t")
		var bit keyScopes
		switch member {
		case "inference":
			bit = keyScopeInference
		case "read":
			bit = keyScopeRead
		case "admin":
			bit = keyScopeAdmin
		default:
			return "", 0, fmt.Errorf("%w: %q", ErrInvalidKeyScopes, raw)
		}
		if scopes&bit != 0 { // repeated member
			return "", 0, fmt.Errorf("%w: %q", ErrInvalidKeyScopes, raw)
		}
		scopes |= bit
	}
	if scopes&keyScopeAdmin != 0 && scopes&(keyScopeInference|keyScopeRead) != 0 {
		return "", 0, fmt.Errorf("%w: %q", ErrInvalidKeyScopes, raw)
	}
	return name, scopes, nil
}

// validKeyScopeLabel reports whether label is 1–64 letters, digits, dots,
// underscores or hyphens.
func validKeyScopeLabel(label string) bool {
	if label == "" || len(label) > 64 {
		return false
	}
	for i := 0; i < len(label); i++ {
		c := label[i]
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' ||
			c == '_' || c == '.' || c == '-') {
			return false
		}
	}
	return true
}

// keyScopeRoutePath normalizes a raw request path the way the dispatcher reads
// it: the query is dropped and trailing slashes trimmed, so a grant is matched
// EXACTLY against the normalized route — never by prefix or suffix
// (TensorFold auth.zig routePath + auth_scope.zig allows).
func keyScopeRoutePath(raw string) string {
	if i := strings.IndexByte(raw, '?'); i >= 0 {
		raw = raw[:i]
	}
	return strings.TrimRight(raw, "/")
}

// The closed grant tables, one row per route exactly as the dispatcher reads
// it. Listing routes are readable by either an inference or a read key (a model
// roster is needed to call a model); reading routes are the observation-plane
// reads; inferring routes are the mutating inference surface, POST only. A
// route absent from every table is granted by admin alone.
var (
	keyScopeListingRoutes = []string{"/v1/models"}
	keyScopeReadingRoutes = []string{
		"/v1/fak/observation",
		"/v1/fak/observation/requests",
	}
	keyScopeInferringRoutes = []string{
		"/v1/chat/completions",
		"/v1/messages",
	}
)

// allows reports whether scopes may make (method, path): the exact
// method+route matrix. Admin passes everything; GET listing needs inference or
// read; GET reading needs read; POST inferring needs inference. Every other
// method or resource is refused — a grant for /v1/messages does NOT cover
// /v1/messages/count_tokens or /v1/messagesX.
func (s keyScopes) allows(method, path string) bool {
	if s&keyScopeAdmin != 0 {
		return true
	}
	route := keyScopeRoutePath(path)
	switch method {
	case http.MethodGet:
		if s&(keyScopeInference|keyScopeRead) != 0 && keyScopeOneOf(route, keyScopeListingRoutes) {
			return true
		}
		return s&keyScopeRead != 0 && keyScopeOneOf(route, keyScopeReadingRoutes)
	case http.MethodPost:
		return s&keyScopeInference != 0 && keyScopeOneOf(route, keyScopeInferringRoutes)
	}
	return false
}

func keyScopeOneOf(route string, names []string) bool {
	for _, name := range names {
		if route == name {
			return true
		}
	}
	return false
}

// keyHasWhitespace reports whether key holds any whitespace or control rune:
// such text is not a key, and accepting it would let a padded duplicate of a
// real key ride the file under a second scope set.
func keyHasWhitespace(key string) bool {
	for _, r := range key {
		if r < 32 || r == 0x7f || unicode.IsSpace(r) {
			return true
		}
	}
	return false
}

// parseKeyFileLines parses a key file's contents into keyset entries. Line
// grammar: `label[scopes]: key` or a bare `key` (which keeps fullKeyScopes);
// blank lines and `#` comments are skipped. One key digest may repeat across
// lines ONLY with identical scopes — "one key, one set of scopes: a restricted
// line cannot be shadowed by a full copy of the same key" (auth.zig:119) —
// otherwise the WHOLE parse is refused. Identical-scope repeats dedupe to the
// first entry (rotation); conflicting ones refuse the load. Refusals wrap the
// closed sentinel classes above and name the offending line number, never the
// key bytes.
func parseKeyFileLines(text string) ([]keysetEntry, error) {
	assigned := make(map[[sha256.Size]byte]keyScopes)
	var entries []keysetEntry
	for n, rawLine := range strings.Split(text, "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		label := ""
		scopes := fullKeyScopes
		key := line
		if i := strings.IndexByte(line, ':'); i >= 0 {
			l, s, err := parseKeyScopeLabel(strings.TrimSpace(line[:i]))
			if err != nil {
				return nil, fmt.Errorf("line %d: %w", n+1, err)
			}
			label, scopes, key = l, s, line[i+1:]
		} else {
			label = fmt.Sprintf("file-%d", len(entries)+1)
		}
		key = strings.TrimSpace(key)
		if key == "" || keyHasWhitespace(key) {
			return nil, fmt.Errorf("line %d: %w", n+1, errors.New("api keys must be nonempty text without whitespace"))
		}
		d := sha256.Sum256([]byte(key))
		if prev, dup := assigned[d]; dup {
			if prev != scopes {
				return nil, fmt.Errorf("line %d: %w", n+1, ErrConflictingKeyScopes)
			}
			continue // identical-scope repeat: rotation, deduped to the first entry
		}
		assigned[d] = scopes
		entries = append(entries, keysetEntry{digest: d, principal: label, scopes: scopes})
	}
	if len(entries) == 0 {
		return nil, ErrEmptyKeyFile
	}
	return entries, nil
}

// scopedKeyFile is the file arm of per-key scope auth: the keys parsed from a
// key file, swapped atomically on reload. It mirrors the file arm of
// TensorFold's key store (auth.zig @pin):
//
//   - the zero value is CLOSED: until the first successful load no file key
//     authenticates;
//   - a successful load replaces the file's keys wholesale;
//   - a refused load (any parse failure) closes EVERY file key — fail-closed,
//     the authenticated routes stay closed until a valid load succeeds again;
//   - match returns a VALUE COPY of the entry: the request's own copy of its
//     label and scopes outlives any later reload, valid or refused, so an
//     in-flight request keeps the scopes it was accepted with.
type scopedKeyFile struct {
	mu      sync.RWMutex
	valid   bool
	entries []keysetEntry
}

// reload re-parses the key file's contents and swaps the file's keys on
// success. On ANY failure it closes every file key (fail-closed) and returns
// the refusal, wrapping the closed sentinel class.
func (f *scopedKeyFile) reload(text string) error {
	entries, err := parseKeyFileLines(text)
	f.mu.Lock()
	defer f.mu.Unlock()
	if err != nil {
		f.entries = nil
		f.valid = false
		return err
	}
	f.entries = entries
	f.valid = true
	return nil
}

// match resolves a presented key to a COPIED entry: the caller holds its own
// principal and scopes, so a later reload cannot change what this request was
// accepted with. While the file arm is closed (no successful load yet, or a
// refused reload) every file key misses. The scan compares digests in constant
// time and runs to completion even after a hit, so reject latency reveals
// neither the set's size nor which entry matched.
func (f *scopedKeyFile) match(presentedKey string) (keysetEntry, bool) {
	if presentedKey == "" {
		return keysetEntry{}, false
	}
	got := sha256.Sum256([]byte(presentedKey))
	f.mu.RLock()
	defer f.mu.RUnlock()
	if !f.valid {
		return keysetEntry{}, false
	}
	var hit keysetEntry
	matched := false
	for i := range f.entries {
		if subtle.ConstantTimeCompare(got[:], f.entries[i].digest[:]) == 1 {
			hit = f.entries[i]
			matched = true
			// No break: keep the scan length independent of the match position.
		}
	}
	return hit, matched
}
