package accounts

import (
	"os"
	"regexp"
	"strings"
)

// apikey.go teaches the config-home registry a SECOND credential KIND beside the
// historical subscription-OAuth seat: an Anthropic API-key seat (the initial spine for
// corporate / API-based fak usage, #5331). The two kinds differ only in WHERE the seat's
// identity comes from and WHAT credential launch fronts the guard with:
//
//   - CredKindOAuth (the default, the zero value) is the existing seat: its credential
//     lives on disk (.credentials.json / .oauth-token) and its identity is DISK-DERIVED by
//     probing the OAuth profile that login authenticates as (DeriveIdentity).
//   - CredKindAPIKey is an API-key seat: its credential is an Anthropic API key held in an
//     ENVIRONMENT VARIABLE, and the registry stores ONLY the env var's NAME — never the
//     secret — exactly the reference posture internal/modelroute's CredEnv keeps. Its
//     identity is derived from the KEY (its org/workspace) rather than an OAuth profile.
//
// The design is strictly ADDITIVE: an OAuth seat carries an empty CredKind/APIKeyEnv and is
// untouched by everything here, so NameLie, identity_metadata_stale, AccountKey, rotation,
// drift, and reconcile keep their exact prior behavior.

// CredKind is the closed set of credential kinds a config-home seat can carry. The zero
// value ("") reads as CredKindOAuth so every pre-#5331 registry — none of which carry a
// cred_kind field — decodes as the subscription-OAuth seat it always was.
type CredKind string

const (
	// CredKindOAuth is the historical subscription-OAuth seat: a disk credential
	// (.credentials.json / .oauth-token) whose identity is the OAuth profile it logs into.
	// It is also the meaning of the empty/absent value (see Home.CredentialKind).
	CredKindOAuth CredKind = "oauth"
	// CredKindAPIKey is an Anthropic API-key seat: its credential is an API key held in the
	// env var named by Home.APIKeyEnv (the registry stores the NAME, never the secret), and
	// its identity is derived from the key's org/workspace rather than an OAuth profile.
	CredKindAPIKey CredKind = "api_key"
)

// apiKeyEnvNameRE is the shape a credential ENV-VAR REFERENCE must match: a valid POSIX
// environment variable name. It deliberately mirrors internal/modelroute's unexported
// envNameRE so a pasted SECRET — which carries `-`, `.`, or is far too long to be a var
// name — fails validation instead of being persisted as if it were a reference.
var apiKeyEnvNameRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// ValidAPIKeyEnvName reports whether s is a syntactically valid environment-variable NAME
// (e.g. "ANTHROPIC_API_KEY"). It is the guard that keeps a seat's stored api_key_env a
// reference and refuses a pasted key: `sk-ant-...` has a `-` and so never matches. The
// name is trimmed first so surrounding whitespace does not defeat the check.
func ValidAPIKeyEnvName(s string) bool {
	return apiKeyEnvNameRE.MatchString(strings.TrimSpace(s))
}

// APIKeyBucketKey builds the account-dedup bucket for an API-key seat that has not (yet)
// been probed for its org/workspace: the "apikey:"-prefixed env-var reference. It is the
// offline-safe, stable fallback AccountKey elects for a CredKindAPIKey seat when no
// org/workspace UUID is known, so two seats pointing at the same key env collapse onto one
// bucket. TODO(#5331): once a live Console/profile probe of the key resolves the org UUID,
// prefer UUIDBucketKey(orgUUID) so seats on the same org collapse regardless of env name.
func APIKeyBucketKey(env string) string {
	return "apikey:" + strings.TrimSpace(env)
}

// CredentialKind returns this seat's credential KIND with the zero value normalized: an
// empty CredKind reads as CredKindOAuth, so an old registry (no cred_kind field) and a seat
// explicitly marked "oauth" are indistinguishable — both the historical subscription seat.
func (h Home) CredentialKind() CredKind {
	if h.CredKind == "" || h.CredKind == CredKindOAuth {
		return CredKindOAuth
	}
	return h.CredKind
}

// DerivedIdentity re-derives this seat's Identity by its credential KIND. An API-key seat
// (CredKindAPIKey) derives from its env-var REFERENCE via DeriveAPIKeyIdentity — never the
// disk OAuth probe, which would mis-read it as a credential-less home; every other seat is
// the historical disk probe (DeriveIdentity). Refresh and MergeDiscovered call this instead
// of DeriveIdentity directly so an API-key seat survives a rescan with a truthful identity.
func (h Home) DerivedIdentity() Identity {
	if h.CredentialKind() == CredKindAPIKey {
		return DeriveAPIKeyIdentity(h.Dir, h.APIKeyEnv, os.LookupEnv)
	}
	return DeriveIdentity(h.Dir)
}

// DeriveAPIKeyIdentity derives an API-key seat's identity from its env-var REFERENCE (never
// the secret) plus, secondarily, its config dir. The env var NAME is recorded as the stable
// credential reference AccountKey buckets on; HasCreds reflects whether the key is actually
// PRESENT in the process env (set and non-empty), so LoginStatus reports "key present" vs
// "key missing" rather than OAuth login readiness. Exists is true once the seat carries a
// reference — an API-key seat's existence is its key, not a directory — so a dir-less seat
// still reads as present rather than missing_dir.
//
// It is OFFLINE and fail-open by construction: it derives NO org email/UUID, because that
// needs a live Console/profile probe of the key which fak cannot make here. lookup is a
// seam for tests (os.LookupEnv in production). TODO(#5331): wire the live org/workspace
// probe and, when it resolves, set AccountUUID so AccountKey prefers the org bucket.
func DeriveAPIKeyIdentity(dir, env string, lookup func(string) (string, bool)) Identity {
	id, _ := statConfigHome(dir) // Exists from the dir, if any; overridden below when a ref is present.
	env = strings.TrimSpace(env)
	if env == "" {
		return id
	}
	if lookup == nil {
		lookup = os.LookupEnv
	}
	id.Exists = true
	id.APIKeyEnv = env
	if v, ok := lookup(env); ok && strings.TrimSpace(v) != "" {
		id.HasCreds = true
	}
	return id
}
