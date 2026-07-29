package gateway

// keyset.go — org-scoped, multi-key gateway auth (#5332). The single RequireKey
// bearer authenticates ONE anonymous caller; a keyset authenticates MANY api keys,
// each bound to an org/project PRINCIPAL, so audit attributes every served turn to
// the tenant whose key presented it — the principal rides the access log and
// /v1/fak/events, both joinable by X-Trace-Id.
//
// Keys are never held or written in the clear: newKeyset hashes each key to a
// SHA-256 digest at construction and drops the raw bytes, and lookup compares an
// inbound credential's digest against the set in constant time. A miss and a
// wrong-key present are indistinguishable to the caller (a uniform 401 at the
// withAuth call site), so the set never reveals which key — or how many — it holds,
// nor leaks a key's bytes or length through reject latency. This is the SAME auth
// primitive as RequireKey (constant-time SHA-256 compare), generalized from one
// secret to a set.
//
// The keyset principal is the tenant ISOLATION principal — an org/project string
// identity, the same axis as principalFor / X-Fak-Principal / traceOwner — and is
// deliberately distinct from the AUTHORITY principal (human / peer-agent / …, see
// principal.go): a key names WHO the tenant is, not whether a turn may consume user
// consent.
//
// The residency arm closes the loop: the authenticated principal rides the request
// (withAuth stamps it via WithPrincipal → principalFromContext, and buildCall lowers it
// onto ToolCall.Meta[vdso.MetaPrincipal]) down to the routing seam, where resolveRoute
// adjudicates it against the resolving account's modelroute.Account.Principals
// allowlist before binding Engine — so a keyset-bound org reaches only the
// RouteAccounts targets its tenancy admits and a route it is not provisioned for fails
// CLOSED, before Submit rather than after. An account naming no principals is
// unrestricted, so a pre-#5332 roster is unchanged.

import (
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"fmt"
	"sort"
	"strings"
)

// ParseKeyPrincipals resolves repeated `PRINCIPAL=ENV_VAR` specs (the operator-facing
// `fak serve --key-principal` surface, #5332) into a Config.KeyPrincipals map, reading
// each tenant's api key from the ENVIRONMENT at boot via lookupEnv (os.Getenv in the
// live path). A spec names an env var, never a secret — so a systemd unit, a launchd
// plist, or a shell history carries only the variable NAME and the raw key never lands
// in a file at rest, the same contract --require-key-env already holds. The resolved
// keys live only until New hands them to newKeyset, which hashes and drops them.
//
// Every failure mode is fail-CLOSED — an error the caller must refuse to boot on, never
// a silently dropped binding:
//
//   - a malformed spec (no `=`, empty principal, or empty env name) is a typo, and a
//     typo that silently armed a SMALLER keyset would leave a tenant unauthenticated
//     while the operator believes their key works;
//   - an unset/empty env var is the same refusal --require-key-env makes, for the same
//     reason: a keyset that quietly forgot a tenant's key looks armed and is not;
//   - two specs naming the SAME env var, or two env vars holding a byte-identical key,
//     would collapse in the key→principal map and authenticate one org AS another —
//     the exact tenant-isolation break this issue exists to prevent.
//
// Binding one principal to SEVERAL env vars is explicitly allowed: that is key rotation
// (the old and new key both authenticate as the same tenant during the overlap).
//
// An empty spec list is the only "no keyset" answer and returns a nil map, so an
// operator who passes no --key-principal keeps the RequireKey-only path byte-for-byte.
func ParseKeyPrincipals(specs []string, lookupEnv func(string) string) (map[string]string, error) {
	if len(specs) == 0 {
		return nil, nil
	}
	if lookupEnv == nil {
		return nil, fmt.Errorf("no environment reader")
	}
	out := make(map[string]string, len(specs))
	byEnv := make(map[string]string, len(specs))
	for _, spec := range specs {
		principal, envName, hasSep := strings.Cut(spec, "=")
		principal = strings.TrimSpace(principal)
		envName = strings.TrimSpace(envName)
		if !hasSep || principal == "" || envName == "" {
			return nil, fmt.Errorf("%q: want PRINCIPAL=ENV_VAR, naming the env var that HOLDS the tenant's api key (never the key itself)", spec)
		}
		if prev, dup := byEnv[envName]; dup {
			return nil, fmt.Errorf("%q: env var %s is already bound to principal %q — one api key cannot authenticate as two tenants", spec, envName, prev)
		}
		byEnv[envName] = principal
		// Trim exactly as newKeyset does, so the value compared for collisions here is
		// the value that will actually be hashed into the set.
		key := strings.TrimSpace(lookupEnv(envName))
		if key == "" {
			return nil, fmt.Errorf("%q: env var %s is unset or empty — refusing to start with principal %q unauthenticated", spec, envName, principal)
		}
		if prev, dup := out[key]; dup && prev != principal {
			return nil, fmt.Errorf("%q: the key in %s is byte-identical to the key bound to principal %q — one api key cannot authenticate as two tenants", spec, envName, prev)
		}
		out[key] = principal
	}
	return out, nil
}

// keysetEntry binds one api-key DIGEST (never the raw key) to the org/project
// principal that key authenticates as.
type keysetEntry struct {
	digest    [sha256.Size]byte
	principal string
}

// keyset is an immutable set of api-key→principal bindings, matched by constant-time
// digest comparison. A nil *keyset authenticates nothing, so the RequireKey-only path
// stays byte-for-byte unchanged.
type keyset struct {
	entries []keysetEntry
}

// newKeyset builds a keyset from a key→principal map, hashing each key to a digest and
// discarding the raw key. An empty key OR empty principal is skipped (a half-specified
// binding must never silently authenticate as the empty single-tenant principal).
// Returns nil when no usable binding remains, so an unconfigured keyset leaves the
// single-key path exactly as it was.
func newKeyset(keyPrincipals map[string]string) *keyset {
	if len(keyPrincipals) == 0 {
		return nil
	}
	ks := &keyset{}
	for key, principal := range keyPrincipals {
		key = strings.TrimSpace(key)
		principal = strings.TrimSpace(principal)
		if key == "" || principal == "" {
			continue
		}
		ks.entries = append(ks.entries, keysetEntry{
			digest:    sha256.Sum256([]byte(key)),
			principal: principal,
		})
	}
	if len(ks.entries) == 0 {
		return nil
	}
	// Deterministic order (principal, then digest) so the scan order and any diagnostic
	// are stable regardless of Go map iteration — it never depends on the raw keys.
	sort.Slice(ks.entries, func(i, j int) bool {
		if ks.entries[i].principal != ks.entries[j].principal {
			return ks.entries[i].principal < ks.entries[j].principal
		}
		return bytes.Compare(ks.entries[i].digest[:], ks.entries[j].digest[:]) < 0
	})
	return ks
}

// lookup resolves the principal an inbound credential authenticates as, matched by
// constant-time digest comparison. It scans EVERY entry even after a hit so the
// accept/reject latency reveals neither the set's size nor which entry matched. An
// empty presented key never matches (a caller that sent nothing is not the empty-key
// tenant). A nil keyset never matches. matched is false when no binding's digest
// equals the credential's.
func (ks *keyset) lookup(presentedKey string) (principal string, matched bool) {
	if ks == nil || presentedKey == "" {
		return "", false
	}
	got := sha256.Sum256([]byte(presentedKey))
	for i := range ks.entries {
		if subtle.ConstantTimeCompare(got[:], ks.entries[i].digest[:]) == 1 {
			principal = ks.entries[i].principal
			matched = true
			// No break: keep the scan length independent of the match position.
		}
	}
	return principal, matched
}
