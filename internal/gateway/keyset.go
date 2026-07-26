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
// TODO(#5332, residency arm): scope the residency PDP by this principal so a
// keyset-bound org reaches only the modelroute.RouteAccounts targets its tenancy
// admits, failing CLOSED on a route its principal is not provisioned for. The
// authenticated principal already rides the request (withAuth stamps it via
// WithPrincipal → principalFromContext) down to the routing seam, so the tie-in is a
// RouteAccounts lookup there; it is deferred so the auth + attribution core ships
// first, per the issue's own guidance.

import (
	"bytes"
	"crypto/sha256"
	"crypto/subtle"
	"sort"
	"strings"
)

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
