package gateway

// l3share.go — child D of the L3 disaggregated-cache epic (#75 / epic #504; study
// docs/notes/L3-DISAGGREGATED-CACHE-REIMAGINED.md §4 Option D + §3 G1+G4): make
// CAMA's riskiest optimization — cross-tenant prefix-sharing on an external L3 tier —
// PROVABLY SAFE on the read path.
//
// THE EDGE. An external L3 KV store (CAMA is the reference target) shares prefixes:
// two requests with the same prefix hit the same content-addressed L3 pages. That is
// its headline efficiency win and its sharpest unguarded edge — the store "does NOT
// verify content", so it cannot tell a SAFE share (a public system prompt — share
// freely) from an UNSAFE one (a prefix carrying tenant-private data that collided on a
// hash or was mis-tagged). It trusts the connector's hash blindly.
//
// THE CLOSE. fak holds the two things the store threw away at the syscall boundary:
// the page's content digest (G1 — Ref.Digest, internal/abi/types.go) and the page's
// isolation scope (G4 — ShareScope, internal/abi/types.go). So fak admits a shared
// page across a trust boundary ONLY IF both hold:
//
//   - G1 (verify, don't trust): the bytes the L3 get returned must hash to the digest
//     the page CLAIMS. A collision or mis-tag (the store handed back bytes that are not
//     the page the digest names) fails here and is REFUSED, never silently served —
//     the same content-verification step child A's sidecar runs and child B's
//     L3RegionBackend.Resolve already enforces over the tier (internal/l3region).
//   - G4 (scope gate): the reader's tenant must be permitted by the page's ShareScope.
//     A SAME-tenant read is always admitted (no regression to the efficiency win). A
//     CROSS-tenant read is admitted ONLY for a page the owner marked shareable across
//     the trust boundary (ScopeFleet — "public"/"fleet"); a page private to one agent
//     (ScopeAgent) or bound to the owner's tenant (ScopeTenant) is REFUSED.
//
// CONTROL PATH ONLY. This decides GET authorization; the verified bytes still flow
// client-direct. The digest fak verifies (hex(sha256(page))) is byte-identical to the
// content address child B's L3RegionBackend and the in-memory blob tier mint, so the
// page identity fak checks here is the same identity the tier addresses by — the gate
// rides the existing tier rather than reimplementing addressing.
//
// HONEST LIMIT (carried from the study). This guards the READ path — who may page a
// shared cell IN. It does not prove deletion (child E) or do middle-eviction (child
// B's later stages). The ShareScope and owning tag are the source of truth; a
// mis-stamped scope is a silent hole, so the producer side that stamps them is where
// the remaining trust sits.

import (
	"crypto/sha256"
	"encoding/hex"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// l3ShareBy is the forensics id this gate stamps on the decision it renders (the L3
// read-path counterpart of ifc's "ifc-scope-ceiling").
const l3ShareBy = "gateway-l3-cross-tenant-gate"

// The CLOSED refusal vocabulary this gate cites — the typed reasons the issue names.
// They are gateway control-path tokens (the same closed-string-vocabulary posture as
// the session run-state tokens), distinct from the kernel's abi.ReasonCode space so a
// gateway test that resets the abi registry (abi.ResetForTest) cannot strand them.
// Both are ALSO declared in the DOS closed refusal vocabulary (dos.toml [reasons], #57),
// so a caller can validate either with dos_check_reason and tell a content collision
// (L3_PAGE_DIGEST_MISMATCH, the G1 bite) apart from an access denial
// (L3_CROSS_TENANT_SCOPE_DENIED, the G4 bite) — distinct, closed, refusable tokens.
const (
	// L3ReasonScopeDenied is the G4 bite: a cross-tenant reader asked for a page whose
	// ShareScope does not reach across the trust boundary (private or tenant-bound).
	L3ReasonScopeDenied = "L3_CROSS_TENANT_SCOPE_DENIED"
	// L3ReasonDigestMismatch is the G1 bite: the bytes the L3 get returned do not hash
	// to the digest the page claims (a collision or a mis-tag).
	L3ReasonDigestMismatch = "L3_PAGE_DIGEST_MISMATCH"
)

// L3SharedGet is the control-path record of one cross-tenant prefix-share GET: WHO is
// asking (ReaderTag, the reader's tenant) for WHICH L3-resident page (Page, the Ref
// the tier minted — carrying the content Digest for G1 and the ShareScope for G4),
// WHO owns it (OwnerTag, the tenant that wrote the page), and the Fetched bytes the L3
// get returned. Fetched is verified out of band here; in production it flows
// client-direct (the gate touches only the control path).
type L3SharedGet struct {
	Page      abi.Ref
	OwnerTag  string
	ReaderTag string
	Fetched   []byte
}

// L3ShareVerdict is the gate's admission decision for one shared-page GET. Admitted
// reports whether the page may be paged into the reader's context; Reason is the typed
// refusal token (one of the closed vocabulary above, empty on admit); Detail is a
// bounded, payload-free witness for forensics; By names the deciding gate.
type L3ShareVerdict struct {
	Admitted bool
	Reason   string
	Detail   string
	By       string
}

// AdmitL3SharedPage runs the two checks — G1 then G4 — that make a cross-tenant
// prefix hit provably safe, before any byte is admitted to the reader's context:
//
//   - G1 first: the fetched bytes MUST hash to the page's claimed Digest. A mismatch
//     (collision / mis-tag) — or a page with no digest to verify against — is refused
//     for EVERY reader, including a same-tenant one: a corrupt page is served to no
//     one. This is the shared content-verification path (child A / child B's Resolve).
//   - G4 next: a same-tenant read is admitted. A cross-tenant read is admitted only
//     when the page's ShareScope reaches across the boundary (ScopeFleet). ScopeAgent
//     (private) and ScopeTenant (owner-tenant-bound) — and any future scope — fail
//     closed with L3_CROSS_TENANT_SCOPE_DENIED.
//
// The witness discloses only the scopes/tags and the digest verdict, never the payload.
func AdmitL3SharedPage(req L3SharedGet) L3ShareVerdict {
	// G1 — verify, don't trust. An empty claimed digest cannot be verified, so it fails
	// closed exactly like a mismatch (the store handed back an unidentifiable page).
	if req.Page.Digest == "" || l3Digest(req.Fetched) != req.Page.Digest {
		return L3ShareVerdict{
			Reason: L3ReasonDigestMismatch,
			Detail: "fetched L3 page does not hash to its claimed digest (collision or mis-tag)",
			By:     l3ShareBy,
		}
	}

	// G4 — scope gate. Same tenant shares freely (the efficiency win is untouched).
	if req.ReaderTag == req.OwnerTag {
		return L3ShareVerdict{Admitted: true, By: l3ShareBy}
	}
	// Cross a tenant boundary only for a fleet/public page.
	if !crossTenantPermits(req.Page.Scope) {
		return L3ShareVerdict{
			Reason: L3ReasonScopeDenied,
			Detail: "cross-tenant read of a " + l3ScopeLabel(req.Page.Scope) +
				"-scoped page (reader " + tagLabel(req.ReaderTag) + " ≠ owner " + tagLabel(req.OwnerTag) + ")",
			By: l3ShareBy,
		}
	}
	return L3ShareVerdict{Admitted: true, By: l3ShareBy}
}

// crossTenantPermits reports whether a page's ShareScope reaches ACROSS a tenant
// boundary. Only ScopeFleet ("public"/"fleet" — shareable across the fleet's trusted
// partition) does. ScopeAgent (private to one agent) and ScopeTenant (bound to the
// OWNER's tenant — a different tenant is outside it) do not, and any future additive
// scope fails closed. NOTE the enum is not monotone for this question (ScopeTenant's
// value exceeds ScopeFleet's), so this is an explicit allowlist, not a `>=` compare.
func crossTenantPermits(s abi.ShareScope) bool {
	return s == abi.ScopeFleet
}

// l3Digest is the content address of a byte span — hex(sha256), byte-identical to the
// digest internal/l3region and the blob tier mint, so the identity G1 verifies here is
// the same identity the L3 tier addresses pages by.
func l3Digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// l3ScopeLabel renders a ShareScope as its lowercase token for the bounded witness.
func l3ScopeLabel(s abi.ShareScope) string {
	switch s {
	case abi.ScopeFleet:
		return "fleet"
	case abi.ScopeTenant:
		return "tenant"
	default:
		return "agent"
	}
}

// tagLabel renders a tenant tag for the witness, naming the empty (single-tenant /
// unscoped) tag explicitly rather than emitting a blank.
func tagLabel(tag string) string {
	if tag == "" {
		return "<unscoped>"
	}
	return tag
}
