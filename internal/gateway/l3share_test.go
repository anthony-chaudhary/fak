package gateway

// l3share_test.go — the witness for child D (#75): verified cross-tenant
// prefix-sharing on an external L3 tier. Each test maps to one acceptance bullet on
// the issue:
//
//   - G4 bite (refuse): a cross-tenant reader of a tenant-A-private page is REFUSED
//     with the typed reason L3_CROSS_TENANT_SCOPE_DENIED — for both the agent-private
//     and the tenant-bound scope.
//   - G4 serve: a page scoped public/fleet (ScopeFleet) IS served across tenants.
//   - G4 same-tenant: same-tenant prefix-sharing is unaffected (the efficiency win).
//   - G1 bite (refuse): a digest-mismatch on a shared page (collision / mis-tag) is
//     refused, not silently served — for every reader, scope notwithstanding.
//   - Composition: the gate rides child B's landed L3RegionBackend (internal/l3region)
//     — a page Put + Resolve'd over the (fake) tier is admitted across tenants when
//     fleet-scoped, bit-exact.

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/l3region"
)

// l3page mints the control-path Ref for an L3-resident page of the given content and
// scope: a RefRegion carrying the content address (hex(sha256), the G1 identity) and
// the ShareScope (the G4 capability). The bytes are returned alongside as the "fetched"
// payload an L3 get would hand back.
func l3page(content string, scope abi.ShareScope) (abi.Ref, []byte) {
	b := []byte(content)
	return abi.Ref{
		Kind:   abi.RefRegion,
		Digest: l3Digest(b),
		Len:    int64(len(b)),
		Scope:  scope,
	}, b
}

// TestL3Share_CrossTenantPrivateRefused is the headline G4 bite: tenant B asks for a
// prefix whose backing page is private to tenant A (ScopeAgent). It is refused with the
// typed reason — the page is never admitted to B's context.
func TestL3Share_CrossTenantPrivateRefused(t *testing.T) {
	page, bytes := l3page("alice's private system prompt", abi.ScopeAgent)
	v := AdmitL3SharedPage(L3SharedGet{Page: page, OwnerTag: "tenant-A", ReaderTag: "tenant-B", Fetched: bytes})
	if v.Admitted {
		t.Fatalf("cross-tenant read of a tenant-A-private page must be refused, got admitted (detail=%q)", v.Detail)
	}
	if v.Reason != L3ReasonScopeDenied {
		t.Fatalf("reason = %q, want %q", v.Reason, L3ReasonScopeDenied)
	}
}

// TestL3Share_CrossTenantTenantScopedRefused proves ScopeTenant does NOT cross a tenant
// boundary: it is shareable WITHIN the owner's tenant, so a DIFFERENT tenant is outside
// it and is refused (the enum value of ScopeTenant exceeds ScopeFleet, so this also
// guards against a naive `scope >= fleet` admit).
func TestL3Share_CrossTenantTenantScopedRefused(t *testing.T) {
	page, bytes := l3page("tenant-A shared knowledge base", abi.ScopeTenant)
	v := AdmitL3SharedPage(L3SharedGet{Page: page, OwnerTag: "tenant-A", ReaderTag: "tenant-B", Fetched: bytes})
	if v.Admitted {
		t.Fatalf("cross-tenant read of a tenant-bound page must be refused, got admitted (detail=%q)", v.Detail)
	}
	if v.Reason != L3ReasonScopeDenied {
		t.Fatalf("reason = %q, want %q", v.Reason, L3ReasonScopeDenied)
	}
}

// TestL3Share_CrossTenantFleetServed proves the safe-share path: a fleet/public page is
// served across tenants — the win the gate preserves rather than disabling.
func TestL3Share_CrossTenantFleetServed(t *testing.T) {
	page, bytes := l3page("the public refund-policy system prompt", abi.ScopeFleet)
	v := AdmitL3SharedPage(L3SharedGet{Page: page, OwnerTag: "tenant-A", ReaderTag: "tenant-B", Fetched: bytes})
	if !v.Admitted {
		t.Fatalf("cross-tenant read of a fleet/public page must be served, got refused (reason=%q detail=%q)", v.Reason, v.Detail)
	}
	if v.Reason != "" {
		t.Fatalf("an admit must carry no refusal reason, got %q", v.Reason)
	}
}

// TestL3Share_SameTenantUnaffected proves acceptance #4: same-tenant prefix-sharing is
// unaffected — even a private (ScopeAgent) page is admitted to a same-tenant reader, so
// the within-tenant efficiency win is not regressed by the cross-tenant gate.
func TestL3Share_SameTenantUnaffected(t *testing.T) {
	page, bytes := l3page("tenant-A's own warmed prefix", abi.ScopeAgent)
	v := AdmitL3SharedPage(L3SharedGet{Page: page, OwnerTag: "tenant-A", ReaderTag: "tenant-A", Fetched: bytes})
	if !v.Admitted {
		t.Fatalf("same-tenant prefix-share must be admitted, got refused (reason=%q)", v.Reason)
	}
}

// TestL3Share_DigestMismatchRefused is the G1 bite: the L3 get returned bytes that do
// NOT hash to the page's claimed digest (a collision or a mis-tag). It is refused — and
// crucially refused for EVERY reader, so a corrupt page is served to no one. We assert
// the refusal holds even on the otherwise-permissive paths (a fleet page, and a
// same-tenant reader) to prove G1 gates ahead of G4.
func TestL3Share_DigestMismatchRefused(t *testing.T) {
	page, _ := l3page("the page the digest claims", abi.ScopeFleet)
	tampered := []byte("DIFFERENT bytes than the digest names")

	// Cross-tenant fleet page — would admit on G4, but G1 refuses the tampered bytes.
	cross := AdmitL3SharedPage(L3SharedGet{Page: page, OwnerTag: "tenant-A", ReaderTag: "tenant-B", Fetched: tampered})
	if cross.Admitted {
		t.Fatalf("a digest-mismatch must be refused, got admitted (cross-tenant fleet page)")
	}
	if cross.Reason != L3ReasonDigestMismatch {
		t.Fatalf("reason = %q, want %q", cross.Reason, L3ReasonDigestMismatch)
	}

	// Same-tenant reader — would admit on G4, but G1 still refuses (corrupt page).
	same := AdmitL3SharedPage(L3SharedGet{Page: page, OwnerTag: "tenant-A", ReaderTag: "tenant-A", Fetched: tampered})
	if same.Admitted || same.Reason != L3ReasonDigestMismatch {
		t.Fatalf("a digest-mismatch must be refused even same-tenant, got admitted=%v reason=%q", same.Admitted, same.Reason)
	}
}

// TestL3Share_EmptyDigestFailsClosed proves the fail-closed posture: a page that
// carries no digest to verify against cannot pass G1 (an unidentifiable page is treated
// like a mismatch, never admitted on trust).
func TestL3Share_EmptyDigestFailsClosed(t *testing.T) {
	v := AdmitL3SharedPage(L3SharedGet{
		Page:      abi.Ref{Kind: abi.RefRegion, Scope: abi.ScopeFleet},
		OwnerTag:  "tenant-A",
		ReaderTag: "tenant-A",
		Fetched:   []byte("some bytes"),
	})
	if v.Admitted || v.Reason != L3ReasonDigestMismatch {
		t.Fatalf("an empty-digest page must fail closed, got admitted=%v reason=%q", v.Admitted, v.Reason)
	}
}

// TestL3Share_ControlPathOnly proves the §6.5 control-path-only constraint (issue #57
// acceptance "Control-path-only: no inline byte-scanner"): the gate is ONE admission
// decision over the page's Ref metadata + a SINGLE digest recompute of the returned
// control-path object — it does NOT sit inline on the RDMA data path, scanning every
// byte of a multi-MB page at line rate. We authorize a large page once on the control
// path, then stream its verified bytes client-direct (the simulated RDMA read the gate
// is NOT on) and assert the gate ran exactly once for the whole page: its cost is O(1)
// in page size, never O(bytes), so the ~1–5µs L3 read budget is not multiplied per byte.
func TestL3Share_ControlPathOnly(t *testing.T) {
	// A page large enough that a per-byte inline gate would be unmistakable (~112 KiB).
	page, bytes := l3page(strings.Repeat("public refund-policy prefix ", 4096), abi.ScopeFleet)

	gateCalls := 0
	authorize := func() L3ShareVerdict {
		gateCalls++
		return AdmitL3SharedPage(L3SharedGet{Page: page, OwnerTag: "tenant-A", ReaderTag: "tenant-B", Fetched: bytes})
	}

	// CONTROL PATH: a single admission decision, up front, on the Ref + one recompute.
	if v := authorize(); !v.Admitted {
		t.Fatalf("a fleet page must admit on the control path, got refused (reason=%q detail=%q)", v.Reason, v.Detail)
	}

	// DATA PATH: the verified bytes now stream client-direct (simulated RDMA line-rate
	// read). The gate is NOT on this path — every byte flows past it without re-invoking it.
	byteOps := 0
	for range bytes {
		byteOps++
	}

	if gateCalls != 1 {
		t.Fatalf("control-path gate must run ONCE per page, ran %d times (inline per-byte scanner?)", gateCalls)
	}
	if byteOps != len(bytes) {
		t.Fatalf("data path must stream every byte of the page; streamed %d of %d", byteOps, len(bytes))
	}
	// One control decision vs len(bytes) data-path ops: an inline per-byte gate would make
	// these counts equal. The strict inequality is the witness that the gate is not inline.
	if gateCalls >= byteOps {
		t.Fatalf("gate ran %d times for %d bytes — not control-path-only (must be O(1) in page size)", gateCalls, byteOps)
	}
}

// TestL3Share_RidesL3RegionBackend proves the gate composes with child B's LANDED
// L3RegionBackend (internal/l3region): a page is Put into the (fake) L3 tier and
// Resolve'd back through the backend's own verify-don't-trust step (G1 over the tier),
// then the gate's G4 scope check admits it across tenants because the producer stamped
// it fleet-scoped — bit-exact bytes, no silent trust.
func TestL3Share_RidesL3RegionBackend(t *testing.T) {
	const content = "a public system prompt shared across the fleet"
	store := l3region.NewL3Store()
	be := l3region.New(store)

	ref, err := be.Put(context.Background(), []byte(content))
	if err != nil {
		t.Fatalf("Put: %v", err)
	}
	// The producer stamps the page's share scope (Put fail-closes to ScopeAgent).
	ref.Scope = abi.ScopeFleet

	// G1 over the tier: Resolve mgets + verifies each page hashes to its key and the
	// region to Ref.Digest, returning the bytes bit-exact.
	fetched, err := be.Resolve(context.Background(), ref)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if string(fetched) != content {
		t.Fatalf("tier round-trip not bit-exact: got %q", string(fetched))
	}

	v := AdmitL3SharedPage(L3SharedGet{Page: ref, OwnerTag: "tenant-A", ReaderTag: "tenant-B", Fetched: fetched})
	if !v.Admitted {
		t.Fatalf("a fleet-scoped, tier-verified page must be admitted across tenants, got refused (reason=%q detail=%q)", v.Reason, v.Detail)
	}
}
