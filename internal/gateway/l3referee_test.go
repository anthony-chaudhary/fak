package gateway

// l3referee_test.go — the witness for child A (#58): the referee-sidecar seam in front
// of an external L3 get/set. Each test maps to one acceptance bullet on the issue, all
// against MockL3Backend with NO real external-store wire dependency:
//
//   - round-trip: a clean set then get round-trips the page BYTE-IDENTICAL (the control
//     path does not mutate the data path).
//   - G1: a get whose bytes hash to a digest other than the recorded Ref.Digest is
//     REFUSED, fail-closed, never served.
//   - G4: a reader whose Scope does not satisfy the page's ShareScope is REFUSED before
//     any bytes are returned.
//   - G5: every set stamps a provenance taint label the matching get reads back; the
//     label survives the round-trip.
//   - G8: every refusal carries a token from the closed refusal vocabulary; no
//     UNCLASSIFIED reason.
//   - §6.5: the forwarded byte slice is identity-equal on the hot read path (the sidecar
//     split, not an inline byte-scanner).
//   - #496: durability classification on set DEFERS to the injected verdict — the
//     sidecar calls it, it does not duplicate the S7 gate.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestRefereeSidecar_CleanRoundTripByteExact is acceptance #1: a fleet/public page set
// then read back across tenants is ADMITTED and the bytes are byte-identical — the
// control-path adjudication does not touch the data path.
func TestRefereeSidecar_CleanRoundTripByteExact(t *testing.T) {
	page, content := l3page("the public refund-policy system prompt", abi.ScopeFleet)
	side := NewRefereeSidecar(NewMockL3Backend(), nil)

	side.Set(page, "tenant-A", content)

	got, _, found, v := side.Get(page.Digest, "tenant-B")
	if !found {
		t.Fatalf("a page just set must be found on get")
	}
	if !v.Admitted {
		t.Fatalf("a fleet page must admit across tenants, got refused (reason=%q detail=%q)", v.Reason, v.Detail)
	}
	if string(got) != string(content) {
		t.Fatalf("round-trip not byte-exact: got %q want %q", string(got), string(content))
	}
}

// TestRefereeSidecar_DigestMismatchRefused is the G1 bite: the page is stored under the
// digest it CLAIMS but the bytes the tier holds are different (a collision or mis-tag the
// semantics-free store cannot detect). The get recomputes the digest, finds it does not
// match, and REFUSES — fail-closed, bytes never served.
func TestRefereeSidecar_DigestMismatchRefused(t *testing.T) {
	claim, _ := l3page("the page the digest claims", abi.ScopeFleet)
	tampered := []byte("DIFFERENT bytes than the digest names")
	side := NewRefereeSidecar(NewMockL3Backend(), nil)

	// Set the page under its CLAIMED digest but with the tampered bytes (a mis-tag).
	side.Set(claim, "tenant-A", tampered)

	got, _, found, v := side.Get(claim.Digest, "tenant-B")
	if !found {
		t.Fatalf("the mis-tagged page is resident, so the get must find it (then refuse it)")
	}
	if v.Admitted {
		t.Fatalf("a digest-mismatched page must be refused, got admitted")
	}
	if v.Reason != L3ReasonDigestMismatch {
		t.Fatalf("reason = %q, want %q", v.Reason, L3ReasonDigestMismatch)
	}
	if got != nil {
		t.Fatalf("a refused page must return no bytes, got %d bytes", len(got))
	}
}

// TestRefereeSidecar_CrossTenantScopeRefused is the G4 bite: tenant B reads a page
// private to tenant A (ScopeAgent). It is refused with the typed reason BEFORE any byte
// is returned.
func TestRefereeSidecar_CrossTenantScopeRefused(t *testing.T) {
	page, content := l3page("alice's private system prompt", abi.ScopeAgent)
	side := NewRefereeSidecar(NewMockL3Backend(), nil)

	side.Set(page, "tenant-A", content)

	got, _, found, v := side.Get(page.Digest, "tenant-B")
	if !found {
		t.Fatalf("the page is resident, so the get must find it (then refuse on scope)")
	}
	if v.Admitted {
		t.Fatalf("a cross-tenant read of a private page must be refused, got admitted")
	}
	if v.Reason != L3ReasonScopeDenied {
		t.Fatalf("reason = %q, want %q", v.Reason, L3ReasonScopeDenied)
	}
	if got != nil {
		t.Fatalf("a scope-refused page must return no bytes before adjudication, got %d bytes", len(got))
	}
}

// TestRefereeSidecar_SameTenantUnaffected proves the within-tenant efficiency win is not
// regressed: a same-tenant reader reads even an agent-private page, byte-exact.
func TestRefereeSidecar_SameTenantUnaffected(t *testing.T) {
	page, content := l3page("tenant-A's own warmed prefix", abi.ScopeAgent)
	side := NewRefereeSidecar(NewMockL3Backend(), nil)

	side.Set(page, "tenant-A", content)

	got, _, _, v := side.Get(page.Digest, "tenant-A")
	if !v.Admitted {
		t.Fatalf("a same-tenant read must be admitted, got refused (reason=%q)", v.Reason)
	}
	if string(got) != string(content) {
		t.Fatalf("same-tenant round-trip not byte-exact: got %q", string(got))
	}
}

// TestRefereeSidecar_ProvenanceLabelSurvives is the G5 bite: the IFC taint label the
// producer stamped on the page is recorded on set and read back on get — the provenance
// survives the L3 round-trip. We check both a trusted page (a non-default label, so a
// dropped stamp would be visible) and the fail-closed default.
func TestRefereeSidecar_ProvenanceLabelSurvives(t *testing.T) {
	side := NewRefereeSidecar(NewMockL3Backend(), nil)

	// A producer-trusted fleet page: the Trusted label must survive the round-trip.
	page, content := l3page("a vetted public prompt", abi.ScopeFleet)
	page.Taint = abi.TaintTrusted
	setMeta := side.Set(page, "tenant-A", content)
	if setMeta.Taint != abi.TaintTrusted {
		t.Fatalf("set must stamp the producer's taint label, recorded %v", setMeta.Taint)
	}
	_, getMeta, _, v := side.Get(page.Digest, "tenant-B")
	if !v.Admitted {
		t.Fatalf("the fleet page must admit; reason=%q", v.Reason)
	}
	if getMeta.Taint != abi.TaintTrusted {
		t.Fatalf("the G5 provenance label did not survive the round-trip: got %v want %v", getMeta.Taint, abi.TaintTrusted)
	}

	// A page with no explicit stamp: the label is the fail-closed default (TaintTainted),
	// and that floor must also survive rather than silently becoming Trusted.
	plain, body := l3page("an unvetted prefix", abi.ScopeFleet)
	side.Set(plain, "tenant-A", body)
	_, plainMeta, _, _ := side.Get(plain.Digest, "tenant-A")
	if plainMeta.Taint != abi.TaintTainted {
		t.Fatalf("an unstamped page must round-trip the fail-closed TaintTainted floor, got %v", plainMeta.Taint)
	}
}

// TestRefereeSidecar_EveryRefusalCarriesKnownReason is the G8 bite: every refusal the
// sidecar emits carries a token from the closed gateway L3 vocabulary — never an empty or
// UNCLASSIFIED reason. (DOS membership of both tokens is verified out of band via
// dos_check_reason; here we assert the gate only ever emits those two.)
func TestRefereeSidecar_EveryRefusalCarriesKnownReason(t *testing.T) {
	known := map[string]bool{
		L3ReasonDigestMismatch: true,
		L3ReasonScopeDenied:    true,
	}
	side := NewRefereeSidecar(NewMockL3Backend(), nil)

	// G4 refusal.
	priv, privBytes := l3page("private", abi.ScopeAgent)
	side.Set(priv, "tenant-A", privBytes)
	_, _, _, scopeV := side.Get(priv.Digest, "tenant-B")

	// G1 refusal.
	claim, _ := l3page("claimed", abi.ScopeFleet)
	side.Set(claim, "tenant-A", []byte("not the claimed bytes"))
	_, _, _, digV := side.Get(claim.Digest, "tenant-A")

	for _, v := range []L3ShareVerdict{scopeV, digV} {
		if v.Admitted {
			t.Fatalf("expected a refusal, got an admit")
		}
		if v.Reason == "" {
			t.Fatalf("a refusal must carry a typed reason, got empty (UNCLASSIFIED)")
		}
		if !known[v.Reason] {
			t.Fatalf("refusal reason %q is not in the closed gateway L3 vocabulary", v.Reason)
		}
		if v.By != l3ShareBy {
			t.Fatalf("a refusal must name the deciding gate, got By=%q", v.By)
		}
	}
}

// TestRefereeSidecar_ControlPathOnlyIdentityEqual is the §6.5 witness: on an admit the
// sidecar forwards the tier's EXACT byte slice, identity-equal — it does not defensively
// copy or re-scan the payload on the hot read path. A large page makes a per-byte copy
// unmistakable. Two gets returning the same backing array prove the read path is the
// sidecar split (control-path adjudication once, bytes passed through), not an inline
// byte-scanner that would re-materialize the page each read.
func TestRefereeSidecar_ControlPathOnlyIdentityEqual(t *testing.T) {
	big := make([]byte, 0, 112*1024)
	for i := 0; i < 4096; i++ {
		big = append(big, []byte("public refund-policy prefix ")...)
	}
	page := abi.Ref{Kind: abi.RefRegion, Digest: l3Digest(big), Len: int64(len(big)), Scope: abi.ScopeFleet}

	be := NewMockL3Backend()
	side := NewRefereeSidecar(be, nil)
	side.Set(page, "tenant-A", big)

	a, _, _, va := side.Get(page.Digest, "tenant-B")
	b, _, _, vb := side.Get(page.Digest, "tenant-B")
	if !va.Admitted || !vb.Admitted {
		t.Fatalf("the fleet page must admit on the control path; reasons=%q,%q", va.Reason, vb.Reason)
	}
	if len(a) != len(big) || len(b) != len(big) {
		t.Fatalf("the data path must forward the whole page; got %d and %d of %d", len(a), len(b), len(big))
	}
	// Identity-equal backing array across reads: a copying/re-scanning sidecar would
	// allocate a fresh slice per get, so &a[0] != &b[0]. The pointer equality is the
	// witness that the gate is control-path-only, O(1) in page size on the data path.
	if &a[0] != &b[0] {
		t.Fatalf("forwarded byte slice is not identity-equal across reads — the sidecar copied/re-scanned the data path (not control-path-only)")
	}
}

// TestRefereeSidecar_DurabilityDefersToVerdict proves the #496 deferral: by default a set
// records the explicit deferred sentinel (the sidecar does NOT invent a durability
// class), and when an S7 verdict IS injected the sidecar records exactly what that
// verdict returns — it calls the verdict, it does not duplicate the judgment.
func TestRefereeSidecar_DurabilityDefersToVerdict(t *testing.T) {
	page, content := l3page("a system prompt", abi.ScopeFleet)

	// Default: deferred to #496, not classified here.
	deferred := NewRefereeSidecar(NewMockL3Backend(), nil)
	if got := deferred.Set(page, "tenant-A", content).Durability; got != L3DurabilityDeferred {
		t.Fatalf("default set must defer durability, recorded %q want %q", got, L3DurabilityDeferred)
	}

	// Injected verdict: the sidecar records exactly the class the verdict returns, and
	// passes through the real page+bytes so the verdict can classify them.
	calls := 0
	verdict := func(p abi.Ref, payload []byte) string {
		calls++
		if p.Digest != page.Digest || len(payload) != len(content) {
			t.Fatalf("the verdict must receive the real page and bytes")
		}
		return "durable"
	}
	wired := NewRefereeSidecar(NewMockL3Backend(), verdict)
	meta := wired.Set(page, "tenant-A", content)
	if calls != 1 {
		t.Fatalf("set must CALL the injected #496 verdict exactly once, called %d", calls)
	}
	if meta.Durability != "durable" {
		t.Fatalf("set must record the verdict's class, recorded %q want %q", meta.Durability, "durable")
	}
}

// TestRefereeSidecar_MissIsNotARefusal proves a get for a key the tier never held is a
// MISS (found=false), not a refusal — the caller takes its cold path, and the gate emits
// no refusal token for a simple absence.
func TestRefereeSidecar_MissIsNotARefusal(t *testing.T) {
	side := NewRefereeSidecar(NewMockL3Backend(), nil)
	got, _, found, v := side.Get("a digest never set", "tenant-A")
	if found {
		t.Fatalf("a key never set must miss, got found")
	}
	if got != nil {
		t.Fatalf("a miss returns no bytes, got %d", len(got))
	}
	if v.Reason != "" || v.Admitted {
		t.Fatalf("a miss is not a refusal and not an admit, got reason=%q admitted=%v", v.Reason, v.Admitted)
	}
}
