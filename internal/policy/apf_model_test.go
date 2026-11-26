package policy

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestAPFPolicy exercises the APF signed-policy artifact model: artifact shape,
// the admission-time enforcement point, policy-version binding, and fail-closed
// refusal on a version mismatch.
func TestAPFPolicy(t *testing.T) {
	const bound = "1.4.0"
	doc := APFDocument{
		Version: bound,
		Issuer:  "fak-policy-authority",
		Rules:   []string{"allow:read", "deny:self_modify"},
	}
	sig := []byte{0x01, 0x02, 0x03, 0x04} // opaque; presence-only, never verified

	t.Run("ArtifactShape", func(t *testing.T) {
		art := APFArtifact{Document: doc, Signature: sig}
		dig, err := APFArtifactDigest(doc)
		if err != nil {
			t.Fatalf("APFArtifactDigest: %v", err)
		}
		raw, err := hex.DecodeString(dig)
		if err != nil || len(raw) != sha256.Size {
			t.Fatalf("digest %q is not a hex sha256 (%d bytes decoded, err %v)", dig, len(raw), err)
		}
		// Content binding: same document, different signature bytes -> same digest.
		dig2, err := APFArtifactDigest(doc)
		if err != nil {
			t.Fatalf("APFArtifactDigest (repeat): %v", err)
		}
		if dig != dig2 {
			t.Fatalf("digest not deterministic: %s != %s", dig, dig2)
		}
		// Different document content -> different digest.
		other := doc
		other.Rules = []string{"allow:read"}
		digOther, err := APFArtifactDigest(other)
		if err != nil {
			t.Fatalf("APFArtifactDigest (other): %v", err)
		}
		if digOther == dig {
			t.Fatalf("distinct documents share digest %s", dig)
		}
		_ = art
	})

	t.Run("AdmissionTimeEnforcement", func(t *testing.T) {
		a, err := NewAPFAdmission(bound)
		if err != nil {
			t.Fatalf("NewAPFAdmission: %v", err)
		}
		if a.BoundVersion() != bound {
			t.Fatalf("BoundVersion() = %q, want %q", a.BoundVersion(), bound)
		}
		if got := a.AdmittedDigest(); got != "" {
			t.Fatalf("pre-admission AdmittedDigest() = %q, want empty", got)
		}
		// A malformed artifact is refused at admission and leaves no admitted state.
		bad := APFArtifact{Document: APFDocument{Version: bound, Issuer: "x"}, Signature: sig}
		if d := a.Admit(bad); d.Admitted || d.Reason != abi.ReasonMalformed {
			t.Fatalf("malformed admit = %+v, want refused MALFORMED", d)
		}
		if got := a.AdmittedDigest(); got != "" {
			t.Fatalf("refused admission recorded digest %q", got)
		}
		// A well-formed artifact is admitted at the enforcement point.
		d := a.Admit(APFArtifact{Document: doc, Signature: sig})
		if !d.Admitted || d.Reason != abi.ReasonNone {
			t.Fatalf("admit = %+v, want admitted NONE", d)
		}
		want, err := APFArtifactDigest(doc)
		if err != nil {
			t.Fatalf("APFArtifactDigest: %v", err)
		}
		if d.ArtifactDigest != want {
			t.Fatalf("decision digest = %s, want %s", d.ArtifactDigest, want)
		}
		if d.PolicyVersion != bound {
			t.Fatalf("decision version = %q, want %q", d.PolicyVersion, bound)
		}
		if got := a.AdmittedDigest(); got != want {
			t.Fatalf("AdmittedDigest() = %s, want %s", got, want)
		}
	})

	t.Run("PolicyVersionBinding", func(t *testing.T) {
		a, err := NewAPFAdmission(bound)
		if err != nil {
			t.Fatalf("NewAPFAdmission: %v", err)
		}
		// A waiver bound to the admitted version is honored.
		if d := a.Waive(APFWaiver{ID: "w1", PolicyVersion: bound, Reason: abi.ReasonRateLimited, GrantedBy: "ops"}); d.Admitted {
			t.Fatalf("waiver before any admission = %+v, want refused", d)
		}
		if d := a.Admit(APFArtifact{Document: doc, Signature: sig}); !d.Admitted {
			t.Fatalf("admit = %+v, want admitted", d)
		}
		d := a.Waive(APFWaiver{ID: "w1", PolicyVersion: bound, Reason: abi.ReasonRateLimited, GrantedBy: "ops"})
		if !d.Admitted || d.Reason != abi.ReasonRateLimited {
			t.Fatalf("waiver = %+v, want honored RATE_LIMITED", d)
		}
		if d.PolicyVersion != bound {
			t.Fatalf("waiver version = %q, want %q", d.PolicyVersion, bound)
		}
		// A waiver citing NONE or an out-of-range code is malformed.
		if d := a.Waive(APFWaiver{ID: "w2", PolicyVersion: bound, Reason: abi.ReasonNone, GrantedBy: "ops"}); d.Admitted || d.Reason != abi.ReasonMalformed {
			t.Fatalf("NONE waiver = %+v, want refused MALFORMED", d)
		}
		if d := a.Waive(APFWaiver{ID: "w3", PolicyVersion: bound, Reason: abi.ReasonCoreMax, GrantedBy: "ops"}); d.Admitted || d.Reason != abi.ReasonMalformed {
			t.Fatalf("out-of-range waiver = %+v, want refused MALFORMED", d)
		}
	})

	t.Run("FailClosedVersionMismatch", func(t *testing.T) {
		a, err := NewAPFAdmission(bound)
		if err != nil {
			t.Fatalf("NewAPFAdmission: %v", err)
		}
		// Presenting a document of a different version is refused fail-closed.
		stale := doc
		stale.Version = "1.3.9"
		d := a.Admit(APFArtifact{Document: stale, Signature: sig})
		if d.Admitted || d.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("mismatched-version admit = %+v, want refused POLICY_BLOCK", d)
		}
		if got := a.AdmittedDigest(); got != "" {
			t.Fatalf("mismatched admission recorded digest %q", got)
		}
		// Admit the bound version, then a waiver against the stale version must
		// still be refused: the binding is to what was admitted, not what was
		// merely presented.
		if d := a.Admit(APFArtifact{Document: doc, Signature: sig}); !d.Admitted {
			t.Fatalf("admit = %+v, want admitted", d)
		}
		if d := a.Waive(APFWaiver{ID: "w-stale", PolicyVersion: "1.3.9", Reason: abi.ReasonRateLimited, GrantedBy: "ops"}); d.Admitted || d.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("stale-version waiver = %+v, want refused POLICY_BLOCK", d)
		}
		// An admission point with no version binding cannot be constructed.
		if _, err := NewAPFAdmission(""); err == nil {
			t.Fatalf("NewAPFAdmission(\"\") succeeded, want refusal")
		}
	})
}
