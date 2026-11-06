package landingvoucher

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// allPassingAuditOptions returns a fresh AuditOptions fixture in which all 8
// pre-landing invariant axes pass. Callers may mutate the returned value.
func allPassingAuditOptions() AuditOptions {
	return AuditOptions{
		RepoRoot:            ".",
		CandidateSHA:        "a1b2c3d4e5f60123456789abcdef0123456789ab",
		BaseSHA:             "f0e1d2c3b4a50123456789abcdef0123456789ab",
		Lane:                "safesync",
		LeasedGlobs:         []string{"internal/landingvoucher/**"},
		ChangedFiles:        []string{"internal/landingvoucher/voucher.go", "internal/landingvoucher/auditor.go"},
		WitnessCommand:      "go test ./internal/landingvoucher",
		WitnessReceipt:      "PASS",
		HasLockContention:   false,
		HasActiveLease:      true,
		ABIBreakageDetected: false,
		ForbiddenFilesFound: nil,
	}
}

// TestAuditRefusalWrapsSentinel pins the closed-token contract that guards and
// journals rely on: every axis refusal wraps ErrLandingInvariantRefused so
// errors.Is matches, names the failing axis, and carries the verbatim reason
// token. voucher_test.go asserts the same refusals through error strings only;
// the sentinel-wrap property is the delta this test owns.
func TestAuditRefusalWrapsSentinel(t *testing.T) {
	opts := allPassingAuditOptions()
	opts.HasLockContention = true
	_, err := AuditLandingInvariants(context.Background(), opts)
	if !errors.Is(err, ErrLandingInvariantRefused) {
		t.Fatalf("expected errors.Is(err, ErrLandingInvariantRefused), got: %v", err)
	}
	errStr := err.Error()
	if !strings.Contains(errStr, "ZeroLockContention") {
		t.Errorf("error should name failing axis %q, got: %q", "ZeroLockContention", errStr)
	}
	if !strings.Contains(errStr, ReasonLandingInvariantRefused) {
		t.Errorf("error should carry closed token %q, got: %q", ReasonLandingInvariantRefused, errStr)
	}
}

// TestComputeDigestSensitiveToFields pins tamper-evidence: any attested field
// change must move the digest, or the voucher stops being tamper-evident.
// voucher_test.go covers determinism (identical inputs -> identical digest);
// sensitivity to a changed input is the delta this test owns.
func TestComputeDigestSensitiveToFields(t *testing.T) {
	fixedTime := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
	invariants := InvariantChecklist{
		SpatialDisjointness: true,
		BaseFreshness:       true,
		ABIStability:        true,
		CleanPathspec:       true,
		WitnessProof:        true,
		ZeroLockContention:  true,
		WriterLeaseValidity: true,
		BoundaryAdmission:   true,
	}
	v1 := LandingVoucher{
		Schema:         VoucherSchemaV1,
		Timestamp:      fixedTime,
		CandidateSHA:   "c1",
		BaseSHA:        "b1",
		Lane:           "safesync",
		LeasedGlobs:    []string{"internal/landingvoucher/**"},
		ChangedFiles:   []string{"internal/landingvoucher/voucher.go"},
		WitnessReceipt: "PASS",
		Invariants:     invariants,
	}
	d1 := v1.ComputeDigest()
	if d1 == "" {
		t.Fatal("expected non-empty digest")
	}
	v2 := v1
	v2.WitnessReceipt = "CHANGED"
	if v1.ComputeDigest() == v2.ComputeDigest() {
		t.Error("changing WitnessReceipt should change the digest")
	}
}

// TestComputeDigestCanonicalizesNilSlices pins the nil -> []string{} canonicalization
// in ComputeDigest: an absent list field must not reach the digest payload as JSON
// null, or a minted voucher and its zero-list twin would hash differently.
func TestComputeDigestCanonicalizesNilSlices(t *testing.T) {
	fixedTime := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
	invariants := InvariantChecklist{
		SpatialDisjointness: true,
		BaseFreshness:       true,
		ABIStability:        true,
		CleanPathspec:       true,
		WitnessProof:        true,
		ZeroLockContention:  true,
		WriterLeaseValidity: true,
		BoundaryAdmission:   true,
	}
	absent := LandingVoucher{
		Schema:         VoucherSchemaV1,
		Timestamp:      fixedTime,
		CandidateSHA:   "c1",
		BaseSHA:        "b1",
		Lane:           "safesync",
		LeasedGlobs:    nil,
		ChangedFiles:   nil,
		WitnessReceipt: "PASS",
		Invariants:     invariants,
	}
	explicitEmpty := absent
	explicitEmpty.LeasedGlobs = []string{}
	explicitEmpty.ChangedFiles = []string{}
	if d := absent.ComputeDigest(); d == "" {
		t.Fatal("expected non-empty digest")
	}
	if absent.ComputeDigest() != explicitEmpty.ComputeDigest() {
		t.Error("nil list fields must canonicalize to empty slices; digests diverge")
	}
}
