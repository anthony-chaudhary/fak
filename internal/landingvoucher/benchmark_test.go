package landingvoucher

import (
	"context"
	"errors"
	"testing"
	"time"
)

// benchmarkDigestSink retains the last computed digest so the compiler cannot
// elide the ComputeDigest calls in BenchmarkComputeDigest.
var benchmarkDigestSink string

// BenchmarkAuditLandingInvariants measures the happy-path audit: all 8 axes
// pass and a voucher is minted, including the SHA-256 digest computation.
func BenchmarkAuditLandingInvariants(b *testing.B) {
	b.ReportAllocs()
	opts := allPassingAuditOptions()
	for i := 0; i < b.N; i++ {
		v, err := AuditLandingInvariants(context.Background(), opts)
		if err != nil || v == nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkAuditLandingInvariantsRefusal measures the refusal path where
// exactly one axis (ZeroLockContention) fails, exercising the failing-axis
// error formatting without minting a voucher.
func BenchmarkAuditLandingInvariantsRefusal(b *testing.B) {
	b.ReportAllocs()
	opts := allPassingAuditOptions()
	opts.HasLockContention = true
	if _, err := AuditLandingInvariants(context.Background(), opts); !errors.Is(err, ErrLandingInvariantRefused) {
		b.Fatalf("fixture must fail with ErrLandingInvariantRefused, got: %v", err)
	}
	for i := 0; i < b.N; i++ {
		AuditLandingInvariants(context.Background(), opts)
	}
}

// BenchmarkComputeDigest measures the tamper-evident digest over a fixed
// voucher with a stable UTC timestamp.
func BenchmarkComputeDigest(b *testing.B) {
	b.ReportAllocs()
	v := LandingVoucher{
		Schema:         VoucherSchemaV1,
		Timestamp:      time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC),
		CandidateSHA:   "a1b2c3d4e5f60123456789abcdef0123456789ab",
		BaseSHA:        "f0e1d2c3b4a50123456789abcdef0123456789ab",
		Lane:           "safesync",
		LeasedGlobs:    []string{"internal/landingvoucher/**"},
		ChangedFiles:   []string{"internal/landingvoucher/voucher.go"},
		WitnessReceipt: "PASS",
		Invariants: InvariantChecklist{
			SpatialDisjointness: true,
			BaseFreshness:       true,
			ABIStability:        true,
			CleanPathspec:       true,
			WitnessProof:        true,
			ZeroLockContention:  true,
			WriterLeaseValidity: true,
			BoundaryAdmission:   true,
		},
	}
	for i := 0; i < b.N; i++ {
		benchmarkDigestSink = v.ComputeDigest()
	}
}
