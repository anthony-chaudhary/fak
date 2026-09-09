package landingvoucher

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestMintAndAuditLandingVoucher(t *testing.T) {
	baseOpts := AuditOptions{
		RepoRoot:            ".",
		CandidateSHA:        "a1b2c3d4e5f60123456789abcdef0123456789ab",
		BaseSHA:             "f0e1d2c3b4a50123456789abcdef0123456789ab",
		Lane:                "safesync",
		LeasedGlobs:         []string{"internal/landingvoucher/**"},
		ChangedFiles:        []string{"internal/landingvoucher/voucher.go", "internal/landingvoucher/auditor.go"},
		WitnessCommand:      "go test -v ./internal/landingvoucher",
		WitnessReceipt:      "PASS: TestMintAndAuditLandingVoucher",
		HasLockContention:   false,
		HasActiveLease:      true,
		ABIBreakageDetected: false,
		ForbiddenFilesFound: nil,
	}

	t.Run("full passing case", func(t *testing.T) {
		ctx := context.Background()
		voucher, err := AuditLandingInvariants(ctx, baseOpts)
		if err != nil {
			t.Fatalf("expected no error, got: %v", err)
		}
		if voucher == nil {
			t.Fatal("expected non-nil voucher")
		}
		if voucher.Schema != VoucherSchemaV1 {
			t.Errorf("expected schema %q, got %q", VoucherSchemaV1, voucher.Schema)
		}
		if voucher.Digest == "" {
			t.Fatal("expected non-empty digest")
		}
		if voucher.Digest != voucher.ComputeDigest() {
			t.Errorf("voucher digest mismatch: got %q, recomputed %q", voucher.Digest, voucher.ComputeDigest())
		}
		if !voucher.Invariants.AllPassed() {
			t.Fatalf("expected Invariants.AllPassed() == true, got false")
		}
		if !voucher.Invariants.SpatialDisjointness ||
			!voucher.Invariants.BaseFreshness ||
			!voucher.Invariants.ABIStability ||
			!voucher.Invariants.CleanPathspec ||
			!voucher.Invariants.WitnessProof ||
			!voucher.Invariants.ZeroLockContention ||
			!voucher.Invariants.WriterLeaseValidity ||
			!voucher.Invariants.BoundaryAdmission {
			t.Errorf("expected all 8 invariant flags to be true, got %+v", voucher.Invariants)
		}
		if voucher.CandidateSHA != baseOpts.CandidateSHA {
			t.Errorf("expected candidate SHA %q, got %q", baseOpts.CandidateSHA, voucher.CandidateSHA)
		}
		if voucher.BaseSHA != baseOpts.BaseSHA {
			t.Errorf("expected base SHA %q, got %q", baseOpts.BaseSHA, voucher.BaseSHA)
		}
		if voucher.Lane != baseOpts.Lane {
			t.Errorf("expected lane %q, got %q", baseOpts.Lane, voucher.Lane)
		}
		if voucher.Timestamp.IsZero() {
			t.Error("expected non-zero voucher timestamp")
		}
	})

	t.Run("deterministic digest verification", func(t *testing.T) {
		fixedTime := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
		v1 := LandingVoucher{
			Schema:         VoucherSchemaV1,
			Timestamp:      fixedTime,
			CandidateSHA:   "c1",
			BaseSHA:        "b1",
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
		v2 := v1
		d1 := v1.ComputeDigest()
		d2 := v2.ComputeDigest()
		if d1 == "" || d2 == "" {
			t.Fatal("expected non-empty digest")
		}
		if d1 != d2 {
			t.Fatalf("expected deterministic digest, got %q vs %q", d1, d2)
		}
	})

	t.Run("context cancellation", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		voucher, err := AuditLandingInvariants(ctx, baseOpts)
		if err == nil {
			t.Fatal("expected context cancellation error, got nil")
		}
		if voucher != nil {
			t.Errorf("expected nil voucher on context error, got %+v", voucher)
		}
	})

	negativeCases := []struct {
		name       string
		failedAxis string
		mutate     func(o *AuditOptions)
	}{
		{
			name:       "SpatialDisjointness failure (file outside leased globs)",
			failedAxis: "SpatialDisjointness",
			mutate: func(o *AuditOptions) {
				o.ChangedFiles = []string{"cmd/fak/main.go"}
			},
		},
		{
			name:       "BaseFreshness failure (CandidateSHA equals BaseSHA)",
			failedAxis: "BaseFreshness",
			mutate: func(o *AuditOptions) {
				o.CandidateSHA = o.BaseSHA
			},
		},
		{
			name:       "BaseFreshness failure (BaseSHA empty)",
			failedAxis: "BaseFreshness",
			mutate: func(o *AuditOptions) {
				o.BaseSHA = ""
			},
		},
		{
			name:       "ABIStability failure (breakage detected)",
			failedAxis: "ABIStability",
			mutate: func(o *AuditOptions) {
				o.ABIBreakageDetected = true
			},
		},
		{
			name:       "CleanPathspec failure (empty changed files)",
			failedAxis: "CleanPathspec",
			mutate: func(o *AuditOptions) {
				o.ChangedFiles = []string{}
			},
		},
		{
			name:       "WitnessProof failure (empty witness receipt)",
			failedAxis: "WitnessProof",
			mutate: func(o *AuditOptions) {
				o.WitnessReceipt = ""
			},
		},
		{
			name:       "WitnessProof failure (empty witness command)",
			failedAxis: "WitnessProof",
			mutate: func(o *AuditOptions) {
				o.WitnessCommand = ""
			},
		},
		{
			name:       "ZeroLockContention failure (lock contention flag true)",
			failedAxis: "ZeroLockContention",
			mutate: func(o *AuditOptions) {
				o.HasLockContention = true
			},
		},
		{
			name:       "WriterLeaseValidity failure (no active lease)",
			failedAxis: "WriterLeaseValidity",
			mutate: func(o *AuditOptions) {
				o.HasActiveLease = false
			},
		},
		{
			name:       "WriterLeaseValidity failure (empty lane)",
			failedAxis: "WriterLeaseValidity",
			mutate: func(o *AuditOptions) {
				o.Lane = ""
			},
		},
		{
			name:       "BoundaryAdmission failure (forbidden files detected)",
			failedAxis: "BoundaryAdmission",
			mutate: func(o *AuditOptions) {
				o.ForbiddenFilesFound = []string{"tools/bad.ps1"}
			},
		},
		{
			name:       "BoundaryAdmission failure (script in changed files)",
			failedAxis: "BoundaryAdmission",
			mutate: func(o *AuditOptions) {
				o.ChangedFiles = []string{"internal/landingvoucher/script.sh"}
			},
		},
	}

	for _, tc := range negativeCases {
		t.Run(tc.name, func(t *testing.T) {
			opts := baseOpts
			tc.mutate(&opts)
			ctx := context.Background()
			voucher, err := AuditLandingInvariants(ctx, opts)
			if err == nil {
				t.Fatalf("expected error, got nil with voucher: %+v", voucher)
			}
			if voucher != nil {
				t.Errorf("expected nil voucher on refusal, got %+v", voucher)
			}
			errStr := err.Error()
			if !strings.Contains(errStr, ReasonLandingInvariantRefused) {
				t.Errorf("expected error string to contain %q, got %q", ReasonLandingInvariantRefused, errStr)
			}
			if !strings.Contains(errStr, tc.failedAxis) {
				t.Errorf("expected error string to contain failed axis %q, got %q", tc.failedAxis, errStr)
			}
		})
	}
}
