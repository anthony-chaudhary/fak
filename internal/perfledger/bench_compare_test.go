package perfledger

import (
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/perfledger/**
func TestBenchTokenComparison(t *testing.T) {
	full := strings.Repeat("a", 64)
	other := full[:12] + strings.Repeat("b", 52)
	tests := []struct {
		name    string
		change  func(*Report)
		outcome BenchTokenOutcome
		reason  BenchTokenReason
	}{
		{"equal", func(*Report) {}, BenchTokensEqual, BenchTokenVerified},
		{"different_same_short_prefix", func(r *Report) { r.Records[1].BenchEvidence.TokenSHA256 = other }, BenchTokensDifferent, BenchTokenVerified},
		{"missing", func(r *Report) { r.Records = r.Records[:1] }, BenchTokensUnverified, BenchTokenMissingRequest},
		{"duplicate", func(r *Report) { r.Records = append(r.Records, r.Records[0]) }, BenchTokensUnverified, BenchTokenAmbiguousRequest},
		{"error_after_http_success", func(r *Report) { r.Records[1].Error = ErrorUpstreamStatus }, BenchTokensUnverified, BenchTokenFailedRequest},
		{"http_failure", func(r *Report) { r.Records[1].Status = 500 }, BenchTokensUnverified, BenchTokenFailedRequest},
		{"unknown_status", func(r *Report) { r.Records[1].Status = 0 }, BenchTokensUnverified, BenchTokenFailedRequest},
		{"incomplete", func(r *Report) { r.Records[1].BenchEvidence.Complete = false }, BenchTokensUnverified, BenchTokenIncomplete},
		{"absent_evidence", func(r *Report) { r.Records[1].BenchEvidence = nil }, BenchTokensUnverified, BenchTokenIncomplete},
		{"missing_full", func(r *Report) { r.Records[1].BenchEvidence.TokenSHA256 = "" }, BenchTokensUnverified, BenchTokenInvalidDigest},
		{"short_only", func(r *Report) { r.Records[1].BenchEvidence.TokenSHA256 = full[:12] }, BenchTokensUnverified, BenchTokenInvalidDigest},
		{"uppercase", func(r *Report) { r.Records[1].BenchEvidence.TokenSHA256 = strings.ToUpper(full) }, BenchTokensUnverified, BenchTokenInvalidDigest},
		{"nonhex", func(r *Report) { r.Records[1].BenchEvidence.TokenSHA256 = strings.Repeat("z", 64) }, BenchTokensUnverified, BenchTokenInvalidDigest},
		{"no_legacy", func(r *Report) { r.Records[1].BenchEvidence.TokenSHA = "" }, BenchTokensEqual, BenchTokenVerified},
		{"full_legacy_equal", func(r *Report) { r.Records[1].BenchEvidence.TokenSHA = full }, BenchTokensEqual, BenchTokenVerified},
		{"full_legacy_conflict_after_prefix", func(r *Report) { r.Records[1].BenchEvidence.TokenSHA = other }, BenchTokensUnverified, BenchTokenLegacyConflict},
		{"short_legacy_conflict", func(r *Report) { r.Records[1].BenchEvidence.TokenSHA = strings.Repeat("b", 12) }, BenchTokensUnverified, BenchTokenLegacyConflict},
		{"invalid_legacy", func(r *Report) { r.Records[1].BenchEvidence.TokenSHA = "bad" }, BenchTokensUnverified, BenchTokenLegacyConflict},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			report := Report{Source: "fixture", Window: Window{Requests: 2, Capped: true, DroppedWrites: 3}}
			for _, id := range []string{"left", "right"} {
				report.Records = append(report.Records, Record{RequestID: id, Status: 200,
					BenchEvidence: &BenchEvidence{Complete: true, TokenSHA: full[:12], TokenSHA256: full, OutputSHA256: "same-output"}})
			}
			tc.change(&report)
			for _, pair := range [][2]string{{"left", "right"}, {"right", "left"}} {
				got := CompareBenchTokens(report, pair[0], pair[1])
				if got.Outcome != tc.outcome || got.Reason != tc.reason {
					t.Fatalf("outcome/reason = %s/%s, want %s/%s", got.Outcome, got.Reason, tc.outcome, tc.reason)
				}
				if got.Source != report.Source || got.Window != report.Window || got.LeftRequestID != pair[0] || got.RightRequestID != pair[1] {
					t.Fatal("comparison lost its selected report identity")
				}
			}
		})
	}
	for _, pair := range [][2]string{{"", "right"}, {"left", " "}, {"left", "left"}} {
		if got := CompareBenchTokens(Report{}, pair[0], pair[1]); got.Outcome != BenchTokensUnverified || got.Reason != BenchTokenInvalidSelection {
			t.Fatal("invalid selection received a token verdict")
		}
	}
}
