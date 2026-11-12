// Adapted from TensorFold at 7cf02bdb401a79e7b13eb27b451c29a781583bbc.
// Copyright 2026 TensorFold contributors. Apache-2.0; see THIRD_PARTY_NOTICES.md ("TensorFold").
package perfledger

import "strings"

type BenchTokenOutcome string

const (
	BenchTokensEqual      BenchTokenOutcome = "equal"
	BenchTokensDifferent  BenchTokenOutcome = "different"
	BenchTokensUnverified BenchTokenOutcome = "unverified"
)

type BenchTokenReason string

const (
	BenchTokenVerified         BenchTokenReason = "verified_full_digests"
	BenchTokenInvalidSelection BenchTokenReason = "invalid_selection"
	BenchTokenMissingRequest   BenchTokenReason = "request_not_in_window"
	BenchTokenAmbiguousRequest BenchTokenReason = "ambiguous_request_id"
	BenchTokenFailedRequest    BenchTokenReason = "unsuccessful_request"
	BenchTokenIncomplete       BenchTokenReason = "incomplete_evidence"
	BenchTokenInvalidDigest    BenchTokenReason = "invalid_full_digest"
	BenchTokenLegacyConflict   BenchTokenReason = "inconsistent_legacy_digest"
)

type BenchTokenComparison struct {
	Schema         string            `json:"schema"`
	Source         string            `json:"source"`
	Window         Window            `json:"window"`
	LeftRequestID  string            `json:"left_request_id"`
	RightRequestID string            `json:"right_request_id"`
	Outcome        BenchTokenOutcome `json:"outcome"`
	Reason         BenchTokenReason  `json:"reason"`
}

func CompareBenchTokens(report Report, leftID, rightID string) BenchTokenComparison {
	result := BenchTokenComparison{
		Schema: "fak.bench-token-comparison.v1", Source: report.Source, Window: report.Window,
		LeftRequestID: leftID, RightRequestID: rightID, Outcome: BenchTokensUnverified,
		Reason: BenchTokenInvalidSelection,
	}
	if strings.TrimSpace(leftID) == "" || strings.TrimSpace(rightID) == "" || leftID == rightID {
		return result
	}
	var digests [2]string
	for side, id := range []string{leftID, rightID} {
		var found *Record
		for i := range report.Records {
			if report.Records[i].RequestID != id {
				continue
			}
			if found != nil {
				result.Reason = BenchTokenAmbiguousRequest
				return result
			}
			found = &report.Records[i]
		}
		if found == nil {
			result.Reason = BenchTokenMissingRequest
			return result
		}
		if found.Error != "" || found.Status < 200 || found.Status >= 300 {
			result.Reason = BenchTokenFailedRequest
			return result
		}
		e := found.BenchEvidence
		if e == nil || !e.Complete {
			result.Reason = BenchTokenIncomplete
			return result
		}
		if len(e.TokenSHA256) != 64 || !validBenchTokenSHA(e.TokenSHA256) {
			result.Reason = BenchTokenInvalidDigest
			return result
		}
		if e.TokenSHA != "" && (!validBenchTokenSHA(e.TokenSHA) || !strings.HasPrefix(e.TokenSHA256, e.TokenSHA)) {
			result.Reason = BenchTokenLegacyConflict
			return result
		}
		digests[side] = e.TokenSHA256
	}
	result.Reason = BenchTokenVerified
	result.Outcome = BenchTokensDifferent
	if digests[0] == digests[1] {
		result.Outcome = BenchTokensEqual
	}
	return result
}
