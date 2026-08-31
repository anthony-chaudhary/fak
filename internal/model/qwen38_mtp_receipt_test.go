package model

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestQwen38MTPReceiptGoldenStates(t *testing.T) {
	tests := []struct {
		name string
		got  Qwen38MTPReceipt
		want string
	}{
		{
			name: "ordinary target-only downgrade",
			got: Qwen38MTPReceipt{
				SchemaVersion:   Qwen38MTPReceiptSchema,
				Outcome:         Qwen38MTPOutcomeTargetOnly,
				Engine:          Qwen38EngineTargetDecode,
				RequestedDepth:  4,
				DowngradeReason: Qwen38MTPMemoryUnsafe,
				LatencyNS:       Qwen38MTPLatencyNS{Setup: 8, Total: 8},
				MemoryBytes:     Qwen38MTPMemoryBytes{Peak: 4096},
			},
			want: `{"schema_version":"fak/qwen38-mtp-receipt/v1","outcome":"target_only","engine":"fak-native-qwen3.8-target-decode","requested_depth":4,"effective_depth":0,"tokens":{"proposed":0,"accepted":0,"rejected":0},"latency_ns":{"setup":8,"draft":0,"verify":0,"rollback":0,"sync":0,"recovery":0,"total":8},"memory_bytes":{"draft_workspace":0,"verify_workspace":0,"rollback_state":0,"peak":4096},"downgrade_reason":"memory_headroom_unsafe"}`,
		},
		{
			name: "successful speculative attempt",
			got: Qwen38MTPReceipt{
				SchemaVersion:  Qwen38MTPReceiptSchema,
				Outcome:        Qwen38MTPOutcomeSucceeded,
				Engine:         Qwen38EngineMTP,
				RequestedDepth: 4,
				EffectiveDepth: 3,
				Tokens: Qwen38MTPTokenAccounting{
					Proposed: 12, Accepted: 9, Rejected: 3,
					Distribution: []Qwen38MTPAcceptanceBucket{
						{Depth: 1, Proposed: 4, Accepted: 4},
						{Depth: 2, Proposed: 4, Accepted: 3, Rejected: 1},
						{Depth: 3, Proposed: 4, Accepted: 2, Rejected: 2},
					},
				},
				LatencyNS:   Qwen38MTPLatencyNS{Setup: 10, Draft: 20, Verify: 30, Rollback: 4, Sync: 5, Recovery: 1, Total: 70},
				MemoryBytes: Qwen38MTPMemoryBytes{DraftWorkspace: 1024, VerifyWorkspace: 2048, RollbackState: 512, Peak: 4096},
			},
			want: `{"schema_version":"fak/qwen38-mtp-receipt/v1","outcome":"speculative_succeeded","engine":"fak-native-qwen3.8-mtp","requested_depth":4,"effective_depth":3,"tokens":{"proposed":12,"accepted":9,"rejected":3,"distribution":[{"depth":1,"proposed":4,"accepted":4,"rejected":0},{"depth":2,"proposed":4,"accepted":3,"rejected":1},{"depth":3,"proposed":4,"accepted":2,"rejected":2}]},"latency_ns":{"setup":10,"draft":20,"verify":30,"rollback":4,"sync":5,"recovery":1,"total":70},"memory_bytes":{"draft_workspace":1024,"verify_workspace":2048,"rollback_state":512,"peak":4096}}`,
		},
		{
			name: "failed speculative attempt",
			got: Qwen38MTPReceipt{
				SchemaVersion:  Qwen38MTPReceiptSchema,
				Outcome:        Qwen38MTPOutcomeFailed,
				Engine:         Qwen38EngineMTP,
				FallbackEngine: Qwen38EngineTargetDecode,
				RequestedDepth: 4,
				EffectiveDepth: 2,
				Tokens: Qwen38MTPTokenAccounting{
					Proposed: 6, Accepted: 2, Rejected: 4,
					Distribution: []Qwen38MTPAcceptanceBucket{
						{Depth: 1, Proposed: 3, Accepted: 2, Rejected: 1},
						{Depth: 2, Proposed: 3, Rejected: 3},
					},
				},
				LatencyNS:       Qwen38MTPLatencyNS{Setup: 10, Draft: 20, Verify: 12, Rollback: 7, Sync: 3, Recovery: 8, Total: 60},
				MemoryBytes:     Qwen38MTPMemoryBytes{DraftWorkspace: 1024, VerifyWorkspace: 2048, RollbackState: 512, Peak: 4096},
				DowngradeReason: Qwen38MTPAttemptFailed,
				FailureReason:   Qwen38MTPVerificationFailed,
			},
			want: `{"schema_version":"fak/qwen38-mtp-receipt/v1","outcome":"speculative_failed","engine":"fak-native-qwen3.8-mtp","fallback_engine":"fak-native-qwen3.8-target-decode","requested_depth":4,"effective_depth":2,"tokens":{"proposed":6,"accepted":2,"rejected":4,"distribution":[{"depth":1,"proposed":3,"accepted":2,"rejected":1},{"depth":2,"proposed":3,"accepted":0,"rejected":3}]},"latency_ns":{"setup":10,"draft":20,"verify":12,"rollback":7,"sync":3,"recovery":8,"total":60},"memory_bytes":{"draft_workspace":1024,"verify_workspace":2048,"rollback_state":512,"peak":4096},"downgrade_reason":"attempt_failed","failure_reason":"verification_failed"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.got.Validate(); err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
			got, err := json.Marshal(tt.got)
			if err != nil {
				t.Fatalf("Marshal() error = %v", err)
			}
			if string(got) != tt.want {
				t.Fatalf("receipt golden mismatch\n got: %s\nwant: %s", got, tt.want)
			}
		})
	}
}

func TestQwen38MTPReceiptValidationRejectsInvalidEvidence(t *testing.T) {
	valid := Qwen38MTPReceipt{
		SchemaVersion:  Qwen38MTPReceiptSchema,
		Outcome:        Qwen38MTPOutcomeSucceeded,
		Engine:         Qwen38EngineMTP,
		RequestedDepth: 2,
		EffectiveDepth: 2,
		Tokens: Qwen38MTPTokenAccounting{
			Proposed: 4, Accepted: 3, Rejected: 1,
			Distribution: []Qwen38MTPAcceptanceBucket{
				{Depth: 1, Proposed: 2, Accepted: 2},
				{Depth: 2, Proposed: 2, Accepted: 1, Rejected: 1},
			},
		},
		LatencyNS:   Qwen38MTPLatencyNS{Setup: 1, Draft: 2, Verify: 3, Rollback: 1, Sync: 1, Recovery: 1, Total: 9},
		MemoryBytes: Qwen38MTPMemoryBytes{DraftWorkspace: 10, VerifyWorkspace: 20, RollbackState: 5, Peak: 20},
	}

	tests := []struct {
		name   string
		mutate func(*Qwen38MTPReceipt)
		match  string
	}{
		{"absent engine identity", func(r *Qwen38MTPReceipt) { r.Engine = "" }, "absent or not fak-native"},
		{"llama cpp engine", func(r *Qwen38MTPReceipt) { r.Engine = "llama.cpp" }, "not fak-native"},
		{"silent non-native fallback", func(r *Qwen38MTPReceipt) { r.FallbackEngine = "llama.cpp" }, "fallback engine"},
		{"non-additive latency", func(r *Qwen38MTPReceipt) { r.LatencyNS.Total++ }, "non-additive"},
		{"accepted exceeds proposed", func(r *Qwen38MTPReceipt) { r.Tokens.Accepted = 5 }, "impossible token totals"},
		{"distribution does not sum", func(r *Qwen38MTPReceipt) { r.Tokens.Distribution[0].Accepted-- }, "impossible acceptance bucket"},
		{"distribution depth exceeds effective", func(r *Qwen38MTPReceipt) { r.Tokens.Distribution[1].Depth = 3 }, "impossible acceptance bucket"},
		{"duplicate distribution depth", func(r *Qwen38MTPReceipt) { r.Tokens.Distribution[1].Depth = 1 }, "impossible acceptance bucket"},
		{"effective exceeds requested", func(r *Qwen38MTPReceipt) { r.EffectiveDepth = 3 }, "impossible depths"},
		{"peak below workspace", func(r *Qwen38MTPReceipt) { r.MemoryBytes.Peak = 19 }, "peak memory"},
		{"success with downgrade", func(r *Qwen38MTPReceipt) { r.DowngradeReason = Qwen38MTPMemoryUnsafe }, "cannot contain"},
		{"unknown failure reason", func(r *Qwen38MTPReceipt) { r.FailureReason = "foreign_failure" }, "unknown failure reason"},
		{"unknown downgrade reason", func(r *Qwen38MTPReceipt) { r.DowngradeReason = "foreign_downgrade" }, "unknown downgrade reason"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := valid
			got.Tokens.Distribution = append([]Qwen38MTPAcceptanceBucket(nil), valid.Tokens.Distribution...)
			tt.mutate(&got)
			err := got.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.match) {
				t.Fatalf("Validate() error = %v, want containing %q", err, tt.match)
			}
		})
	}
}

func TestQwen38MTPReceiptTargetOnlyFromEligibility(t *testing.T) {
	decision := EvaluateQwen38MTPEligibility(Qwen38MTPEligibilityInput{
		Qwen38MTPArtifact: true,
		MTPBackendReady:   true,
		F32:               true,
		Greedy:            true,
		Depth:             1,
		FreshSession:      true,
		MemoryHeadroomOK:  false,
		OperatorEnabled:   true,
	})
	receipt := Qwen38MTPReceipt{
		SchemaVersion:   Qwen38MTPReceiptSchema,
		Outcome:         Qwen38MTPOutcomeTargetOnly,
		Engine:          decision.Engine,
		RequestedDepth:  1,
		DowngradeReason: decision.DowngradeReason,
	}
	if err := receipt.Validate(); err != nil {
		t.Fatalf("eligibility downgrade receipt invalid: %v", err)
	}
	if receipt.Engine != Qwen38EngineTargetDecode || receipt.DowngradeReason != Qwen38MTPMemoryUnsafe {
		t.Fatalf("receipt = %+v", receipt)
	}
}
