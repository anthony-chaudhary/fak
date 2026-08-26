package qwenworkbudget

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func TestEvaluateUsesCanonicalQwenContributorsBelowBudget(t *testing.T) {
	audit := auditWith(
		transcript("claude", "q1", "Qwen3", 40, 10, 50, 2),
		transcript("codex", "q2", "qwen-2.5", 20, 5, 25, 1),
		transcript("codex", "other", "gpt-5", 999, 999, 1, 1),
	)
	receipt := (Policy{MaxInputPerOutput: 2}).Evaluate(Packet{Boundary: BoundaryLaunch, Engine: "fak-native/qwen3", Audit: &audit})
	if !receipt.Eligible || receipt.Action != trajectory.QwenAmplificationObserve {
		t.Fatalf("decision = eligible %v action %q", receipt.Eligible, receipt.Action)
	}
	if got, want := *receipt.InputPerOutput, 1.0; got != want {
		t.Fatalf("input/output = %v, want %v", got, want)
	}
	if len(receipt.Contributors) != 2 || receipt.Usage.InputTokens != 60 || receipt.Usage.CacheReadTokens != 15 || receipt.Usage.OutputTokens != 75 {
		t.Fatalf("attribution = %+v usage=%+v", receipt.Contributors, receipt.Usage)
	}
}

func TestEvaluateHoldsAmplificationBreachBeforeContinuation(t *testing.T) {
	audit := auditWith(transcript("claude", "giant", "qwen3", 100, 200, 1, 1))
	policy := Policy{QwenAmplificationPolicy: trajectory.QwenAmplificationPolicy{Enforce: true}, MaxInputPerOutput: 100}
	receipt := policy.Evaluate(Packet{Boundary: BoundaryContinuation, Engine: "fak-native/qwen3", Audit: &audit})
	if receipt.Action != trajectory.QwenAmplificationHold || receipt.Boundary != BoundaryContinuation {
		t.Fatalf("receipt = %+v", receipt)
	}
}

func TestEvaluateObserveOnlyAlertsAndOverrideIsExplicit(t *testing.T) {
	audit := auditWith(transcript("claude", "q", "qwen3", 100, 100, 1, 1))
	observe := (Policy{MaxInputPerOutput: 10}).Evaluate(Packet{Engine: "fak-native/qwen3", Audit: &audit})
	if observe.Action != trajectory.QwenAmplificationAlert {
		t.Fatalf("observe-only action = %q", observe.Action)
	}
	policy := Policy{QwenAmplificationPolicy: trajectory.QwenAmplificationPolicy{Enforce: true}, MaxInputPerOutput: 10}
	override := policy.Evaluate(Packet{Engine: "fak-native/qwen3", Audit: &audit, OverrideReason: "finish controlled witness"})
	if override.Action != trajectory.QwenAmplificationAlert || override.Override == nil || override.Override.Reason != "finish controlled witness" {
		t.Fatalf("override receipt = %+v", override)
	}
}

func TestEvaluateZeroOutputBreachesWithoutNonJSONRatio(t *testing.T) {
	audit := auditWith(transcript("claude", "q", "qwen3", 1, 0, 0, 1))
	policy := Policy{QwenAmplificationPolicy: trajectory.QwenAmplificationPolicy{Enforce: true}, MaxInputPerOutput: 10}
	receipt := policy.Evaluate(Packet{Engine: "fak-native/qwen3", Audit: &audit})
	if receipt.Action != trajectory.QwenAmplificationHold || receipt.InputPerOutput != nil {
		t.Fatalf("zero-output receipt = %+v", receipt)
	}
}

func TestObservedCohortRatioMatchesScopedAudit(t *testing.T) {
	audit := auditWith(transcript("claude", "2026-08-23-cohort", "qwen", 146810316, 1103765248, 3864539, 1))
	receipt := (Policy{MaxInputPerOutput: 400}).Evaluate(Packet{Engine: "fak-native/qwen", Audit: &audit})
	if got, want := *receipt.InputPerOutput, 323.603; got < want-0.0005 || got > want+0.0005 {
		t.Fatalf("observed ratio = %.6f, want %.3f (+/- 0.0005)", got, want)
	}
	if receipt.Action != trajectory.QwenAmplificationObserve {
		t.Fatalf("observed cohort action = %q, want observe below declared budget", receipt.Action)
	}
}
func TestEvaluateMissingUsageAndNativeEngineIdentity(t *testing.T) {
	policy := Policy{QwenAmplificationPolicy: trajectory.QwenAmplificationPolicy{Enforce: true}, MaxInputPerOutput: 10}
	missing := policy.Evaluate(Packet{Engine: "fak-native/qwen3"})
	if missing.Eligible || missing.Reason != trajectory.QwenAmplificationMissingUsage {
		t.Fatalf("missing receipt = %+v", missing)
	}
	audit := auditWith(transcript("claude", "q", "qwen3", 1, 1, 1, 1))
	invalid := policy.Evaluate(Packet{Engine: "llama.cpp/qwen3", Audit: &audit})
	if invalid.Eligible || invalid.Reason != trajectory.QwenAmplificationInvalidEngine || invalid.Engine != "llama.cpp/qwen3" {
		t.Fatalf("engine receipt = %+v", invalid)
	}
}

func auditWith(rows ...trajectory.AuditTranscriptRow) trajectory.AuditResult {
	return trajectory.AuditResult{Transcripts: rows}
}

func transcript(source, id, model string, input, cacheRead, output int64, records int) trajectory.AuditTranscriptRow {
	return trajectory.AuditTranscriptRow{
		Source: source, TranscriptID: id, Models: []string{model}, UsageRecords: records,
		Tokens: trajectory.AuditTokens{InputTokens: input, CacheReadTokens: cacheRead, OutputTokens: output},
	}
}
