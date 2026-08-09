package main

import "testing"

func TestVerifyArtifact(t *testing.T) {
	if err := verifyArtifact("../../experiments/microcontext/s1-gcp-realendpoint-workers4-pass-2026-08-06.json"); err != nil {
		t.Fatal(err)
	}
}

func TestVerifyReportAcceptsDeclaredScaleLadder(t *testing.T) {
	for _, contexts := range []int{100, 1000, 10000} {
		r := validRealEndpointReport(contexts)
		if err := verifyReport(r); err != nil {
			t.Errorf("contexts=%d: %v", contexts, err)
		}
	}
}

func TestVerifyReportRejectsUnsupportedOrIncompleteScale(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*report)
	}{
		{"undeclared scale", func(r *report) { r.LogicalShards, r.Completed, r.TurnCount, r.UsageResponses = 101, 101, 101, 101 }},
		{"missing completion", func(r *report) { r.Completed-- }},
		{"missing usage", func(r *report) { r.UsageResponses-- }},
		{"unbounded workers", func(r *report) { r.PhysicalWorkers = r.LogicalShards }},
		{"peak exceeds workers", func(r *report) { r.PeakInFlight = int64(r.PhysicalWorkers + 1) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := validRealEndpointReport(1000)
			tt.mutate(&r)
			if err := verifyReport(r); err == nil {
				t.Fatal("expected verifier refusal")
			}
		})
	}
}

func validRealEndpointReport(contexts int) report {
	r := report{
		Schema: "fak-microcontext-spine/1", Verdict: "PASS", Mode: "openai-compatible",
		LogicalShards: contexts, PhysicalWorkers: 1, Completed: contexts, TurnCount: int64(contexts),
		UsageResponses: contexts, SharedBaseInstalls: 1, PeakInFlight: 1, ElapsedMS: 1,
		Provider: "fixture", Model: "fixture", Hardware: "fixture", BaseFingerprint: "base",
		PromptTokens: int64(contexts), CompletionTokens: int64(contexts), TTFTP50MS: 1, TTFTP95MS: 1,
		PromptTokensPerSec: 1, DecodeTokensPerSec: 1, Scope: "fixture",
		ResourceSamples: 2, ClientPeakRSSBytes: 1, ServerPeakRSSBytes: 1, ServerPeakHeapBytes: 1, EndpointPeakRequests: 1,
		KVCapacityEvidence: "fixture", QueueEvidence: "fixture", ResultCheck: "fixture", VerifiedResultsPerSec: 1,
	}
	if contexts == 10000 {
		r.SoakContract = "controlled-10k-v1"
		r.CanaryContexts, r.CanaryPassed, r.BaseRollbackCount = 1, 1, 1
		r.RetryInjected, r.RetryRecovered = 1, 1
		r.CancellationInjected, r.CancellationRecovered = 1, 1
		r.MaxAttempts = 3
		r.QueuePeakContexts = contexts
		r.HibernatedContexts = contexts - r.PhysicalWorkers
		r.RestoredContexts = r.HibernatedContexts
	}
	return r
}
