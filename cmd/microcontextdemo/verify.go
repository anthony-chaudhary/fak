package main

import (
	"encoding/json"
	"fmt"
	"os"
)

func verifyArtifact(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var r report
	if err := json.Unmarshal(data, &r); err != nil {
		return err
	}
	return verifyReport(r)
}

func verifyReport(r report) error {
	if r.Schema != "fak-microcontext-spine/1" {
		return fmt.Errorf("schema=%q", r.Schema)
	}
	if r.Verdict != "PASS" || r.Mode != "openai-compatible" {
		return fmt.Errorf("verdict=%q mode=%q", r.Verdict, r.Mode)
	}
	if !declaredScale(r.LogicalShards) {
		return fmt.Errorf("logical_shards=%d, want one of 100, 1000, or 10000", r.LogicalShards)
	}
	if r.Completed != r.LogicalShards || r.Failed != 0 || r.TurnCount != int64(r.LogicalShards) || r.UsageResponses != r.LogicalShards {
		return fmt.Errorf("accounting mismatch: submitted=%d completed=%d failed=%d turns=%d usage_responses=%d", r.LogicalShards, r.Completed, r.Failed, r.TurnCount, r.UsageResponses)
	}
	if r.PhysicalWorkers < 1 || r.PhysicalWorkers >= r.LogicalShards {
		return fmt.Errorf("workers=%d are not bounded below contexts=%d", r.PhysicalWorkers, r.LogicalShards)
	}
	if r.PeakInFlight < 1 || r.PeakInFlight > int64(r.PhysicalWorkers) {
		return fmt.Errorf("peak_in_flight=%d is outside bounded worker envelope 1..%d", r.PeakInFlight, r.PhysicalWorkers)
	}
	if r.SharedBaseInstalls != 1 || r.BaseFingerprint == "" {
		return fmt.Errorf("shared base evidence missing")
	}
	if r.PromptTokens <= 0 || r.CompletionTokens <= 0 || r.TTFTP50MS <= 0 || r.TTFTP95MS < r.TTFTP50MS || (r.TTFTMaxMS > 0 && r.TTFTMaxMS < r.TTFTP95MS) {
		return fmt.Errorf("telemetry incomplete")
	}
	if r.ElapsedMS <= 0 || r.PromptTokensPerSec <= 0 || r.DecodeTokensPerSec <= 0 {
		return fmt.Errorf("wall-rate telemetry incomplete")
	}
	if r.Scope == "" || r.Provider == "" || r.Model == "" || r.Hardware == "" {
		return fmt.Errorf("scope or provenance missing")
	}
	if r.LogicalShards >= 1000 {
		if r.ResourceSamples < 2 || r.ClientPeakRSSBytes <= 0 || r.ServerPeakRSSBytes <= 0 || r.ServerPeakHeapBytes <= 0 {
			return fmt.Errorf("scale resource evidence incomplete")
		}
		if r.EndpointPeakRequests < 1 || r.EndpointPeakRequests > r.PhysicalWorkers+1 {
			return fmt.Errorf("endpoint_peak_requests=%d is outside request plus metrics-probe envelope", r.EndpointPeakRequests)
		}
		if r.KVCapacityEvidence == "" || r.QueueEvidence == "" || r.ResultCheck == "" || r.VerifiedResultsPerSec <= 0 {
			return fmt.Errorf("scale claim-boundary evidence incomplete")
		}
	}
	return nil
}

func declaredScale(n int) bool {
	switch n {
	case 100, 1000, 10000:
		return true
	default:
		return false
	}
}

