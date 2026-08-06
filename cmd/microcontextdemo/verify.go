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
	if r.Schema != "fak-microcontext-spine/1" {
		return fmt.Errorf("schema=%q", r.Schema)
	}
	if r.Verdict != "PASS" || r.Mode != "openai-compatible" {
		return fmt.Errorf("verdict=%q mode=%q", r.Verdict, r.Mode)
	}
	if r.LogicalShards != 100 || r.Completed != 100 || r.Failed != 0 || r.TurnCount != 100 || r.UsageResponses != 100 {
		return fmt.Errorf("accounting mismatch: %+v", r)
	}
	if r.PhysicalWorkers < 1 || r.PhysicalWorkers >= r.LogicalShards {
		return fmt.Errorf("workers=%d are not bounded below contexts=%d", r.PhysicalWorkers, r.LogicalShards)
	}
	if r.SharedBaseInstalls != 1 || r.BaseFingerprint == "" {
		return fmt.Errorf("shared base evidence missing")
	}
	if r.PromptTokens <= 0 || r.CompletionTokens <= 0 || r.TTFTP50MS <= 0 || r.TTFTP95MS < r.TTFTP50MS {
		return fmt.Errorf("telemetry incomplete")
	}
	if r.Scope == "" {
		return fmt.Errorf("scope missing")
	}
	return nil
}
