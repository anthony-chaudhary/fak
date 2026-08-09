package main

import (
	"encoding/json"
	"fmt"
	"os"
)

type apiOnlyArtifact struct {
	Schema             string  `json:"schema"`
	Verdict            string  `json:"verdict"`
	LogicalShards      int     `json:"logical_shards"`
	PhysicalWorkers    int     `json:"physical_workers"`
	Completed          int     `json:"completed"`
	Failed             int     `json:"failed"`
	TurnCount          int64   `json:"turn_count"`
	UsageResponses     int     `json:"usage_responses"`
	CachedPromptTokens int64   `json:"cached_prompt_tokens"`
	TTFTP50MS          float64 `json:"ttft_p50_ms"`
	TTFTP95MS          float64 `json:"ttft_p95_ms"`
	Provider           string  `json:"provider"`
	Model              string  `json:"model"`
	ObservedReuse      string  `json:"observed_reuse"`
	ClaimBoundary      string  `json:"claim_boundary"`
	Adapter            struct {
		Mode             string `json:"mode"`
		CredentialClass  string `json:"credential_class"`
		ReuseControl     string `json:"reuse_control"`
		ReuseEvidence    string `json:"reuse_evidence"`
		RPMLimit         int    `json:"rpm_limit"`
		TPMLimit         int    `json:"tpm_limit"`
		ConcurrencyLimit int    `json:"concurrency_limit"`
		ProviderManaged  bool   `json:"provider_managed"`
	} `json:"adapter"`
}

func verifyAPIOnlyArtifact(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var r apiOnlyArtifact
	if err := json.Unmarshal(data, &r); err != nil {
		return err
	}
	if r.Schema != "fak-microcontext-spine/1" || r.Verdict != "PASS" {
		return fmt.Errorf("schema=%q verdict=%q", r.Schema, r.Verdict)
	}
	if r.LogicalShards < 2 || r.Completed != r.LogicalShards || r.Failed != 0 || r.TurnCount != int64(r.LogicalShards) || r.UsageResponses != r.LogicalShards {
		return fmt.Errorf("API-only accounting mismatch")
	}
	if r.PhysicalWorkers <= r.Adapter.ConcurrencyLimit || r.Adapter.ConcurrencyLimit < 1 {
		return fmt.Errorf("provider admission was not narrower than orchestration workers")
	}
	if r.Adapter.Mode != "api-only" || r.Adapter.CredentialClass != "keyed-billing-credential" || !r.Adapter.ProviderManaged {
		return fmt.Errorf("API-only boundary missing")
	}
	if r.Adapter.RPMLimit <= 0 || r.Adapter.TPMLimit <= 0 || r.Adapter.ReuseControl == "" || r.Adapter.ReuseEvidence == "" {
		return fmt.Errorf("provider shape incomplete")
	}
	if r.CachedPromptTokens <= 0 || r.ObservedReuse == "" || r.ClaimBoundary == "" || r.TTFTP50MS <= 0 || r.TTFTP95MS < r.TTFTP50MS {
		return fmt.Errorf("observable evidence incomplete")
	}
	if r.Provider == "" || r.Model == "" {
		return fmt.Errorf("provider provenance missing")
	}
	return nil
}
