package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

type abArm struct {
	Name               string  `json:"name"`
	PrefixMode         string  `json:"prefix_mode"`
	LogicalShards      int     `json:"logical_shards"`
	PhysicalWorkers    int     `json:"physical_workers"`
	Completed          int     `json:"completed"`
	Failed             int     `json:"failed"`
	WallMS             int64   `json:"wall_ms"`
	TTFTP50MS          float64 `json:"ttft_p50_ms"`
	TTFTP95MS          float64 `json:"ttft_p95_ms"`
	PromptTokens       int64   `json:"prompt_tokens"`
	CompletionTokens   int64   `json:"completion_tokens"`
	CachedPromptTokens int64   `json:"cached_prompt_tokens"`
	ShardsPerSecond    float64 `json:"shards_per_second"`
	PromptTokensPerSec float64 `json:"prompt_tokens_per_wall_second"`
	DecodeTokensPerSec float64 `json:"decode_tokens_per_wall_second"`
}

type nativeBatchProbe struct {
	Supported  bool   `json:"supported"`
	HTTPStatus int    `json:"http_status"`
	Reason     string `json:"reason"`
}

type abReport struct {
	Schema                   string           `json:"schema"`
	Verdict                  string           `json:"verdict"`
	ObservedAt               string           `json:"observed_at"`
	Provider                 string           `json:"provider"`
	Model                    string           `json:"model"`
	Hardware                 string           `json:"hardware"`
	BaseFingerprint          string           `json:"base_fingerprint"`
	Canonicalization         string           `json:"canonicalization"`
	ShardsPerArm             int              `json:"shards_per_arm"`
	NativeBatch              nativeBatchProbe `json:"provider_native_batch"`
	Arms                     []abArm          `json:"arms"`
	SharedVsUniqueThroughput float64          `json:"shared_vs_unique_throughput_ratio"`
	SharedVsUniqueTTFT       float64          `json:"shared_vs_unique_ttft_ratio"`
	ReuseEvidence            string           `json:"reuse_evidence"`
	ReuseLimits              []string         `json:"reuse_limits"`
	ClaimVerdict             string           `json:"claim_verdict"`
	Scope                    string           `json:"scope"`
}

func armFromReport(name, prefixMode string, r report) abArm {
	return abArm{
		Name: name, PrefixMode: prefixMode, LogicalShards: r.LogicalShards,
		PhysicalWorkers: r.PhysicalWorkers, Completed: r.Completed, Failed: r.Failed,
		WallMS: r.ElapsedMS, TTFTP50MS: r.TTFTP50MS, TTFTP95MS: r.TTFTP95MS,
		PromptTokens: r.PromptTokens, CompletionTokens: r.CompletionTokens,
		CachedPromptTokens: r.CachedPromptTokens, ShardsPerSecond: r.ShardsPerSecond,
		PromptTokensPerSec: r.PromptTokensPerSec, DecodeTokensPerSec: r.DecodeTokensPerSec,
	}
}

func probeNativeBatch(ctx context.Context, endpoint string, timeout time.Duration) nativeBatchProbe {
	url := strings.TrimRight(endpoint, "/")
	if !strings.HasSuffix(url, "/v1") {
		url += "/v1"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url+"/batches", nil)
	if err != nil {
		return nativeBatchProbe{Reason: err.Error()}
	}
	client := &http.Client{Timeout: timeout}
	resp, err := client.Do(req)
	if err != nil {
		return nativeBatchProbe{Reason: err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed {
		return nativeBatchProbe{HTTPStatus: resp.StatusCode, Reason: "endpoint exposes no provider-native chat batch API"}
	}
	return nativeBatchProbe{Supported: resp.StatusCode >= 200 && resp.StatusCode < 300, HTTPStatus: resp.StatusCode, Reason: "GET /v1/batches capability probe"}
}

func runAB(ctx context.Context, cfg config) (abReport, error) {
	if cfg.Endpoint == "" || cfg.Model == "" {
		return abReport{}, fmt.Errorf("A/B requires endpoint and model")
	}
	if cfg.Contexts < 2 {
		return abReport{}, fmt.Errorf("A/B requires at least two contexts per arm")
	}
	workers := cfg.Workers
	if workers < 2 {
		return abReport{}, fmt.Errorf("A/B requires at least two concurrent workers")
	}
	probe := probeNativeBatch(ctx, cfg.Endpoint, cfg.RequestTimeout)
	arms := make([]abArm, 0, 3)
	armCfg := cfg
	armCfg.Selfcheck = false
	// Run the two concurrent arms first with equal admission; sequential is the
	// tuned no-batch alternative and intentionally uses one physical slot.
	for _, spec := range []struct {
		name, prefix string
		workers      int
	}{
		{"concurrent-unique-full-prompts", "unique", workers},
		{"concurrent-byte-identical-base", "shared", workers},
		{"tuned-sequential-byte-identical-base", "shared", 1},
	} {
		armCfg.PrefixMode, armCfg.Workers = spec.prefix, spec.workers
		r, err := run(ctx, armCfg)
		if err != nil {
			return abReport{}, fmt.Errorf("arm %s: %w", spec.name, err)
		}
		arms = append(arms, armFromReport(spec.name, spec.prefix, r))
	}
	unique, shared := arms[0], arms[1]
	report := abReport{
		Schema: "fak-microcontext-prefix-ab/1", Verdict: "PASS", ObservedAt: time.Now().UTC().Format(time.RFC3339),
		Provider: cfg.Provider, Model: cfg.Model, Hardware: cfg.Hardware,
		BaseFingerprint: canonicalBaseFingerprint(), Canonicalization: "fixed system bytes; stable message/tool ordering; task delta is the only shared-arm mutation",
		ShardsPerArm: cfg.Contexts, NativeBatch: probe, Arms: arms,
		ReuseLimits:  []string{"endpoint returned zero cached_prompt_tokens", "endpoint metrics expose no prompt-prefix cache hit/miss counter", "provider-native /v1/batches API unavailable"},
		ClaimVerdict: "not-yet", Scope: "observed endpoint A/B; ratios are descriptive, not a cache-gain claim without cache provenance or replicated significance",
	}
	if unique.ShardsPerSecond > 0 {
		report.SharedVsUniqueThroughput = shared.ShardsPerSecond / unique.ShardsPerSecond
	}
	if unique.TTFTP50MS > 0 {
		report.SharedVsUniqueTTFT = shared.TTFTP50MS / unique.TTFTP50MS
	}
	if shared.CachedPromptTokens > 0 {
		report.ReuseEvidence = "provider reported cached prompt tokens"
		report.ClaimVerdict = "observed-cache-reuse"
	} else {
		report.ReuseEvidence = "no provider cache-hit evidence; shared-prefix benefit is falsified/not-yet on this endpoint"
	}
	for _, arm := range arms {
		if arm.Completed != cfg.Contexts || arm.Failed != 0 {
			report.Verdict = "FAIL"
			return report, fmt.Errorf("arm %s did not complete", arm.Name)
		}
	}
	return report, nil
}

func writeAB(path string, r abReport) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if path == "-" {
		_, err = os.Stdout.Write(data)
		return err
	}
	return os.WriteFile(path, data, 0o644)
}

func verifyABArtifact(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var r abReport
	if err := json.Unmarshal(data, &r); err != nil {
		return err
	}
	if r.Schema != "fak-microcontext-prefix-ab/1" || r.Verdict != "PASS" {
		return fmt.Errorf("bad schema/verdict")
	}
	if len(r.Arms) != 3 || r.ShardsPerArm < 2 || r.BaseFingerprint == "" {
		return fmt.Errorf("incomplete A/B dimensions")
	}
	for _, arm := range r.Arms {
		if arm.Completed != r.ShardsPerArm || arm.Failed != 0 || arm.PromptTokens == 0 || arm.TTFTP50MS <= 0 {
			return fmt.Errorf("arm %s incomplete", arm.Name)
		}
	}
	if r.ReuseEvidence == "" || r.ClaimVerdict == "" || r.Scope == "" {
		return fmt.Errorf("honesty fields missing")
	}
	return nil
}
