package qwen38quantrun

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

// Observation is independent runtime readback. It deliberately excludes secrets.
type Observation struct {
	Identity       qwen38quant.Identity `json:"identity"`
	Hardware       string               `json:"hardware"`
	Software       string               `json:"software"`
	Device         string               `json:"device"`
	ContextTokens  int                  `json:"context_tokens"`
	CacheMode      string               `json:"cache_mode"`
	Resident       bool                 `json:"resident"`
	FallbackActive bool                 `json:"fallback_active"`
	MemoryBytes    uint64               `json:"memory_bytes,omitempty"`
	PowerWatts     float64              `json:"power_watts,omitempty"`
}

type Probe interface {
	Observe(context.Context) (Observation, error)
}
type Lifecycle interface {
	Restart(context.Context) error
	Ready(context.Context) error
	Cleanup(context.Context) error
}

type CampaignConfig struct {
	Endpoint          Config
	Arm               string
	Expected          qwen38quant.Identity
	Command           []string
	RequireDevice     string
	StaleAfter        string
	RollbackThreshold string
	Probe             Probe
	Lifecycle         Lifecycle
}

type Archive struct {
	Schema       string      `json:"schema"`
	CorpusID     string      `json:"corpus_id"`
	Arm          string      `json:"arm"`
	Before       Observation `json:"before"`
	After        Observation `json:"after"`
	Results      []Result    `json:"results"`
	RestartReady bool        `json:"restart_ready"`
	CleanupOK    bool        `json:"cleanup_ok"`
}

type Campaign struct {
	Report  qwen38quant.Report
	Archive []byte
}

func (r Runner) RunCampaign(ctx context.Context, cfg CampaignConfig, corpus qwen38quant.Corpus) (campaign Campaign, err error) {
	if cfg.Probe == nil || cfg.Lifecycle == nil {
		return Campaign{}, errors.New("probe and lifecycle are required")
	}
	cleanupDone := false
	defer func() {
		if cleanupDone {
			return
		}
		if cleanupErr := cfg.Lifecycle.Cleanup(ctx); cleanupErr != nil {
			if err == nil {
				err = fmt.Errorf("cleanup: %w", cleanupErr)
			} else {
				err = errors.Join(err, fmt.Errorf("cleanup: %w", cleanupErr))
			}
		}
	}()
	before, err := cfg.Probe.Observe(ctx)
	if err != nil {
		return Campaign{}, fmt.Errorf("evidence preflight: %w", err)
	}
	if err := admitObservation(before, cfg); err != nil {
		return Campaign{}, err
	}

	restartReady := false
	endpoint := cfg.Endpoint
	if endpoint.Sample == nil {
		endpoint.Sample = func(sampleCtx context.Context) (ResourceSample, error) {
			observed, sampleErr := cfg.Probe.Observe(sampleCtx)
			if sampleErr != nil {
				return ResourceSample{}, sampleErr
			}
			if err := admitObservation(observed, cfg); err != nil {
				return ResourceSample{}, err
			}
			return ResourceSample{MemoryBytes: observed.MemoryBytes, PowerWatts: observed.PowerWatts}, nil
		}
	}
	userBefore := endpoint.BeforeTrial
	endpoint.BeforeTrial = func(trialCtx context.Context, fixture qwen38quant.Fixture, repetition int) error {
		if userBefore != nil {
			if err := userBefore(trialCtx, fixture, repetition); err != nil {
				return err
			}
		}
		phase := cachePhase(fixture, repetition)
		if fixture.Workload != "repeated_workflow_cache" || (phase != "restart" && phase != "warm_after_restart") {
			return nil
		}
		if err := cfg.Lifecycle.Restart(trialCtx); err != nil {
			return fmt.Errorf("restart: %w", err)
		}
		if err := cfg.Lifecycle.Ready(trialCtx); err != nil {
			return fmt.Errorf("restart readiness: %w", err)
		}
		restartReady = true
		return nil
	}
	results, err := r.Run(ctx, endpoint, corpus)
	if err != nil {
		return Campaign{}, err
	}
	after, err := cfg.Probe.Observe(ctx)
	if err != nil {
		return Campaign{}, fmt.Errorf("post-run evidence: %w", err)
	}
	if err := admitObservation(after, cfg); err != nil {
		return Campaign{}, fmt.Errorf("post-run: %w", err)
	}
	cleanupErr := cfg.Lifecycle.Cleanup(ctx)
	cleanupDone = true
	cleanupOK := cleanupErr == nil

	archive := Archive{Schema: "fak.qwen38-quant-raw/1", CorpusID: corpus.ID, Arm: cfg.Arm, Before: before, After: after, Results: results, RestartReady: restartReady, CleanupOK: cleanupOK}
	raw, err := canonicalJSON(archive)
	if err != nil {
		return Campaign{}, err
	}
	raw = scrubSecret(raw, cfg.Endpoint.APIKey)
	sum := sha256.Sum256(raw)
	verdict := "PROMOTE"
	trials := make([]qwen38quant.Trial, 0, len(results))
	for _, result := range results {
		trials = append(trials, qwen38quant.Trial{Workload: result.Workload, Repetition: result.Repeat, Quality: result.Quality, LatencyMS: result.LatencyMS, Failure: result.Failure})
		if result.Quality != "PASS" {
			verdict = "HOLD"
		}
	}
	report := qwen38quant.Report{
		Schema: qwen38quant.Schema, CorpusID: corpus.ID, CorpusSHA256: qwen38quant.CorpusDigest(corpus), Arm: cfg.Arm,
		Identity:    before.Identity,
		Environment: qwen38quant.Environment{Command: append([]string(nil), cfg.Command...), Hardware: before.Hardware, Software: before.Software, ContextTokens: before.ContextTokens, CacheMode: before.CacheMode, RequireDevice: cfg.RequireDevice, DenyFallback: true},
		Trials:      trials, Verdict: verdict, EvidenceClass: "CAMPAIGN", RawArchiveSHA256: hex.EncodeToString(sum[:]), StaleAfter: cfg.StaleAfter, RollbackThreshold: cfg.RollbackThreshold,
	}
	if cleanupErr != nil {
		return Campaign{Report: report, Archive: raw}, fmt.Errorf("cleanup: %w", cleanupErr)
	}
	if err := qwen38quant.Validate(report, corpus); err != nil {
		return Campaign{Report: report, Archive: raw}, fmt.Errorf("campaign report: %w", err)
	}
	return Campaign{Report: report, Archive: raw}, nil
}

func admitObservation(got Observation, cfg CampaignConfig) error {
	if !reflect.DeepEqual(got.Identity, cfg.Expected) {
		return errors.New("immutable identity mismatch")
	}
	if got.Hardware == "" || got.Software == "" || got.Device == "" || got.ContextTokens <= 0 || got.CacheMode == "" {
		return errors.New("incomplete runtime observation")
	}
	if !got.Resident {
		return errors.New("model is not resident")
	}
	if got.FallbackActive {
		return errors.New("fallback is active")
	}
	if cfg.RequireDevice == "" || !strings.Contains(got.Device, cfg.RequireDevice) {
		return fmt.Errorf("required device %q not observed in %q", cfg.RequireDevice, got.Device)
	}
	return nil
}

func scrubSecret(raw []byte, secret string) []byte {
	if secret == "" {
		return raw
	}
	raw = bytes.ReplaceAll(raw, []byte(secret), []byte("[REDACTED]"))
	quoted, _ := json.Marshal(secret)
	if len(quoted) >= 2 {
		raw = bytes.ReplaceAll(raw, quoted[1:len(quoted)-1], []byte("[REDACTED]"))
	}
	return raw
}

func canonicalJSON(v any) ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}
