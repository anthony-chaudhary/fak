package qwen38quantrun

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

type staticProbe struct {
	observation Observation
	calls       int
}

func (p *staticProbe) Observe(context.Context) (Observation, error) {
	p.calls++
	return p.observation, nil
}

type lifecycleSpy struct{ restarts, ready, cleanup int }

func (l *lifecycleSpy) Restart(context.Context) error { l.restarts++; return nil }
func (l *lifecycleSpy) Ready(context.Context) error   { l.ready++; return nil }
func (l *lifecycleSpy) Cleanup(context.Context) error { l.cleanup++; return nil }

func TestCampaignBuildsValidatorCleanReportAndArchive(t *testing.T) {
	corpus := qwen38quant.DefaultCorpus()
	calls := 0
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"id": "exact"}}})
			return
		}
		calls++
		msg := map[string]any{"content": "ok"}
		switch {
		case calls <= 3:
			msg["content"] = "ok"
		case calls <= 6:
			msg["content"] = `{"ok":true}`
		case calls <= 9:
			msg["content"] = ""
			msg["tool_calls"] = []any{map[string]any{"function": map[string]any{"name": "x", "arguments": `null`}}}
		default:
			msg["content"] = "ok"
		}
		json.NewEncoder(w).Encode(map[string]any{"model": "exact", "choices": []any{map[string]any{"message": msg}}, "usage": map[string]int{"prompt_tokens": 10, "completion_tokens": 2}})
	}))
	defer s.Close()
	id := qwen38quant.Identity{Model: "Qwen3.8-27B", CheckpointSHA256: hash64('a'), ArtifactSHA256: hash64('b'), TokenizerSHA256: hash64('c'), TemplateSHA256: hash64('d'), QuantizerRevision: "quant-r1", RuntimeRevision: "runtime-r1", FakModuleRev: "internal/gateway r1+gabcdef0"}
	observation := Observation{Identity: id, Hardware: "A100 40GB", Software: "fak runtime-r1", Device: "NVIDIA A100-SXM4-40GB", ContextTokens: 16384, CacheMode: "prefix", Resident: true, MemoryBytes: 20 << 30, PowerWatts: 211}
	probe, lifecycle := &staticProbe{observation: observation}, &lifecycleSpy{}
	got, err := (Runner{}).RunCampaign(context.Background(), CampaignConfig{Endpoint: Config{Endpoint: s.URL, APIKey: "secret", Model: "exact"}, ExecutionEngine: qwen38quant.EngineFakNative, Arm: "q4_k_m", Expected: id, Command: []string{"fak", "serve"}, RequireDevice: "A100", StaleAfter: "2026-09-20", RollbackThreshold: "quality pass rate below 100%", Probe: probe, Lifecycle: lifecycle}, corpus)
	if err != nil {
		t.Fatal(err)
	}
	if err := qwen38quant.Validate(got.Report, corpus); err != nil {
		t.Fatal(err)
	}
	if got.Report.ExecutionEngine != qwen38quant.EngineFakNative || got.Report.Verdict != "PROMOTE" || len(got.Archive) == 0 || probe.calls != 20 || lifecycle.restarts != 1 || lifecycle.ready != 1 || lifecycle.cleanup != 1 {
		t.Fatalf("unexpected campaign: %#v probe=%d lifecycle=%+v", got.Report, probe.calls, lifecycle)
	}
	if contains(string(got.Archive), "secret") {
		t.Fatal("archive leaked API key")
	}

	calls = 0
	comparisons := []struct{ engine, arm string }{
		{qwen38quant.EngineLlamaCpp, "q4_k_m"},
		{qwen38quant.EngineVLLM, "fp8"},
	}
	for _, comparisonCase := range comparisons {
		calls = 0
		comparisonProbe, comparisonLifecycle := &staticProbe{observation: observation}, &lifecycleSpy{}
		comparison, err := (Runner{}).RunCampaign(context.Background(), CampaignConfig{Endpoint: Config{Endpoint: s.URL, APIKey: "secret", Model: "exact"}, ExecutionEngine: comparisonCase.engine, Arm: comparisonCase.arm, Expected: id, Command: []string{comparisonCase.engine, "serve"}, RequireDevice: "A100", StaleAfter: "2026-09-20", RollbackThreshold: "quality pass rate below 100%", Probe: comparisonProbe, Lifecycle: comparisonLifecycle}, corpus)
		if err != nil {
			t.Fatal(err)
		}
		if comparison.Report.ExecutionEngine != comparisonCase.engine || comparison.Report.Verdict != "HOLD" {
			t.Fatalf("comparison engine/verdict = %q/%q, want %s/HOLD", comparison.Report.ExecutionEngine, comparison.Report.Verdict, comparisonCase.engine)
		}
		if err := qwen38quant.Validate(comparison.Report, corpus); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCampaignRejectsMissingOrUnknownEngineBeforeRequests(t *testing.T) {
	for _, engine := range []string{"", "other"} {
		t.Run(engine, func(t *testing.T) {
			probe, lifecycle := &staticProbe{}, &lifecycleSpy{}
			_, err := (Runner{}).RunCampaign(context.Background(), CampaignConfig{ExecutionEngine: engine, Probe: probe, Lifecycle: lifecycle}, qwen38quant.DefaultCorpus())
			if err == nil || !contains(err.Error(), "execution engine") {
				t.Fatalf("err=%v", err)
			}
			if probe.calls != 0 || lifecycle.cleanup != 0 {
				t.Fatalf("invalid engine touched runtime: probe=%d cleanup=%d", probe.calls, lifecycle.cleanup)
			}
		})
	}
}

func TestCampaignDeniesFallbackBeforeRequests(t *testing.T) {
	corpus := qwen38quant.DefaultCorpus()
	id := qwen38quant.Identity{Model: "x"}
	probe := &staticProbe{observation: Observation{Identity: id, Hardware: "h", Software: "s", Device: "A100", ContextTokens: 1, CacheMode: "none", Resident: true, FallbackActive: true}}
	_, err := (Runner{}).RunCampaign(context.Background(), CampaignConfig{Endpoint: Config{Endpoint: "http://invalid", APIKey: "secret", Model: "x"}, ExecutionEngine: qwen38quant.EngineFakNative, Arm: "q4_k_m", Expected: id, Command: []string{"x"}, RequireDevice: "A100", Probe: probe, Lifecycle: &lifecycleSpy{}}, corpus)
	if err == nil || !contains(err.Error(), "fallback") {
		t.Fatalf("err=%v", err)
	}
}
func hash64(c byte) string {
	b := make([]byte, 64)
	for i := range b {
		b[i] = c
	}
	return string(b)
}

func TestCampaignCleansUpAfterPreflightFailure(t *testing.T) {
	corpus := qwen38quant.DefaultCorpus()
	id := qwen38quant.Identity{Model: "x"}
	probe := &staticProbe{observation: Observation{Identity: id, Hardware: "h", Software: "s", Device: "A100", ContextTokens: 1, CacheMode: "none", Resident: true, FallbackActive: true}}
	lifecycle := &lifecycleSpy{}
	_, err := (Runner{}).RunCampaign(context.Background(), CampaignConfig{Endpoint: Config{Endpoint: "http://invalid", Model: "x"}, ExecutionEngine: qwen38quant.EngineFakNative, Arm: "q4_k_m", Expected: id, Command: []string{"x"}, RequireDevice: "A100", Probe: probe, Lifecycle: lifecycle}, corpus)
	if err == nil || lifecycle.cleanup != 1 {
		t.Fatalf("err=%v cleanup=%d", err, lifecycle.cleanup)
	}
}
