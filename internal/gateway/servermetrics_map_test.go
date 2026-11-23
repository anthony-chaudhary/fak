package gateway

import (
	"strings"
	"testing"
)

func smGauge(v float64) ServerMetricSample {
	return ServerMetricSample{Kind: ServerMetricGauge, Value: v}
}
func smCounter(v float64) ServerMetricSample {
	return ServerMetricSample{Kind: ServerMetricCounter, Value: v}
}

func TestServerMetricMapVLLMFirstWhenBothPresent(t *testing.T) {
	scraped := map[string]ServerMetricSample{
		"vllm:num_requests_running": smGauge(7),
		"sglang:num_running_reqs":   smGauge(3),
	}
	r, ok := ResolveServerMetric(ServerMetricNumRunning, scraped)
	if !ok {
		t.Fatal("expected resolution")
	}
	if r.Source != "vllm:num_requests_running" || r.Value != 7 || r.Neutral != ServerMetricNumRunning {
		t.Fatalf("got %+v, want vLLM source with value 7", r)
	}
}

func TestServerMetricMapSGLangFallbackWhenVLLMAbsent(t *testing.T) {
	scraped := map[string]ServerMetricSample{
		"sglang:num_queue_reqs": smGauge(4),
		"sglang:token_usage":    smGauge(0.25),
	}
	r, ok := ResolveServerMetric(ServerMetricNumWaiting, scraped)
	if !ok || r.Source != "sglang:num_queue_reqs" || r.Value != 4 {
		t.Fatalf("waiting: got %+v ok=%v", r, ok)
	}
	r, ok = ResolveServerMetric(ServerMetricKVCacheUsage, scraped)
	if !ok || r.Source != "sglang:token_usage" || r.Value != 0.25 {
		t.Fatalf("kv usage: got %+v ok=%v", r, ok)
	}
}

func TestServerMetricMapLegacyVLLMNameBeatsSGLang(t *testing.T) {
	scraped := map[string]ServerMetricSample{
		"vllm:gpu_cache_usage_perc": smGauge(0.5),
		"sglang:token_usage":        smGauge(0.9),
	}
	r, ok := ResolveServerMetric(ServerMetricKVCacheUsage, scraped)
	if !ok || r.Source != "vllm:gpu_cache_usage_perc" || r.Value != 0.5 {
		t.Fatalf("got %+v ok=%v", r, ok)
	}
}

func TestServerMetricMapMissingIsNotZero(t *testing.T) {
	scraped := map[string]ServerMetricSample{"unrelated:metric": smGauge(0)}
	r, ok := ResolveServerMetric(ServerMetricNumPreemptions, scraped)
	if ok {
		t.Fatalf("expected missing, got %+v", r)
	}
	if r != (ServerMetricResolution{}) {
		t.Fatalf("missing resolution must be empty, got %+v", r)
	}
	if _, ok := ResolveServerMetric(ServerMetricNumPreemptions, nil); ok {
		t.Fatal("nil scrape must resolve missing")
	}
	if _, ok := ResolveServerMetric("no_such_neutral", map[string]ServerMetricSample{"vllm:num_preemptions": smCounter(1)}); ok {
		t.Fatal("unknown neutral name must resolve missing")
	}
	all := ResolveServerMetrics(scraped)
	if len(all) != 0 {
		t.Fatalf("expected no resolutions, got %v", all)
	}
}

func TestServerMetricMapPresentZeroStillWins(t *testing.T) {
	scraped := map[string]ServerMetricSample{
		"vllm:num_preemptions":      smCounter(0),
		"sglang:num_retracted_reqs": smCounter(5),
	}
	r, ok := ResolveServerMetric(ServerMetricNumPreemptions, scraped)
	if !ok || r.Source != "vllm:num_preemptions" || r.Value != 0 {
		t.Fatalf("got %+v ok=%v, want present zero from vLLM", r, ok)
	}
}

func TestServerMetricMapRetractedGaugeIsNotPreemptionCounter(t *testing.T) {
	gaugeOnly := map[string]ServerMetricSample{"sglang:num_retracted_reqs": smGauge(9)}
	if r, ok := ResolveServerMetric(ServerMetricNumPreemptions, gaugeOnly); ok {
		t.Fatalf("gauge-typed num_retracted_reqs must not resolve preemptions, got %+v", r)
	}
	counterOnly := map[string]ServerMetricSample{"sglang:num_retracted_reqs": smCounter(9)}
	r, ok := ResolveServerMetric(ServerMetricNumPreemptions, counterOnly)
	if !ok || r.Source != "sglang:num_retracted_reqs" || r.Value != 9 {
		t.Fatalf("counter-typed num_retracted_reqs: got %+v ok=%v", r, ok)
	}
}

func TestServerMetricMapKindMismatchFallsThrough(t *testing.T) {
	scraped := map[string]ServerMetricSample{
		"vllm:num_requests_running": smCounter(7),
		"sglang:num_running_reqs":   smGauge(3),
	}
	r, ok := ResolveServerMetric(ServerMetricNumRunning, scraped)
	if !ok || r.Source != "sglang:num_running_reqs" || r.Value != 3 {
		t.Fatalf("got %+v ok=%v, want SGLang gauge after counter-typed vLLM entry is skipped", r, ok)
	}
	counterOnly := map[string]ServerMetricSample{"vllm:num_requests_running": smCounter(7)}
	if r, ok := ResolveServerMetric(ServerMetricNumRunning, counterOnly); ok {
		t.Fatalf("gauge chain must reject a counter-typed sample, got %+v", r)
	}
}

func TestServerMetricMapResolveAllMixesBackendsPerChain(t *testing.T) {
	scraped := map[string]ServerMetricSample{
		"vllm:prompt_tokens":        smCounter(100),
		"sglang:generation_tokens":  smCounter(40),
		"sglang:num_running_reqs":   smGauge(2),
		"vllm:num_requests_waiting": smGauge(0),
	}
	got := ResolveServerMetrics(scraped)
	want := map[string]string{
		ServerMetricPromptTokens:     "vllm:prompt_tokens",
		ServerMetricGenerationTokens: "sglang:generation_tokens",
		ServerMetricNumRunning:       "sglang:num_running_reqs",
		ServerMetricNumWaiting:       "vllm:num_requests_waiting",
	}
	if len(got) != len(want) {
		t.Fatalf("got %d resolutions %v, want %d", len(got), got, len(want))
	}
	for neutral, src := range want {
		if got[neutral].Source != src {
			t.Errorf("%s: source %q, want %q", neutral, got[neutral].Source, src)
		}
	}
}

func TestServerMetricMapChainTable(t *testing.T) {
	chains := ServerMetricChains()
	if len(chains) == 0 {
		t.Fatal("no chains")
	}
	neutrals := map[string]bool{}
	backendNames := map[string]string{}
	for _, c := range chains {
		if c.Neutral == "" {
			t.Fatal("chain with empty neutral name")
		}
		if neutrals[c.Neutral] {
			t.Fatalf("duplicate neutral name %q", c.Neutral)
		}
		neutrals[c.Neutral] = true
		if len(c.Names) == 0 {
			t.Fatalf("%s: empty chain", c.Neutral)
		}
		if !strings.HasPrefix(c.Names[0], "vllm:") {
			t.Errorf("%s: first entry %q is not a vLLM name", c.Neutral, c.Names[0])
		}
		seenSGLang := false
		for _, n := range c.Names {
			if prev, dup := backendNames[n]; dup {
				t.Errorf("backend name %q appears in %s and %s", n, prev, c.Neutral)
			}
			backendNames[n] = c.Neutral
			switch {
			case strings.HasPrefix(n, "sglang:"):
				seenSGLang = true
			case strings.HasPrefix(n, "vllm:"):
				if seenSGLang {
					t.Errorf("%s: vLLM name %q after an SGLang name", c.Neutral, n)
				}
			default:
				t.Errorf("%s: unexpected backend prefix in %q", c.Neutral, n)
			}
			if c.Kind == ServerMetricCounter && strings.HasSuffix(n, "_total") {
				t.Errorf("%s: counter name %q must use the family form without _total", c.Neutral, n)
			}
		}
		if !seenSGLang {
			t.Errorf("%s: no SGLang fallback", c.Neutral)
		}
	}
}

func TestServerMetricMapChainsReturnsCopy(t *testing.T) {
	first := ServerMetricChains()
	first[0].Names[0] = "mutated"
	if ServerMetricChains()[0].Names[0] == "mutated" {
		t.Fatal("ServerMetricChains leaked the internal table")
	}
}
