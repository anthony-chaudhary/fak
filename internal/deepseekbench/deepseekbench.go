// Package deepseekbench is the pure core of the DeepSeek V4 Pro/Flash
// TTFT/TPOT/context-scaling SCORECARD (#3014, under the DeepSeek V4 support
// program #3006; complements the self-host wire-readiness runbook #3013).
//
// It is the CORE half of a wire+core split: this package owns the locked JSONL
// row schema, the deterministic keyless dry-run fixture, the speedup-refusal
// honesty gate, and the live streaming measurement — all testable in isolation
// (`go test ./internal/deepseekbench`), immune to any concurrent shared-tree WIP
// in cmd/fak. The thin `fak deepseekbench` command in cmd/fak/deepseekbench.go is
// only the flag/I-O wire over these functions.
//
// HONESTY FENCE: this scorecard reports OBSERVED PROVIDER SPEED. A DeepSeek number
// is never a fak-authored saving — the same OBSERVED-vs-WITNESSED discipline
// internal/gateway/deepseek_pricing.go applies to the cache counters. CompareSpeedup
// REFUSES to print a delta unless the two rows share a prompt shape, both carry a
// verified quality parity, and are both live measurements.
package deepseekbench

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Row is ONE scorecard cell. The JSON tags are the locked schema the field-lock
// test (TestRequiredFields) pins — adding/renaming a field without updating
// RequiredFields fails the test on purpose.
type Row struct {
	// Provenance / honesty — read these BEFORE any latency number.
	Measurement     string `json:"measurement"`      // "dry-run-fixture" | "live"
	SpeedProvenance string `json:"speed_provenance"` // "fixture-placeholder-not-measured" | "provider-observed"

	// Route identity.
	ModelID       string `json:"model_id"`
	ProviderRoute string `json:"provider_route"` // e.g. "deepseek" | "anthropic" | "openai-compatible"
	EngineRoute   string `json:"engine_route"`   // e.g. "hosted-api" | "vllm" | "sglang"
	Hosting       string `json:"hosting"`        // "hosted" | "self-hosted"

	// Prompt / output shape.
	ContextBucket string `json:"context_bucket"` // 4K | 32K | 128K | 512K | 1M
	OutputTarget  string `json:"output_target"`  // short | 1K | 8K | long-reasoning
	ReasoningMode string `json:"reasoning_mode"` // non-thinking | high | max
	Stream        bool   `json:"stream"`

	// Latency / throughput.
	TTFTMillis       float64 `json:"ttft_ms"`
	TPOTMillis       float64 `json:"tpot_ms"`
	E2EMillis        float64 `json:"e2e_ms"`
	OutputToksPerSec float64 `json:"output_toks_per_s"`

	// Token counters (reasoning_tokens is 0 when the provider does not expose them).
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	ReasoningTokens  int `json:"reasoning_tokens"`

	// Prompt-cache counters + attribution (provider-observed; never fak-authored).
	PromptCacheHitTokens  int    `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens int    `json:"prompt_cache_miss_tokens"`
	CacheAttribution      string `json:"cache_attribution"` // "provider-observed" | "unknown-dry-run"

	// Comparability keys — the scorecard refuses a speedup unless these line up.
	PromptShapeKey string `json:"prompt_shape"`   // shape descriptor (bucket|output|reasoning|stream)
	QualityParity  string `json:"quality_parity"` // "unknown" | "verified" | "differs"
}

// RequiredFields is the locked set of JSON keys every emitted row MUST carry. The
// field-lock test marshals a row and asserts each key is present (and that the row
// carries no others), so the issue's required-field list can never silently drift.
func RequiredFields() []string {
	return []string{
		"measurement", "speed_provenance",
		"model_id", "provider_route", "engine_route", "hosting",
		"context_bucket", "output_target", "reasoning_mode", "stream",
		"ttft_ms", "tpot_ms", "e2e_ms", "output_toks_per_s",
		"prompt_tokens", "completion_tokens", "reasoning_tokens",
		"prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "cache_attribution",
		"prompt_shape", "quality_parity",
	}
}

// The locked axis vocabularies (the issue's buckets/targets/modes).
var (
	ContextBuckets = []string{"4K", "32K", "128K", "512K", "1M"}
	OutputTargets  = []string{"short", "1K", "8K", "long-reasoning"}
	ReasoningModes = []string{"non-thinking", "high", "max"}
)

// The two current DeepSeek V4 model ids (both, always, side by side — Flash is
// never inferred from Pro).
const (
	ModelV4Pro   = "deepseek-v4-pro"
	ModelV4Flash = "deepseek-v4-flash"
)

// contextBucketTokens maps a bucket label to its nominal prompt-token count.
func contextBucketTokens(bucket string) int {
	switch bucket {
	case "4K":
		return 4096
	case "32K":
		return 32768
	case "128K":
		return 131072
	case "512K":
		return 524288
	case "1M":
		return 1048576
	}
	return 0
}

// outputTargetTokens maps an output target to its nominal completion-token count.
func outputTargetTokens(target string) int {
	switch target {
	case "short":
		return 64
	case "1K":
		return 1024
	case "8K":
		return 8192
	case "long-reasoning":
		return 16384
	}
	return 0
}

// PromptShape is the comparability key: two rows are comparable only when it matches.
func PromptShape(bucket, output, reasoning string, stream bool) string {
	return fmt.Sprintf("%s|%s|%s|stream=%t", bucket, output, reasoning, stream)
}

// benchModel is one model's fixture-latency parameters. These are PLACEHOLDERS used
// only to shape the dry-run fixture; they are NOT measurements and never leave a
// dry-run row (Measurement stays "dry-run-fixture").
type benchModel struct {
	id           string
	prefillMsPer float64 // fixture prefill ms per 1K prompt tokens
	tpotMs       float64 // fixture per-output-token ms
}

// fixtureModels are V4 Pro and V4 Flash — both, always, side by side. The numbers
// reflect only the coarse expectation that the 49B-active Pro decodes slower per
// token than the 13B-active Flash; they are labelled placeholders, not evidence.
func fixtureModels() []benchModel {
	return []benchModel{
		{id: ModelV4Pro, prefillMsPer: 0.18, tpotMs: 22},
		{id: ModelV4Flash, prefillMsPer: 0.09, tpotMs: 9},
	}
}

// DryRunRows builds the deterministic no-key fixture: the full model × bucket ×
// output × reasoning × stream matrix (skipping the contradictory non-thinking +
// long-reasoning cells), plus one existing-fak-route baseline row so the scorecard
// has a same-harness comparand to (correctly) REFUSE comparing against in dry-run.
func DryRunRows() []Row {
	var rows []Row
	for _, m := range fixtureModels() {
		for _, bucket := range ContextBuckets {
			for _, output := range OutputTargets {
				for _, reasoning := range ReasoningModes {
					if output == "long-reasoning" && reasoning == "non-thinking" {
						continue // a non-thinking route does not emit a long reasoning trace
					}
					for _, stream := range []bool{true, false} {
						rows = append(rows, fixtureRow(m, bucket, output, reasoning, stream))
					}
				}
			}
		}
	}
	// One existing-fak-route baseline row, same harness, same schema. It is also a
	// fixture, so the scorecard MUST refuse to compare it (Measurement != live).
	rows = append(rows, Row{
		Measurement:      "dry-run-fixture",
		SpeedProvenance:  "fixture-placeholder-not-measured",
		ModelID:          "claude-sonnet-5",
		ProviderRoute:    "anthropic",
		EngineRoute:      "hosted-api",
		Hosting:          "hosted",
		ContextBucket:    "32K",
		OutputTarget:     "1K",
		ReasoningMode:    "non-thinking",
		Stream:           true,
		CacheAttribution: "unknown-dry-run",
		PromptShapeKey:   PromptShape("32K", "1K", "non-thinking", true),
		QualityParity:    "unknown",
		PromptTokens:     contextBucketTokens("32K"),
		CompletionTokens: outputTargetTokens("1K"),
	})
	return rows
}

// fixtureRow computes one deterministic placeholder row. No time/rand: every number
// is a pure function of the axes, so the fixture is byte-stable across runs.
func fixtureRow(m benchModel, bucket, output, reasoning string, stream bool) Row {
	prompt := contextBucketTokens(bucket)
	completion := outputTargetTokens(output)
	reasoningTokens := 0
	switch reasoning {
	case "high":
		reasoningTokens = completion
	case "max":
		reasoningTokens = completion * 3
	}
	genTokens := completion + reasoningTokens
	ttft := (float64(prompt) / 1000.0) * m.prefillMsPer
	tpot := m.tpotMs
	var e2e, toksPerSec float64
	if stream {
		e2e = ttft + tpot*float64(genTokens)
	} else {
		// Non-stream: no incremental token stream, so TPOT is not observable; the whole
		// generation lands at once and TTFT collapses into E2E.
		tpot = 0
		e2e = ttft + m.tpotMs*float64(genTokens)
		ttft = e2e
	}
	if e2e > 0 {
		toksPerSec = float64(genTokens) / (e2e / 1000.0)
	}
	return Row{
		Measurement:      "dry-run-fixture",
		SpeedProvenance:  "fixture-placeholder-not-measured",
		ModelID:          m.id,
		ProviderRoute:    "deepseek",
		EngineRoute:      "hosted-api",
		Hosting:          "hosted",
		ContextBucket:    bucket,
		OutputTarget:     output,
		ReasoningMode:    reasoning,
		Stream:           stream,
		TTFTMillis:       round2(ttft),
		TPOTMillis:       round2(tpot),
		E2EMillis:        round2(e2e),
		OutputToksPerSec: round2(toksPerSec),
		PromptTokens:     prompt,
		CompletionTokens: completion,
		ReasoningTokens:  reasoningTokens,
		// The fixture asserts nothing about cache behavior — a real hit/miss split is a
		// provider-observed number only a live run may fill.
		PromptCacheHitTokens:  0,
		PromptCacheMissTokens: 0,
		CacheAttribution:      "unknown-dry-run",
		PromptShapeKey:        PromptShape(bucket, output, reasoning, stream),
		QualityParity:         "unknown",
	}
}

func round2(v float64) float64 {
	return float64(int64(v*100+0.5)) / 100
}

// CompareSpeedup is the honesty gate the issue demands: it prints a speedup delta
// ONLY when the DeepSeek row and the baseline row (a) share a prompt shape, (b) both
// carry a verified quality parity, and (c) are both live measurements. Any other case
// returns a "[NOT COMPARABLE: …]" line and printed=false — a dry-run fixture, a shape
// mismatch, or an unverified parity can never surface as a speed headline.
func CompareSpeedup(subject, baseline Row) (line string, printed bool) {
	switch {
	case subject.Measurement != "live" || baseline.Measurement != "live":
		return "[NOT COMPARABLE: dry-run fixture — no measured latency; run --live with a parity check]", false
	case subject.PromptShapeKey != baseline.PromptShapeKey:
		return fmt.Sprintf("[NOT COMPARABLE: prompt shape differs (%s vs %s)]", subject.PromptShapeKey, baseline.PromptShapeKey), false
	case subject.QualityParity != "verified" || baseline.QualityParity != "verified":
		return "[NOT COMPARABLE: quality parity not verified for both rows]", false
	case baseline.E2EMillis <= 0 || subject.E2EMillis <= 0:
		return "[NOT COMPARABLE: a measured E2E is missing]", false
	}
	ratio := baseline.E2EMillis / subject.E2EMillis
	return fmt.Sprintf("OBSERVED provider speed: %s is %.2f× the E2E of %s at shape %s (provider-observed, not a fak-authored saving)",
		subject.ModelID, ratio, baseline.ModelID, subject.PromptShapeKey), true
}

// LiveGate is the pure opt-in guard for a live run: it admits a live measurement
// ONLY when a key is present AND spend was explicitly acknowledged, returning the
// refusal message (and ok=false) otherwise. The wire consults this BEFORE any
// network call, so default CI — no key, no --spend — never touches the network.
func LiveGate(hasKey, spend bool) (msg string, ok bool) {
	if !hasKey {
		return "--live needs DEEPSEEK_API_KEY set (refusing before any network call)", false
	}
	if !spend {
		return "--live costs money — pass --spend to acknowledge (refusing)", false
	}
	return "", true
}

// MeasureStreamed issues one streaming chat/completions request against an
// OpenAI-compatible endpoint (hosted DeepSeek or a self-hosted vLLM/SGLang server)
// and times TTFT (first content/reasoning delta), TPOT (mean inter-delta), E2E, and
// reads the final usage block for the token + prompt-cache counters. The http client
// is injectable so a canned httptest SSE server can witness the parsing/timing logic
// with no key; production passes a real client. A live row — and only a live row —
// carries Measurement="live"/SpeedProvenance="provider-observed".
func MeasureStreamed(client *http.Client, baseURL, key, model string) (Row, error) {
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Minute}
	}
	body, _ := json.Marshal(map[string]any{
		"model":          model,
		"stream":         true,
		"messages":       []map[string]string{{"role": "user", "content": "Reply with a short sentence."}},
		"stream_options": map[string]any{"include_usage": true},
	})
	url := strings.TrimRight(baseURL, "/") + "/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return Row{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return Row{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return Row{}, fmt.Errorf("upstream status %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
	}

	var ttft time.Duration
	var deltas int
	var lastDelta time.Time
	var interSum time.Duration
	prompt, completion, reasoning, hit, miss := 0, 0, 0, 0, 0
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var ev struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens          int `json:"prompt_tokens"`
				CompletionTokens      int `json:"completion_tokens"`
				ReasoningTokens       int `json:"reasoning_tokens"`
				PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
				PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}
		if len(ev.Choices) > 0 && (ev.Choices[0].Delta.Content != "" || ev.Choices[0].Delta.ReasoningContent != "") {
			now := time.Now()
			if ttft == 0 {
				ttft = now.Sub(start)
			} else {
				interSum += now.Sub(lastDelta)
			}
			lastDelta = now
			deltas++
		}
		if ev.Usage != nil {
			prompt = ev.Usage.PromptTokens
			completion = ev.Usage.CompletionTokens
			reasoning = ev.Usage.ReasoningTokens
			hit = ev.Usage.PromptCacheHitTokens
			miss = ev.Usage.PromptCacheMissTokens
		}
	}
	if err := sc.Err(); err != nil {
		return Row{}, err
	}
	e2e := time.Since(start)
	tpot := 0.0
	if deltas > 1 {
		tpot = float64(interSum.Milliseconds()) / float64(deltas-1)
	}
	toksPerSec := 0.0
	if e2e > 0 && completion > 0 {
		toksPerSec = float64(completion) / e2e.Seconds()
	}
	attribution := "unknown"
	if hit > 0 || miss > 0 {
		attribution = "provider-observed"
	}
	return Row{
		Measurement:           "live",
		SpeedProvenance:       "provider-observed",
		ModelID:               model,
		ProviderRoute:         "deepseek",
		EngineRoute:           "hosted-api",
		Hosting:               "hosted",
		ContextBucket:         "4K",
		OutputTarget:          "short",
		ReasoningMode:         "non-thinking",
		Stream:                true,
		TTFTMillis:            round2(float64(ttft.Microseconds()) / 1000.0),
		TPOTMillis:            round2(tpot),
		E2EMillis:             round2(float64(e2e.Microseconds()) / 1000.0),
		OutputToksPerSec:      round2(toksPerSec),
		PromptTokens:          prompt,
		CompletionTokens:      completion,
		ReasoningTokens:       reasoning,
		PromptCacheHitTokens:  hit,
		PromptCacheMissTokens: miss,
		CacheAttribution:      attribution,
		PromptShapeKey:        PromptShape("4K", "short", "non-thinking", true),
		// A single request cannot assert quality parity against another route.
		QualityParity: "unknown",
	}, nil
}
