package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// fak deepseekbench — the DeepSeek V4 Pro/Flash TTFT/TPOT/context-scaling
// SCORECARD (#3014, under the DeepSeek V4 support program #3006; complements the
// self-host wire-readiness runbook #3013).
//
// It emits one JSONL row per (model × context bucket × output target × reasoning
// mode × stream/non-stream) cell with the performance axes the issue locks: TTFT,
// TPOT, E2E, output tok/s, prompt/completion/reasoning tokens, and the provider's
// prompt-cache hit/miss counters. V4 Pro and V4 Flash are BOTH in the same
// scorecard — Flash behavior is never inferred from Pro.
//
// TWO modes, one schema:
//
//	fak deepseekbench                 DEFAULT: a no-key DRY-RUN FIXTURE. Every latency
//	                                  number is a labelled PLACEHOLDER (measurement=
//	                                  "dry-run-fixture"), NOT a measurement — the fixture
//	                                  exists to lock the row schema and prove the harness
//	                                  runs keyless in CI, exactly like the #3013 smoke
//	                                  that skips cleanly with no upstream.
//	fak deepseekbench --live --spend  OPT-IN LIVE run. Refuses unless DEEPSEEK_API_KEY is
//	                                  set AND --spend is passed (an explicit money ack).
//	                                  Only a live row carries measurement="live" and
//	                                  speed_provenance="provider-observed".
//
// HONESTY FENCE: this scorecard reports OBSERVED PROVIDER SPEED. A DeepSeek number is
// never a fak-authored saving — the split is the same OBSERVED-vs-WITNESSED discipline
// internal/gateway/deepseek_pricing.go applies to the cache counters. The scorecard's
// speedup line REFUSES to print a delta unless the two rows share a prompt shape and a
// verified quality parity and are both live measurements (see compareSpeedup).
//
//	fak deepseekbench --out rows.jsonl        write the JSONL rows to a file
//	fak deepseekbench --base-url URL --model M  point the live run at a self-hosted
//	                                            OpenAI-compatible endpoint (default hosted)
func cmdDeepSeekBench(argv []string) { os.Exit(runDeepSeekBench(os.Stdout, os.Stderr, argv)) }

// DeepSeekBenchRow is ONE scorecard cell. The JSON tags are the locked schema the
// field-lock test (TestDeepSeekBenchRequiredFields) pins — adding/renaming a field
// without updating requiredBenchFields fails the test on purpose.
type DeepSeekBenchRow struct {
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
	PromptShape   string `json:"prompt_shape"`   // shape descriptor (bucket|output|reasoning|stream)
	QualityParity string `json:"quality_parity"` // "unknown" | "verified" | "differs"
}

// requiredBenchFields is the locked set of JSON keys every emitted row MUST carry.
// The field-lock test marshals a row and asserts each key is present, so the issue's
// required-field list can never silently drift.
func requiredBenchFields() []string {
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
	benchContextBuckets = []string{"4K", "32K", "128K", "512K", "1M"}
	benchOutputTargets  = []string{"short", "1K", "8K", "long-reasoning"}
	benchReasoningModes = []string{"non-thinking", "high", "max"}
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

// promptShape is the comparability key: two rows are comparable only when it matches.
func promptShape(bucket, output, reasoning string, stream bool) string {
	return fmt.Sprintf("%s|%s|%s|stream=%t", bucket, output, reasoning, stream)
}

// benchModel is one model's fixture-latency parameters. These are PLACEHOLDERS used
// only to shape the dry-run fixture; they are NOT measurements and never leave a
// dry-run row (measurement stays "dry-run-fixture").
type benchModel struct {
	id           string
	prefillMsPer float64 // fixture prefill ms per 1K prompt tokens
	tpotMs       float64 // fixture per-output-token ms
}

// deepseekFixtureModels are V4 Pro and V4 Flash — both, always, side by side. The
// numbers reflect only the coarse expectation that the 49B-active Pro decodes slower
// per token than the 13B-active Flash; they are labelled placeholders, not evidence.
func deepseekFixtureModels() []benchModel {
	return []benchModel{
		{id: "deepseek-v4-pro", prefillMsPer: 0.18, tpotMs: 22},
		{id: "deepseek-v4-flash", prefillMsPer: 0.09, tpotMs: 9},
	}
}

// dryRunRows builds the deterministic no-key fixture: the full model × bucket ×
// output × reasoning × stream matrix (skipping the contradictory non-thinking +
// long-reasoning cells), plus one fak-route baseline row so the scorecard has a
// same-harness comparand to (correctly) REFUSE comparing against in dry-run.
func dryRunRows() []DeepSeekBenchRow {
	var rows []DeepSeekBenchRow
	for _, m := range deepseekFixtureModels() {
		for _, bucket := range benchContextBuckets {
			for _, output := range benchOutputTargets {
				for _, reasoning := range benchReasoningModes {
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
	// fixture, so the scorecard MUST refuse to compare it (measurement != live).
	rows = append(rows, DeepSeekBenchRow{
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
		PromptShape:      promptShape("32K", "1K", "non-thinking", true),
		QualityParity:    "unknown",
		PromptTokens:     contextBucketTokens("32K"),
		CompletionTokens: outputTargetTokens("1K"),
	})
	return rows
}

// fixtureRow computes one deterministic placeholder row. No time/rand: every number
// is a pure function of the axes, so the fixture is byte-stable across runs.
func fixtureRow(m benchModel, bucket, output, reasoning string, stream bool) DeepSeekBenchRow {
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
	return DeepSeekBenchRow{
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
		PromptShape:           promptShape(bucket, output, reasoning, stream),
		QualityParity:         "unknown",
	}
}

func round2(v float64) float64 {
	return float64(int64(v*100+0.5)) / 100
}

// compareSpeedup is the honesty gate the issue demands: it prints a speedup delta
// ONLY when the DeepSeek row and the baseline row (a) share a prompt shape, (b) both
// carry a verified quality parity, and (c) are both live measurements. Any other case
// returns a "[NOT COMPARABLE: …]" line and printed=false — a dry-run fixture, a shape
// mismatch, or an unverified parity can never surface as a speed headline.
func compareSpeedup(subject, baseline DeepSeekBenchRow) (line string, printed bool) {
	switch {
	case subject.Measurement != "live" || baseline.Measurement != "live":
		return "[NOT COMPARABLE: dry-run fixture — no measured latency; run --live with a parity check]", false
	case subject.PromptShape != baseline.PromptShape:
		return fmt.Sprintf("[NOT COMPARABLE: prompt shape differs (%s vs %s)]", subject.PromptShape, baseline.PromptShape), false
	case subject.QualityParity != "verified" || baseline.QualityParity != "verified":
		return "[NOT COMPARABLE: quality parity not verified for both rows]", false
	case baseline.E2EMillis <= 0 || subject.E2EMillis <= 0:
		return "[NOT COMPARABLE: a measured E2E is missing]", false
	}
	ratio := baseline.E2EMillis / subject.E2EMillis
	return fmt.Sprintf("OBSERVED provider speed: %s is %.2f× the E2E of %s at shape %s (provider-observed, not a fak-authored saving)",
		subject.ModelID, ratio, baseline.ModelID, subject.PromptShape), true
}

// runDeepSeekBench is the testable core (exit code, no os.Exit): 0 ok, 1 a run error,
// 2 a usage/gating error.
func runDeepSeekBench(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("deepseekbench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	live := fs.Bool("live", false, "opt-in LIVE run (requires DEEPSEEK_API_KEY and --spend); default is a no-key dry-run fixture")
	spend := fs.Bool("spend", false, "explicit acknowledgement that a --live run costs money")
	baseURL := fs.String("base-url", "https://api.deepseek.com", "OpenAI-compatible root for a --live run (point at a self-hosted vLLM/SGLang endpoint for the self-hosted arm)")
	model := fs.String("model", "deepseek-v4-flash", "model id to route for a --live run")
	outPath := fs.String("out", "", "write the JSONL rows to this file (default: stdout)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	var rows []DeepSeekBenchRow
	if *live {
		got, rc := liveRows(stderr, *baseURL, *model, *spend)
		if rc != 0 {
			return rc
		}
		rows = got
	} else {
		rows = dryRunRows()
	}

	sink := stdout
	if *outPath != "" {
		f, err := os.Create(*outPath)
		if err != nil {
			fmt.Fprintln(stderr, "fak deepseekbench:", err)
			return 1
		}
		defer f.Close()
		sink = f
	}
	if err := writeBenchJSONL(sink, rows); err != nil {
		fmt.Fprintln(stderr, "fak deepseekbench:", err)
		return 1
	}

	printBenchSummary(stderr, rows, *live)
	return 0
}

// writeBenchJSONL emits one compact JSON object per line — the locked scorecard format.
func writeBenchJSONL(w io.Writer, rows []DeepSeekBenchRow) error {
	bw := bufio.NewWriter(w)
	enc := json.NewEncoder(bw)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return bw.Flush()
}

// printBenchSummary writes the human-facing coverage + honesty summary to stderr (the
// JSONL rows themselves are the machine artifact on stdout/--out).
func printBenchSummary(w io.Writer, rows []DeepSeekBenchRow, live bool) {
	models := map[string]int{}
	for _, r := range rows {
		models[r.ModelID]++
	}
	ids := make([]string, 0, len(models))
	for id := range models {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	fmt.Fprintln(w, "== fak deepseekbench — DeepSeek V4 Pro/Flash TTFT/TPOT/context scorecard (#3014) ==")
	if live {
		fmt.Fprintln(w, "mode: LIVE (measurement=\"live\"; latency is provider-observed)")
	} else {
		fmt.Fprintln(w, "mode: DRY-RUN FIXTURE (measurement=\"dry-run-fixture\"; every latency is a labelled placeholder, NOT measured)")
	}
	fmt.Fprintf(w, "rows: %d across models:\n", len(rows))
	for _, id := range ids {
		fmt.Fprintf(w, "  %-22s %d rows\n", id, models[id])
	}
	// Demonstrate the speedup-refusal gate on the first two DeepSeek/baseline rows.
	line, _ := scorecardExample(rows)
	fmt.Fprintf(w, "scorecard speedup gate: %s\n", line)
	fmt.Fprintln(w, "OBSERVED provider speed is never a fak-authored saving — see docs/benchmarks/DEEPSEEK-V4-PERF-SCORECARD.md")
	if !live {
		fmt.Fprintln(w, "live recipe: DEEPSEEK_API_KEY=… fak deepseekbench --live --spend --model deepseek-v4-pro")
	}
}

// scorecardExample picks a DeepSeek subject row and a non-DeepSeek baseline row (if
// present) and runs the refusal gate over them, so the summary always shows what the
// gate decides on the current rows.
func scorecardExample(rows []DeepSeekBenchRow) (string, bool) {
	var subject, baseline *DeepSeekBenchRow
	for i := range rows {
		if rows[i].ProviderRoute == "deepseek" && subject == nil {
			subject = &rows[i]
		}
		if rows[i].ProviderRoute != "deepseek" && baseline == nil {
			baseline = &rows[i]
		}
	}
	if subject == nil || baseline == nil {
		return "[NOT COMPARABLE: need one DeepSeek row and one baseline row]", false
	}
	return compareSpeedup(*subject, *baseline)
}

// --- live measurement arm (opt-in, gated, unwitnessed in CI) --------------------

// liveRows gates the live arm and, when admitted, measures one streamed request. It
// returns (rows, exitCode); a non-zero code means the gate refused BEFORE any network.
func liveRows(stderr io.Writer, baseURL, model string, spend bool) ([]DeepSeekBenchRow, int) {
	key := strings.TrimSpace(os.Getenv("DEEPSEEK_API_KEY"))
	if key == "" {
		fmt.Fprintln(stderr, "fak deepseekbench: --live needs DEEPSEEK_API_KEY set (refusing before any network call)")
		return nil, 2
	}
	if !spend {
		fmt.Fprintln(stderr, "fak deepseekbench: --live costs money — pass --spend to acknowledge (refusing)")
		return nil, 2
	}
	row, err := measureStreamed(baseURL, key, model)
	if err != nil {
		fmt.Fprintln(stderr, "fak deepseekbench: live run failed:", err)
		return nil, 1
	}
	return []DeepSeekBenchRow{row}, 0
}

// measureStreamed issues one streaming chat/completions request and times TTFT (first
// content delta), TPOT (mean inter-delta), E2E, and reads the final usage block for the
// token + prompt-cache counters. It is the untested-in-CI arm: it only runs behind the
// key+spend gate, mirroring the #3013 self-host smoke that skips cleanly without a node.
func measureStreamed(baseURL, key, model string) (DeepSeekBenchRow, error) {
	body, _ := json.Marshal(map[string]any{
		"model":    model,
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "Reply with a short sentence."}},
		"stream_options": map[string]any{"include_usage": true},
	})
	url := strings.TrimRight(baseURL, "/") + "/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return DeepSeekBenchRow{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)

	client := &http.Client{Timeout: 5 * time.Minute}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return DeepSeekBenchRow{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return DeepSeekBenchRow{}, fmt.Errorf("upstream status %d: %s", resp.StatusCode, strings.TrimSpace(string(snippet)))
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
		return DeepSeekBenchRow{}, err
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
	return DeepSeekBenchRow{
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
		PromptShape:           promptShape("4K", "short", "non-thinking", true),
		// A single request cannot assert quality parity against another route.
		QualityParity: "unknown",
	}, nil
}
