// Package qwen38quantrun executes the frozen Qwen3.8 quantization corpus
// against an OpenAI-compatible endpoint and independently grades effects.
package qwen38quantrun

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

type Config struct {
	Endpoint, APIKey, Model string
	Repetitions             int
	Timeout                 time.Duration
	// NativeDecodeTrace requests the buffered fak-native token-commit receipt.
	// It never enables streaming or infers tokens from response fragments.
	NativeDecodeTrace bool
	// NativeInferenceReceipt requests the deterministic native token IDs and
	// logprobs/logits receipt from the gateway.
	NativeInferenceReceipt bool
	BeforeTrial            func(context.Context, qwen38quant.Fixture, int) error
	Sample                 func(context.Context) (ResourceSample, error)
}

const LongDecodeWorkload = "long_output_decode"

type ResourceSample struct {
	MemoryBytes uint64  `json:"memory_bytes,omitempty"`
	PowerWatts  float64 `json:"power_watts,omitempty"`
}

type Result struct {
	FixtureID              string                        `json:"fixture_id"`
	Workload               string                        `json:"workload"`
	Repeat                 int                           `json:"repeat"`
	LatencyMS              float64                       `json:"latency_ms"`
	Usage                  map[string]int                `json:"usage,omitempty"`
	Quality                string                        `json:"quality"`
	Failure                string                        `json:"failure,omitempty"`
	Output                 string                        `json:"output,omitempty"`
	ToolName               string                        `json:"tool_name,omitempty"`
	ToolArgs               map[string]any                `json:"tool_args,omitempty"`
	Phase                  string                        `json:"phase,omitempty"`
	CachedInputTokens      int                           `json:"cached_input_tokens,omitempty"`
	Resource               ResourceSample                `json:"resource,omitempty"`
	DecodeTrace            *DecodeTrace                  `json:"decode_trace,omitempty"`
	DecodeWindows          *DecodeWindowReport           `json:"decode_windows,omitempty"`
	NativeInferenceReceipt *model.NativeInferenceReceipt `json:"native_inference_receipt,omitempty"`
}

type Runner struct{ Client *http.Client }

type chatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Function struct{ Name, Arguments string } `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
	Usage        map[string]int `json:"usage"`
	UsageDetails struct {
		CachedTokens int
	}
	Fak *struct {
		DecodeTrace            *DecodeTrace                  `json:"decode_trace,omitempty"`
		NativeInferenceReceipt *model.NativeInferenceReceipt `json:"native_inference_receipt,omitempty"`
	} `json:"fak,omitempty"`
	DecodeWindows *DecodeWindowReport `json:"-"`
}

func (r *chatResponse) UnmarshalJSON(data []byte) error {
	type wireResponse chatResponse
	var wire struct {
		wireResponse
		Usage json.RawMessage `json:"usage"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*r = chatResponse(wire.wireResponse)
	if len(wire.Usage) == 0 || bytes.Equal(wire.Usage, []byte("null")) {
		return nil
	}
	var usage map[string]json.RawMessage
	if err := json.Unmarshal(wire.Usage, &usage); err != nil {
		return fmt.Errorf("usage: %w", err)
	}
	r.Usage = make(map[string]int)
	for _, name := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
		if raw := usage[name]; len(raw) != 0 {
			var value int
			if err := json.Unmarshal(raw, &value); err != nil {
				return fmt.Errorf("usage.%s: %w", name, err)
			}
			r.Usage[name] = value
		}
	}
	if raw := usage["prompt_tokens_details"]; len(raw) != 0 && !bytes.Equal(raw, []byte("null")) {
		var details struct {
			CachedTokens int `json:"cached_tokens"`
		}
		if err := json.Unmarshal(raw, &details); err != nil {
			return fmt.Errorf("usage.prompt_tokens_details: %w", err)
		}
		r.UsageDetails.CachedTokens = details.CachedTokens
	}
	return nil
}

func (r Runner) Run(ctx context.Context, cfg Config, corpus qwen38quant.Corpus) ([]Result, error) {
	if err := corpus.Validate(); err != nil {
		return nil, err
	}
	if cfg.Endpoint == "" || cfg.APIKey == "" || cfg.Model == "" {
		return nil, fmt.Errorf("endpoint, API key, and exact model are required")
	}
	reps := cfg.Repetitions
	if reps == 0 {
		reps = corpus.MinimumRepetitions
	}
	if reps < corpus.MinimumRepetitions {
		return nil, fmt.Errorf("repetitions %d below corpus minimum", reps)
	}
	client := r.Client
	if client == nil {
		client = &http.Client{Timeout: cfg.Timeout}
		if client.Timeout == 0 {
			client.Timeout = 10 * time.Minute
		}
	}
	if err := preflightModel(ctx, client, cfg); err != nil {
		return nil, err
	}
	return r.runFixtures(ctx, client, cfg, corpus.Fixtures, reps)
}

// RunLongDecode runs the issue-8385 measurement fixture exactly three times
// without adding a benchmark-only workload to the frozen quality corpus.
func (r Runner) RunLongDecode(ctx context.Context, cfg Config, fixture qwen38quant.Fixture) ([]Result, error) {
	if cfg.Endpoint == "" || cfg.APIKey == "" || cfg.Model == "" {
		return nil, fmt.Errorf("endpoint, API key, and exact model are required")
	}
	if fixture.ID == "" || fixture.Prompt == "" || fixture.Workload != LongDecodeWorkload || fixture.MaxOutputTokens < MinimumLongDecodeTokens {
		return nil, fmt.Errorf("long decode fixture requires identity, prompt, workload %q, and at least %d max output tokens", LongDecodeWorkload, MinimumLongDecodeTokens)
	}
	if cfg.Repetitions != 0 && cfg.Repetitions != MinimumDecodeRepetitions {
		return nil, fmt.Errorf("long decode campaign requires exactly %d repetitions", MinimumDecodeRepetitions)
	}
	cfg.Repetitions = MinimumDecodeRepetitions
	cfg.NativeDecodeTrace = true
	client := r.Client
	if client == nil {
		client = &http.Client{Timeout: cfg.Timeout}
		if client.Timeout == 0 {
			client.Timeout = 10 * time.Minute
		}
	}
	if err := preflightModel(ctx, client, cfg); err != nil {
		return nil, err
	}
	return r.runFixtures(ctx, client, cfg, []qwen38quant.Fixture{fixture}, MinimumDecodeRepetitions)
}

func (r Runner) runFixtures(ctx context.Context, client *http.Client, cfg Config, fixtures []qwen38quant.Fixture, reps int) ([]Result, error) {
	var out []Result
	for fixtureIndex, f := range fixtures {
		for n := 1; n <= reps; n++ {
			res := Result{FixtureID: f.ID, Workload: f.Workload, Repeat: n, Quality: "FAIL"}
			if f.Workload == "repeated_workflow_cache" {
				res.Phase = cachePhase(f, n)
			}
			if cfg.BeforeTrial != nil {
				if err := cfg.BeforeTrial(ctx, f, n); err != nil {
					res.Failure = "before trial: " + err.Error()
					out = append(out, res)
					continue
				}
			}
			start := time.Now()
			resp, err := runOne(ctx, client, cfg, f)
			res.LatencyMS = float64(time.Since(start).Nanoseconds()) / float64(time.Millisecond)
			if err != nil {
				res.Failure = err.Error()
				out = append(out, res)
				if fixtureIndex == 0 && n == 1 {
					return nil, fmt.Errorf("API canary %s failed: %w; fix the serving contract before running the campaign matrix", f.ID, err)
				}
				continue
			}
			res.Usage = resp.Usage
			res.CachedInputTokens = resp.UsageDetails.CachedTokens
			if cfg.NativeDecodeTrace {
				trace := *resp.Fak.DecodeTrace
				trace.Events = append([]DecodeTraceEvent(nil), trace.Events...)
				report := *resp.DecodeWindows
				report.Windows = append([]DecodeWindow(nil), report.Windows...)
				res.DecodeTrace, res.DecodeWindows = &trace, &report
			}
			if cfg.NativeInferenceReceipt && resp.Fak != nil && resp.Fak.NativeInferenceReceipt != nil {
				receipt := *resp.Fak.NativeInferenceReceipt
				receipt.TokenIDs = append([]int(nil), receipt.TokenIDs...)
				receipt.TokenLogprobs = append([]float64(nil), receipt.TokenLogprobs...)
				res.NativeInferenceReceipt = &receipt
			}
			if cfg.Sample != nil {
				if sample, sampleErr := cfg.Sample(ctx); sampleErr == nil {
					res.Resource = sample
				} else {
					res.Failure = "resource sample: " + sampleErr.Error()
					out = append(out, res)
					continue
				}
			}
			if f.Workload == LongDecodeWorkload {
				gradeLongDecode(&res)
			} else {
				grade(&res, f, resp)
			}
			out = append(out, res)
			if fixtureIndex == 0 && n == 1 && res.Usage["completion_tokens"] == 0 {
				return nil, fmt.Errorf("API canary %s produced zero completion tokens; fix the serving contract before running the campaign matrix", f.ID)
			}
		}
	}
	return out, nil
}

func gradeLongDecode(result *Result) {
	if result.Usage["completion_tokens"] < MinimumLongDecodeTokens {
		result.Quality = "FAIL"
		result.Failure = fmt.Sprintf("completion_tokens=%d below %d", result.Usage["completion_tokens"], MinimumLongDecodeTokens)
		return
	}
	if result.DecodeTrace == nil || result.DecodeWindows == nil {
		result.Quality = "FAIL"
		result.Failure = "missing native decode trace"
		return
	}
	result.Quality = "PASS"
	result.Failure = ""
}

func preflightModel(ctx context.Context, client *http.Client, cfg Config) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(cfg.Endpoint, "/")+"/v1/models", nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	rsp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("model identity preflight: %w", err)
	}
	defer rsp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(rsp.Body, 1<<20))
	if err != nil {
		return err
	}
	if rsp.StatusCode/100 != 2 {
		return fmt.Errorf("model identity preflight HTTP %d", rsp.StatusCode)
	}
	var listing struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &listing); err != nil {
		return fmt.Errorf("model identity preflight: %w", err)
	}
	matches := 0
	for _, model := range listing.Data {
		if model.ID == cfg.Model {
			matches++
		}
	}
	if matches != 1 {
		return fmt.Errorf("model identity preflight: exact model %q occurred %d times", cfg.Model, matches)
	}
	return nil
}

func cachePhase(f qwen38quant.Fixture, repetition int) string {
	if repetition <= len(f.CacheSequence) {
		return f.CacheSequence[repetition-1]
	}
	return "warm"
}

func runOne(ctx context.Context, client *http.Client, cfg Config, f qwen38quant.Fixture) (chatResponse, error) {
	prompt := materialize(f)
	body := map[string]any{"model": cfg.Model, "messages": []map[string]string{{"role": "user", "content": prompt}}, "temperature": 0, "max_tokens": f.MaxOutputTokens, "chat_template_kwargs": map[string]bool{"enable_thinking": false}}
	if cfg.NativeDecodeTrace {
		body["fak_decode_trace"] = true
	}
	if cfg.NativeInferenceReceipt {
		body["fak"] = map[string]any{"native_inference_receipt": true}
	}
	if f.Workload == "json_schema" {
		body["response_format"] = map[string]any{
			"type": "json_schema",
			"json_schema": map[string]any{
				"name":   f.ID,
				"strict": true,
				"schema": jsonSchemaFor(f.ExpectedJSON),
			},
		}
	}
	if len(f.Tools) > 0 {
		body["tools"] = f.Tools
		body["tool_choice"] = "required"
	}
	b, _ := json.Marshal(body)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.Endpoint, "/")+"/v1/chat/completions", bytes.NewReader(b))
	if err != nil {
		return chatResponse{}, err
	}
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	rsp, err := client.Do(req)
	if err != nil {
		return chatResponse{}, err
	}
	defer rsp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(rsp.Body, 4<<20))
	if err != nil {
		return chatResponse{}, err
	}
	if rsp.StatusCode/100 != 2 {
		return chatResponse{}, fmt.Errorf("HTTP %d: %s", rsp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var out chatResponse
	if err = json.Unmarshal(raw, &out); err != nil {
		return out, err
	}
	if out.Model != "" && out.Model != cfg.Model {
		return out, fmt.Errorf("response model mismatch: got %q want %q", out.Model, cfg.Model)
	}
	if len(out.Choices) == 0 {
		return out, fmt.Errorf("response has no choices")
	}
	if cfg.NativeDecodeTrace {
		if out.Fak == nil || out.Fak.DecodeTrace == nil {
			return out, fmt.Errorf("response missing fak.decode_trace")
		}
		if got := out.Fak.DecodeTrace.Provenance; got != "" && got != NativeDecodeTraceProvenance {
			return out, fmt.Errorf("decode trace provenance mismatch: got %q want %q", got, NativeDecodeTraceProvenance)
		}
		out.Fak.DecodeTrace.Provenance = NativeDecodeTraceProvenance
		report, traceErr := BuildDecodeWindows(*out.Fak.DecodeTrace, NativeDecodeContract, out.Usage["completion_tokens"])
		if traceErr != nil {
			return out, traceErr
		}
		out.DecodeWindows = &report
	}
	if cfg.NativeInferenceReceipt {
		if out.Fak == nil || out.Fak.NativeInferenceReceipt == nil {
			return out, fmt.Errorf("response missing fak.native_inference_receipt")
		}
	}
	return out, nil
}

func jsonSchemaFor(value any) map[string]any {
	switch value := value.(type) {
	case map[string]any:
		properties := make(map[string]any, len(value))
		required := make([]string, 0, len(value))
		for name, child := range value {
			properties[name] = jsonSchemaFor(child)
			required = append(required, name)
		}
		slices.Sort(required)
		return map[string]any{"type": "object", "properties": properties, "required": required, "additionalProperties": false}
	case []any:
		items := map[string]any{}
		if len(value) > 0 {
			items = jsonSchemaFor(value[0])
		}
		return map[string]any{"type": "array", "items": items}
	case string:
		return map[string]any{"type": "string"}
	case bool:
		return map[string]any{"type": "boolean"}
	case float64, float32, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		return map[string]any{"type": "number"}
	case nil:
		return map[string]any{"type": "null"}
	default:
		return map[string]any{}
	}
}
func grade(res *Result, f qwen38quant.Fixture, r chatResponse) {
	m := r.Choices[0].Message
	res.Output = strings.TrimSpace(m.Content)
	switch f.Workload {
	case "json_schema":
		var got any
		if json.Unmarshal([]byte(res.Output), &got) != nil {
			res.Failure = "invalid JSON"
			return
		}
		if !reflect.DeepEqual(got, any(f.ExpectedJSON)) {
			res.Failure = "JSON effect mismatch"
			return
		}
	case "correlated_tools":
		if len(m.ToolCalls) != 1 {
			res.Failure = "expected exactly one tool call"
			return
		}
		res.ToolName = m.ToolCalls[0].Function.Name
		if json.Unmarshal([]byte(m.ToolCalls[0].Function.Arguments), &res.ToolArgs) != nil {
			res.Failure = "invalid tool arguments"
			return
		}
		wantName, _ := f.ExpectedTool["name"].(string)
		wantArgs, _ := f.ExpectedTool["arguments"].(map[string]any)
		wantJSON, _ := json.Marshal(wantArgs)
		wantArgs = map[string]any{}
		_ = json.Unmarshal(wantJSON, &wantArgs)
		if res.ToolName != wantName || !reflect.DeepEqual(res.ToolArgs, wantArgs) {
			res.Failure = "tool effect mismatch"
			return
		}
	default:
		if res.Output != f.ExpectedExact {
			res.Failure = fmt.Sprintf("exact effect mismatch: got %q", res.Output)
			return
		}
	}
	res.Quality = "PASS"
}

func materialize(f qwen38quant.Fixture) string {
	if f.Workload != "long_context_retrieval" {
		return f.Prompt
	}
	records, _ := number(f.Generator["records"])
	needleRecord, _ := number(f.Generator["needle_record"])
	needle, _ := f.Generator["needle"].(string)
	format, _ := f.Generator["filler"].(string)
	var b strings.Builder
	b.WriteString(f.Prompt)
	b.WriteByte('\n')
	for i := 1; i <= records; i++ {
		if i == needleRecord {
			fmt.Fprintf(&b, "record-%04d: secret %s\n", i, needle)
		} else {
			fmt.Fprintf(&b, format+"\n", i, i)
		}
	}
	return b.String()
}
func number(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), n == float64(int(n))
	case int:
		return n, true
	default:
		return 0, false
	}
}
