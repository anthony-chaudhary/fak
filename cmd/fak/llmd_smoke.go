package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/engine"
)

func cmdLLMDSmoke(argv []string) { os.Exit(runLLMDSmoke(os.Stdout, os.Stderr, argv)) }

type llmdSmokeOptions struct {
	BaseURL    string
	Model      string
	APIKeyEnv  string
	MetricsURL string
	Prompt     string
	Timeout    time.Duration
	JSON       bool
}

type llmdSmokeReport struct {
	OK         bool             `json:"ok"`
	Engine     string           `json:"engine"`
	BaseURL    string           `json:"base_url"`
	Model      string           `json:"model"`
	Models     llmdSmokeModels  `json:"models"`
	Chat       llmdSmokeChat    `json:"chat"`
	Metrics    llmdSmokeMetrics `json:"metrics"`
	DurationMS int64            `json:"duration_ms"`
	Error      string           `json:"error,omitempty"`
}

type llmdSmokeModels struct {
	OK       bool     `json:"ok"`
	Endpoint string   `json:"endpoint"`
	Status   int      `json:"status"`
	Count    int      `json:"count"`
	IDs      []string `json:"ids,omitempty"`
}

type llmdSmokeChat struct {
	OK            bool            `json:"ok"`
	Endpoint      string          `json:"endpoint"`
	Status        int             `json:"status"`
	DataEvents    int             `json:"data_events"`
	ContentChunks int             `json:"content_chunks"`
	ContentChars  int             `json:"content_chars"`
	Done          bool            `json:"done"`
	Model         string          `json:"model,omitempty"`
	Usage         *llmdSmokeUsage `json:"usage,omitempty"`
}

type llmdSmokeUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type llmdSmokeMetrics struct {
	Checked          bool    `json:"checked"`
	OK               bool    `json:"ok"`
	Endpoint         string  `json:"endpoint,omitempty"`
	Status           int     `json:"status,omitempty"`
	Engine           string  `json:"engine,omitempty"`
	WorkerID         string  `json:"worker_id,omitempty"`
	RequestsRunning  float64 `json:"requests_running,omitempty"`
	RequestsWaiting  float64 `json:"requests_waiting,omitempty"`
	RequestSuccesses float64 `json:"request_successes,omitempty"`
	SkippedReason    string  `json:"skipped_reason,omitempty"`
}

func runLLMDSmoke(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("llmd-smoke", flag.ContinueOnError)
	fs.SetOutput(stderr)
	baseURL := fs.String("base-url", llmdEnvFirst("FAK_LLMD_BASE_URL", "FAK_LLM_D_BASE_URL"), "llm-d OpenAI-compatible /v1 root (default: FAK_LLMD_BASE_URL or FAK_LLM_D_BASE_URL)")
	model := fs.String("model", llmdEnvFirst("FAK_LLMD_MODEL", "FAK_LLM_D_MODEL"), "model id to send in the chat smoke request (default: first /v1/models id)")
	apiKeyEnv := fs.String("api-key-env", "", "environment variable containing a bearer token; defaults to FAK_LLMD_API_KEY/FAK_LLM_D_API_KEY when set")
	metricsURL := fs.String("metrics-url", llmdEnvFirst("FAK_LLMD_METRICS_URL", "FAK_LLM_D_METRICS_URL"), "optional Prometheus metrics endpoint to normalize as engine=llm-d")
	prompt := fs.String("prompt", "Reply with the single word fak.", "short user prompt for the streamed chat smoke request")
	timeout := fs.Duration("timeout", 15*time.Second, "overall HTTP timeout")
	asJSON := fs.Bool("json", false, "emit the smoke report as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	opts := llmdSmokeOptions{
		BaseURL:    *baseURL,
		Model:      *model,
		APIKeyEnv:  *apiKeyEnv,
		MetricsURL: *metricsURL,
		Prompt:     *prompt,
		Timeout:    *timeout,
		JSON:       *asJSON,
	}
	ctx, cancel := context.WithTimeout(context.Background(), opts.Timeout)
	defer cancel()
	report, err := llmdSmoke(ctx, opts)
	if err != nil {
		report.OK = false
		report.Error = err.Error()
		if opts.JSON {
			writeLLMDSmokeJSON(stdout, report)
		} else {
			fmt.Fprintf(stderr, "fak llmd-smoke: %v\n", err)
		}
		return 1
	}
	if opts.JSON {
		writeLLMDSmokeJSON(stdout, report)
		return 0
	}
	printLLMDSmokeText(stdout, report)
	return 0
}

func llmdSmoke(ctx context.Context, opts llmdSmokeOptions) (report llmdSmokeReport, err error) {
	start := time.Now()
	report = llmdSmokeReport{
		Engine:  engine.LLMDEngineID,
		BaseURL: strings.TrimSpace(opts.BaseURL),
		Metrics: llmdSmokeMetrics{
			Checked:       false,
			SkippedReason: "no metrics endpoint configured",
		},
	}
	defer func() {
		report.DurationMS = time.Since(start).Milliseconds()
	}()
	if report.BaseURL == "" {
		return report, errors.New("--base-url or FAK_LLMD_BASE_URL is required")
	}
	if opts.Timeout <= 0 {
		return report, errors.New("--timeout must be positive")
	}
	apiKey, err := llmdResolveAPIKey(opts.APIKeyEnv)
	if err != nil {
		return report, err
	}
	client := &http.Client{Timeout: opts.Timeout}

	models, ids, err := llmdProbeModels(ctx, client, report.BaseURL, apiKey)
	report.Models = models
	if err != nil {
		return report, err
	}
	model := strings.TrimSpace(opts.Model)
	if model == "" && len(ids) > 0 {
		model = ids[0]
	}
	if model == "" {
		return report, errors.New("llm-d /v1/models returned no model ids; pass --model")
	}
	report.Model = model

	chat, err := llmdProbeChat(ctx, client, report.BaseURL, apiKey, model, opts.Prompt)
	report.Chat = chat
	if err != nil {
		return report, err
	}

	if strings.TrimSpace(opts.MetricsURL) != "" {
		metrics, err := llmdProbeMetrics(ctx, client, opts.MetricsURL, apiKey)
		report.Metrics = metrics
		if err != nil {
			return report, err
		}
	}

	report.OK = report.Models.OK && report.Chat.OK && (!report.Metrics.Checked || report.Metrics.OK)
	if !report.OK {
		return report, errors.New("one or more llm-d smoke checks failed")
	}
	return report, nil
}

func llmdProbeModels(ctx context.Context, client *http.Client, baseURL, apiKey string) (llmdSmokeModels, []string, error) {
	endpoint, err := llmdEndpoint(baseURL, "/models")
	if err != nil {
		return llmdSmokeModels{}, nil, err
	}
	step := llmdSmokeModels{Endpoint: endpoint}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return step, nil, err
	}
	req.Header.Set("Accept", "application/json")
	llmdApplyAuth(req, apiKey)
	resp, err := client.Do(req)
	if err != nil {
		return step, nil, fmt.Errorf("llm-d models probe: %w", err)
	}
	defer resp.Body.Close()
	step.Status = resp.StatusCode
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return step, nil, fmt.Errorf("llm-d models probe returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&payload); err != nil {
		return step, nil, fmt.Errorf("llm-d models probe returned invalid JSON: %w", err)
	}
	ids := make([]string, 0, len(payload.Data))
	for _, item := range payload.Data {
		id := strings.TrimSpace(item.ID)
		if id != "" {
			ids = append(ids, id)
		}
	}
	step.OK = true
	step.Count = len(ids)
	step.IDs = ids
	return step, ids, nil
}

func llmdProbeChat(ctx context.Context, client *http.Client, baseURL, apiKey, model, prompt string) (llmdSmokeChat, error) {
	endpoint, err := llmdEndpoint(baseURL, "/chat/completions")
	if err != nil {
		return llmdSmokeChat{}, err
	}
	step := llmdSmokeChat{Endpoint: endpoint}
	body := map[string]any{
		"model": model,
		"messages": []map[string]string{{
			"role":    "user",
			"content": prompt,
		}},
		"stream":         true,
		"stream_options": map[string]bool{"include_usage": true},
		"max_tokens":     8,
		"temperature":    0,
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return step, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return step, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	llmdApplyAuth(req, apiKey)
	resp, err := client.Do(req)
	if err != nil {
		return step, fmt.Errorf("llm-d chat probe: %w", err)
	}
	defer resp.Body.Close()
	step.Status = resp.StatusCode
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return step, fmt.Errorf("llm-d chat probe returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	if err := llmdConsumeSSE(resp.Body, &step); err != nil {
		return step, err
	}
	if step.DataEvents == 0 {
		return step, errors.New("llm-d chat probe returned no SSE data events")
	}
	if !step.Done {
		return step, errors.New("llm-d chat probe ended without the OpenAI [DONE] sentinel")
	}
	step.OK = true
	return step, nil
}

func llmdProbeMetrics(ctx context.Context, client *http.Client, metricsURL, apiKey string) (llmdSmokeMetrics, error) {
	metricsURL = strings.TrimSpace(metricsURL)
	step := llmdSmokeMetrics{Checked: true, Endpoint: metricsURL}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return step, err
	}
	llmdApplyAuth(req, apiKey)
	resp, err := client.Do(req)
	if err != nil {
		return step, fmt.Errorf("llm-d metrics probe: %w", err)
	}
	defer resp.Body.Close()
	step.Status = resp.StatusCode
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return step, fmt.Errorf("llm-d metrics probe returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return step, err
	}
	snap := engine.ParseLLMDPrometheus(engine.LLMDEngineID, string(raw))
	step.OK = true
	step.Engine = snap.Engine
	step.WorkerID = snap.WorkerID
	step.RequestsRunning = snap.RequestsRunning
	step.RequestsWaiting = snap.RequestsWaiting
	step.RequestSuccesses = snap.RequestSuccesses
	return step, nil
}

func llmdConsumeSSE(r io.Reader, step *llmdSmokeChat) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var event strings.Builder
	flush := func() error {
		data := strings.TrimSpace(event.String())
		event.Reset()
		if data == "" {
			return nil
		}
		return llmdDecodeSSEData(data, step)
	}
	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r")
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			if event.Len() > 0 {
				event.WriteByte('\n')
			}
			event.WriteString(strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return flush()
}

func llmdDecodeSSEData(data string, step *llmdSmokeChat) error {
	if strings.TrimSpace(data) == "[DONE]" {
		step.Done = true
		return nil
	}
	step.DataEvents++
	var chunk struct {
		Model   string          `json:"model"`
		Choices []llmdSSEChoice `json:"choices"`
		Usage   *llmdSmokeUsage `json:"usage"`
	}
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return fmt.Errorf("llm-d chat probe returned invalid SSE JSON: %w", err)
	}
	if chunk.Model != "" {
		step.Model = chunk.Model
	}
	if chunk.Usage != nil {
		step.Usage = chunk.Usage
	}
	for _, choice := range chunk.Choices {
		content := strings.TrimSpace(choice.Text)
		if content == "" {
			content = llmdRawText(choice.Delta.Content)
		}
		if content == "" {
			continue
		}
		step.ContentChunks++
		step.ContentChars += len(content)
	}
	return nil
}

type llmdSSEChoice struct {
	Delta struct {
		Content json.RawMessage `json:"content"`
	} `json:"delta"`
	Text string `json:"text"`
}

func llmdRawText(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var b strings.Builder
		for _, part := range parts {
			b.WriteString(part.Text)
		}
		return b.String()
	}
	var obj struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil {
		return obj.Text
	}
	return ""
}

func llmdEndpoint(baseURL, suffix string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return "", err
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("invalid llm-d base URL %q", baseURL)
	}
	u.Path = strings.TrimRight(u.Path, "/") + suffix
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), nil
}

func llmdResolveAPIKey(apiKeyEnv string) (string, error) {
	apiKeyEnv = strings.TrimSpace(apiKeyEnv)
	if apiKeyEnv != "" {
		v := os.Getenv(apiKeyEnv)
		if strings.TrimSpace(v) == "" {
			return "", fmt.Errorf("--api-key-env %s is set but empty", apiKeyEnv)
		}
		return v, nil
	}
	return llmdEnvFirst("FAK_LLMD_API_KEY", "FAK_LLM_D_API_KEY"), nil
}

func llmdApplyAuth(req *http.Request, apiKey string) {
	if strings.TrimSpace(apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
}

func llmdEnvFirst(keys ...string) string {
	for _, key := range keys {
		if v := os.Getenv(key); strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func writeLLMDSmokeJSON(w io.Writer, report llmdSmokeReport) {
	raw, _ := json.MarshalIndent(report, "", "  ")
	fmt.Fprintln(w, string(raw))
}

func printLLMDSmokeText(w io.Writer, report llmdSmokeReport) {
	fmt.Fprintln(w, "llm-d smoke OK")
	fmt.Fprintf(w, "  models: ok (%d model id(s); selected %s)\n", report.Models.Count, report.Model)
	fmt.Fprintf(w, "  chat: ok (stream events=%d, content chunks=%d, done=%t)\n", report.Chat.DataEvents, report.Chat.ContentChunks, report.Chat.Done)
	if report.Chat.Usage != nil {
		fmt.Fprintf(w, "  usage: prompt=%d completion=%d total=%d\n", report.Chat.Usage.PromptTokens, report.Chat.Usage.CompletionTokens, report.Chat.Usage.TotalTokens)
	}
	if report.Metrics.Checked {
		fmt.Fprintf(w, "  metrics: ok (engine=%s worker=%s)\n", report.Metrics.Engine, report.Metrics.WorkerID)
	} else {
		fmt.Fprintf(w, "  metrics: skipped (%s)\n", report.Metrics.SkippedReason)
	}
}
