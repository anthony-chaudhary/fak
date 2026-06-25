package webbench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const ServingParitySchema = "fak.serving-parity.v1"

type ServingTrack string

const (
	TrackOurs           ServingTrack = "ours"
	TrackSGLang         ServingTrack = "sglang"
	TrackVLLM           ServingTrack = "vllm"
	TrackFakFrontsFleet ServingTrack = "fak-fronts-fleet"
)

var AllServingTracks = []ServingTrack{
	TrackOurs,
	TrackSGLang,
	TrackVLLM,
	TrackFakFrontsFleet,
}

// ParseServingTrack is the track-selector extension point for issue #44.
func ParseServingTrack(s string) (ServingTrack, error) {
	n := strings.ToLower(strings.TrimSpace(s))
	n = strings.ReplaceAll(n, "_", "-")
	switch ServingTrack(n) {
	case TrackOurs, TrackSGLang, TrackVLLM, TrackFakFrontsFleet:
		return ServingTrack(n), nil
	default:
		return "", fmt.Errorf("unknown serving track %q", s)
	}
}

func ParseServingTracks(s string) ([]ServingTrack, error) {
	if strings.TrimSpace(s) == "" {
		return append([]ServingTrack(nil), AllServingTracks...), nil
	}
	seen := make(map[ServingTrack]bool)
	var tracks []ServingTrack
	for _, part := range strings.Split(s, ",") {
		tr, err := ParseServingTrack(part)
		if err != nil {
			return nil, err
		}
		if !seen[tr] {
			seen[tr] = true
			tracks = append(tracks, tr)
		}
	}
	return tracks, nil
}

type ServingPlan struct {
	Track    ServingTrack `json:"track"`
	Model    string       `json:"model,omitempty"`
	BaseURL  string       `json:"base_url,omitempty"`
	Replicas int          `json:"replicas,omitempty"`
}

// ScriptFor returns the operator command shape for a track. It is deliberately
// descriptive: the harness measures OpenAI-compatible endpoints and does not
// fork or vendor vLLM/SGLang internals.
func ScriptFor(p ServingPlan) (string, error) {
	track, err := ParseServingTrack(string(p.Track))
	if err != nil {
		return "", err
	}
	model := p.Model
	if model == "" {
		model = "<model>"
	}
	baseURL := strings.TrimRight(p.BaseURL, "/")
	if baseURL == "" {
		baseURL = "<engine-or-router-base-url>/v1"
	}
	replicas := p.Replicas
	if replicas <= 0 {
		replicas = 1
	}
	switch track {
	case TrackOurs:
		return "go build -o fak ./cmd/fak && ./fak serve --engine inkernel --addr 127.0.0.1:8080", nil
	case TrackSGLang:
		return fmt.Sprintf("python -m sglang.launch_server --model-path %s --host 127.0.0.1 --port 30000", model), nil
	case TrackVLLM:
		return fmt.Sprintf("vllm serve %s --host 127.0.0.1 --port 8000", model), nil
	case TrackFakFrontsFleet:
		return fmt.Sprintf("start %d OpenAI-compatible engine replica(s) behind a fleet router, then: go build -o fak ./cmd/fak && ./fak serve --provider openai --base-url %s --model %s --addr 127.0.0.1:8080", replicas, baseURL, model), nil
	default:
		return "", fmt.Errorf("unknown serving track %q", p.Track)
	}
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ServingRequest struct {
	ID                   string        `json:"id"`
	Messages             []ChatMessage `json:"messages"`
	MaxOutputTokens      int           `json:"max_output_tokens"`
	PromptTokensEstimate int           `json:"prompt_tokens_estimate"`
}

func BuildServingWorkload(d *Dataset, gm GeometryModel, limit, agents, maxOutputTokens int, sharedPrefix string) []ServingRequest {
	if d == nil || d.Len() == 0 {
		return nil
	}
	if limit <= 0 || limit > d.Len() {
		limit = d.Len()
	}
	if agents <= 0 {
		agents = limit
	}
	if maxOutputTokens <= 0 {
		maxOutputTokens = 64
	}
	if strings.TrimSpace(sharedPrefix) == "" {
		sharedPrefix = "Shared browser-agent serving prefix: answer each task concisely. This prefix is intentionally identical across requests so vLLM, SGLang, and fak-fronted fleet runs exercise the same cacheable prompt prefix."
	}

	base := make([]ServingRequest, 0, limit)
	for i := 0; i < limit; i++ {
		in := d.Instances[i]
		geom := gm.Derive(in)
		body := strings.TrimSpace(in.Description + "\n" + in.Instructions)
		if body == "" {
			body = in.TaskID
		}
		id := in.TaskID
		if id == "" {
			id = fmt.Sprintf("task-%03d", i+1)
		}
		base = append(base, ServingRequest{
			ID: id,
			Messages: []ChatMessage{
				{Role: "system", Content: sharedPrefix},
				{Role: "user", Content: "Task: " + body},
			},
			MaxOutputTokens:      maxOutputTokens,
			PromptTokensEstimate: geom.Prefix,
		})
	}

	out := make([]ServingRequest, 0, agents)
	for len(out) < agents {
		for _, req := range base {
			if len(out) >= agents {
				break
			}
			cp := req
			if agents > len(base) {
				cp.ID = fmt.Sprintf("%s#agent-%03d", req.ID, len(out)+1)
			}
			out = append(out, cp)
		}
	}
	return out
}

type ServingTrackConfig struct {
	Track      ServingTrack `json:"track"`
	BaseURL    string       `json:"base_url,omitempty"`
	MetricsURL string       `json:"metrics_url,omitempty"`
	Model      string       `json:"model,omitempty"`
	APIKey     string       `json:"-"`
	APIKeyEnv  string       `json:"api_key_env,omitempty"`
	Replicas   int          `json:"replicas,omitempty"`
}

type ServingParityConfig struct {
	GeneratedAt string
	MachineID   string
	Model       string
	Tracks      []ServingTrackConfig
	Workload    []ServingRequest
	Concurrency int
	SLO         time.Duration
	Timeout     time.Duration
	Client      *http.Client
}

type ServingParityReport struct {
	Schema      string                 `json:"schema"`
	GeneratedAt string                 `json:"generated_at"`
	MachineID   string                 `json:"machine_id"`
	Model       string                 `json:"model"`
	Workload    ServingWorkloadInfo    `json:"workload"`
	Tracks      []ServingTrackResult   `json:"tracks"`
	Honesty     ServingHonestyContract `json:"honesty"`
	Artifact    string                 `json:"artifact,omitempty"`
}

type ServingWorkloadInfo struct {
	Requests        int    `json:"requests"`
	Concurrency     int    `json:"concurrency"`
	SLOMillis       int64  `json:"slo_ms"`
	TokenCountBasis string `json:"token_count_basis"`
}

type ServingHonestyContract struct {
	IdenticalWorkload bool     `json:"identical_workload"`
	AbsentAs          string   `json:"absent_as"`
	RequiredTracks    []string `json:"required_tracks"`
	ParityClaimGate   string   `json:"parity_claim_gate"`
}

type ServingTrackResult struct {
	Track      ServingTrack    `json:"track"`
	Status     string          `json:"status"`
	Reason     string          `json:"reason,omitempty"`
	BaseURL    string          `json:"base_url,omitempty"`
	MetricsURL string          `json:"metrics_url,omitempty"`
	PlanScript string          `json:"plan_script,omitempty"`
	Samples    []ServingSample `json:"samples,omitempty"`
	Stats      ServingStats    `json:"stats"`
}

type ServingSample struct {
	ID                   string    `json:"id"`
	Status               string    `json:"status"`
	Error                string    `json:"error,omitempty"`
	HTTPStatus           int       `json:"http_status,omitempty"`
	StreamMode           string    `json:"stream_mode,omitempty"`
	TTFTMillis           *float64  `json:"ttft_ms,omitempty"`
	ITLMillis            []float64 `json:"itl_ms,omitempty"`
	TPOTMillis           *float64  `json:"tpot_ms,omitempty"`
	EndToEndMillis       float64   `json:"end_to_end_ms"`
	OutputEvents         int       `json:"output_events"`
	OutputTokenEstimate  int       `json:"output_token_estimate"`
	PromptTokensEstimate int       `json:"prompt_tokens_estimate"`
}

type ServingStats struct {
	Requests           int            `json:"requests"`
	OK                 int            `json:"ok"`
	Failed             int            `json:"failed"`
	TTFTMillis         QuantileMetric `json:"ttft_ms"`
	ITLMillis          QuantileMetric `json:"itl_ms"`
	TPOTMillis         QuantileMetric `json:"tpot_ms"`
	EndToEndMillis     QuantileMetric `json:"end_to_end_ms"`
	ThroughputTokensS  ScalarMetric   `json:"throughput_tok_s"`
	GoodputRPS         ScalarMetric   `json:"goodput_rps"`
	PrefixCacheHitRate ScalarMetric   `json:"prefix_cache_hit_rate"`
	TokenCountBasis    string         `json:"token_count_basis"`
}

type QuantileMetric struct {
	Status string   `json:"status"`
	Unit   string   `json:"unit,omitempty"`
	P50    *float64 `json:"p50,omitempty"`
	P90    *float64 `json:"p90,omitempty"`
	P99    *float64 `json:"p99,omitempty"`
	Reason string   `json:"reason,omitempty"`
}

type ScalarMetric struct {
	Status string   `json:"status"`
	Unit   string   `json:"unit,omitempty"`
	Value  *float64 `json:"value,omitempty"`
	Reason string   `json:"reason,omitempty"`
	Source string   `json:"source,omitempty"`
}

func RunServingParity(ctx context.Context, cfg ServingParityConfig) (*ServingParityReport, error) {
	if len(cfg.Tracks) == 0 {
		for _, tr := range AllServingTracks {
			cfg.Tracks = append(cfg.Tracks, ServingTrackConfig{Track: tr})
		}
	}
	if len(cfg.Workload) == 0 {
		return nil, errors.New("serving parity workload is empty")
	}
	if cfg.GeneratedAt == "" {
		cfg.GeneratedAt = time.Now().UTC().Format(time.RFC3339)
	}
	if cfg.MachineID == "" {
		cfg.MachineID = "unknown"
	}
	if cfg.Model == "" {
		cfg.Model = "unknown"
	}
	if cfg.Concurrency <= 0 {
		cfg.Concurrency = 1
	}
	if cfg.SLO <= 0 {
		cfg.SLO = 2 * time.Second
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 60 * time.Second
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{Timeout: cfg.Timeout + 5*time.Second}
	}

	rep := &ServingParityReport{
		Schema:      ServingParitySchema,
		GeneratedAt: cfg.GeneratedAt,
		MachineID:   cfg.MachineID,
		Model:       cfg.Model,
		Workload: ServingWorkloadInfo{
			Requests:        len(cfg.Workload),
			Concurrency:     cfg.Concurrency,
			SLOMillis:       cfg.SLO.Milliseconds(),
			TokenCountBasis: "stream_content_events",
		},
		Honesty: ServingHonestyContract{
			IdenticalWorkload: true,
			AbsentAs:          "not_measured",
			RequiredTracks:    []string{string(TrackVLLM), string(TrackSGLang), string(TrackFakFrontsFleet)},
			ParityClaimGate:   "A parity-or-better claim requires a serving parity artifact with measured vllm, sglang, and fak-fronts-fleet tracks.",
		},
	}
	for _, tc := range cfg.Tracks {
		tr, err := measureTrack(ctx, cfg, tc)
		if err != nil {
			return nil, err
		}
		rep.Tracks = append(rep.Tracks, tr)
	}
	return rep, nil
}

func measureTrack(ctx context.Context, cfg ServingParityConfig, tc ServingTrackConfig) (ServingTrackResult, error) {
	track, err := ParseServingTrack(string(tc.Track))
	if err != nil {
		return ServingTrackResult{}, err
	}
	tc.Track = track
	model := tc.Model
	if model == "" {
		model = cfg.Model
	}
	script, _ := ScriptFor(ServingPlan{Track: track, Model: model, BaseURL: tc.BaseURL, Replicas: tc.Replicas})
	res := ServingTrackResult{
		Track:      track,
		BaseURL:    tc.BaseURL,
		MetricsURL: tc.MetricsURL,
		PlanScript: script,
	}
	if strings.TrimSpace(tc.BaseURL) == "" {
		res.Status = "not_measured"
		res.Reason = "no base URL configured for this track"
		res.Stats = emptyServingStats("track was not measured")
		return res, nil
	}

	start := time.Now()
	samples := runSamples(ctx, cfg.Client, tc, model, cfg.Workload, cfg.Concurrency, cfg.Timeout)
	wall := time.Since(start)
	res.Samples = samples
	res.Status = "measured"
	res.Stats = FoldServingSamples(samples, wall.Seconds(), cfg.SLO)
	res.Stats.PrefixCacheHitRate = FetchPrefixCacheHitRate(ctx, cfg.Client, tc.MetricsURL)
	return res, nil
}

func emptyServingStats(reason string) ServingStats {
	return ServingStats{
		TTFTMillis:         notMeasuredQuantile("ms", reason),
		ITLMillis:          notMeasuredQuantile("ms", reason),
		TPOTMillis:         notMeasuredQuantile("ms", reason),
		EndToEndMillis:     notMeasuredQuantile("ms", reason),
		ThroughputTokensS:  notMeasuredScalar("stream_content_events/s", reason),
		GoodputRPS:         notMeasuredScalar("requests/s", reason),
		PrefixCacheHitRate: notMeasuredScalar("ratio", reason),
		TokenCountBasis:    "stream_content_events",
	}
}

func runSamples(ctx context.Context, client *http.Client, tc ServingTrackConfig, model string, workload []ServingRequest, concurrency int, timeout time.Duration) []ServingSample {
	type job struct {
		index int
		req   ServingRequest
	}
	if concurrency <= 0 {
		concurrency = 1
	}
	if concurrency > len(workload) {
		concurrency = len(workload)
	}
	out := make([]ServingSample, len(workload))
	jobs := make(chan job)
	var wg sync.WaitGroup
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				out[j.index] = MeasureSSERequest(ctx, client, tc, model, j.req, timeout)
			}
		}()
	}
	for i, req := range workload {
		jobs <- job{index: i, req: req}
	}
	close(jobs)
	wg.Wait()
	return out
}

func MeasureSSERequest(ctx context.Context, client *http.Client, tc ServingTrackConfig, model string, req ServingRequest, timeout time.Duration) ServingSample {
	sample := ServingSample{
		ID:                   req.ID,
		Status:               "fail",
		PromptTokensEstimate: req.PromptTokensEstimate,
	}
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	if req.MaxOutputTokens <= 0 {
		req.MaxOutputTokens = 64
	}
	payload := map[string]any{
		"model":       model,
		"messages":    req.Messages,
		"max_tokens":  req.MaxOutputTokens,
		"temperature": 0,
		"stream":      true,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		sample.Error = err.Error()
		return sample
	}
	url := strings.TrimRight(tc.BaseURL, "/") + "/chat/completions"
	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	httpReq, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		sample.Error = err.Error()
		return sample
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if key := tc.APIKey; key != "" {
		httpReq.Header.Set("Authorization", "Bearer "+key)
	} else if tc.APIKeyEnv != "" {
		if key := os.Getenv(tc.APIKeyEnv); key != "" {
			httpReq.Header.Set("Authorization", "Bearer "+key)
		}
	}

	start := time.Now()
	resp, err := client.Do(httpReq)
	if err != nil {
		sample.EndToEndMillis = millis(time.Since(start))
		sample.Error = err.Error()
		return sample
	}
	defer resp.Body.Close()
	sample.HTTPStatus = resp.StatusCode
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		sample.EndToEndMillis = millis(time.Since(start))
		sample.Error = strings.TrimSpace(string(b))
		if sample.Error == "" {
			sample.Error = resp.Status
		}
		return sample
	}

	ct := strings.ToLower(resp.Header.Get("Content-Type"))
	if !strings.Contains(ct, "text/event-stream") {
		b, _ := io.ReadAll(resp.Body)
		sample.StreamMode = "non_sse"
		sample.EndToEndMillis = millis(time.Since(start))
		sample.Status = "ok"
		content := completionContent(b)
		sample.OutputTokenEstimate = EstimateTokens(content)
		return sample
	}

	sample.StreamMode = "sse"
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var first, last time.Time
	var content strings.Builder
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}
		part, ok := streamContent(data)
		if !ok {
			continue
		}
		now := time.Now()
		if first.IsZero() {
			first = now
			ttft := millis(first.Sub(start))
			sample.TTFTMillis = &ttft
		} else {
			sample.ITLMillis = append(sample.ITLMillis, millis(now.Sub(last)))
		}
		last = now
		sample.OutputEvents++
		content.WriteString(part)
	}
	if err := sc.Err(); err != nil {
		sample.EndToEndMillis = millis(time.Since(start))
		sample.Error = err.Error()
		return sample
	}
	end := time.Now()
	sample.EndToEndMillis = millis(end.Sub(start))
	sample.OutputTokenEstimate = sample.OutputEvents
	if sample.OutputTokenEstimate == 0 {
		sample.OutputTokenEstimate = EstimateTokens(content.String())
	}
	if sample.OutputEvents > 1 && !first.IsZero() {
		tpot := millis(end.Sub(first)) / float64(sample.OutputEvents-1)
		sample.TPOTMillis = floatPtr(tpot)
	}
	sample.Status = "ok"
	return sample
}

func streamContent(data string) (string, bool) {
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return data, true
	}
	for _, ch := range chunk.Choices {
		if ch.Delta.Content != "" {
			return ch.Delta.Content, true
		}
		if ch.Text != "" {
			return ch.Text, true
		}
	}
	return "", false
}

func completionContent(data []byte) string {
	var doc struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Text string `json:"text"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &doc); err != nil {
		return string(data)
	}
	var b strings.Builder
	for _, ch := range doc.Choices {
		b.WriteString(ch.Message.Content)
		b.WriteString(ch.Text)
	}
	return b.String()
}

func FoldServingSamples(samples []ServingSample, wallSeconds float64, slo time.Duration) ServingStats {
	stats := ServingStats{
		Requests:        len(samples),
		TokenCountBasis: "stream_content_events",
	}
	var ttft, itl, tpot, e2e []float64
	var good, tokens int
	for _, s := range samples {
		if s.Status == "ok" {
			stats.OK++
			e2e = append(e2e, s.EndToEndMillis)
			if slo > 0 && time.Duration(s.EndToEndMillis*float64(time.Millisecond)) <= slo {
				good++
			}
			if s.TTFTMillis != nil {
				ttft = append(ttft, *s.TTFTMillis)
			}
			itl = append(itl, s.ITLMillis...)
			if s.TPOTMillis != nil {
				tpot = append(tpot, *s.TPOTMillis)
			}
			tokens += s.OutputTokenEstimate
		} else {
			stats.Failed++
		}
	}
	stats.TTFTMillis = quantiles(ttft, "ms", "no streaming first-token measurements")
	stats.ITLMillis = quantiles(itl, "ms", "no inter-token measurements")
	stats.TPOTMillis = quantiles(tpot, "ms", "no TPOT measurements")
	stats.EndToEndMillis = quantiles(e2e, "ms", "no successful request latencies")
	if wallSeconds > 0 && tokens > 0 {
		stats.ThroughputTokensS = measuredScalar(float64(tokens)/wallSeconds, "stream_content_events/s", "")
	} else {
		stats.ThroughputTokensS = notMeasuredScalar("stream_content_events/s", "no output events measured")
	}
	if wallSeconds > 0 && stats.OK > 0 {
		stats.GoodputRPS = measuredScalar(float64(good)/wallSeconds, "requests/s", "")
	} else {
		stats.GoodputRPS = notMeasuredScalar("requests/s", "no successful requests measured")
	}
	stats.PrefixCacheHitRate = notMeasuredScalar("ratio", "no metrics URL configured")
	return stats
}

func FetchPrefixCacheHitRate(ctx context.Context, client *http.Client, metricsURL string) ScalarMetric {
	if strings.TrimSpace(metricsURL) == "" {
		return notMeasuredScalar("ratio", "no metrics URL configured")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return notMeasuredScalar("ratio", err.Error())
	}
	resp, err := client.Do(req)
	if err != nil {
		return notMeasuredScalar("ratio", err.Error())
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return notMeasuredScalar("ratio", resp.Status)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return notMeasuredScalar("ratio", err.Error())
	}
	value, source, ok := ParsePrefixCacheHitRateMetrics(string(b))
	if !ok {
		return notMeasuredScalar("ratio", "no prefix-cache hit-rate metric found")
	}
	m := measuredScalar(value, "ratio", source)
	return m
}

func ParsePrefixCacheHitRateMetrics(text string) (float64, string, bool) {
	known := []string{
		"vllm:gpu_prefix_cache_hit_rate",
		"vllm:cpu_prefix_cache_hit_rate",
		"vllm:prefix_cache_hit_rate",
		"sglang:prefix_cache_hit_rate",
		"sglang_prefix_cache_hit_rate",
		"fak_gateway_kv_prefix_hit_rate",
		"fak_gateway_kv_prefix_hit_ratio",
	}
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		name := strings.SplitN(fields[0], "{", 2)[0]
		matched := false
		for _, k := range known {
			if name == k {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}
		v, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			continue
		}
		if v > 1 && v <= 100 {
			v = v / 100
		}
		return round(v), name, true
	}
	return 0, "", false
}

func quantiles(values []float64, unit, reason string) QuantileMetric {
	if len(values) == 0 {
		return notMeasuredQuantile(unit, reason)
	}
	sort.Float64s(values)
	return QuantileMetric{
		Status: "measured",
		Unit:   unit,
		P50:    floatPtr(percentileFloat(values, 0.50)),
		P90:    floatPtr(percentileFloat(values, 0.90)),
		P99:    floatPtr(percentileFloat(values, 0.99)),
	}
}

func percentileFloat(sorted []float64, q float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(q*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func notMeasuredQuantile(unit, reason string) QuantileMetric {
	return QuantileMetric{Status: "not_measured", Unit: unit, Reason: reason}
}

func measuredScalar(value float64, unit, source string) ScalarMetric {
	return ScalarMetric{Status: "measured", Unit: unit, Value: floatPtr(value), Source: source}
}

func notMeasuredScalar(unit, reason string) ScalarMetric {
	return ScalarMetric{Status: "not_measured", Unit: unit, Reason: reason}
}

func floatPtr(v float64) *float64 {
	x := round(v)
	return &x
}

func round(v float64) float64 {
	return math.Round(v*1000) / 1000
}

func millis(d time.Duration) float64 {
	return round(float64(d) / float64(time.Millisecond))
}

func DefaultServingArtifactPath(outDir, machineID, generatedAt string, tracks []ServingTrack) string {
	if outDir == "" {
		outDir = filepath.Join("experiments", "benchmark", "runs", "by-machine")
	}
	if machineID == "" {
		machineID = "unknown"
	}
	ts := generatedAt
	if parsed, err := time.Parse(time.RFC3339, generatedAt); err == nil {
		ts = parsed.UTC().Format("20060102T150405Z")
	}
	if ts == "" {
		ts = time.Now().UTC().Format("20060102T150405Z")
	}
	labels := make([]string, 0, len(tracks))
	for _, tr := range tracks {
		labels = append(labels, sanitizePathPart(string(tr)))
	}
	if len(labels) == 0 {
		labels = []string{"serving"}
	}
	return filepath.Join(outDir, sanitizePathPart(machineID), fmt.Sprintf("%s-serving-parity-%s", ts, strings.Join(labels, "-")), "result.json")
}

func WriteServingParityReport(rep *ServingParityReport, path string) error {
	if rep == nil {
		return errors.New("nil serving parity report")
	}
	rep.Artifact = path
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0644)
}

func LoadServingParityReport(path string) (*ServingParityReport, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rep ServingParityReport
	if err := json.Unmarshal(data, &rep); err != nil {
		return nil, err
	}
	return &rep, nil
}

func ClaimRequiresServingArtifact(text string) bool {
	lower := strings.ToLower(text)
	needles := []string{
		"parity or better",
		"parity-or-better",
		"parity with vllm",
		"parity with sglang",
		"vllm/sglang/native parity",
		"vllm/sGLang/native parity",
	}
	for _, needle := range needles {
		if strings.Contains(lower, strings.ToLower(needle)) {
			return true
		}
	}
	return false
}

func ValidateParityClaim(claim string, rep *ServingParityReport) error {
	if !ClaimRequiresServingArtifact(claim) {
		return nil
	}
	if rep == nil {
		return errors.New("parity-or-better claim requires a serving parity artifact")
	}
	if rep.Schema != ServingParitySchema {
		return fmt.Errorf("serving parity artifact schema = %q, want %q", rep.Schema, ServingParitySchema)
	}
	required := []ServingTrack{TrackVLLM, TrackSGLang, TrackFakFrontsFleet}
	for _, want := range required {
		if !rep.TrackMeasured(want) {
			return fmt.Errorf("parity-or-better claim requires measured %s track", want)
		}
	}
	return nil
}

func (r *ServingParityReport) TrackMeasured(track ServingTrack) bool {
	if r == nil {
		return false
	}
	for _, tr := range r.Tracks {
		if tr.Track == track && tr.Status == "measured" && tr.Stats.OK > 0 {
			return true
		}
	}
	return false
}

func sanitizePathPart(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	if s == "" {
		return "unknown"
	}
	var b strings.Builder
	lastDash := false
	for _, r := range s {
		ok := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if ok {
			b.WriteRune(r)
			lastDash = false
			continue
		}
		if !lastDash {
			b.WriteByte('-')
			lastDash = true
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "unknown"
	}
	return out
}
