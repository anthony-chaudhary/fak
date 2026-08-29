package quality

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// ProbeStatus is a preflight result, not a quality or accuracy verdict.
type ProbeStatus string

const (
	probeOK    ProbeStatus = "supported"
	probeNo    ProbeStatus = "unsupported"
	probeInfra ProbeStatus = "infrastructure_error"
)

// ProbeResult records one independently classified endpoint capability.
type ProbeResult struct {
	Status ProbeStatus `json:"status"`
	Detail string      `json:"detail,omitempty"`
}

// EndpointProbeReport is the bounded, dataset-free receipt emitted by quality probe.
type EndpointProbeReport struct {
	Endpoint          string      `json:"endpoint"`
	Model             string      `json:"model"`
	Models            ProbeResult `json:"models"`
	Generation        ProbeResult `json:"generation"`
	PromptLogprobs    ProbeResult `json:"prompt_logprobs"`
	Reasoning         ProbeResult `json:"reasoning"`
	Engine            string      `json:"engine"`
	Fallbacks         int         `json:"fallbacks"`
	Native            bool        `json:"native"`
	AccuracyEvaluated bool        `json:"accuracy_evaluated"`
}

// ProbeEndpoint performs only /v1/models and one-token task probes.
func ProbeEndpoint(ctx context.Context, client *http.Client, endpoint, model string) EndpointProbeReport {
	report := EndpointProbeReport{Endpoint: strings.TrimRight(endpoint, "/"), Model: model, Engine: "openai-compatible"}
	if client == nil {
		client = &http.Client{Timeout: 15 * time.Second}
	}
	if model == "" {
		detail := "exact model identity is required"
		report.Models = infra(detail)
		report.Generation = infra(detail)
		report.PromptLogprobs = infra(detail)
		report.Reasoning = infra(detail)
		return report
	}
	if _, err := url.ParseRequestURI(report.Endpoint); err != nil || !(strings.HasPrefix(report.Endpoint, "http://") || strings.HasPrefix(report.Endpoint, "https://")) {
		detail := "valid http(s) endpoint is required"
		report.Models = infra(detail)
		report.Generation = infra(detail)
		report.PromptLogprobs = infra(detail)
		report.Reasoning = infra(detail)
		return report
	}

	models, _, err := request(ctx, client, http.MethodGet, report.Endpoint+"/v1/models", nil)
	if err != nil {
		detail := "models request: " + err.Error()
		report.Models = infra(detail)
		report.Generation = infra(detail)
		report.PromptLogprobs = infra(detail)
		report.Reasoning = infra(detail)
		return report
	}
	if models.status < 200 || models.status >= 300 {
		detail := fmt.Sprintf("models request returned HTTP %d", models.status)
		report.Models = infra(detail)
		report.Generation = infra(detail)
		report.PromptLogprobs = infra(detail)
		report.Reasoning = infra(detail)
		return report
	}
	var listing struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if json.Unmarshal(models.body, &listing) != nil {
		detail := "models response was not valid JSON"
		report.Models = infra(detail)
		report.Generation = infra(detail)
		report.PromptLogprobs = infra(detail)
		report.Reasoning = infra(detail)
		return report
	}
	found := false
	for _, item := range listing.Data {
		if item.ID == model {
			found = true
			break
		}
	}
	if !found {
		detail := fmt.Sprintf("model %q was not returned by /v1/models", model)
		report.Models = infra(detail)
		report.Generation = infra(detail)
		report.PromptLogprobs = infra(detail)
		report.Reasoning = infra(detail)
		return report
	}
	report.Models = ProbeResult{Status: probeOK}

	generationBody := map[string]any{"model": model, "messages": []map[string]string{{"role": "user", "content": "Reply with one token."}}, "max_tokens": 1, "temperature": 0}
	generation, headers, err := request(ctx, client, http.MethodPost, report.Endpoint+"/v1/chat/completions", generationBody)
	report.Generation = classifyHTTP(generation, err, "generation")
	setProvenance(&report, headers, generation.body)

	logprobBody := map[string]any{"model": model, "prompt": "A", "max_tokens": 1, "temperature": 0, "echo": true, "logprobs": 1}
	logprobs, lpHeaders, err := request(ctx, client, http.MethodPost, report.Endpoint+"/v1/completions", logprobBody)
	report.PromptLogprobs = classifyLogprobs(logprobs, err)
	setProvenance(&report, lpHeaders, logprobs.body)

	reasoningBody := map[string]any{"model": model, "messages": []map[string]string{{"role": "user", "content": "Reply with one token."}}, "max_tokens": 1, "reasoning_effort": "low"}
	reasoning, reasoningHeaders, err := request(ctx, client, http.MethodPost, report.Endpoint+"/v1/chat/completions", reasoningBody)
	report.Reasoning = classifyHTTP(reasoning, err, "reasoning_effort")
	setProvenance(&report, reasoningHeaders, reasoning.body)
	report.Native = report.Engine == "fak-native" && report.Fallbacks == 0
	return report
}

// HasInfrastructureError reports whether configuration or transport prevented a trustworthy preflight.
func (r EndpointProbeReport) HasInfrastructureError() bool {
	return r.Models.Status == probeInfra ||
		r.Generation.Status == probeInfra ||
		r.PromptLogprobs.Status == probeInfra ||
		r.Reasoning.Status == probeInfra
}

type httpResult struct {
	status int
	body   []byte
}

func request(ctx context.Context, client *http.Client, method, target string, payload any) (httpResult, http.Header, error) {
	var body io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			return httpResult{}, nil, err
		}
		body = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, target, body)
	if err != nil {
		return httpResult{}, nil, err
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		return httpResult{}, nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return httpResult{}, resp.Header, err
	}
	return httpResult{status: resp.StatusCode, body: data}, resp.Header, nil
}

func classifyHTTP(result httpResult, err error, name string) ProbeResult {
	if err != nil {
		return infra(name + " request: " + err.Error())
	}
	if result.status >= 200 && result.status < 300 {
		return ProbeResult{Status: probeOK}
	}
	if result.status >= 400 && result.status < 500 {
		return ProbeResult{Status: probeNo, Detail: fmt.Sprintf("HTTP %d", result.status)}
	}
	return infra(fmt.Sprintf("%s request returned HTTP %d", name, result.status))
}

func classifyLogprobs(result httpResult, err error) ProbeResult {
	classified := classifyHTTP(result, err, "prompt logprobs")
	if classified.Status != probeOK {
		return classified
	}
	var response struct {
		Choices []struct {
			Logprobs json.RawMessage `json:"logprobs"`
		} `json:"choices"`
	}
	if json.Unmarshal(result.body, &response) != nil {
		return infra("prompt logprobs response was not valid JSON")
	}
	if len(response.Choices) == 0 || len(response.Choices[0].Logprobs) == 0 || string(response.Choices[0].Logprobs) == "null" {
		return ProbeResult{Status: probeNo, Detail: "response omitted prompt logprobs"}
	}
	return classified
}

func infra(detail string) ProbeResult {
	return ProbeResult{Status: probeInfra, Detail: detail}
}

func setProvenance(report *EndpointProbeReport, headers http.Header, body []byte) {
	engine := headers.Get("X-Fak-Engine")
	fallbacks := headers.Get("X-Fak-Fallbacks")
	var envelope struct {
		Engine    string `json:"engine"`
		Fallbacks int    `json:"fallbacks"`
	}
	_ = json.Unmarshal(body, &envelope)
	if engine == "" {
		engine = envelope.Engine
	}
	if engine != "" {
		report.Engine = engine
	}
	if fallbacks != "" {
		if n, err := strconv.Atoi(fallbacks); err == nil {
			report.Fallbacks = n
		}
	} else if envelope.Fallbacks > report.Fallbacks {
		report.Fallbacks = envelope.Fallbacks
	}
}
