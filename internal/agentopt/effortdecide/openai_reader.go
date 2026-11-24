package effortdecide

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// OpenAIReader reads declared-label scores from an OpenAI-compatible
// /v1/chat/completions endpoint (vLLM, SGLang, llama.cpp server, hosted
// providers) using first-token top_logprobs. It generates one token, but the
// decision is the distribution, not the sampled token.
//
// The decision model must return logprobs and answer on its first token.
// Hosted reasoning models that refuse logprobs cannot serve here; a local
// thinking model needs DisableThinking so its first token is not "<think>".
type OpenAIReader struct {
	BaseURL string // http://host:8000 or http://host:8000/v1
	Model   string
	APIKey  string
	// MaxTokensField names the output-cap field; "" means "max_tokens".
	// Some providers accept only "max_completion_tokens".
	MaxTokensField string
	// DisableThinking sends chat_template_kwargs.enable_thinking=false, which
	// vLLM and SGLang honor for Qwen3/DeepSeek-style templates.
	DisableThinking bool
	Client          *http.Client
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResp struct {
	Choices []struct {
		Logprobs struct {
			Content []struct {
				TopLogprobs []struct {
					Token   string  `json:"token"`
					Logprob float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
}

// ReadOptions implements Reader.
func (r *OpenAIReader) ReadOptions(ctx context.Context, prompt string, labels []string) (map[string]float64, error) {
	maxField := r.MaxTokensField
	if maxField == "" {
		maxField = "max_tokens"
	}
	payload := map[string]any{
		"model": r.Model,
		"messages": []chatMsg{
			{Role: "system", Content: "Reply with exactly one capital letter."},
			{Role: "user", Content: prompt},
		},
		maxField:       1,
		"temperature":  0,
		"logprobs":     true,
		"top_logprobs": 20,
	}
	if r.DisableThinking {
		payload["chat_template_kwargs"] = map[string]any{"enable_thinking": false}
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(r.BaseURL, "/")
	if !strings.HasSuffix(url, "/v1") {
		url += "/v1"
	}
	url += "/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if r.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+r.APIKey)
	}
	client := r.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		snippet := strings.TrimSpace(string(raw))
		if len(snippet) > 300 {
			snippet = snippet[:300]
		}
		return nil, fmt.Errorf("decision endpoint status %d: %s", resp.StatusCode, snippet)
	}
	var cr chatResp
	if err := json.Unmarshal(raw, &cr); err != nil {
		return nil, fmt.Errorf("decode decision response: %w", err)
	}
	if len(cr.Choices) == 0 || len(cr.Choices[0].Logprobs.Content) == 0 {
		return nil, fmt.Errorf("decision response has no logprobs")
	}
	want := make(map[string]bool, len(labels))
	for _, l := range labels {
		want[l] = true
	}
	out := map[string]float64{}
	for _, tl := range cr.Choices[0].Logprobs.Content[0].TopLogprobs {
		label := normalizeLabel(tl.Token)
		if !want[label] {
			continue
		}
		// "A" and " A" are distinct tokens for one answer: combine their mass.
		if prev, ok := out[label]; ok {
			hi, lo := math.Max(prev, tl.Logprob), math.Min(prev, tl.Logprob)
			out[label] = hi + math.Log1p(math.Exp(lo-hi))
		} else {
			out[label] = tl.Logprob
		}
	}
	return out, nil
}

// normalizeLabel maps first-token variants of an answer ("A", " A", "A)",
// "**A", "a.") to the bare label.
func normalizeLabel(tok string) string {
	return strings.ToUpper(strings.Trim(tok, " \t\n*().:[]_`'\""))
}

// Environment knobs for FromEnv.
const (
	EnvDecider       = "FAK_EFFORT_DECIDER" // "decision-model" enables; empty/"off" disables
	EnvURL           = "FAK_EFFORT_DECIDER_URL"
	EnvModel         = "FAK_EFFORT_DECIDER_MODEL"
	EnvAPIKey        = "FAK_EFFORT_DECIDER_API_KEY"
	EnvMinConfidence = "FAK_EFFORT_DECIDER_MIN_CONFIDENCE"
	EnvTimeout       = "FAK_EFFORT_DECIDER_TIMEOUT" // Go duration, e.g. 500ms
	EnvMinMass       = "FAK_EFFORT_DECIDER_MIN_DECLARED_MASS"
	EnvMaxTokens     = "FAK_EFFORT_DECIDER_MAX_TOKENS_FIELD" // max_tokens | max_completion_tokens
	EnvNoThinking    = "FAK_EFFORT_DECIDER_DISABLE_THINKING" // 1/true
)

// FromEnv builds a Decider from environment knobs. It returns (nil, nil) when
// the feature is off, and an error when it is on but misconfigured so a typo
// surfaces instead of silently running the heuristic.
func FromEnv(getenv func(string) string) (*Decider, error) {
	switch mode := strings.ToLower(strings.TrimSpace(getenv(EnvDecider))); mode {
	case "", "off", "0", "false", "heuristic":
		return nil, nil
	case "decision-model", "jev", "on", "1", "true":
	default:
		return nil, fmt.Errorf("%s=%q: want decision-model or off", EnvDecider, mode)
	}
	url, model := strings.TrimSpace(getenv(EnvURL)), strings.TrimSpace(getenv(EnvModel))
	if url == "" || model == "" {
		return nil, fmt.Errorf("%s needs %s and %s", EnvDecider, EnvURL, EnvModel)
	}
	if !strings.HasPrefix(url, "http://") && !strings.HasPrefix(url, "https://") {
		return nil, fmt.Errorf("%s=%q: want an http(s) URL", EnvURL, url)
	}
	reader := &OpenAIReader{BaseURL: url, Model: model, APIKey: getenv(EnvAPIKey)}
	switch f := strings.TrimSpace(getenv(EnvMaxTokens)); f {
	case "", "max_tokens", "max_completion_tokens":
		reader.MaxTokensField = f
	default:
		return nil, fmt.Errorf("%s=%q: want max_tokens or max_completion_tokens", EnvMaxTokens, f)
	}
	if v := strings.TrimSpace(getenv(EnvNoThinking)); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("%s=%q: want a boolean", EnvNoThinking, v)
		}
		reader.DisableThinking = b
	}
	cfg := Config{Reader: reader}
	prob := func(key string, dst *float64) error {
		v := strings.TrimSpace(getenv(key))
		if v == "" {
			return nil
		}
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || f <= 0 || f > 1 {
			return fmt.Errorf("%s=%q: want a probability in (0,1]", key, v)
		}
		*dst = f
		return nil
	}
	if err := prob(EnvMinConfidence, &cfg.MinConfidence); err != nil {
		return nil, err
	}
	if err := prob(EnvMinMass, &cfg.MinDeclaredMass); err != nil {
		return nil, err
	}
	if v := strings.TrimSpace(getenv(EnvTimeout)); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil || d <= 0 {
			return nil, fmt.Errorf("%s=%q: want a positive duration", EnvTimeout, v)
		}
		cfg.Timeout = d
	}
	return New(cfg)
}
