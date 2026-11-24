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
type OpenAIReader struct {
	BaseURL string // e.g. http://127.0.0.1:8000; "/v1/chat/completions" is appended
	Model   string
	APIKey  string
	Client  *http.Client
}

type chatReq struct {
	Model       string    `json:"model"`
	Messages    []chatMsg `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float64   `json:"temperature"`
	Logprobs    bool      `json:"logprobs"`
	TopLogprobs int       `json:"top_logprobs"`
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
	body, err := json.Marshal(chatReq{
		Model: r.Model,
		Messages: []chatMsg{
			{Role: "system", Content: "Reply with exactly one capital letter."},
			{Role: "user", Content: prompt},
		},
		MaxTokens:   1,
		Temperature: 0,
		Logprobs:    true,
		TopLogprobs: 20,
	})
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(r.BaseURL, "/") + "/v1/chat/completions"
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
		return nil, fmt.Errorf("decision endpoint status %d", resp.StatusCode)
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
		label := strings.TrimSpace(tl.Token)
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

// Environment knobs for FromEnv.
const (
	EnvDecider       = "FAK_EFFORT_DECIDER" // "decision-model" enables; empty/"off" disables
	EnvURL           = "FAK_EFFORT_DECIDER_URL"
	EnvModel         = "FAK_EFFORT_DECIDER_MODEL"
	EnvAPIKey        = "FAK_EFFORT_DECIDER_API_KEY"
	EnvMinConfidence = "FAK_EFFORT_DECIDER_MIN_CONFIDENCE"
	EnvTimeout       = "FAK_EFFORT_DECIDER_TIMEOUT" // Go duration, e.g. 500ms
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
	cfg := Config{Reader: &OpenAIReader{BaseURL: url, Model: model, APIKey: getenv(EnvAPIKey)}}
	if v := strings.TrimSpace(getenv(EnvMinConfidence)); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || f <= 0 || f > 1 {
			return nil, fmt.Errorf("%s=%q: want a probability in (0,1]", EnvMinConfidence, v)
		}
		cfg.MinConfidence = f
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
