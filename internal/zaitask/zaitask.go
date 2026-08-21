package zaitask

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

const (
	DefaultBaseURL = "https://api.z.ai/api/coding/paas/v4"
	DefaultModel   = "glm-5.2"
	maxTaskBytes   = 1 << 20
)

// Suitability reports whether the fleet taxonomy considers a task safe for the
// bounded GLM-5 path. Explicit light/gardening/tier-3 classes are accepted;
// frontier and apex work fail closed instead of silently using a weaker model.
type Suitability struct {
	Suitable   bool   `json:"suitable"`
	Class      string `json:"class"`
	TargetTier int    `json:"target_tier"`
	Reason     string `json:"reason"`
}

func Classify(prompt, taskClass string) Suitability {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return Suitability{Class: "invalid", TargetTier: fleetaccounts.TierFrontier, Reason: "no task content"}
	}
	if len(prompt) >= maxTaskBytes {
		return Suitability{Class: "invalid", TargetTier: fleetaccounts.TierFrontier, Reason: "task exceeds the bounded routing input limit"}
	}
	switch taskClass {
	case "bounded":
		return classifyBounded(prompt)
	case "frontier":
		return Suitability{Class: "frontier", TargetTier: fleetaccounts.TierFrontier, Reason: "frontier work is reserved for a frontier model"}
	case "", "auto":
		if strings.HasPrefix(strings.ToLower(prompt), "summarize ") {
			return classifyBounded(prompt)
		}
	default:
		if !supportedFleetClass(taskClass) {
			return Suitability{Class: "invalid", TargetTier: fleetaccounts.TierFrontier, Reason: "unsupported task class"}
		}
	}
	got := fleetaccounts.ClassifyTask(prompt, taskClass, fleetaccounts.DefaultPolicy())
	return Suitability{Suitable: got.TargetTier >= fleetaccounts.TierLight, Class: got.Class, TargetTier: got.TargetTier, Reason: got.Reason}
}

func classifyBounded(prompt string) Suitability {
	got := fleetaccounts.ClassifyTask(prompt, "light", fleetaccounts.DefaultPolicy())
	return Suitability{Suitable: true, Class: got.Class, TargetTier: got.TargetTier, Reason: "bounded task: " + got.Reason}
}

func supportedFleetClass(taskClass string) bool {
	switch taskClass {
	case "light", "easy", "tier2", "t2", "2",
		"hard", "default", "tier1", "t1", "1",
		"tier3", "t3", "3",
		"gardening", "garden", "maintenance", "maint", "cleanup", "chore", "triage",
		"engineering", "eng", "dev", "feature", "implementation",
		"tier0", "t0", "0", "apex", "fable", "fable5", "fable-5":
		return true
	default:
		return false
	}
}

type Client struct {
	BaseURL    string
	APIKey     string
	HTTPClient *http.Client
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type Result struct {
	Content   string `json:"content"`
	Model     string `json:"model"`
	RequestID string `json:"request_id"`
	Usage     Usage  `json:"usage"`
	LatencyMS int64  `json:"latency_ms"`
}

type apiRequest struct {
	Model     string    `json:"model"`
	Messages  []message `json:"messages"`
	MaxTokens int       `json:"max_tokens,omitempty"`
	Stream    bool      `json:"stream"`
	Thinking  *thinking `json:"thinking,omitempty"`
}

type thinking struct {
	Type string `json:"type"`
}
type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
type apiResponse struct {
	Choices []struct {
		Message message `json:"message"`
	} `json:"choices"`
	Model     string `json:"model"`
	RequestID string `json:"request_id"`
	Usage     Usage  `json:"usage"`
	Error     *struct {
		Message string `json:"message"`
		Code    any    `json:"code"`
	} `json:"error,omitempty"`
}

func (c Client) Run(ctx context.Context, prompt, model string, maxTokens int) (Result, error) {
	if strings.TrimSpace(c.APIKey) == "" {
		return Result{}, errors.New("ZAI API key is required; recovery: set Client.APIKey from a ZAI credential")
	}
	if strings.TrimSpace(prompt) == "" {
		return Result{}, errors.New("prompt is required; recovery: provide a non-empty task prompt")
	}
	if model == "" {
		model = DefaultModel
	}
	base := strings.TrimRight(c.BaseURL, "/")
	if base == "" {
		base = DefaultBaseURL
	}
	body, err := json.Marshal(apiRequest{Model: model, Messages: []message{{Role: "user", Content: prompt}}, MaxTokens: maxTokens, Stream: false, Thinking: &thinking{Type: "disabled"}})
	if err != nil {
		return Result{}, fmt.Errorf("encode zai request; recovery: report the request encoder defect: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Result{}, fmt.Errorf("build zai request; recovery: set Client.BaseURL to a valid HTTP(S) URL: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")
	hc := c.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 2 * time.Minute}
	}
	start := time.Now()
	resp, err := hc.Do(req)
	latency := time.Since(start).Milliseconds()
	if err != nil {
		return Result{}, fmt.Errorf("zai request failed; recovery: check endpoint/network availability and retry: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, (8<<20)+1))
	if err != nil {
		return Result{}, fmt.Errorf("read zai response; recovery: retry the request or inspect the endpoint transport: %w", err)
	}
	if len(raw) > 8<<20 {
		return Result{}, errors.New("zai response exceeds 8 MiB limit; recovery: request a smaller max_tokens value")
	}
	var decoded apiResponse
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return Result{}, fmt.Errorf("zai HTTP %d returned invalid JSON; recovery: retry or inspect provider compatibility: %w", resp.StatusCode, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg := strings.TrimSpace(string(raw))
		if decoded.Error != nil && decoded.Error.Message != "" {
			msg = decoded.Error.Message
		}
		return Result{}, fmt.Errorf("zai HTTP %d: %s; recovery: inspect provider error, quota, and credentials before retrying", resp.StatusCode, msg)
	}
	if len(decoded.Choices) == 0 {
		return Result{}, errors.New("zai response contained no choices; recovery: retry or inspect provider response compatibility")
	}
	return Result{Content: decoded.Choices[0].Message.Content, Model: decoded.Model, RequestID: decoded.RequestID, Usage: decoded.Usage, LatencyMS: latency}, nil
}
