package cachevalue

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

const Qwen38ColdArmReceiptSchema = "fak.qwen38_cold_arm_receipt.v1"

// Qwen38ColdArmConfig contains only endpoint declarations and workload input. Measurements
// belong exclusively to Qwen38ColdArmReceipt so a prepared observation cannot masquerade as a
// runner result.
type Qwen38ColdArmConfig struct {
	Endpoint  string             `json:"endpoint"`
	Model     string             `json:"model"`
	APIKeyEnv string             `json:"api_key_env"`
	Trial     Qwen38ColdArmTrial `json:"trial"`
}

// Qwen38ColdArmTrial is the single cold text request this spine executes.
type Qwen38ColdArmTrial struct {
	ID        string `json:"id"`
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"`
}

type Qwen38ColdArmEvidence struct {
	Scope            string `json:"scope"`
	PerformanceClaim bool   `json:"performance_claim"`
	Reason           string `json:"reason"`
}

type Qwen38ColdArmCommand struct {
	Method string `json:"method"`
	Path   string `json:"path"`
}

// Qwen38ColdArmUsage retains presence with pointers: a missing provider field remains absent
// instead of being conflated with a measured zero.
type Qwen38ColdArmUsage struct {
	PromptTokens     *int64 `json:"prompt_tokens,omitempty"`
	CompletionTokens *int64 `json:"completion_tokens,omitempty"`
	TotalTokens      *int64 `json:"total_tokens,omitempty"`
	CachedTokens     *int64 `json:"cached_tokens,omitempty"`
}

type Qwen38UnavailableResource struct {
	Status string `json:"status"`
	Reason string `json:"reason"`
}

// Qwen38ColdArmReceipt is deliberately not a cache-campaign result. It is one versioned raw
// exchange plus fields derived at the HTTP boundary, suitable as an input to later aggregation.
type Qwen38ColdArmReceipt struct {
	Schema       string                    `json:"schema"`
	Evidence     Qwen38ColdArmEvidence     `json:"evidence"`
	Endpoint     string                    `json:"endpoint"`
	Model        string                    `json:"model"`
	ModelOwner   string                    `json:"model_owner"`
	Engine       string                    `json:"engine"`
	Planner      string                    `json:"planner"`
	APIKeyEnv    string                    `json:"api_key_env"`
	TrialID      string                    `json:"trial_id"`
	Command      Qwen38ColdArmCommand      `json:"command"`
	HTTPStatus   int                       `json:"http_status"`
	WallMS       float64                   `json:"wall_ms"`
	Usage        Qwen38ColdArmUsage        `json:"usage"`
	OutputSHA256 string                    `json:"output_sha256"`
	Memory       Qwen38UnavailableResource `json:"memory"`
	Energy       Qwen38UnavailableResource `json:"energy"`
	RawRequest   json.RawMessage           `json:"raw_request"`
	RawResponse  json.RawMessage           `json:"raw_response"`
}

// Qwen38ColdArmRunner owns the request clock and OpenAI-compatible exchange. A nil Client uses a
// bounded default; callers cannot inject a clock, usage, hash, or resource measurement.
type Qwen38ColdArmRunner struct {
	Client *http.Client
}

type qwen38EndpointIdentity struct {
	Engine, Planner string
}

func (r Qwen38ColdArmRunner) Run(ctx context.Context, cfg Qwen38ColdArmConfig) (Qwen38ColdArmReceipt, error) {
	var receipt Qwen38ColdArmReceipt
	endpoint, err := validateQwen38ColdArmConfig(cfg)
	if err != nil {
		return receipt, err
	}
	apiKey, ok := os.LookupEnv(cfg.APIKeyEnv)
	if !ok || strings.TrimSpace(apiKey) == "" {
		return receipt, fmt.Errorf("Qwen3.8 cold arm: API key environment %q is unset or empty", cfg.APIKeyEnv)
	}
	client := r.Client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	identity, err := qwen38PreflightNativeEndpoint(ctx, client, endpoint, cfg.Model, apiKey)
	if err != nil {
		return receipt, err
	}
	owner, err := qwen38PreflightExactModel(ctx, client, endpoint, cfg.Model, apiKey)
	if err != nil {
		return receipt, err
	}

	type wireMessage struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	wireRequest := struct {
		Model              string          `json:"model"`
		Messages           []wireMessage   `json:"messages"`
		Temperature        int             `json:"temperature"`
		MaxTokens          int             `json:"max_tokens"`
		Stream             bool            `json:"stream"`
		ChatTemplateKwargs map[string]bool `json:"chat_template_kwargs"`
	}{
		Model: cfg.Model, Temperature: 0, MaxTokens: cfg.Trial.MaxTokens, Stream: false,
		ChatTemplateKwargs: map[string]bool{"enable_thinking": false},
	}
	wireRequest.Messages = append(wireRequest.Messages, wireMessage{Role: "user", Content: cfg.Trial.Prompt})
	rawRequest, err := json.Marshal(wireRequest)
	if err != nil {
		return receipt, err
	}
	requestURL := endpoint + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, requestURL, bytes.NewReader(rawRequest))
	if err != nil {
		return receipt, err
	}
	qwen38SetAuth(req, apiKey)
	req.Header.Set("Content-Type", "application/json")
	started := time.Now()
	rsp, err := client.Do(req)
	if err != nil {
		return receipt, fmt.Errorf("Qwen3.8 cold arm request: %w", err)
	}
	defer rsp.Body.Close()
	rawResponse, readErr := io.ReadAll(io.LimitReader(rsp.Body, 4<<20))
	wall := time.Since(started)
	if readErr != nil {
		return receipt, fmt.Errorf("Qwen3.8 cold arm response: %w", readErr)
	}
	if rsp.StatusCode/100 != 2 {
		return receipt, fmt.Errorf("Qwen3.8 cold arm HTTP %d: %s", rsp.StatusCode, strings.TrimSpace(string(rawResponse)))
	}
	var response struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage json.RawMessage `json:"usage"`
	}
	if err := json.Unmarshal(rawResponse, &response); err != nil {
		return receipt, fmt.Errorf("Qwen3.8 cold arm response JSON: %w", err)
	}
	if response.Model != cfg.Model {
		return receipt, fmt.Errorf("Qwen3.8 cold arm response model mismatch: got %q want exact %q", response.Model, cfg.Model)
	}
	if len(response.Choices) == 0 {
		return receipt, errors.New("Qwen3.8 cold arm response has no choices")
	}
	usage, err := decodeQwen38ColdArmUsage(response.Usage)
	if err != nil {
		return receipt, err
	}
	semanticOutput := strings.TrimSpace(response.Choices[0].Message.Content)
	outputSum := sha256.Sum256([]byte(semanticOutput))
	unavailable := Qwen38UnavailableResource{Status: "N/A", Reason: "unsupported by the OpenAI chat-completions wire"}
	receipt = Qwen38ColdArmReceipt{
		Schema: Qwen38ColdArmReceiptSchema,
		Evidence: Qwen38ColdArmEvidence{
			Scope: "partial-cold-arm", PerformanceClaim: false,
			Reason: "one cold endpoint observation; no comparison, campaign verdict, or hardware-performance claim",
		},
		Endpoint: endpoint, Model: cfg.Model, ModelOwner: owner, Engine: identity.Engine, Planner: identity.Planner,
		APIKeyEnv: cfg.APIKeyEnv, TrialID: cfg.Trial.ID,
		Command:    Qwen38ColdArmCommand{Method: http.MethodPost, Path: "/v1/chat/completions"},
		HTTPStatus: rsp.StatusCode, WallMS: float64(wall.Nanoseconds()) / float64(time.Millisecond), Usage: usage,
		OutputSHA256: "sha256:" + hex.EncodeToString(outputSum[:]), Memory: unavailable, Energy: unavailable,
		RawRequest: append(json.RawMessage(nil), rawRequest...), RawResponse: append(json.RawMessage(nil), rawResponse...),
	}
	return receipt, nil
}

func validateQwen38ColdArmConfig(cfg Qwen38ColdArmConfig) (string, error) {
	endpoint := strings.TrimRight(strings.TrimSpace(cfg.Endpoint), "/")
	parsed, err := url.Parse(endpoint)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", fmt.Errorf("Qwen3.8 cold arm: endpoint must be an HTTP(S) service root")
	}
	if parsed.Path != "" {
		return "", fmt.Errorf("Qwen3.8 cold arm: endpoint must not include an API path")
	}
	if strings.TrimSpace(cfg.Model) == "" || strings.TrimSpace(cfg.APIKeyEnv) == "" {
		return "", errors.New("Qwen3.8 cold arm: exact model and API-key environment name are required")
	}
	if strings.TrimSpace(cfg.Trial.ID) == "" || strings.TrimSpace(cfg.Trial.Prompt) == "" || cfg.Trial.MaxTokens <= 0 {
		return "", errors.New("Qwen3.8 cold arm: trial id, prompt, and positive max tokens are required")
	}
	return endpoint, nil
}

func qwen38PreflightNativeEndpoint(ctx context.Context, client *http.Client, endpoint, model, apiKey string) (qwen38EndpointIdentity, error) {
	var identity qwen38EndpointIdentity
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"/healthz", nil)
	if err != nil {
		return identity, err
	}
	qwen38SetAuth(req, apiKey)
	rsp, err := client.Do(req)
	if err != nil {
		return identity, fmt.Errorf("Qwen3.8 cold arm fak-native preflight: %w", err)
	}
	defer rsp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(rsp.Body, 1<<20))
	if err != nil {
		return identity, err
	}
	if rsp.StatusCode/100 != 2 {
		return identity, fmt.Errorf("Qwen3.8 cold arm fak-native preflight HTTP %d", rsp.StatusCode)
	}
	var health struct {
		OK      bool   `json:"ok"`
		Engine  string `json:"engine"`
		Planner string `json:"planner"`
		Model   string `json:"model"`
	}
	if err := json.Unmarshal(raw, &health); err != nil {
		return identity, fmt.Errorf("Qwen3.8 cold arm fak-native preflight: %w", err)
	}
	if !health.OK || health.Engine != "inkernel" || health.Planner != "inkernel" {
		return identity, fmt.Errorf("Qwen3.8 cold arm requires fak-native in-kernel execution: ok=%v engine=%q planner=%q", health.OK, health.Engine, health.Planner)
	}
	if health.Model != model {
		return identity, fmt.Errorf("Qwen3.8 cold arm health model mismatch: got %q want exact %q", health.Model, model)
	}
	return qwen38EndpointIdentity{Engine: health.Engine, Planner: health.Planner}, nil
}

func qwen38PreflightExactModel(ctx context.Context, client *http.Client, endpoint, model, apiKey string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"/v1/models", nil)
	if err != nil {
		return "", err
	}
	qwen38SetAuth(req, apiKey)
	rsp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("Qwen3.8 cold arm model identity preflight: %w", err)
	}
	defer rsp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(rsp.Body, 1<<20))
	if err != nil {
		return "", err
	}
	if rsp.StatusCode/100 != 2 {
		return "", fmt.Errorf("Qwen3.8 cold arm model identity preflight HTTP %d", rsp.StatusCode)
	}
	var listing struct {
		Data []struct {
			ID      string `json:"id"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &listing); err != nil {
		return "", fmt.Errorf("Qwen3.8 cold arm model identity preflight: %w", err)
	}
	matches, owner := 0, ""
	for _, advertised := range listing.Data {
		if advertised.ID == model {
			matches++
			owner = advertised.OwnedBy
		}
	}
	if matches != 1 || owner != "fak" {
		return "", fmt.Errorf("Qwen3.8 cold arm model identity preflight: exact model %q owned by fak occurred %d times (owner=%q)", model, matches, owner)
	}
	return owner, nil
}

func decodeQwen38ColdArmUsage(raw json.RawMessage) (Qwen38ColdArmUsage, error) {
	var usage Qwen38ColdArmUsage
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return usage, nil
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return usage, fmt.Errorf("Qwen3.8 cold arm usage: %w", err)
	}
	decode := func(name string, dst **int64) error {
		value, ok := fields[name]
		if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return nil
		}
		var n int64
		if err := json.Unmarshal(value, &n); err != nil || n < 0 {
			return fmt.Errorf("Qwen3.8 cold arm usage.%s must be a non-negative integer", name)
		}
		*dst = &n
		return nil
	}
	if err := decode("prompt_tokens", &usage.PromptTokens); err != nil {
		return usage, err
	}
	if err := decode("completion_tokens", &usage.CompletionTokens); err != nil {
		return usage, err
	}
	if err := decode("total_tokens", &usage.TotalTokens); err != nil {
		return usage, err
	}
	if details := fields["prompt_tokens_details"]; len(details) != 0 && !bytes.Equal(bytes.TrimSpace(details), []byte("null")) {
		cached, present, err := decodeQwen38CachedDetail(details, "prompt_tokens_details")
		if err != nil {
			return usage, err
		}
		if present {
			usage.CachedTokens = &cached
		}
	}
	if usage.CachedTokens == nil {
		if details := fields["input_tokens_details"]; len(details) != 0 && !bytes.Equal(bytes.TrimSpace(details), []byte("null")) {
			cached, present, err := decodeQwen38CachedDetail(details, "input_tokens_details")
			if err != nil {
				return usage, err
			}
			if present {
				usage.CachedTokens = &cached
			}
		}
	}
	if usage.CachedTokens == nil {
		for _, name := range []string{"cache_read_input_tokens", "prompt_cache_hit_tokens"} {
			cached := fields[name]
			if len(cached) == 0 || bytes.Equal(bytes.TrimSpace(cached), []byte("null")) {
				continue
			}
			var n int64
			if err := json.Unmarshal(cached, &n); err != nil || n < 0 {
				return usage, fmt.Errorf("Qwen3.8 cold arm usage.%s must be a non-negative integer", name)
			}
			usage.CachedTokens = &n
			break
		}
	}
	return usage, nil
}

func decodeQwen38CachedDetail(raw json.RawMessage, field string) (int64, bool, error) {
	var details map[string]json.RawMessage
	if err := json.Unmarshal(raw, &details); err != nil {
		return 0, false, fmt.Errorf("Qwen3.8 cold arm usage.%s: %w", field, err)
	}
	cached := details["cached_tokens"]
	if len(cached) == 0 || bytes.Equal(bytes.TrimSpace(cached), []byte("null")) {
		return 0, false, nil
	}
	var n int64
	if err := json.Unmarshal(cached, &n); err != nil || n < 0 {
		return 0, false, fmt.Errorf("Qwen3.8 cold arm usage.%s.cached_tokens must be a non-negative integer", field)
	}
	return n, true, nil
}

func qwen38SetAuth(req *http.Request, apiKey string) {
	req.Header.Set("Authorization", "Bearer "+apiKey)
}
