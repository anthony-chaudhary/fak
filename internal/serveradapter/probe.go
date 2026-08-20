package serveradapter

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
)

const (
	ProbeSchema = "fak.server-adapter-probe.v1"
	maxBodySize = 1 << 20
)

// Capability is a protocol behavior directly observed by a successful probe.
type Capability string

const (
	FeatureHealth    Capability = "health"
	FeatureModelList Capability = "models"
	FeatureChat      Capability = "chat_completions"
)

// ProbeKind identifies one required llama-server protocol probe.
type ProbeKind string

const (
	ProbeHealth ProbeKind = "health"
	ProbeModels ProbeKind = "models"
	ProbeChat   ProbeKind = "chat"
)

// FailureKind is the closed readiness failure vocabulary exposed to callers.
type FailureKind string

const (
	FailureNotListening      FailureKind = "not-listening"
	FailureNotReady          FailureKind = "not-ready"
	FailureWrongModel        FailureKind = "wrong-model"
	FailureChatUnavailable   FailureKind = "unsupported-chat"
	FailureMalformedResponse FailureKind = "malformed-response"
)

// HTTPDoer permits protocol probing through an injected HTTP transport.
type HTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// ProbeTarget binds readiness evidence to one loopback endpoint and model alias.
type ProbeTarget struct {
	BaseURL    string `json:"base_url"`
	ModelAlias string `json:"model_alias"`
}

// ProbeObservation is bounded evidence from one protocol request.
type ProbeObservation struct {
	Probe      ProbeKind `json:"probe"`
	Method     string    `json:"method"`
	Path       string    `json:"path"`
	StatusCode int       `json:"status_code,omitempty"`
	BodyDigest string    `json:"body_digest,omitempty"`
}

// ProbeResult reports only capabilities observed before the first failure.
type ProbeResult struct {
	Schema       string             `json:"schema"`
	BaseURL      string             `json:"base_url"`
	ModelAlias   string             `json:"model_alias"`
	Ready        bool               `json:"ready"`
	Capabilities []Capability       `json:"capabilities"`
	Observations []ProbeObservation `json:"observations"`
	ProbeDigest  string             `json:"probe_digest"`
}

// ProbeError reports the failure class and probe without leaking response bodies.
type ProbeError struct {
	Kind       FailureKind
	Probe      ProbeKind
	StatusCode int
	Cause      error
}

func (e *ProbeError) Error() string {
	detail := ""
	if e.StatusCode != 0 {
		detail = fmt.Sprintf(" (HTTP %d)", e.StatusCode)
	}
	if e.Cause != nil {
		detail += ": " + e.Cause.Error()
	}
	return fmt.Sprintf("%s probe: %s%s", e.Probe, e.Kind, detail)
}

func (e *ProbeError) Unwrap() error { return e.Cause }

// ProbeLlamaServer observes health, model identity, and chat support in that
// order. Ready becomes true only after all three probes pass.
func ProbeLlamaServer(ctx context.Context, client HTTPDoer, target ProbeTarget) (ProbeResult, error) {
	result := ProbeResult{
		Schema:       ProbeSchema,
		BaseURL:      strings.TrimRight(target.BaseURL, "/"),
		ModelAlias:   target.ModelAlias,
		Capabilities: []Capability{},
		Observations: []ProbeObservation{},
	}
	if client == nil {
		return finishProbe(result, FailureMalformedResponse), fmt.Errorf("HTTP client is nil")
	}
	if err := validateProbeTarget(target); err != nil {
		return finishProbe(result, FailureMalformedResponse), err
	}

	status, body, err := makeProbeRequest(ctx, client, http.MethodGet, result.BaseURL+"/health", nil)
	result.Observations = append(result.Observations, observation(ProbeHealth, http.MethodGet, "/health", status, body))
	if err != nil {
		return failProbe(result, FailureNotListening, ProbeHealth, status, err)
	}
	if status != http.StatusOK {
		return failProbe(result, FailureNotReady, ProbeHealth, status, nil)
	}
	var health struct {
		Status string `json:"status"`
	}
	if err := decodeObject(body, &health); err != nil {
		return failProbe(result, FailureMalformedResponse, ProbeHealth, status, err)
	}
	if health.Status != "ok" {
		return failProbe(result, FailureNotReady, ProbeHealth, status, nil)
	}
	result.Capabilities = append(result.Capabilities, FeatureHealth)

	status, body, err = makeProbeRequest(ctx, client, http.MethodGet, result.BaseURL+"/v1/models", nil)
	result.Observations = append(result.Observations, observation(ProbeModels, http.MethodGet, "/v1/models", status, body))
	if err != nil {
		return failProbe(result, FailureNotListening, ProbeModels, status, err)
	}
	if status == http.StatusServiceUnavailable {
		return failProbe(result, FailureNotReady, ProbeModels, status, nil)
	}
	if status != http.StatusOK {
		return failProbe(result, FailureMalformedResponse, ProbeModels, status, nil)
	}
	var models struct {
		Object string `json:"object"`
		Data   []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := decodeObject(body, &models); err != nil || models.Object != "list" || len(models.Data) == 0 {
		if err == nil {
			err = errors.New("models response is not a non-empty list")
		}
		return failProbe(result, FailureMalformedResponse, ProbeModels, status, err)
	}
	modelFound := false
	for _, model := range models.Data {
		if model.ID == "" {
			return failProbe(result, FailureMalformedResponse, ProbeModels, status, errors.New("models response contains an empty id"))
		}
		modelFound = modelFound || model.ID == target.ModelAlias
	}
	result.Capabilities = append(result.Capabilities, FeatureModelList)
	if !modelFound {
		return failProbe(result, FailureWrongModel, ProbeModels, status, nil)
	}

	requestBody, err := json.Marshal(struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		MaxTokens int  `json:"max_tokens"`
		Stream    bool `json:"stream"`
	}{
		Model: target.ModelAlias,
		Messages: []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{{Role: "user", Content: "Reply with OK."}},
		MaxTokens: 1,
		Stream:    false,
	})
	if err != nil {
		return failProbe(result, FailureMalformedResponse, ProbeChat, 0, err)
	}
	status, body, err = makeProbeRequest(ctx, client, http.MethodPost, result.BaseURL+"/v1/chat/completions", requestBody)
	result.Observations = append(result.Observations, observation(ProbeChat, http.MethodPost, "/v1/chat/completions", status, body))
	if err != nil {
		return failProbe(result, FailureNotListening, ProbeChat, status, err)
	}
	if status == http.StatusServiceUnavailable {
		return failProbe(result, FailureNotReady, ProbeChat, status, nil)
	}
	if status != http.StatusOK {
		return failProbe(result, FailureChatUnavailable, ProbeChat, status, nil)
	}
	var chat struct {
		Object  string `json:"object"`
		Model   string `json:"model"`
		Choices []struct {
			Message *struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := decodeObject(body, &chat); err != nil || chat.Object != "chat.completion" || len(chat.Choices) == 0 ||
		chat.Choices[0].Message == nil || chat.Choices[0].Message.Role == "" || len(chat.Choices[0].Message.Content) == 0 {
		if err == nil {
			err = errors.New("chat response is missing completion fields")
		}
		return failProbe(result, FailureMalformedResponse, ProbeChat, status, err)
	}
	if chat.Model != "" && chat.Model != target.ModelAlias {
		return failProbe(result, FailureWrongModel, ProbeChat, status, nil)
	}
	result.Capabilities = append(result.Capabilities, FeatureChat)
	result.Ready = true
	return finishProbe(result, ""), nil
}

func validateProbeTarget(target ProbeTarget) error {
	if !validModelAlias(target.ModelAlias) {
		return fmt.Errorf("model alias is invalid")
	}
	parsed, err := url.Parse(target.BaseURL)
	if err != nil || parsed.Scheme != "http" || parsed.Host == "" || parsed.User != nil ||
		parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return fmt.Errorf("base URL must be an absolute loopback HTTP origin")
	}
	host := parsed.Hostname()
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return fmt.Errorf("base URL host must be a loopback IP")
	}
	if parsed.Port() == "" {
		return fmt.Errorf("base URL must include a port")
	}
	return nil
}

func makeProbeRequest(ctx context.Context, client HTTPDoer, method, endpoint string, body []byte) (int, []byte, error) {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint, reader)
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	if resp == nil || resp.Body == nil {
		return 0, nil, errors.New("HTTP client returned an empty response")
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxBodySize+1))
	if err != nil {
		return resp.StatusCode, raw, err
	}
	if len(raw) > maxBodySize {
		return resp.StatusCode, raw, errors.New("response body exceeds probe limit")
	}
	return resp.StatusCode, raw, nil
}

func decodeObject(raw []byte, out any) error {
	if len(bytes.TrimSpace(raw)) == 0 || bytes.TrimSpace(raw)[0] != '{' {
		return errors.New("response is not a JSON object")
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}
	return nil
}

func observation(probe ProbeKind, method, path string, status int, body []byte) ProbeObservation {
	obs := ProbeObservation{Probe: probe, Method: method, Path: path, StatusCode: status}
	if body != nil {
		sum := sha256.Sum256(body)
		obs.BodyDigest = "sha256:" + fmt.Sprintf("%x", sum)
	}
	return obs
}

func failProbe(result ProbeResult, kind FailureKind, probe ProbeKind, status int, cause error) (ProbeResult, error) {
	result.Ready = false
	result = finishProbe(result, kind)
	return result, &ProbeError{Kind: kind, Probe: probe, StatusCode: status, Cause: cause}
}

func finishProbe(result ProbeResult, failure FailureKind) ProbeResult {
	var evidence strings.Builder
	fmt.Fprintf(&evidence, "%s\nbase_url=%s\nmodel_alias=%s\nready=%t\nfailure=%s\n", ProbeSchema, result.BaseURL, result.ModelAlias, result.Ready, failure)
	for _, obs := range result.Observations {
		fmt.Fprintf(&evidence, "%s\t%s\t%s\t%d\t%s\n", obs.Probe, obs.Method, obs.Path, obs.StatusCode, obs.BodyDigest)
	}
	sum := sha256.Sum256([]byte(evidence.String()))
	result.ProbeDigest = "sha256:" + fmt.Sprintf("%x", sum)
	return result
}
