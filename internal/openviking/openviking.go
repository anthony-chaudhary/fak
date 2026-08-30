package openviking

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const maxResponseBytes = 8 << 20

const defaultHTTPTimeout = 30 * time.Second

const (
	CodeUnavailable     = "UNAVAILABLE"
	CodeInvalidResponse = "INVALID_RESPONSE"
	CodeHTTPError       = "HTTP_ERROR"
)

// Config describes one optional OpenViking service boundary. Empty identity
// fields are omitted rather than inferred from the environment or another
// backend.
type Config struct {
	BaseURL    string
	APIKey     string
	Account    string
	User       string
	ActorPeer  string
	HTTPClient *http.Client
}

// Client calls the small public OpenViking REST contract fak integrates with.
// It performs no retries and has no fallback backend.
type Client struct {
	baseURL   string
	apiKey    string
	account   string
	user      string
	actorPeer string
	http      *http.Client
}

// ResponseMeta is the optional metadata carried by an OpenViking response
// envelope.
type ResponseMeta struct {
	Telemetry json.RawMessage
	Profile   []string
}

// APIError retains both the HTTP status and OpenViking business error code.
// Its text and Details are scrubbed of the configured API key before return.
type APIError struct {
	Operation  string
	HTTPStatus int
	Code       string
	Message    string
	Details    json.RawMessage
	cause      error
}

func (e *APIError) Error() string {
	if e == nil {
		return "<nil>"
	}
	prefix := "openviking"
	if e.Operation != "" {
		prefix += " " + e.Operation
	}
	if e.HTTPStatus != 0 {
		prefix += fmt.Sprintf(" HTTP %d", e.HTTPStatus)
	}
	if e.Code != "" {
		prefix += " " + e.Code
	}
	if e.Message != "" {
		prefix += ": " + e.Message
	}
	return prefix
}

func (e *APIError) Unwrap() error { return e.cause }

// HealthStatus is the raw, deliberately unauthenticated /health response.
type HealthStatus struct {
	Status string `json:"status"`
}

// SearchContextRequest asks OpenViking to assemble bounded, injection-ready
// context. Mode is fixed to "context" by SearchContext.
type SearchContextRequest struct {
	Query             string         `json:"query"`
	SessionID         string         `json:"session_id,omitempty"`
	Limit             int            `json:"limit,omitempty"`
	ScoreThreshold    *float64       `json:"score_threshold,omitempty"`
	QueryExpansion    string         `json:"query_expansion,omitempty"`
	MaxTokens         int            `json:"max_tokens,omitempty"`
	Quotas            map[string]int `json:"quotas,omitempty"`
	Purpose           string         `json:"purpose,omitempty"`
	DedupTurns        int            `json:"dedup_turns,omitempty"`
	ExcludeURIs       []string       `json:"exclude_uris,omitempty"`
	PeerScope         string         `json:"peer_scope,omitempty"`
	Rewrite           *bool          `json:"rewrite,omitempty"`
	RewriteMaxBullets int            `json:"rewrite_max_bullets,omitempty"`
}

// ContextEntry is one item in an assembled context response.
type ContextEntry struct {
	URI      string  `json:"uri,omitempty"`
	Category string  `json:"category,omitempty"`
	Score    float64 `json:"score,omitempty"`
	Detail   string  `json:"detail,omitempty"`
	Text     string  `json:"text,omitempty"`
	Origin   string  `json:"origin,omitempty"`
}

// SearchContextResult is OpenViking's bounded context result.
type SearchContextResult struct {
	Entries  []ContextEntry `json:"entries,omitempty"`
	Rendered string         `json:"rendered,omitempty"`
	Digest   string         `json:"digest,omitempty"`
	Stats    map[string]any `json:"stats,omitempty"`
	Meta     ResponseMeta   `json:"-"`
}

// Message is one captured agent message. Content or Parts must be supplied.
type Message struct {
	Role             string           `json:"role"`
	Content          *string          `json:"content,omitempty"`
	Parts            []map[string]any `json:"parts,omitempty"`
	CreatedAt        string           `json:"created_at,omitempty"`
	PeerID           string           `json:"peer_id,omitempty"`
	TurnID           string           `json:"turn_id,omitempty"`
	MessageKind      string           `json:"message_kind,omitempty"`
	SourceMessageIDs []string         `json:"source_message_ids,omitempty"`
}

// BatchMessagesRequest captures up to 100 messages in one call.
type BatchMessagesRequest struct {
	Messages []Message `json:"messages"`
}

// BatchMessagesResult reports the post-write session state.
type BatchMessagesResult struct {
	SessionID     string       `json:"session_id,omitempty"`
	MessageCount  int          `json:"message_count,omitempty"`
	Added         int          `json:"added,omitempty"`
	PendingTokens int          `json:"pending_tokens,omitempty"`
	Meta          ResponseMeta `json:"-"`
}

// CommitRequest controls the retained live tail while OpenViking archives and
// schedules memory extraction.
type CommitRequest struct {
	KeepRecentCount            int    `json:"keep_recent_count,omitempty"`
	RetentionMode              string `json:"retention_mode,omitempty"`
	KeepRecentTurnCount        *int   `json:"keep_recent_turn_count,omitempty"`
	RetainedMessageTokenBudget *int   `json:"retained_message_token_budget,omitempty"`
	MinRawTailSteps            *int   `json:"min_raw_tail_steps,omitempty"`
}

// CommitResult contains the stable identifiers commonly returned by a commit.
type CommitResult struct {
	TaskID    string       `json:"task_id,omitempty"`
	SessionID string       `json:"session_id,omitempty"`
	ArchiveID string       `json:"archive_id,omitempty"`
	Status    string       `json:"status,omitempty"`
	Meta      ResponseMeta `json:"-"`
}

type responseEnvelope struct {
	Status    string          `json:"status"`
	Result    json.RawMessage `json:"result"`
	Error     *errorInfo      `json:"error"`
	Telemetry json.RawMessage `json:"telemetry"`
	Profile   []string        `json:"profile"`
}

type errorInfo struct {
	Code    string          `json:"code"`
	Message string          `json:"message"`
	Details json.RawMessage `json:"details"`
}

// NewClient validates an HTTP(S) service root before any network I/O.
func NewClient(cfg Config) (*Client, error) {
	base, err := validateBaseURL(cfg.BaseURL)
	if err != nil {
		return nil, err
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = newDefaultHTTPClient()
	}
	return &Client{
		baseURL: base, apiKey: cfg.APIKey, account: cfg.Account,
		user: cfg.User, actorPeer: cfg.ActorPeer, http: hc,
	}, nil
}

func newDefaultHTTPClient() *http.Client {
	return &http.Client{Timeout: defaultHTTPTimeout}
}

// Health probes the raw health endpoint.
func (c *Client) Health(ctx context.Context) (HealthStatus, error) {
	var out HealthStatus
	raw, status, err := c.exchange(ctx, "health", http.MethodGet, "/health", nil, true)
	if err != nil {
		return out, err
	}
	if status < 200 || status >= 300 {
		return out, c.responseError("health", status, raw)
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return out, c.invalidResponse("health", status, "health response is not valid JSON")
	}
	if out.Status == "" {
		return out, c.invalidResponse("health", status, "health response has no status")
	}
	return out, nil
}

// SearchContext calls OpenViking's context-mode retrieval endpoint.
func (c *Client) SearchContext(ctx context.Context, in SearchContextRequest) (SearchContextResult, error) {
	var out SearchContextResult
	if strings.TrimSpace(in.Query) == "" {
		return out, errors.New("openviking search context: query is required")
	}
	body := struct {
		Mode string `json:"mode"`
		SearchContextRequest
	}{Mode: "context", SearchContextRequest: in}
	meta, err := c.call(ctx, "search context", "/api/v1/search/search", body, &out)
	out.Meta = meta
	return out, err
}

// BatchMessages appends messages to a session, which OpenViking may create on
// first write.
func (c *Client) BatchMessages(ctx context.Context, sessionID string, in BatchMessagesRequest) (BatchMessagesResult, error) {
	var out BatchMessagesResult
	if err := validateSessionID(sessionID); err != nil {
		return out, err
	}
	if len(in.Messages) == 0 || len(in.Messages) > 100 {
		return out, errors.New("openviking batch messages: messages must contain 1 to 100 items")
	}
	for i, message := range in.Messages {
		if strings.TrimSpace(message.Role) == "" {
			return out, fmt.Errorf("openviking batch messages: message %d role is required", i)
		}
		if message.Content == nil && len(message.Parts) == 0 {
			return out, fmt.Errorf("openviking batch messages: message %d requires content or parts", i)
		}
	}
	path := "/api/v1/sessions/" + sessionID + "/messages/batch"
	meta, err := c.call(ctx, "batch messages", path, in, &out)
	out.Meta = meta
	return out, err
}

// Commit archives a session tail and schedules OpenViking memory extraction.
func (c *Client) Commit(ctx context.Context, sessionID string, in CommitRequest) (CommitResult, error) {
	var out CommitResult
	if err := validateSessionID(sessionID); err != nil {
		return out, err
	}
	path := "/api/v1/sessions/" + sessionID + "/commit"
	meta, err := c.call(ctx, "commit", path, in, &out)
	out.Meta = meta
	return out, err
}

func (c *Client) call(ctx context.Context, operation, path string, in, out any) (ResponseMeta, error) {
	var meta ResponseMeta
	body, err := json.Marshal(in)
	if err != nil {
		return meta, fmt.Errorf("openviking %s: encode request: %w", operation, err)
	}
	raw, status, err := c.exchange(ctx, operation, http.MethodPost, path, body, true)
	if err != nil {
		return meta, err
	}
	var envelope responseEnvelope
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return meta, c.invalidResponse(operation, status, "response is not a valid OpenViking envelope")
	}
	meta.Telemetry = cloneRaw(envelope.Telemetry)
	meta.Profile = append([]string(nil), envelope.Profile...)
	if status < 200 || status >= 300 || envelope.Status == "error" {
		return meta, envelopeAPIError(operation, status, envelope)
	}
	if envelope.Status != "ok" {
		return meta, c.invalidResponse(operation, status, "response envelope status is not ok or error")
	}
	if len(envelope.Result) != 0 && string(envelope.Result) != "null" {
		if err := json.Unmarshal(envelope.Result, out); err != nil {
			return meta, c.invalidResponse(operation, status, "response result does not match the public contract")
		}
	}
	return meta, nil
}

func (c *Client) exchange(ctx context.Context, operation, method, path string, body []byte, identity bool) ([]byte, int, error) {
	if ctx == nil {
		return nil, 0, errors.New("openviking: nil context")
	}
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, reader)
	if err != nil {
		return nil, 0, &APIError{Operation: operation, Code: CodeInvalidResponse, Message: "could not build request"}
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Accept", "application/json")
	if identity {
		setIfPresent(req.Header, "X-API-Key", c.apiKey)
		setIfPresent(req.Header, "X-OpenViking-Account", c.account)
		setIfPresent(req.Header, "X-OpenViking-User", c.user)
		setIfPresent(req.Header, "X-OpenViking-Actor-Peer", c.actorPeer)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		if contextFailure := ctx.Err(); contextFailure != nil {
			return nil, 0, &APIError{Operation: operation, Code: CodeUnavailable, Message: contextFailure.Error(), cause: contextFailure}
		}
		message := c.redact(err.Error())
		return nil, 0, &APIError{Operation: operation, Code: CodeUnavailable, Message: message, cause: errors.New(message)}
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes+1))
	if err != nil {
		return nil, resp.StatusCode, &APIError{Operation: operation, HTTPStatus: resp.StatusCode, Code: CodeUnavailable, Message: "could not read response"}
	}
	if len(raw) > maxResponseBytes {
		return nil, resp.StatusCode, &APIError{Operation: operation, HTTPStatus: resp.StatusCode, Code: CodeInvalidResponse, Message: "response exceeds 8 MiB limit"}
	}
	return c.redactBytes(raw), resp.StatusCode, nil
}

func (c *Client) responseError(operation string, status int, raw []byte) error {
	var envelope responseEnvelope
	if json.Unmarshal(raw, &envelope) == nil && envelope.Error != nil {
		return envelopeAPIError(operation, status, envelope)
	}
	return &APIError{Operation: operation, HTTPStatus: status, Code: CodeHTTPError, Message: http.StatusText(status)}
}

func (c *Client) invalidResponse(operation string, status int, message string) error {
	return &APIError{Operation: operation, HTTPStatus: status, Code: CodeInvalidResponse, Message: message}
}

func envelopeAPIError(operation string, status int, envelope responseEnvelope) error {
	code := CodeHTTPError
	message := http.StatusText(status)
	var details json.RawMessage
	if envelope.Error != nil {
		if envelope.Error.Code != "" {
			code = envelope.Error.Code
		}
		if envelope.Error.Message != "" {
			message = envelope.Error.Message
		}
		details = cloneRaw(envelope.Error.Details)
	}
	return &APIError{Operation: operation, HTTPStatus: status, Code: code, Message: message, Details: details}
}

func validateBaseURL(raw string) (string, error) {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return "", errors.New("openviking: base URL is required")
	}
	parsed, err := url.Parse(trimmed)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return "", errors.New("openviking: base URL must be a valid HTTP(S) service root")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" || (parsed.Path != "" && parsed.Path != "/") {
		return "", errors.New("openviking: base URL must not contain credentials, a path, query, or fragment")
	}
	return strings.TrimRight(trimmed, "/"), nil
}

func validateSessionID(sessionID string) error {
	if sessionID == "" || sessionID == "." || sessionID == ".." || url.PathEscape(sessionID) != sessionID || strings.Contains(sessionID, "\\") {
		return errors.New("openviking: session ID must be one non-empty URL path segment")
	}
	return nil
}

func setIfPresent(header http.Header, name, value string) {
	if value != "" {
		header.Set(name, value)
	}
}

func cloneRaw(raw json.RawMessage) json.RawMessage {
	return append(json.RawMessage(nil), raw...)
}

func (c *Client) redact(value string) string {
	if c.apiKey == "" {
		return value
	}
	return strings.ReplaceAll(value, c.apiKey, "[REDACTED]")
}

func (c *Client) redactBytes(value []byte) []byte {
	if c.apiKey == "" {
		return value
	}
	return bytes.ReplaceAll(value, []byte(c.apiKey), []byte("[REDACTED]"))
}
