// Package fakclient is the importable Go client SDK for the fak gateway's
// fak-native verdict surface (F-007, issue #205).
//
// fak fronts the OpenAI, Anthropic, and Gemini wires on one port, so a caller
// who already has an official OpenAI / Anthropic / google-genai SDK adopts fak by
// repointing that SDK's base URL at `fak serve` — no new client needed (see
// docs/integrations). The surface that has NO off-the-shelf SDK is the
// fak-native one: the `/v1/fak/*` adjudicate / syscall / admit / changes /
// revoke endpoints that return a verdict as a value. THIS package is the typed
// Go client for that surface.
//
// Every request/response type below mirrors the gateway's wire DTOs
// (internal/gateway/wire.go) field-for-field, by the exact JSON tags the server
// emits; the OpenAPI source of truth those DTOs are documented against is
// docs/fak/openapi.yaml. The wire agreement is gated: client_test.go marshals the
// server's own types into these and back, so an SDK type can never silently lag
// the served surface (the in-code analogue of the route-drift gate in
// internal/gateway/openapi_spec_test.go).
//
// The package is stdlib-only — a consumer `go get`s it without pulling any fak
// internals.
//
// A refusal is a successful 200 carried in the Verdict, NOT an HTTP error: a DENY
// / QUARANTINE / TRANSFORM verdict returns (resp, nil) with resp.Verdict.Kind set
// accordingly. *APIError is returned only for a real fault — a malformed request
// (400), an auth failure (401), a method error (405), or an upstream/gateway
// fault (5xx).
package fakclient

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// maxResponseBytes caps how much of a response body the client will read, so a
// misbehaving or hostile server cannot exhaust client memory. Every fak-native
// response is a small JSON document; 8 MiB is far above any legitimate body.
const maxResponseBytes = 8 << 20

// Client is a fak gateway client. The zero value is not usable; construct one
// with New. It is safe for concurrent use by multiple goroutines.
type Client struct {
	baseURL    string
	httpClient *http.Client
	apiKey     string
	principal  string
}

// Option configures a Client at construction time.
type Option func(*Client)

// WithHTTPClient sets the underlying *http.Client (for custom timeouts, a proxy,
// or a transport with TLS pinning). Defaults to defaultFakHTTPClient (a 30s-timeout
// client) — NOT http.DefaultClient, whose zero timeout would hang an SDK caller
// forever on a dead or wedged gateway.
func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) {
		if h != nil {
			c.httpClient = h
		}
	}
}

// defaultFakHTTPClient is the client used when a caller injects none. It carries a bounded
// 30s timeout so an SDK consumer never hangs forever on a dead/wedged gateway — the
// MISSING_HTTP_TIMEOUT failure mode the repo's boundarylint flags for http.DefaultClient.
// A caller that needs streaming/long bodies supplies its own via WithHTTPClient.
var defaultFakHTTPClient = &http.Client{Timeout: 30 * time.Second}

// WithAPIKey sets the bearer token sent as `Authorization: Bearer <key>` on every
// request. Required only when the gateway was booted with --require-key-env;
// otherwise leave it empty (loopback, auth-off default).
func WithAPIKey(key string) Option {
	return func(c *Client) { c.apiKey = key }
}

// WithPrincipal sets the isolation principal (a tenant / user / auth subject) sent
// as the X-Fak-Principal header. When set, the gateway scopes its vDSO cache and
// its change feed to this principal. Empty => single-tenant (every caller shares).
func WithPrincipal(p string) Option {
	return func(c *Client) { c.principal = strings.TrimSpace(p) }
}

// New returns a Client for the gateway at baseURL (e.g. "http://127.0.0.1:8080").
// A trailing slash is trimmed.
func New(baseURL string, opts ...Option) *Client {
	c := &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		httpClient: defaultFakHTTPClient,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// ---------------------------------------------------------------------------
// Wire DTOs — mirror internal/gateway/wire.go field-for-field by JSON tag.
// ---------------------------------------------------------------------------

// Verdict is the named projection of the kernel's adjudication outcome (the
// gateway's WireVerdict). Kind is the closed vocabulary:
// ALLOW|DENY|TRANSFORM|QUARANTINE|REQUIRE_WITNESS|DEFER. Disposition is the
// actionable deny-loopback class (RETRYABLE|WAIT|ESCALATE|TERMINAL) on a refusal.
type Verdict struct {
	Kind        string            `json:"kind"`
	Reason      string            `json:"reason,omitempty"`
	By          string            `json:"by,omitempty"`
	Disposition string            `json:"disposition,omitempty"`
	Detail      map[string]string `json:"detail,omitempty"`
}

// Allowed reports whether the verdict permits the call (Kind == "ALLOW").
func (v Verdict) Allowed() bool { return v.Kind == "ALLOW" }

// SyscallRequest is the body of Adjudicate and Syscall. Arguments is either a JSON
// object or a JSON-encoded string (the OpenAI function.arguments convention).
type SyscallRequest struct {
	Tool      string          `json:"tool"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
	ReadOnly  bool            `json:"read_only,omitempty"`
	Witness   string          `json:"witness,omitempty"`
	TraceID   string          `json:"trace_id,omitempty"`
	Principal string          `json:"principal,omitempty"`
}

// AdmitRequest is the body of Admit: a CLIENT-produced tool Result to run through
// the kernel's result-side stack (quarantine + IFC source-stamp).
type AdmitRequest struct {
	Tool    string          `json:"tool"`
	Result  json.RawMessage `json:"result,omitempty"`
	Witness string          `json:"witness,omitempty"`
	TraceID string          `json:"trace_id,omitempty"`
}

// ResultEnvelope is a tool result rendered for the wire.
type ResultEnvelope struct {
	Status  string            `json:"status"`
	Content string            `json:"content"`
	Meta    map[string]string `json:"meta,omitempty"`
}

// SyscallResponse is the result of an adjudicate(-and-execute). Result is present
// only on the execute path (Syscall); RepairedArguments only when the verdict is
// TRANSFORM (the canonical args the client should run instead).
type SyscallResponse struct {
	Verdict           Verdict         `json:"verdict"`
	Result            *ResultEnvelope `json:"result,omitempty"`
	RepairedArguments json.RawMessage `json:"repaired_arguments,omitempty"`
	TraceID           string          `json:"trace_id,omitempty"`
}

// ChangeEvent is one entry on the cross-agent "what changed" feed.
type ChangeEvent struct {
	Kind       string   `json:"kind"`
	Seq        uint64   `json:"seq"`
	Tool       string   `json:"tool,omitempty"`
	Tags       []string `json:"tags,omitempty"`
	Witness    string   `json:"witness,omitempty"`
	Evicted    int      `json:"evicted,omitempty"`
	WorldVer   uint64   `json:"world_ver"`
	TrustEpoch uint64   `json:"trust_epoch"`
}

// ChangesResponse is the drained feed slice plus the client's next cursor.
type ChangesResponse struct {
	Events []ChangeEvent `json:"events"`
	Cursor uint64        `json:"cursor"`
}

// RevokeResponse reports how many pooled entries the refutation stranded locally
// and the post-bump integrity epoch.
type RevokeResponse struct {
	Witness    string `json:"witness"`
	Evicted    int    `json:"evicted"`
	TrustEpoch uint64 `json:"trust_epoch"`
}

// Model is one entry of the OpenAI-style model list.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse is the body of GET /v1/models.
type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// ---------------------------------------------------------------------------
// Errors.
// ---------------------------------------------------------------------------

// APIError is a non-2xx gateway response (a real fault, never a policy refusal —
// a refusal is a 200 carried in the Verdict). It carries the HTTP status and the
// gateway's structured error body (the ErrorResponse schema in openapi.yaml).
type APIError struct {
	StatusCode int
	Type       string
	Message    string
	Code       string
	Param      string
}

func (e *APIError) Error() string {
	if e.Type != "" {
		return fmt.Sprintf("fak: %d %s: %s", e.StatusCode, e.Type, e.Message)
	}
	return fmt.Sprintf("fak: %d: %s", e.StatusCode, e.Message)
}

// errorEnvelope is the gateway's ErrorResponse wire shape: {"error": {...}}.
type errorEnvelope struct {
	Error struct {
		Message string  `json:"message"`
		Type    string  `json:"type"`
		Code    *string `json:"code"`
		Param   *string `json:"param"`
	} `json:"error"`
}

func parseAPIError(status int, body []byte) *APIError {
	e := &APIError{StatusCode: status, Message: strings.TrimSpace(string(body))}
	var env errorEnvelope
	if err := json.Unmarshal(body, &env); err == nil && env.Error.Message != "" {
		e.Message = env.Error.Message
		e.Type = env.Error.Type
		if env.Error.Code != nil {
			e.Code = *env.Error.Code
		}
		if env.Error.Param != nil {
			e.Param = *env.Error.Param
		}
	}
	return e
}

// ---------------------------------------------------------------------------
// Endpoint methods.
// ---------------------------------------------------------------------------

// Adjudicate returns the pre-execution verdict only (POST /v1/fak/adjudicate) —
// the production path for a client that runs its own tools. On a TRANSFORM
// verdict, resp.RepairedArguments carries the canonical args to run instead.
func (c *Client) Adjudicate(ctx context.Context, req SyscallRequest) (*SyscallResponse, error) {
	return c.postSyscall(ctx, "/v1/fak/adjudicate", req)
}

// postSyscall POSTs req to a syscall-family endpoint and decodes a SyscallResponse.
// Shared body of Adjudicate / Syscall / Admit.
func (c *Client) postSyscall(ctx context.Context, path string, req any) (*SyscallResponse, error) {
	var resp SyscallResponse
	if err := c.do(ctx, http.MethodPost, path, req, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Syscall adjudicates AND executes one tool call against the registered engine
// (POST /v1/fak/syscall) — the self-contained / CI path. On ALLOW, resp.Result
// carries the executed result envelope.
func (c *Client) Syscall(ctx context.Context, req SyscallRequest) (*SyscallResponse, error) {
	return c.postSyscall(ctx, "/v1/fak/syscall", req)
}

// Admit runs a CLIENT-produced tool result through the kernel's result-side stack
// (POST /v1/fak/admit). A QUARANTINE verdict means the bytes were paged out.
func (c *Client) Admit(ctx context.Context, req AdmitRequest) (*SyscallResponse, error) {
	return c.postSyscall(ctx, "/v1/fak/admit", req)
}

// Changes drains the cross-agent change feed for events after the since cursor
// (GET /v1/fak/changes?since=N). Pass 0 to read everything retained; pass the
// previous response's Cursor to page forward.
func (c *Client) Changes(ctx context.Context, since uint64) (*ChangesResponse, error) {
	path := "/v1/fak/changes"
	if since > 0 {
		path += "?since=" + strconv.FormatUint(since, 10)
	}
	var resp ChangesResponse
	if err := c.do(ctx, http.MethodGet, path, nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Revoke triggers a fleet-wide refutation of an external world-state witness
// (POST /v1/fak/revoke): every pooled entry admitted under it is causally evicted.
func (c *Client) Revoke(ctx context.Context, witness string) (*RevokeResponse, error) {
	var resp RevokeResponse
	body := map[string]string{"witness": witness}
	if err := c.do(ctx, http.MethodPost, "/v1/fak/revoke", body, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Models lists the model the gateway serves (GET /v1/models).
func (c *Client) Models(ctx context.Context) (*ModelsResponse, error) {
	var resp ModelsResponse
	if err := c.do(ctx, http.MethodGet, "/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// Health checks the unauthenticated liveness endpoint (GET /healthz). It returns
// nil when the gateway answers 2xx, else an *APIError.
func (c *Client) Health(ctx context.Context) error {
	return c.do(ctx, http.MethodGet, "/healthz", nil, nil)
}

// do issues one request and decodes the JSON response into out (out may be nil to
// discard the body). A non-2xx status is mapped to *APIError.
func (c *Client) do(ctx context.Context, method, path string, body, out any) error {
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("fak: marshal request: %w", err)
		}
		rdr = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, rdr)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Accept", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	if c.principal != "" {
		req.Header.Set("X-Fak-Principal", c.principal)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return fmt.Errorf("fak: read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return parseAPIError(resp.StatusCode, data)
	}
	if out == nil || len(bytes.TrimSpace(data)) == 0 {
		return nil
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("fak: decode response: %w", err)
	}
	return nil
}
