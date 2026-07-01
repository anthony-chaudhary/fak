// Package enginecache binds cachemeta's remote invalidation directives to
// documented serving-engine control endpoints.
package enginecache

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

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// Engine names a remote serving engine with a documented cache reset endpoint.
type Engine string

const (
	EngineSGLang Engine = "sglang"
	EngineVLLM   Engine = "vllm"

	ScopeWholePrefixCache = string(cachemeta.KVEvictionScopeWholePrefixCache)
	ScopeExactSpan        = string(cachemeta.KVEvictionScopeExactSpan)
)

const (
	degradeExactSpanUnsupported = "exact_span_unsupported_whole_prefix_flush"
	degradeExactSpanUnnamed     = "exact_span_target_missing_whole_prefix_flush"
)

// Client translates cachemeta invalidation directives into remote engine calls.
// The current SGLang/vLLM public HTTP surfaces expose whole-prefix-cache reset,
// not exact-span eviction, so one or more directives becomes one auditable full
// engine-cache invalidation.
type Client struct {
	Engine        Engine
	BaseURL       string
	AdminAPIKey   string
	IdleTimeout   time.Duration
	RequiredScope string
	HTTPClient    *http.Client
	// ExactSpanEndpoint, when non-empty, is an operator-supplied control endpoint
	// that evicts exactly the named K/V span(s) plus their dependent DSA
	// attention_index entries instead of resetting the whole prefix cache. It is
	// empty by default: fak does not claim the public SGLang/vLLM HTTP surfaces
	// expose exact-span eviction (SupportsExactSpan stays false for them), so this
	// path is taken ONLY when an operator wires an independently witnessed endpoint
	// for their deployment. A full URL (scheme+host) is used verbatim; a bare path
	// resolves against the control base derived from BaseURL.
	ExactSpanEndpoint string
}

// exactSpanRequest is the wire body of an exact-span eviction call: the precise,
// payload-free span identities cachemeta planned, never the bytes.
type exactSpanRequest struct {
	Spans []cachemeta.ExactSpanTarget `json:"spans"`
}

// Result is the witnessed effect of an engine-cache invalidation attempt.
type Result struct {
	Engine             Engine
	Endpoint           string
	Scope              string
	ExactSpanSupported bool
	Degraded           bool
	DegradeReason      string
	Directives         int
	StatusCode         int
	BodySummary        string
	Attestations       []cachemeta.KVEvictionAttestation
}

// Invalidate applies remote invalidation directives. It is a no-op when the
// directive set is empty, and returns an error on unsupported engines or non-2xx
// control endpoint responses.
func (c Client) Invalidate(ctx context.Context, dirs []cachemeta.ExternalInvalidationDirective) (Result, error) {
	var res Result
	if len(dirs) == 0 {
		return res, nil
	}
	engine := c.Engine
	if engine == "" {
		engine = inferEngine(dirs)
	}
	if engine == "" {
		return res, fmt.Errorf("enginecache: no supported engine in directives")
	}
	if v := cachemeta.DefaultKVGovernanceReferee.AdmitInvalidations(dirs); !v.Admitted {
		return Result{
			Engine:             engine,
			Scope:              ScopeWholePrefixCache,
			ExactSpanSupported: c.exactSpanCapable(engine),
			Directives:         len(dirs),
			Attestations:       cachemeta.AttestInvalidations(dirs, cachemeta.KVEvictionScopeWholePrefixCache, c.exactSpanCapable(engine), ""),
		}, fmt.Errorf("enginecache: KV governance referee denied invalidation: %s", v.Reason)
	}
	if err := c.checkRequiredScope(engine, len(dirs)); err != nil {
		return Result{
			Engine:             engine,
			Scope:              ScopeWholePrefixCache,
			ExactSpanSupported: c.exactSpanCapable(engine),
			Directives:         len(dirs),
		}, err
	}
	if c.exactSpanCapable(engine) {
		return c.invalidateExactSpan(ctx, engine, dirs)
	}
	return c.invalidateWholePrefix(ctx, engine, dirs, "")
}

// invalidateExactSpan evicts only the named K/V span(s) and their dependent DSA
// attention_index entries, when an exact-span-capable engine is configured. It is
// fail-closed: a non-2xx control response surfaces as an error (the caller must
// not forward the contaminated turn), and an empty named-span set is NOT silently
// reported as a precise eviction — it either fails closed (when exact-span is
// required) or degrades to the safe whole-prefix reset superset.
func (c Client) invalidateExactSpan(ctx context.Context, engine Engine, dirs []cachemeta.ExternalInvalidationDirective) (Result, error) {
	targets := cachemeta.ExactSpanTargets(dirs)
	if len(targets) == 0 {
		if c.RequiredScope == ScopeExactSpan {
			return Result{
				Engine:             engine,
				Scope:              ScopeExactSpan,
				ExactSpanSupported: true,
				Directives:         len(dirs),
			}, fmt.Errorf("enginecache: exact-span eviction required but no named span in %d directive(s)", len(dirs))
		}
		// No span identity to evict precisely; the safe superset is a whole reset.
		return c.invalidateWholePrefix(ctx, engine, dirs, degradeExactSpanUnnamed)
	}
	endpoint, err := c.exactSpanResolvedEndpoint()
	if err != nil {
		return Result{Engine: engine, Scope: ScopeExactSpan, ExactSpanSupported: true, Directives: len(dirs)}, err
	}
	payload, err := json.Marshal(exactSpanRequest{Spans: targets})
	if err != nil {
		return Result{Engine: engine, Scope: ScopeExactSpan, ExactSpanSupported: true, Directives: len(dirs)}, err
	}
	resp, err := c.post(ctx, endpoint, payload, "application/json")
	if err != nil {
		return Result{Engine: engine, Endpoint: endpoint, Scope: ScopeExactSpan, ExactSpanSupported: true, Directives: len(dirs)}, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	res := Result{
		Engine:             engine,
		Endpoint:           endpoint,
		Scope:              ScopeExactSpan,
		ExactSpanSupported: true,
		Directives:         len(dirs),
		StatusCode:         resp.StatusCode,
		BodySummary:        strings.TrimSpace(string(raw)),
		Attestations:       cachemeta.AttestInvalidations(dirs, cachemeta.KVEvictionScopeExactSpan, true, ""),
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return res, fmt.Errorf("enginecache: %s exact-span HTTP %d: %s", engine, resp.StatusCode, res.BodySummary)
	}
	return res, nil
}

// invalidateWholePrefix collapses any directive set to one documented
// whole-prefix/radix-cache reset — the safe over-invalidation superset of the
// named span. A non-2xx control response surfaces as an error.
func (c Client) invalidateWholePrefix(ctx context.Context, engine Engine, dirs []cachemeta.ExternalInvalidationDirective, degradeReason string) (Result, error) {
	endpoint, err := c.endpoint(engine)
	if err != nil {
		return Result{}, err
	}
	resp, err := c.post(ctx, endpoint, nil, "")
	if err != nil {
		return Result{}, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	exactSpanSupported := c.exactSpanCapable(engine)
	if degradeReason == "" && !exactSpanSupported && len(cachemeta.ExactSpanTargets(dirs)) > 0 {
		degradeReason = degradeExactSpanUnsupported
	}
	res := Result{
		Engine:             engine,
		Endpoint:           endpoint,
		Scope:              ScopeWholePrefixCache,
		ExactSpanSupported: exactSpanSupported,
		Degraded:           degradeReason != "",
		DegradeReason:      degradeReason,
		Directives:         len(dirs),
		StatusCode:         resp.StatusCode,
		BodySummary:        strings.TrimSpace(string(raw)),
		Attestations:       cachemeta.AttestInvalidations(dirs, cachemeta.KVEvictionScopeWholePrefixCache, exactSpanSupported, degradeReason),
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return res, fmt.Errorf("enginecache: %s HTTP %d: %s", engine, resp.StatusCode, res.BodySummary)
	}
	return res, nil
}

// post issues the control-plane POST shared by both reset scopes. A nil body is
// the empty whole-cache reset request; a non-empty body carries the exact-span
// target list.
func (c Client) post(ctx context.Context, endpoint string, body []byte, contentType string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	if c.AdminAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.AdminAPIKey)
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	httpClient := c.HTTPClient
	if httpClient == nil {
		// A defensive default: the request is context-scoped, but a caller-supplied
		// ctx without a deadline would otherwise inherit DefaultClient's no-timeout.
		httpClient = &http.Client{Timeout: 10 * time.Second}
	}
	return httpClient.Do(req)
}

// exactSpanCapable reports whether this client can evict an exact span: either the
// documented public engine exposes it (SupportsExactSpan — currently never), or an
// operator wired a witnessed ExactSpanEndpoint for this deployment.
func (c Client) exactSpanCapable(engine Engine) bool {
	return strings.TrimSpace(c.ExactSpanEndpoint) != "" || SupportsExactSpan(engine)
}

func (c Client) exactSpanResolvedEndpoint() (string, error) {
	raw := strings.TrimSpace(c.ExactSpanEndpoint)
	if raw == "" {
		return "", fmt.Errorf("enginecache: exact-span endpoint is not configured")
	}
	// A "://" marks raw as an attempted absolute URL, not a bare path: resolve it
	// only if it fully validates (scheme+host), rather than silently falling
	// through to the bare-path join below with a malformed absolute URL as the
	// "path" (e.g. "http://" or an unparseable "://bad").
	if strings.Contains(raw, "://") {
		u, err := url.Parse(raw)
		if err != nil || u.Scheme == "" || u.Host == "" {
			return "", fmt.Errorf("enginecache: exact-span endpoint %q must be a full URL with scheme and host", raw)
		}
		return raw, nil
	}
	base, err := controlBase(c.BaseURL)
	if err != nil {
		return "", err
	}
	return base.ResolveReference(&url.URL{Path: joinPath(base.Path, strings.TrimPrefix(raw, "/"))}).String(), nil
}

func (c Client) checkRequiredScope(engine Engine, nDirs int) error {
	switch c.RequiredScope {
	case "", ScopeWholePrefixCache:
		return nil
	case ScopeExactSpan:
		if c.exactSpanCapable(engine) {
			return nil
		}
		return fmt.Errorf("enginecache: exact-span eviction required for %d directive(s), but %s exposes only whole-prefix-cache reset", nDirs, engine)
	default:
		return fmt.Errorf("enginecache: unsupported required scope %q", c.RequiredScope)
	}
}

// SupportsExactSpan reports whether the documented public engine control plane
// can evict exactly the K/V or attention-index span named by cachemeta. Current
// SGLang and vLLM HTTP endpoints reset the whole prefix/radix cache instead.
func SupportsExactSpan(engine Engine) bool {
	switch engine {
	case EngineSGLang, EngineVLLM:
		return false
	default:
		return false
	}
}

func (c Client) endpoint(engine Engine) (string, error) {
	base, err := controlBase(c.BaseURL)
	if err != nil {
		return "", err
	}
	switch engine {
	case EngineSGLang:
		u := base.ResolveReference(&url.URL{Path: joinPath(base.Path, "flush_cache")})
		if c.IdleTimeout > 0 {
			q := u.Query()
			q.Set("timeout", strconv.FormatFloat(c.IdleTimeout.Seconds(), 'f', -1, 64))
			u.RawQuery = q.Encode()
		}
		return u.String(), nil
	case EngineVLLM:
		return base.ResolveReference(&url.URL{Path: joinPath(base.Path, "reset_prefix_cache")}).String(), nil
	default:
		return "", fmt.Errorf("enginecache: unsupported engine %q", engine)
	}
}

func controlBase(raw string) (*url.URL, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, fmt.Errorf("enginecache: base URL is required")
	}
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("enginecache: bad base URL: %w", err)
	}
	if u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("enginecache: base URL must include scheme and host")
	}
	u.Path = strings.TrimRight(u.Path, "/")
	if strings.HasSuffix(u.Path, "/v1") {
		u.Path = strings.TrimSuffix(u.Path, "/v1")
	}
	return u, nil
}

func joinPath(base, leaf string) string {
	base = strings.TrimRight(base, "/")
	if base == "" {
		return "/" + leaf
	}
	return base + "/" + leaf
}

func inferEngine(dirs []cachemeta.ExternalInvalidationDirective) Engine {
	for _, d := range dirs {
		for _, s := range []string{d.Provider, d.Engine, d.Residency.Owner} {
			switch strings.ToLower(strings.TrimSpace(s)) {
			case string(EngineSGLang):
				return EngineSGLang
			case string(EngineVLLM):
				return EngineVLLM
			}
		}
	}
	return ""
}
