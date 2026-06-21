// Package enginecache binds cachemeta's remote invalidation directives to
// documented serving-engine control endpoints.
package enginecache

import (
	"bytes"
	"context"
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

	ScopeWholePrefixCache = "whole_prefix_cache"
	ScopeExactSpan        = "exact_span"
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
}

// Result is the witnessed effect of an engine-cache invalidation attempt.
type Result struct {
	Engine             Engine
	Endpoint           string
	Scope              string
	ExactSpanSupported bool
	Directives         int
	StatusCode         int
	BodySummary        string
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
	if err := c.checkRequiredScope(engine, len(dirs)); err != nil {
		return Result{
			Engine:             engine,
			Scope:              ScopeWholePrefixCache,
			ExactSpanSupported: SupportsExactSpan(engine),
			Directives:         len(dirs),
		}, err
	}
	endpoint, err := c.endpoint(engine)
	if err != nil {
		return res, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(nil))
	if err != nil {
		return res, err
	}
	if c.AdminAPIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.AdminAPIKey)
	}
	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return res, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	res = Result{
		Engine:             engine,
		Endpoint:           endpoint,
		Scope:              ScopeWholePrefixCache,
		ExactSpanSupported: SupportsExactSpan(engine),
		Directives:         len(dirs),
		StatusCode:         resp.StatusCode,
		BodySummary:        strings.TrimSpace(string(raw)),
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return res, fmt.Errorf("enginecache: %s HTTP %d: %s", engine, resp.StatusCode, res.BodySummary)
	}
	return res, nil
}

func (c Client) checkRequiredScope(engine Engine, nDirs int) error {
	switch c.RequiredScope {
	case "", ScopeWholePrefixCache:
		return nil
	case ScopeExactSpan:
		if SupportsExactSpan(engine) {
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
