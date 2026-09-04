package systools

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// ---------------------------------------------------------------------------
// Engine 1: systools.get_time
// ---------------------------------------------------------------------------

type getTimeEngine struct{ t *Toolset }

func (getTimeEngine) Caps() []abi.Capability { return nil }
func (getTimeEngine) WeightBearing() bool    { return false }

func (e getTimeEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, isErr := e.t.getTime(ctx, in)
	return result(ctx, c, in, out, isErr, EngineGetTime), nil
}

func (t *Toolset) getTime(ctx context.Context, body []byte) ([]byte, bool) {
	var a GetTimeArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	if err := ctx.Err(); err != nil {
		return refuse(CodeCanceled, err.Error()).JSON(), true
	}
	loc := time.UTC
	if a.Timezone != "" {
		var err error
		loc, err = time.LoadLocation(a.Timezone)
		if err != nil {
			return refuse(CodeMalformed, "invalid timezone: "+err.Error()).JSON(), true
		}
	}
	now := time.Now().In(loc)
	return okJSON(map[string]any{
		"time":          now.Format(time.RFC3339),
		"timezone":      loc.String(),
		"epoch_seconds": now.Unix(),
		"epoch_millis":  now.UnixMilli(),
	}), false
}

// ---------------------------------------------------------------------------
// Engine 2: systools.fetch_web
// ---------------------------------------------------------------------------

type fetchWebEngine struct{ t *Toolset }

func (fetchWebEngine) Caps() []abi.Capability { return nil }
func (fetchWebEngine) WeightBearing() bool    { return false }

func (e fetchWebEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, isErr := e.t.fetchWeb(ctx, in)
	return result(ctx, c, in, out, isErr, EngineFetchWeb), nil
}

func (t *Toolset) fetchWeb(ctx context.Context, body []byte) ([]byte, bool) {
	var a FetchWebArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	if err := ctx.Err(); err != nil {
		return refuse(CodeCanceled, err.Error()).JSON(), true
	}
	u, err := url.Parse(a.URL)
	if err != nil || u.Host == "" {
		return refuse(CodeMalformed, "invalid URL: "+a.URL).JSON(), true
	}
	host := u.Hostname()
	if !t.domainAllowed(host) {
		return refuse(CodePolicyBlock, "domain "+host+" is not in the allowlist").JSON(), true
	}
	if !t.allowPrivateIPs {
		if r := t.checkSSRF(ctx, host); r != nil {
			return r.JSON(), true
		}
	}

	capBytes := t.defaultMaxFetchBytes
	if a.MaxBytes > 0 {
		capBytes = a.MaxBytes
		if t.maxFetchBytes > 0 && capBytes > t.maxFetchBytes {
			capBytes = t.maxFetchBytes
		}
	}

	reqCtx, cancel := context.WithTimeout(ctx, t.fetchTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, a.URL, nil)
	if err != nil {
		return refuse(CodeMalformed, err.Error()).JSON(), true
	}
	req.Header.Set("User-Agent", "fak-systools/1.0")

	resp, err := t.httpClient.Do(req)
	if err != nil {
		if strings.Contains(err.Error(), "SSRF protection") {
			return refuse(CodeSSRFBlock, err.Error()).JSON(), true
		}
		if reqCtx.Err() != nil {
			return refuse(CodeCanceled, reqCtx.Err().Error()).JSON(), true
		}
		return refuse(CodeIO, err.Error()).JSON(), true
	}
	defer resp.Body.Close()

	lr := io.LimitReader(resp.Body, int64(capBytes+1))
	data, err := io.ReadAll(lr)
	if err != nil && err != io.EOF {
		return refuse(CodeIO, "error reading response: "+err.Error()).JSON(), true
	}

	truncated := false
	if len(data) > capBytes {
		data = data[:capBytes]
		truncated = true
	}

	return okJSON(map[string]any{
		"url":         a.URL,
		"status_code": resp.StatusCode,
		"content":     string(data),
		"bytes":       len(data),
		"truncated":   truncated,
	}), false
}

// ---------------------------------------------------------------------------
// Engine 3: systools.web_search
// ---------------------------------------------------------------------------

type webSearchEngine struct{ t *Toolset }

func (webSearchEngine) Caps() []abi.Capability { return nil }
func (webSearchEngine) WeightBearing() bool    { return false }

func (e webSearchEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	in := bytesOf(ctx, c.Args)
	out, isErr := e.t.webSearch(ctx, in)
	return result(ctx, c, in, out, isErr, EngineWebSearch), nil
}

func (t *Toolset) webSearch(ctx context.Context, body []byte) ([]byte, bool) {
	var a WebSearchArgs
	if r := decodeArgs(body, &a); r != nil {
		return r.JSON(), true
	}
	if r := a.Validate(); r != nil {
		return r.JSON(), true
	}
	if err := ctx.Err(); err != nil {
		return refuse(CodeCanceled, err.Error()).JSON(), true
	}

	maxResults := a.MaxResults
	if maxResults <= 0 {
		maxResults = 5
	}
	if maxResults > 20 {
		maxResults = 20
	}

	results, err := t.searchAdapter(ctx, a.Query, maxResults)
	if err != nil {
		return refuse(CodeIO, err.Error()).JSON(), true
	}

	return okJSON(map[string]any{
		"query":   a.Query,
		"results": results,
		"count":   len(results),
	}), false
}
