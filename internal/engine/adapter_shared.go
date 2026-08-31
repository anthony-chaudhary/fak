package engine

// adapter_shared.go holds the handful of genuinely-identical helpers shared across
// the per-engine adapter files (dynamo.go, sglang.go, vllm.go, llmd.go,
// lifecycle_adapter.go, engine.go, on_device.go). Each per-engine file otherwise
// intentionally mirrors its siblings — the config/request/response shapes differ per
// upstream project — so only the byte-for-byte identical spans live here.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Transport and SSE-idle deadlines shared by every HTTP engine adapter. Client.Timeout
// stays 0 (a long but healthy streaming generation must not be cut off mid-body); these
// bound the two vectors that otherwise hang forever — a peer that accepts the connection
// but never sends headers, and a peer that goes silent mid-stream.
const (
	engineDialTimeout           = 15 * time.Second
	engineTLSHandshakeTimeout   = 10 * time.Second
	engineResponseHeaderTimeout = 30 * time.Second
)

// sseIdleTimeout bounds the gap between reads on an SSE body. A healthy generation emits
// tokens well within this window; a peer that goes silent mid-stream is unblocked here
// (its request context is cancelled) instead of wedging Admit/Complete and kernel.Reap
// forever. It is a var, not a const, only so a test can lower it — production never
// mutates it.
var sseIdleTimeout = 120 * time.Second

// defaultHTTPClient returns c, or a streaming-safe *http.Client when c is nil. Every
// HTTP-based engine constructor (Dynamo/SGLang/vLLM/llm-d) applies this same nil-client
// fallback over its own config type.
//
// Client.Timeout stays 0 so a long but healthy streaming generation is never cut off
// mid-body, but the transport carries the bounded dial/TLS/response-header deadlines
// documented as the "download-safe form" in internal/boundarylint/rules_http.go — so a
// peer that never sends headers fails fast instead of wedging Admit/Complete forever.
func defaultHTTPClient(c *http.Client) *http.Client {
	if c != nil {
		return c
	}
	return &http.Client{Timeout: 0, Transport: streamingTransport()}
}

// streamingTransport returns a private clone of http.DefaultTransport (preserving proxy/HTTP2
// defaults without mutating process-global transport state) and pins the three deadlines that bound a dead-or-silent peer: dial, TLS handshake, and —
// the one DefaultTransport leaves unset — the response-header wait. Client.Timeout is
// deliberately NOT set here; the caller keeps it 0 for unbounded healthy streams.
func streamingTransport() *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.DialContext = (&net.Dialer{
		Timeout:   engineDialTimeout,
		KeepAlive: 30 * time.Second,
	}).DialContext
	t.TLSHandshakeTimeout = engineTLSHandshakeTimeout
	t.ResponseHeaderTimeout = engineResponseHeaderTimeout
	return t
}

// defaultWorkerID returns id, or def when id is empty. Every engine constructor
// fills in the same worker-id default over its own config type.
func defaultWorkerID(id, def string) string {
	if id != "" {
		return id
	}
	return def
}

// defaultCacheAndResidency fills in the SGLang/vLLM constructors' shared
// cache-recorder and prefix-residency-index defaults. Dynamo has neither field, so
// its constructor does not call this helper.
func defaultCacheAndResidency(cache *CacheEventRecorder, residency *PrefixResidencyIndex) (*CacheEventRecorder, *PrefixResidencyIndex) {
	if cache == nil {
		cache = NewCacheEventRecorder()
	}
	if residency == nil {
		residency = NewPrefixResidencyIndex()
	}
	return cache, residency
}

// postStreamingRequest issues one streaming POST for the OpenAI-frontend adapters
// (Dynamo/vLLM/llm-d) and the SGLang native adapter. It applies the shared
// header/bearer-token plumbing shared by every Admit implementation in this
// package, and on a non-200 response classifies the body into the
// "<errPrefix>: <kind> returned N: body" shape every adapter's error already used,
// cancelling the request context before returning. On success (nil err) the caller
// owns cctx/cancel/resp and builds its own request handle from them.
func postStreamingRequest(ctx context.Context, client *http.Client, endpoint, apiKey string, body []byte, errPrefix, kind string) (cctx context.Context, cancel context.CancelFunc, resp *http.Response, err error) {
	cctx, cancel = context.WithCancel(ctx)
	req, err := http.NewRequestWithContext(cctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		cancel()
		return nil, nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err = client.Do(req)
	if err != nil {
		cancel()
		return nil, nil, nil, err
	}
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		_ = resp.Body.Close()
		cancel()
		return nil, nil, nil, fmt.Errorf("%s: %s returned %d: %s", errPrefix, kind, resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	return cctx, cancel, resp, nil
}

// scrapeMetricsText reads one worker's Prometheus exposition with the shared
// bearer-token plumbing and returns it as text. A transport or read error is
// returned as an error; a NON-200 status is NOT — it comes back as a non-empty
// `disabled` note and a nil error, because an engine serving its /metrics endpoint
// off is a legitimate "exposes no such surface" OBSERVATION rather than a failed
// read. Exactly one of text/disabled is non-empty when err is nil.
func scrapeMetricsText(ctx context.Context, client *http.Client, metricsURL, apiKey string) (text, disabled string, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return "", "", err
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Sprintf("/metrics returned %d (endpoint disabled)", resp.StatusCode), nil
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return "", "", err
	}
	return string(raw), "", nil
}

// decodeOrEmptyJSONObject parses args as a JSON object, returning a fresh empty
// object when args does not decode to a non-empty object. Every request-body
// builder (SGLang's native /generate, vLLM's OpenAI chat/completions) starts from
// this same caller-supplied-JSON-or-empty seed.
func decodeOrEmptyJSONObject(args []byte) map[string]json.RawMessage {
	obj := map[string]json.RawMessage{}
	if json.Unmarshal(args, &obj) != nil || len(obj) == 0 {
		obj = map[string]json.RawMessage{}
	}
	return obj
}

// runSSEPump drives the SSE line-scanning loop shared by the SGLang and vLLM
// streaming requests: scan lines, skip non-"data:" lines, finish on "[DONE]" or ctx
// cancellation, hand each data line to decode, and forward a non-empty delta onto
// tokens (finishing early if ctx is cancelled first). decode applies
// engine-specific fields to the request and returns the newly observed delta text.
func runSSEPump(ctx context.Context, body io.ReadCloser, cancel context.CancelFunc, tokens chan abi.EngineToken, finish func(*abi.Result, error), assemble func() *abi.Result, decode func(data string) (delta string, err error)) {
	// Wrap the body so a mid-stream stall is bounded: if no byte arrives for
	// sseIdleTimeout the request context is cancelled, which unblocks the otherwise
	// forever-blocking sc.Scan below and lets finish/Result return so kernel.Reap
	// completes. Client.Timeout stays 0, so this is the only bound on a healthy stream.
	body = newIdleTimeoutReader(body, sseIdleTimeout, cancel)
	defer body.Close()
	defer cancel()
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		if err := ctx.Err(); err != nil {
			finish(nil, err)
			return
		}
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			finish(assemble(), nil)
			return
		}
		delta, err := decode(data)
		if err != nil {
			finish(nil, err)
			return
		}
		if delta == "" {
			continue
		}
		select {
		case tokens <- abi.EngineToken{Text: delta}:
		case <-ctx.Done():
			finish(nil, ctx.Err())
			return
		}
	}
	if err := sc.Err(); err != nil {
		finish(nil, err)
		return
	}
	finish(assemble(), nil)
}

// idleTimeoutReader bounds the gap between successive reads on an SSE body. Each Read
// (re)arms a single timer; if idle elapses before that read returns, onIdle fires. The
// only caller passes the request's cancel func as onIdle: cancelling the context aborts
// the in-flight net/http body read, so the blocked read returns an error instead of
// hanging forever. onIdle must be idempotent (cancel is), since a healthy read that
// returns right as the timer fires may still trigger it.
type idleTimeoutReader struct {
	rc    io.ReadCloser
	idle  time.Duration
	timer *time.Timer
}

// newIdleTimeoutReader wraps rc with an idle-read deadline. A non-positive idle disables
// the guard (rc is returned unchanged), so callers can opt out without a branch.
func newIdleTimeoutReader(rc io.ReadCloser, idle time.Duration, onIdle func()) io.ReadCloser {
	if idle <= 0 {
		return rc
	}
	return &idleTimeoutReader{
		rc:    rc,
		idle:  idle,
		timer: time.AfterFunc(idle, onIdle),
	}
}

// Read arms the idle timer for this read, then delegates. A read that blocks past idle
// lets the timer fire onIdle (cancelling the stream); a read that returns in time leaves
// the timer to be re-armed by the next Read.
func (r *idleTimeoutReader) Read(p []byte) (int, error) {
	r.timer.Reset(r.idle)
	return r.rc.Read(p)
}

// Close stops the idle timer and closes the underlying body.
func (r *idleTimeoutReader) Close() error {
	r.timer.Stop()
	return r.rc.Close()
}

// decodeRawJSONOrBareString handles the two trivial json.RawMessage shapes shared by
// every loosely-typed field these adapters decode (SGLang's finish_reason, vLLM's
// chat delta content): absent/null decodes to "", and a bare JSON string decodes to
// itself. ok reports whether one of those two shapes matched, so the caller falls
// through to its own structured shape when it doesn't.
func decodeRawJSONOrBareString(raw json.RawMessage) (value string, ok bool) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", true
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, true
	}
	return "", false
}

// estimateOfflineUsage is the coarse token-usage heuristic shared by every
// network-free engine (Mock, OnDevice): tokens are approximated at ~4 chars/token,
// with a flat input floor of 50 so short prompts don't estimate to zero. A real
// network adapter overwrites these with the usage the served engine reports; this
// exists so every offline engine's Meta carries the same keys.
func estimateOfflineUsage(inputLen, outputLen int) Usage {
	u := Usage{InputTokens: 50 + inputLen/4, OutputTokens: outputLen / 4}
	u.TotalTokens = u.InputTokens + u.OutputTokens
	return u
}

// setMetaIfNonEmpty writes meta[key] = value when value is non-empty. Both the
// SGLang and vLLM stream-result builders populate Meta with this
// only-when-present convention for their model/finish_reason fields.
func setMetaIfNonEmpty(meta map[string]string, key, value string) {
	if value != "" {
		meta[key] = value
	}
}

// setMetaIfPositive writes meta[key] = strconv.Itoa(n) when n > 0. Both the SGLang
// and vLLM stream-result builders populate Meta with this only-when-present
// convention for their token-count fields.
func setMetaIfPositive(meta map[string]string, key string, n int) {
	if n > 0 {
		meta[key] = strconv.Itoa(n)
	}
}
