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
	"net/http"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// defaultHTTPClient returns c, or an unbounded-timeout *http.Client when c is nil.
// Every HTTP-based engine constructor (Dynamo/SGLang/vLLM) applies this same
// nil-client fallback over its own config type.
func defaultHTTPClient(c *http.Client) *http.Client {
	if c != nil {
		return c
	}
	return &http.Client{Timeout: 0}
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

// closeRequestChannels closes the token/done channels an EngineRequest signals
// completion through. Callers must set their res/err fields BEFORE calling this —
// Result() receives on done as its only synchronization point, so closing first
// would let a reader observe done closed before res/err are written.
func closeRequestChannels(tokens chan abi.EngineToken, done chan struct{}) {
	close(tokens)
	close(done)
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
