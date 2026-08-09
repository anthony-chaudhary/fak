package gateway

// http_epfanout.go — the expert-parallel (EP) request-fanout bridge (#2955), split
// out of http.go along its concern seam (#2999). Everything here serves one job:
// mirror an inbound served request from the EP front rank to its follower ranks so
// every process of a sharded serve reaches the collectives.

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const epFollowerHeader = "X-Fak-EP-Follower"

// The served routes this bridge mirrors. A follower has to be asked for the SAME
// route the front rank is serving: /v1/completions and /v1/chat/completions take
// different request schemas, so a legacy request mirrored onto the chat route is not
// the same forward pass — it is a 400 on the follower and a front rank still alone in
// the collective (#5523). Each call site passes the constant route it serves rather
// than r.URL.Path, so no part of an outbound follower URL is client-steerable.
const (
	epRouteChatCompletions = "/v1/chat/completions"
	epRouteCompletions     = "/v1/completions"
)

// epFanoutClient carries loopback follower requests. It deliberately has no
// independent wall-clock timeout: the follower forward must live exactly as long as
// the front-rank request, including slow cold GLM/MoE turns. NewRequestWithContext
// below propagates the inbound deadline and cancellation, so a disconnected caller
// still releases every follower without a shorter transport timer splitting ranks.
var epFanoutClient = &http.Client{}

// startEPFanoutFollowers mirrors an inbound served request from the EP front rank to
// its follower rank endpoints before the front rank enters the local in-kernel decode.
// Rank-local expert parallelism reduces the routed-expert delta through a
// process-group AllReduce; that collective makes progress only if every rank runs the
// same forward pass. FAK_EP_FANOUT_ADDRS is therefore the temporary single-endpoint
// bridge for the sharded serve: the front rank receives the client request, followers
// receive the identical loopback request with X-Fak-EP-Follower set, and every process
// reaches the collectives concurrently. The follower header prevents recursive fanout
// if an operator points the bridge at another front rank — it is checked HERE, on
// entry, before any route-specific work, so every wire that calls this is guarded the
// same way.
//
// route is the served route to mirror onto, and is the caller's own constant (see
// epRouteChatCompletions / epRouteCompletions) rather than r.URL.Path. It used to be
// hardcoded to the chat wire, which is half of why the legacy text-completion wire
// entered a multi-rank decode no follower was released into (#5523); the other half
// was that handleCompletions never called this at all.
func (s *Server) startEPFanoutFollowers(w http.ResponseWriter, r *http.Request, route string) (func(), bool) {
	if r.Header.Get(epFollowerHeader) != "" {
		return func() {}, true
	}
	urls := epFanoutURLsFromEnv(route)
	if len(urls) == 0 {
		return func() {}, true
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxTranscriptBody))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return nil, false
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	var meta struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &meta); err != nil {
		// Malformed bodies will be rejected by the normal decoder below.
		return func() {}, true
	}
	// A streaming request must still fan out: rank-local expert parallelism makes
	// progress only if every rank runs the same forward pass, and each decode step is
	// a collective. Skipping the followers for stream:true stranded the front rank on
	// the first AllReduce — rank 0 pinned at 100% GPU with collapsed residency while
	// ranks 1-7 sat idle and no SSE byte ever left the socket (#4855). Followers are
	// mirrored either way; only how the helper drains their response differs.
	stream := meta.Stream
	// Read the inbound trace id HERE, on the handler goroutine, not inside the follower
	// below. handleChatCompletions calls useHTTPTrace after this function returns, and
	// that mints-and-writes r.Header when the client supplied no X-Trace-Id — so a Get
	// from a still-running follower raced the front rank's own Set on the same map and
	// tripped the detector under `go test -race` (TestEPStreamingEmitsFirstSSEChunkBefore
	// CollectiveJoin). Hoisting the read puts it in program order before that write, which
	// also makes what the follower propagates deterministic: the client's own trace id, or
	// nothing. Never touch r from a follower goroutine — the front rank still owns it.
	inboundTrace := r.Header.Get(traceHeader)
	results := make(chan epFanoutResult, len(urls))
	for _, target := range urls {
		target := target
		go func() {
			req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, target, bytes.NewReader(body))
			if err != nil {
				results <- epFanoutResult{target: target, err: err}
				return
			}
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set(epFollowerHeader, "1")
			if inboundTrace != "" {
				req.Header.Set(traceHeader, inboundTrace)
			}
			resp, err := epFanoutClient.Do(req)
			if err != nil {
				results <- epFanoutResult{target: target, err: err}
				return
			}
			defer resp.Body.Close()
			// The follower must run its decode to completion so it reaches every
			// per-step collective in lockstep with the front rank. For a streaming
			// follower that means draining the whole SSE stream: closing the body
			// after a fixed prefix would cancel the follower mid-decode and re-strand
			// the front rank's collective. Non-streaming responses are already whole,
			// so a bounded snippet is enough to log a non-2xx follower.
			var snippet []byte
			if stream {
				_, _ = io.Copy(io.Discard, resp.Body)
			} else {
				snippet, _ = io.ReadAll(io.LimitReader(resp.Body, 512))
			}
			results <- epFanoutResult{target: target, status: resp.StatusCode, body: string(snippet)}
		}()
	}
	return func() {
		timeout := durEnv("FAK_EP_FANOUT_WAIT_TIMEOUT_S", 5*time.Second)
		var timer <-chan time.Time
		if timeout > 0 {
			timer = time.After(timeout)
		}
		for range urls {
			select {
			case res := <-results:
				if res.err != nil {
					s.logEPFanout("gateway: EP fanout follower %s failed: %v", res.target, res.err)
					continue
				}
				if res.status < 200 || res.status >= 300 {
					s.logEPFanout("gateway: EP fanout follower %s status=%d body=%q", res.target, res.status, res.body)
				}
			case <-timer:
				s.logEPFanout("gateway: EP fanout follower wait timed out after %s", timeout)
				return
			}
		}
	}, true
}

type epFanoutResult struct {
	target string
	status int
	body   string
	err    error
}

func (s *Server) logEPFanout(format string, args ...any) {
	if s != nil && s.logf != nil {
		s.logf(format, args...)
	}
}

// epFanoutURLsFromEnv expands FAK_EP_FANOUT_ADDRS — a delimiter-separated list of
// follower rank addresses — into one absolute follower URL per rank on the given
// served route. An address may omit the scheme (http:// is assumed) and may carry a
// trailing slash. An unset or empty variable yields no URLs, which is what makes the
// bridge inert on a single-rank serve.
func epFanoutURLsFromEnv(route string) []string {
	raw := os.Getenv("FAK_EP_FANOUT_ADDRS")
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.FieldsFunc(raw, func(r rune) bool {
		return r == ',' || r == ' ' || r == '\n' || r == '\t' || r == ';'
	})
	urls := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if !strings.HasPrefix(part, "http://") && !strings.HasPrefix(part, "https://") {
			part = "http://" + part
		}
		urls = append(urls, strings.TrimRight(part, "/")+route)
	}
	return urls
}
