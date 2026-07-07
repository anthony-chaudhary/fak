package gateway

// http_epfanout.go — the expert-parallel (EP) request-fanout bridge (#2955), split
// out of http.go along its concern seam (#2999). Everything here serves one job:
// mirror an inbound non-streaming chat request from the EP front rank to its
// follower ranks so every process of a sharded serve reaches the collectives.

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

// epFanoutClient carries the loopback follower requests. It MUST have a timeout:
// a hung or slow follower rank must not pin the front rank's fanout goroutine
// forever (boundarylint MISSING_HTTP_TIMEOUT). The fanout body is a full chat
// request replayed to a same-host follower, so the ceiling is generous and
// env-tunable, matching http.go's FAK_HTTP_* convention.
var epFanoutClient = &http.Client{Timeout: durEnv("FAK_EP_FANOUT_TIMEOUT_S", 120*time.Second)}

// startEPFanoutFollowers mirrors an inbound non-streaming chat request from the EP
// front rank to its follower rank endpoints before the front rank enters the local
// in-kernel decode. Rank-local expert parallelism reduces the routed-expert delta
// through a process-group AllReduce; that collective makes progress only if every
// rank runs the same forward pass. FAK_EP_FANOUT_ADDRS is therefore the temporary
// single-endpoint bridge for the sharded serve: the front rank receives the client
// request, followers receive the identical loopback request with X-Fak-EP-Follower
// set, and every process reaches the collectives concurrently. The follower header
// prevents recursive fanout if an operator points the bridge at another front rank.
func (s *Server) startEPFanoutFollowers(w http.ResponseWriter, r *http.Request) (func(), bool) {
	if r.Header.Get(epFollowerHeader) != "" {
		return func() {}, true
	}
	urls := epFanoutURLsFromEnv()
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
	if err := json.Unmarshal(body, &meta); err != nil || meta.Stream {
		// Malformed bodies will be rejected by the normal decoder below. Streaming EP
		// needs a coordinated streaming bridge; do not start follower requests whose
		// bodies the helper would close before the decode finishes.
		return func() {}, true
	}
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
			if trace := r.Header.Get("X-Trace-Id"); trace != "" {
				req.Header.Set("X-Trace-Id", trace)
			}
			resp, err := epFanoutClient.Do(req)
			if err != nil {
				results <- epFanoutResult{target: target, err: err}
				return
			}
			defer resp.Body.Close()
			snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
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

func epFanoutURLsFromEnv() []string {
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
		urls = append(urls, strings.TrimRight(part, "/")+"/v1/chat/completions")
	}
	return urls
}
