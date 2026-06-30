package agent

// anthropic_stream.go is the streaming twin of the buffered Anthropic passthrough in
// chat.go. Where Complete forwards the inbound client's raw bytes with stream forced
// OFF (so a non-streaming planner can parse the buffered JSON), StreamAnthropicRaw
// forwards those SAME bytes with stream ON and relays the real Anthropic Messages SSE
// to the caller event-by-event. This is the half that turns the flagship
// `fak guard -- claude` from "wait for the whole turn, then synthesize SSE" (TTFT ==
// full generation) into a true live token stream whose first token tracks the model —
// and lets the prompt-cache hit's fast prefill actually be FELT, not buffered away.
//
// The kernel boundary is preserved by the caller, not here: this file only frames the
// upstream SSE. The gateway interprets the events — relaying text/thinking deltas live
// while HOLDING every tool_use block for k.Decide before the client ever sees it.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// AnthropicSSEEvent is one event parsed from an upstream Anthropic Messages SSE
// stream: the SSE `event:` name and its raw `data:` JSON payload, verbatim. The
// gateway interprets these — relaying text/thinking deltas the instant they arrive and
// buffering tool_use blocks for kernel adjudication — so the Data is kept raw rather
// than decoded into a typed shape this layer would have to keep in lock-step with the
// wire.
type AnthropicSSEEvent struct {
	Event string
	Data  json.RawMessage
}

// StreamAnthropicRaw opens a TRUE token stream against the real Anthropic Messages API
// by forwarding rawBody (the inbound client's bytes, so its prompt-cache prefix
// survives byte-for-byte → a real cache hit) with stream:true, and invokes onEvent for
// each SSE event as it arrives. It is the streaming counterpart of the buffered
// passthrough in Complete: same raw-body + credential + beta pass-through, but the
// upstream delivers an SSE token stream instead of one buffered JSON body.
//
// A transient transport error or a retryable status (429 rate-limit, 503/529 overload,
// 408/5xx transient) is RETRIED here with backoff+jitter+Retry-After — BEFORE any onEvent
// call, where the retry is invisible to the client — exactly as Complete/CompleteStream
// do, so a real Anthropic 429/529 window no longer collapses the flagship stream to the
// slower buffered fallback on the first hit. A connection or non-retryable status failure
// (or a retryable one that survived every attempt) surfaces BEFORE any onEvent call, so the
// caller can still fall back to the buffered path having sent the client nothing AND
// without a second generation having been billed (a non-200 produced no tokens). Once
// events have flowed, a read error is returned as-is for the caller to terminate the
// open stream. Only the Anthropic wire is supported — any other provider (or an
// upstream that ignores stream:true and answers with buffered JSON) returns
// ErrStreamingUnsupported without leaking a half-stream.
func (p *HTTPPlanner) StreamAnthropicRaw(ctx context.Context, rawBody []byte, apiKey, beta string, onEvent func(AnthropicSSEEvent) error) error {
	adapter, err := p.transcriptAdapter()
	if err != nil {
		return err
	}
	if adapter.Provider() != ProviderAnthropic {
		return ErrStreamingUnsupported
	}

	body := forceAnthropicStreaming(rawBody)
	// Transparent hop: authenticate with the inbound client's own credential when it
	// supplied one (passthrough), else the planner's EFFECTIVE key — the same scheme the
	// buffered path resolves in prepareUpstream. effectiveAPIKey re-resolves a rotating
	// subscription token per request, so a long streamed session never sends a stale
	// boot-time bearer (the 401-after-relogin bug).
	key := p.effectiveAPIKey()
	if apiKey != "" {
		key = apiKey
	}
	// Retry a transient transport error OR a retryable status (429 rate-limit, 503/529
	// overload, 408/5xx transient) with the SAME backoff+jitter+Retry-After policy as
	// Complete/CompleteStream — but ONLY here, before the first SSE byte reaches the client,
	// where a retry is invisible and the caller can still choose an HTTP status. Until now
	// this flagship `fak guard -- claude` passthrough retried ONLY a 401; a real Anthropic
	// 429/529 collapsed the live stream to the slower buffered fallback on the very first
	// one. A fleet sharing one upstream account rides out a long overload window far better
	// when the streaming path itself backs off and retries (plannerMaxAttempts, default 8),
	// instead of giving up after a single hit. A deterministic dial failure still fails fast
	// (no backoff). A 401 on the rotating-subscription path self-heals ONCE: the on-disk
	// OAuth token may have rotated or been briefly torn between resolve and send, so we
	// re-read it fresh and re-send immediately (uncounted). Every other status is a request
	// error a retry cannot fix and is returned as-is. Each non-200 body is drained+closed in
	// the loop; only the successful 200 escapes to the SSE reader below.
	authRefreshable := apiKey == "" && p.APIKeyFunc != nil
	triedAuthRefresh := false
	maxAttempts, deadline, budgetOn := retryBounds(time.Now())
	var lastErr error
	// lastStatusErr: see Complete (#1358) — the last real-HTTP-status error, never cleared
	// by a later transport glitch, so exhaustion surfaces the true status + Retry-After.
	var lastStatusErr *UpstreamStatusError
	lastStatus := 0      // the status that triggered the pending retry (0 = a transient transport error)
	lastRetryAfter := "" // the triggering response's Retry-After, honored as the next wait
	var resp *http.Response
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			// Surface the retry BEFORE the otherwise-invisible backoff sleep (the same hook
			// the buffered + OpenAI-stream paths use, so the gateway's `fak-turn … retry`
			// line and retry counter fire for the streaming passthrough too), then wait —
			// honoring a named Retry-After, else the jittered exponential schedule. A
			// cancelled context aborts the wait promptly rather than sleeping it out. When the
			// time budget is the bound, the wait is clamped to the remaining budget.
			var wait time.Duration
			if budgetOn {
				wait = retryWaitWithin(attempt, lastRetryAfter, deadline, time.Now())
				if wait < 0 {
					break
				}
			} else {
				wait = retryWait(attempt, lastRetryAfter)
			}
			if p.RetryNotify != nil {
				p.RetryNotify(attempt, lastStatus, wait)
			}
			if err := sleepCtx(ctx, wait); err != nil {
				return err
			}
		}
		req, err := http.NewRequestWithContext(ctx, "POST", adapter.Endpoint(p.BaseURL, p.ModelID), bytes.NewReader(body))
		if err != nil {
			return err
		}
		h := adapter.Headers(key)
		if beta != "" {
			h["anthropic-beta"] = mergeBeta(h["anthropic-beta"], beta)
		}
		for k, v := range h {
			if v != "" {
				req.Header.Set(k, v)
			}
		}
		req.Header.Set("Accept", "text/event-stream")

		r, derr := p.Client.Do(req)
		if derr != nil {
			// A deterministic dial failure (refused/NXDOMAIN/TLS) cannot be retried away —
			// fail fast and tagged. A transient transport error (timeout, mid-flight reset)
			// gets the same backoff as a retryable status.
			if deterministicTransportError(derr) {
				return &UpstreamUnreachableError{Err: derr}
			}
			lastErr = derr
			lastStatus = 0
			lastRetryAfter = ""
			continue // lastStatusErr left intact
		}
		if r.StatusCode == http.StatusOK {
			resp = r
			break
		}
		raw, _ := io.ReadAll(io.LimitReader(r.Body, 4096))
		r.Body.Close()
		if retryableStatus(r.StatusCode) {
			ra := r.Header.Get("Retry-After")
			se := &UpstreamStatusError{Status: r.StatusCode, Body: truncate(raw, 400), RetryAfter: ra}
			lastErr = se
			lastStatusErr = se
			lastStatus = r.StatusCode
			lastRetryAfter = ra
			continue
		}
		// A 401 on the rotating-subscription path: re-resolve the credential fresh and retry
		// ONCE (attempt-- so the refresh re-send is immediate and uncounted), mirroring
		// Complete/CompleteStream. waitForFreshAPIKey polls the on-disk token across the
		// re-login grace window so a user logging back in mid-stream is adopted and the live
		// session self-heals; a no-op (func gone, or no fresher token within the window)
		// falls through to the raw 401.
		if r.StatusCode == http.StatusUnauthorized && !triedAuthRefresh && authRefreshable {
			if fresh, ok := waitForFreshAPIKey(ctx, p, key); ok {
				key = fresh
				triedAuthRefresh = true
				attempt--
				continue
			}
		}
		return &UpstreamStatusError{Status: r.StatusCode, Body: truncate(raw, 400), RetryAfter: r.Header.Get("Retry-After")}
	}
	if resp == nil {
		// Prefer the real upstream status (and Retry-After) over a later transport glitch (#1358).
		if lastStatusErr != nil {
			return fmt.Errorf("planner: streaming failed after retries: %w", lastStatusErr)
		}
		return fmt.Errorf("planner: streaming failed after retries: %w", lastErr)
	}
	defer resp.Body.Close()
	// The gateway only takes this path against the real Anthropic API, but guard anyway:
	// an upstream that ignores stream and replies with one buffered JSON body cannot be
	// framed as SSE, so surface that as unsupported BEFORE any event (the caller falls
	// back to the buffered path) rather than emit a malformed stream.
	if ct := resp.Header.Get("Content-Type"); ct != "" && !strings.Contains(ct, "event-stream") {
		return ErrStreamingUnsupported
	}
	// Wrap the body in an idle-read deadline so an upstream that opens the stream and then
	// goes silent (a transient overload / "API issue") fails in ≤streamStallTimeout()
	// instead of blocking parseAnthropicSSE on resp.Body.Read until the 600s whole-request
	// Client.Timeout fires. A healthy stream's `ping`/keepalive/delta frames keep resetting
	// the window, so only true silence trips it. Surface the trip as the typed
	// UpstreamStalledError the gateway logs distinctly from a normal read failure.
	sr := newStallReader(resp.Body, streamStallTimeout())
	defer sr.Close()
	if err := parseAnthropicSSE(sr, onEvent); err != nil {
		if errors.Is(err, ErrUpstreamStalled) {
			return &UpstreamStalledError{Idle: streamStallTimeout(), Err: err}
		}
		return err
	}
	return nil
}

// parseAnthropicSSE reads an Anthropic Messages SSE body and invokes onEvent once per
// `event:`/`data:` frame (frames are separated by a blank line). Multi-line data
// payloads are joined with newlines per the SSE spec; non-event lines (comments, id:,
// retry:) are ignored. A frame with no data is dropped. The scanner ceiling is raised
// well past the 64 KiB default so a large input_json_delta (a big tool argument) is
// never truncated.
func parseAnthropicSSE(r io.Reader, onEvent func(AnthropicSSEEvent) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	var (
		event string
		data  strings.Builder
	)
	flush := func() error {
		defer func() { event = ""; data.Reset() }()
		if data.Len() == 0 {
			return nil
		}
		return onEvent(AnthropicSSEEvent{Event: event, Data: json.RawMessage(data.String())})
	}
	for sc.Scan() {
		line := sc.Text()
		if line == "" { // frame boundary
			if err := flush(); err != nil {
				return err
			}
			continue
		}
		if v, ok := strings.CutPrefix(line, "event:"); ok {
			event = strings.TrimSpace(v)
			continue
		}
		if v, ok := strings.CutPrefix(line, "data:"); ok {
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(strings.TrimSpace(v))
			continue
		}
		// ignore SSE comment / id: / retry: lines
	}
	if err := sc.Err(); err != nil {
		return err
	}
	return flush() // a final frame not terminated by a trailing blank line
}

// forceAnthropicStreaming returns the raw Anthropic request body with its top-level
// "stream" flag set to true, so the upstream delivers an SSE token stream. A body that
// ALREADY carries stream:true is returned UNCHANGED (byte-identical) — the common case,
// since the gateway only takes the streaming path when the inbound client itself asked
// to stream — so its exact cache prefix is preserved. Only a body missing or with a
// non-true stream flag is re-marshalled; the cached prefix is the system/tools/messages
// content, unaffected by the top-level key order or the stream flag (the mirror of
// forceAnthropicNonStreaming). A body that does not parse as a JSON object is returned
// unchanged.
func forceAnthropicStreaming(raw []byte) []byte {
	return setAnthropicStreamFlag(raw, "true", func(v json.RawMessage, present bool) bool {
		return present && strings.TrimSpace(string(v)) == "true" // already streaming — keep its exact prefix
	})
}
