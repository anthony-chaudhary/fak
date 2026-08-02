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
	// Carry the resolved credential + adapter in an upstreamCall so this flagship streaming
	// path shares the SAME 429/403 cap-rehome seam (noteRetryableCapMaybeRehome) and 401
	// self-heal the buffered/planner-stream paths use — instead of the bare backoff-only retry
	// it had before, which left a live `fak guard -- claude` session sleeping on a capped seat
	// toward an hours-away reset while a free sibling seat sat idle. failoverAccountCred mutates
	// call.apiKey in place, so a rehome persists across attempts here exactly as it does there.
	call := &upstreamCall{
		adapter:         adapter,
		apiKey:          key,
		upstreamBeta:    beta,
		extraHeaders:    p.effectiveExtraHeaders(),
		authRefreshable: apiKey == "" && p.APIKeyFunc != nil,
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
	triedAuthRefresh := false
	var fbState forbiddenRetryState // bounded transient-403 recovery arm (see retry.go), mirrors Complete
	// 429/403-account-cap seat rehome, one swap max (see Complete/streamConnect). Safe on this
	// pre-first-byte loop: a retryable status here has emitted nothing to the client yet, so
	// swapping seats and re-sending is invisible; rehomePending defers the confirmed-rehome
	// notify to the 200.
	triedRehome := false
	rehomePending := false
	// Account-scoped entitlement/org walls use the same one-shot sibling-account failover as
	// Complete and CompleteStream. Keep this distinct from cap rehome: a hard 403 is not a
	// cooldown and must not enter the hours-away cap wait path.
	triedAccountFailover := false
	accountFailoverPending := false
	maxAttempts, deadline, budgetOn := retryBounds(time.Now())
	var rs retryState // shared between-attempt truth (#1358, #1362) — see retry_state.go
	var resp *http.Response
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			// Surface the retry BEFORE the otherwise-invisible backoff sleep (the same hook the
			// buffered + OpenAI-stream paths use, so the gateway's `fak-turn … retry` line fires
			// for the streaming passthrough too), then wait. See retryBackoffWait.
			stop, err := p.retryBackoffWait(ctx, attempt, rs.lastStatus, rs.lastRetryAfter, rs.lastCapWait, rs.lastStatusErr, deadline, budgetOn)
			if err != nil {
				return err
			}
			if stop {
				break
			}
		}
		req, err := http.NewRequestWithContext(ctx, "POST", adapter.Endpoint(p.BaseURL, p.ModelID), bytes.NewReader(body))
		if err != nil {
			return err
		}
		// Send from call.headers() so a seat rehome (which mutates call.apiKey/extraHeaders in
		// place) or a 401 self-heal takes effect on the very next re-send — the same credential
		// source the buffered/planner-stream paths use, folding in the anthropic-beta union.
		call.applyHeaders(req)
		req.Header.Set("Accept", "text/event-stream")

		r, derr := p.Client.Do(req)
		if derr != nil {
			// A deterministic dial failure (refused/NXDOMAIN/TLS) cannot be retried away —
			// fail fast and tagged. A transient transport error (timeout, mid-flight reset)
			// gets the same backoff as a retryable status.
			if uerr := classifyDoError(derr, &rs); uerr != nil {
				return uerr
			}
			continue
		}
		if r.StatusCode == http.StatusOK {
			// A 200 after the 403 arm fired is a CONFIRMED transient-403 self-heal (see Complete).
			fbState.noteRecovered(p, attempt)
			// A 200 after a 429/403-account-cap seat rehome is a CONFIRMED rehome (see Complete).
			notifyRehomeRecovered(p, &rehomePending, attempt)
			if accountFailoverPending {
				notifyAccountFailover(p, AccountFailoverRecovered, attempt)
				accountFailoverPending = false
			}
			resp = r
			break
		}
		raw, _ := io.ReadAll(io.LimitReader(r.Body, 4096))
		r.Body.Close()
		if retryableStatus(r.StatusCode) {
			// A 429 ACCOUNT CAP (session/weekly/usage, or a generic 429 whose relayed wait is
			// hours-away) rehomes the live stream ONCE to a permitted sibling seat rather than
			// sleep the guarded session on the capped one toward a reset no wrapped client can
			// outlast; a transient throttle keeps its seat. The SAME seam the buffered and planner-
			// stream loops call, so the flagship path no longer diverges (its prior bare backoff
			// was the reason a capped `fak guard -- claude` session never rehomed).
			if call.noteRetryableCapMaybeRehome(p, &rs, r.StatusCode, raw, r.Header, 400, false, &triedRehome, &rehomePending, attempt) == capRehomeResend {
				attempt--
			}
			continue
		}
		// A 401 on the rotating-subscription path: re-resolve the credential fresh and retry
		// ONCE (attempt-- so the refresh re-send is immediate and uncounted), mirroring
		// Complete/CompleteStream. refreshAPIKeyWait polls the on-disk token across the re-login
		// grace window so a user logging back in mid-stream is adopted (updating call.apiKey) and
		// the live session self-heals; a no-op (func gone, or no fresher token within the window)
		// falls through to the raw 401.
		if r.StatusCode == http.StatusUnauthorized && !triedAuthRefresh && call.authRefreshable {
			if call.refreshAPIKeyWait(ctx, p) {
				triedAuthRefresh = true
				notifyAuthRefresh(p, AuthRefreshRecovered, attempt)
				attempt--
				continue
			}
			notifyAuthRefresh(p, AuthRefreshExhausted, attempt)
		}
		// A 403/402 USAGE/OVERAGE cap (see Complete): a self-recovering rolling-window cap that
		// Anthropic surfaces as a 403 with an org-flavored body — only the unified/overage headers
		// reveal it. The overage headers already proved a recovering cap (mustRehome=true): swap to
		// a free seat or ride the cap-aware backoff toward the reset — never the seconds-scale
		// forbidden arm and not a terminal wall. Precedes the transient arm below.
		if (r.StatusCode == http.StatusForbidden || r.StatusCode == http.StatusPaymentRequired) &&
			usageOrOverageRejected(r.Header) {
			if call.noteRetryableCapMaybeRehome(p, &rs, r.StatusCode, raw, r.Header, 400, true, &triedRehome, &rehomePending, attempt) == capRehomeResend {
				attempt--
			}
			continue
		}
		// A 403's bounded transient-recovery arm (self-contained short paced wait; see Complete).
		if r.StatusCode == http.StatusForbidden {
			if fbState.step403(ctx, p, raw, attempt) {
				attempt--
				continue
			}
		}
		// The raw Anthropic SSE path is the path used by an interactive `fak guard -- claude`
		// turn. It must not diverge from the buffered/planner-stream paths here: once the
		// transient-403 arm has ruled out a clearing flap, an account-scoped org/entitlement
		// wall gets one sibling-account swap and an immediate invisible re-send.
		if classifyUpstream(r.StatusCode, raw, r.Header) == RemedyFailoverAccount && !triedAccountFailover && p.AccountFailoverFunc != nil {
			if call.failoverAccountCred(p, RemedyFailoverAccount.String()) {
				triedAccountFailover = true
				accountFailoverPending = true
				rs.lastErr = &UpstreamStatusError{Status: r.StatusCode, Body: truncate(raw, 400), RetryAfter: r.Header.Get("Retry-After")}
				rs.lastStatus = r.StatusCode
				rs.lastRetryAfter = ""
				attempt--
				continue
			}
			triedAccountFailover = true
			notifyAccountFailover(p, AccountFailoverExhausted, attempt)
		}
		return &UpstreamStatusError{Status: r.StatusCode, Body: truncate(raw, 400), RetryAfter: r.Header.Get("Retry-After")}
	}
	if resp == nil {
		return rs.exhausted("planner: streaming failed after retries")
	}
	defer resp.Body.Close()
	// The gateway only takes this path against the real Anthropic API, but guard anyway:
	// an upstream that ignores stream and replies with one buffered JSON body cannot be
	// framed as SSE, so surface that as unsupported BEFORE any event (the caller falls
	// back to the buffered path) rather than emit a malformed stream.
	if !upstreamStreamsSSE(resp) {
		return ErrStreamingUnsupported
	}
	// Wrap the body in an idle-read deadline so an upstream that opens the stream and then
	// goes silent (a transient overload / "API issue") fails in ≤streamStallTimeout()
	// instead of blocking parseAnthropicSSE on resp.Body.Read until the 600s whole-request
	// Client.Timeout fires. A healthy stream's `ping`/keepalive/delta frames keep resetting
	// the window, so only true silence trips it. Surface the trip as the typed
	// UpstreamStalledError the gateway logs distinctly from a normal read failure.
	//
	// The second, longer deadline covers the case the byte one cannot see (#5486): re-arm it
	// ONLY on a frame that advances the turn. A `ping` still re-arms the byte window (it IS
	// bytes) but deliberately not this one, so an upstream warm enough to keep pinging while
	// producing no content trips in ≤streamProgressTimeout() instead of riding the ceiling.
	sr := newStallReader(resp.Body, streamStallTimeout(), streamProgressTimeout())
	defer sr.Close()
	onProgressingEvent := func(ev AnthropicSSEEvent) error {
		if anthropicFrameAdvancesTurn(ev) {
			sr.noteProgress()
		}
		return onEvent(ev)
	}
	if err := parseAnthropicSSE(sr, onProgressingEvent); err != nil {
		if errors.Is(err, ErrUpstreamStalled) {
			kind, window := sr.stallCause()
			return &UpstreamStalledError{Idle: window, Kind: kind, Err: err}
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
