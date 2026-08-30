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
// slower buffered fallback on the first hit. Anthropic refuses a streamed turn two ways,
// and the SECOND way wears a 200: HTTP 200 + text/event-stream, then an SSE `error` frame
// as the first event, before any message_start. That in-band refusal is the same condition
// in different clothing, so it is classified back onto its equivalent status
// (anthropicInBandErrorStatus) and takes the SAME arms — a transient one re-sends under the
// same budget, a request error surfaces at once with its real status (#5491). A connection
// or non-retryable status failure
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
	triedTransientRetry := false
	triedTransientTarget := false
	maxAttempts, deadline, budgetOn := retryBounds(time.Now())
	var rs retryState // shared between-attempt truth (#1358, #1362) — see retry_state.go
	for attempt := 0; attempt < maxAttempts; attempt++ {
		// Surface the retry before the otherwise-invisible backoff sleep (the same hook the
		// buffered + OpenAI-stream paths use), then wait. Attempt zero is a no-op.
		stop, err := p.waitBeforeAttempt(ctx, attempt, &rs, deadline, budgetOn)
		if err != nil {
			return err
		}
		if stop {
			break
		}
		req, err := http.NewRequestWithContext(ctx, "POST", adapter.Endpoint(p.BaseURL, p.ModelID), bytes.NewReader(body))
		if err != nil {
			return err
		}
		// Send from call.headers() so a seat rehome (which mutates call.apiKey/extraHeaders in
		// place) or a 401 self-heal takes effect on the very next re-send — the same credential
		// source the buffered/planner-stream paths use, folding in the anthropic-beta union.
		call.applyHeaders(req)
		finishProvider := BeginProviderCall(req)
		req.Header.Set("Accept", "text/event-stream")

		r, derr := p.Client.Do(req)
		finishProvider(providerResponseStatus(r), derr)
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
			// The stream is framed INSIDE the loop because a 200 is not yet an acceptance
			// (#5491): Anthropic refuses a streamed turn two ways, and the second one is an
			// HTTP 200 + text/event-stream carrying an SSE `error` frame as its FIRST event,
			// before any message_start. That in-band refusal is exactly as transient as the
			// HTTP 529 the arm below already retries, and it has emitted nothing to the
			// caller, so it belongs on the same invisible retry — not outside the loop where
			// it used to become an un-retried, un-counted 502.
			refusalStatus, refusalFrame, serr := p.relayAnthropicStream(r, onEvent)
			if refusalStatus == 0 {
				return serr // a relayed turn (nil), or a real read/stall/unsupported failure
			}
			if !retryableStatus(refusalStatus) {
				// A request error the upstream merely chose to report in-band (an
				// invalid_request_error / authentication_error): surface the status it maps
				// to immediately, exactly as the same status on the wire would be — no retry
				// burst, and no 502 costume over a 400.
				return newUpstreamStatusError(refusalStatus, refusalFrame, r.Header, 400)
			}
			// Project an in-band pre-start refusal into the shared rejected-response policy so
			// Anthropic's HTTP-200 overload envelope gets the same quick retry and transient
			// target failover as an HTTP 529.
			rejected := &http.Response{StatusCode: refusalStatus, Header: r.Header}
			retry, rewind, statusErr := call.handleRejectedResponse(ctx, p, &rs, rejected, refusalFrame, attempt, rejectedResponseRetry{
				triedAuthRefresh: &triedAuthRefresh, forbidden: &fbState,
				triedRehome: &triedRehome, rehomePending: &rehomePending,
				triedFailover: &triedAccountFailover, failoverPending: &accountFailoverPending,
				triedTransientRetry: &triedTransientRetry, triedTransientTarget: &triedTransientTarget,
				bodyCap: 400,
			})
			if statusErr != nil {
				return statusErr
			}
			if rewind {
				attempt--
			}
			if retry {
				continue
			}
		}
		raw, _ := io.ReadAll(io.LimitReader(r.Body, 4096))
		r.Body.Close()
		retry, rewind, statusErr := call.handleRejectedResponse(ctx, p, &rs, r, raw, attempt, rejectedResponseRetry{
			triedAuthRefresh: &triedAuthRefresh, forbidden: &fbState,
			triedRehome: &triedRehome, rehomePending: &rehomePending,
			triedFailover: &triedAccountFailover, failoverPending: &accountFailoverPending,
			triedTransientRetry: &triedTransientRetry, triedTransientTarget: &triedTransientTarget,
			bodyCap: 400,
		})
		if statusErr != nil {
			return statusErr
		}
		if rewind {
			attempt--
		}
		if retry {
			continue
		}
	}
	return rs.exhausted("planner: streaming failed after retries")
}

// errAnthropicInBandRefusal stops the SSE parse the moment an in-band pre-start refusal is
// recognized. It never escapes relayAnthropicStream — the refusal is reported through that
// method's status/frame returns instead — so no caller ever has to match on it.
var errAnthropicInBandRefusal = errors.New("agent: anthropic refused the stream in-band before message_start")

// relayAnthropicStream frames ONE accepted (HTTP 200) upstream response into onEvent calls
// and closes its body. It is the post-acceptance half of StreamAnthropicRaw, kept as its own
// method so the retry loop can call it per attempt.
//
// The three outcomes are disjoint:
//
//   - (0, nil, nil) — the turn was relayed to a clean end.
//   - (0, nil, err) — a real failure the caller cannot retry away: the wire cannot be framed
//     as SSE (ErrStreamingUnsupported), the upstream went silent (UpstreamStalledError), the
//     read failed, or onEvent itself asked to stop.
//   - (status, frame, nil) — the upstream answered 200 + text/event-stream and then refused
//     IN-BAND with an SSE `error` frame BEFORE any message_start (#5491). Nothing was handed
//     to onEvent, so the caller still owns the response: it re-sends a transient refusal under
//     the normal backoff and surfaces a request error with the status this maps to.
//
// The pre-start window is bounded by message_start deliberately: that frame is what opens the
// caller's own client stream, so an `error` frame arriving after it is a MID-stream failure the
// caller must forward, and it is passed through to onEvent untouched exactly as before.
func (p *HTTPPlanner) relayAnthropicStream(resp *http.Response, onEvent func(AnthropicSSEEvent) error) (refusalStatus int, refusalFrame []byte, err error) {
	defer resp.Body.Close()
	// The gateway only takes this path against the real Anthropic API, but guard anyway:
	// an upstream that ignores stream and replies with one buffered JSON body cannot be
	// framed as SSE, so surface that as unsupported BEFORE any event (the caller falls
	// back to the buffered path) rather than emit a malformed stream.
	if !upstreamStreamsSSE(resp) {
		return 0, nil, ErrStreamingUnsupported
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
	// producing no content trips in ≤the planner's configured progress window instead of
	// riding the ceiling.
	sr := newStallReader(resp.Body, streamStallTimeout(), p.streamProgressWindow())
	defer sr.Close()
	started := false
	var frame []byte
	onProgressingEvent := func(ev AnthropicSSEEvent) error {
		if ev.Event == "message_start" {
			started = true
		}
		if !started && ev.Event == "error" {
			if status, ok := anthropicInBandErrorStatus(ev.Data); ok {
				refusalStatus, frame = status, append([]byte(nil), ev.Data...)
				return errAnthropicInBandRefusal
			}
			// An `error` frame whose type this build does not recognize cannot be mapped to
			// a status, so it is NOT claimed as a classified refusal — it falls through to
			// onEvent, whose pre-start handling still owns it.
		}
		if anthropicFrameAdvancesTurn(ev) {
			sr.noteProgress()
		}
		return onEvent(ev)
	}
	if perr := parseAnthropicSSE(sr, onProgressingEvent); perr != nil {
		if refusalStatus != 0 && errors.Is(perr, errAnthropicInBandRefusal) {
			return refusalStatus, frame, nil
		}
		if errors.Is(perr, ErrUpstreamStalled) {
			kind, window := sr.stallCause()
			return 0, nil, &UpstreamStalledError{Idle: window, Kind: kind, Err: perr}
		}
		return 0, nil, perr
	}
	return 0, nil, nil
}

// anthropicInBandErrorStatus maps an Anthropic in-band SSE `error` frame — the
// {"type":"error","error":{"type":…}} payload the API sends on a 200 when it decides to
// refuse a turn after the headers are already out — to the HTTP status the SAME refusal
// carries when the API reports it on the wire instead. The two are the same conditions
// reported two ways (which is why #5491 was intermittent), so mapping the in-band form back
// onto its status is what lets ONE classification serve both: retryableStatus decides the
// retry, and the gateway's upstreamErrorStatus/errType ladder gives the client the upstream's
// own type (a 529 reads as overloaded_error) rather than a flattened 502/server_error.
//
// ok is false for any type this build does not recognize: an unknown refusal is left to the
// caller's own pre-start handling rather than guessed into a status that might silently make
// it retryable.
func anthropicInBandErrorStatus(data []byte) (status int, ok bool) {
	var frame struct {
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	if json.Unmarshal(data, &frame) != nil {
		return 0, false
	}
	switch strings.ToLower(strings.TrimSpace(frame.Error.Type)) {
	case "invalid_request_error":
		return http.StatusBadRequest, true
	case "authentication_error":
		return http.StatusUnauthorized, true
	case "permission_error":
		return http.StatusForbidden, true
	case "not_found_error":
		return http.StatusNotFound, true
	case "request_too_large":
		return http.StatusRequestEntityTooLarge, true
	case "timeout_error":
		return http.StatusRequestTimeout, true
	case "rate_limit_error":
		return http.StatusTooManyRequests, true
	case "api_error":
		return http.StatusInternalServerError, true
	case "overloaded_error":
		return statusOverloaded, true
	}
	return 0, false
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
