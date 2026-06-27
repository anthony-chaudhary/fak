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
	"io"
	"net/http"
	"strings"
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
// A connection or non-200 status failure surfaces BEFORE any onEvent call, so the
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
	req, err := http.NewRequestWithContext(ctx, "POST", adapter.Endpoint(p.BaseURL, p.ModelID), bytes.NewReader(body))
	if err != nil {
		return err
	}
	// Transparent hop: authenticate with the inbound client's own credential when it
	// supplied one (passthrough), else the planner's configured key — the same scheme
	// the buffered path resolves in prepareUpstream.
	key := p.APIKey
	if apiKey != "" {
		key = apiKey
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

	resp, err := p.Client.Do(req)
	if err != nil {
		if deterministicTransportError(err) {
			return &UpstreamUnreachableError{Err: err}
		}
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return &UpstreamStatusError{Status: resp.StatusCode, Body: truncate(raw, 400)}
	}
	// The gateway only takes this path against the real Anthropic API, but guard anyway:
	// an upstream that ignores stream and replies with one buffered JSON body cannot be
	// framed as SSE, so surface that as unsupported BEFORE any event (the caller falls
	// back to the buffered path) rather than emit a malformed stream.
	if ct := resp.Header.Get("Content-Type"); ct != "" && !strings.Contains(ct, "event-stream") {
		return ErrStreamingUnsupported
	}
	return parseAnthropicSSE(resp.Body, onEvent)
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
