package agent

// anthropic_inband.go classifies the OTHER way an Anthropic upstream refuses a streamed
// turn, and the one the retry loop in anthropic_stream.go structurally could not see:
// the upstream answers HTTP 200 + text/event-stream and THEN refuses IN-BAND with an SSE
// `error` frame instead of a message_start.
//
// The retry loop decides on r.StatusCode, so a 200 escapes it entirely — the refusal
// arrives later, inside the SSE reader, as an onEvent verdict. An in-band
// `overloaded_error` is exactly as transient as the HTTP 529 the status path already
// retries, so before this classification existed the flagship `fak guard -- claude`
// stream died on the FIRST in-band refusal: one upstream hit, no retry, no buffered
// fallback, and an opaque 502 whose `server_error` type had lost the upstream's own
// retryable signal. UpstreamInBandError is how the SSE reader hands that verdict back to
// the retry loop so the re-send happens where it is still invisible to the client.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// UpstreamInBandError is an Anthropic SSE `error` frame that arrived on an otherwise-200
// stream BEFORE anything was relayed to the client. Status is the HTTP status the frame's
// error type corresponds to (anthropicInBandStatus), which is what makes the frame
// comparable to a non-200: the ordinary retryableStatus policy then decides it, so an
// in-band overload and an HTTP 529 get the same backoff, and an in-band request error and
// an HTTP 400 both surface immediately.
//
// It is deliberately NOT an UpstreamStatusError: the distinction matters to the retry loop
// (an in-band refusal was read from a 200 body, so the loop must re-send rather than
// return), and StatusError() converts it at the boundary where a terminal answer is owed.
type UpstreamInBandError struct {
	// Type is the upstream's own error type verbatim ("overloaded_error",
	// "api_error", …), or "" when the frame carried none.
	Type string
	// Message is the upstream's error text, for the OPERATOR LOG only — like
	// UpstreamStatusError.Body it is not meant to cross the trust boundary verbatim
	// to a possibly-unauthenticated downstream caller.
	Message string
	// Status is Type mapped onto the equivalent HTTP status, so one retry policy
	// covers both the status path and this one.
	Status int
}

// Error formats the frame as "planner: in-band HTTP <status> (<type>): <message>", keeping
// it distinguishable in an operator log from the status path's "planner: HTTP <status>".
func (e *UpstreamInBandError) Error() string {
	t := e.Type
	if t == "" {
		t = "unlabeled"
	}
	return fmt.Sprintf("planner: in-band HTTP %d (%s): %s", e.Status, t, e.Message)
}

// Retryable reports whether re-sending can plausibly clear this in-band refusal, under the
// SAME closed policy the status path uses (retryableStatus): an overload/throttle/transient
// yes, a request error no.
func (e *UpstreamInBandError) Retryable() bool { return retryableStatus(e.Status) }

// StatusError converts the frame into the UpstreamStatusError the rest of the stack already
// knows how to surface — so an exhausted in-band overload reaches the client as the honest
// 529 + overloaded_error rather than the opaque 502 + server_error it used to be flattened
// into. RetryAfter is empty by construction: an in-band frame carries no response header.
func (e *UpstreamInBandError) StatusError() *UpstreamStatusError {
	return &UpstreamStatusError{Status: e.Status, Body: truncate([]byte(e.Message), 400)}
}

// anthropicInBandStatus maps an Anthropic error `type` onto the HTTP status the same
// condition would carry had the upstream refused before the stream opened, so the single
// retryableStatus policy covers both paths.
//
// An UNRECOGNIZED type maps to 500 — i.e. RETRYABLE. That direction is deliberate and
// matches the 403 arm's reasoning in retry.go: an in-band refusal arrives on a request the
// upstream already ACCEPTED (a malformed request is rejected with a real 4xx before the
// stream ever opens), so an unlabeled in-band failure is far more likely a capacity flap
// than a permanent error, and the bounded attempt/time budget still ends the turn promptly
// if it is not.
func anthropicInBandStatus(errType string) int {
	switch strings.ToLower(strings.TrimSpace(errType)) {
	case "invalid_request_error":
		return http.StatusBadRequest // 400
	case "authentication_error":
		return http.StatusUnauthorized // 401
	case "billing_error":
		return http.StatusPaymentRequired // 402
	case "permission_error":
		return http.StatusForbidden // 403
	case "not_found_error":
		return http.StatusNotFound // 404
	case "request_too_large":
		return http.StatusRequestEntityTooLarge // 413
	case "timeout_error":
		return http.StatusRequestTimeout // 408, retryable
	case "rate_limit_error":
		return http.StatusTooManyRequests // 429, retryable
	case "overloaded_error":
		return statusOverloaded // 529, retryable
	case "api_error":
		return http.StatusInternalServerError // 500, retryable
	}
	return http.StatusInternalServerError // unlabeled: treat as a transient the loop may clear
}

// NewAnthropicInBandError parses an Anthropic SSE `error` frame's raw `data:` payload into
// the typed verdict a StreamAnthropicRaw onEvent callback returns when the frame arrived
// before anything was relayed to the client. The frame shape is
// {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}; a payload
// that does not parse (or carries no inner type) still yields a non-nil error with the
// unlabeled 500 mapping, so a malformed frame is retried-then-surfaced rather than silently
// treated as a served turn.
func NewAnthropicInBandError(data []byte) *UpstreamInBandError {
	var frame struct {
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	_ = json.Unmarshal(data, &frame) // a malformed frame keeps the zero values (unlabeled)
	return &UpstreamInBandError{
		Type:    frame.Error.Type,
		Message: frame.Error.Message,
		Status:  anthropicInBandStatus(frame.Error.Type),
	}
}
