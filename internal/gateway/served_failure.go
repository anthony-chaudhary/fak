package gateway

import (
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

// This file is the #2336 serving-honesty seam: a native served model that
// panics on every completion turn must (a) answer each request with the
// structured OpenAI-style 5xx envelope instead of a bare body or a reset
// connection, and (b) stop /healthz from reporting an unqualified ok:true
// while the failure is recent — a green liveness probe over crashing
// completions keeps watchdogs routing work to a broken native serve.

// servedFailureWindow bounds how long a recovered served-turn panic keeps
// /healthz qualified. In-memory and time-bounded only: long enough that a
// polling watchdog cannot miss it, short enough that one aged-out panic does
// not permanently condemn a serve that has recovered. A serve that panics on
// every request (the #2336 GLM-5.2 q3 shape) keeps refreshing the marker and
// stays unhealthy for as long as it is broken.
const servedFailureWindow = 5 * time.Minute

// servedFailure is the most recent recovered panic on a served completion
// route. Zero value ready; guarded by its own mutex so the health read never
// contends with request accounting.
type servedFailure struct {
	mu    sync.Mutex
	at    time.Time
	route string
	msg   string
}

func (f *servedFailure) note(route string, p any, now time.Time) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.at, f.route, f.msg = now, route, fmt.Sprint(p)
}

// recent returns the last served-turn failure while it is inside the honesty
// window; ok=false once it has aged out (or never happened).
func (f *servedFailure) recent(now time.Time) (route, msg string, age time.Duration, ok bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.at.IsZero() || now.Sub(f.at) > servedFailureWindow {
		return "", "", 0, false
	}
	return f.route, f.msg, now.Sub(f.at), true
}

// servedTurnPath reports whether path is a served model-turn surface — a
// route whose handler runs the configured planner/model for a completion. A
// panic here means "serving is broken" and must reach /healthz; a
// control-plane handler panic must not condemn serving health.
// /v1/messages/count_tokens is deliberately absent (tokenizer only, no turn).
func servedTurnPath(path string) bool {
	switch path {
	case "/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages":
		return true
	}
	// Native Gemini generateContent surface (/v1beta/models/{model}:{method}).
	return strings.HasPrefix(path, "/v1beta/")
}

// noteServedPanic records a recovered handler panic into the health-honesty
// marker when (and only when) it happened on a served turn.
func (s *Server) noteServedPanic(path string, p any) {
	if servedTurnPath(path) {
		s.servedFailure.note(path, p, time.Now())
	}
}

// writeRecoveredPanicErr converts a recovered handler panic into the
// structured OpenAI-style 5xx envelope (server_error), so a client or
// watchdog can classify the failure instead of parsing a bare text body. The
// panic value is code-authored diagnostic text (e.g. "glm_moe_dsa attention
// step failed"), not operator-private data, so surfacing it is safe and is
// what makes the error actionable.
func writeRecoveredPanicErr(w http.ResponseWriter, p any) {
	writeErrCode(w, http.StatusInternalServerError, "handler_panic",
		fmt.Sprintf("internal error: recovered handler panic: %v", p))
}
