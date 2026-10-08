package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// upstreamErrorStatus maps a planner error to the HTTP status, an OpenAI-style
// error `code`, and a client-facing message the proxy should return. An
// *agent.UpstreamUnreachableError (a deterministic dial failure — refused / DNS
// NXDOMAIN / TLS) becomes a 502 with the distinct code "upstream_unreachable" so a
// client can tell a misconfigured --base-url apart from a 5xx or a parse failure,
// instead of the opaque code:null "upstream model error" (#346). An
// *agent.UpstreamStatusError carries the upstream provider's OWN status: a 4xx (a
// request error the client can act on — an unknown model 404, a malformed argument
// 400) is SURFACED to the client with that same status, so it is no longer masked
// as a misleading 200 or a generic 502 (#82); a 5xx (the upstream itself failed)
// becomes a 502 Bad Gateway. Any other planner error (transient transport failure,
// response parse error) is also a 502. The provider's raw body / underlying dial
// detail is NEVER forwarded — only the status + classification cross the boundary —
// so an upstream error message cannot leak to a possibly-unauthenticated caller.
func upstreamErrorStatus(err error) (status int, code, msg string) {
	if status, code, msg, ok := admissionErrorStatus(err); ok {
		return status, code, msg
	}
	if recurrentEvictUnsupported(err) {
		return http.StatusConflict, "in_kernel_recurrent_evict_unsupported",
			"context budget exhausted but this Gated-DeltaNet recurrent cache cannot evict in place; retry in a fresh session or start fak serve with --reset-on-budget"
	}
	var contextErr *agent.InKernelContextLengthError
	if errors.As(err, &contextErr) {
		return http.StatusBadRequest, "context_length_exceeded",
			fmt.Sprintf("in-kernel request exceeds the context window (prompt_tokens=%d, max_tokens=%d, context_window=%d); reduce the prompt or max_tokens",
				contextErr.PromptTokens, contextErr.MaxNewTokens, contextErr.MaxContext)
	}
	// An in-kernel device-allocation failure (e.g. the model decode OOM'd on a small GPU under
	// a large prompt) is a LOCAL resource exhaustion the caller can act on, not an upstream
	// failure. It is in-kernel by construction (only the in-kernel planner produces it), so the
	// specific, actionable message is safe and reachable only on a genuine local OOM — a real
	// upstream error can never be this type. 503 (retryable with a smaller request) over 502.
	var oom *agent.InKernelOOMError
	if errors.As(err, &oom) {
		class := strings.TrimSpace(string(oom.Class))
		if class == "" || class == "unknown" {
			class = "device"
		}
		class = strings.ReplaceAll(class, "_", " ")
		return http.StatusServiceUnavailable, "in_kernel_oom",
			fmt.Sprintf("in-kernel GPU out of memory for this request (%s allocation of %d bytes failed); "+
				"reduce the prompt/context size or max_tokens, or serve a smaller model / shorter --ctx", class, oom.Bytes)
	}
	// A native Metal command-buffer wait that exceeded its bound is a LOCAL stall the caller can
	// retry — a transiently-busy or wedged GPU, not an upstream fault. It is in-kernel by
	// construction (only the native planner produces it), so the specific operation + wait detail
	// is safe and reachable only on a genuine stall — a real upstream error can never be this
	// type. 503 (retryable) over 502, matching the in-kernel OOM arm's reasoning.
	var stall metalgemm.MetalCommandBufferStallError
	if errors.As(err, &stall) {
		return http.StatusServiceUnavailable, "metal_command_buffer_stalled",
			fmt.Sprintf("Metal command buffer stall during %s: waited %.3fms at/over %.3fms limit; "+
				"retry the request — if it persists, reduce the batch/prompt or restart the GPU server",
				stall.Operation, stall.WaitedMilliseconds, stall.LimitMilliseconds)
	}
	var capErr *agent.InKernelCapacityError
	if errors.As(err, &capErr) {
		if errors.Is(capErr, agent.ErrInKernelRuntimeExtrasUnknown) {
			// Not a byte-budget verdict: Want/Avail are zero. Surface the agent's message, which
			// names the runtime-extras-unknown site and the missing bound, instead of rendering
			// an ambiguous "plan needs 0 bytes; available budget is 0 bytes".
			return http.StatusServiceUnavailable, "in_kernel_oom",
				capErr.Error() + "; the engine could not bound its runtime-extras memory for this request"
		}
		class := strings.TrimSpace(string(capErr.Class))
		if class == "" || class == "unknown" {
			class = "device"
		}
		class = strings.ReplaceAll(class, "_", " ")
		scope := strings.TrimSpace(string(capErr.Scope))
		if scope == "" {
			scope = "device"
		}
		subject := "GPU"
		if scope == "host" {
			subject = "host memory"
		}
		return http.StatusServiceUnavailable, "in_kernel_oom",
			fmt.Sprintf("in-kernel %s capacity precheck refused this request (%s %s plan needs %d bytes; available budget is %d bytes); "+
				"reduce the prompt/context size or max_tokens, or serve a smaller model / shorter --ctx", subject, scope, class, capErr.Want, capErr.Avail)
	}
	var ue *agent.UpstreamUnreachableError
	if errors.As(err, &ue) {
		return http.StatusBadGateway, "upstream_unreachable",
			"upstream unreachable — check that --base-url points at a running server"
	}
	// An upstream that opened the stream then went SILENT mid-turn (the idle-deadline trip in
	// the streaming planner). This is a timeout, not a generic upstream failure: 504 Gateway
	// Timeout with a distinct code so a client/harness can tell a stalled provider apart from a
	// 4xx request error or a parse failure — instead of the opaque code:null "upstream model
	// error". Placed before the status/fallthrough cases for the same reason as unreachable.
	var stalled *agent.UpstreamStalledError
	if errors.As(err, &stalled) {
		if agent.IsFirstTokenStall(err) {
			return http.StatusGatewayTimeout, "upstream_stalled",
				"upstream stalled — no first token within the first-token watchdog window (long prefill or wedged model); the request was terminated instead of hanging"
		}
		return http.StatusGatewayTimeout, "upstream_stalled",
			"upstream stalled — the model or provider opened the stream then went silent within the idle window"
	}
	// A 429/5xx that fak's retry loop DELIBERATELY stopped at the in-handler ceiling (#2258):
	// the provider-named wait was longer than a wrapped client can hold open, so fak surfaced
	// the truth NOW instead of sleeping past the client's own request timeout. This unwraps to
	// a *UpstreamStatusError, so it MUST precede the generic `se` arm below — otherwise a
	// ceiling bail wears the costume of a plain throttle and the message tells the model to
	// "back off and retry," the one thing that cannot work here: an immediate in-handler retry
	// hits the same wall, and the wait exceeds what the client can absorb. That misdirection is
	// exactly the "if the model is confused it should be able to query and recover" gap — the
	// generic 429 wording steers a confused agent into a futile retry loop instead of recovery.
	//
	// So give it a distinct code and a message aimed at the wrapped MODEL, not just an operator:
	// stop retrying this turn, the condition will NOT self-heal in-handler, and the recovery is
	// to PARK/RESUME (a supervisor — `fak guard`, #2256 — carries it across the reset) or to
	// re-issue the whole turn AFTER the named wait, not to hammer it now. The status stays the
	// real upstream status and the Retry-After still rides downstream (writeUpstreamErr), so a
	// harness that DOES honor Retry-After backs off correctly; the code+message are the
	// machine- and human-branchable signal that this was fak's own ceiling stop, not a bare
	// provider throttle. The announced wait and ceiling are already on the operator-only
	// /debug/vars FAILED line (debugErrorDetail, relayed=true); here we name the wait inline so
	// the model reading the wire knows how long the real reset is.
	var rc *agent.RetryCeilingError
	if errors.As(err, &rc) {
		status := http.StatusTooManyRequests
		if rc.Cause != nil && rc.Cause.Status != 0 {
			status = rc.Cause.Status
		}
		return status, "upstream_retry_ceiling",
			fmt.Sprintf("upstream is rate-limited/overloaded (HTTP %d) and fak already retried: the provider-named wait (~%s) is LONGER than this client can hold a request open, so fak surfaced the truth now instead of sleeping past your timeout. Do NOT retry this turn immediately — an in-handler retry hits the same wall. Recover instead: let a supervisor park and resume it across the reset (fak guard, which carries the turn), or re-issue the whole turn only AFTER the wait named in the Retry-After response header. This will not self-heal by hammering.",
				status, rc.Wait.Round(time.Second))
	}
	var se *agent.UpstreamStatusError
	if errors.As(err, &se) {
		// A 4xx is a REQUEST error the client can act on. Until now every 4xx collapsed
		// into one opaque code:null "upstream rejected the request (HTTP n)" — a 401
		// (bad credential), a 403 (login/org/permission denied), a 429 (rate limit), a
		// 413 (too large), and a 404 (unknown model) were indistinguishable except by the
		// bare number, even though each calls for a DIFFERENT fix. Split them into
		// distinct, actionable OpenAI-style codes + messages so the wrapped agent — and
		// an operator reading the wire — sees WHICH 4xx hit and what to do, not just "not
		// 200". The upstream status passes straight through in every arm (no remap):
		// remapping 429 in particular would silently break both fak's own backoff
		// (chat.go retryableStatus keys on the literal 429) and any downstream client's.
		// Every message is built from se.Status + fixed literals ONLY — the upstream's
		// raw Body (and err.Error(), which embeds it) NEVER crosses the trust boundary to
		// a possibly-unauthenticated downstream caller (#82/#346 invariant). The `type`
		// the client sees comes from errType(status) at the writeErrCode site (401/403 ->
		// authentication_error, 429 -> rate_limit_error), so these code strings are the
		// machine-branchable companion to that human-facing type.
		if se.Status >= 400 && se.Status < 500 {
			return upstream4xxStatus(se)
		}
		// A 529 is Anthropic's non-standard "Overloaded": the PROVIDER is over capacity,
		// which is a different failure from a 500 crash AND from a 429 rate limit. Its
		// recovery is the OPPOSITE of a 429's: a 429 carries a trustworthy Retry-After to
		// honor, whereas a 529 has no retry-after a client can trust and wants exponential
		// backoff + jitter (and, past a couple tries, provider failover). Surfacing it as a
		// generic 502 "upstream model error" (the fallthrough below) flattens it into a
		// crash and strips the client's ability to apply the right posture — so give it a
		// distinct status + code. The matching `type` (overloaded_error) comes from
		// errType(529) at the write site. The Retry-After echo (if the provider sent one)
		// still happens at writeUpstreamErr, harmlessly.
		if se.Status == 529 { // statusOverloaded (agent.statusOverloaded is unexported)
			return 529, "upstream_overloaded",
				"upstream is overloaded (HTTP 529) — the provider is over capacity (not a rate limit and not a crash); back off with exponential jitter, and consider failing over"
		}
		// A 503 the upstream tagged with a Retry-After is an OVERLOAD the client should
		// back off on, not a generic gateway fault — surface the real 503 (and the
		// Retry-After echo happens at the write site) so a wrapped agent waits instead of
		// hammering. Every other 5xx (except the 529 above) stays the opaque 502 below: the
		// upstream itself failed in a way the client cannot time. code stays "" (the
		// historical code:null shape) for both so the 5xx envelope is byte-identical apart
		// from the genuinely-overloaded 503/529.
		if se.Status == http.StatusServiceUnavailable && se.RetryAfter != "" {
			return http.StatusServiceUnavailable, "",
				"upstream temporarily unavailable (HTTP 503) — back off and retry (see the Retry-After response header)"
		}
	}
	return http.StatusBadGateway, "", "upstream model error"
}

func (s *Server) plannerErrorStatus(err error) (status int, code, msg string) {
	if s != nil && s.metrics != nil {
		if _, _, _, ok := admissionErrorStatus(err); !ok {
			s.metrics.observeInKernelOOM(err)
			s.metrics.observeUpstreamError(err)
			// A PERSISTENT 403 (one that survived the agent's bounded transient-recovery arm):
			// snapshot its scrubbed body to the operator-only /debug/vars drilldown so the reason
			// for the denial — org-disabled vs model-not-permitted vs abuse gate — is not lost.
			// The body never crosses to the downstream client (upstreamErrorStatus builds the
			// client message from fixed literals only); this is the operator's private copy.
			var se *agent.UpstreamStatusError
			if errors.As(err, &se) && se.Status == http.StatusForbidden {
				s.metrics.recordForbiddenDetail(se.Body)
			}
		}
	}
	status, code, msg = upstreamErrorStatus(err)
	// On the TRUSTED LOCAL path (fak guard, loopback-bound — s.exposeUpstreamErrorDetail),
	// fold the upstream's OWN 400 detail into the message so the wrapped agent sees WHICH
	// field it got wrong and can self-correct, instead of the generic "check the model name,
	// roles, and parameter ranges". This is the one place the #82/#346 no-leak boundary is
	// relaxed, and only here: the caller is the trusted child, not a possibly-unauthenticated
	// remote. The detail is scrubbed (secret-shaped runs redacted) and bounded by the same
	// scrubForbiddenDetail used for the operator-only 403 drilldown, so a credential an
	// upstream echoed into its body can never ride out. Off (the default) — and every
	// externally-exposed serve — keeps the generic string byte-for-byte. Scoped to 400 (the
	// reported case); 401/403 stay generic (see follow-up note on ExposeUpstreamErrorDetail).
	if s != nil && s.exposeUpstreamErrorDetail && status == http.StatusBadRequest {
		var se *agent.UpstreamStatusError
		if errors.As(err, &se) {
			if detail := scrubForbiddenDetail(se.Body); detail != "" {
				msg = msg + " — upstream said: " + detail
				if s.upstreamBadRequestNotify != nil {
					s.upstreamBadRequestNotify(detail)
				}
			}
		}
	}
	// Name WHICH account/seat hit an account-scoped block (a 403 wall, a 429 ceiling, a
	// usage cap) so the operator/wrapped agent reading the message that STOPPED the turn
	// knows which seat to switch off or wait on — the roster on /debug/vars shows the
	// active seat, but the blocking message never did, so a multi-account session could not
	// tell which of its seats got walled. Gated on the SAME trusted-local path as the 400
	// fold (fak guard, loopback-bound): the seat name is display metadata, but on an
	// externally-exposed serve the caller may be untrusted, so it stays off there — and it
	// is empty anyway on a plain serve, which wires no endpoints provider. Purely additive
	// (the generic message is preserved as a prefix), and only for the account-scoped codes,
	// so a request-shaped error (bad model, too large) is never dressed up as an account
	// problem. Reads the live pull provider, so a mid-run failover names the CURRENT seat.
	if s != nil && s.exposeUpstreamErrorDetail && isAccountBlockCode(code) {
		if seat := s.activeAccountLabel(); seat != "" {
			msg = msg + " " + seat
		}
	}
	return status, code, msg
}

// writeUpstreamErr is the single buffered-error fold for the proxy/planner paths:
// it classifies the upstream failure (observing the metric + mapping it to an
// HTTP status/code/message via plannerErrorStatus), echoes the upstream's
// Retry-After header downstream when the failure carried one, then writes the
// OpenAI-style error envelope. Centralizing it here is what lets EVERY served
// wire — chat, completions, responses, gemini, both Anthropic messages paths,
// and the streaming proxy — surface the same distinct codes AND the same
// Retry-After signal without each handler re-deriving it. The Retry-After value
// is the upstream's header VERBATIM; fak never parses it, so a malformed provider
// value can never reach fak's control flow — it only ever becomes a response
// header (or, absent, a clean no-op). It must be set BEFORE writeErrCode, which
// calls w.WriteHeader and freezes the header block.
func (s *Server) writeUpstreamErr(w http.ResponseWriter, err error) {
	status, code, msg := s.plannerErrorStatus(err)
	if ra := upstreamRetryAfter(err); ra != "" {
		w.Header().Set("Retry-After", ra)
	}
	writeErrCodeFields(w, status, code, msg, s.upstreamErrorFields(err, code))
}

// upstreamRetryAfter returns the upstream's Retry-After header from a
// *agent.UpstreamStatusError, or "" for any other error (or an absent header).
// It is the only place the gateway reaches into the upstream error for that one
// safe-to-forward field; everything else about the upstream error stays off the
// wire.
func upstreamRetryAfter(err error) string {
	var se *agent.UpstreamStatusError
	if errors.As(err, &se) {
		return se.RetryAfter
	}
	return ""
}

// decodeJSON reads a bounded body and decodes JSON. It does NOT reject unknown
// fields — drop-in OpenAI compatibility requires ignoring extra fields — but the
// DTOs have no Ref field, so a client cannot smuggle a kernel CAS handle.
func decodeJSON(w http.ResponseWriter, r *http.Request, v any) error {
	// The chat-completions passthrough carries the same resumed transcript as the
	// Anthropic wire, so it gets the larger transcript cap, not the tool-args cap.
	r.Body = http.MaxBytesReader(w, r.Body, maxTranscriptBody)
	return json.NewDecoder(r.Body).Decode(v)
}

// decodeRequestBody decodes the request body into v via decodeJSON. On a malformed
// body it writes the standard 400 ("malformed request body: ...") and returns false
// so the caller returns immediately; on success it returns true. It centralizes the
// decode-or-400 block repeated across every JSON handler entry (parallel to
// requireMethod). Handlers that need a different error phrasing keep their own block.
func decodeRequestBody(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := decodeJSON(w, r, v); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return false
	}
	return true
}

// rejectInvalidSampling writes the sampling-validation 400 and returns true when a
// validator reports a problem (a non-empty msg), so the caller returns immediately;
// an empty msg returns false to continue. It folds the "validator message → 400"
// block the chat, completions, and responses sampling wires share (#326).
func rejectInvalidSampling(w http.ResponseWriter, msg string) bool {
	if msg != "" {
		writeErr(w, http.StatusBadRequest, msg)
		return true
	}
	return false
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeErr emits an OpenAI-style error envelope, which both the fak-native and
// OpenAI-compatible clients understand. The error `type` is derived from the
// status class so a client that branches on it (retry server_error, not
// invalid_request_error) classifies a transient 502 correctly.
func writeErr(w http.ResponseWriter, status int, msg string) {
	writeErrCode(w, status, "", msg)
}

// requireMethod enforces a single allowed HTTP method on a handler entry. On a
// mismatch it writes the standard 405 ("use <METHOD>") and returns false so the
// caller returns immediately; on a match it returns true.
func requireMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r.Method != method {
		writeErr(w, http.StatusMethodNotAllowed, "use "+method)
		return false
	}
	return true
}

// writeErrCode is writeErr with an explicit OpenAI-style error `code`. An empty
// code keeps the historical code:null shape; a non-empty code (e.g.
// "upstream_unreachable") lets a client branch on the specific failure class
// rather than guessing from the message text (#346).
func writeErrCode(w http.ResponseWriter, status int, code, msg string) {
	writeErrCodeFields(w, status, code, msg, nil)
}

// writeErrCodeFields is writeErrCode plus typed extra fields on the error object (e.g.
// context_window on a context_length_exceeded). The four standard keys always win.
func writeErrCodeFields(w http.ResponseWriter, status int, code, msg string, fields map[string]any) {
	writeJSON(w, status, map[string]any{"error": errObject(status, code, msg, fields)})
}

func errObject(status int, code, msg string, fields map[string]any) map[string]any {
	var codeVal any
	if code != "" {
		codeVal = code
	}
	obj := make(map[string]any, 4+len(fields))
	for k, v := range fields {
		obj[k] = v
	}
	obj["message"], obj["type"], obj["code"], obj["param"] = msg, errType(status), codeVal, nil
	return obj
}

func errType(status int) string {
	switch {
	case status == http.StatusBadRequest:
		return "invalid_request_error"
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		return "authentication_error"
	case status == http.StatusTooManyRequests:
		// 429 gets the OpenAI/Anthropic-standard rate_limit_error type so a client that
		// branches on `type` (back off on a rate limit, don't on a malformed request)
		// classifies it correctly — without this it fell through to invalid_request_error
		// and was indistinguishable from a 400.
		return "rate_limit_error"
	case status == 529:
		// 529 is Anthropic's "Overloaded": a PROVIDER-capacity failure, distinct from a 500
		// crash. Give it the Anthropic-standard overloaded_error type so a client can apply
		// backoff+jitter (a 529 has no trustworthy Retry-After) instead of treating it like
		// a generic server_error. Must precede the status>=500 arm.
		return "overloaded_error"
	case status >= 500:
		return "server_error"
	default:
		return "invalid_request_error"
	}
}
