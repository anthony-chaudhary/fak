package gateway

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

// maxBody bounds an inbound request body (defense against an unbounded read from
// an untrusted client). 4 MiB is far above any real tool-args / chat payload.
const maxBody = 4 << 20

// gatewayRoute pairs a ServeMux registration pattern with its handler. Handler
// builds the mux from routeTable() rather than a sequence of inline HandleFunc
// calls so that the served HTTP surface has a single, enumerable source of
// truth — which the OpenAPI spec drift gate (openapi_spec_test.go) ranges over
// to assert docs/fak/openapi.yaml documents every route (#205, F-007: the spec
// the client SDKs are generated from must not drift behind the served surface).
type gatewayRoute struct {
	pattern string
	handler http.HandlerFunc
}

// routeTable is the canonical, ordered list of the gateway's HTTP routes — the
// single source of truth Handler registers and the OpenAPI spec test verifies
// against. ServeMux dispatch is by pattern specificity, not registration order,
// so building the mux from this slice is behavior-identical to inline
// registration.
func (s *Server) routeTable() []gatewayRoute {
	return []gatewayRoute{
		// OpenAI-compatible surface.
		{"/v1/chat/completions", s.handleChatCompletions},
		{"/v1/embeddings", s.handleEmbeddings},
		{"/v1/moderations", s.handleModerations},
		// Anthropic Messages surface.
		{"/v1/messages", s.handleAnthropicMessages},
		{"/v1/messages/count_tokens", s.handleAnthropicCountTokens},
		// Native Gemini generateContent surface (/v1beta/models/{model}:{method}).
		{"/v1beta/", s.handleGeminiGenerateContent},
		// fak-native surface — one POST, one verdict.
		{"/v1/fak/syscall", s.handleFakSyscall},
		{"/v1/fak/adjudicate", s.handleFakAdjudicate},
		{"/v1/fak/admit", s.handleFakAdmit},
		{"/v1/fak/changes", s.handleFakChanges},
		{"/v1/fak/events", s.handleFakEvents},
		{"/v1/fak/revoke", s.handleFakRevoke},
		{"/v1/fak/context/change", s.handleFakContextChange},
		{"/v1/fak/policy/reload", s.handleFakPolicyReload},
		{"/v1/fak/trace/reset", s.handleFakTraceReset},
		{"/v1/fak/trace/", s.handleFakTraceObserve},
		// /v1/fak/session/ is the DRIVE-state control surface: GET /v1/fak/session/{id}
		// observes one session's run-state/budget/priority/pace; POST
		// /v1/fak/session/{id}/{verb} applies a control verb (run|budget|pace|priority).
		// One subtree handler dispatches on method + the trailing path segments.
		{"/v1/fak/session/", s.handleFakSession},
		// /v1/fak/sessions (no trailing slash) is the MULTI-session read: a snapshot of
		// every live session's drive state. Registered distinctly from the singular
		// /v1/fak/session/ subtree, so a single-id request never lands here.
		{"/v1/fak/sessions", s.handleFakSessions},
		// /v1/fak/tasks is the read-only process task-manager snapshot. Inert (404)
		// unless a host installs a provider via SetTasksSnapshotProvider and the
		// operator enables it; the snapshot carries accounting only, no payload bytes.
		{"/v1/fak/tasks", s.handleFakTasks},
		{"/v1/models", s.handleModels},
		// MCP-over-HTTP, operational endpoints.
		{"/mcp", s.handleMCPHTTP},
		{"/healthz", s.handleHealth},
		{"/metrics", s.handleMetrics},
		{"/debug/vars", s.handleDebugVars},
	}
}

// Handler builds the gateway's HTTP routes (routeTable) wrapped in the metrics
// and optional bearer-auth middleware. Routes: the OpenAI-compatible surface
// (/v1/chat/completions, /v1/embeddings, /v1/moderations, /v1/models), the
// Anthropic Messages and native Gemini surfaces, the fak-native
// syscall/adjudicate JSON endpoints, policy reload, Prometheus metrics
// (/metrics), expvar-style diagnostics (/debug/vars), MCP-over-HTTP (/mcp), and
// an unauthenticated health check (/healthz).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	for _, rt := range s.routeTable() {
		mux.HandleFunc(rt.pattern, rt.handler)
	}
	return s.withMetrics(s.withAuth(mux))
}

// ListenAndServe binds the HTTP surface on addr, then serves it via Serve until
// ctx is done. It warns loudly if a no-auth gateway is bound beyond loopback. The
// bind is SYNCHRONOUS (not via hs.ListenAndServe in a goroutine) for three reasons:
// (1) the bind duration is measured as the "listener-bind" boot phase so the
// dashboard can show it; (2) a bind error (addr in use, permission denied) surfaces
// and fails BEFORE MarkReady closes the timeline, rather than racing the ready mark
// and lying about readiness; (3) Serve then runs against the already-bound listener.
func (s *Server) ListenAndServe(ctx context.Context, addr string) error {
	if s.requireKey == "" && !loopbackOnly(addr) {
		s.logf("WARNING: binding %s with NO --require-key set — the kernel gateway is exposed without authentication", addr)
	}
	tBind := time.Now()
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	s.startup.phase("listener-bind", time.Since(tBind))
	return s.Serve(ctx, ln)
}

// Serve runs the gateway HTTP surface on an already-bound listener until ctx is
// done, then drains gracefully within a bounded shutdown window. ListenAndServe is
// Serve over a freshly bound socket; a caller that needs the chosen port up front
// — a test binding 127.0.0.1:0, or a host handing fak a pre-opened socket — binds
// its own listener and calls Serve directly. It mirrors net/http.Server's
// ListenAndServe/Serve split.
func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	// Bounded timeouts so a single slow/idle connection cannot pin a goroutine +
	// socket indefinitely (slow-loris-on-body / idle-keepalive DoS). ReadTimeout
	// also caps body-delivery TIME (MaxBytesReader only caps SIZE).
	//
	// WriteTimeout bounds the WHOLE handler, and a live upstream model round-trip
	// rides it — so a SLOW LOCAL backend (a multi-thousand-token prefill on a CPU
	// model can take minutes) needs a far higher ceiling than a hosted API. The
	// default stays conservative for a network-exposed deployment; FAK_HTTP_*_TIMEOUT_S
	// raises (or, with 0, disables) it for local dogfood serving. The dogfood
	// launchers set FAK_HTTP_WRITE_TIMEOUT_S generously.
	hs := &http.Server{
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       durEnv("FAK_HTTP_READ_TIMEOUT_S", 30*time.Second),
		WriteTimeout:      durEnv("FAK_HTTP_WRITE_TIMEOUT_S", 90*time.Second),
		IdleTimeout:       durEnv("FAK_HTTP_IDLE_TIMEOUT_S", 120*time.Second),
	}
	// Disable Nagle on accepted TCP connections. Without TCP_NODELAY the kernel
	// coalesces small writes (Nagle), adding 40-200ms of buffering on a high-RTT
	// link — felt on streamed chat-completion deltas and the small fak-native verdict
	// replies. nodelayListener sets NoDelay(true) on every accepted *net.TCPConn; it
	// wraps the listener here so BOTH entry points get it (ListenAndServe's freshly
	// bound socket AND a Serve caller that handed us its own listener). A non-TCP
	// listener (e.g. a test net.Pipe) passes through untouched.
	errc := make(chan error, 1)
	go func() { errc <- hs.Serve(nodelayListener(ln)) }()
	// The boot timeline closes here: the listener is bound and the gateway is
	// ready to adjudicate. Any eager model load the host did (fak serve --gguf) has
	// already completed before this point, so time-to-ready spans it.
	s.MarkReady()
	s.logf("fak gateway listening on http://%s  (engine=%s model=%s vdso=%v auth=%v)",
		ln.Addr(), s.engineID, s.model, s.k.VDSOEnabled(), s.requireKey != "")
	select {
	case <-ctx.Done():
		shctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return hs.Shutdown(shctx)
	case err := <-errc:
		return err
	}
}

// nodelayListener wraps ln so every accepted *net.TCPConn has Nagle disabled
// (TCP_NODELAY). It is a pass-through for a listener whose Accept does not yield a
// *net.TCPConn — a test's in-memory pipe or a Unix socket — so wrapping is always
// safe. Returning the bare net.Listener interface keeps Serve's signature unchanged.
func nodelayListener(ln net.Listener) net.Listener {
	return &noDelayTCPListener{Listener: ln}
}

type noDelayTCPListener struct {
	net.Listener
}

// Accept returns the next connection from the wrapped listener with Nagle disabled (TCP_NODELAY) on any *net.TCPConn, best-effort.
func (l *noDelayTCPListener) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return c, err
	}
	if tc, ok := c.(*net.TCPConn); ok {
		// Best-effort: a SetNoDelay failure (already-closed conn) is not fatal to the
		// connection — let the handler proceed and surface any real error on use.
		_ = tc.SetNoDelay(true)
	}
	return c, nil
}

// withAuth enforces the configured secret on every route except /healthz when
// RequireKey is set. With no key configured it is a pass-through (drop-in, loopback
// default). The comparison is constant-time over SHA-256 digests so the reject
// latency leaks neither the secret's bytes nor its length — this is the gateway's
// only auth primitive on a network-reachable security kernel.
func (s *Server) withAuth(next http.Handler) http.Handler {
	want := sha256.Sum256([]byte(s.requireKey))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.requireKey != "" && r.URL.Path != "/healthz" {
			tok, ok := gatewayCredential(r)
			got := sha256.Sum256([]byte(tok))
			if !ok || subtle.ConstantTimeCompare(got[:], want[:]) != 1 {
				writeErr(w, http.StatusUnauthorized, "missing or invalid credentials")
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// gatewayCredential extracts the presented secret from any of the auth schemes a
// fak gateway fronts. The OpenAI/fak-native surfaces send
// "Authorization: Bearer <tok>"; the native Anthropic surface (/v1/messages) is
// driven by clients — Claude Code, the Anthropic SDKs — that authenticate with the
// "x-api-key: <tok>" header instead; the native Gemini surface
// (/v1beta/models/{model}:generateContent) is driven by clients — Gemini CLI, the
// google-genai SDKs — that authenticate with "x-goog-api-key: <tok>" (or, for raw
// REST, "?key=<tok>"). Accepting all of them is what lets an authenticated
// (non-loopback) gateway serve any native client wire over its base-URL redirect;
// without the matching arm every such client 401s even though the gateway speaks
// its wire. All schemes compare against the same single secret in constant time at
// the call site.
func gatewayCredential(r *http.Request) (string, bool) {
	if tok, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer "); ok {
		return tok, true
	}
	if k := r.Header.Get("X-Api-Key"); k != "" {
		return k, true
	}
	if g := r.Header.Get("X-Goog-Api-Key"); g != "" {
		return g, true
	}
	if q := r.URL.Query().Get("key"); q != "" {
		return q, true
	}
	return "", false
}

// ---------------------------------------------------------------------------
// OpenAI-compatible surface.
// ---------------------------------------------------------------------------

// handleChatCompletions is the adjudication PROXY. It forwards the chat to the
// configured model (upstream HTTPPlanner or the offline mock), then runs each
// PROPOSED tool_call through k.Decide BEFORE the caller sees it: denied calls are
// dropped, grammar-repaired calls have their arguments rewritten to the canonical
// form, and a fak-aware client gets the full per-call adjudication in the `fak`
// extension. It NEVER executes the client's tools — the client does.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req ChatRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	// An empty/missing messages array is a CLIENT error, not an upstream failure.
	// Reject it here with a 400 ("messages: field required") rather than forwarding
	// a degenerate request and surfacing the upstream's own 400 as a confusing 502
	// gateway error (#82). This is the same well-formedness floor a real provider
	// applies, applied before we spend an upstream round-trip on it.
	if len(req.Messages) == 0 {
		writeErr(w, http.StatusBadRequest, "messages: field required")
		return
	}
	// Validate the sampling params on ingress (#326). A negative max_tokens or an
	// out-of-range temperature/top_p is a CLIENT error — reject it here with a 400
	// rather than forwarding bad input that the upstream silently answers anyway (a
	// wire-contract deviation the proxy used to swallow). Same well-formedness floor
	// as the empty-messages check above, applied before an upstream round-trip is spent.
	if msg := validateSampling(req); msg != "" {
		writeErr(w, http.StatusBadRequest, msg)
		return
	}
	ctx := r.Context()
	// Request-model pass-through (#82): forward the client's requested model to the
	// upstream verbatim, falling back to the gateway's configured model only when the
	// client omitted one. This stops the gateway silently serving a DIFFERENT model
	// than the client asked for — an unknown model now reaches the upstream and
	// surfaces its 404 instead of a misleading 200. --model stays the advertised
	// /v1/models id and the default. reqModel is also the response-model fallback
	// when the upstream omits a served-model field.
	reqModel := req.Model
	if reqModel == "" {
		reqModel = s.model
	}

	// Thread one request TraceID across every proposed call in this chat so the IFC
	// ledger, plan-CFI, response header, and access log all correlate. The
	// middleware honors a client-supplied X-Trace-Id or mints one.
	reqTrace := s.useHTTPTrace(w, r, "")
	// Operator control / budget / pace at the served request boundary. With
	// DecideSession wired this mutates the live session table (TurnsLeft debit,
	// budget exhaustion, pace cap); without it the legacy observe-only admission guard
	// still refuses paused/draining/stopped sessions.
	sessionTurn, ok, canceled := s.beginServedSessionTurn(ctx, reqTrace)
	if canceled {
		return
	}
	if !ok {
		// Budget drained: opt-in human-like reset (distill seed, re-arm, continue on
		// the fresh trace) when wired; else the historical 409 directive. Mirrors the
		// Anthropic wire in messages.go.
		if newTrace, seed, reset := s.maybeResetOnBudget(ctx, sessionTurn.state, req.Messages); reset {
			req.Messages = spliceSeed(seed, req.Messages)
			reqTrace = newTrace
			if sessionTurn, ok, canceled = s.beginServedSessionTurn(ctx, reqTrace); canceled {
				return
			}
			if !ok {
				writeSessionRefusal(w, sessionTurn.state)
				return
			}
		} else {
			writeSessionRefusal(w, sessionTurn.state)
			return
		}
	}
	resultAdmissions, err := s.admitInboundResults(ctx, req.Messages, reqTrace)
	if err != nil {
		writeErr(w, http.StatusBadGateway, "upstream cache invalidation failed")
		return
	}

	// True streaming fast path: when the client asked to stream AND the planner can
	// stream this wire, forward the upstream tokens live for a real time-to-first-token
	// instead of synthesizing the SSE from a fully-buffered turn. Tool-bearing requests
	// take this path too: CompleteStream HOLDS every proposed call off-wire for
	// adjudication and the lift-guard keeps a text-form call from leaking into the live
	// content, so the buffered path's trust posture is preserved (see streamChatLive). A
	// non-streaming-wire request falls through to the buffered path below, whose tail
	// still synthesizes a stream for stream=true. streamChatLive returns false having
	// written nothing when it cannot stream, so the fall-through is safe.
	if req.Stream {
		if s.streamChatLive(ctx, w, req, reqModel, reqTrace, sessionTurn, resultAdmissions) {
			return
		}
	}

	// Forward the client's per-request sampling params to the upstream model. Each
	// option is a no-op when its field is absent (max_tokens 0, nil temperature/top_p,
	// empty stop), so an OpenAI client that omits them gets the planner default —
	// identical to the pre-seam behavior — while one asking for a long completion is
	// no longer hard-capped at the planner's 1024-token floor (#62).
	began := time.Now()
	comp, err := s.completeServed(ctx, sessionTurn, req.Messages, req.Tools,
		agent.WithModel(req.Model), // no-op when the client omitted model
		agent.WithMaxTokens(sessionTurn.maxTokensFor(req.MaxTokens)),
		agent.WithTemperature(req.Temperature),
		agent.WithTopP(req.TopP),
		agent.WithStop(normalizeStop(req.Stop)),
		// Structured-output passthrough (#907): forward the client's response_format /
		// logit_bias to the ride engine verbatim so vLLM/SGLang enforce the constraint
		// during generation; the resulting tool candidate still enters adjudication
		// below. Each option is a no-op when its field is absent (bit-exact drop-in).
		agent.WithResponseFormat(req.ResponseFormat),
		agent.WithLogitBias(req.LogitBias),
	)
	if err != nil {
		// Map the upstream failure to an honest status. Log the detail for the operator
		// but return a GENERIC message — the planner error embeds up to 400 bytes of the
		// upstream provider's raw body, which must not cross the trust boundary to a
		// (possibly unauthenticated) downstream caller.
		s.logf("gateway: upstream model error: %v", err)
		status, code, msg := s.plannerErrorStatus(err)
		writeErrCode(w, status, code, msg)
		return
	}

	asst := comp.Message
	asst.Role = agent.RoleAssistant

	// Tool-call conformance: the upstream's finish_reason announced tool calls but
	// NONE survived parsing + the text-lift fallback. Proceeding would skip
	// adjudication on a call the model intended to make — the exact silent-no-op a
	// non-OpenAI-shaped emitter (e.g. a GLM-5.2 variant burying calls in
	// reasoning_content) causes. Fail closed: never let an unparsed tool call cross
	// the gateway as a benign empty turn.
	if comp.ToolCallsDropped && len(asst.ToolCalls) == 0 {
		s.logf("gateway: upstream announced tool_calls but none parsed (conformance fail-closed); model=%s", s.model)
		writeErr(w, http.StatusBadGateway, "upstream tool-call format not recognized; refusing to skip adjudication")
		return
	}

	kept, adjs, dropped := s.adjudicateProposed(ctx, asst.ToolCalls, reqTrace)
	asst.ToolCalls = kept

	finish := comp.FinishReason
	if len(kept) > 0 {
		finish = "tool_calls"
	} else if dropped > 0 {
		// Every proposed call was refused. Give even a fak-unaware client something
		// actionable in-band rather than an empty turn.
		finish = "stop"
		if asst.Content == "" {
			asst.Content = denySummary(adjs)
		}
	}

	// Echo the model the UPSTREAM reported it served (#82); fall back to the client's
	// requested model (or, if it omitted one, the configured model) when the upstream
	// did not name a served model. Never just s.model — that is the silent-substitution
	// this fix removes.
	respModel := comp.Model
	if respModel == "" {
		respModel = reqModel
	}
	s.logInferenceTurn(reqTrace, "openai_chat_completions", req.Stream, comp.Usage, finish, time.Since(began), false)
	resp := ChatResponse{
		ID:      "chatcmpl-fak-" + itoa(uint64(time.Now().UnixNano())),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   respModel,
		Choices: []ChatChoice{{Index: 0, Message: asst, FinishReason: finish}},
		Usage:   comp.Usage,
	}
	redactions := wireRedactionsFrom(comp.PreSendRedactionRecords)
	if len(adjs) > 0 || len(resultAdmissions) > 0 || len(redactions) > 0 {
		resp.Fak = &FakExt{Adjudications: adjs, ResultAdmissions: resultAdmissions, Redactions: redactions}
	}
	if req.Stream {
		writeChatCompletionStream(w, resp)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// validateSampling enforces the OpenAI sampling-param contract on an inbound chat
// request, returning a client-facing 400 message for the first invalid field (or ""
// when every present field is in range). It catches the unambiguous wire-contract
// violations the proxy otherwise forwarded verbatim — a negative max_tokens, a
// temperature outside [0, 2], a top_p outside [0, 1] — so bad client input surfaces
// as a 400 instead of the model silently answering it (#326).
//
// max_tokens == 0 is deliberately NOT rejected. The wire field is an omitempty int,
// so an explicit "max_tokens":0 and an omitted field both decode to Go 0 and are
// indistinguishable here; 0 therefore falls through to the planner default (the
// documented semantics). Only values that cannot be a zero-value default — negatives
// and out-of-band floats — are caught, which keeps the check free of false positives
// on a client that simply omitted a field.
func validateSampling(req ChatRequest) string {
	if req.MaxTokens < 0 {
		return "max_tokens: must be a positive integer"
	}
	if req.Temperature != nil && (*req.Temperature < 0 || *req.Temperature > 2) {
		return "temperature: must be in [0, 2]"
	}
	if req.TopP != nil && (*req.TopP < 0 || *req.TopP > 1) {
		return "top_p: must be in [0, 1]"
	}
	return ""
}

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
	var capErr *agent.InKernelCapacityError
	if errors.As(err, &capErr) {
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
	var se *agent.UpstreamStatusError
	if errors.As(err, &se) && se.Status >= 400 && se.Status < 500 {
		return se.Status, "", fmt.Sprintf("upstream rejected the request (HTTP %d)", se.Status)
	}
	return http.StatusBadGateway, "", "upstream model error"
}

func (s *Server) plannerErrorStatus(err error) (status int, code, msg string) {
	if s != nil && s.metrics != nil {
		s.metrics.observeInKernelOOM(err)
	}
	return upstreamErrorStatus(err)
}

func writeChatCompletionStream(w http.ResponseWriter, resp ChatResponse) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)

	choice := resp.Choices[0]

	chunk := func(d ChatDelta, finish *string, usage *agent.Usage) ChatStreamResponse {
		return ChatStreamResponse{
			ID:      resp.ID,
			Object:  "chat.completion.chunk",
			Created: resp.Created,
			Model:   resp.Model,
			Choices: []ChatStreamChoice{{Index: choice.Index, Delta: d, FinishReason: finish}},
			Usage:   usage,
		}
	}

	// Opening chunk: announce the assistant role and the surviving (adjudicated)
	// tool calls. OpenAI sends the role before any content fragment, so a client
	// that keys on the first delta's role sees it immediately.
	opening := chunk(ChatDelta{
		Role:      choice.Message.Role,
		ToolCalls: streamToolCalls(choice.Message.ToolCalls),
	}, nil, nil)
	if err := writeSSEData(w, opening); err != nil {
		return
	}

	// Content chunks: stream the adjudicated content as incremental fragments, one
	// SSE event per fragment, the way a real OpenAI stream delivers tokens — rather
	// than collapsing the whole reply into a single delta. segmentContent preserves
	// every byte, so concatenating the content deltas reproduces the reply exactly.
	for _, seg := range segmentContent(choice.Message.Content) {
		if err := writeSSEData(w, chunk(ChatDelta{Content: seg}, nil, nil)); err != nil {
			return
		}
	}

	finish := choice.FinishReason
	final := chunk(ChatDelta{}, &finish, &resp.Usage)
	final.Fak = resp.Fak
	if err := writeSSEData(w, final); err != nil {
		return
	}
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// segmentContent splits assistant content into incremental streaming fragments at
// word boundaries (each fragment keeps its trailing space), so concatenating the
// fragments in order reproduces the content byte-for-byte. Empty content yields no
// fragments: a pure tool-call turn streams no content delta, matching OpenAI, which
// emits a content delta only when there is content to deliver.
func segmentContent(content string) []string {
	if content == "" {
		return nil
	}
	segs := strings.SplitAfter(content, " ")
	out := segs[:0]
	for _, s := range segs {
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

func streamToolCalls(calls []agent.ToolCall) []ChatDeltaToolCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]ChatDeltaToolCall, 0, len(calls))
	for i, tc := range calls {
		out = append(out, ChatDeltaToolCall{
			Index:    i,
			ID:       tc.ID,
			Type:     tc.Type,
			Function: tc.Function,
		})
	}
	return out
}

func writeSSEData(w http.ResponseWriter, v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", raw); err != nil {
		return err
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}

// denySummary renders a short human-readable note when every proposed tool_call
// was refused, so a client that ignores the `fak` extension still adapts.
func denySummary(adjs []ToolAdjudication) string {
	parts := make([]string, 0, len(adjs))
	for _, a := range adjs {
		parts = append(parts, fmt.Sprintf("%s: %s (%s/%s)", a.Tool, a.Verdict.Kind, a.Verdict.Reason, a.Verdict.Disposition))
	}
	return "All proposed tool calls were refused by the fak kernel: " + strings.Join(parts, "; ")
}

// adjudicationNote renders a short, agent-readable summary of the kernel's
// non-trivial decisions (drops + repairs) on a turn, for clients that read only
// the in-band content channel and never the `fak` extension — Claude Code on the
// Anthropic wire is exactly that client. It is the difference between a denied
// call SILENTLY VANISHING (the agent re-proposes it, or proceeds on a false
// premise) and the agent being told "fak refused rm -rf for POLICY_BLOCK" so it
// can adapt. Returns "" when every call was a clean ALLOW (nothing worth saying).
func adjudicationNote(adjs []ToolAdjudication) string {
	denied := make([]string, 0, len(adjs))
	repaired := make([]string, 0, len(adjs))
	for _, a := range adjs {
		switch {
		case !a.Admitted:
			denied = append(denied, fmt.Sprintf("%s (%s/%s)", a.Tool, a.Verdict.Reason, a.Verdict.Disposition))
		case a.Verdict.Kind == "TRANSFORM":
			repaired = append(repaired, a.Tool)
		}
	}
	if len(denied) == 0 && len(repaired) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("[fak] ")
	if len(denied) > 0 {
		b.WriteString("refused ")
		b.WriteString(strconv.Itoa(len(denied)))
		b.WriteString(" tool call(s): ")
		b.WriteString(strings.Join(denied, "; "))
		b.WriteString(". Do not re-propose a refused call unchanged; choose an allowed alternative.")
	}
	if len(repaired) > 0 {
		if len(denied) > 0 {
			b.WriteString(" ")
		}
		b.WriteString("repaired arguments for: ")
		b.WriteString(strings.Join(repaired, ", "))
		b.WriteString(".")
	}
	return b.String()
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   []map[string]any{{"id": s.model, "object": "model", "owned_by": "fak"}},
	})
}

// ---------------------------------------------------------------------------
// fak-native surface — the simplest non-Go integration: one POST, one verdict.
// ---------------------------------------------------------------------------

// handleFakSyscall adjudicates AND executes a single tool call through the kernel
// (the self-contained / CI path).
func (s *Server) handleFakSyscall(w http.ResponseWriter, r *http.Request) {
	req, ok := s.decodeSyscall(w, r)
	if !ok {
		return
	}
	ctx := WithPrincipal(r.Context(), principalFor(r, req.Principal))
	wv, env, err := s.syscall(ctx, req.Tool, rawArgs(req.Arguments), req.ReadOnly, req.Witness, req.TraceID)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, SyscallResponse{Verdict: wv, Result: env, TraceID: req.TraceID})
}

// principalFor resolves a request's isolation principal: the X-Fak-Principal header
// (set by an auth proxy / tenant router in front of the gateway) takes precedence, else
// the request body's principal field. Empty => single-tenant (every caller shares).
func principalFor(r *http.Request, bodyPrincipal string) string {
	if h := strings.TrimSpace(r.Header.Get("X-Fak-Principal")); h != "" {
		return h
	}
	return strings.TrimSpace(bodyPrincipal)
}

// handleFakAdmit runs a CLIENT-PRODUCED tool result through the kernel's
// result-side stack (context-MMU quarantine + IFC source-stamp). This is the
// served-path complement to handleFakAdjudicate: adjudicate gates the CALL before
// the client runs it; admit contains the RESULT after. A poisoned result is
// quarantined and the session's taint ledger raised before it is admitted.
func (s *Server) handleFakAdmit(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req AdmitRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	req.TraceID = s.useHTTPTrace(w, r, req.TraceID)
	wv, env, err := s.admit(r.Context(), req.Tool, rawArgs(req.Result), req.Witness, req.TraceID)
	if err != nil {
		// A REMOTE engine-cache reset failure is a gateway/upstream fault — surface it
		// as a 502 (the same fail-closed signal the proxy returns), with a generic
		// message so the upstream error body never crosses the trust boundary. Any
		// other admit error is a client-side 400.
		if errors.Is(err, errEngineCacheReset) {
			s.logf("gateway: native admit engine cache reset failed: %v", err)
			writeErr(w, http.StatusBadGateway, "upstream cache invalidation failed")
			return
		}
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, SyscallResponse{Verdict: wv, Result: env, TraceID: req.TraceID})
}

// handleFakAdjudicate returns the pre-execution verdict only (the production path
// for a client that runs its own tools): no dispatch, no engine, no pending state.
func (s *Server) handleFakAdjudicate(w http.ResponseWriter, r *http.Request) {
	req, ok := s.decodeSyscall(w, r)
	if !ok {
		return
	}
	wv, repaired, err := s.adjudicate(r.Context(), req.Tool, rawArgs(req.Arguments), req.ReadOnly, req.Witness, req.TraceID)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	resp := SyscallResponse{Verdict: wv, TraceID: req.TraceID}
	if repaired != "" {
		resp.RepairedArguments = json.RawMessage(repaired)
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleFakChanges drains the cross-agent "what changed" feed for events after the
// client's ?since= (or {"since":N}) cursor. GET or POST.
func (s *Server) handleFakChanges(w http.ResponseWriter, r *http.Request) {
	var since uint64
	var bodyPrincipal string
	if r.Method == http.MethodPost {
		var req ChangesRequest
		if err := decodeJSON(w, r, &req); err != nil {
			writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
			return
		}
		since = req.Since
		bodyPrincipal = req.Principal
	} else if v := r.URL.Query().Get("since"); v != "" {
		var n uint64
		for _, c := range v {
			if c < '0' || c > '9' {
				writeErr(w, http.StatusBadRequest, "since must be a non-negative integer")
				return
			}
			n = n*10 + uint64(c-'0')
		}
		since = n
	}
	events, cursor := s.changes(principalFor(r, bodyPrincipal), since)
	writeJSON(w, http.StatusOK, ChangesResponse{Events: events, Cursor: cursor})
}

// activeJournal returns the process-global durable audit journal, or nil if
// FAK_AUDIT_JOURNAL was unset at boot. Indirected through a var so a test can
// inject an in-memory journal without process-global env setup.
var activeJournal = journal.Active

// EventsResponse is the drained durable audit-journal tail plus the client's next
// cursor (mirrors ChangesResponse for the coherence feed).
type EventsResponse struct {
	Events []journal.Row `json:"events"`
	Cursor uint64        `json:"cursor"`
}

// handleFakEvents drains the durable, hash-chained audit journal
// (internal/journal) after the client's ?since= cursor — the Seq of the last row
// it saw; 0 returns the whole retained tail. It mirrors the /v1/fak/changes
// cursor protocol but over the persisted verdict ledger rather than the live
// coherence bus. It serves the bounded in-memory tail without re-reading disk;
// the full tamper-evident history is the on-disk JSONL. Returns 404 if no journal
// is configured (FAK_AUDIT_JOURNAL unset at boot). GET or POST {"since":N}.
func (s *Server) handleFakEvents(w http.ResponseWriter, r *http.Request) {
	j := activeJournal()
	if j == nil {
		writeErr(w, http.StatusNotFound, "audit journal not enabled (set FAK_AUDIT_JOURNAL to a path)")
		return
	}
	var since uint64
	if r.Method == http.MethodPost {
		var req ChangesRequest
		if err := decodeJSON(w, r, &req); err != nil {
			writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
			return
		}
		since = req.Since
	} else if v := r.URL.Query().Get("since"); v != "" {
		n, err := strconv.ParseUint(v, 10, 64)
		if err != nil {
			writeErr(w, http.StatusBadRequest, "since must be a non-negative integer")
			return
		}
		since = n
	}
	rows := j.Recent(0)
	out := make([]journal.Row, 0, len(rows))
	cursor := since
	for _, row := range rows {
		if row.Seq > since {
			out = append(out, row)
		}
		if row.Seq > cursor {
			cursor = row.Seq
		}
	}
	writeJSON(w, http.StatusOK, EventsResponse{Events: out, Cursor: cursor})
}

// handleFakRevoke triggers a fleet-wide refutation of an external world-state witness.
func (s *Server) handleFakRevoke(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req RevokeRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	if req.Witness == "" {
		writeErr(w, http.StatusBadRequest, "revoke requires a non-empty witness")
		return
	}
	evicted, te := s.revoke(req.Witness)
	writeJSON(w, http.StatusOK, RevokeResponse{Witness: req.Witness, Evicted: evicted, TrustEpoch: te})
}

// handleFakContextChange records a safe requester-initiated mutation against a
// persisted recall core image. The only shipped mutation is a tombstone that
// suppresses one page from future model-visible recall without deleting evidence.
func (s *Server) handleFakContextChange(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var req ContextChangeRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	resp, err := s.contextChange(r.Context(), req)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleFakPolicyReload reloads the configured policy manifest in-place. The
// actual loader is injected by cmd/fak so this package stays policy-schema blind.
func (s *Server) handleFakPolicyReload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	if s.reloadPolicy == nil {
		writeErr(w, http.StatusNotFound, "policy reload is not configured")
		return
	}
	resp, err := s.reloadPolicy(r.Context())
	if err != nil {
		s.logf("gateway: policy reload failed: %v", err)
		writeErr(w, http.StatusBadRequest, "policy reload failed: "+err.Error())
		return
	}
	if resp.Source != "" {
		s.logf("gateway: reloaded capability floor from %s", resp.Source)
	} else {
		s.logf("gateway: reloaded capability floor")
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleFakTraceReset clears the per-trace IFC taint high-water mark for a live
// served session. The reset implementation is injected by cmd/fak.
func (s *Server) handleFakTraceReset(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	if s.resetTrace == nil {
		writeErr(w, http.StatusNotFound, "trace reset is not configured")
		return
	}
	var req TraceResetRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	traceID := strings.TrimSpace(req.TraceID)
	if traceID == "" {
		writeErr(w, http.StatusBadRequest, "trace_id is required")
		return
	}
	if err := s.resetTrace(r.Context(), traceID); err != nil {
		s.logf("gateway: trace reset failed: %v", err)
		writeErr(w, http.StatusBadRequest, "trace reset failed: "+err.Error())
		return
	}
	s.logf("gateway: reset trace %s", traceID)
	writeJSON(w, http.StatusOK, TraceResetResponse{Reset: true, TraceID: traceID})
}

// handleFakTraceObserve is the read-only complement of /v1/fak/trace/reset (#411):
// GET /v1/fak/trace/{trace_id} returns the current IFC taint high-water mark for a
// live/recent served session, so an operator can see whether a session's taint is
// rising WITHOUT parsing stderr. It is mounted on the /v1/fak/trace/ subtree; the
// exact /v1/fak/trace/reset route (POST) is matched first by the mux, so only the
// observe id-path lands here. The observe implementation is injected by cmd/fak so
// this package stays IFC-internals blind, mirroring resetTrace.
func (s *Server) handleFakTraceObserve(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeErr(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	if s.observeTrace == nil {
		writeErr(w, http.StatusNotFound, "trace observe is not configured")
		return
	}
	traceID := strings.TrimSpace(strings.TrimPrefix(r.URL.Path, "/v1/fak/trace/"))
	if traceID == "" {
		writeErr(w, http.StatusBadRequest, "trace_id is required")
		return
	}
	level, dangerous := s.observeTrace(r.Context(), traceID)
	writeJSON(w, http.StatusOK, TraceObserveResponse{TraceID: traceID, Taint: level, Dangerous: dangerous})
}

// handleFakSession is the session DRIVE-state control surface, the read-write
// generalization of /v1/fak/trace (which carries exactly one bit — taint). It is
// mounted on the /v1/fak/session/ subtree; the remainder is "{trace_id}" for an
// observe (GET) or "{trace_id}/{verb}" for a control verb (POST). The observe and
// control implementations are injected by cmd/fak so this package stays
// session-internals-blind, mirroring resetTrace/observeTrace. A nil injection ⇒
// 404 (never a silent clean reading) — the same fail-closed posture as the trace
// routes with no ledger.
func (s *Server) handleFakSession(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/v1/fak/session/")
	rest = strings.TrimSuffix(rest, "/")
	parts := strings.Split(rest, "/")
	if len(parts) == 0 || strings.TrimSpace(parts[0]) == "" {
		writeErr(w, http.StatusBadRequest, "trace_id is required")
		return
	}
	traceID := strings.TrimSpace(parts[0])
	verb := ""
	if len(parts) >= 2 {
		verb = strings.TrimSpace(parts[1])
	}

	switch r.Method {
	case http.MethodGet:
		// GET observes one session. A verb on the path is not the observe shape.
		if verb != "" {
			writeErr(w, http.StatusMethodNotAllowed, "use GET /v1/fak/session/{trace_id}")
			return
		}
		if s.observeSession == nil {
			writeErr(w, http.StatusNotFound, "session observe is not configured")
			return
		}
		writeJSON(w, http.StatusOK, s.observeSession(r.Context(), traceID))
	case http.MethodPost:
		// POST applies a control verb. The verb is required from the path.
		if verb == "" {
			writeErr(w, http.StatusBadRequest, "control verb is required: POST /v1/fak/session/{trace_id}/{run|budget|pace|priority}")
			return
		}
		// steer is its own shape (operator input to a RUNNING session, #760): a different
		// body and a different sink (the a2achan bus, not the drive table). Dispatch it
		// before the generic control path. A refused steer (tainted/over-scoped/uncapped)
		// is the adjudication floor's deny — distinct from the control 409 — so it maps to
		// 422 (unprocessable), not 409 (terminal/stale rev).
		if verb == "steer" {
			if s.steerSession == nil {
				writeErr(w, http.StatusNotFound, "session steer is not configured")
				return
			}
			var sr SteerRequest
			if err := decodeJSON(w, r, &sr); err != nil {
				writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
				return
			}
			if strings.TrimSpace(sr.Text) == "" {
				writeErr(w, http.StatusBadRequest, "steer text is required")
				return
			}
			if err := s.steerSession(r.Context(), traceID, sr.Text); err != nil {
				writeErr(w, http.StatusUnprocessableEntity, "steer refused: "+err.Error())
				return
			}
			s.logf("gateway: session %s steer accepted (%d bytes)", traceID, len(sr.Text))
			writeJSON(w, http.StatusAccepted, map[string]any{"trace_id": traceID, "steered": true})
			return
		}
		if s.controlSession == nil {
			writeErr(w, http.StatusNotFound, "session control is not configured")
			return
		}
		var req SessionControlRequest
		if err := decodeJSON(w, r, &req); err != nil {
			writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
			return
		}
		st, ok, err := s.controlSession(r.Context(), traceID, verb, req)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		if !ok {
			// Terminal session, or an if_rev CAS guard lost the race: the client
			// re-reads and retries. Not a malformed request.
			writeErr(w, http.StatusConflict, "session control refused (terminal or stale rev)")
			return
		}
		s.logf("gateway: session %s %s -> rev %d (%s)", traceID, verb, st.Rev, st.Run)
		writeJSON(w, http.StatusOK, st)
	default:
		writeErr(w, http.StatusMethodNotAllowed, "use GET or POST")
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	// planner names the /v1/chat/completions backend ("mock" | "proxy" | "inkernel")
	// so a probe can detect the silent offline-mock fallback that New also warns
	// about at boot — scripted responses must never be mistaken for model output.
	writeJSON(w, http.StatusOK, map[string]any{"ok": true, "engine": s.engineID, "model": s.model, "planner": plannerKind(s.planner)})
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func (s *Server) decodeSyscall(w http.ResponseWriter, r *http.Request) (SyscallRequest, bool) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return SyscallRequest{}, false
	}
	var req SyscallRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return SyscallRequest{}, false
	}
	req.TraceID = s.useHTTPTrace(w, r, req.TraceID)
	return req, true
}

// decodeJSON reads a bounded body and decodes JSON. It does NOT reject unknown
// fields — drop-in OpenAI compatibility requires ignoring extra fields — but the
// DTOs have no Ref field, so a client cannot smuggle a kernel CAS handle.
func decodeJSON(w http.ResponseWriter, r *http.Request, v any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxBody)
	return json.NewDecoder(r.Body).Decode(v)
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

// writeErrCode is writeErr with an explicit OpenAI-style error `code`. An empty
// code keeps the historical code:null shape; a non-empty code (e.g.
// "upstream_unreachable") lets a client branch on the specific failure class
// rather than guessing from the message text (#346).
func writeErrCode(w http.ResponseWriter, status int, code, msg string) {
	var codeVal any
	if code != "" {
		codeVal = code
	}
	writeJSON(w, status, map[string]any{
		"error": map[string]any{"message": msg, "type": errType(status), "code": codeVal, "param": nil},
	})
}

func errType(status int) string {
	switch {
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		return "authentication_error"
	case status >= 500:
		return "server_error"
	default:
		return "invalid_request_error"
	}
}

// durEnv reads an integer-seconds timeout override from the environment, returning
// def when the var is unset or unparseable. A non-negative integer wins: 0 selects
// Go's "no timeout" semantics (an explicit, documented opt-out for a long-running
// local backend); a negative value is rejected and def is kept. This is the seam
// that lets a slow CPU-served model finish a turn without tripping WriteTimeout.
func durEnv(name string, def time.Duration) time.Duration {
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return def
	}
	return time.Duration(n) * time.Second
}
