package gateway

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// maxBody bounds an inbound tool-args / MCP-frame body (defense against an
// unbounded read from an untrusted client). 4 MiB is far above any real
// tool-args payload.
const maxBody = 4 << 20

// maxTranscriptBody bounds an inbound /v1/messages or /v1/chat/completions body.
// A RESUMED long-context session re-sends its whole transcript every turn, so the
// request body legitimately grows past the 4 MiB tool-args cap — a 388k-token
// resume serializes to several MiB of JSON. 32 MiB matches the real Anthropic
// request-body ceiling, so the gateway never refuses a body the upstream would
// have accepted (the silent-truncation 400 in #-resume).
const maxTranscriptBody = 32 << 20

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
		{"/", s.handleHome},
		// A2A Agent-to-Agent protocol surface (#1019).
		{"/a2a/v1/messages", s.handleA2ASendMessage},
		{"/a2a/v1/tasks", s.handleA2AListTasks},
		{"/a2a/v1/agent-card", s.handleA2AGetExtendedAgentCard},
		// /a2a/v1/tasks/{id} subtree: GET reads one task, POST /cancel cancels it.
		{"/a2a/v1/tasks/", s.handleA2ATask},
		// OpenAI-compatible surface.
		{"/v1/chat/completions", s.handleChatCompletions},
		// Legacy OpenAI text-completion wire — the pre-chat surface vLLM/SGLang/
		// llama.cpp-server all still serve, for older clients and eval harnesses. No
		// tools on this wire, so no tool-call adjudication; adapts onto the same served
		// completion path as the chat route.
		{"/v1/completions", s.handleCompletions},
		// OpenAI Responses API — a client-facing inbound route so a Responses-native
		// agent (Codex CLI, the Terminal-Bench terminus agent) can route its model
		// traffic through the kernel's tool-call adjudication, the same as the chat
		// wire. Buffered only; stream:true is refused (#925).
		{"/v1/responses", s.handleResponses},
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
		{"/v1/fak/vcache/score", s.handleFakVCacheScore},
		{"/v1/fak/vcache/actions", s.handleFakVCacheActions},
		// /v1/fak/usage/cache-alignment is the per-request provider prompt-cache
		// alignment read (#10670): the last N completed requests, the share
		// cache-aligned at the canonical threshold, and each request's join
		// against the native warm-state receipt. GET, read-only, counts and
		// ratios only.
		{"/v1/fak/usage/cache-alignment", s.handleFakUsageCacheAlignment},
		{"/v1/fak/session-audit/actions", s.handleFakSessionAuditActions},
		// /v1/fak/ctxvalue is the managed-context arm of the value API: the per-session
		// multi-level (tokens / turns / session) long-session context report plus the
		// closed step-advice verdict. GET; ?trace=<id> narrows to one session.
		{"/v1/fak/ctxvalue", s.handleFakCtxValue},
		{"/v1/fak/revoke", s.handleFakRevoke},
		{"/v1/fak/context/change", s.handleFakContextChange},
		// /v1/fak/policy (exact, GET) is the read-only floor attestation (#3960); the
		// longer exact /v1/fak/policy/reload (POST) is matched independently by the mux,
		// so the observe route never shadows the reload route.
		{"/v1/fak/policy", s.handleFakPolicyObserve},
		{"/v1/fak/policy/reload", s.handleFakPolicyReload},
		{"/v1/fak/cache/posture", s.handleFakCachePosture},
		{"/v1/fak/route/reload", s.handleFakRouteReload},
		{"/v1/fak/trace/reset", s.handleFakTraceReset},
		{"/v1/fak/trace/", s.handleFakTraceObserve},
		// /v1/fak/session/changes is the DRIVE-state revision stream (#630): a
		// cursor-drained tail of every session-table Rev bump. Registered as an EXACT
		// path so net/http.ServeMux matches it ahead of the /v1/fak/session/ subtree
		// (a longer, exact pattern wins) — a session whose id is literally "changes"
		// is not addressable, which is fine (ids are gateway-minted gw-<n>).
		{"/v1/fak/discovery/", s.handleFakSessionDiscovery},
		{"/v1/fak/session/changes", s.handleFakSessionChanges},
		// /v1/fak/session/ is the DRIVE-state control surface: GET /v1/fak/session/{id}
		// observes one session's run-state/budget/priority/pace; POST
		// /v1/fak/session/{id}/{verb} applies a control verb
		// (run|budget|pace|priority|wall|throughput).
		// One subtree handler dispatches on method + the trailing path segments.
		{"/v1/fak/session/", s.handleFakSession},
		// /v1/fak/sessions (no trailing slash) is the MULTI-session read: a snapshot of
		// every live session's drive state. Registered distinctly from the singular
		// /v1/fak/session/ subtree, so a single-id request never lands here.
		{"/v1/fak/sessions", s.handleFakSessions},
		// /v1/fak/observation is the versioned aggregate diagnostic read: one
		// point-in-time set of typed source envelopes for sessions, cache
		// attribution, managed-cache posture, and harness resources.
		{"/v1/fak/observation", s.handleFakObservation},
		{"/v1/fak/fleet", s.handleFakFleet},
		// /v1/fak/tasks is the read-only process task-manager snapshot. Inert (404)
		// unless a host installs a provider via SetTasksSnapshotProvider and the
		// operator enables it; the snapshot carries accounting only, no payload bytes.
		{"/v1/fak/tasks", s.handleFakTasks},
		// /v1/fak/sharedtask/ is the shared-task record co-editing subtree (#3885):
		// GET /v1/fak/sharedtask/{task_id} is the scope-redacted record view, GET
		// {task_id}/events the same-policy historical catch-up, POST {task_id}
		// creates a record, POST {task_id}/patch is the adjudicated write through
		// the internal/sharedtask fold (accept / conflict / deny / quarantine).
		// Inert (404) unless the host installs a provider via SetSharedTaskProvider
		// and the operator enables it (FAK_SHAREDTASK=1).
		{"/v1/fak/sharedtask/", s.handleFakSharedTask},
		// /v1/fak/agent/sessions is the agent-runtime spine (#3258, epic #3256): POST
		// a goal and stream back ONE kernel-governed owned-loop session as NDJSON
		// events — session.start, per-call adjudicated `call` rows, session.end with
		// the ArmMetrics witness. The loop is agent.RunGovernedArm over the server's
		// planner (offline mock / --gguf in-kernel / proxy), so every tool call
		// crosses the in-kernel syscall boundary and the route runs offline in CI.
		{"/v1/fak/agent/sessions", s.handleFakAgentSessions},
		// /v1/fak/loops is the in-kernel background-loop runtime view: a JSON snapshot
		// of every supervised loop and its live progress (the observability half of the
		// loop control plane; complements the loopmgr ledger `fak loop status` reads).
		{"/v1/fak/loops", s.handleFakLoops},
		// /v1/fak/account/rehome is the operator "switch seat now" button: force the
		// live guarded session onto the next available account (the on-demand form of
		// the 403-triggered account failover). Inert (404) unless the host installs a
		// swap function via SetAccountRehomeFunc — fak guard does, on the pinned
		// Claude-subscription path. See account_rehome.go.
		{"/v1/fak/account/rehome", s.handleFakAccountRehome},
		{"/v1/models", s.handleModels},
		// Multi-node dev-server READ plane (#2297, epic #2254 plane 1): the
		// coordinator clone's live lease view (the dos_arbitrate live_leases
		// projection) and presence (session descriptors + lease liveness
		// classification), observed from refs/fak/locks/* at request time.
		// Read-only; injected by the host CLI (SetLeasePlaneProviders), 404 when
		// unwired. /v1/sessions is the cross-machine guard-session presence view —
		// distinct from /v1/fak/sessions, the served-session DRIVE-state snapshot.
		{"/v1/leases", s.handleLeases},
		// Multi-node dev-server WRITE plane (#2299, epic #2254 plane 1 — the atomicity
		// closure): POST /v1/leases/{acquire,renew,release} is the single-arbiter fenced
		// write over the coordinator clone's refs/fak/locks/* store. Registered as the
		// /v1/leases/ subtree so a longer, exact /v1/leases (the read plane) still wins;
		// the subtree handler routes on the trailing verb segment. Serialized through the
		// gateway (leaseWriteMu) so the coordinator is a single arbiter. Injected by the
		// host CLI (SetLeaseWriteFunc), 404 when unwired. See leasewrite.go.
		{"/v1/leases/", s.handleLeaseWrite},
		{"/v1/sessions", s.handleLeaseSessions},
		// MCP-over-HTTP, operational endpoints.
		{"/mcp", s.handleMCPHTTP},
		{"/healthz", s.handleHealth},
		{"/metrics", s.handleMetrics},
		{"/debug/vars", s.handleDebugVars},
		{"/debug/guard-audit", handleGuardAuditDebug},
	}
}

// Handler builds the gateway's HTTP routes (routeTable) wrapped in the metrics
// and optional bearer-auth middleware. Routes: the OpenAI-compatible surface
// (/v1/chat/completions, /v1/embeddings, /v1/moderations, /v1/models), the
// Anthropic Messages and native Gemini surfaces, the fak-native
// syscall/adjudicate JSON endpoints, policy reload, Prometheus metrics
// (/metrics), expvar-style diagnostics (/debug/vars), the typed aggregate
// observation (/v1/fak/observation), MCP-over-HTTP (/mcp), and an
// unauthenticated health check (/healthz).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	for _, rt := range s.routeTable() {
		mux.HandleFunc(rt.pattern, rt.handler)
	}
	// /readyz is an orchestration probe rather than a product API route. Keep it
	// outside routeTable so API-spec and follower-fanout coverage remain scoped
	// to callable runtime surfaces.
	mux.HandleFunc("/readyz", s.handleReady)
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
	// Record the address we actually bound so a served descriptor can name this
	// process instead of a literal (#5642). This is the ONLY point where the chosen
	// address is known — with an ephemeral ":0" bind the port does not exist until
	// the listener does — and both entry points funnel through here.
	if a := ln.Addr(); a != nil {
		addr := a.String()
		s.boundAddr.Store(&addr)
	}
	// Bounded timeouts so a single slow/idle connection cannot pin a goroutine +
	// socket indefinitely (slow-loris-on-body / idle-keepalive DoS). ReadTimeout
	// also caps body-delivery TIME (MaxBytesReader only caps SIZE).
	//
	// WriteTimeout bounds the WHOLE handler measured from the end of the request
	// headers — and a NON-streaming turn writes the body only AFTER the model finishes,
	// so a slow LOCAL backend whose single turn takes minutes (a multi-thousand-token
	// prefill, or an in-kernel cpu-offload GLM-5.2 decode at ~0.17 tok/s) trips the
	// deadline DURING the decode: the handler logs a clean 200 but the connection is
	// already torn down, so the client sees an empty reply with zero bytes (#1015). The
	// default therefore depends on the backend the gateway is actually serving
	// (serveWriteTimeoutDefault): a local in-kernel model gets NO write timeout, while a
	// proxy-to-hosted-API (the fast, network-exposed surface) keeps the conservative
	// 90s. FAK_HTTP_WRITE_TIMEOUT_S overrides either way.
	hs := &http.Server{
		Handler:           s.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       durEnv("FAK_HTTP_READ_TIMEOUT_S", 30*time.Second),
		WriteTimeout:      durEnv("FAK_HTTP_WRITE_TIMEOUT_S", serveWriteTimeoutDefault(plannerKind(s.planner))),
		IdleTimeout:       durEnv("FAK_HTTP_IDLE_TIMEOUT_S", 120*time.Second),
		// Route net/http's own diagnostics (per-request panic recovery, TLS/keepalive
		// errors) through the SAME sink as every other gateway log. Left nil, net/http
		// falls back to the std logger → os.Stderr, which UNDER `fak guard` is the child
		// harness's controlling TTY: a recovered handler panic then dumps a multi-line
		// goroutine stack straight into the agent's TUI and corrupts the display (#2772).
		// s.logf already honors the operator's --log choice — muted by default to keep the
		// terminal clean, streamed to a file/stderr when asked — so binding ErrorLog to it
		// makes those diagnostics obey the same policy instead of bypassing it. Zero flags
		// so we don't double-stamp the timestamp s.logf already adds.
		ErrorLog: log.New(logfWriter{logf: s.logf}, "", 0),
	}
	if s.richDashboards != nil {
		defer s.richDashboards.close()
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
	// Start the in-kernel background loops on the serve lifecycle context: from here
	// until ctx is done, registered loops keep progressing (the heartbeat, plus any a
	// host registered), observable at /v1/fak/loops and via fak_bgloop_* metrics.
	s.startLoops(ctx)
	s.logf("fak gateway listening on http://%s  (engine=%s model=%s vdso=%v auth=%v)",
		ln.Addr(), s.engineID, s.model, s.k.VDSOEnabled(), s.requireKey != "")
	// Surface fak's core value-add — realized in-kernel KV-prefix reuse — at startup so it
	// is discoverable without scraping /metrics or waiting for a long --debug-stats session
	// (epic #1072). The cacheobs tap is the SAME WITNESSED signal /metrics renders; at boot
	// it is idle (no served turn yet) and climbs per in-kernel turn. A pure-proxy workload
	// never feeds it, so the honest startup line is "idle until the first in-kernel turn".
	s.logf("fak cache: %s", cacheBootSummary(cacheobs.Default.Snapshot()))
	select {
	case <-ctx.Done():
		// Join the background loops first (bounded), then drain the HTTP surface, so a
		// wedged loop is reported rather than silently outliving the gateway.
		s.stopLoops()
		shctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		return hs.Shutdown(shctx)
	case err := <-errc:
		return err
	}
}

// logfWriter adapts the gateway's structured logf onto io.Writer so an http.Server.ErrorLog
// (which speaks io.Writer) drains into the same sink. net/http hands ErrorLog one whole
// pre-formatted message per line (a panic recovery arrives as a single multi-line write), so
// we forward it verbatim minus the trailing newline; when logf is the guard default no-op the
// message is dropped and nothing reaches the terminal. A nil logf is tolerated for the same
// reason the Serve path assumes it non-nil — belt-and-braces, never a nil-deref.
type logfWriter struct {
	logf func(format string, args ...any)
}

func (w logfWriter) Write(p []byte) (int, error) {
	if w.logf != nil {
		w.logf("%s", strings.TrimRight(string(p), "\n"))
	}
	return len(p), nil
}

// cacheBootSummary renders the startup cache-state line from the process-global cacheobs
// snapshot (the WITNESSED in-kernel KV-prefix reuse). Idle at boot (no served turn yet); once
// turns accumulate it reports the realized reuse ratio AND the absolute prompt tokens served
// from cache (saved=N tok, the snapshot's ReusedTokens) so an operator sees the cliff live (#1076).
func cacheBootSummary(s cacheobs.Stats) string {
	if s.Turns == 0 {
		return "idle — realized KV-prefix reuse appears here per in-kernel turn (scrape /metrics fak_gateway_kv_prefix_* for the full family)"
	}
	return fmt.Sprintf("reuse %.0f%% (saved=%d tok) over %d turns (frozen=%d partial=%d cold=%d) — WITNESSED, by=vdso",
		s.ReuseRatio*100, s.ReusedTokens, s.Turns, s.FrozenTurns, s.PartialTurns, s.ColdTurns)
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

// withAuth enforces the configured secret on every route except the auth-exempt
// set (authExempt) when RequireKey is set. With no key configured it is a
// pass-through (drop-in, loopback default). The comparison is constant-time over
// SHA-256 digests so the reject latency leaks neither the secret's bytes nor its
// length — this is the gateway's only auth primitive on a network-reachable
// security kernel.
func (s *Server) withAuth(next http.Handler) http.Handler {
	want := sha256.Sum256([]byte(s.requireKey))
	wantRead := sha256.Sum256([]byte(s.readBearer))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if (s.requireKey != "" || s.keyset != nil) && !authExempt(r) {
			tok, ok := gatewayCredential(r)
			got := sha256.Sum256([]byte(tok))
			// The single RequireKey bearer authenticates the anonymous single-tenant
			// caller; the keyset (#5332) authenticates a bound org/project principal.
			// Both are the SAME constant-time SHA-256 compare — a request presenting ANY
			// accepted key is authenticated. A keyset match additionally ATTRIBUTES the
			// turn to its tenant principal, stamped onto the context below so principalFor
			// / traceOwner / the access log / /v1/fak/events all name the same tenant. On
			// the RequireKey-only path (keyset == nil) lookup is a nil no-op and this is
			// byte-for-byte the prior behavior.
			authed := ok && s.requireKey != "" && subtle.ConstantTimeCompare(got[:], want[:]) == 1
			principal := ""
			if ok {
				if p, matched := s.keyset.lookup(tok); matched {
					authed = true
					principal = p
				}
			}
			// The read-scoped bearer (Config.ReadBearer) is consulted only AFTER the
			// full-strength credential has already failed, and only on the read-only
			// observability paths. That ordering is what keeps it strictly widening: it
			// can admit a caller the main key would have rejected, but it can never
			// reject one the main key accepted, and it is never reachable from a
			// mutating route. Guarding on a non-empty readBearer is load-bearing, not
			// belt-and-braces — without it an UNSET read bearer would hash to the same
			// digest as an empty presented bearer and silently authorize `Bearer `.
			if !authed && s.readBearer != "" && ok && readScopedPath(r) {
				authed = subtle.ConstantTimeCompare(got[:], wantRead[:]) == 1
			}
			if !authed {
				writeErr(w, http.StatusUnauthorized, "missing or invalid credentials")
				return
			}
			if principal != "" {
				r = r.WithContext(WithPrincipal(r.Context(), principal))
			}
		}
		if s.startup.childStartupPending() && strings.HasPrefix(r.URL.Path, "/v1/") && !strings.HasPrefix(r.URL.Path, "/v1/fak/") {
			s.MarkChildUsable(time.Now())
		}
		next.ServeHTTP(w, r)
	})
}

// authExempt reports whether a request may skip the bearer check on an
// authenticated gateway. Two cases:
//
//   - /healthz is ALWAYS exempt — it is the unauthenticated liveness probe a
//     load balancer or a `fak claude-mac-fak --debug` preflight hits before it
//     holds a token, and it carries only the planner/engine words (no counts).
//   - /metrics, /debug/vars, and /v1/fak/observation are exempt ONLY for a
//     loopback caller. They are read-only observability surfaces (Prometheus,
//     legacy diagnostics, and the typed snapshot) that an operator clicks
//     straight from the host or an SSH tunnel; gating them behind the bearer is
//     what makes those panel links 401. A REMOTE caller still needs the token,
//     so request counts, token volumes, the model id, and uptime are never
//     exposed to the open internet.
func authExempt(r *http.Request) bool {
	if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
		return true
	}
	// Human/agent discovery is directly clickable from the loopback URL shown
	// in the TUI, but remains bearer-gated when the gateway is exposed off-box.
	if (r.URL.Path == "/" || r.URL.Path == "/a2a/v1/agent-card") && requestFromLoopback(r) {
		return true
	}
	if readScopedPath(r) {
		return requestFromLoopback(r)
	}
	return false
}

// readScopedPath reports whether the path is one of the read-only observability
// surfaces — the set the loopback exemption opens, and the same set the read-scoped
// bearer may unlock off-loopback. Both callers share this one predicate so the two
// grants can never drift into disagreeing about what "read-only" means: adding a
// surface here widens both at once, which is the intent.
func readScopedPath(r *http.Request) bool {
	switch r.URL.Path {
	case "/metrics", "/debug/vars", "/v1/fak/observation":
		return true
	}
	return false
}

// requestFromLoopback reports whether the request's peer is the loopback
// interface, classifying by IP VALUE (net.ParseIP + IsLoopback) rather than a
// string prefix so a spoofed RemoteAddr host cannot masquerade as local. An
// unparseable RemoteAddr is treated as NOT loopback (fail closed). RemoteAddr is
// the kernel-observed peer of the TCP connection, set by net/http — it is not a
// client-supplied header (unlike X-Forwarded-For, which is deliberately ignored
// here so a proxied header can never grant the exemption).
func requestFromLoopback(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	host = strings.Trim(host, "[]")
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
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
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	waitEPFanout, ok := s.startEPFanoutFollowers(w, r, epRouteChatCompletions)
	if !ok {
		return
	}
	defer waitEPFanout()
	var req ChatRequest
	if !decodeRequestBody(w, r, &req) {
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
	if rejectInvalidSampling(w, validateSampling(req)) {
		return
	}
	// Stamp the causal input on the untouched wire envelope before admission
	// transforms, request routing, planner selection, or model execution.
	inputTriggerRoute, routedModel, err := s.admitAndRouteChatInputTrigger(req)
	if err != nil {
		if errors.Is(err, errInvalidExplicitInputTrigger) {
			writeErr(w, http.StatusBadRequest, "invalid input_trigger")
			return
		}
		s.logf("gateway: input-trigger request route failed: %v", err)
		writeErr(w, http.StatusInternalServerError, "input-trigger request routing failed")
		return
	}
	if routedModel != "" {
		req.Model = routedModel
	}
	receiptRequested := req.Fak != nil && req.Fak.NativeInferenceReceipt
	decodeTraceRequested := req.FakDecodeTrace
	decodeTokenIDsRequested := req.Fak != nil && req.Fak.NativeDecodeTokenIDs
	if decodeTokenIDsRequested && !decodeTraceRequested {
		writeErr(w, http.StatusBadRequest, "native decode token IDs require fak_decode_trace")
		return
	}
	if decodeTraceRequested && req.Stream {
		writeErr(w, http.StatusBadRequest, "fak_decode_trace requires a buffered fak-native request")
		return
	}
	if receiptRequested && req.Stream {
		writeErr(w, http.StatusBadRequest, "native inference receipts require a buffered request")
		return
	}
	if receiptRequested && ((req.Temperature != nil && *req.Temperature != 0) || (req.TopP != nil && *req.TopP != 0) || len(req.LogitBias) > 0 || (req.FrequencyPenalty != nil && *req.FrequencyPenalty != 0) || (req.PresencePenalty != nil && *req.PresencePenalty != 0)) {
		writeErr(w, http.StatusBadRequest, "native inference receipts require greedy sampling over unmodified logits")
		return
	}
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
	if decodeTraceRequested && !s.chatDecodeTraceSupported(req.Model) {
		writeErr(w, http.StatusBadRequest, "fak_decode_trace requires a fak-native model route")
		return
	}

	// Thread one request TraceID across every proposed call in this chat so the IFC
	// ledger, plan-CFI, response header, and access log all correlate. The
	// middleware honors a client-supplied X-Trace-Id or mints one.
	ctx, reqTrace, messages, sessionTurn, admitted := s.admitServedRequest(w, r, req.Messages)
	if !admitted {
		return
	}
	req.Messages = messages
	resultAdmissions, err := s.admitInboundResults(ctx, req.Messages, req.Tools, reqTrace)
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
		if s.streamChatLive(ctx, w, req, reqModel, reqTrace, sessionTurn, resultAdmissions, inputTriggerRoute) {
			return
		}
	}

	// #5399 (the remaining half of #4855): the buffered path below blocks in
	// completeServed for the WHOLE decode, and used to write its first byte only after
	// that returned — so a stream:true request served by a Complete-only planner (every
	// in-kernel serve: agent.InKernelPlanner is not an agent.StreamingPlanner) emitted no
	// status line, no headers and no SSE byte for the entire multi-rank decode. Open the
	// stream NOW instead: 200 + SSE headers + the opening role chunk, flushed, so the
	// client can tell an accepted streaming request from a dead socket.
	//
	// Placement is load-bearing. Every PRE-decode refusal — the method/body/sampling
	// 400s, writeSessionRefusal, the inbound-result 502 — is already behind us and kept
	// its real HTTP status. Everything that can still fail below (the upstream error,
	// the tool-call conformance fail-closed) has to report in-band now, as an SSE error
	// event + [DONE]; see chatStreamWriter.fail.
	var stream *chatStreamWriter
	if req.Stream {
		stream = newChatStreamWriter(w, reqModel)
		if err := stream.open(); err != nil {
			// The client is already gone; do not spend a decode on a socket nobody reads.
			s.logf("gateway: client vanished before the streamed preamble landed: %v", err)
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
		agent.WithToolChoice(req.ToolChoice),
		agent.WithLogitBias(req.LogitBias),
		agent.WithGuidedDecode(req.GuidedDecodeFields()),
		// Repetition-penalty passthrough (#1705): forward frequency_penalty/
		// presence_penalty to the in-kernel sampler so a reasoning model can break a
		// non-terminating repetition loop the way an upstream ride engine already
		// could. No-op when the client omitted them (nil pointer).
		agent.WithFrequencyPenalty(req.FrequencyPenalty),
		agent.WithPresencePenalty(req.PresencePenalty),
		agent.WithNativeInferenceReceipt(receiptRequested),
		agent.WithDecodeTrace(decodeTraceRequested),
		agent.WithNativeDecodeTokenIDs(decodeTokenIDsRequested),
	)
	if err != nil {
		// Map the upstream failure to an honest status. Log the detail for the operator
		// but return a GENERIC message — the planner error embeds up to 400 bytes of the
		// upstream provider's raw body, which must not cross the trust boundary to a
		// (possibly unauthenticated) downstream caller.
		s.logf("gateway: upstream model error: %v", err)
		if stream != nil {
			// The 200 + SSE headers went out before the decode, so the status line is
			// spent: report the SAME classified failure in-band as an SSE error event +
			// [DONE] rather than truncating the stream. plannerErrorStatus carries the
			// identical metric/observation side effects writeUpstreamErr would have run,
			// and msg is the same client-facing string (never the upstream's raw body).
			status, code, msg := s.plannerErrorStatus(err)
			stream.fail(status, code, msg)
			return
		}
		s.writeUpstreamErr(w, err)
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
		const conformanceMsg = "upstream tool-call format not recognized; refusing to skip adjudication"
		if stream != nil {
			// Fail closed in-band: the preamble already spent the status line, but the
			// client must still never read a benign empty stop on a skipped adjudication.
			stream.fail(http.StatusBadGateway, "", conformanceMsg)
			return
		}
		writeErr(w, http.StatusBadGateway, conformanceMsg)
		return
	}
	if receiptRequested && comp.NativeInference == nil {
		writeErr(w, http.StatusBadGateway, "configured planner cannot produce a native inference receipt")
		return
	}
	if inputTriggerRoute != nil && comp.NativeInference != nil &&
		(comp.NativeInference.Engine != TurnIngressEngine ||
			comp.NativeInference.Model != TurnIngressModel ||
			comp.NativeInference.FallbackActive) {
		s.logf("gateway: input-trigger route execution identity mismatch")
		writeErr(w, http.StatusBadGateway, "fak-native route execution identity mismatch")
		return
	}
	if decodeTraceRequested && comp.DecodeTrace == nil {
		writeErr(w, http.StatusBadGateway, "fak-native planner did not produce a decode trace")
		return
	}
	if decodeTokenIDsRequested && comp.NativeDecodeTokenIDs == nil {
		writeErr(w, http.StatusBadGateway, "fak-native planner did not produce native decode token IDs")
		return
	}

	kept, adjs, dropped, servedText, servedHits, bodyRefused := s.adjudicateProposedTurn(ctx, asst, reqTrace)
	finish := s.applyAdjudicatedTurn(&asst, adjs, kept, dropped, servedHits, servedText, bodyRefused, comp.FinishReason)

	// Echo the model the UPSTREAM reported it served (#82); fall back to the client's
	// requested model (or, if it omitted one, the configured model) when the upstream
	// did not name a served model. Never just s.model — that is the silent-substitution
	// this fix removes.
	respModel := s.responseModel(comp.Model, reqModel, chatStreamModel(stream), "#5399")
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
	if len(adjs) > 0 || len(resultAdmissions) > 0 || len(redactions) > 0 || inputTriggerRoute != nil {
		resp.Fak = &FakExt{Adjudications: adjs, ResultAdmissions: resultAdmissions, Redactions: redactions}
	}
	if inputTriggerRoute != nil {
		resp.Fak.InputTriggerRoute = inputTriggerRoute
	}
	if comp.NativeInference != nil {
		if s.nativeReceiptMetrics != nil {
			s.nativeReceiptMetrics.Observe(comp.NativeInference, time.Now())
		}
		if resp.Fak == nil {
			resp.Fak = &FakExt{}
		}
		resp.Fak.NativeInferenceReceipt = comp.NativeInference
	}
	if decodeTraceRequested {
		if resp.Fak == nil {
			resp.Fak = &FakExt{}
		}
		resp.Fak.DecodeTrace = comp.DecodeTrace
	}
	if decodeTokenIDsRequested {
		if resp.Fak == nil {
			resp.Fak = &FakExt{}
		}
		resp.Fak.NativeDecodeTokenIDs = comp.NativeDecodeTokenIDs
	}
	if stream != nil {
		writeChatCompletionStream(stream, resp)
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) chatDecodeTraceSupported(reqModel string) bool {
	p := s.planner
	if dual, ok := p.(*DualPlanner); ok {
		if !dual.RoutesLocal(reqModel) {
			return false
		}
		p = dual.Local()
	}
	capable, ok := p.(agent.NativeDecodeTracePlanner)
	return ok && capable.NativeDecodeTraceSupported()
}

func (s *Server) responseModel(served, requested, streamModel, issue string) string {
	if served == "" {
		served = requested
	}
	if streamModel != "" && served != streamModel {
		s.logf("gateway: streamed turn announced model %q in its preamble; upstream served %q (constant-model SSE, %s)", streamModel, served, issue)
		return streamModel
	}
	return served
}

func chatStreamModel(stream *chatStreamWriter) string {
	if stream == nil {
		return ""
	}
	return stream.model
}

// applyAdjudicatedTurn folds one adjudicated proposal set into the assistant
// message the OpenAI wire will carry — refused-body blanking, vDSO served-inline
// text, the livelock advisory note — and returns the finish_reason the choice
// reports: "tool_calls" while calls survive, "stop" when every proposal was
// refused or served inline (with the deny summary in-band when the content is
// otherwise empty), else the upstream's own finish.
func (s *Server) applyAdjudicatedTurn(asst *agent.Message, adjs []ToolAdjudication, kept []agent.ToolCall, dropped, servedHits int, servedText string, bodyRefused bool, upstreamFinish string) string {
	asst.ToolCalls = kept
	// #3567 output-side shadow: classify the MODEL's own outbound prose (sampled,
	// observe-only) before fak blanks/appends anything, folding the negative-framing
	// finding count into fak_negframe_output_negatives_total. Pure telemetry.
	outputNegframeAudit.observe(asst.Content)
	if bodyRefused {
		asst.Content = ""
	}
	// vDSO served-inline (vDSO live in the hot path): a re-proposed read-only call the
	// vDSO already holds fresh is answered LOCALLY and folded into the assistant content
	// (the OpenAI assistant message carries tool_calls, never results), the call dropped
	// from kept so the client never re-runs it — the engine round-trip is saved.
	if servedText != "" {
		if asst.Content != "" {
			asst.Content += "\n" + servedText
		} else {
			asst.Content = servedText
		}
	}
	if servedHits > 0 {
		s.metrics.recordServedInline(servedHits)
	}
	if anyLivelock(adjs) {
		asst.Content = prependAdjudicationContentNote(asst.Content, adjs)
	}
	finish := upstreamFinish
	if len(kept) > 0 {
		finish = "tool_calls"
	} else if dropped > 0 || servedHits > 0 {
		// Every proposed call was refused OR served inline. A pure-served turn has
		// dropped==0, so broaden the guard: the turn ends (no surviving tool_calls
		// array for the client to act on). On a deny, surface the reason in-band.
		finish = "stop"
		if asst.Content == "" {
			asst.Content = denySummary(adjs)
		}
	}
	return finish
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
	return validateSamplingRanges(req.Temperature, req.TopP)
}

// validateSamplingRanges enforces the temperature/top_p range contract shared by
// every inbound wire (chat, completions, responses): a present temperature must be
// in [0, 2] and a present top_p in [0, 1]. A nil pointer (the field was omitted) is
// always valid. Returns the first client-facing 400 message, or "" when both are in
// range. Each wire keeps its own max-tokens check inline because the field name
// differs (max_tokens vs max_output_tokens).
func validateSamplingRanges(temperature, topP *float64) string {
	if temperature != nil && (*temperature < 0 || *temperature > 2) {
		return "temperature: must be in [0, 2]"
	}
	if topP != nil && (*topP < 0 || *topP > 1) {
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
	if status, code, msg, ok := admissionErrorStatus(err); ok {
		return status, code, msg
	}
	if recurrentEvictUnsupported(err) {
		return http.StatusConflict, "in_kernel_recurrent_evict_unsupported",
			"context budget exhausted but this Gated-DeltaNet recurrent cache cannot evict in place; retry in a fresh session or start fak serve with --reset-on-budget"
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
	// An upstream that opened the stream then went SILENT mid-turn (the idle-deadline trip in
	// the streaming planner). This is a timeout, not a generic upstream failure: 504 Gateway
	// Timeout with a distinct code so a client/harness can tell a stalled provider apart from a
	// 4xx request error or a parse failure — instead of the opaque code:null "upstream model
	// error". Placed before the status/fallthrough cases for the same reason as unreachable.
	var stalled *agent.UpstreamStalledError
	if errors.As(err, &stalled) {
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
	writeErrCode(w, status, code, msg)
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

func (s *Server) decodeSyscall(w http.ResponseWriter, r *http.Request) (SyscallRequest, bool) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return SyscallRequest{}, false
	}
	var req SyscallRequest
	if !decodeRequestBody(w, r, &req) {
		return SyscallRequest{}, false
	}
	req.TraceID = s.useHTTPTrace(w, r, req.TraceID)
	return req, true
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

// serveWriteTimeoutDefault picks the WriteTimeout default for the backend the gateway is
// serving, so a non-streaming turn never trips the deadline DURING a long synchronous decode
// (#1015). A LOCAL model — the in-kernel fused model — answers a single turn in seconds to
// MINUTES (a cpu-offload GLM-5.2 decode is multi-minute), and the whole response is written
// only after that turn finishes, so any finite write deadline measured from the request
// headers races the decode: 0 (no timeout) is the only correct default there. A "proxy" to a
// hosted API is fast and is the network-exposed surface a slow-loris cares about, so it keeps
// the conservative 90s. The mock/unknown backends are instant; the conservative default is
// harmless for them. FAK_HTTP_WRITE_TIMEOUT_S overrides this in every case (durEnv).
func serveWriteTimeoutDefault(kind string) time.Duration {
	if kind == "inkernel" {
		return 0 // a local model turn can legitimately run for minutes — no whole-handler deadline
	}
	return 90 * time.Second
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
