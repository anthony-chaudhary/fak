package gateway

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/pkg/deploykit/probe"
	"github.com/anthony-chaudhary/fak/pkg/gatewayauth"
	"github.com/anthony-chaudhary/fak/pkg/turncost"
)

// maxBody bounds an inbound tool-args / MCP-frame body (defense against an
// unbounded read from an untrusted client). 4 MiB is far above any real
// tool-args payload.
const maxBody = 4 << 20

// maxTokenizeBody bounds the stateless native prompt-preparation request. The
// response contains token identities and counts, never the source transcript.
const maxTokenizeBody = 1 << 20

const (
	healthAuthChallengeHeader = "X-Fak-Auth-Challenge"
	healthAuthProofHeader     = "X-Fak-Auth-Proof"
	healthAuthProofDomain     = "fak-health-v1\x00"
	healthAuthNonceBytes      = 32
)

type nativePromptEncoder interface {
	EncodePrompt(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (agent.PromptEncoding, error)
}

// Native batch evidence headers (#13317). The resident in-kernel coordinator
// already stamps agent.Completion.InKernelBatch with the authoritative cohort
// receipt; these headers carry that same receipt to an external benchmark so it
// can distinguish a coordinated cohort from serial fallback without importing
// public internal/* packages or inventing a second counter. The set is additive
// and opt-in: it is emitted only for a buffered request that asked for the
// native inference receipt and whose completion carries a valid batch receipt.
// CohortSize is coordinator membership, never an observed GPU batch width;
// SharedPanels=0 is authoritative serial/no-shared-panel evidence, and an absent
// header set always means unknown.
const (
	// HeaderNativeCohortID names the coordinator cohort that served the turn.
	HeaderNativeCohortID = "x-fak-native-cohort-id"
	// HeaderNativeCohortSize names the coordinator membership of that cohort.
	HeaderNativeCohortSize = "x-fak-native-cohort-size"
	// HeaderNativeSharedPanels names how many panels the cohort shared.
	HeaderNativeSharedPanels = "x-fak-native-shared-panels"
)

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
		{"/v1/fak/features", s.handleFeatures},
		{"/v1/fak/features/proof", s.handleFeatureProof},
		{"/v1/fak/tokenize", s.handleFakTokenize},
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
		// Tier 0 hot-swap control plane: dynamic scalar configuration table (#10867).
		{"/v1/control/config", s.handleControlConfig},
		{"/v1/fak/control/config", s.handleControlConfig},
		// Shift-left dry-run validation, relational invariants, and canary auto-rollback (#10869).
		{"/v1/control/apply", s.handleControlApply},
		{"/v1/fak/control/apply", s.handleControlApply},
		{"/v1/control/events", s.handleControlEvents},
		{"/v1/fak/control/events", s.handleControlEvents},
		{"/v1/control/telemetry", s.handleControlTelemetry},
		{"/v1/fak/control/telemetry", s.handleControlTelemetry},
		// The provisional durable directive store is reachable through the same
		// gateway auth door. It does not confer human C0 authority or priority.
		{"/v1/fak/control/directives", s.handleControlDirectives},
		{"/v1/fak/control/directives/", s.handleControlDirectives},
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
		// /v1/fak/observation/requests is the per-request arm of the observation
		// family: one bounded snapshot of the live-request registry (id, route,
		// start, elapsed). GET, read-only, payload-free (route + timing only).
		{"/v1/fak/observation/requests", s.handleFakObservationRequests},
		// /v1/fak/observation/engine is the per-step arm: the native
		// continuous-batching loop's phase/decode-step/cohort summary plus a
		// bounded ring of recent steps. GET, read-only, counts and timings only.
		{"/v1/fak/observation/engine", s.handleFakObservationEngine},
		// /v1/fak/perf/recent is the per-request serving-performance read: the last
		// N served turns' TTFT / prefill / decode / e2e / cache rows and their
		// quantile summary, seeded from the durable perf ledger across restarts.
		{"/v1/fak/perf/recent", s.handleFakPerfRecent},
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
		{"/mcp", s.handleAuthenticatedMCPHTTP},
		{gatewayauth.KeyProofPath, s.handleKeyProof},
		{"/healthz", s.handleHealthWithAuthProof},
		{"/metrics", s.handleMetrics},
		// Fak-native engine introspection, projected from the SAME live state
		// /metrics renders (see serving_props.go): /props is the engine's
		// self-description (admission running-set cap, loaded-model context
		// window, build label, live running/waiting/hit-rate), /slots is the
		// resident KV-prefix accounting. Registered beside /metrics because they
		// are the same class of read-only observability surface and take the same
		// read-scoped auth treatment (readScopedPath below), so a consumer that
		// speaks only the llama-server shape can observe a Fak engine instead of
		// reading a 404 as "no engine here".
		{"/props", s.handleProps},
		{"/slots", s.handleSlots},
		{"/debug/vars", s.handleDebugVars},
		{"/debug/guard-audit", handleGuardAuditDebug},
	}
}

// handleControlDirectives refuses an unconfigured credential door even on the
// loopback default, where withAuth otherwise passes requests through. The
// durable store performs method, body, id, and journal validation itself.
func isControlDirectivePath(path string) bool {
	return path == durableControlDirectivePath || strings.HasPrefix(path, durableControlDirectivePath+"/")
}

func (s *Server) handleControlDirectives(w http.ResponseWriter, r *http.Request) {
	if s.controlIngress == nil || (s.requireKey == "" && s.keyset == nil) {
		writeControlReceipt(w, http.StatusServiceUnavailable,
			rejectedControlReceipt(ControlDirective{}, "unavailable", "journal_unavailable"))
		return
	}
	s.controlIngress.ServeHTTP(w, r)
}

func (s *Server) handleKeyProof(w http.ResponseWriter, r *http.Request) {
	gatewayauth.ServeKeyProof(w, r, s.requireKey)
}

// handleHealthWithAuthProof preserves the unauthenticated health response while
// allowing a client that already holds the configured gateway key to prove
// endpoint key possession before sending that key as a bearer token. Invalid or absent
// challenges reveal nothing and leave /healthz byte-for-byte unchanged.
func (s *Server) handleHealthWithAuthProof(w http.ResponseWriter, r *http.Request) {
	gatewayauth.WriteHealthProof(w, r, s.requireKey)
	s.handleHealth(w, r)
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
	// /readyz and /version are orchestration probes rather than product API
	// routes. The shared deploykit probe contract mounts them outside routeTable
	// so API-spec and follower-fanout coverage remain scoped to callable runtime
	// surfaces. /healthz stays in routeTable, so Mount keeps its handler, and
	// Drain is nil, so the gateway mounts no drain route.
	if err := probe.Mount(mux, probe.Hooks{Ready: s.handleReady, Version: probe.SelfVersion}); err != nil {
		panic("gateway: mount probe routes: " + err.Error())
	}
	mux.HandleFunc("/v1/fak/arms", s.handleFakArms)
	mux.HandleFunc("/v1/fak/arms/traffic", s.handleFakArmsTraffic)
	mux.HandleFunc("/v1/fak/arms/lease", s.handleFakArmsLease)
	mux.HandleFunc("/v1/fak/arms/limits", s.handleFakArmsLimits)
	mux.HandleFunc("/a2a/v1/director/digest", s.handleA2AGetDirectorDigest)
	if s.richDashboards != nil && s.richDashboards.proxyGrafana && s.richDashboards.proxy != nil {
		prefix := s.richDashboards.proxyPrefix
		if prefix == "" {
			prefix = "/grafana"
		}
		prefix = "/" + strings.Trim(prefix, "/")
		mux.Handle(prefix+"/", s.richDashboards.proxy)
	}
	return s.withFeatureActivations(s.withMetrics(s.withAuth(s.withNativeFeatureProof(mux))))
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
	drainTimeout := parseHTTPDrainTimeout(os.Getenv("FAK_HTTP_DRAIN_TIMEOUT_S"))
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
	s.metrics.setHTTPWriteTimeout(hs.WriteTimeout)
	errc := make(chan error, 1)
	go func() { errc <- hs.Serve(nodelayListener(ln)) }()
	// The boot timeline closes here: the listener is bound and the gateway is
	// ready to adjudicate. Any eager model load the host did (fak serve --gguf) has
	// already completed before this point, so time-to-ready spans it.
	s.MarkReady()
	// Start the in-kernel background loops on the serve lifecycle context: from here
	// until ctx is done, registered loops keep progressing (the heartbeat, plus any a
	// host registered), observable at /v1/fak/loops and via fak_bgloop_* metrics. The
	// replica fleet's live health/drain loop (issue fak-private#2417) is one of them —
	// registered as "fleet-health" in newBgloopSupervisor, so it starts here and is
	// cancelled AND joined by stopLoops on shutdown.
	s.startLoops(ctx)
	if s.logf != nil {
		s.logf("fak gateway listening on http://%s (planner=%s engine=%s model=%s vdso=%v auth=%v)",
			ln.Addr(), plannerKind(s.planner), s.engineID, s.model, s.k.VDSOEnabled(), s.requireKey != "")
		if !s.warmup.pending() {
			s.logf("[READY] fak gateway is ready to accept requests on http://%s", ln.Addr())
		}
		// Surface fak's core value-add — realized in-kernel KV-prefix reuse — at startup so it
		// is discoverable without scraping /metrics or waiting for a long --debug-stats session
		// (epic #1072). The cacheobs tap is the SAME WITNESSED signal /metrics renders; at boot
		// it is idle (no served turn yet) and climbs per in-kernel turn. A pure-proxy workload
		// never feeds it, so the honest startup line is "idle until the first in-kernel turn".
		s.logf("fak cache: %s", cacheBootSummary(cacheobs.Default.Snapshot()))
	}
	select {
	case <-ctx.Done():
		s.stopping.Store(true)
		if s.logf != nil {
			var inflight int64
			if s.metrics != nil {
				inflight = atomic.LoadInt64(&s.metrics.inflight)
			}
			s.logf("fak gateway shutdown: drain_timeout=%s inflight_requests=%d", drainTimeout, inflight)
		}
		// Join the background loops first (bounded), then drain the HTTP surface, so a
		// wedged loop is reported rather than silently outliving the gateway.
		s.stopLoops()
		shctx, cancel := context.WithTimeout(context.Background(), drainTimeout)
		defer cancel()
		return hs.Shutdown(shctx)
	case err := <-errc:
		return err
	}
}

const defaultHTTPDrainTimeout = 5 * time.Second

// parseHTTPDrainTimeout accepts positive whole seconds that fit in a Duration.
// Invalid values retain the default so shutdown always has a finite deadline.
func parseHTTPDrainTimeout(raw string) time.Duration {
	n, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64)
	if err != nil || n <= 0 || n > int64((1<<63-1)/time.Second) {
		return defaultHTTPDrainTimeout
	}
	return time.Duration(n) * time.Second
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
		// The X-Fak-Auth-Scope guard (#12761) runs BEFORE the bearer check so a
		// scope-refused request answers 403 without ever consulting credentials.
		// An absent header falls through unchanged (the legacy path is byte-for-
		// byte identical); enforceAuthScope returns false only when it already
		// wrote the 403 scope_forbidden response.
		if !s.enforceAuthScope(w, r) {
			return
		}
		if (s.requireKey != "" || s.keyset != nil) && !s.authExempt(r) {
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
				writeAuthError(w, r, ok)
				return
			}
			if principal != "" {
				r = r.WithContext(WithPrincipal(r.Context(), principal))
			}
		} else if s.keyset != nil {
			if tok, ok := gatewayCredential(r); ok {
				if p, matched := s.keyset.lookup(tok); matched {
					r = r.WithContext(WithPrincipal(r.Context(), p))
				}
			}
		}
		if s.startup.childStartupPending() && strings.HasPrefix(r.URL.Path, "/v1/") && !strings.HasPrefix(r.URL.Path, "/v1/fak/") {
			s.MarkChildUsable(time.Now())
		}
		next.ServeHTTP(w, r)
	})
}

// authExempt reports whether a request may skip the bearer check on an
// authenticated gateway.
func (s *Server) authExempt(r *http.Request) bool {
	// A LAN peer must still present the configured credential
	// before it can submit or read a provisional control directive.
	if isControlDirectivePath(r.URL.Path) {
		return false
	}
	if r.URL.Path == gatewayauth.KeyProofPath {
		return true
	}
	if r.URL.Path == "/healthz" || r.URL.Path == "/readyz" {
		return true
	}
	if requestViaProxy(r) {
		return false
	}
	if s.allowLAN && requestFromLAN(r) {
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
	if s.richDashboards != nil && s.richDashboards.proxyGrafana && requestFromLoopback(r) {
		prefix := s.richDashboards.proxyPrefix
		if prefix == "" {
			prefix = "/grafana"
		}
		prefix = "/" + strings.Trim(prefix, "/")
		if r.URL.Path == prefix || strings.HasPrefix(r.URL.Path, prefix+"/") {
			return true
		}
	}
	return false
}

var proxyHeaders = []string{
	"CF-Connecting-IP", "CF-Ray", "CF-Visitor",
	"X-Forwarded-For", "X-Forwarded-Host", "X-Forwarded-Proto",
	"X-Real-IP", "Forwarded", "True-Client-IP",
}

// requestViaProxy reports whether the request carries a forwarding/edge header,
// meaning an on-box proxy (e.g. cloudflared) may be relaying an off-box caller.
// These headers are client-spoofable, but presence can only REMOVE a peer-address
// exemption, so it fails closed.
func requestViaProxy(r *http.Request) bool {
	for _, h := range proxyHeaders {
		if _, ok := r.Header[http.CanonicalHeaderKey(h)]; ok {
			return true
		}
	}
	return false
}

// requestFromLAN reports whether the request's peer is on a private/local network
// interface (RFC 1918 IPv4, link-local IPv4/IPv6, unique-local IPv6, or loopback).
func requestFromLAN(r *http.Request) bool {
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	host = strings.Trim(host, "[]")
	if zone := strings.IndexByte(host, '%'); zone >= 0 {
		host = host[:zone]
	}
	ip := net.ParseIP(host)
	return ip != nil && (ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast())
}

// readScopedPath reports whether the path is one of the read-only observability
// surfaces — the set the loopback exemption opens, and the same set the read-scoped
// bearer may unlock off-loopback. Both callers share this one predicate so the two
// grants can never drift into disagreeing about what "read-only" means: adding a
// surface here widens both at once, which is the intent.
func readScopedPath(r *http.Request) bool {
	switch r.URL.Path {
	case "/v1/fak/features/proof", "/metrics", "/debug/vars", "/v1/fak/observation", "/v1/fak/observation/requests", "/v1/fak/observation/engine", "/v1/fak/perf/recent", "/v1/fak/arms", "/v1/fak/arms/traffic",
		// /props and /slots are the llama-server-shaped engine introspection
		// pair, served from the same live state as /metrics and carrying the
		// same class of information: counts, ratios, and build labels. They join
		// /metrics here so the read-scoped floor cannot drift — an engine that
		// admits /metrics must admit the introspection that explains it.
		"/props", "/slots":
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
	// Control directives use only their declared bearer and X-Api-Key doors.
	// A query-string secret can be logged or copied with a URL; Gemini's
	// X-Goog-Api-Key scheme is not a credential for this operator route.
	if isControlDirectivePath(r.URL.Path) {
		return "", false
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
	turnCostBegan := time.Now()
	if s.checkWarmupPending(w) {
		return
	}
	if dl := s.ScalarConfig().CompletionDeadlineMs; dl > 0 {
		ctx, cancel := context.WithTimeout(r.Context(), time.Duration(dl)*time.Millisecond)
		defer cancel()
		r = r.WithContext(ctx)
	}
	releaseEPFanout, waitEPFanout, ok := s.prepareChatEPFanout(w, r, epRouteChatCompletions)
	if !ok {
		return
	}
	defer waitEPFanout()
	var req ChatRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}
	normalizeChatMaxTokens(&req)
	req.AffinityKey = extractAffinityKey(r, req.AffinityKey)
	if !validateChatRequestIngress(w, req) {
		return
	}
	// Bind the client deadline before routing or preparation. Cache credit is
	// selected only after the execution route is fixed below.
	r, cancelDeadline := bindClientDeadline(r, turnCostBegan)
	defer cancelDeadline()
	// Retain served-prefix history as observation only. Admission below uses
	// current request-owned native state or measured residency evidence.
	r = withDeadlineCacheObservation(r, req.Model, req.Tools, req.Messages)
	deadlineCtx := r.Context()
	// Stamp the causal input on the untouched wire envelope before admission
	// transforms, request routing, planner selection, or model execution.
	inputTriggerRoute, routedModel, err := s.admitAndRouteChatInputTriggerWithContext(r.Context(), req)
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
	// Request-invariant system+tools head so parent and subagent requests share a prefix-KV hit.
	req.Messages, req.Tools = normalizeHarnessPrefix(req.Messages, req.Tools)
	r, ok = s.prepareChatRoute(w, r, req.Model)
	if !ok {
		return
	}
	if s.refuseNativeModelMismatch(w, r, req.Model, routedModel) {
		return
	}
	r, releaseDeadline, ok := s.admitRoutedClientDeadlineModel(w, r, turnCostBegan, req.Model, req.Messages, req.MaxTokens)
	if !ok {
		return
	}
	defer releaseDeadline()
	if !releaseEPFanout(r) {
		return
	}
	receiptRequested := req.Fak != nil && req.Fak.NativeInferenceReceipt
	decodeTraceRequested := req.FakDecodeTrace
	decodeTokenIDsRequested := req.Fak != nil && req.Fak.NativeDecodeTokenIDs
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
	applyChatCompletionSpeculativeHeaders(w, s, reqModel)
	if req.Stream && s.isVulkanMTPEnabled(reqModel) && chatRequestVulkanMTPEligible(req) {
		// The buffered in-kernel stream opens before decode completes. Declare
		// trailers now, then populate them from the request-local execution
		// receipt after completion; configuration alone never claims MTP ran.
		w.Header().Add("Trailer", HeaderSpeculative)
		w.Header().Add("Trailer", HeaderSpeculativeDowngrade)
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
	sessionTurn.turnCost = newTurnCostRecord(req.Stream)
	defer func() {
		sessionTurn.complete()
		s.finishTurnCost(sessionTurn, reqTrace, reqModel, turnCostBegan)
	}()
	req.Messages = messages
	// Oversize tool results page out to restore stubs only for a client that can call
	// the restore tool; the decision rides ctx into both admission passes.
	paging := resolveChatCtxPaging(r, req.Tools)
	w.Header().Set(HeaderCtxPaging, paging.header)
	ctx = ctxmmu.WithOversizePagingSuppressed(ctx, paging.suppressReason)
	resultAdmissions, err := s.admitInboundResults(ctx, req.Messages, req.Tools, reqTrace)
	if err != nil {
		writeErr(w, http.StatusBadGateway, "upstream cache invalidation failed")
		return
	}

	// Microcontext tool result elision (#11648)
	restoreToolName, restoreToolPresent, autoAdvertise := determineChatRestoreTool(req.Tools)
	req.Messages = s.maybeElideResponsesToolResults(reqTrace, req.Messages, restoreToolName)
	if len(req.Tools) > 0 && !restoreToolPresent && (autoAdvertise || hasElidedToolResult(req.Messages)) {
		req.Tools = append(req.Tools, agent.ToolDef{
			Type: "function",
			Function: agent.ToolDefFunction{
				Name:        restoreToolName,
				Description: "Restore dropped context by content-addressed sha256 id. Returns verbatim stashed bytes plus orientation; optional trace_id defaults to the current guarded session. Read-only and trust-gated.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"id":{"type":"string","description":"the content-address handle (sha256 hex) a compaction tombstone embedded as id=<hex>, or a recall page digest"},"trace_id":{"type":"string","description":"session trace id; omitted uses the gateway default trace"}},"required":["id"]}`),
			},
		})
	}
	ctx = withClientToolSchemas(ctx, req.Tools)

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
		stream = newChatStreamWriter(w, reqModel, req.DeclaredStreamUsage())
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
		agent.WithClientTemperature(req.Temperature),
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
		agent.WithChatWireTopK(req.TopK),
		agent.WithMinP(req.MinP),
		agent.WithChatTemplateKwargs(req.ChatTemplateKwargs),
		agent.WithNativeInferenceReceipt(receiptRequested),
		agent.WithDecodeTrace(decodeTraceRequested),
		agent.WithNativeDecodeTokenIDs(decodeTokenIDsRequested),
	)
	if err != nil {
		if comp != nil {
			applyVulkanMTPExecutionHeaders(w, comp.VulkanMTP)
		}
		s.renderTurnDebugError(reqTrace, "openai_chat_completions", err, time.Since(began))
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
			stream.failFields(status, code, msg, s.upstreamErrorFields(err, code))
			return
		}
		s.writeUpstreamErr(w, err)
		return
	}
	applyVulkanMTPExecutionHeaders(w, comp.VulkanMTP)
	if receiptRequested {
		applyNativeBatchReceiptHeaders(w, comp.InKernelBatch)
	}

	asst := comp.Message
	asst.Role = agent.RoleAssistant

	if !s.validateChatCompletionConformance(w, stream, comp, asst, receiptRequested, decodeTraceRequested, decodeTokenIDsRequested, inputTriggerRoute) {
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
	s.recordDeadlineWarmPrefix(deadlineCtx, comp.Usage)
	stampTurnCost(sessionTurn.turnCost, reqTrace, respModel, turnCostBegan)
	resp := s.buildChatResponse(comp, asst, finish, respModel, adjs, resultAdmissions, inputTriggerRoute, decodeTraceRequested, decodeTokenIDsRequested, sessionTurn.turnCost)
	if stream != nil {
		timePhase(sessionTurn.turnCost, turncost.PhaseStream, func() {
			writeChatCompletionStream(stream, resp)
		})
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func validateChatRequestIngress(w http.ResponseWriter, req ChatRequest) bool {
	if rejectStopLimits(w, normalizeStop(req.Stop)) {
		return false
	}
	if rejectGuidedRegexLimits(w, req) {
		return false
	}
	if len(req.Messages) == 0 {
		writeErr(w, http.StatusBadRequest, "messages: field required")
		return false
	}
	if rejectInvalidSampling(w, validateSampling(req)) {
		return false
	}
	if rejectInvalidResponseFormatCarrier(w, req.ResponseFormat) {
		return false
	}
	receiptRequested := req.Fak != nil && req.Fak.NativeInferenceReceipt
	decodeTraceRequested := req.FakDecodeTrace
	decodeTokenIDsRequested := req.Fak != nil && req.Fak.NativeDecodeTokenIDs
	if decodeTokenIDsRequested && !decodeTraceRequested {
		writeErr(w, http.StatusBadRequest, "native decode token IDs require fak_decode_trace")
		return false
	}
	if decodeTraceRequested && req.Stream {
		writeErr(w, http.StatusBadRequest, "fak_decode_trace requires a buffered fak-native request")
		return false
	}
	if receiptRequested && req.Stream {
		writeErr(w, http.StatusBadRequest, "native inference receipts require a buffered request")
		return false
	}
	if receiptRequested && ((req.Temperature != nil && *req.Temperature != 0) || (req.TopP != nil && *req.TopP != 0) || len(req.LogitBias) > 0 || (req.FrequencyPenalty != nil && *req.FrequencyPenalty != 0) || (req.PresencePenalty != nil && *req.PresencePenalty != 0)) {
		writeErr(w, http.StatusBadRequest, "native inference receipts require greedy sampling over unmodified logits")
		return false
	}
	return true
}

func determineChatRestoreTool(tools []agent.ToolDef) (restoreToolName string, present bool, autoAdvertise bool) {
	for _, t := range tools {
		if t.Function.Name == "mcp__fak_guard__fak_context_restore" {
			return "mcp__fak_guard__fak_context_restore", true, false
		}
	}
	for _, t := range tools {
		if t.Function.Name == "mcp__fak__fak_context_restore" {
			return "mcp__fak__fak_context_restore", true, false
		}
	}
	for _, t := range tools {
		if t.Function.Name == "fak_context_restore" {
			return "fak_context_restore", true, false
		}
	}
	for _, t := range tools {
		if isRestoreTool(t.Function.Name) {
			return t.Function.Name, true, false
		}
	}
	for _, t := range tools {
		if strings.HasPrefix(t.Function.Name, "mcp__fak_guard__") {
			return "mcp__fak_guard__fak_context_restore", false, true
		}
		if strings.HasPrefix(t.Function.Name, "mcp__fak__") {
			return "mcp__fak__fak_context_restore", false, true
		}
	}
	return "fak_context_restore", false, false
}

func hasElidedToolResult(messages []agent.Message) bool {
	for _, m := range messages {
		if strings.HasPrefix(m.Content, "...[fak: tool output elided") {
			return true
		}
	}
	return false
}

func (s *Server) validateChatCompletionConformance(w http.ResponseWriter, stream *chatStreamWriter, comp *agent.Completion, asst agent.Message, receiptRequested, decodeTraceRequested, decodeTokenIDsRequested bool, inputTriggerRoute *InputTriggerRouteReceipt) bool {
	if comp.ToolCallsDropped && len(asst.ToolCalls) == 0 {
		s.logf("gateway: upstream announced tool_calls but none parsed (conformance fail-closed); model=%s reason=%s", s.model, toolCallDropReasonName(comp))
		if msg, code, typed := toolCallDropRefusal(comp); typed {
			if stream != nil {
				stream.fail(http.StatusBadGateway, code, msg)
				return false
			}
			writeErrCode(w, http.StatusBadGateway, code, msg)
			return false
		}
		const conformanceMsg = "upstream tool-call format not recognized; refusing to skip adjudication"
		if stream != nil {
			stream.fail(http.StatusBadGateway, "", conformanceMsg)
			return false
		}
		writeErr(w, http.StatusBadGateway, conformanceMsg)
		return false
	}
	if receiptRequested && comp.NativeInference == nil {
		writeErr(w, http.StatusBadGateway, "configured planner cannot produce a native inference receipt")
		return false
	}
	if inputTriggerRoute != nil && comp.NativeInference != nil &&
		(comp.NativeInference.Engine != TurnIngressEngine ||
			comp.NativeInference.Model != TurnIngressModel ||
			comp.NativeInference.FallbackActive) {
		s.logf("gateway: input-trigger route execution identity mismatch")
		writeErr(w, http.StatusBadGateway, "fak-native route execution identity mismatch")
		return false
	}
	if decodeTraceRequested && comp.DecodeTrace == nil {
		writeErr(w, http.StatusBadGateway, "fak-native planner did not produce a decode trace")
		return false
	}
	if decodeTokenIDsRequested && comp.NativeDecodeTokenIDs == nil {
		writeErr(w, http.StatusBadGateway, "fak-native planner did not produce native decode token IDs")
		return false
	}
	return true
}

// applyNativeBatchReceiptHeaders emits the additive native-cohort headers from
// the coordinator's own batch receipt (#13317). The set is atomic: it is
// written only when the cohort id is non-empty and header-safe, the cohort
// size is positive, and the shared-panel count is non-negative. An absent or
// malformed receipt writes NO header, so a downstream benchmark always sees
// either the complete evidence triple or "unknown" — never a partial set that
// requested concurrency could be mistaken for. SharedPanels is written even
// when zero: a present 0 is authoritative serial/no-shared-panel evidence,
// distinct from an absent header. CohortSize is coordinator membership; this
// function never labels it a GPU batch width.
func applyNativeBatchReceiptHeaders(w http.ResponseWriter, receipt *agent.InKernelBatchReceipt) {
	if w == nil || receipt == nil {
		return
	}
	cohortID := strconv.FormatUint(receipt.CohortID, 10)
	if receipt.CohortID == 0 || !headerSafeValue(cohortID) {
		return
	}
	if receipt.CohortSize <= 0 || receipt.SharedPanels < 0 {
		return
	}
	h := w.Header()
	h.Set(HeaderNativeCohortID, cohortID)
	h.Set(HeaderNativeCohortSize, strconv.Itoa(receipt.CohortSize))
	h.Set(HeaderNativeSharedPanels, strconv.Itoa(receipt.SharedPanels))
}

// headerSafeValue reports whether a value is safe to place in a response
// header: non-empty, ASCII printable, and free of control characters. The
// coordinator's cohort id is numeric today, so this is a fail-closed guard
// against a future non-numeric or injected value reaching the wire.
func headerSafeValue(v string) bool {
	if v == "" {
		return false
	}
	for i := 0; i < len(v); i++ {
		if c := v[i]; c < 0x20 || c > 0x7e {
			return false
		}
	}
	return true
}

func extractAffinityKey(r *http.Request, explicit string) string {
	if s := strings.TrimSpace(explicit); s != "" {
		return s
	}
	if r != nil {
		if s := strings.TrimSpace(r.Header.Get("X-Session-ID")); s != "" {
			return s
		}
	}
	return ""
}

func (s *Server) buildChatResponse(comp *agent.Completion, asst agent.Message, finish, respModel string, adjs []ToolAdjudication, resultAdmissions []ResultAdmission, inputTriggerRoute *InputTriggerRouteReceipt, decodeTraceRequested, decodeTokenIDsRequested bool, turnCost *turncost.TurnCostRecord) ChatResponse {
	resp := ChatResponse{
		ID:      "chatcmpl-fak-" + itoa(uint64(time.Now().UnixNano())),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   respModel,
		Choices: []ChatChoice{{Index: 0, Message: asst, FinishReason: finish}},
		Usage:   comp.Usage,
		Timings: comp.Timings,
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
		// The per-turn cost record rides the native receipt (#924): opt-in, so an
		// ordinary turn's response bytes are unchanged whether the surface is on or off.
		if turncost.Enabled() && turnCost != nil {
			resp.Fak.TurnCost = turnCost
		}
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
	return resp
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
	// A refused OpenAI call can coexist with model prose. Preserve that prose,
	// but make the refusal visible to clients that ignore the fak extension;
	// otherwise a no-tool stop looks like completed work to their harness.
	if anyLivelock(adjs) || (dropped > 0 && len(kept) == 0 && asst.Content != "") {
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

// handleFakTokenize prepares the exact native prompt without entering decode,
// admission, session mutation, or a fallback planner. Planners that do not
// expose native prompt encoding fail closed.
func (s *Server) handleFakTokenize(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	encoder, ok := s.planner.(nativePromptEncoder)
	if !ok {
		writeErr(w, http.StatusNotImplemented, "native prompt encoding unavailable")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxTokenizeBody)
	decoder := json.NewDecoder(r.Body)
	var req ChatRequest
	if err := decoder.Decode(&req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	normalizeChatMaxTokens(&req)
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}

	servedModel := strings.TrimSpace(s.model)
	if requested := strings.TrimSpace(req.Model); requested != "" && requested != servedModel {
		writeErrCode(w, http.StatusBadRequest, "model_mismatch", "requested model does not match the loaded native model")
		return
	}
	encoding, err := encoder.EncodePrompt(r.Context(), req.Messages, req.Tools, nativeTokenizeSampleOpts(req)...)
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "native prompt encoding failed")
		return
	}
	if strings.TrimSpace(encoding.ModelID) != servedModel {
		writeErr(w, http.StatusInternalServerError, "native prompt encoding model identity mismatch")
		return
	}
	writeJSON(w, http.StatusOK, encoding)
}

func nativeTokenizeSampleOpts(req ChatRequest) []agent.SampleOpt {
	return []agent.SampleOpt{
		agent.WithModel(req.Model),
		agent.WithMaxTokens(req.MaxTokens),
		agent.WithTemperature(req.Temperature),
		agent.WithTopP(req.TopP),
		agent.WithStop(normalizeStop(req.Stop)),
		agent.WithResponseFormat(req.ResponseFormat),
		agent.WithToolChoice(req.ToolChoice),
		agent.WithLogitBias(req.LogitBias),
		agent.WithGuidedDecode(req.GuidedDecodeFields()),
		agent.WithFrequencyPenalty(req.FrequencyPenalty),
		agent.WithPresencePenalty(req.PresencePenalty),
		agent.WithReasoningEffort(req.ReasoningEffort),
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
