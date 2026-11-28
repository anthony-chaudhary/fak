package gateway

import (
	"context"
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
	"github.com/anthony-chaudhary/fak/pkg/deploykit/probe"
	"github.com/anthony-chaudhary/fak/pkg/gatewayauth"
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
// request body legitimately grows past the 4 MiB tool-args cap â€” a 388k-token
// resume serializes to several MiB of JSON. 32 MiB matches the real Anthropic
// request-body ceiling, so the gateway never refuses a body the upstream would
// have accepted (the silent-truncation 400 in #-resume).
const maxTranscriptBody = 32 << 20

// gatewayRoute pairs a ServeMux registration pattern with its handler. Handler
// builds the mux from routeTable() rather than a sequence of inline HandleFunc
// calls so that the served HTTP surface has a single, enumerable source of
// truth â€” which the OpenAPI spec drift gate (openapi_spec_test.go) ranges over
// to assert docs/fak/openapi.yaml documents every route (#205, F-007: the spec
// the client SDKs are generated from must not drift behind the served surface).
type gatewayRoute struct {
	pattern string
	handler http.HandlerFunc
}

// routeTable is the canonical, ordered list of the gateway's HTTP routes â€” the
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
		// Legacy OpenAI text-completion wire â€” the pre-chat surface vLLM/SGLang/
		// llama.cpp-server all still serve, for older clients and eval harnesses. No
		// tools on this wire, so no tool-call adjudication; adapts onto the same served
		// completion path as the chat route.
		{"/v1/completions", s.handleCompletions},
		// OpenAI Responses API â€” a client-facing inbound route so a Responses-native
		// agent (Codex CLI, the Terminal-Bench terminus agent) can route its model
		// traffic through the kernel's tool-call adjudication, the same as the chat
		// wire. Buffered only; stream:true is refused (#925).
		{"/v1/responses", s.handleResponses},
		// GET / DELETE /v1/responses/{id} over the same owner-scoped store
		// previous_response_id reads.
		{"/v1/responses/", s.handleResponsesRetrieve},
		{"/v1/embeddings", s.handleEmbeddings},
		{"/v1/moderations", s.handleModerations},
		// /v1/decisions is the single-choice decision-probability surface: one
		// validated option set (unique ids, A-Z labels, temperature > 0), engine-
		// scored logits through the installed scoring hook, stable softmax out.
		// Inert (404 decisions_hook_unwired) until a host installs the hook via
		// InstallDecisionsScorer. See decisions.go.
		{"/v1/decisions", s.handleDecisions},
		// Anthropic Messages surface.
		{"/v1/messages", s.handleAnthropicMessages},
		{"/v1/messages/count_tokens", s.handleAnthropicCountTokens},
		// Native Gemini generateContent surface (/v1beta/models/{model}:{method}).
		{"/v1beta/", s.handleGeminiGenerateContent},
		// fak-native surface â€” one POST, one verdict.
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
		// /v1/fak/usage/ledger is the billing-grade per-request usage export
		// (#14451): last-N rows plus whole-file per-principal token sums and the
		// chain verdict. GET, read-only, counts only.
		{"/v1/fak/usage/ledger", s.handleFakUsageLedger},
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
		// (a longer, exact pattern wins) â€” a session whose id is literally "changes"
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
		// /v1/fak/perf/phases is the per-request native phase receipt read
		// (fak-request-phases/1 JSONL) joined to perf rows by request_id.
		{"/v1/fak/perf/phases", s.handleFakPerfPhases},
		{"/v1/fak/fleet", s.handleFakFleet},
		{"/v1/fak/capacity/tiers", s.handleFakCapacityTiers},
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
		// events â€” session.start, per-call adjudicated `call` rows, session.end with
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
		// swap function via SetAccountRehomeFunc â€” fak guard does, on the pinned
		// Claude-subscription path. See account_rehome.go.
		{"/v1/fak/account/rehome", s.handleFakAccountRehome},
		{"/v1/models", s.handleModels},
		// Multi-node dev-server READ plane (#2297, epic #2254 plane 1): the
		// coordinator clone's live lease view (the dos_arbitrate live_leases
		// projection) and presence (session descriptors + lease liveness
		// classification), observed from refs/fak/locks/* at request time.
		// Read-only; injected by the host CLI (SetLeasePlaneProviders), 404 when
		// unwired. /v1/sessions is the cross-machine guard-session presence view â€”
		// distinct from /v1/fak/sessions, the served-session DRIVE-state snapshot.
		{"/v1/leases", s.handleLeases},
		// Multi-node dev-server WRITE plane (#2299, epic #2254 plane 1 â€” the atomicity
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
		{"/v1/serve/status", s.handleServeStatus},
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
		s.logf("WARNING: binding %s with NO --require-key set â€” the kernel gateway is exposed without authentication", addr)
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
// â€” a test binding 127.0.0.1:0, or a host handing fak a pre-opened socket â€” binds
// its own listener and calls Serve directly. It mirrors net/http.Server's
// ListenAndServe/Serve split.
func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	// The drain deadline derives from the SAME write timeout the handlers run
	// under, so an in-flight stream that legitimately runs up to WriteTimeout
	// finishes instead of being cut by a fixed 5s on every rolling update
	// (#14454). FAK_HTTP_DRAIN_TIMEOUT_S still overrides either way.
	writeTimeout := durEnv("FAK_HTTP_WRITE_TIMEOUT_S", serveWriteTimeoutDefault(plannerKind(s.planner)))
	drainTimeout := parseHTTPDrainTimeout(os.Getenv("FAK_HTTP_DRAIN_TIMEOUT_S"), defaultHTTPDrainTimeout(writeTimeout))
	// Record the address we actually bound so a served descriptor can name this
	// process instead of a literal (#5642). This is the ONLY point where the chosen
	// address is known â€” with an ephemeral ":0" bind the port does not exist until
	// the listener does â€” and both entry points funnel through here.
	if a := ln.Addr(); a != nil {
		addr := a.String()
		s.boundAddr.Store(&addr)
	}
	// Bounded timeouts so a single slow/idle connection cannot pin a goroutine +
	// socket indefinitely (slow-loris-on-body / idle-keepalive DoS). ReadTimeout
	// also caps body-delivery TIME (MaxBytesReader only caps SIZE).
	//
	// WriteTimeout bounds the WHOLE handler measured from the end of the request
	// headers â€” and a NON-streaming turn writes the body only AFTER the model finishes,
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
		WriteTimeout:      writeTimeout,
		IdleTimeout:       durEnv("FAK_HTTP_IDLE_TIMEOUT_S", 120*time.Second),
		// Route net/http's own diagnostics (per-request panic recovery, TLS/keepalive
		// errors) through the SAME sink as every other gateway log. Left nil, net/http
		// falls back to the std logger â†’ os.Stderr, which UNDER `fak guard` is the child
		// harness's controlling TTY: a recovered handler panic then dumps a multi-line
		// goroutine stack straight into the agent's TUI and corrupts the display (#2772).
		// s.logf already honors the operator's --log choice â€” muted by default to keep the
		// terminal clean, streamed to a file/stderr when asked â€” so binding ErrorLog to it
		// makes those diagnostics obey the same policy instead of bypassing it. Zero flags
		// so we don't double-stamp the timestamp s.logf already adds.
		ErrorLog: log.New(logfWriter{logf: s.logf}, "", 0),
	}
	if s.richDashboards != nil {
		defer s.richDashboards.close()
	}
	// Disable Nagle on accepted TCP connections. Without TCP_NODELAY the kernel
	// coalesces small writes (Nagle), adding 40-200ms of buffering on a high-RTT
	// link â€” felt on streamed chat-completion deltas and the small fak-native verdict
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
	// replica fleet's live health/drain loop (issue fak-private#2417) is one of them â€”
	// registered as "fleet-health" in newBgloopSupervisor, so it starts here and is
	// cancelled AND joined by stopLoops on shutdown.
	s.startLoops(ctx)
	if s.logf != nil {
		s.logf("fak gateway listening on http://%s (planner=%s engine=%s model=%s vdso=%v auth=%v)",
			ln.Addr(), plannerKind(s.planner), s.engineID, s.model, s.k.VDSOEnabled(), s.requireKey != "")
		if !s.warmup.pending() {
			s.logf("[READY] fak gateway is ready to accept requests on http://%s", ln.Addr())
		}
		// Surface fak's core value-add â€” realized in-kernel KV-prefix reuse â€” at startup so it
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

// defaultHTTPDrainTimeout derives the graceful-shutdown deadline from the handler
// write timeout, so an in-flight stream that legitimately runs up to WriteTimeout
// finishes instead of being cut by a fixed 5s on every rolling update (#14454).
// The +10s settle covers the tail of chunked delivery after the last token. A zero
// write timeout - a local in-kernel model whose turn can run for minutes - gets
// the longer floor; operators with multi-minute decodes raise
// FAK_HTTP_DRAIN_TIMEOUT_S explicitly.
func defaultHTTPDrainTimeout(writeTimeout time.Duration) time.Duration {
	if writeTimeout <= 0 {
		return 120 * time.Second
	}
	return writeTimeout + 10*time.Second
}

// parseHTTPDrainTimeout accepts positive whole seconds that fit in a Duration.
// Unset or invalid values retain def so shutdown always has a finite deadline.
func parseHTTPDrainTimeout(raw string, def time.Duration) time.Duration {
	n, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64)
	if err != nil || n <= 0 || n > int64((1<<63-1)/time.Second) {
		return def
	}
	return time.Duration(n) * time.Second
}

// logfWriter adapts the gateway's structured logf onto io.Writer so an http.Server.ErrorLog
// (which speaks io.Writer) drains into the same sink. net/http hands ErrorLog one whole
// pre-formatted message per line (a panic recovery arrives as a single multi-line write), so
// we forward it verbatim minus the trailing newline; when logf is the guard default no-op the
// message is dropped and nothing reaches the terminal. A nil logf is tolerated for the same
// reason the Serve path assumes it non-nil â€” belt-and-braces, never a nil-deref.
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
		return "idle â€” realized KV-prefix reuse appears here per in-kernel turn (scrape /metrics fak_gateway_kv_prefix_* for the full family)"
	}
	return fmt.Sprintf("reuse %.0f%% (saved=%d tok) over %d turns (frozen=%d partial=%d cold=%d) â€” WITNESSED, by=vdso",
		s.ReuseRatio*100, s.ReusedTokens, s.Turns, s.FrozenTurns, s.PartialTurns, s.ColdTurns)
}

// nodelayListener wraps ln so every accepted *net.TCPConn has Nagle disabled
// (TCP_NODELAY). It is a pass-through for a listener whose Accept does not yield a
// *net.TCPConn â€” a test's in-memory pipe or a Unix socket â€” so wrapping is always
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
		// connection â€” let the handler proceed and surface any real error on use.
		_ = tc.SetNoDelay(true)
	}
	return c, nil
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
	if requested := strings.TrimSpace(req.Model); requested != "" && !sameServedModel(requested, servedModel) {
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
		agent.WithMultimodalMode(req.DeclaredMultimodalMode()),
	}
}
