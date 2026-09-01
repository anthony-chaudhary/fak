// Package gateway is the kernel-adjudicated wire: it fronts the fak kernel over
// MCP (newline-delimited JSON-RPC) and an OpenAI-compatible HTTP surface so an
// agent written in ANY language can route its tool calls through the in-process
// syscall boundary WITHOUT writing Go.
//
// Direction (DIRECTION.md). The gateway is Go and sits ON the request path — it
// adjudicates every call through the existing abi.Kernel. That is in-direction:
// the typed core does the deciding. It adds NO non-Go surface in-tree; the non-Go
// CLIENT lives in the adopter's repo. Everything crossing the wire is untrusted
// input that typed Go re-validates before it reaches the kernel — the same
// posture the policy loader takes toward a manifest and the kernel takes toward a
// tool result. A wire client NEVER supplies an abi.Ref (a content-addressed CAS
// handle); it supplies raw argument bytes and the gateway mints a tainted,
// agent-scoped Ref itself, so the IFC/secret/self-modify rungs stay armed.
//
// The three operations, all funnelling into one adjudication helper:
//
//	fak_adjudicate  — k.Decide only (no dispatch, no pending state): the pre-exec
//	                  verdict a client-side executor asks for BEFORE running a tool.
//	fak_syscall     — k.Syscall (adjudicate + dispatch to the registered engine +
//	                  context-MMU admit): the self-contained / CI / demo path.
//	/v1/chat/completions — an adjudication PROXY: it forwards the chat to an
//	                  upstream model, then runs each PROPOSED tool_call through
//	                  k.Decide, dropping denied calls and rewriting grammar-repaired
//	                  ones before the caller ever sees them. It does NOT execute the
//	                  client's tools (the client does).
package gateway

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/auditreceipt"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/enginecache"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/rungobs"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// New builds a Server. It validates that the ABI is wired (a resolver is
// registered — i.e. internal/registrations was imported) and that EngineID names
// a registered engine. It fails loud rather than degrade to a permissive default.
func New(cfg Config) (*Server, error) {
	// fak_read is part of this server's default MCP inventory, so arm its confined
	// execution route before the server can advertise the tool. Preserve any caller-
	// supplied narrower root; only the missing default route needs startup repair.
	if abi.Engine(agent.FakReadEngineID) == nil {
		agent.RegisterReadEngine("")
	}
	if abi.ActiveResolver() == nil {
		return nil, errors.New("gateway: no Ref resolver registered (blank-import internal/registrations before New)")
	}
	engineID := cfg.EngineID
	if engineID == "" {
		engineID = "inkernel"
	}
	if !engineRegistered(engineID) {
		return nil, fmt.Errorf("gateway: engine %q is not registered (have: %s)", engineID, strings.Join(abi.EngineIDs(), ", "))
	}
	// A misconfigured routing policy is a security boundary (it decides which model
	// — local or remote — a tenant payload reaches), so validate it at New and fail
	// loud rather than fall through to a silent default model at dispatch time.
	if cfg.RouteManifest != nil {
		if err := cfg.RouteManifest.Validate(); err != nil {
			return nil, fmt.Errorf("gateway: route manifest: %w", err)
		}
	}
	// A misconfigured account roster is the SAME class of security boundary: it decides
	// which provider/account (local or remote) a routed model — and thus a tenant payload
	// — reaches. Validate at New and fail loud rather than fall through to a silent
	// mis-bind or a residency-floor bypass at dispatch time (#2528).
	if cfg.RouteAccounts != nil {
		if err := cfg.RouteAccounts.Validate(); err != nil {
			return nil, fmt.Errorf("gateway: route accounts: %w", err)
		}
	}
	// The MCP tool-exposure allowlist (--expose) narrows which tools are advertised
	// AND callable. It is the same class of boundary as the route manifest — it
	// decides what a client can reach — so compile-and-validate it here and fail
	// loud on a malformed or zero-match pattern rather than let a typo silently
	// shrink the surface to nothing at first connect. Empty ⇒ nil ⇒ full surface.
	exposeAllow, err := compileToolExposeAllow(cfg.ExposeTools)
	if err != nil {
		return nil, err
	}
	model := cfg.Model
	if model == "" {
		model = engineID
	}
	logf := cfg.Logf
	if logf == nil {
		logf = func(format string, args ...any) { log.Printf(format, args...) }
	}
	version := cfg.Version
	if version == "" {
		version = "dev"
	}

	// Boot timeline: start from the host-supplied process-start instant (so phases
	// the CLI timed before New are on the same clock) and append each New-internal
	// phase as we complete it.
	startup := newStartupProfile(cfg.StartTime)
	for _, ph := range cfg.StartupPhases {
		startup.phaseWithProvenance(ph.Name, ph.Dur, ph.Provenance)
	}

	proxyURLs, err := proxyBaseURLs(cfg)
	if err != nil {
		return nil, err
	}

	planner, servedSide, upstreamSide, inKernelModelButChatIsMock, err := selectChatPlanner(cfg, model, proxyURLs, logf, startup)
	if err != nil {
		return nil, err
	}

	remoteCache, err := newEngineCacheClient(cfg)
	if err != nil {
		return nil, err
	}

	// Select the live fleet's tier-2 invalidation granularity (process-global vDSO).
	// Fail loud on an unknown name rather than silently degrading to a full flush.
	t := time.Now()
	if g, ok := vdso.ParseGranularity(cfg.Invalidation); ok {
		vdso.Default.SetGranularity(g)
	} else {
		return nil, fmt.Errorf("gateway: unknown invalidation granularity %q (want global|namespace|resource)", cfg.Invalidation)
	}
	startup.phase("vdso-config", time.Since(t))

	t = time.Now()
	k := kernel.New(engineID)
	k.SetVDSO(cfg.VDSO)
	startup.phase("kernel-init", time.Since(t))

	// Unified cache-stream observability: subscribe the live tier-2 cache-event sink
	// of the process-global vDSO (the SAME instance writeVDSOMetrics reads Stats from)
	// so every fill/hit/evict/revoke folds into the fak_cache_* family. The sink fires
	// OUTSIDE the vDSO lock and Observe only takes its own cheap lock, so it never
	// blocks the hot path. Close detaches it. This is the gateway's single production
	// consumer of the sink (only tests set it otherwise), so owning it is safe.
	cacheStream := cachemeta.NewStreamMetrics()
	vdso.Default.SetCacheEventSink(func(ev vdso.CacheEvent) {
		cacheStream.Observe(string(ev.Kind), ev.Entry)
	})

	// Passive rung-decision observability (issue #693): register a rungobs Emitter that
	// folds the kernel's verdict stream into a per-(rung,kind,reason) histogram,
	// exposed on /metrics as fak_kernel_decisions_total. It subscribes to ONLY
	// EvDecide/EvDeny/EvVDSOHit, so it adds zero work to the every-syscall event path,
	// and it is passive (re-folds off the hot path; never mutates verdict or Counters).
	rungObs := rungobs.New()
	abi.RegisterEmitter(rungObs)

	// The ctxplan view planner is OFF unless the host set a resident-token budget. nil
	// leaves maybePlanMessages an inert identity (the byte-for-byte-unchanged guard).
	var ctxView *agent.CtxViewPlanner
	if cfg.CtxViewBudget > 0 {
		ctxView = &agent.CtxViewPlanner{Enabled: true, Budget: cfg.CtxViewBudget}
	}
	var admissionCtl *AdmissionController
	if cfg.InKernelModel != nil && cfg.Tokenizer != nil && len(proxyURLs) == 0 {
		admissionCtl = NewAdmissionController(DefaultAdmissionPolicy())
	}

	var nativeCodeCatalog []agent.ToolDef
	if cfg.NativeCodeWorkspace != "" {
		var armErr error
		nativeCodeCatalog, armErr = agent.ArmFocusedCodeTools(cfg.NativeCodeWorkspace)
		if armErr != nil {
			return nil, fmt.Errorf("native code workspace: %w", armErr)
		}
	}

	otlpCapacity := cfg.OTLPQueueCapacity
	if otlpCapacity == 0 {
		otlpCapacity = 256
	}
	otlpTimeout := cfg.OTLPTimeout
	if otlpTimeout == 0 {
		otlpTimeout = 2 * time.Second
	}
	otlp, err := newOTLPExporter(cfg.OTLPEndpoint, otlpCapacity, otlpTimeout)
	if err != nil {
		return nil, err
	}
	orgAudit, err := auditreceipt.New(cfg.OrgAudit)
	if err != nil {
		return nil, err
	}

	s := &Server{
		k:                            k,
		toolPlugins:                  append([]toolplugin.Plugin(nil), cfg.ToolPlugins...),
		toolPreferences:              cfg.ToolPreferences,
		engineID:                     engineID,
		model:                        model,
		servedSide:                   servedSide,
		upstream:                     upstreamSide,
		requireKey:                   cfg.RequireKey,
		readBearer:                   cfg.ReadBearer,
		keyset:                       newKeyset(cfg.KeyPrincipals),
		exposeUpstreamErrorDetail:    cfg.ExposeUpstreamErrorDetail,
		denialRecoveryOff:            cfg.DenialRecoveryOff,
		upstreamBadRequestNotify:     cfg.UpstreamBadRequestNotify,
		version:                      version,
		logf:                         logf,
		debugStatsf:                  cfg.DebugStatsf,
		reloadPolicy:                 cfg.ReloadPolicy,
		policyCanaryTurns:            cfg.PolicyCanaryTurns,
		observePolicy:                cfg.ObservePolicy,
		resetTrace:                   cfg.ResetTrace,
		observeTrace:                 cfg.ObserveTrace,
		observeSession:               cfg.ObserveSession,
		controlSession:               cfg.ControlSession,
		steerSession:                 cfg.SteerSession,
		listSessions:                 cfg.ListSessions,
		trajctlMetrics:               cfg.TrajctlMetrics,
		decideSession:                cfg.DecideSession,
		stopGate:                     cfg.StopGate,
		debitSession:                 cfg.DebitSession,
		resetOnBudget:                cfg.ResetOnBudget,
		budgetDrained:                cfg.OnBudgetExhausted,
		defaultTraceID:               strings.TrimSpace(cfg.DefaultTraceID),
		guardRecoveryPrompt:          strings.TrimSpace(cfg.GuardRecoveryPrompt),
		startup:                      startup,
		planner:                      planner,
		inKernelModelButChatIsMock:   inKernelModelButChatIsMock,
		engineCache:                  remoteCache,
		admissionCtl:                 admissionCtl,
		ctxView:                      ctxView,
		compactHistoryBudget:         cfg.CompactHistoryBudget,
		positiveResidualSubstitution: cfg.PositiveResidualSubstitution,
		autoCheckpoint:               cfg.AutoCheckpoint,
		ctxExpenseWarn:               ctxExpenseThresholdOr(cfg.CtxExpenseWarnTokens, ctxExpenseWarnTokensDefault),
		ctxExpenseBlock:              ctxExpenseThresholdOr(cfg.CtxExpenseBlockTokens, ctxExpenseBlockTokensDefault),
		ctxExpenseGate:               cfg.CtxExpenseGate || envEnabled("FAK_CTX_EXPENSE_GATE"),
		compactAnchorHead:            cfg.CompactAnchorHead,
		assumeSessionTurns:           cfg.AssumeSessionTurns,
		compactSolvencyFloorTokens:   cfg.CompactSolvencyFloorTokens,
		exposeProfile:                cfg.ExposeProfile,
		elideResultBytes:             ablateUncachedTrimBytes(cfg.ElideResultBytes),
		elideStaleReads:              cfg.ElideStaleReads,
		provider:                     strings.TrimSpace(cfg.Provider),
		prefixGuard:                  cfg.PrefixGuard || envEnabled("FAK_ABLATE_PREFIX_GUARD"),
		vcacheAnchor:                 cfg.VCacheAnchor || envEnabled("FAK_ABLATE_BP_PLAN"),
		vcacheCalibration:            cloneVCacheRuntimeCalibration(cfg.VCacheCalibration),
		toolFloorDenies:              cfg.ToolFloorDenies,
		exposeAllow:                  exposeAllow,
		deferMCPTools:                cfg.DeferMCPTools || envEnabled("FAK_DEFER_MCP_TOOLS"),
		deferColdTools:               cfg.DeferColdTools || envEnabled("FAK_DEFER_COLD_TOOLS"),
		cacheStream:                  cacheStream,
		rungObs:                      rungObs,
		observers:                    newObserverStratum(observerJournalPath(), logf),
		feed:                         newCoherenceFeed(0),
		sessionFeed:                  newSessionFeed(0),
		activity:                     newSessionActivity(),
		toolPages:                    ctxmmu.NewToolPageTable(nil), // nil ⇒ the process-global MMU pager (#2440)
		metrics:                      newGatewayMetrics(time.Now()),
		nativeReceiptMetrics:         nativeperf.NewReceiptMetrics(0),
		otlp:                         otlp,
		orgAudit:                     orgAudit,
		route:                        newRouteLive(cfg.RouteManifest),
		roster:                       cfg.RouteAccounts,
		native:                       cfg.Native,
		nativeMaxTurns:               nativeMaxTurnsOr(cfg.NativeMaxTurns),
		nativeCodeCatalog:            nativeCodeCatalog,
		nativeSpeculate:              cfg.NativeSpeculate,
		vdsoProxyFill:                cfg.VDSOProxyFill,
		maxAgeByTool:                 cfg.ToolMaxAge,

		pinUpstreamCredential: cfg.PinUpstreamCredential,
	}
	s.installRichDashboardManager(cfg.RichDashboards)

	// #4003: seed the model-routing hot-reload seam behind POST /v1/fak/route/reload.
	// The watcher is normally installed AFTER New via SetRouteWatcher (it needs the
	// server's live routing holder), so this is a no-op unless a host/test supplies a
	// pre-built watcher in the config. A nil watcher leaves the route disabled (404).
	if cfg.ReloadRoute != nil {
		s.routeWatcher.Store(cfg.ReloadRoute)
	}
	s.cacheTTL1H.Store(cfg.CacheTTL1H || envEnabled("FAK_ABLATE_TTL_1H"))

	// Wire retry observability onto the proxy planner (#793 follow-on): Complete's 429/5xx
	// backoff is otherwise invisible — up to ~8s of silent waiting. The hook bumps a retry
	// counter and prints a glanceable `fak-turn … retry` line to the default --debug-stats
	// sink, so an operator sees the backoff happening instead of a frozen terminal. Only the
	// direct HTTPPlanner carries the loop — unwrapped from a dual planner's proxy side, which
	// fronts the same upstream; the mock/in-kernel/replica planners don't, so this is a
	// no-op for them.
	if hp := unwrapHTTPPlanner(planner); hp != nil {
		hp.RetryNotify = s.onUpstreamRetry
		hp.AuthRefreshNotify = s.onAuthRefresh
		hp.ForbiddenRetryNotify = s.onForbiddenRetry
		hp.AccountFailoverNotify = s.onAccountFailover
	}

	// Build the in-kernel background-loop supervisor and register the built-in loops
	// (a liveness heartbeat). It is not running yet — Serve starts it on the lifecycle
	// context and joins it on shutdown, so the loops progress exactly while the kernel
	// is up.
	s.loops = newBgloopSupervisor(s)

	return s, nil
}

func (s *Server) installRichDashboardManager(cfg RichDashboardConfig) {
	s.richDashboards = newRichDashboardManager(cfg)
	s.richDashboards.listenerAddress = func() string {
		if addr := s.boundAddr.Load(); addr != nil {
			return *addr
		}
		return ""
	}
}

// selectChatPlanner picks the chat backend for the /v1/chat/completions and
// /v1/messages surfaces from the wired configuration — a dual (local-alongside-API)
// planner, a proxy planner, the in-kernel chat planner, or the deterministic mock —
// records the planner-init startup phase, and reports whether real weights are loaded
// for syscalls but chat still falls back to the mock (missing tokenizer, #1115).
func selectChatPlanner(cfg Config, model string, proxyURLs []string, logf func(string, ...any), startup *startupProfile) (agent.Planner, servingLocality, *servingLocality, bool, error) {
	var planner agent.Planner
	var err error
	t := time.Now()
	inKernelModelButChatIsMock := false
	// Which SIDE serves a turn under this deployment — the attribution the usage
	// ledger's self-hosted split needs and could not previously get. It is resolved
	// here, and only here, because this switch is the one place the four modes are
	// exhaustive by construction: a fifth branch added later must state its own side
	// or inherit the honest unknown, whereas a type-switch somewhere downstream would
	// quietly read a new planner as unclassified and shrink coverage without saying so.
	side := localityUnknown
	switch {
	case len(proxyURLs) != 0 && cfg.InKernelModel != nil && cfg.Tokenizer != nil:
		// DUAL (small local model ALONGSIDE the API upstream, dual_planner.go): a live
		// proxy AND a loaded in-kernel model in ONE gateway. Requests addressed to the
		// local model id (Config.LocalModelID, default "local") decode in-kernel with no
		// upstream call; every other request — including the wrapped agent's default
		// turns — proxies upstream byte-for-byte as the proxy-only path does.
		// Historically the proxy silently WON this combination and the loaded weights
		// were dead; now the combination is the alongside deployment.
		proxy, perr := newProxyPlanner(cfg, model, proxyURLs)
		if perr != nil {
			return nil, localityUnknown, nil, false, perr
		}
		localID := localModelIDOr(cfg.LocalModelID)
		planner, err = NewDualPlanner(proxy, newInKernelChatPlanner(cfg, localID, logf), localID)
		if err != nil {
			return nil, localityUnknown, nil, false, err
		}
		logf("gateway: dual planner — model id %q (and \"local\") decodes in-kernel; every other model id proxies upstream", localID)
		// Deliberately left unknown: dual is the one mode with no deployment-constant
		// side, because the side is a property of each REQUEST's model id. servedLocality
		// asks the planner itself (RoutesLocal) rather than guessing from a default here.
	case len(proxyURLs) != 0:
		planner, err = newProxyPlanner(cfg, model, proxyURLs)
		if err != nil {
			return nil, localityUnknown, nil, false, err
		}
		// Deliberately left unknown, for the same reason dual is: a proxying
		// deployment has no deployment-CONSTANT side. `--base-url` is how both
		// self-hosting rungs are reached in the common case — an on-box server and a
		// company-operated one — so reading the flag itself as "a third-party API"
		// told an operator who bought nothing that they bought everything. What the
		// upstream URL does establish is resolved by upstreamAttribution and rides on
		// Server.upstream, where a per-model roster binding can still override it;
		// servedLocality composes the two.
	case cfg.InKernelModel != nil && cfg.Tokenizer != nil:
		// Serve the model fused into the kernel as the chat backend on BOTH
		// /v1/chat/completions and /v1/messages (they share s.planner.Complete):
		// real ChatML chat via internal/tokenizer, the cmd/fakchat recipe factored
		// into a Planner. Falls through to MockPlanner if the host didn't preload.
		planner = newInKernelChatPlanner(cfg, model, logf)
		// Every turn decodes on this box against weights we host.
		side = localitySelfHosted
	default:
		// No upstream (--base-url) and no in-kernel model (--gguf/FAK_MODEL_DIR): the
		// chat surface silently fell back to the deterministic offline mock. Warn
		// LOUDLY so an operator never mistakes scripted demo text for real model
		// output — the /healthz planner:"mock" field carries the same signal to a
		// liveness probe.
		if cfg.InKernelModel != nil && cfg.Tokenizer == nil {
			// #1115: kernel has real weights loaded (for fak_syscalls) but chat
			// falls back to mock due to missing tokenizer. Flag for witness fidelity.
			inKernelModelButChatIsMock = true
			logf("gateway: WARNING — POST /v1/chat/completions is served by the DETERMINISTIC MOCK planner: responses are SCRIPTED, not model output. --gguf was passed but no BPE tokenizer was found (GGUF has no embedded BPE tokenizer and no --tokenizer was provided). Pass --tokenizer <dir|file> to enable real chat, or --base-url to proxy a real provider.")
		} else {
			logf("gateway: WARNING — POST /v1/chat/completions is served by the DETERMINISTIC MOCK planner: responses are SCRIPTED, not model output. Pass --base-url (proxy a real provider) or --gguf/FAK_MODEL_DIR (serve the in-kernel model) to disable the mock.")
		}
		planner = agent.NewMockPlanner(model)
		// Scripted text is not inference, from here or from a vendor. It stays
		// UNCLASSIFIED so a mock run can never pad the self-hosted share with turns
		// no hardware ever served.
	}
	startup.phase("planner-init", time.Since(t))
	return planner, side, upstreamAttribution(proxyURLs), inKernelModelButChatIsMock, nil
}

// ablateUncachedTrimBytes composes Config.ElideResultBytes with the FAK_ABLATE_UNCACHED_TRIM
// ablation arm (#2182): a configured positive threshold always wins verbatim, and an arm
// sweeping the uncached-tail trim on a construction that left the lever inert (<= 0) arms it
// at the documented default. This is the int-shaped form of the `cfg.X || envEnabled(...)`
// compose the boolean levers (CacheTTL1H, VCacheAnchor, PrefixGuard) use, so guard flags and
// ablation env compose instead of replacing each other.
func ablateUncachedTrimBytes(configured int) int {
	if configured <= 0 && envEnabled("FAK_ABLATE_UNCACHED_TRIM") {
		return DefaultElideResultBytes
	}
	return configured
}

func envEnabled(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// onUpstreamRetry is the planner's RetryNotify hook: count the retry and surface it on the
// default debug-stats line so the otherwise-silent 429/5xx backoff window is visible. status is
// the upstream HTTP status that triggered the retry (0 for a transient transport error).
func (s *Server) onUpstreamRetry(attempt, status int, wait time.Duration) {
	if s == nil {
		return
	}
	if s.metrics != nil {
		s.metrics.observeUpstreamRetry(wait)
	}
	if s.debugStatsf != nil {
		s.debugStatsf("fak-turn retry attempt=%d status=%d wait=%s", attempt, status, wait.Round(100*time.Millisecond))
	}
}

// onAuthRefresh is the planner's AuthRefreshNotify hook: surface a 401 token-rotation self-heal
// on the rotating-subscription path. It is SEPARATE from onUpstreamRetry so a credential expiry
// is never conflated with a 429/5xx backoff. outcome is "recovered" (a fresh token was adopted
// mid-session and the call re-sent in place — the live guarded session healed across a re-login)
// or "exhausted" (no fresher token landed within the grace window, so the 401 is about to surface
// and the wrapped agent will drop into its own /login). This is the otherwise-INVISIBLE event the
// "fak guard gets stuck on login sometimes" class hinges on: with this line an operator sees the
// self-heal happen — or sees it give up — instead of a silent session loss.
func (s *Server) onAuthRefresh(outcome string, attempt int) {
	if s == nil {
		return
	}
	if s.metrics != nil {
		s.metrics.observeUpstreamAuthRefresh(outcome)
	}
	if s.debugStatsf != nil {
		s.debugStatsf("fak-turn auth-refresh outcome=%s attempt=%d", outcome, attempt)
	}
}

// onForbiddenRetry is the planner's ForbiddenRetryNotify hook: surface a 403 transient-recovery
// outcome. It is SEPARATE from onAuthRefresh (a 401 credential rotation) and onUpstreamRetry (a
// 429/5xx backoff) so a transient-permission flap is its own signal, not conflated with the
// other two. outcome is "recovered" (a retry within the short window returned 200 — a transient
// abuse/capacity gate cleared and the live session healed in place instead of dropping into a
// spurious /login) or "exhausted" (the bounded window/attempts elapsed still 403ing, so the
// denial is the permanent entitlement kind now surfacing with the actionable answer). This is the
// otherwise-INVISIBLE event the 2026-07-03 gem8 storm exposed: five sessions 403'd for ~9 minutes
// then recovered on the same token — with this line an operator sees the flap self-heal, or sees
// it give up on a real permission denial, instead of a silent session loss.
func (s *Server) onForbiddenRetry(outcome string, attempt int) {
	if s == nil {
		return
	}
	if s.metrics != nil {
		s.metrics.observeUpstreamForbiddenRetry(outcome)
	}
	if s.debugStatsf != nil {
		s.debugStatsf("fak-turn forbidden-retry outcome=%s attempt=%d", outcome, attempt)
	}
}

// onAccountFailover is the planner's AccountFailoverNotify hook: surface an ACCOUNT-SCOPED failover
// outcome — the response to a 403/402 whose body says this credential's organization (or region or
// billing) is walled, even though the credential is valid. It is SEPARATE from the other three
// notify hooks (a 429/5xx backoff, a 401 token rotation, a transient-403 flap) because the cause is
// distinct — a permanent PER-CREDENTIAL wall that no retry or re-login clears — and so is the fix:
// swap to a permitted sibling account. outcome is "recovered" (a permitted sibling credential was
// adopted and the walled turn completed in place, so the session healed onto a working account
// instead of dropping into a futile /login) or "exhausted" (no failover target existed, so the
// account-scoped 403 now surfaces). The SAME hook also reports the 429-account-cap seat rehome that
// rides this swap mechanism: "rehomed_seat" (a session/weekly/usage cap — whose reset can be hours
// away — was answered by moving to a free sibling seat that served the turn now, instead of sleeping
// toward the reset) or "rehome_seat_unavailable" (every sibling seat was capped/walled, so the
// cap-aware backoff rode it out). This is the otherwise-INVISIBLE event behind the org-OAuth-disabled
// failure AND the "429 that was longer than it looked": with this line an operator sees the session
// auto-switch accounts/seats — or sees it give up because every one is walled — instead of a silent
// session loss.
func (s *Server) onAccountFailover(outcome string, attempt int) {
	if s == nil {
		return
	}
	if s.metrics != nil {
		s.metrics.observeUpstreamAccountFailover(outcome)
	}
	if s.debugStatsf != nil {
		s.debugStatsf("fak-turn account-failover outcome=%s attempt=%d", outcome, attempt)
	}
}

// newRouteLive wraps the validated config manifest in an atomic Live holder, or
// returns nil when routing is off (no --route-manifest). A nil Live leaves
// routeDecision on the kernel-default path, byte-for-byte the pre-routing behavior.
func newRouteLive(m *modelroute.Manifest) *modelroute.Live {
	if m == nil {
		return nil
	}
	return modelroute.NewLive(m)
}

// RouteLive returns the atomic holder of the live routing policy, or nil when no
// --route-manifest is installed. The host (cmd/fak serve) hands this to a
// modelroute.Watcher so a manifest edit hot-swaps the policy this server reads —
// the same Live, so the swap is visible on the hot path with no restart (#842).
func (s *Server) RouteLive() *modelroute.Live { return s.route }

// SetRouteWatcher installs (or clears, with nil) the model-routing manifest
// hot-reload watcher that backs POST /v1/fak/route/reload (#4003). The host calls
// this from the serve lifecycle right after it constructs the watcher over the
// server's RouteLive holder, so the manual reload route drives the SAME watcher as
// the background poll loop. The store is atomic, so a concurrent request that hits
// the route observes either the old or the new watcher, never a torn pointer. A nil
// watcher leaves the route disabled (404), mirroring an unset reloadPolicy.
func (s *Server) SetRouteWatcher(w *modelroute.Watcher) { s.routeWatcher.Store(w) }

// currentRouteWatcher loads the installed route-reload watcher, or nil when routing
// hot-reload is not configured for this deployment (the 404 predicate for the route
// handler).
func (s *Server) currentRouteWatcher() *modelroute.Watcher { return s.routeWatcher.Load() }

func newEngineCacheClient(cfg Config) (*enginecache.Client, error) {
	engineName := strings.ToLower(strings.TrimSpace(cfg.EngineCacheEngine))
	baseURL := strings.TrimSpace(cfg.EngineCacheBaseURL)
	if engineName == "" && baseURL == "" && strings.TrimSpace(cfg.EngineCacheAdminKey) == "" && cfg.EngineCacheIdleTimeout == 0 && !cfg.EngineCacheRequireExactSpan {
		return nil, nil
	}
	if engineName == "" {
		return nil, errors.New("gateway: engine cache reset requires EngineCacheEngine (sglang|vllm)")
	}
	if baseURL == "" {
		urls, err := proxyBaseURLs(cfg)
		if err != nil {
			return nil, err
		}
		if len(urls) > 1 {
			return nil, errors.New("gateway: engine cache reset with replica base URLs requires EngineCacheBaseURL")
		}
		if len(urls) == 1 {
			// urls[0] may carry a name=URL identity; the cache-control target is the URL.
			_, baseURL = parseReplicaEntry(urls[0])
		}
	}
	if baseURL == "" {
		return nil, errors.New("gateway: engine cache reset requires EngineCacheBaseURL or BaseURL")
	}
	engine := enginecache.Engine(engineName)
	switch engine {
	case enginecache.EngineSGLang, enginecache.EngineVLLM:
	default:
		return nil, fmt.Errorf("gateway: unsupported engine cache engine %q (want sglang|vllm)", cfg.EngineCacheEngine)
	}
	requiredScope := ""
	if cfg.EngineCacheRequireExactSpan {
		requiredScope = enginecache.ScopeExactSpan
	}
	return &enginecache.Client{
		Engine:        engine,
		BaseURL:       baseURL,
		AdminAPIKey:   cfg.EngineCacheAdminKey,
		IdleTimeout:   cfg.EngineCacheIdleTimeout,
		RequiredScope: requiredScope,
	}, nil
}

func proxyBaseURLs(cfg Config) ([]string, error) {
	urls := make([]string, 0, 1+len(cfg.ReplicaBaseURLs))
	if base := strings.TrimSpace(cfg.BaseURL); base != "" {
		urls = append(urls, base)
	}
	for i, base := range cfg.ReplicaBaseURLs {
		base = strings.TrimSpace(base)
		if base == "" {
			return nil, fmt.Errorf("gateway: replica base URL %d is empty", i+1)
		}
		urls = append(urls, base)
	}
	return urls, nil
}

// newInKernelChatPlanner builds the in-kernel chat planner (the model fused into the
// kernel) advertising modelID, wiring expert parallelism onto the model exactly as the
// single-planner path always has. Shared by the pure in-kernel case and the dual
// (local-alongside-API) case so the EP semantics cannot drift between them.
// Expert parallelism is model state, set on the in-kernel Model here (the EP rank
// lives on the Model, consumed by ffnForLayer); 0/1 is the no-op default.
func newInKernelChatPlanner(cfg Config, modelID string, logf func(string, ...any)) agent.Planner {
	if cfg.ExpertParallelRanks > 1 && !cfg.InKernelModel.IsExpertParallelRankLocal() {
		cfg.InKernelModel.SetExpertParallelRanks(cfg.ExpertParallelRanks)
		// Reduce the routed-expert partials through the DEVICE collective the serve
		// initialized — serve.go gates ranks>1 on a backend advertising Caps().Collective
		// (the NCCL CollectiveBackend), so the decode AllReduceSum must cross those GPUs,
		// not the hardcoded single-box LocalCollective glmMoeEPFFN reduced through before.
		// On cpu-ref the bridge is byte-identical to LocalCollective (collective_bridge_test.go),
		// so this changes no host-tested bytes; on the NCCL backend the SAME call all-reduces
		// across the rank fleet. Fail-soft: a backend without the seam leaves the bit-exact
		// LocalCollective default (the EP output stays correct, just reduced host-side).
		if cfg.Backend != nil {
			if err := cfg.InKernelModel.SetExpertParallelDeviceCollective(cfg.Backend); err == nil {
				logf("gateway: expert-parallel ranks=%d → routed-expert AllReduceSum reduces through device collective %q (Caps().Collective=%v)", cfg.ExpertParallelRanks, cfg.Backend.Name(), cfg.Backend.Caps().Collective)
			} else {
				logf("gateway: expert-parallel ranks=%d: backend %q exposes no device collective (%v) — reducing host-side via LocalCollective (correct, single-box)", cfg.ExpertParallelRanks, cfg.Backend.Name(), err)
			}
		}
	} else if cfg.ExpertParallelRanks > 1 && cfg.InKernelModel.IsExpertParallelRankLocal() {
		// A SHARDED EP rank: the serve already set the rank, the world size, and the DistComm
		// process-group collective (each rank holds only its band, reduces cross-process). Do
		// NOT re-wire a single-process device/Local collective here — it would clobber the
		// cross-process reduce and break the sharded serve (#971).
		logf("gateway: expert-parallel ranks=%d rank-local (sharded serve) — reducing through the serve's DistComm process group, device-collective wiring skipped", cfg.ExpertParallelRanks)
	}
	plannerCfg := cfg.InKernelPlanner
	plannerCfg.CPUOffloadExperts = cfg.CPUOffloadExperts
	return agent.NewInKernelPlannerWithConfig(cfg.InKernelModel, cfg.Tokenizer, modelID, cfg.InKernelQ4K, cfg.Backend, cfg.Metal, plannerCfg)
}

func newProxyPlanner(cfg Config, model string, baseURLs []string) (agent.Planner, error) {
	if len(baseURLs) == 1 {
		// A lone upstream needs no replica identity, but still honor a name=URL form
		// (an operator may pin one) by dialing only the URL part.
		_, dialURL := parseReplicaEntry(baseURLs[0])
		return newConfiguredHTTPPlanner(cfg, model, dialURL)
	}
	replicas := make([]PlannerReplica, 0, len(baseURLs))
	for _, base := range baseURLs {
		// Stable, order-independent identity (#3968): use the operator-chosen id from a
		// name=URL entry, else derive replica-<digest> from the endpoint so the same
		// upstream keeps one identity — and one set of metric/residency labels — no
		// matter its flag position or a membership change that drops a peer.
		name, dialURL := parseReplicaEntry(base)
		if name == "" {
			name = deriveReplicaName(dialURL)
		}
		p, err := newConfiguredHTTPPlanner(cfg, model, dialURL)
		if err != nil {
			return nil, err
		}
		replicas = append(replicas, PlannerReplica{
			Name:    name,
			Planner: p,
		})
	}
	router, err := NewReplicaRouter(model, replicas)
	if err != nil {
		return nil, err
	}
	router.Hedge = cfg.HedgePolicy
	return router, nil
}

// newConfiguredHTTPPlanner dials one upstream and applies every Config-derived knob to
// the resulting planner — the wiring newProxyPlanner performs once for a lone upstream
// and once per replica, so a new Config field is threaded through in exactly one place.
func newConfiguredHTTPPlanner(cfg Config, model, dialURL string) (*agent.HTTPPlanner, error) {
	p, err := agent.NewProviderHTTPPlanner(cfg.Provider, dialURL, model, cfg.APIKey)
	if err != nil {
		return nil, err
	}
	p.APIKeyFunc = cfg.APIKeyFunc
	p.AccountFailoverFunc = cfg.AccountFailoverFunc
	p.TransientTargetFunc = cfg.TransientTargetFunc
	p.ExtraHeaders = cloneConfigHeaders(cfg.ExtraHeaders)
	p.ExtraHeadersFunc = cfg.ExtraHeadersFunc
	p.ForceResponsesStream = cfg.ForceResponsesStream
	// Passed through verbatim: Config.StreamProgressTimeout carries the planner field's own
	// encoding (0 = the agent default, negative = disabled, out-of-band = the default), so a
	// Config nobody configures leaves the planner at the 300s default byte-for-byte.
	p.StreamProgressTimeout = cfg.StreamProgressTimeout
	wrapUpstreamObserver(p.Client, cfg.UpstreamResponseObserver, cfg.UpstreamTransportErrorObserver)
	return p, nil
}

func cloneConfigHeaders(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		if strings.TrimSpace(k) != "" {
			out[k] = v
		}
	}
	return out
}

// MarkReady stamps the instant the gateway became able to serve requests, closing
// the boot timeline (fak_gateway_time_to_ready_seconds / fak_gateway_ready_time_
// seconds). Idempotent and safe on a nil-startup Server; the first call wins.

// RecordStartupPhase appends a host-timed startup stage, including work after MarkReady.
func (s *Server) RecordStartupPhase(name string, dur time.Duration, provenance string) {
	s.startup.phaseWithProvenance(name, dur, provenance)
}

// BeginChildStartup starts the external child cold-start observation window.
func (s *Server) BeginChildStartup(at time.Time) { s.startup.beginChildStartup(at) }

// MarkChildUsable closes the child window at the first authenticated API request.
func (s *Server) MarkChildUsable(at time.Time) { s.startup.markChildUsable(at) }

func (s *Server) MarkReady() {
	if s == nil {
		return
	}
	s.startup.markReady(time.Now())
}

// AdjudicationSummary returns a verdict roll-up over every kernel decision this
// gateway has made so far — proposed-call adjudication, direct syscalls, and inbound
// result admission. It is the live tally `fak guard` prints on exit (what the kernel
// allowed vs denied / repaired / quarantined), read straight from the same operation
// counters /metrics exposes. Safe on a nil Server (returns the zero summary).
func (s *Server) AdjudicationSummary() AdjudicationSummary {
	if s == nil {
		return AdjudicationSummary{ByReason: map[string]uint64{}}
	}
	sum := s.metrics.adjudicationSummary()
	// The compaction budget lives on the Server, not the metrics ledger; attach it here so
	// the exit line can distinguish "enabled but idle" from "disabled" (0).
	sum.CompactionBudget = s.compactHistoryBudget
	return sum
}

// KernelCounters returns a snapshot of the kernel's call-path tallies (engine
// dispatches, vDSO hits, in-syscall repairs, fast-reject denies) — the raw counts a
// tier-4 caller folds through internal/callavoid to render the avoided-call
// amplification headline for the `fak guard` exit summary. The verdict roll-up
// (allowed/denied/…) is AdjudicationSummary; this is the orthogonal call-path axis
// (was the call avoided, and how much further did the agent get per real dispatch?),
// read straight from the same kernel.Counters the fak_kernel_* metrics expose. Safe
// on a nil Server (returns the zero Counters).
func (s *Server) KernelCounters() kernel.Counters {
	if s == nil {
		return kernel.Counters{}
	}
	return s.k.Counters()
}

// HarnessCoherenceSummary returns the operator-line roll-up of the harness-coherence
// family (the same numbers the fak_harness_coherence_* scrape reports). It is exposed
// next to AdjudicationSummary/KernelCounters so a host can persist the session's
// ObservedTurns — the REAL count of served passthrough turns — at exit. That count is
// the honest session-length signal the gateway-usage ledger records: Submits is 0 on the
// guard proxy path (the kernel submit boundary is not on the pass-through wire), so the
// durable turn-distribution corpus would otherwise have only CachedTurns as a proxy.
// Safe on a nil Server (returns the zero summary — a nil Server served no turns).
func (s *Server) HarnessCoherenceSummary() HarnessCoherenceSummary {
	if s == nil {
		return HarnessCoherenceSummary{}
	}
	return s.metrics.harnessCoherenceSummary()
}

// AssumeSessionTurns returns the resolved session-length prior (Config.AssumeSessionTurns;
// 0 = the head-anchored burst gate's prior is disabled) this Server booted with. Exposed
// so a host can stamp it into a persisted row's Provenance — the corpus that CALIBRATES
// DefaultAssumedSessionTurns must be able to exclude override sessions from the fit. Safe
// on a nil Server (returns 0).
func (s *Server) AssumeSessionTurns() int {
	if s == nil {
		return 0
	}
	return s.assumeSessionTurns
}

// CompactHistoryBudget returns the resident-token budget compaction fires against
// (Config.CompactHistoryBudget; 0 = compaction OFF), the sibling provenance knob to
// AssumeSessionTurns. Safe on a nil Server (returns 0). AdjudicationSummary also surfaces
// this as CompactionBudget; this dedicated accessor lets a provenance stamp read it without
// folding the whole counter roll-up.
func (s *Server) CompactHistoryBudget() int {
	if s == nil {
		return 0
	}
	return s.compactHistoryBudget
}

// ExposeProfile returns the descriptive expose-profile label this session launched under
// (Config.ExposeProfile; "" when unset), the sibling provenance knob to AssumeSessionTurns
// and CompactHistoryBudget. Safe on a nil Server (returns ""); the caller normalizes an
// empty value to "interactive".
func (s *Server) ExposeProfile() string {
	if s == nil {
		return ""
	}
	return s.exposeProfile
}

// VCacheTurnsSnapshot returns a copy of the per-turn provider-cache window this session
// observed (input/cache_read/cache_creation tokens per turn, the OBSERVED axis fed by
// observeVCacheTurn on every streamed passthrough turn), plus whether the bounded window
// has dropped older turns. It is the live source `fak vcache score` reads to report the
// REALIZED cache multiplier instead of the synthetic-Zipf forecast — exposed here, next to
// AdjudicationSummary, so a host can persist it at session exit. Safe on a nil Server.
func (s *Server) VCacheTurnsSnapshot() ([]vcacheobserve.Turn, bool) {
	if s == nil {
		return nil, false
	}
	turns, capped := s.metrics.vcacheTurnsSnapshot()
	attachVCacheContextEconomics(turns, s.compactHistoryBudget)
	return turns, capped
}

// SetModelLoadProfile records the boot-time weight-load breakdown the host captured
// while eagerly loading a model (fak serve --gguf), exposing it as the
// fak_model_load_* metric family. Passing nil clears it. Safe for concurrent use
// and on a nil Server.
func (s *Server) SetModelLoadProfile(p *ModelLoadProfile) {
	if s == nil {
		return
	}
	s.modelLoadMu.Lock()
	s.modelLoad = p.clone()
	s.modelLoadMu.Unlock()
}

func (s *Server) modelLoadProfile() *ModelLoadProfile {
	s.modelLoadMu.Lock()
	defer s.modelLoadMu.Unlock()
	return s.modelLoad.clone()
}

// maybePlanMessages is the live-loop integration point for the ctxplan context PLANNER
// (issue #555): when the view planner is enabled, each buffered turn's history is lowered
// into a lossless store and re-materialized as an O(1) resident view under the configured
// budget — a planned view in place of appending the whole transcript. When the planner is
// off (the default) it returns the input UNCHANGED, so a deploy that leaves the flag off is
// byte-for-byte identical to the pre-seam path. It is FAIL-SAFE: any planner error or empty
// render falls back to the full lossless history, so an experimental rewrite can never
// break or empty a turn — the planner only ever SHORTENS, and on doubt it shortens nothing.
func (s *Server) maybePlanMessages(ctx context.Context, trace string, messages []agent.Message) []agent.Message {
	if s.ctxView == nil || !s.ctxView.Enabled {
		return messages
	}
	if hasStructuredToolContinuation(messages) {
		return messages
	}
	// With a stable session trace, plan through the PERSISTENT per-session index — the
	// incremental O(c·N) path, output-equivalent to the stateless full-scan but without
	// rebuilding the lossless store every turn. Without a trace (a one-shot caller), fall
	// back to the stateless shared planner so behavior is unchanged for an unkeyed request.
	if sp := s.sessionPlannerFor(trace); sp != nil {
		planned := sp.RenderTurn(ctx, messages)
		if len(planned) == 0 {
			return messages // fail-safe: never empty a turn
		}
		return planned
	}
	planned, err := s.ctxView.RenderTurn(ctx, messages)
	if err != nil || len(planned) == 0 {
		s.logf("gateway: ctxplan view planning fell back to full history: %v", err)
		return messages
	}
	return planned
}

func hasStructuredToolContinuation(messages []agent.Message) bool {
	for _, m := range messages {
		if len(m.ToolCalls) > 0 || m.FunctionCall != nil || m.ToolCallID != "" {
			return true
		}
	}
	return false
}

func (s *Server) observeDecodedCtxViewRewrite(trace string, full, planned []agent.Message) {
	if s == nil || len(planned) >= len(full) {
		return
	}
	fullTokens := estimateMessageContentTokens(full)
	plannedTokens := estimateMessageContentTokens(planned)
	shed := 0
	if fullTokens > plannedTokens {
		shed = fullTokens - plannedTokens
	}
	s.metrics.observeCtxViewRewrite(agent.CompactOutcome{
		Reason:     agent.CompactReasonNone,
		Dropped:    len(full) - len(planned),
		ShedTokens: shed,
	})
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return
	}
	s.ctxViewPendingMu.Lock()
	if s.ctxViewPending == nil {
		s.ctxViewPending = map[string]int{}
	}
	s.ctxViewPending[trace]++
	s.ctxViewPendingMu.Unlock()
}

func (s *Server) consumeDecodedCtxViewEvent(trace string) bool {
	if s == nil {
		return false
	}
	trace = strings.TrimSpace(trace)
	if trace == "" {
		return false
	}
	s.ctxViewPendingMu.Lock()
	defer s.ctxViewPendingMu.Unlock()
	n := s.ctxViewPending[trace]
	if n <= 0 {
		return false
	}
	if n == 1 {
		delete(s.ctxViewPending, trace)
	} else {
		s.ctxViewPending[trace] = n - 1
	}
	return true
}

func estimateMessageContentTokens(messages []agent.Message) int {
	total := 0
	for _, m := range messages {
		if m.Content == "" {
			continue
		}
		total += (len(m.Content) + 3) / 4
	}
	return total
}

type suppressDecodedCtxViewKey struct{}

func withDecodedCtxViewSuppressed(ctx context.Context) context.Context {
	return context.WithValue(ctx, suppressDecodedCtxViewKey{}, true)
}

func decodedCtxViewSuppressed(ctx context.Context) bool {
	v, _ := ctx.Value(suppressDecodedCtxViewKey{}).(bool)
	return v
}

// maybeElideKVResidency drives the model-side PLANNED-ELISION residency bridge (issue #579, the
// kvmmu-planned-eviction half) when the context planner shrank the turn history. It is the
// capacity-plan twin of evictInKernelPoison's KVSpanEvictor (which enforces a trust quarantine):
// the planner's O(1) text view becomes a real O(1) KV RESIDENCY, with the elided spans' K/V
// actually evicted from the kernel-owned cache instead of physically held behind an O(1) view.
//
// Honest posture (issue #579, "bit-exact provable direction only"): the bridge engages on the
// plan the context planner produced, and ElideKVSpans evicts every elided span — but it asserts
// the bit-exact O(1)-residency invariant ONLY when the elided spans are the positional SUFFIX (the
// over-budget direction the kvmmu witness proves: a re-RoPE renumbers survivors with no surviving
// earlier token having attended to an evicted later one). Eliding an old prefix the recent
// resident tail already attended to still shrinks residency but is NOT reported bit-exact, rather
// than overclaiming an invariant a re-RoPE cannot satisfy.
//
// It is a no-op (returns silently) unless the planner implements KVSpanElider (the in-kernel
// engine with FAK_INKERNEL_KVMMU on) AND the planned view is a clean sub-sequence of the history
// (a reorder/rewrite is left untouched — the bridge fails OPEN rather than evict the wrong span).
// Default posture is therefore unchanged unless an operator opts the in-kernel bridge in.
func (s *Server) maybeElideKVResidency(fullHistory, planned []agent.Message) {
	elider, ok := s.planner.(agent.KVSpanElider)
	if !ok {
		return
	}
	// Recover which fullHistory messages the planner elided. The planned view must be a clean
	// trailing SUFFIX of fullHistory (planning dropped a leading prefix); a reorder/rewrite is
	// not a shape the residency bridge can map safely, so it is skipped (fail-open).
	elided, ok := elidedPrefixMask(fullHistory, planned)
	if !ok {
		return
	}
	plan := agent.SegElisionPlan(fullHistory, elided)
	if len(plan.Elided) == 0 {
		return // planning kept everything resident — nothing to evict
	}
	if freed, exact := elider.ElideKVSpans(fullHistory, plan); freed > 0 {
		s.logf("gateway: in-kernel KV residency shrank to planned view elided=%d freed=%dpos reposition_exact=%v", len(plan.Elided), freed, exact)
	}
}

// elidedPrefixMask recovers which fullHistory messages the planner elided, for the case the
// planned view is a trailing SUFFIX of fullHistory (planning dropped a leading prefix). It returns
// a mask where the leading prefix is elided (true) and the resident suffix is kept (false), and
// ok=false when the planned view is not a clean trailing suffix (a reorder or rewrite). Compares
// role+content, the fields renderTranscript lowers into the spans the bridge evicts.
//
// NOTE the bit-exactness direction is decided downstream: a trailing-suffix RESIDENT view means
// the ELIDED spans are the leading prefix — the non-bit-exact direction — so ElideKVSpans will
// shrink residency but report reposition_exact=false here. The provable (suffix-elided) direction
// is exercised by the unit witness driving SegElisionPlan directly. This gate is the conservative
// pre-filter; ElideKVSpans re-checks positional order and is the load-bearing proof.
func elidedPrefixMask(fullHistory, planned []agent.Message) (mask []bool, ok bool) {
	if len(planned) == 0 || len(planned) >= len(fullHistory) {
		return nil, false
	}
	off := len(fullHistory) - len(planned)
	for i := range planned {
		if planned[i].Role != fullHistory[off+i].Role || planned[i].Content != fullHistory[off+i].Content {
			return nil, false
		}
	}
	mask = make([]bool, len(fullHistory))
	for i := 0; i < off; i++ {
		mask[i] = true
	}
	return mask, true
}

// maybeElideMessages shrinks oversized OLD tool-role message Content to a bounded head+tail on
// the DECODED []Message path — the OpenAI / in-kernel wire a LOCAL model served by fak takes
// (GLM-5.2 / Qwen-3.6-27B via an OpenAI backend or the in-kernel engine), where the byte-splice
// ElideAnthropicResults — which only fires on the real-Anthropic passthrough — never runs. It is
// the decoded-path twin of maybeElideAnthropicRaw, so oversized-result elision is enabled by
// default on BOTH wires. Guarded OFF on the passthrough (handled there on req.Raw) and when
// --elide-result-bytes is 0. agent.ElideMessages is copy-on-write and fail-safe (it only ever
// SHORTENS an old tool message, never empties a turn, recent working set protected), so this can
// never break a turn or mutate the caller's slice.
func (s *Server) maybeElideMessages(messages []agent.Message) []agent.Message {
	if s.elideResultBytes <= 0 || s.anthropicPassthrough() {
		return messages
	}
	out, outcome := agent.ElideMessages(messages, s.elideResultBytes)
	s.metrics.observeUncachedTrim(outcome)
	return out
}

// maxSessionPlanners bounds the per-session planner cache so a long-lived gateway serving
// many distinct traces cannot grow it without limit. When the cache is full a new trace
// evicts the whole map (a cheap generational reset) rather than tracking per-entry LRU —
// the planners are reconstructible from the next turn's full history, so eviction only
// costs that session one O(N) rebuild, never correctness.
const maxSessionPlanners = 8192

// sessionPlannerFor returns the persistent SessionPlanner for a trace, minting one lazily
// from the shared ctxView config (CtxViewPlanner.NewSession). It returns nil when the
// planner is disabled or the trace is empty, so the caller falls back to the stateless
// path. Concurrency-safe: the per-session planner is mutated only under sessionPlannerMu
// by the single in-flight turn for that trace (turns of one session are serial).
func (s *Server) sessionPlannerFor(trace string) *agent.SessionPlanner {
	if s.ctxView == nil || !s.ctxView.Enabled || trace == "" {
		return nil
	}
	s.sessionPlannerMu.Lock()
	defer s.sessionPlannerMu.Unlock()
	if s.sessionPlanners == nil {
		s.sessionPlanners = make(map[string]*agent.SessionPlanner)
	}
	if sp, ok := s.sessionPlanners[trace]; ok {
		return sp
	}
	if len(s.sessionPlanners) >= maxSessionPlanners {
		s.sessionPlanners = make(map[string]*agent.SessionPlanner) // generational reset
	}
	sp := s.ctxView.NewSession()
	s.sessionPlanners[trace] = sp
	return sp
}

// existingSessionPlanner returns the retained per-trace SessionPlanner WITHOUT minting one — the
// read-only counterpart of sessionPlannerFor. The restore path (fak_context_restore, #3062) uses it
// so a restore call for a trace that never planned a turn is a plain miss that falls through to the
// next source, rather than a freshly-minted empty planner (which would allocate, pollute the map,
// and still miss). It does not gate on ctxView.Enabled: if a planner was retained for the trace its
// lossless store is a valid ctxview-elision source regardless of the current seam flag. The returned
// planner is safe to call Spans/Materialize on concurrently — those methods take the planner's own
// lock — so the read holds sessionPlannerMu only for the map lookup.
func (s *Server) existingSessionPlanner(trace string) *agent.SessionPlanner {
	if trace == "" {
		return nil
	}
	s.sessionPlannerMu.Lock()
	defer s.sessionPlannerMu.Unlock()
	return s.sessionPlanners[trace]
}

// complete runs the configured planner for one turn and records the inference
// metrics that make real model work visible at /metrics — the token counts the
// planner reports plus the wall-clock spent generating. Both /v1/chat/completions
// and /v1/messages route through it so the fak_gateway_inference_* family reflects
// every served turn on either wire. On a planner error nothing is recorded (a turn
// that produced no tokens is not a generation); the error is returned untouched so
// the caller's existing error handling is unchanged.
func (s *Server) complete(ctx context.Context, trace string, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (comp *agent.Completion, err error) {
	// Bind authenticated request identity to in-kernel prefix-cache visibility.
	// Empty principal preserves legacy single-user reuse; authenticated prefixes
	// enter tenant-private storage and cannot shape another tenant's hit timing.
	if principal := principalFromContext(ctx); principal != "" {
		ctx = agent.WithPrefixCacheIdentity(ctx, principal, "")
	}
	defer func() {
		if r := recover(); r != nil {
			if evictErr, ok := recoverRecurrentEvictUnsupported(r); ok {
				comp, err = nil, evictErr
				return
			}
			panic(r)
		}
	}()
	// Re-plan the turn history into an O(1) resident view before the model sees it —
	// the "replace append+compact with a planned view" rung (issue #555). Inert (an
	// identity) unless CtxViewBudget > 0, so the default path is unchanged. The trace keys
	// the persistent per-session planner so the rewrite is O(c·N), not O(N²), across turns.
	fullHistory := messages
	messages = s.maybeElideStaleReadMessages(trace, messages)
	messages = s.maybePlanMessages(ctx, trace, messages)
	if !decodedCtxViewSuppressed(ctx) {
		s.observeDecodedCtxViewRewrite(trace, fullHistory, messages)
	}
	// Shrink the kernel-owned KV residency to match the planned O(1) view (issue #579, the
	// kvmmu-planned-eviction half): when planning ELIDED older history (the view is a strict
	// trailing window of the full transcript), drive the model-side residency bridge so the
	// elided spans' K/V is actually evicted via model.KVCache.Evict — making the "O(1) view" a
	// real O(1) KV RESIDENCY instead of an O(1) text view over an O(N) cache. Default OFF /
	// fail-open: a no-op unless FAK_INKERNEL_KVMMU opted the in-kernel bridge in.
	s.maybeElideKVResidency(fullHistory, messages)
	// Oversized tool_result elision on the DECODED path — the OpenAI / in-kernel wire a LOCAL
	// model served by fak takes (GLM-5.2 / Qwen-3.6-27B), where the byte-splice passthrough
	// elision never fires. Shrinks old oversized tool-role content to head+tail; default-on,
	// fail-safe, recent working set protected. No-op on the Anthropic passthrough (handled on
	// req.Raw there).
	messages = s.maybeElideMessages(messages)
	start := time.Now()
	comp, err = s.planner.Complete(ctx, messages, tools, opts...)
	dur := time.Since(start)
	if err != nil {
		if _, _, _, ok := inKernelOOMObservation(err); ok {
			s.observePlannerRequestMemory()
		}
		return nil, err
	}
	s.metrics.observeInferenceUsageServed(s.servedLocalityOf(opts), comp.Usage, comp.FinishReason, dur)
	s.observePlannerRequestMemory()
	// The served turn has mutated the KV cache; relieve HBM pressure by demoting a hot span to
	// the colder tier instead of dropping it (#1073, the live serve-path call site for the
	// capacity executor). Fail-open + gated: a no-op unless FAK_INKERNEL_KVMMU armed the bridge
	// AND the host injected a device-backed provider+sweeper via SetKVPressureRelief.
	s.maybeRelieveKVPressure(ctx)
	return comp, nil
}

func recurrentEvictUnsupported(err error) bool {
	var evictErr *model.RecurrentEvictUnsupportedError
	return errors.As(err, &evictErr)
}

func recoverRecurrentEvictUnsupported(r any) (error, bool) {
	err, ok := r.(error)
	if !ok || !recurrentEvictUnsupported(err) {
		return nil, false
	}
	return err, true
}

// completeServed is complete plus the served-session usage debit. The request
// boundary has already called beginServedSessionTurn (and therefore Decide); after a
// successful planner response the provider usage is finally known, so debit the
// output/context budgets here. Planner errors keep the old behavior: no usage was
// reported, so there is nothing to debit beyond the turn admission already taken.
func (s *Server) completeServed(ctx context.Context, turn servedSessionTurn, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	ctx = providerSpanContext(ctx, s.otlp)
	messages = applyNegframeRequestPass(messages, turn.traceID)
	// Mark the trace as holding an open model request so the agents pane can show its
	// in-flight age (#2627). The window closes when this call returns; adjudication then
	// stamps last_tool/idle from the served turn. Cleared on every exit path via defer.
	began := time.Now()
	s.activity.beginTurn(turn.traceID, began)
	defer s.activity.endTurn(turn.traceID)
	lease, err := s.beginServedAdmission(ctx, turn, messages, tools, sampleMaxTokens(opts))
	if err != nil {
		return nil, err
	}
	defer lease.Release()
	comp, err := s.complete(ctx, turn.traceID, messages, tools, opts...)
	if err != nil {
		return nil, err
	}
	// The provider's real usage is now known — settle the token-rate window with it
	// (#2019), replacing the admission-time estimate.
	lease.SettleUsage(comp.Usage)
	s.debitServedSessionTurn(ctx, turn, comp.Usage, time.Since(began), messages)
	return comp, nil
}
