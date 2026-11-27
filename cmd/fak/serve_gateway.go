package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/macobs"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// buildGateway loads the optional model-routing policy, constructs the gateway
// server from the resolved planes, and arms the admission controller for a pure
// in-kernel serve.
func (rt *serveRuntime) buildGateway(sf *serveFlags) (result *gateway.DurableControlIngress, buildErr error) {
	ensureServeProxyTimeoutFloors(sf)
	rt.ensureServeInkling2WriteTimeout()
	resolveServeEngine(sf, rt.explicitFlags, rt.inKernelModel != nil)
	native, _ := abi.Engine(*sf.engineID).(*modelengine.Engine)
	if err := configureServeHostSwapBudget(sf, native); err != nil {
		return nil, err
	}
	startupMessages := append([]gateway.StartupMessage(nil), rt.startupMessages...)
	nativePlannerConfig := serveNativePlannerConfigWithContext(sf, rt.nativeContext.ResolvedTokens)
	nativePlannerConfig.RequireDeviceExecution = rt.requireDeviceExecution
	nativePlannerConfig.DeviceOnlyBypass = rt.executionPolicy.sessionBypass()
	if err := applyServeNativeChatTemplate(sf, &nativePlannerConfig, rt.inKernelModel != nil && rt.inKernelTok != nil); err != nil {
		return nil, err
	}
	// The backend tier check (resolveServeKVPrecision) does not see the architecture;
	// refuse an arch-unrealized --kv-precision (e.g. q8_0 on a Qwen hybrid) before the
	// listener binds, so the service fails admission instead of refusing every request.
	if rt.inKernelModel != nil {
		if err := agent.ValidateKVPrecision(rt.inKernelModel.Cfg, nativePlannerConfig.KVPrecision, rt.chatBackend, rt.useMetal); err != nil {
			return nil, err
		}
	}
	// Resolve the optional model-routing policy. Off by default: an empty --route-manifest
	// leaves routeMan nil, so gateway.New gets a nil RouteManifest and Engine stays unset —
	// byte-for-byte the pre-routing behavior. A malformed file fails loud here rather than
	// silently default-routing every call to the kernel default (a mis-routed model is a
	// security boundary). gateway.New also re-validates the loaded manifest.
	var routeMan *modelroute.Manifest
	if *sf.routeManifest != "" {
		routeMan = loadServeRouteFile("route-manifest", *sf.routeManifest, "a modelroute manifest whose every plan member resolves", modelroute.LoadManifest)
		startupMessages = append(startupMessages, gateway.StartupMessage{Source: "serve", Kind: "route-manifest", Level: "info", Text: "model-routing policy loaded from " + *sf.routeManifest})
	}

	// Resolve the optional model-ACCOUNT roster (#2528). Off by default: an empty
	// --route-accounts leaves routeRoster nil, so the plan-member string stays the route
	// (byte-for-byte the pre-#2528 behavior). A malformed roster fails loud here rather
	// than mis-binding a routed model to the wrong account (a residency-floor boundary);
	// LoadRoster validates on load and gateway.New re-validates. The roster carries only
	// env-var NAMES, never secrets.
	var routeRoster *modelroute.Roster
	if *sf.routeAccounts != "" {
		routeRoster = loadServeRouteFile("route-accounts", *sf.routeAccounts, "a fak-accounts/v1 roster carrying env var NAMES, never secrets", modelroute.LoadRoster)
		startupMessages = append(startupMessages, gateway.StartupMessage{Source: "serve", Kind: "route-accounts", Level: "info", Text: "model-account roster loaded from " + *sf.routeAccounts})
	}

	// Refuse a reused node-global roster whose loopback account binding disagrees with
	// this process's --base-url, BEFORE the gateway.Config is built (BaseURL / RouteAccounts
	// below) so a mismatched replica never binds a listener. The H200 half-node witness:
	// replica2 launched with the correct --base-url :8010 but replica1's roster binding
	// coder-local -> :8000, so GPUs 4-7 served nothing while every request returned 200
	// from replica1 and the facade stayed healthy. The check is scoped to loopback
	// accounts and to host:port, so a mixed local+remote roster stays legal.
	if err := checkReplicaBindingConsistency(*sf.baseURL, routeRoster); err != nil {
		writeConfigBail(os.Stderr, configBail{
			Verb:    "fak serve",
			Reason:  bailReplicaBindingMismatch,
			Summary: "the account roster's local-engine binding points at a different host:port than this gateway's --base-url",
			Knobs: []bailKnob{
				bailFlag("base-url", *sf.baseURL),
				bailFile(*sf.routeAccounts, "loopback account binding disagrees with --base-url"),
			},
			Bind: []string{"base_url=" + *sf.baseURL, "roster=" + *sf.routeAccounts},
		})
		os.Exit(1)
	}

	// Resolve the optional multi-tenant KEYSET (#5332). Off by default: no --key-principal
	// leaves keyPrincipals nil, so gateway.New builds a nil *keyset and the RequireKey-only
	// auth path stays byte-for-byte what it was. A malformed spec, an unset/empty env var, or
	// two tenants sharing one key fails LOUD here — the same refusal --require-key-env makes
	// in resolveSessionPlane, and for the stronger reason: without a matched keyset the
	// gateway attributes the turn from the caller-asserted X-Fak-Principal header, so a
	// silently-forgotten binding does not just mislabel a tenant, it lets one assert another.
	trajctlMetricsFile := trajctl.NewMetricsFile(filepath.Join(repoRoot(), trajctl.DefaultLedgerRel))
	trajctlMetrics := func() gateway.TrajctlMetrics {
		m := trajctlMetricsFile.Snapshot()
		out := gateway.TrajctlMetrics{Objectives: map[string]int{}, Scores: map[string]float64{}, Signals: map[string]int{}, Nudges: map[string]int{}}
		for k, v := range m.Objectives {
			out.Objectives[string(k)] = v
		}
		for k, v := range m.Scores {
			out.Scores[k] = v
		}
		for k, v := range m.Signals {
			out.Signals[string(k)] = v
		}
		for k, v := range m.Nudges {
			out.Nudges[k] = v
		}
		return out
	}

	orgAudit, loadErr := loadServeOrgAuditConfig()
	if loadErr != nil {
		fmt.Fprintf(os.Stderr, "fak serve: load organization enrollment: %v\n", loadErr)
		os.Exit(2)
	}

	keyPrincipals, keysetOK := serveKeyPrincipals(sf.keyPrincipal.Values(), os.Getenv, os.Stderr)
	if !keysetOK {
		os.Exit(2)
	}
	nativePlannerConfig.WarmDiskModelIdentity = rt.warmDiskModelIdentity
	controlHandler, controlIngress, err := openServeControlIngress(*sf.stdio, rt.requireKey, keyPrincipals, os.Getenv)
	if err != nil {
		return nil, err
	}

	// Gateway lease-admission posture (--workspace-admission): an explicit mode
	// (flag or FAK_WORKSPACE_ADMISSION) wins; the legacy permissive bool resolves
	// to permissive; otherwise the going-forward open default. An unknown token
	// already failed loud at flag-parse time; New re-validates defensively.
	wsAdmissionMode, err := resolveWorkspaceAdmission(*sf.workspaceAdmission, *sf.workspaceAdmissionPermissive)
	if err != nil {
		return nil, err
	}

	// Tool-guard posture (--guard / FAK_GUARD): the serve default is RAW PASS-THROUGH
	// (off) — the router is an inference pipe with zero interception processing unless
	// the operator arms the guard by name. An explicitly passed --guard wins over
	// FAK_GUARD; an unparsable FAK_GUARD fails loud here instead of booting with a
	// silently-resolved security posture. The resolved posture is stated on the
	// startup banner so the served surface can never disagree about its own floor.
	toolGuardMode, err := resolveServeToolGuard(*sf.guard, rt.explicitFlags, os.Getenv)
	if err != nil {
		return nil, err
	}
	if toolGuardMode == gateway.ToolGuardOn {
		startupMessages = append(startupMessages, gateway.StartupMessage{Source: "serve", Kind: "tool-guard", Level: "info", Text: "tool guard: on — tool-call adjudication + tool-result screening armed"})
	} else {
		startupMessages = append(startupMessages, gateway.StartupMessage{Source: "serve", Kind: "tool-guard", Level: "info", Text: "tool guard: off — raw pass-through (arm with --guard or FAK_GUARD=on)"})
	}

	// Context-plane postures (--ctx-paging / FAK_CTX_PAGING, --chat-elide /
	// FAK_CHAT_ELIDE): the serve default is RAW CONTEXT — no oversize paging, no
	// tool-result elision, no restore-tool auto-advertisement, and the per-lever
	// serving flags (--ctx-view-budget, --compact-history-budget, --elide-*)
	// default to their off values. Each lever re-arms by name; an unparsable
	// posture fails loud here instead of booting with a silently-rewriting plane.
	ctxPagingMode, err := resolveServeCtxPaging(*sf.ctxPaging, rt.explicitFlags, os.Getenv)
	if err != nil {
		return nil, err
	}
	chatElideMode, err := resolveServeChatToolResultElide(*sf.chatElide, rt.explicitFlags, os.Getenv)
	if err != nil {
		return nil, err
	}
	if ctxPagingMode == gateway.CtxPagingOff && chatElideMode == gateway.ChatToolResultElideOff {
		startupMessages = append(startupMessages, gateway.StartupMessage{Source: "serve", Kind: "context-plane", Level: "info", Text: "context plane: raw — no oversize paging, no tool-result elision, no restore-tool injection (re-arm with --ctx-paging on / --chat-elide / the --elide-* and --compact-* levers)"})
	} else {
		startupMessages = append(startupMessages, gateway.StartupMessage{Source: "serve", Kind: "context-plane", Level: "info", Text: "context plane: armed — oversize paging " + ctxPagingMode + ", tool-result elision " + chatElideMode})
	}

	srv, err := gateway.New(gateway.Config{
		WorkspaceAdmission:           wsAdmissionMode,
		WorkspaceAdmissionPermissive: *sf.workspaceAdmissionPermissive,
		ToolGuard:                    toolGuardMode,
		CtxPaging:                    ctxPagingMode,
		ChatToolResultElide:          chatElideMode,
		EngineID:                     *sf.engineID,
		OTLPEndpoint:                 *sf.otlpEndpoint,
		OrgAudit:                     orgAudit,
		Model:                        *sf.model,
		BaseURL:                      *sf.baseURL,
		ReplicaBaseURLs:              sf.replicaBaseURLs.Values(),
		Provider:                     *sf.provider,
		VCacheCalibration:            loadVCacheRuntimeCalibration(*sf.provider, *sf.model),
		APIKey:                       rt.apiKey,
		EngineCacheEngine:            *sf.engineCacheEngine,
		EngineCacheBaseURL:           *sf.engineCacheBaseURL,
		EngineCacheAdminKey:          rt.engineCacheAdminKey,
		EngineCacheIdleTimeout:       *sf.engineCacheIdleTimeout,
		EngineCacheRequireExactSpan:  *sf.engineCacheRequireExactSpan,
		EmbeddingPooling:             *sf.embeddingPooling,
		EmbeddingsBackend:            *sf.embeddingsBackend,
		InKernelModel:                rt.inKernelModel,
		NativeChatPlanner:            rt.inkling2Planner,
		ServeStatusProvider:          rt.inkling2ServeStatus,
		Tokenizer:                    rt.inKernelTok,
		InKernelQ4K:                  rt.inKernelQ4K,
		InKernelPlanner:              nativePlannerConfig,
		Backend:                      rt.chatBackend,
		CPUOffloadExperts:            *sf.cpuOffloadExperts,
		Metal:                        rt.useMetal,
		// Backend-aware buffered first-token window (#13592): resolveServeFirstTokenWatchdog
		// resolves it against the chat backend/metal decision set above and records the
		// window and its source as a startup message; without this line the resolver is
		// dead code and the gateway keeps the backend-blind 60s default.
		FirstTokenWatchdog:           rt.resolveServeFirstTokenWatchdog(sf),
		SpeculativeMode:              *sf.speculative,
		ExpertParallelRanks:          *sf.expertParallel,
		RequireKey:                   rt.requireKey,
		ReadBearer:                   rt.metricsKey,
		AllowLAN:                     *sf.allowLAN,
		KeyPrincipals:                keyPrincipals,
		ControlIngress:               controlHandler,
		VDSO:                         *sf.vdso,
		ToolPlugins:                  rt.toolPlugins,
		ToolPreferences:              rt.toolPreferences,
		Invalidation:                 *sf.invalidation,
		Version:                      appversion.Current(),
		ReloadPolicy:                 policyReloader(*sf.policyPath),
		PolicyCanaryTurns:            *sf.policyCanaryTurns,
		ResetTrace:                   resetTrace,
		ObserveTrace:                 observeTrace,
		ObserveSession:               observeSession,
		ControlSession:               controlSession,
		SteerSession:                 steerSession,
		ListSessions:                 listSessions,
		TrajctlMetrics:               trajctlMetrics,
		DecideSession:                decideSession,
		DebitSession:                 debitSession,
		Table:                        serveSessions,
		ResetOnBudget:                resetOnBudgetHook(*sf.resetOnBudget, *sf.contextBudgetTokens),
		DefaultTraceID:               rt.defaultTraceID,
		StartTime:                    rt.t0,
		StartupPhases:                rt.startupPhases,
		CtxViewBudget:                *sf.ctxViewBudget,
		CompactHistoryBudget:         *sf.compactHistoryBudget,
		PositiveResidualSubstitution: *sf.positiveResidualSubstitution,
		CompactAnchorHead:            *sf.compactAnchorHead,
		AssumeSessionTurns:           *sf.assumeSessionTurns,
		ElideResultBytes:             *sf.elideResultBytes,
		ElideStaleReads:              *sf.elideStaleReads,
		ModelPassthrough:             *sf.modelPassthrough,
		// M2 star-anchor pre-flight gate (#1493): DEFAULT-ON (--vcache-anchor), DECOUPLED
		// from CompactHistoryBudget so --compact-history-budget=0 no longer takes anchoring
		// down with it. Applies cachemeta.RecommendLayout on the Anthropic passthrough;
		// fail-safe identity on any ambiguity. No effect on non-passthrough wires.
		VCacheAnchor: *sf.vcacheAnchor,
		// The 10x floor lever (--defer-cold-tools, #3232): defer the COLD tool tail and
		// inject a tool_search_tool on the outbound Anthropic body. DEFAULT OFF; gateway.New
		// also ORs in FAK_DEFER_COLD_TOOLS. Deterministic, cache-safe, fail-safe identity.
		DeferColdTools:  *sf.deferColdTools,
		DisableMCPDefer: !*sf.deferTools,
		MCPToolCeiling:  *sf.toolCeiling,
		DebugStatsf:     debugStatsSink(*sf.debugStats),
		// MCP tool-exposure allowlist (--expose). Empty (default) leaves ExposeTools nil
		// so gateway.New exposes the full tool surface byte-for-byte; a non-empty set
		// narrows both tools/list and tools/call, and New fails loud on a bad pattern.
		ExposeTools: sf.expose.Values(),
		// Inbound twin of #555: prune tool DEFINITIONS the installed floor can never admit
		// from the Anthropic passthrough's tools[], cache-prefix-preserving. The predicate
		// reads adjudicator.Default (the floor serve installs via applyPolicy) under its lock,
		// and is fail-safe against an unconfigured floor (NeverAdmits returns false when there
		// is nothing to admit), so it is a no-op until a real floor is in place — never an
		// over-drop. Behavior-preserving: a pruned tool stays DEFAULT_DENY at the kernel.
		ToolFloorDenies: adjudicator.Default.NeverAdmits,
		// Model-routing policy (#601). nil (the default, no --route-manifest) leaves
		// ToolCall.Engine unset → the kernel default engine, byte-for-byte the pre-routing
		// path. When set, each fak_syscall call is classified and a PICK plan binds the
		// chosen model before Submit so the residency PDP adjudicates the real route.
		RouteManifest: routeMan,
		// Model-account roster (#2528). nil (the default, no --route-accounts) leaves the
		// plan-member string as the route. When set, the routed model id is BOUND through
		// the roster to its account-resolved EngineRoute before Submit, so the residency
		// PDP adjudicates the ACCOUNT-resolved route and a native provider with no wired
		// adapter fails loud instead of running through the default engine.
		RouteAccounts: routeRoster,
		// Native-harness keystone (#1316): drive agent.RunArm for a non-streaming
		// /v1/messages turn. Off by default — the proxy path is byte-for-byte unchanged.
		Native:                     *sf.native,
		NativeMaxTurns:             *sf.nativeMaxTurns,
		NativeCodeWorkspace:        *sf.nativeCodeWorkspace,
		NativeExactAllowedCommands: append([]string(nil), sf.nativeAllowBashCommands...),
		NativeMaxCommandTime:       *sf.nativeBashCommandTimeout,
		NativeSpeculate:            *sf.nativeSpeculate,
		VDSOProxyFill:              *sf.vdsoProxyFill,
		// Streaming CONTENT-progress deadline (#5486, --stream-progress-timeout): the
		// window a warm-but-unadvancing proxied stream is given before the turn is ended.
		// gateway.newConfiguredHTTPPlanner carries it onto every proxy planner, where
		// agent.(*HTTPPlanner).streamProgressWindow band-checks it. The flag's 0 means OFF
		// and is translated here into that resolver's negative encoding; every other value
		// (including the 300s default) passes through untouched.
		StreamProgressTimeout: serveStreamProgressTimeout(*sf.streamProgressTimeout),
		// #10638 soft diagnostic: passed through verbatim (0 = derive-from-hard). It already
		// uses the negative-off encoding, so no front-door translation is needed.
		StreamSoftProgressTimeout: *sf.streamSoftProgressTimeout,
		LlamaSlotAffinity:         *sf.llamaSlotAffinity,
		LlamaSoftSlot:             *sf.llamaSoftSlot,
		LlamaSoftSlotPolicy:       *sf.llamaSoftSlotPolicy,
		RichDashboards: gateway.RichDashboardConfig{
			ApplianceProfile: *sf.applianceObservability || rt.strixPreflight.Detected,
		},
	})
	if err != nil {
		if controlIngress != nil {
			_ = controlIngress.Close()
		}
		return nil, err
	}
	if sf.toolprocReuseRoot != nil && strings.TrimSpace(*sf.toolprocReuseRoot) != "" {
		coalesce := sf.toolprocReuseQueries != nil && *sf.toolprocReuseQueries
		srv.SetToolprocReuse(toolproc.ArmedConfig{CoalesceQueries: coalesce}, toolproc.FileDigest(*sf.toolprocReuseRoot))
	}
	if native, ok := abi.Engine(*sf.engineID).(*modelengine.Engine); ok {
		srv.SetKVPreemptor(native.KVPreemptionMetricWriter())
		if err := native.KVPreemptionEnvErr(); err != nil {
			startupMessages = append(startupMessages, newServeStartupMessage("serve", "native-kv-env", "warning", modelengine.NativeKVEnvNotice(err)))
		}
	}
	srv.AddStartupMessages(startupMessages...)
	// fak#13567: gateway.New built the in-kernel planner, which performs the Metal residency
	// promotion; sample the LM-head route NOW so the operator sees the head decode will use
	// (resident-layout was formatted before promotion).
	if line := serveLMHeadReadyLine(rt.inKernelModel); line != "" {
		fmt.Fprintln(os.Stderr, "fak serve: "+line)
		srv.AddStartupMessages(serveStartupMessage("lm-head", "info", line))
	}
	srv.SetModelLoadProfile(rt.loadProfile)
	srv.SetExecutionPolicy(rt.executionPolicy.report())
	if err := rt.configureServeArmLeases(sf, srv); err != nil {
		if controlIngress != nil {
			_ = controlIngress.Close()
		}
		return nil, err
	}
	armSetupComplete := false
	defer func() {
		if !armSetupComplete {
			buildErr = errors.Join(buildErr, rt.closeServeArmLeases())
			if controlIngress != nil {
				buildErr = errors.Join(buildErr, controlIngress.Close())
			}
		}
	}()
	if rt.inKernelModel != nil && rt.inKernelTok != nil && strings.TrimSpace(*sf.baseURL) == "" && len(sf.replicaBaseURLs.Values()) == 0 {
		controller, message, err := newServeNativeAdmissionController(sf)
		if err != nil {
			return nil, err
		}
		srv.SetAdmissionController(controller)
		srv.AddStartupMessages(message)
	}
	if maxTokens := sf.effectiveMaxTotalTokens(); maxTokens > 0 {
		srv.SetMaxTotalTokens(maxTokens)
		if err := srv.CheckMaxTotalTokens(maxTokens); err != nil {
			return nil, err
		}
	}
	// Per-tenant admission gates (#14453): --tenant-* arms the request-rate, in-flight,
	// and token-window gates PER SERVER, so one noisy tenant can no longer starve the
	// others on a shared serve. All-off leaves them nil — the served request path stays
	// byte-for-byte historical; a malformed value fails startup loud.
	if tenantMsgs, err := installTenantAdmissionGates(sf, srv); err != nil {
		return nil, err
	} else {
		startupMessages = append(startupMessages, tenantMsgs...)
	}
	// Control-plane SPEND CAP (#4859, the CLI half of #3273): --spend-cap builds the
	// governor, --spend-scope-trace the trace->ScopeKey resolver, and --budget-webhook is
	// wired as the breach sink. No --spend-cap ⇒ a nil governor and no SetSpendGovernor
	// call at all, so the served request path is byte-for-byte historical. A malformed or
	// unenforceable cap fails startup loud (serve_spend.go) — a budget that silently does
	// not bind is worse than a refused boot.
	gov, scopeOf, err := buildSpendGovernor(sf.spendCap.Values(), *sf.spendScopeTrace, *sf.budgetWebhook)
	if err != nil {
		return nil, err
	}
	if gov != nil {
		srv.SetSpendGovernor(gov, scopeOf)
		srv.AddStartupMessages(gateway.StartupMessage{Source: "serve", Kind: "spend-cap", Level: "info", Text: fmt.Sprintf("spend cap armed on %d scope budget(s)", len(sf.spendCap.Values()))})
	}
	if rt.useMetal || (sf.memoryGovernor != nil && *sf.memoryGovernor) || os.Getenv("FAK_MEMORY_GOVERNOR") == "1" {
		hw := macobs.CollectHardware(context.Background())
		cfg := macobs.DefaultHeadroomConfig()
		var govOpts []macobs.GovernorOption
		if rawCeiling := os.Getenv("FAK_WIRED_MEMORY_CEILING_BYTES"); rawCeiling != "" {
			if ceilingBytes, err := strconv.ParseUint(rawCeiling, 10, 64); err == nil && ceilingBytes > 0 {
				govOpts = append(govOpts, macobs.WithGovernorCeilingBytes(ceilingBytes))
			}
		}
		memGov := macobs.NewMemoryGovernor(hw, cfg, govOpts...)
		srv.SetMemoryGovernor(memGov)
		srv.AddStartupMessages(gateway.StartupMessage{
			Source: "serve",
			Kind:   "memory-governor",
			Level:  "info",
			Text:   fmt.Sprintf("zero-swap memory governor armed (wired ceiling: %d MB, 24-agent target)", memGov.Telemetry().WiredMemoryCeilingBytes/(1024*1024)),
		})
	}
	rt.srv = srv
	armSetupComplete = true
	return controlIngress, nil
}

// persistCacheValueObservations writes the post-session cache-value ledger row (tagged
// kind/name) and persists this session's OBSERVED provider-cache window so a later
// `fak vcache score` reports the REALIZED multiplier from real traffic instead of the
// synthetic-Zipf forecast (#1090). Best-effort: a write failure never fails the session.
// It is the shared shutdown tail of the guard and serve (stdio + http) front doors.
func persistCacheValueObservations(srv *gateway.Server, kind, name, provider string) {
	appendObservedCacheSavings(kind, provider, name, srv.AdjudicationSummary())
	if turns, _ := srv.VCacheTurnsSnapshot(); len(turns) > 0 {
		_, _, _ = writeConfiguredVCacheSnapshot(turns)
		persistVCacheCalibration(provider, kind+":"+name, turns)
	}
}

// gatewayUsageCounters folds a live gateway Server's exported counter accessors
// (KernelCounters + AdjudicationSummary) into a gatewayusageledger.Counters snapshot
// — the #1610 bridge between the gateway's in-memory-only counter family and the
// durable ledger. It is the ONLY place cmd/fak knows the shape of both source
// structs; internal/gatewayusageledger itself stays free of any internal/gateway or
// internal/kernel import.
func gatewayUsageCounters(srv *gateway.Server) gatewayusageledger.Counters {
	kc := srv.KernelCounters()
	adj := srv.AdjudicationSummary()
	return gatewayusageledger.Counters{
		Submits:      kc.Submits,
		VDSOHits:     kc.VDSOHits,
		EngineCalls:  kc.EngineCalls,
		Denies:       kc.Denies,
		Transforms:   kc.Transforms,
		Quarantines:  kc.Quarantines,
		ResultDenies: kc.ResultDenies,
		Admitted:     kc.Admitted,

		Total:       adj.Total,
		Allowed:     adj.Allowed,
		Denied:      adj.Denied,
		Transformed: adj.Transformed,
		Quarantined: adj.Quarantined,
		Deferred:    adj.Deferred,
		Escalated:   adj.Escalated,
		Errored:     adj.Errored,

		// The REAL per-session turn count (see gatewayusageledger.Counters.ObservedTurns):
		// Submits above is 0 on the guard proxy path, so this — not CachedTurns — is the
		// honest session-length signal the DefaultAssumedSessionTurns calibration percentiles.
		ObservedTurns: srv.HarnessCoherenceSummary().ObservedTurns,

		InputTokens:          adj.InputTokens,
		OutputTokens:         adj.OutputTokens,
		CachedPromptTokens:   adj.CachedPromptTokens,
		CachedTurns:          adj.CachedTurns,
		CacheCreationTokens:  adj.CacheCreationTokens,
		KVPrefixPromptTokens: adj.KVPrefixPromptTokens,
		KVPrefixReusedTokens: adj.KVPrefixReusedTokens,

		// WHO ACTUALLY SERVED THE VOLUME above — the self-hosted split, carried
		// from the routing decision the planner makes per request. The two groups
		// are disjoint subsets of InputTokens/OutputTokens; what falls in neither
		// is the volume whose side this build could not resolve.
		//
		// Six omitempty fields on both ends, and that is load-bearing: a gateway
		// that classified nothing leaves all six at zero here, they never reach
		// the wire, and an absent field keeps reading NOT INSTRUMENTED. Filling
		// any of them with a derived or defaulted value would convert "nobody
		// measured" into "everyone paid a vendor" — or worse, the reverse.
		SelfHostedTurns:        adj.SelfHostedTurns,
		SelfHostedInputTokens:  adj.SelfHostedInputTokens,
		SelfHostedOutputTokens: adj.SelfHostedOutputTokens,
		VendorTurns:            adj.VendorTurns,
		VendorInputTokens:      adj.VendorInputTokens,
		VendorOutputTokens:     adj.VendorOutputTokens,

		CompactionFired:           adj.CompactionFired,
		CompactionBailed:          adj.CompactionBailed,
		CompactionOff:             adj.CompactionOff,
		CompactionDroppedTurns:    adj.CompactionDroppedTurns,
		CompactionRestoredTurns:   adj.CompactionRestoredTurns,
		CompactionShedTokens:      adj.CompactionShedTokens,
		CompactionCacheReadTokens: adj.CompactionCacheReadTokens,
		CompactionBailReasons:     adj.CompactionBailReasons,
		CompactionAnchorStarved:   adj.CompactionAnchorStarved,

		ToolPruneTurns: adj.ToolPruneTurns,
		ToolPruneCount: adj.ToolPruneCount,

		DenyAllStops: adj.DenyAllStops,

		// WHY turns failed upstream (#5487) — carried off the in-memory /metrics surface
		// and into the durable row, which is the only record that survives a
		// per-invocation `fak guard` process. Note adj.Errored above does NOT cover this:
		// it counts kernel adjudication ERROR verdicts, a different population, so before
		// this line a stalled turn moved nothing at all in the row. Sourced from the FULL
		// snapshot on purpose — RotationEvidenceSnapshot and TransientWireErrorSnapshot
		// are deliberately narrower and would both drop "stalled". Deliberately nil when
		// the session hit no upstream error, so the omitempty field stays ABSENT (not
		// instrumented) instead of asserting a measured zero.
		UpstreamErrorKinds: srv.UpstreamErrorKindsSnapshot(),

		CacheTTLUpgradesUpgraded: adj.CacheTTLUpgraded,
		CacheTTLUpgradeReasons:   adj.CacheTTLUpgradeReasons,

		ByReason: adj.ByReason,
	}
}

// persistGatewayUsageObservation appends ONE "exit" row to the gateway-usage ledger
// (#1610) — the full served-turn counter-family snapshot, restart-durable via the
// same append-only JSONL pattern persistCacheValueObservations already uses for the
// narrower cache-value axis (#1303). Best-effort: a write failure never fails the
// session. context is a free-form label (e.g. "http"/"stdio").
//
// sessionID is the row's JOIN KEY back to a named session, and it is the caller's job to
// pass one ONLY when the row really describes a single session — see
// gatewayUsageSessionID. The ledger's session_id has been optional since the schema
// landed and every caller passed "", so no historical row carries one; that is why the
// fleet exporter publishes an identified/unidentified census rather than assuming the
// per-session drill-down covers the corpus.
func persistGatewayUsageObservation(srv *gateway.Server, sessionType, context string, uptime time.Duration, sessionID string) {
	stats := cacheobs.Default.Snapshot()
	if stats.Turns > 0 {
		_ = cachevalueledger.AppendSession(sessionType, context, sessionID, nightrunLedgerPath(cachevalueledger.DefaultLedgerRel), stats)
	}
	row := gatewayusageledger.NewRow("exit", sessionType, context, sessionID, uptime, gatewayUsageProvenance(srv), gatewayUsageCounters(srv), time.Now())
	if err := gatewayusageledger.Append(nightrunLedgerPath(gatewayusageledger.DefaultLedgerRel), row); err != nil {
		fmt.Fprintf(os.Stderr, "fak: gateway-usage ledger append failed (non-fatal): %v\n", err)
	}
}

// guardSharedTraceSentinel is the CONSTANT trace resolveGuardSessionID hands an ordinary
// non-durable `fak guard` launch. Every such launch on every machine gets this same
// string, so it identifies the guard PATH, not a session.
const guardSharedTraceSentinel = "guard"

// gatewayUsageSessionID returns the id a usage row may be stamped with, or "" when the
// caller's trace is not a per-session identity.
//
// WHY THIS FILTER EXISTS. The join key is only useful if it is unique per session, and a
// non-durable guard launch resolves to the shared "guard" sentinel. Stamping that would be
// strictly WORSE than leaving the field empty: thousands of unrelated sessions would
// collapse into one enormous series on a per-session panel, and it would look like a real
// session rather than like missing data. An empty id is honestly absent — it shows up in
// the exporter's unidentified census — while a shared id is a wrong answer that reads as a
// right one.
func gatewayUsageSessionID(trace string) string {
	trace = strings.TrimSpace(trace)
	if trace == "" || trace == guardSharedTraceSentinel {
		return ""
	}
	return trace
}

// serveUsageSessionID names the session a `fak serve` exit row describes — but ONLY when
// the process hosted exactly one.
//
// A serve gateway MULTIPLEXES a session table, so unlike guard (one process, one wrapped
// agent) its exit counters are a process total that can span many traces. Stamping any one
// of those traces would attribute the WHOLE process's tokens, turns and verdicts to a
// single session — a per-session cost panel would then show one session having spent what
// five of them did, and nothing on the panel would reveal the blend.
//
// So the rule is the one case where the process total and the session total are the same
// number: Len()==1. A single-session serve (the dispatch launcher's usual shape — one
// worker process for one lane's work) becomes drillable; a genuinely multiplexed one stays
// honestly unidentified and lands in the exporter's unidentified census, where it reads as
// "this row covers more than one session" instead of as a wrong attribution.
//
// The table is read at EXIT, after the last turn, so Len() is the count of sessions that
// survived to the end. An LRU-evicted mid-run session is not counted — which is the
// conservative direction: eviction can only turn a would-be-stamped row into an
// unidentified one, never the reverse.
func serveUsageSessionID(tbl *session.Table) string {
	if tbl == nil || tbl.Len() != 1 {
		return ""
	}
	snap := tbl.Snapshot()
	if len(snap) != 1 {
		return ""
	}
	return gatewayUsageSessionID(snap[0].TraceID)
}

// gatewayUsageProvenance stamps the calibration-relevant config the Server actually ran
// under (not what a flag said — the Server is the source of truth) plus the fak binary's
// build revision, so a gateway-usage row is self-describing: a reader recomputing the
// DefaultAssumedSessionTurns / headHorizonHeavyResidentFloor calibration can exclude
// override sessions and scope to a known build. The build revision is the short VCS SHA
// (binstamp), suffixed "-dirty" for an uncommitted build; an unstamped build (e.g. `go
// run`) leaves it empty, which the ledger omits.
func resolvedGatewayExposeProfile(profile string) string {
	profile = strings.ToLower(strings.TrimSpace(profile))
	if profile == "" {
		return "interactive"
	}
	return profile
}

func gatewayUsageProvenance(srv *gateway.Server) *gatewayusageledger.Provenance {
	rev := binstamp.Self().Revision
	if len(rev) > 12 {
		rev = rev[:12]
	}
	if rev != "" && binstamp.Self().Dirty {
		rev += "-dirty"
	}
	return &gatewayusageledger.Provenance{
		AssumeSessionTurns:   srv.AssumeSessionTurns(),
		CompactHistoryBudget: srv.CompactHistoryBudget(),
		ExposeProfile:        resolvedGatewayExposeProfile(srv.ExposeProfile()),
		BuildRevision:        rev,
	}
}

// startGatewayUsageSnapshotLoop starts the optional --metrics-snapshot periodic
// ledger writer (#1610) for a long-lived `fak serve`: every interval it appends a
// "periodic" row so a crash before a clean exit still leaves an OBSERVED counter
// trail on disk. interval<=0 disables it (byte-for-byte no-op, the default). The
// returned stop func cancels the loop; it is safe to call even when the loop was
// never started. The loop also exits on its own once ctx is done, so a caller that
// forgets to invoke stop still cannot leak the goroutine past the serve lifecycle.
func startGatewayUsageSnapshotLoop(ctx context.Context, srv *gateway.Server, interval time.Duration, sessionType string, startedAt time.Time) func() {
	if interval <= 0 {
		return func() {}
	}
	loopCtx, cancel := context.WithCancel(ctx)
	// Anchor to the repo root ONCE (not per tick) so the periodic append lands in the
	// real docs/nightrun regardless of cwd, without re-walking the tree every interval.
	ledgerPath := nightrunLedgerPath(gatewayusageledger.DefaultLedgerRel)
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-loopCtx.Done():
				return
			case <-t.C:
				now := time.Now()
				row := gatewayusageledger.NewRow("periodic", sessionType, "snapshot", "", now.Sub(startedAt), gatewayUsageProvenance(srv), gatewayUsageCounters(srv), now)
				if err := gatewayusageledger.Append(ledgerPath, row); err != nil {
					fmt.Fprintf(os.Stderr, "fak: gateway-usage periodic snapshot failed (non-fatal): %v\n", err)
				}
			}
		}
	}()
	return cancel
}

func validateServeHostSwapByteLimit(limit int64) error {
	if limit < 0 {
		return fmt.Errorf("--native-kv-host-swap-bytes must be nonnegative")
	}
	return nil
}

// The explicit flag reaches the selected ordinary syscall engine before
// gateway construction/listener bind. It does not configure chat planners or
// remote providers and does not implicitly arm KV pressure or recompute.
func configureServeHostSwapBudget(sf *serveFlags, native *modelengine.Engine) error {
	if sf == nil || sf.nativeHostSwapBytes == nil {
		return nil
	}
	limit := *sf.nativeHostSwapBytes
	if err := validateServeHostSwapByteLimit(limit); err != nil {
		return err
	}
	if limit == 0 {
		return nil
	}
	if native == nil {
		return fmt.Errorf("--native-kv-host-swap-bytes requires the selected native engine")
	}
	if native.KVPreemptionStats().MaxBlocks <= 0 {
		return fmt.Errorf("--native-kv-host-swap-bytes requires a positive native KV block budget")
	}
	return native.SetKVHostSwapByteLimit(limit)
}
