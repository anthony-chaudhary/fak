package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/snapshot"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
)

type repeatedStringFlag []string

func (f *repeatedStringFlag) String() string {
	if f == nil {
		return ""
	}
	return strings.Join(*f, ",")
}

func (f *repeatedStringFlag) Set(value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return errors.New("value must be non-empty")
	}
	*f = append(*f, value)
	return nil
}

func (f *repeatedStringFlag) Values() []string {
	if f == nil || len(*f) == 0 {
		return nil
	}
	out := make([]string, len(*f))
	copy(out, *f)
	return out
}

// debugStatsSink returns the per-turn debug sink for `--debug-stats` (#793): a stderr
// line-writer when on, nil (the no-op default) when off. The gateway emits one compact,
// payload-free line per served turn through it.
func debugStatsSink(on bool) func(string, ...any) {
	if !on {
		return nil
	}
	return func(format string, args ...any) {
		fmt.Fprintf(os.Stderr, format+"\n", args...)
	}
}

func configureServeToolEngines() {
	// Serve exposes fak_read over MCP even when it is not running the demo agent loop.
	// Register only the confined read miss engine; agent.Configure would also install
	// the demo airline tool policy and is intentionally not part of serve startup.
	agent.RegisterReadEngine("")
}

// serveFlags holds the parsed `fak serve` flag values, one field per flag in
// definition order, so the boot stages consume them without threading four dozen
// locals through every call.
type serveFlags struct {
	addr                        *string
	stdio                       *bool
	provider                    *string
	baseURL                     *string
	replicaBaseURLs             repeatedStringFlag
	model                       *string
	apiKeyEnv                   *string
	engineCacheEngine           *string
	engineCacheBaseURL          *string
	engineCacheAdminKeyEnv      *string
	engineCacheIdleTimeout      *time.Duration
	engineCacheRequireExactSpan *bool
	engineID                    *string
	backendName                 *string
	cudaGraph                   *bool
	policyPath                  *string
	policyCheck                 *bool
	vdso                        *bool
	invalidation                *string
	requireKeyEnv               *string
	routeManifest               *string
	ggufPath                    *string
	tokPath                     *string
	ctxViewBudget               *int
	compactHistoryBudget        *int
	compactAnchorHead           *bool
	elideResultBytes            *int
	sessionID                   *string
	sessionStatePath            *string
	contextBudgetTokens         *int
	resetOnBudget               *bool
	cpuOffloadExperts           *bool
	metal                       *bool
	expertParallel              *int
	tensorParallel              *int
	budgetWebhook               *string
	budgetWarnFraction          *float64
	notifyNative                *bool
	notifyWebhook               *string
	notifySlack                 *string
	debugStats                  *bool
	dojoMode                    *bool
	native                      *bool
	nativeMaxTurns              *int
	vdsoProxyFill               *bool
	metricsSnapshot             *time.Duration
}

// newServeFlagSet defines the full `fak serve` flag surface and returns the set
// plus the struct the boot stages read the parsed values through.
func newServeFlagSet() (*flag.FlagSet, *serveFlags) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	sf := &serveFlags{}
	sf.addr = fs.String("addr", "127.0.0.1:8080", "HTTP listen address (OpenAI + fak + /mcp surface); ignored with --stdio")
	sf.stdio = fs.Bool("stdio", false, "serve MCP over stdin/stdout (newline-delimited JSON-RPC) instead of HTTP")
	sf.provider = fs.String("provider", "openai", "upstream provider transcript wire: openai, anthropic, gemini, or xai")
	sf.baseURL = fs.String("base-url", "", "upstream provider base URL for the /v1/chat/completions proxy (empty = offline mock planner)")
	fs.Var(&sf.replicaBaseURLs, "replica-base-url", "additional upstream provider base URL for a static round-robin replica fleet; repeat for N replicas. If --base-url is set, it is replica 1.")
	sf.model = fs.String("model", "mock", "model id (advertised by /v1/models; used for the upstream call)")
	sf.apiKeyEnv = fs.String("api-key-env", "", "env var holding the upstream API key (proxy mode)")
	sf.engineCacheEngine = fs.String("engine-cache-engine", "", "self-hosted upstream cache reset engine for quarantined provider-bound tool results: sglang|vllm (empty disables)")
	sf.engineCacheBaseURL = fs.String("engine-cache-base-url", "", "serving-engine control/base URL for cache reset (default: --base-url when --engine-cache-engine is set)")
	sf.engineCacheAdminKeyEnv = fs.String("engine-cache-admin-key-env", "", "env var holding the serving-engine admin API key for cache reset")
	sf.engineCacheIdleTimeout = fs.Duration("engine-cache-idle-timeout", 0, "SGLang /flush_cache idle timeout, e.g. 30s (0 fails fast)")
	sf.engineCacheRequireExactSpan = fs.Bool("engine-cache-require-exact-span", false, "require exact remote K/V/index span eviction; fail closed if the selected engine only supports whole-cache reset")
	sf.engineID = fs.String("engine", "inkernel", "registered engine id that fak_syscall dispatches an allowed call to: inkernel, mock, vllm, sglang, llm-d, dynamo, or another registered driver (default: the fused in-kernel model)")
	sf.backendName = fs.String("backend", "", "compute backend for the in-kernel chat decode (with --gguf, no --base-url): empty = the CPU reference path; a registered device name like 'cuda' runs prefill+decode through the GPU HAL. Requires a `-tags cuda` build AND a reachable GPU at runtime; fails loud if named but unavailable so a typo never silently runs on CPU.")
	sf.cudaGraph = fs.Bool("cuda-graph", false, "with --backend cuda: capture each decode token's whole op stream into a CUDA graph and replay it as ONE launch instead of N kernel launches (#483), the per-token launch-overhead lever for large single-stream decode (e.g. Qwen3.6-27B on an A100). OFF by default (a measured no-win on a tiny 0.5B/L4 where launch overhead is already small); witness tok/s before/after on YOUR node before relying on it. Equivalent to FAK_CUDA_GRAPH=1; inert on a non-cuda build or CPU backend.")
	sf.policyPath = fs.String("policy", "", "capability-floor manifest to load (default: the built-in adjudicator floor — the tau2 airline-demo tools, NOT the `fak guard` coding floor; see `fak policy --dump`)")
	sf.policyCheck = fs.Bool("policy-check", false, "validate --policy and exit without binding a listener")
	sf.vdso = fs.Bool("vdso", true, "enable the vDSO dedup fast path")
	sf.invalidation = fs.String("invalidation", "global", "vDSO tier-2 invalidation granularity for the live fleet: global|namespace|resource")
	sf.requireKeyEnv = fs.String("require-key-env", "", "env var holding a bearer token to REQUIRE on every request (default: no auth)")
	sf.routeManifest = fs.String("route-manifest", "", "model-routing policy to install: each fak_syscall call is classified into a modelroute.Subject and a single-model (PICK) plan binds abi.ToolCall.Engine before Submit, so the residency PDP adjudicates the real route (#601). Empty (default) leaves Engine unset → the kernel default engine, byte-for-byte the pre-routing behavior. A malformed manifest fails startup loud (a mis-routed model is a security boundary, never a silent default). The installed file is HOT-RELOADED: an edit is picked up without a restart and swapped atomically (a request classifies against the whole old or whole new policy, never a torn read); a malformed edit is rejected and the last-good policy stays installed (#842).")
	sf.ggufPath = fs.String("gguf", "", "load these GGUF weights into the in-kernel engine at boot; the load is part of the measured startup sequence and its phase breakdown is exposed on /metrics. Default path is lean-Q8 (Q4→f32→Q8 round-trip); set FAK_Q4K=1 for the direct-resident-Q4_K path (Qwen3.6-27B q4_k_m, the P1/P2 decode lever)")
	sf.tokPath = fs.String("tokenizer", "", "OPTIONAL override for the in-kernel CHAT planner's tokenizer. With --gguf and no --base-url, /v1/chat/completions AND /v1/messages already serve the in-kernel model (real ChatML chat) using the GGUF's EMBEDDED tokenizer; pass this only to override it (e.g. an SPM-only checkpoint with no embedded BPE tokenizer, or a custom vocab). Accepts a tokenizer.json or its directory. e.g. ~/.cache/fak-models/tokenizers/qwen3.6")
	sf.ctxViewBudget = fs.Int("ctx-view-budget", 8000, "wire the ctxplan context PLANNER into the live serve loop: each buffered turn, re-materialize the forwarded history as an O(1) planned VIEW under this resident-token budget (a planned view in place of appending the whole transcript, #555). DEFAULT-ON at a conservative 8000 resident tokens; pass 0 to disable (leaves the existing path byte-for-byte unchanged). The planner only ever SHORTENS and falls open to the full history on any doubt; on the Anthropic passthrough it keeps the cached prefix byte-identical (witness: docs/notes/CTXVIEW-DEFAULT-ON-WITNESS-2026-06-28.md). The streaming fast-path bypasses this; the buffered turn path is what gets planned.")
	sf.compactHistoryBudget = fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "on the Anthropic PASSTHROUGH (an upstream --base-url anthropic), compact OLD conversation turns in the OUTBOUND request body down to this resident-token budget while keeping the cache_control prefix BYTE-IDENTICAL, so the upstream cache hit survives. This reaches the flagship passthrough the streaming ctxplan view cannot (#555). DEFAULT-ON: once a conversation sprawls past ~48k resident tokens the cut fires and sheds the un-cacheable middle the provider re-bills every turn; a typical short session stays untouched. Pass 0 to disable (body forwarded byte-for-byte). No effect on non-passthrough wires.")
	sf.compactAnchorHead = fs.Bool("compact-anchor-head", false, "re-anchor --compact-history-budget's protected prefix on the stable system/tools head instead of the default first-breakpoint anchor, fixing the anchor-starved trap (#1407) where real Claude Code traffic's recent cache_control breakpoint protects almost the whole conversation so the budget can never shed anything. OPT-IN: re-anchoring bursts the recent breakpoint's cached suffix once, so it only fires when the burst repays (CacheBurstPaysBack, #1408) — without a wired session-turn horizon it only fires zero-penalty bursts.")
	sf.elideResultBytes = fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes, "ON by default at gateway.DefaultElideResultBytes (the reviewed gateway.DocumentedElideResultBytes threshold): shrink oversized tool_result bodies outside the active working set to a bounded head+tail form once they exceed this byte threshold. 0 disables.")
	sf.sessionID = fs.String("session-id", "", "default trace/session id for callers that omit X-Trace-Id or MCP trace_id (empty = mint gw-N per request unless --context-budget-tokens is set)")
	sf.sessionStatePath = fs.String("session-state", "", "COLD-RESUME the per-session DRIVE state across a process restart (#629): a fleet-snapshot file this `fak serve` RESTORES at boot — re-attaching every session at the budget/priority/run-state/pace it held, not its defaults (a STOPPED session reloads STOPPED with its reason, never silently RUNNING) — and REWRITES on a clean shutdown. Empty (default) = off, byte-for-byte today's path. Distinct from the live Paused→Running resume the /v1/fak/session control verbs already do.")
	sf.contextBudgetTokens = fs.Int("context-budget-tokens", 0, "seed the default session with this prompt/context-token budget; exhaustion returns a reset directive with continuation_id (0 = off)")
	sf.resetOnBudget = fs.Bool("reset-on-budget", false, "on context-budget exhaustion, re-arm the continuation trace with a carryover seed and continue transparently instead of returning 409 (requires --context-budget-tokens)")
	sf.cpuOffloadExperts = fs.Bool("cpu-offload-experts", false, "with --gguf --backend: keep the MoE expert GEMMs on host RAM while dense projections + router + attention run on the device — the `--n-cpu-moe` hybrid that lets a model whose experts dwarf VRAM (e.g. GLM-5.2 Q4 ~424GB experts) serve at all on a smaller VRAM pool. The device load uses the memory-lean Q8 quantize-at-load path when the backend advertises quantized upload; otherwise it falls back to F32 weights until that backend implements UploadDtype.")
	sf.metal = fs.Bool("metal", false, "with --gguf (no --base-url), require the Apple-Silicon Metal GPU forward — GPU prefill + GPU-resident Q8 decode (#67, ~0.99x of llama.cpp-Metal on dense Qwen2.5-7B Q8). Apple-Silicon+cgo builds auto-select Metal when a usable device is present; this flag/FAK_METAL=1 makes absence fail loud instead of falling back to CPU. Mutually exclusive with --backend (Metal is the CPU-session seam, not a compute HAL device). Dense Qwen-class Q8 GGUFs only — a MoE/hybrid model (GLM-5.2, GDN) self-declines to CPU decode.")
	sf.expertParallel = fs.Int("expert-parallel", 1, "with --gguf: shard the routed MoE experts of a glm_moe_dsa model (GLM-5.2) across N expert-parallel ranks — the lever to move the expert GEMM off the host (the `--cpu-offload-experts` wall) onto resident GPUs (#971). The per-rank residual partials are reduced by one AllReduceSum through the wired Collective. 1 (default) = the unchanged monolith forward. N>1 requires an initialized non-cpu-ref compute.CollectiveBackend; CUDA builds provide that only with -tags cuda,nccl (build_cuda.sh: FAK_CUDA_NCCL=1) on a box with enough visible GPUs.")
	sf.tensorParallel = fs.Int("tensor-parallel", 1, "with --gguf: tensor-parallel rank count for the dense projections (the Megatron column/row split, tensor_parallel.go). 1 (default) = no split. N>1 uses the same initialized device-collective gate as --expert-parallel; CUDA builds require -tags cuda,nccl (build_cuda.sh: FAK_CUDA_NCCL=1).")
	sf.budgetWebhook = fs.String("budget-webhook", "", "POST a JSON event to this URL when a served session's context budget crosses the warning threshold (--budget-warn-fraction) or is exhausted (the reset trigger), so an operator/monitor is notified before exhaustion (#743). Empty = off. Needs --context-budget-tokens to have a budget to watch.")
	sf.budgetWarnFraction = fs.Float64("budget-warn-fraction", 0.8, "consumed share (0..1) of the context budget at which --budget-webhook fires its pre-exhaustion warning (default 0.8 = 80%); <=0 or >=1 disables the warning while the exhaustion event still fires")
	sf.notifyNative = fs.Bool("notify-native", true, "emit a one-line native notification to stderr when a served session hits a PAUSED/DRAINING/STOPPED or budget boundary, carrying the closed stop-reason token — the SIGCHLD-equivalent so a waiting agent is never silent (#761); default on")
	sf.notifyWebhook = fs.String("notify-webhook", "", "POST a JSON StopEvent to this URL on each served-session terminal/paused/budget boundary (#761), carrying the closed reason token; empty = off. Extends the #743 budget webhook to the full stop-reason vocabulary.")
	sf.notifySlack = fs.String("notify-slack", "", "POST a Slack incoming-webhook payload ({\"text\":…}) on each served-session boundary (#761); empty = off")
	sf.debugStats = fs.Bool("debug-stats", false, "print ONE compact, payload-free line per served turn to stderr: request/cache_read/cache_creation tokens, the compaction action, and the resetScore SHADOW health (healthy_cache|cache_decay|stale_prefix|cooldown|unknown_provider). Independent of --log (#793); default off.")
	sf.dojoMode = fs.Bool("dojo", false, "enable live dojo mode: write a start-marker for each serve session into the live-episode corpus (.dojo/live-episodes/ under the workspace root) for issue #956. NOTE: live-episode scoring is not yet wired into `fak dojo run` (which today scores Claude Code transcripts passed via --corpus), so this records the boundary but does not yet feed the scorer.")
	sf.native = fs.Bool("native", false, "NATIVE HARNESS (#1316): drive fak's OWN agent loop (agent.RunArm) for a non-streaming /v1/messages turn instead of the single-shot proxy turn — fak owns dispatch, the in-kernel syscall boundary is the sole tool path, and the per-turn session gate + per-call routing + operator steer bus run on the served loop. The loop is seeded with the request's last user message and drives the kernel-owned tool catalog to a final answer; the per-turn ArmMetrics ride back on the response `fak.native_arm` extension. Off by default (the proxy path is byte-for-byte unchanged). A streaming request falls through to the proxy path.")
	sf.nativeMaxTurns = fs.Int("native-max-turns", gateway.DefaultNativeMaxTurns, "with --native: cap the owned loop's model round-trips per served request (<=0 uses the built-in default)")
	sf.vdsoProxyFill = fs.Bool("vdso-proxy-fill", false, "warm the vDSO from ADMITTED inbound tool_result blocks on the proxy path: an allowed, read-only-shaped result the client sends back fills (tool,args)->result so a LATER identical read is served inline (no client re-execution). Off by default — sound only when the principal is named and writes that touch the same resource reach fak (a proxy-closed world), so it is an explicit operator opt-in. Scoped per-principal; never fills a Shareable or write-shaped tool.")
	sf.metricsSnapshot = fs.Duration("metrics-snapshot", 0, "periodically append an interim gateway-usage counter snapshot (internal/gatewayusageledger, docs/nightrun/gateway-usage.jsonl) while this long-lived `fak serve` is up, so a crash before a clean exit still leaves a trail (#1610). 0 (default) disables periodic snapshots; the exit-time snapshot is always written regardless of this flag.")
	return fs, sf
}

func cmdServe(argv []string) {
	// t0 anchors the boot timeline exposed as fak_gateway_time_to_ready_seconds; it
	// must be the FIRST statement so flag parse + policy + weight load are accounted.
	t0 := time.Now()
	fs, sf := newServeFlagSet()
	tParse := time.Now()
	_ = fs.Parse(argv)
	parseDur := time.Since(tParse)

	resolveServeModelSources(sf)

	// --policy-check: validate the manifest and exit, binding no listener.
	if *sf.policyCheck {
		runServePolicyCheck(*sf.policyPath)
		return
	}

	// Install the capability floor fail-loud: a bad manifest aborts startup rather
	// than silently falling back to a more permissive default. Time it as the first
	// startup phase.
	tPolicy := time.Now()
	applyPolicy(*sf.policyPath)
	rt := &serveRuntime{t0: t0, startupPhases: []gateway.StartupPhase{
		{Name: "flag-parse", Dur: parseDur},
		{Name: "policy-load", Dur: time.Since(tPolicy)},
	}}
	configureServeToolEngines()

	// Boot stages (serve_stages.go). The order is load-bearing: compute before the
	// weight load, the session plane before the gateway, the observer seams resolved
	// before the gateway exists but installed only after it does.
	rt.resolveCompute(sf)
	defer rt.closeEPGroup()
	rt.loadModel(sf)
	rt.resolveSessionPlane(sf)
	rt.resolveObservers(sf)
	rt.buildGateway(sf)
	rt.wireGateway(sf)
	rt.run(sf)
}

// buildGateway loads the optional model-routing policy, constructs the gateway
// server from the resolved planes, and arms the admission controller for a pure
// in-kernel serve.
func (rt *serveRuntime) buildGateway(sf *serveFlags) {
	// Resolve the optional model-routing policy. Off by default: an empty --route-manifest
	// leaves routeMan nil, so gateway.New gets a nil RouteManifest and Engine stays unset —
	// byte-for-byte the pre-routing behavior. A malformed file fails loud here rather than
	// silently default-routing every call to the kernel default (a mis-routed model is a
	// security boundary). gateway.New also re-validates the loaded manifest.
	var routeMan *modelroute.Manifest
	if *sf.routeManifest != "" {
		loaded, err := modelroute.LoadManifest(*sf.routeManifest)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve: --route-manifest:", err)
			os.Exit(1)
		}
		routeMan = &loaded
		fmt.Printf("fak: model-routing policy loaded from %s\n", *sf.routeManifest)
	}

	srv, err := gateway.New(gateway.Config{
		EngineID:                    *sf.engineID,
		Model:                       *sf.model,
		BaseURL:                     *sf.baseURL,
		ReplicaBaseURLs:             sf.replicaBaseURLs.Values(),
		Provider:                    *sf.provider,
		APIKey:                      rt.apiKey,
		EngineCacheEngine:           *sf.engineCacheEngine,
		EngineCacheBaseURL:          *sf.engineCacheBaseURL,
		EngineCacheAdminKey:         rt.engineCacheAdminKey,
		EngineCacheIdleTimeout:      *sf.engineCacheIdleTimeout,
		EngineCacheRequireExactSpan: *sf.engineCacheRequireExactSpan,
		InKernelModel:               rt.inKernelModel,
		Tokenizer:                   rt.inKernelTok,
		InKernelQ4K:                 rt.inKernelQ4K,
		Backend:                     rt.chatBackend,
		CPUOffloadExperts:           *sf.cpuOffloadExperts,
		Metal:                       rt.useMetal,
		ExpertParallelRanks:         *sf.expertParallel,
		RequireKey:                  rt.requireKey,
		VDSO:                        *sf.vdso,
		Invalidation:                *sf.invalidation,
		Version:                     appversion.Current(),
		ReloadPolicy:                policyReloader(*sf.policyPath),
		ResetTrace:                  resetTrace,
		ObserveTrace:                observeTrace,
		ObserveSession:              observeSession,
		ControlSession:              controlSession,
		SteerSession:                steerSession,
		ListSessions:                listSessions,
		DecideSession:               decideSession,
		DebitSession:                debitSession,
		ResetOnBudget:               resetOnBudgetHook(*sf.resetOnBudget, *sf.contextBudgetTokens),
		DefaultTraceID:              rt.defaultTraceID,
		StartTime:                   rt.t0,
		StartupPhases:               rt.startupPhases,
		CtxViewBudget:               *sf.ctxViewBudget,
		CompactHistoryBudget:        *sf.compactHistoryBudget,
		CompactAnchorHead:           *sf.compactAnchorHead,
		ElideResultBytes:            *sf.elideResultBytes,
		DebugStatsf:                 debugStatsSink(*sf.debugStats),
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
		// Native-harness keystone (#1316): drive agent.RunArm for a non-streaming
		// /v1/messages turn. Off by default — the proxy path is byte-for-byte unchanged.
		Native:         *sf.native,
		NativeMaxTurns: *sf.nativeMaxTurns,
		VDSOProxyFill:  *sf.vdsoProxyFill,
	})
	must(err)
	srv.SetModelLoadProfile(rt.loadProfile)
	if rt.inKernelModel != nil && rt.inKernelTok != nil && strings.TrimSpace(*sf.baseURL) == "" && len(sf.replicaBaseURLs.Values()) == 0 {
		srv.SetAdmissionController(gateway.NewAdmissionController(gateway.DefaultAdmissionPolicy()))
	}
	rt.srv = srv
}

// persistCacheValueObservations writes the post-session cache-value ledger row (tagged
// kind/name) and persists this session's OBSERVED provider-cache window so a later
// `fak vcache score` reports the REALIZED multiplier from real traffic instead of the
// synthetic-Zipf forecast (#1090). Best-effort: a write failure never fails the session.
// It is the shared shutdown tail of the guard and serve (stdio + http) front doors.
func persistCacheValueObservations(srv *gateway.Server, kind, name, provider string) {
	stats := cacheobs.Default.Snapshot()
	if stats.Turns > 0 {
		_ = cachevalueledger.Append(kind, name, cachevalueledger.DefaultLedgerRel, stats)
	}
	appendObservedCacheSavings(kind, provider, name, srv.AdjudicationSummary())
	if turns, _ := srv.VCacheTurnsSnapshot(); len(turns) > 0 {
		_ = vcachesnapshot.Write(vcachesnapshot.DefaultPath(), turns)
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

		InputTokens:          adj.InputTokens,
		OutputTokens:         adj.OutputTokens,
		CachedPromptTokens:   adj.CachedPromptTokens,
		CachedTurns:          adj.CachedTurns,
		CacheCreationTokens:  adj.CacheCreationTokens,
		KVPrefixPromptTokens: adj.KVPrefixPromptTokens,
		KVPrefixReusedTokens: adj.KVPrefixReusedTokens,

		CompactionFired:           adj.CompactionFired,
		CompactionBailed:          adj.CompactionBailed,
		CompactionOff:             adj.CompactionOff,
		CompactionDroppedTurns:    adj.CompactionDroppedTurns,
		CompactionShedTokens:      adj.CompactionShedTokens,
		CompactionCacheReadTokens: adj.CompactionCacheReadTokens,

		ToolPruneTurns: adj.ToolPruneTurns,
		ToolPruneCount: adj.ToolPruneCount,

		DenyAllStops: adj.DenyAllStops,

		ByReason: adj.ByReason,
	}
}

// persistGatewayUsageObservation appends ONE "exit" row to the gateway-usage ledger
// (#1610) — the full served-turn counter-family snapshot, restart-durable via the
// same append-only JSONL pattern persistCacheValueObservations already uses for the
// narrower cache-value axis (#1303). Best-effort: a write failure never fails the
// session. context is a free-form label (e.g. "http"/"stdio").
func persistGatewayUsageObservation(srv *gateway.Server, sessionType, context string) {
	row := gatewayusageledger.NewRow("exit", sessionType, context, "", 0, gatewayUsageCounters(srv), time.Now())
	if err := gatewayusageledger.Append(gatewayusageledger.DefaultLedgerRel, row); err != nil {
		fmt.Fprintf(os.Stderr, "fak: gateway-usage ledger append failed (non-fatal): %v\n", err)
	}
}

// startGatewayUsageSnapshotLoop starts the optional --metrics-snapshot periodic
// ledger writer (#1610) for a long-lived `fak serve`: every interval it appends a
// "periodic" row so a crash before a clean exit still leaves an OBSERVED counter
// trail on disk. interval<=0 disables it (byte-for-byte no-op, the default). The
// returned stop func cancels the loop; it is safe to call even when the loop was
// never started. The loop also exits on its own once ctx is done, so a caller that
// forgets to invoke stop still cannot leak the goroutine past the serve lifecycle.
func startGatewayUsageSnapshotLoop(ctx context.Context, srv *gateway.Server, interval time.Duration, sessionType string) func() {
	if interval <= 0 {
		return func() {}
	}
	loopCtx, cancel := context.WithCancel(ctx)
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for {
			select {
			case <-loopCtx.Done():
				return
			case <-t.C:
				row := gatewayusageledger.NewRow("periodic", sessionType, "snapshot", "", 0, gatewayUsageCounters(srv), time.Now())
				if err := gatewayusageledger.Append(gatewayusageledger.DefaultLedgerRel, row); err != nil {
					fmt.Fprintf(os.Stderr, "fak: gateway-usage periodic snapshot failed (non-fatal): %v\n", err)
				}
			}
		}
	}()
	return cancel
}

// restoreServeSessions re-attaches the persisted DRIVE state of every session (the COLD
// resume of #629) from a fleet-snapshot file a prior `fak serve` wrote on shutdown. It is
// the load-time inverse of dumpServeSessions: each session re-attaches at the budget /
// priority / run-state / pace it held — a STOPPED session reloads STOPPED with its reason
// (session.Table.Restore is the one write that re-establishes a terminal record), never
// silently resurrected as RUNNING. An empty path is off (no-op). A missing file is a clean
// first boot (not an error). A PRESENT-but-corrupt file fails loud — a tampered/truncated
// drive record is worse than none, the same fail-closed posture the policy/route loaders
// take, and the snapshot envelope's own sha256 body digest is what catches the tamper. This
// is the process-restart half the design note SESSION-CONTROL-STATE-AS-FIRST-CLASS §5
// named; it is DISTINCT from the live Paused→Running resume the control verbs already do.
func restoreServeSessions(tbl *session.Table, path string) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	snap, err := snapshot.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // first boot — nothing persisted yet
		}
		return fmt.Errorf("--session-state %s: %w", path, err)
	}
	n, err := snap.RestoreFleet(tbl)
	if err != nil {
		return fmt.Errorf("--session-state %s: %w", path, err)
	}
	if n > 0 {
		fmt.Printf("fak: cold resume (#629) — re-attached %d session(s) drive state from %s\n", n, path)
	}
	return nil
}

// dumpServeSessions writes the live DRIVE table to path as an integrity-checked fleet
// snapshot so the NEXT `fak serve` cold-resumes it (#629). An empty path is off (no-op).
// Best-effort on a clean shutdown: a write failure is logged, never fatal — a failed dump
// must not turn a graceful stop into a crash (worst case the next boot starts at defaults,
// exactly today's behavior). A hard kill skips the dump; the last clean shutdown's file
// stands. An empty table writes an empty (still valid) snapshot.
func dumpServeSessions(tbl *session.Table, path string) {
	path = strings.TrimSpace(path)
	if path == "" {
		return
	}
	snap, err := snapshot.DumpFleet("serve", tbl, 0)
	if err == nil {
		var b []byte
		if b, err = snap.Encode(); err == nil {
			err = os.WriteFile(path, b, 0o644)
		}
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak: persist session state to %s failed: %v\n", path, err)
		return
	}
	fmt.Printf("fak: persisted live session drive state → %s (#629)\n", path)
}

// toGatewayLoadProfile mirrors a ggufload.LoadProfile into the gateway's import-
// decoupled ModelLoadProfile so the boot-time weight-load breakdown surfaces on
// /metrics. Returns nil for a nil profile (no eager load happened).
func toGatewayLoadProfile(p *ggufload.LoadProfile) *gateway.ModelLoadProfile {
	if p == nil {
		return nil
	}
	out := &gateway.ModelLoadProfile{
		Source:       p.Source,
		Mode:         p.Mode,
		TotalSeconds: float64(p.TotalNanos) / 1e9,
		Tensors:      p.TensorCount,
		Bottleneck:   p.Bottleneck,
	}
	for _, ph := range p.Phases {
		out.Bytes += ph.Bytes
		out.Phases = append(out.Phases, gateway.ModelLoadPhase{
			Phase:   ph.Phase,
			Seconds: float64(ph.Nanos) / 1e9,
			Bytes:   ph.Bytes,
			Tensors: ph.Tensors,
		})
	}
	for _, lp := range p.LoadPaths {
		out.LoadPaths = append(out.LoadPaths, gateway.ModelLoadPath{
			QuantType:       lp.QuantType,
			Expert:          lp.Expert,
			ResidentTensors: lp.ResidentTensors,
			ResidentBytes:   lp.ResidentBytes,
			DequantTensors:  lp.DequantTensors,
			DequantBytes:    lp.DequantBytes,
		})
	}
	return out
}

func withServeGGUFMemoryProfile(p *gateway.ModelLoadProfile, plan compute.MemoryPlan, be compute.Backend) *gateway.ModelLoadProfile {
	if p == nil {
		return nil
	}
	p.MemoryPlan = toGatewayLoadMemoryPlan(plan)
	if be != nil {
		p.MemoryCapacities = toGatewayLoadMemoryCapacities(be)
		if len(p.MemoryPlan) > 0 {
			p.MemoryHeadroomRatio = serveGGUFDeviceHeadroom
		}
	}
	return p
}

func toGatewayLoadMemoryPlan(plan compute.MemoryPlan) []gateway.ModelLoadMemoryDemand {
	if len(plan) == 0 {
		return nil
	}
	out := make([]gateway.ModelLoadMemoryDemand, 0, len(plan))
	for _, d := range plan {
		if d.Bytes <= 0 {
			continue
		}
		class := d.Class
		if class == "" {
			class = compute.MemoryUnknown
		}
		out = append(out, gateway.ModelLoadMemoryDemand{
			Class:  string(class),
			Scope:  string(d.ScopeOrDefault()),
			Bytes:  d.Bytes,
			Detail: d.Detail,
			DType:  d.DType,
		})
	}
	return out
}

func toGatewayLoadMemoryCapacities(be compute.Backend) []gateway.ModelLoadMemoryCapacity {
	if be == nil {
		return nil
	}
	deviceTotal, deviceFree, deviceKnown := compute.DeviceMemoryInfo(be)
	hostTotal, hostFree, hostKnown := compute.HostMemoryInfo(be)
	return []gateway.ModelLoadMemoryCapacity{
		toGatewayLoadMemoryCapacity(string(compute.MemoryScopeDevice), deviceTotal, deviceFree, deviceKnown),
		toGatewayLoadMemoryCapacity(string(compute.MemoryScopeHost), hostTotal, hostFree, hostKnown),
	}
}

func toGatewayLoadMemoryCapacity(scope string, total, free int64, known bool) gateway.ModelLoadMemoryCapacity {
	cap := gateway.ModelLoadMemoryCapacity{
		Scope:      scope,
		TotalBytes: total,
		Known:      known,
		FreeKnown:  known && free >= 0,
	}
	if !known {
		cap.TotalBytes = 0
		return cap
	}
	if cap.FreeKnown {
		cap.FreeBytes = free
	}
	return cap
}

// loadServeInKernelModel eagerly loads the GGUF weights (when ggufPath is set) BEFORE the
// listener binds, so the load counts toward time-to-ready and its phase breakdown reaches
// /metrics rather than being a lazy cost on first request. It returns the resident model
// (nil if no --gguf), whether the direct-resident-Q4_K path was taken, the load profile for
// /metrics, and the model-load startup phase (zero Name when no load happened). The path
// selection mirrors cmd/fakchat with one device-specific split: a device --backend that
// advertises quantized upload takes the lean-Q8 load, because the served planner runs
// Session.Quant=true and the HAL can consume Q8_0 directly. Backends without UploadDtype keep
// the F32 fallback until they can consume quantized resident weights. FAK_Q4K takes the
// direct-resident-Q4_K CPU path, and the CPU default is the lean-Q8 round-trip; the Q8 path
// stays byte-identical when the env is unset.
func resolveServeChatBackend(backendName string) (compute.Backend, error) {
	backendName = strings.TrimSpace(backendName)
	if backendName == "" {
		return nil, nil
	}
	be, found := compute.Lookup(backendName)
	if !found {
		return nil, fmt.Errorf("fak serve: --backend %q is not available (registered backends: %v). A device backend needs both a matching build tag (e.g. -tags %s) and a reachable device at runtime.", backendName, compute.Registered(), backendName)
	}
	return be, nil
}

// resolveServeMetal decides whether `fak serve` runs the in-kernel chat through the
// Apple-Silicon Metal GPU forward. Metal auto-selects when this binary has the backend
// linked and a usable device is present; --metal/FAK_METAL=1 only changes the unavailable
// case from CPU fallback to a fail-loud error. The error distinguishes a wrong build
// (`metalgemm.Compiled()` false → build on Apple Silicon with cgo) from a right build with
// no device (`Available()` false). Metal is the CPU-session seam (the served session keeps
// s.Backend nil and gets s.Metal=true), so it is mutually exclusive with a device --backend.
// Kept side-effect free (no os.Exit) so the decision is unit-testable; on a non-Metal build
// metalgemm.Available()/Compiled() are the stub's deterministic false.
func resolveServeMetal(flag, env bool, backendName string) (bool, error) {
	requested := flag || env
	if strings.TrimSpace(backendName) != "" {
		if requested {
			return false, fmt.Errorf("fak serve: --metal and --backend %q are mutually exclusive — Metal is the Apple-Silicon CPU-session forward, not a compute HAL device. Pass one.", backendName)
		}
		return false, nil
	}
	if !metalgemm.Available() {
		if !requested {
			return false, nil
		}
		if !metalgemm.Compiled() {
			return false, fmt.Errorf("fak serve: --metal requested but this binary has no Metal support — build on darwin/arm64 with cgo enabled.")
		}
		return false, fmt.Errorf("fak serve: --metal requested but no usable Metal device is available on this host.")
	}
	return true, nil
}

const serveGGUFDeviceHeadroom = 0.15

// refuseEPPlanIfUnfit fails the serve closed when `--expert-parallel N>1` cannot fit the model
// resident across N GPUs. It partitions the loaded model's resident weights into the replicated
// remainder and the routed experts (model.MoEResidentWeightBytes), builds the BUSIEST rank's
// per-card plan (compute.ExpertParallelPerRankPlan: replicated + largest expert band), and checks
// it against the device backend's PER-GPU capacity with the same headroom the load-time fit uses.
//
// It is FAIL-OPEN by construction (the contract every capacity check here keeps): a non-MoE model,
// a model whose weights cannot be accounted, ranks<=1, a nil backend, or a backend whose capacity is
// unknown (cpu-ref, a non-probing device) all return nil. So it can ONLY turn a KNOWN per-card
// overflow (e.g. a 434 GiB model at N=4 ≈ 118 GiB/card on 80 GiB GPUs) into a clean pre-serve
// refusal — instead of an OOM that surfaces minutes in, when rank r uploads its expert band to GPU r.
func refuseEPPlanIfUnfit(m *fakmodel.Model, be compute.Backend, ranks, contextBudgetTokens int) error {
	if m == nil || be == nil || ranks <= 1 {
		return nil
	}
	replicated, expert, ok := m.MoEResidentWeightBytes()
	if !ok {
		return nil // nothing accounted (non-MoE / unloaded) -> fail open
	}
	// KV is a per-rank cost: pure EP replicates attention, so each rank holds the full KV for the
	// context it serves. Size it from the model geometry at the context budget — the SAME KV the
	// load-time fit plan sizes from contextBudgetTokens — so the per-card check is weights + KV, not
	// weights alone (matching the established serve fit pattern). 0 budget leaves a weights-only plan.
	var extra compute.MemoryPlan
	if contextBudgetTokens > 0 {
		extra = compute.EstimateKVStoreMemoryPlan(compute.KVConfig{
			NumLayers:  m.Cfg.NumLayers,
			NumKVHeads: m.Cfg.NumKVHeads,
			HeadDim:    m.Cfg.HeadDim,
			RopeTheta:  m.Cfg.RopeTheta,
		}, contextBudgetTokens)
	}
	plan := compute.ExpertParallelPerRankPlan(replicated, expert, m.Cfg.NumExperts, ranks, extra)
	return compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

// serveGGUFHostHeadroom reserves a fraction of the process host's allocatable RAM (MemAvailable)
// for the pure-CPU reference serve path's costs NOT in the header estimate: the resident-Q4K
// struct overshoot over the raw-payload estimate (~458 GiB resident vs ~433 GiB on-disk on
// GLM-5.2 UD-Q4_K_M, #974), gateway and KV init, and MemAvailable jitter as clean page cache is
// evicted during the multi-minute load. Matched to serveGGUFDeviceHeadroom for parity with the
// device fit plan, and comfortably above the observed ~6% resident overshoot.
const serveGGUFHostHeadroom = 0.15

// serveFitBudget is the memory ceiling the #1046 context auto-sizer derives the largest fitting
// context against: the raw budget base (a backend's device free-or-total, or the host's
// MemAvailable) and the headroom fraction the matching load-time fit check reserves. A
// non-positive Base means the ceiling is unprobeable (the cpu-ref floor, a device that cannot
// report capacity) — avail() then yields FreeUnknown and the auto-sizer falls open to the model's
// full declared window, exactly as before #1046.
type serveFitBudget struct {
	Base     int64
	Headroom float64
}

// avail is the headroom-adjusted budget passed to compute.AutoSizeContextPlan — byte-identical to
// the budget the matching RefuseMemoryPlanIfTooBig* check computes (same compute.BudgetAfterHeadroom
// formula), so a context derived against it provably passes that check. An unknown base yields
// FreeUnknown so the sizer fails open to the full window.
func (b serveFitBudget) avail() int64 {
	if b.Base <= 0 {
		return compute.FreeUnknown
	}
	return compute.BudgetAfterHeadroom(b.Base, b.Headroom)
}

// serveDeviceFitBudget reads the device memory ceiling a device serve arm's fit check uses
// (DeviceMemoryInfo: free, or the total ceiling when free is unprobeable). Unknown capacity → a
// zero base → the auto-sizer keeps the full window.
func serveDeviceFitBudget(be compute.Backend) serveFitBudget {
	total, free, known := compute.DeviceMemoryInfo(be)
	return serveFitBudget{Base: serveFitBudgetBase(total, free, known), Headroom: serveGGUFDeviceHeadroom}
}

// serveHostFitBudget reads the process host's allocatable RAM the pure-CPU serve arm's fit check
// uses (HostSystemMemoryInfo → Linux MemAvailable). Unknown → a zero base → the full window.
func serveHostFitBudget() serveFitBudget {
	total, free, known := compute.HostSystemMemoryInfo()
	return serveFitBudget{Base: serveFitBudgetBase(total, free, known), Headroom: serveGGUFHostHeadroom}
}

// serveFitBudgetBase collapses a (total, free, known) capacity report into the raw budget base
// the fit check would size against: free when known, the total ceiling when free is unprobeable
// (parity with fitsWithinReportedMemory), and 0 when capacity is unknown.
func serveFitBudgetBase(total, free int64, known bool) int64 {
	if !known || total <= 0 {
		return 0
	}
	if free < 0 { // FreeUnknown -> the total ceiling, conservatively
		return total
	}
	return free
}

func fitServeGGUFOnDevice(ws *ggufload.WeightSource, be compute.Backend, f32Resident bool, contextBudgetTokens int) error {
	if ws == nil || be == nil {
		return nil
	}
	plan, err := serveGGUFMemoryPlan(ws, f32Resident, contextBudgetTokens, serveDeviceFitBudget(be))
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

func fitServeGGUFCPUOffloadOnDevice(ws *ggufload.WeightSource, be compute.Backend, contextBudgetTokens int) error {
	if ws == nil || be == nil {
		return nil
	}
	plan, err := serveGGUFCPUOffloadMemoryPlan(ws, contextBudgetTokens, serveDeviceFitBudget(be))
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

func serveGGUFMemoryPlan(ws *ggufload.WeightSource, f32Resident bool, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	var plan compute.MemoryPlan
	if f32Resident {
		weights, err := ws.EstimateF32LoadMemoryPlan()
		if err != nil {
			return nil, err
		}
		plan = append(plan, weights...)
	} else {
		weights, err := ws.EstimateLoadMemoryPlan()
		if err != nil {
			return nil, err
		}
		plan = append(plan, weights...)
	}
	return appendServeGGUFDevicePlan(ws, plan, contextBudgetTokens, fit), nil
}

func serveGGUFCPUOffloadMemoryPlan(ws *ggufload.WeightSource, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	plan, err := ws.EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		return nil, err
	}
	return appendServeGGUFDevicePlan(ws, plan, contextBudgetTokens, fit), nil
}

func appendServeGGUFDevicePlan(ws *ggufload.WeightSource, plan compute.MemoryPlan, contextBudgetTokens int, fit serveFitBudget) compute.MemoryPlan {
	cfg, err := ws.File.Config()
	if err != nil {
		return plan
	}
	// Delegate to the single context auto-sizer (#1049) so the serve boot path sizes its
	// KV+scratch plan exactly as the in-kernel per-request planner does. #1046: pass the real
	// (headroom-adjusted) memory ceiling so that when no --context-budget-tokens is set the sizer
	// derives the LARGEST context that fits this box — instead of sizing against the full
	// MaxPositionEmbeddings window and refusing — and log the derived size for the operator.
	csc := cfg.ContextSizeConfig()
	avail := fit.avail()
	tokens, ctxPlan := compute.AutoSizeContextPlan(csc, plan, avail, serveContextTokenOverride(contextBudgetTokens))
	logServeAutoSizedContext(csc, plan, fit, avail, contextBudgetTokens, tokens)
	return append(plan, ctxPlan...)
}

// logServeAutoSizedContext prints the #1046 one-line auto-size record when the boot path DERIVED a
// context (no --context-budget-tokens, and a probeable memory ceiling) that is smaller than the
// model's full declared window — the case the operator needs to see, because the full window would
// have overflowed the box and refused. It is silent when an explicit budget was given, when the
// ceiling is unprobeable (the full window is kept, unchanged), or when the full window already fits
// (nothing was shrunk).
func logServeAutoSizedContext(csc compute.ContextSizeConfig, weights compute.MemoryPlan, fit serveFitBudget, avail int64, contextBudgetTokens, tokens int) {
	if contextBudgetTokens > 0 || avail <= 0 || csc.MaxContext <= 0 || tokens >= csc.MaxContext {
		return
	}
	kv := compute.EstimateKVStoreBytes(csc.KV, tokens)
	headroom := fit.Base - avail
	if headroom < 0 {
		headroom = 0
	}
	fmt.Fprintf(os.Stderr,
		"fak: auto-sized context to %d tokens (kv=%s, weights=%s, headroom=%s) — no --context-budget-tokens set; the model's full %d-token window would overflow the %s fit budget\n",
		tokens, bytesText(uint64(max(kv, 0))), bytesText(uint64(max(weights.DeviceTotal(), 0))),
		bytesText(uint64(headroom)), csc.MaxContext, bytesText(uint64(max(avail, 0))))
}

// serveContextTokenOverride maps the serve flag convention (0 = unset, fall back to the
// model's full window) to the auto-sizer's override convention (<0 = unset, >=0 = explicit).
func serveContextTokenOverride(contextBudgetTokens int) int {
	if contextBudgetTokens > 0 {
		return contextBudgetTokens
	}
	return -1
}

// fitServeGGUFPathOnHost is the pure-CPU reference-path memory-fit pre-flight (#974). The CPU
// serve path (loadServeInKernelModel's FAK_Q4K and default cases) copies every super-block to
// ANONYMOUS host RAM with NO HAL backend to refuse via RefuseMemoryPlanIfTooBig, so without this
// it loads until the host OOM-wedges. It sizes the resident weights + KV + scratch off the GGUF
// HEADER ALONE (no tensor read — same EstimateLoadMemoryPlan proxy the device lean path uses) and
// refuses with a typed FitTooBig naming the shortfall when the plan exceeds MemAvailable less
// headroom — parity with the device path's fit plan. Fail-open: a platform that cannot report
// host memory loads exactly as before.
func fitServeGGUFPathOnHost(ggufPath string, f32Resident bool, contextBudgetTokens int) error {
	if ggufPath == "" {
		return nil
	}
	plan, err := serveGGUFPathMemoryPlan(ggufPath, f32Resident, contextBudgetTokens, serveHostFitBudget())
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBigForHost(plan, serveGGUFHostHeadroom)
}

// refuseIfTooBigOnDevice applies the device-headroom refusal to a freshly-built plan —
// the err-check + nil-backend passthrough + RefuseMemoryPlanIfTooBig tail the two
// fitAndPlan…OnDevice helpers share.
func refuseIfTooBigOnDevice(plan compute.MemoryPlan, err error, be compute.Backend) (compute.MemoryPlan, error) {
	if err != nil {
		return nil, err
	}
	if be == nil {
		return plan, nil
	}
	return plan, compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

// withGGUFWeights opens the GGUF weights at ggufPath (an empty path plans nothing) and runs
// plan against them, closing the source after — the open+defer-close prelude the
// serveGGUF…PathMemoryPlan helpers share.
func withGGUFWeights(ggufPath string, plan func(*ggufload.WeightSource) (compute.MemoryPlan, error)) (compute.MemoryPlan, error) {
	if ggufPath == "" {
		return nil, nil
	}
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return plan(ws)
}

func fitAndPlanServeGGUFPathOnDevice(ggufPath string, be compute.Backend, f32Resident bool, contextBudgetTokens int) (compute.MemoryPlan, error) {
	plan, err := serveGGUFPathMemoryPlan(ggufPath, f32Resident, contextBudgetTokens, serveDeviceFitBudget(be))
	return refuseIfTooBigOnDevice(plan, err, be)
}

func fitAndPlanServeGGUFCPUOffloadPathOnDevice(ggufPath string, be compute.Backend, contextBudgetTokens int) (compute.MemoryPlan, error) {
	plan, err := serveGGUFCPUOffloadPathMemoryPlan(ggufPath, contextBudgetTokens, serveDeviceFitBudget(be))
	return refuseIfTooBigOnDevice(plan, err, be)
}

func serveGGUFPathMemoryPlan(ggufPath string, f32Resident bool, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	return withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		return serveGGUFMemoryPlan(ws, f32Resident, contextBudgetTokens, fit)
	})
}

func serveGGUFCPUOffloadPathMemoryPlan(ggufPath string, contextBudgetTokens int, fit serveFitBudget) (compute.MemoryPlan, error) {
	return withGGUFWeights(ggufPath, func(ws *ggufload.WeightSource) (compute.MemoryPlan, error) {
		return serveGGUFCPUOffloadMemoryPlan(ws, contextBudgetTokens, fit)
	})
}

func loadServeInKernelModel(ggufPath string, backend compute.Backend, cpuOffloadExperts bool, contextBudgetTokens int, expertShard *ggufload.ExpertShard) (inKernelModel *fakmodel.Model, inKernelQ4K bool, loadProfile *gateway.ModelLoadProfile, phase gateway.StartupPhase) {
	if ggufPath == "" {
		return nil, false, nil, gateway.StartupPhase{}
	}
	tLoad := time.Now()
	// A sharded expert-parallel rank (expertShard != nil) admits ONLY its routed-expert band into
	// the resident store — the residency that fits GLM-5.2 across the fleet (#971). It rides ONLY
	// the resident-Q4_K arms (cpu-offload, device FAK_Q4K, pure-CPU FAK_Q4K): those carry the
	// WithExpertShard seam (the raw-super-block splitter filters the batched routed experts). The
	// Q8/f32 arms have no shard seam, so a sharded serve that would land on one is REFUSED here —
	// loading a full model on a rank sized only for its band would OOM or silently defeat the shard.
	var q4kOpts []ggufload.Q4KLoadOption
	if expertShard != nil {
		q4kOpts = append(q4kOpts, ggufload.WithExpertShard(expertShard.Lo, expertShard.Hi))
		q4kArm := cpuOffloadExperts || os.Getenv("FAK_Q4K") != ""
		if !q4kArm {
			must(fmt.Errorf("fak serve: --expert-parallel sharded load requires the resident-Q4K path; set FAK_Q4K=1 (or --cpu-offload-experts) so this rank admits only its expert band"))
		}
	}
	// #1062 pre-launch load-path check: warn (don't refuse) before a large GGUF load when the
	// weights sit on a network filesystem. NFS/CIFS read at network speed — the ~50-100x
	// time-to-ready tax a CPU server hit loading GLM-5.2 off /projects (NFS, ~82 min) vs a local NVMe
	// (minutes). Probed once here so it covers every serve arm (device + CPU); fail-open, so a
	// local or unclassifiable weights path prints nothing and loads exactly as before.
	if w := compute.WarnSlowLoadPath(compute.ProbeLoadPath(ggufPath)); w != "" {
		fmt.Fprintln(os.Stderr, "fak: "+w)
	}
	switch {
	case backend != nil && cpuOffloadExperts:
		if !backend.Caps().UploadDtype {
			must(fmt.Errorf("fak serve: --cpu-offload-experts requires backend %q to advertise quantized UploadDtype (Q8_0 upload); use a quantized-upload backend or omit --cpu-offload-experts", backend.Name()))
		}
		fmt.Printf("fak: GGUF device load -> direct-resident Q4_K on backend %q (raw super-blocks copied to VRAM/host, dequant FUSED into the GEMM tile, NO f32/Q8 round-trip; experts host-resident)\n", backend.Name())
		// Device backend + CPU expert-offload: the DIRECT-RESIDENT-Q4_K path. Q4_K matmul weights
		// are copied to VRAM (dense) / host RAM (experts) as raw super-blocks and served with the
		// dequant-fused k_q4k_gemm tile (#485) — skipping the lean path's Q4_K->f32->Q8 round-trip
		// entirely. That round-trip was the load bottleneck on the 466 GB GLM-5.2 (every tensor
		// decompressed to f32 then re-quantized); the resident path is I/O-bound only, cutting the
		// load from ~100 min to minutes. The per-request session decodes Q4_K (s.Q4K=true) on both
		// the device (dense) and host (offloaded experts). The fit check still uses the dense-vs-
		// expert split so experts dwarfing VRAM stay host-scoped while the dense side must fit.
		memPlan, err := fitAndPlanServeGGUFCPUOffloadPathOnDevice(ggufPath, backend, contextBudgetTokens)
		must(err)
		// #971 blocker 3: the dense weights fit-checked above land in VRAM, but the routed MoE
		// experts (~424 GiB for GLM-5.2 Q4_K) are pinned in HOST RAM — and a device backend does not
		// advertise HostCapacity, so the device fit check fails OPEN on them. Guard the host expert
		// pool against the box's real MemAvailable so a load that would OOM-kill the host (or a second
		// concurrent large load on a contended box) refuses cleanly here instead of wedging the box.
		must(compute.RefuseHostScopedPlanIfTooBigForHost(memPlan, serveGGUFHostHeadroom))
		return loadResidentQ4KDevice(ggufPath, tLoad, memPlan, backend, q4kOpts...)
	case backend != nil && os.Getenv("FAK_Q4K") != "" && backend.Caps().UploadDtype:
		// Standard-arch device serve with FAK_Q4K: hold raw Q4_K matmul tensors RESIDENT on the
		// device (dequant fused into the GEMM tile, no Q4_K->f32->Q8 round-trip), instead of the
		// Q8-resident default below. Without this arm the `case backend != nil:` Q8 path matched
		// first and silently ignored FAK_Q4K — loading ~1 B/param instead of raw Q4_K's
		// ~0.56 B/param, ~2x the weight VRAM (#949). No expert offload here (that is the
		// cpuOffloadExperts arm) — all weights are device-resident, so the fit uses the
		// non-offload device plan (EstimateLoadMemoryPlan, quant-aware), same helper the Q8 arm
		// uses; only the loader differs. A backend without UploadDtype falls through to the Q8/
		// f32 arms unchanged (the device Q4_K GEMM needs the quantized-upload seam).
		fmt.Printf("fak: GGUF device load -> resident Q4_K on backend %q (raw super-blocks, dequant-fused GEMM, ~0.56 B/param vs Q8 ~1 B/param)\n", backend.Name())
		memPlan, err := fitAndPlanServeGGUFPathOnDevice(ggufPath, backend, false, contextBudgetTokens)
		must(err)
		return loadResidentQ4KDevice(ggufPath, tLoad, memPlan, backend, q4kOpts...)
	case backend != nil:
		if backend.Caps().UploadDtype {
			// A device backend that can consume Q8_0 uploads should not be forced through
			// the f32 resident path. The served planner runs Session.Quant=true, so this
			// is the memory-lean representation it will actually execute.
			fmt.Printf("fak: GGUF device load -> mixed precision on backend %q (Q8 resident weights, f32 activations/KV)\n", backend.Name())
			memPlan, err := fitAndPlanServeGGUFPathOnDevice(ggufPath, backend, false, contextBudgetTokens)
			must(err)
			prof := ggufload.NewLoadProfiler()
			mm, err := ggufload.LoadModelQuantProfile(ggufPath, prof)
			must(err)
			modelengine.Preload(mm)
			loadNanos := time.Since(tLoad).Nanoseconds()
			profile := withServeGGUFMemoryProfile(toGatewayLoadProfile(prof.Snapshot("gguf-lean-q8-device", ggufPath, loadNanos)), memPlan, backend)
			return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
		}
		// Backends without quantized upload still need f32-resident weights; a lean-Q8
		// model would drop the f32 matmul weights they fall back to.
		fmt.Printf("fak: GGUF device load -> f32 resident weights on backend %q (backend has no quantized UploadDtype)\n", backend.Name())
		memPlan, err := fitAndPlanServeGGUFPathOnDevice(ggufPath, backend, true, contextBudgetTokens)
		must(err)
		mm, err := ggufload.LoadModel(ggufPath)
		must(err)
		modelengine.Preload(mm)
		loadNanos := time.Since(tLoad).Nanoseconds()
		profile := withServeGGUFMemoryProfile(toGatewayLoadProfile(&ggufload.LoadProfile{
			Mode:       "gguf-f32-device",
			Source:     ggufPath,
			TotalNanos: loadNanos,
			TotalMS:    float64(loadNanos) / 1e6,
			Phases:     []ggufload.LoadPhaseStat{{Phase: "f32-load", Calls: 1, Nanos: loadNanos, MS: float64(loadNanos) / 1e6, TimePct: 100}},
			Bottleneck: "f32-load",
		}), memPlan, backend)
		return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	case os.Getenv("FAK_Q4K") != "":
		// CPU-path memory-fit pre-flight (#974): refuse cleanly with a typed FitTooBig BEFORE the
		// all-resident load can drive MemAvailable to ~0 and OOM-wedge the host (parity with the
		// device path's fit plan). Fail-open where host RAM is not probeable.
		must(fitServeGGUFPathOnHost(ggufPath, false, contextBudgetTokens))
		// Pure-CPU reference serve via the direct-resident-Q4_K loader. That loader already
		// routes the mixed Q5_K/Q6_K experts (GLM-5.2's ~417 GB bulk) to a raw-resident byte
		// copy keyed on the GGUF quant type — the SAME resident-K-quant lever the device
		// cpu-offload case above uses (internal/ggufload/quant_q4k_loader.go). The only thing
		// missing on this path was the WITNESS: the old call threaded no LoadProfiler, so a
		// multi-minute GLM-5.2 load ran silent AND emitted no per-quant-type load-path summary,
		// leaving an operator unable to SEE whether the expert bulk took the raw-resident path
		// (dequant≈0) or the slow f32 round-trip. Thread a real profiler here too (parity with
		// the device cpu-offload case) so both the streamed summary and the gateway /metrics
		// profile carry the resident-vs-dequant breakdown — the witness #975 needs.
		mm, prof, loadNanos := loadResidentQ4KProfiled(ggufPath, tLoad, q4kOpts...)
		profile := toGatewayLoadProfile(prof.Snapshot("gguf-resident-q4k", ggufPath, loadNanos))
		return mm, true, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	default:
		// CPU-path memory-fit pre-flight (#974): same clean FitTooBig refusal as the FAK_Q4K arm
		// above, so the default lean CPU serve cannot OOM-wedge the host either.
		must(fitServeGGUFPathOnHost(ggufPath, false, contextBudgetTokens))
		prof := ggufload.NewLoadProfiler()
		prof.Progress = os.Stderr // stream load % to stderr so a large multi-minute load is not silent
		mm, err := ggufload.LoadModelQuantProfile(ggufPath, prof)
		must(err)
		modelengine.Preload(mm)
		loadNanos := time.Since(tLoad).Nanoseconds()
		profile := toGatewayLoadProfile(prof.Snapshot("gguf-lean-q8", ggufPath, loadNanos))
		return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	}
}

// loadResidentQ4KProfiled runs the profiled raw-Q4_K resident load shared by the three Q4_K
// serve arms (device cpu-offload, device-resident, pure-CPU): it streams load % to stderr,
// loads + preloads the resident super-blocks, prints the post-load resident split (so a glance
// confirms the mixed-quant expert bulk loaded resident, the slow f32 round-trip avoided), and
// returns the model, the profiler (the caller folds prof.Snapshot into its own profile), and
// the elapsed load nanos.
func loadResidentQ4KProfiled(ggufPath string, tLoad time.Time, opts ...ggufload.Q4KLoadOption) (*fakmodel.Model, *ggufload.LoadProfiler, int64) {
	prof := ggufload.NewLoadProfiler()
	prof.Progress = os.Stderr // stream load % to stderr so a large multi-minute load is not silent
	// opts carries the per-rank expert shard (ggufload.WithExpertShard) for a sharded expert-
	// parallel serve: this process admits ONLY its band's routed experts into the resident store,
	// so its footprint is the replicated remainder + one band (≈ model/ranks), not the full model.
	// Empty opts (the default, every non-EP serve) is byte-identical to the old LoadModelQ4KProfile.
	mm, err := ggufload.LoadModelQ4KProfileOptions(ggufPath, prof, opts...)
	must(err)
	modelengine.PreloadQ4K(mm)
	fmt.Fprintln(os.Stderr, "fak: "+fakmodel.FormatResidentReport(mm.ResidentReport()))
	loadNanos := time.Since(tLoad).Nanoseconds()
	return mm, prof, loadNanos
}

// loadResidentQ4KDevice is the device Q4_K arm's shared tail: it runs the profiled resident
// load and folds the host/device memory plan into the streamed profile. The cpu-offload and
// device-resident arms differ only in how memPlan is derived upstream. opts threads the per-rank
// expert shard (see loadResidentQ4KProfiled).
func loadResidentQ4KDevice(ggufPath string, tLoad time.Time, memPlan compute.MemoryPlan, backend compute.Backend, opts ...ggufload.Q4KLoadOption) (*fakmodel.Model, bool, *gateway.ModelLoadProfile, gateway.StartupPhase) {
	mm, prof, loadNanos := loadResidentQ4KProfiled(ggufPath, tLoad, opts...)
	profile := withServeGGUFMemoryProfile(toGatewayLoadProfile(prof.Snapshot("gguf-resident-q4k-device", ggufPath, loadNanos)), memPlan, backend)
	return mm, true, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
}

// resolveServeTokenizer picks the in-kernel chat planner's tokenizer: an explicit
// --tokenizer (a tokenizer.json or its directory, matching cmd/fakchat's resolution) wins;
// otherwise, with --gguf set, the GGUF's EMBEDDED tokenizer is used so /v1/chat/completions
// and /v1/messages serve real in-kernel chat by default (like cmd/simpledemo); otherwise it
// returns nil, leaving the gateway's offline MockPlanner fallback. The bool reports whether
// a tokenizer-load startup phase should be recorded.
//
// On a real load it ALSO arms the in-kernel engine's detokenizer (modelengine.SetTokenizer),
// symmetric with how loadServeInKernelModel preloads the weights: that closes #463's named
// gap — the lower-level /v1/fak/syscall route then NL-tokenizes a call's arguments and
// returns decoded TEXT (generated_text) instead of raw token ids. With no real tokenizer the
// engine keeps its byte-level default, so the CI/no-export path is unchanged.
func resolveServeTokenizer(tokPath, ggufPath string) (*tokenizer.Tokenizer, bool) {
	if tokPath != "" {
		tokFile := tokPath
		if info, err := os.Stat(tokFile); err == nil && info.IsDir() {
			tokFile = filepath.Join(tokFile, "tokenizer.json")
		}
		tok, err := tokenizer.LoadJSON(tokFile)
		must(err)
		modelengine.SetTokenizer(tok)
		return tok, true
	}
	if ggufPath != "" {
		// No explicit --tokenizer: fall back to the tokenizer EMBEDDED in the GGUF. Virtually
		// every GGUF carries its full vocab+merges, so `fak serve --gguf X` (no --base-url)
		// serves real in-kernel chat out of the box instead of silently dropping
		// /v1/chat/completions to the offline MockPlanner. If the GGUF embeds no usable BPE
		// tokenizer (e.g. an SPM-only checkpoint), we keep the MockPlanner fallback — pass
		// --tokenizer to override.
		if tok, err := embeddedGGUFTokenizer(ggufPath); err == nil {
			modelengine.SetTokenizer(tok)
			return tok, true
		} else {
			fmt.Fprintf(os.Stderr, "fak serve: --gguf set without --tokenizer and no embedded BPE tokenizer (%v);\n"+
				"  /v1/chat/completions will use the offline mock planner. Pass --tokenizer <dir|file> for real chat.\n", err)
		}
	}
	return nil, false
}

// resetOnBudgetHook gates the human-like auto-reset behind the --reset-on-budget flag.
// When the flag is off it returns nil, so the gateway keeps the historical 409 + reset
// directive verbatim (the reset is strictly opt-in). When on, it returns the host hook
// that distills a carryover seed and re-arms the continuation trace with a fresh context
// budget. The flag is validated to require --context-budget-tokens, so freshContextTokens
// is positive here whenever enabled is true.
func resetOnBudgetHook(enabled bool, freshContextTokens int) gateway.ResetOnBudgetFunc {
	if !enabled {
		return nil
	}
	return resetServedSessionOnBudget(freshContextTokens)
}
