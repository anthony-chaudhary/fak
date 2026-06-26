package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
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

func cmdServe(argv []string) {
	// t0 anchors the boot timeline exposed as fak_gateway_time_to_ready_seconds; it
	// must be the FIRST statement so flag parse + policy + weight load are accounted.
	t0 := time.Now()
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:8080", "HTTP listen address (OpenAI + fak + /mcp surface); ignored with --stdio")
	stdio := fs.Bool("stdio", false, "serve MCP over stdin/stdout (newline-delimited JSON-RPC) instead of HTTP")
	provider := fs.String("provider", "openai", "upstream provider transcript wire: openai, anthropic, gemini, or xai")
	baseURL := fs.String("base-url", "", "upstream provider base URL for the /v1/chat/completions proxy (empty = offline mock planner)")
	var replicaBaseURLs repeatedStringFlag
	fs.Var(&replicaBaseURLs, "replica-base-url", "additional upstream provider base URL for a static round-robin replica fleet; repeat for N replicas. If --base-url is set, it is replica 1.")
	model := fs.String("model", "mock", "model id (advertised by /v1/models; used for the upstream call)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the upstream API key (proxy mode)")
	engineCacheEngine := fs.String("engine-cache-engine", "", "self-hosted upstream cache reset engine for quarantined provider-bound tool results: sglang|vllm (empty disables)")
	engineCacheBaseURL := fs.String("engine-cache-base-url", "", "serving-engine control/base URL for cache reset (default: --base-url when --engine-cache-engine is set)")
	engineCacheAdminKeyEnv := fs.String("engine-cache-admin-key-env", "", "env var holding the serving-engine admin API key for cache reset")
	engineCacheIdleTimeout := fs.Duration("engine-cache-idle-timeout", 0, "SGLang /flush_cache idle timeout, e.g. 30s (0 fails fast)")
	engineCacheRequireExactSpan := fs.Bool("engine-cache-require-exact-span", false, "require exact remote K/V/index span eviction; fail closed if the selected engine only supports whole-cache reset")
	engineID := fs.String("engine", "inkernel", "registered engine id that fak_syscall dispatches an allowed call to (default: the fused in-kernel model)")
	backendName := fs.String("backend", "", "compute backend for the in-kernel chat decode (with --gguf, no --base-url): empty = the CPU reference path; a registered device name like 'cuda' runs prefill+decode through the GPU HAL. Requires a `-tags cuda` build AND a reachable GPU at runtime; fails loud if named but unavailable so a typo never silently runs on CPU.")
	policyPath := fs.String("policy", "", "capability-floor manifest to load (default: the built-in adjudicator floor — the tau2 airline-demo tools, NOT the `fak guard` coding floor; see `fak policy --dump`)")
	policyCheck := fs.Bool("policy-check", false, "validate --policy and exit without binding a listener")
	vdso := fs.Bool("vdso", true, "enable the vDSO dedup fast path")
	invalidation := fs.String("invalidation", "global", "vDSO tier-2 invalidation granularity for the live fleet: global|namespace|resource")
	requireKeyEnv := fs.String("require-key-env", "", "env var holding a bearer token to REQUIRE on every request (default: no auth)")
	routeManifest := fs.String("route-manifest", "", "model-routing policy to install: each fak_syscall call is classified into a modelroute.Subject and a single-model (PICK) plan binds abi.ToolCall.Engine before Submit, so the residency PDP adjudicates the real route (#601). Empty (default) leaves Engine unset → the kernel default engine, byte-for-byte the pre-routing behavior. A malformed manifest fails startup loud (a mis-routed model is a security boundary, never a silent default). The installed file is HOT-RELOADED: an edit is picked up without a restart and swapped atomically (a request classifies against the whole old or whole new policy, never a torn read); a malformed edit is rejected and the last-good policy stays installed (#842).")
	ggufPath := fs.String("gguf", "", "load these GGUF weights into the in-kernel engine at boot; the load is part of the measured startup sequence and its phase breakdown is exposed on /metrics. Default path is lean-Q8 (Q4→f32→Q8 round-trip); set FAK_Q4K=1 for the direct-resident-Q4_K path (Qwen3.6-27B q4_k_m, the P1/P2 decode lever)")
	tokPath := fs.String("tokenizer", "", "OPTIONAL override for the in-kernel CHAT planner's tokenizer. With --gguf and no --base-url, /v1/chat/completions AND /v1/messages already serve the in-kernel model (real ChatML chat) using the GGUF's EMBEDDED tokenizer; pass this only to override it (e.g. an SPM-only checkpoint with no embedded BPE tokenizer, or a custom vocab). Accepts a tokenizer.json or its directory. e.g. ~/.cache/fak-models/tokenizers/qwen3.6")
	ctxViewBudget := fs.Int("ctx-view-budget", 0, "wire the ctxplan context PLANNER into the live serve loop: each buffered turn, re-materialize the forwarded history as an O(1) planned VIEW under this resident-token budget (a planned view in place of appending the whole transcript, #555). 0 (default) leaves the existing path byte-for-byte unchanged. OFF by default: it rewrites in-flight turn history, so gate it until you have watched a real session. The streaming fast-path bypasses this; the buffered turn path is what gets planned.")
	compactHistoryBudget := fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "on the Anthropic PASSTHROUGH (an upstream --base-url anthropic), compact OLD conversation turns in the OUTBOUND request body down to this resident-token budget while keeping the cache_control prefix BYTE-IDENTICAL, so the upstream cache hit survives. This reaches the flagship passthrough the streaming ctxplan view cannot (#555). DEFAULT-ON: once a conversation sprawls past ~48k resident tokens the cut fires and sheds the un-cacheable middle the provider re-bills every turn; a typical short session stays untouched. Pass 0 to disable (body forwarded byte-for-byte). No effect on non-passthrough wires.")
	sessionID := fs.String("session-id", "", "default trace/session id for callers that omit X-Trace-Id or MCP trace_id (empty = mint gw-N per request unless --context-budget-tokens is set)")
	contextBudgetTokens := fs.Int("context-budget-tokens", 0, "seed the default session with this prompt/context-token budget; exhaustion returns a reset directive with continuation_id (0 = off)")
	resetOnBudget := fs.Bool("reset-on-budget", false, "on context-budget exhaustion, re-arm the continuation trace with a carryover seed and continue transparently instead of returning 409 (requires --context-budget-tokens)")
	cpuOffloadExperts := fs.Bool("cpu-offload-experts", false, "with --gguf --backend: keep the MoE expert GEMMs on host RAM while dense projections + router + attention run on the device — the `--n-cpu-moe` hybrid that lets a model whose experts dwarf VRAM (e.g. GLM-5.2 Q4 ~424GB experts) serve at all on a smaller VRAM pool. The device load uses the memory-lean Q8 quantize-at-load path when the backend advertises quantized upload; otherwise it falls back to F32 weights until that backend implements UploadDtype.")
	budgetWebhook := fs.String("budget-webhook", "", "POST a JSON event to this URL when a served session's context budget crosses the warning threshold (--budget-warn-fraction) or is exhausted (the reset trigger), so an operator/monitor is notified before exhaustion (#743). Empty = off. Needs --context-budget-tokens to have a budget to watch.")
	budgetWarnFraction := fs.Float64("budget-warn-fraction", 0.8, "consumed share (0..1) of the context budget at which --budget-webhook fires its pre-exhaustion warning (default 0.8 = 80%); <=0 or >=1 disables the warning while the exhaustion event still fires")
	notifyNative := fs.Bool("notify-native", true, "emit a one-line native notification to stderr when a served session hits a PAUSED/DRAINING/STOPPED or budget boundary, carrying the closed stop-reason token — the SIGCHLD-equivalent so a waiting agent is never silent (#761); default on")
	notifyWebhook := fs.String("notify-webhook", "", "POST a JSON StopEvent to this URL on each served-session terminal/paused/budget boundary (#761), carrying the closed reason token; empty = off. Extends the #743 budget webhook to the full stop-reason vocabulary.")
	notifySlack := fs.String("notify-slack", "", "POST a Slack incoming-webhook payload ({\"text\":…}) on each served-session boundary (#761); empty = off")
	debugStats := fs.Bool("debug-stats", false, "print ONE compact, payload-free line per served turn to stderr: request/cache_read/cache_creation tokens, the compaction action, and the resetScore SHADOW health (healthy_cache|cache_decay|stale_prefix|cooldown|unknown_provider). Independent of --log (#793); default off.")
	tParse := time.Now()
	_ = fs.Parse(argv)
	parseDur := time.Since(tParse)

	// Expand a leading ~ in the model/tokenizer paths: PowerShell and most quoting
	// pass ~ through literally and Go never expands it, so `--gguf ~/...` (as the
	// docs and the --tokenizer help itself show) would otherwise fail to open.
	*ggufPath = pathutil.ExpandTilde(*ggufPath)
	*tokPath = pathutil.ExpandTilde(*tokPath)

	// An hf:// --gguf resolves to a locally cached file before the loader sees it,
	// so `fak serve --gguf hf://owner/repo/model.gguf` works without a manual
	// `fak model load` first (issue #294). Download progress goes to stderr.
	if hfhub.IsURI(*ggufPath) {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		resolved, err := hfhub.FetchURI(ctx, *ggufPath, os.Stderr)
		stop()
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: --gguf %v\n", err)
			os.Exit(1)
		}
		*ggufPath = resolved
	}

	// --policy-check: validate the manifest and exit, binding no listener.
	if *policyCheck {
		if *policyPath == "" {
			fmt.Fprintln(os.Stderr, "fak serve: --policy-check requires --policy FILE")
			os.Exit(2)
		}
		rt, err := policy.LoadRuntime(*policyPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve:", err)
			os.Exit(1)
		}
		fmt.Printf("OK  %s  (manifest valid; every deny cites a closed-vocabulary reason)\n\n%s", *policyPath, policy.SummaryRuntime(rt))
		return
	}

	// Install the capability floor fail-loud: a bad manifest aborts startup rather
	// than silently falling back to a more permissive default. Time it as the first
	// startup phase.
	tPolicy := time.Now()
	applyPolicy(*policyPath)
	startupPhases := []gateway.StartupPhase{
		{Name: "flag-parse", Dur: parseDur},
		{Name: "policy-load", Dur: time.Since(tPolicy)},
	}

	// Resolve the optional in-kernel chat decode backend BEFORE eager model loading, so
	// a known device can refuse an oversize GGUF from its header instead of OOMing during
	// the load. Lookup (not Pick) keeps typos fail-loud rather than silently degrading to CPU.
	chatBackend, err := resolveServeChatBackend(*backendName)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if chatBackend != nil {
		fmt.Printf("fak: in-kernel chat decode → device backend %q\n", chatBackend.Name())
	}

	// Eager GGUF load: pull the weights resident BEFORE binding the listener so the
	// (potentially multi-second) load is measured as part of time-to-ready and its
	// phase breakdown is on /metrics, rather than a lazy cost paid on first request.
	//
	// Two load paths, selected by the FAK_Q4K env (mirroring cmd/fakchat and
	// cmd/q4kdiag): the default lean-Q8 round-trip, or the direct-resident-Q4_K path
	// (QWEN36-NATIVE-PERF-PLAN P1/P2) that holds eligible Q4_K matmul tensors raw and
	// engages the NEON SDOT int8 decode GEMV — ~10× faster load and the Qwen3.6-27B
	// decode lever. The Q8 path stays byte-identical when the env is unset.
	//
	// The loaded *model.Model is ALSO kept for the gateway chat planner: with a tokenizer
	// (explicit --tokenizer or the GGUF's embedded one) and no --base-url,
	// /v1/chat/completions and /v1/messages serve it directly.
	inKernelModel, inKernelQ4K, loadProfile, loadPhase := loadServeInKernelModel(*ggufPath, chatBackend, *cpuOffloadExperts, *contextBudgetTokens)
	if loadPhase.Name != "" {
		startupPhases = append(startupPhases, loadPhase)
	}

	inKernelTok, tokLoaded := resolveServeTokenizer(*tokPath, *ggufPath)
	if tokLoaded {
		startupPhases = append(startupPhases, gateway.StartupPhase{Name: "tokenizer-load", Dur: 0})
	}

	apiKey := ""
	if *apiKeyEnv != "" {
		apiKey = os.Getenv(*apiKeyEnv)
	}
	engineCacheAdminKey, ok := resolveRequiredKey(*engineCacheAdminKeyEnv, os.Getenv)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak serve: --engine-cache-admin-key-env %s is set but unset/empty — refusing to send cache-reset requests with NO admin auth (set the secret or omit the flag)\n", *engineCacheAdminKeyEnv)
		os.Exit(2)
	}
	if *engineCacheIdleTimeout < 0 {
		fmt.Fprintln(os.Stderr, "fak serve: --engine-cache-idle-timeout must be non-negative")
		os.Exit(2)
	}
	requireKey, ok := resolveRequiredKey(*requireKeyEnv, os.Getenv)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak serve: --require-key-env %s is set but unset/empty — refusing to start a network-facing gateway with NO authentication (set the secret or omit the flag)\n", *requireKeyEnv)
		os.Exit(2)
	}
	if *contextBudgetTokens < 0 {
		fmt.Fprintln(os.Stderr, "fak serve: --context-budget-tokens must be non-negative")
		os.Exit(2)
	}
	if *resetOnBudget && *contextBudgetTokens <= 0 {
		fmt.Fprintln(os.Stderr, "fak serve: --reset-on-budget requires --context-budget-tokens N")
		os.Exit(2)
	}
	defaultTraceID := strings.TrimSpace(*sessionID)
	if *contextBudgetTokens > 0 {
		if defaultTraceID == "" {
			defaultTraceID = "default"
		}
		serveSessions.SetBudget(defaultTraceID, session.Budget{
			TurnsLeft:         session.Unbounded,
			TokensLeft:        session.Unbounded,
			ContextTokensLeft: *contextBudgetTokens,
		})
	}
	// Wire the optional operator webhook (#743) and the tiered stop-reason push notifier
	// (#761). The #743 budget webhook stays byte-identical when it is the only thing set:
	// combineBudgetObservers returns the lone observer unchanged, so WatchBudget is called
	// once exactly as before. The notifier (native default-on; webhook/Slack opt-in) adds a
	// SECOND budget fan-out plus the run-state TRANSITION observer that covers
	// PAUSED/DRAINING/STOPPED — the rest of the closed stop-reason vocabulary the budget seam
	// alone never sees. newNotifier returns nil when no sink is configured, leaving the
	// transition seam its byte-identical no-op default.
	notifier := newNotifier(*notifyNative, os.Stderr, *notifyWebhook, *notifySlack)
	if notifier != nil {
		serveSessions.WatchTransitions(notifier.transitionObserver())
	}
	var budgetObs session.BudgetObserver
	if obs := budgetWebhookObserver(*budgetWebhook); obs != nil {
		budgetObs = obs
	}
	if notifier != nil {
		budgetObs = combineBudgetObservers(budgetObs, notifier.budgetObserver())
	}
	if budgetObs != nil {
		serveSessions.WatchBudget(*budgetWarnFraction, budgetObs)
	}

	// Resolve the optional model-routing policy. Off by default: an empty --route-manifest
	// leaves routeMan nil, so gateway.New gets a nil RouteManifest and Engine stays unset —
	// byte-for-byte the pre-routing behavior. A malformed file fails loud here rather than
	// silently default-routing every call to the kernel default (a mis-routed model is a
	// security boundary). gateway.New also re-validates the loaded manifest.
	var routeMan *modelroute.Manifest
	if *routeManifest != "" {
		loaded, err := modelroute.LoadManifest(*routeManifest)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve: --route-manifest:", err)
			os.Exit(1)
		}
		routeMan = &loaded
		fmt.Printf("fak: model-routing policy loaded from %s\n", *routeManifest)
	}

	srv, err := gateway.New(gateway.Config{
		EngineID:                    *engineID,
		Model:                       *model,
		BaseURL:                     *baseURL,
		ReplicaBaseURLs:             replicaBaseURLs.Values(),
		Provider:                    *provider,
		APIKey:                      apiKey,
		EngineCacheEngine:           *engineCacheEngine,
		EngineCacheBaseURL:          *engineCacheBaseURL,
		EngineCacheAdminKey:         engineCacheAdminKey,
		EngineCacheIdleTimeout:      *engineCacheIdleTimeout,
		EngineCacheRequireExactSpan: *engineCacheRequireExactSpan,
		InKernelModel:               inKernelModel,
		Tokenizer:                   inKernelTok,
		InKernelQ4K:                 inKernelQ4K,
		Backend:                     chatBackend,
		CPUOffloadExperts:           *cpuOffloadExperts,
		RequireKey:                  requireKey,
		VDSO:                        *vdso,
		Invalidation:                *invalidation,
		Version:                     appversion.Current(),
		ReloadPolicy:                policyReloader(*policyPath),
		ResetTrace:                  resetTrace,
		ObserveTrace:                observeTrace,
		ObserveSession:              observeSession,
		ControlSession:              controlSession,
		SteerSession:                steerSession,
		ListSessions:                listSessions,
		DecideSession:               decideSession,
		DebitSession:                debitSession,
		ResetOnBudget:               resetOnBudgetHook(*resetOnBudget, *contextBudgetTokens),
		DefaultTraceID:              defaultTraceID,
		StartTime:                   t0,
		StartupPhases:               startupPhases,
		CtxViewBudget:               *ctxViewBudget,
		CompactHistoryBudget:        *compactHistoryBudget,
		DebugStatsf:                 debugStatsSink(*debugStats),
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
	})
	must(err)
	srv.SetModelLoadProfile(loadProfile)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Hot-reload the routing policy (#842): when a manifest is installed, follow the
	// file and atomically swap the live policy on a validated edit — no restart. A
	// malformed edit is rejected and the last-good policy is kept (the fail-loud
	// startup contract extended to reload). The watcher reads the SAME atomic Live
	// the gateway classifies through, so a swap is visible on the hot path; it is
	// bound to ctx, so it stops with the server. Reloads/rejections are logged so an
	// operator can confirm the swap landed.
	if live := srv.RouteLive(); live != nil {
		watcher := modelroute.NewWatcher(*routeManifest, live, 0, func(ev modelroute.ReloadEvent) {
			if ev.Err != nil {
				fmt.Fprintf(os.Stderr, "fak: route-manifest reload REJECTED: %v\n", ev.Err)
				return
			}
			if ev.Reloaded {
				fmt.Fprintf(os.Stderr, "fak: model-routing policy hot-reloaded from %s (reload #%d)\n", *routeManifest, ev.Reloads)
			}
		})
		go func() { _ = watcher.Run(ctx) }()
	}

	if *stdio {
		// MCP over stdio: stdout carries the protocol; the log package writes to
		// stderr, so diagnostics never corrupt the frames.
		if err := srv.ServeStdio(ctx, os.Stdin, os.Stdout); err != nil && !errors.Is(err, context.Canceled) {
			must(err)
		}
		return
	}
	if *addr == "" {
		fmt.Fprintln(os.Stderr, "fak serve: --addr is required (or pass --stdio)")
		os.Exit(2)
	}
	if err := srv.ListenAndServe(ctx, *addr); err != nil && !errors.Is(err, http.ErrServerClosed) {
		must(err)
	}
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
	return out
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

const serveGGUFDeviceHeadroom = 0.15

func fitServeGGUFOnDevice(ws *ggufload.WeightSource, be compute.Backend, f32Resident bool, contextBudgetTokens int) error {
	if ws == nil || be == nil {
		return nil
	}
	plan, err := serveGGUFMemoryPlan(ws, f32Resident, contextBudgetTokens)
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

func fitServeGGUFCPUOffloadOnDevice(ws *ggufload.WeightSource, be compute.Backend, contextBudgetTokens int) error {
	if ws == nil || be == nil {
		return nil
	}
	plan, err := serveGGUFCPUOffloadMemoryPlan(ws, contextBudgetTokens)
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

func serveGGUFMemoryPlan(ws *ggufload.WeightSource, f32Resident bool, contextBudgetTokens int) (compute.MemoryPlan, error) {
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
	return appendServeGGUFKVPlan(ws, plan, contextBudgetTokens), nil
}

func serveGGUFCPUOffloadMemoryPlan(ws *ggufload.WeightSource, contextBudgetTokens int) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	plan, err := ws.EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		return nil, err
	}
	return appendServeGGUFKVPlan(ws, plan, contextBudgetTokens), nil
}

func appendServeGGUFKVPlan(ws *ggufload.WeightSource, plan compute.MemoryPlan, contextBudgetTokens int) compute.MemoryPlan {
	if cfg, err := ws.File.Config(); err == nil {
		if tokens := serveGGUFContextPlanTokens(cfg, contextBudgetTokens); tokens > 0 {
			plan = append(plan, compute.EstimateKVStoreMemoryPlan(compute.KVConfig{
				NumLayers:  cfg.NumLayers,
				NumKVHeads: cfg.NumKVHeads,
				HeadDim:    cfg.HeadDim,
				RopeTheta:  cfg.RopeTheta,
			}, tokens)...)
		}
	}
	return plan
}

func serveGGUFContextPlanTokens(cfg fakmodel.Config, contextBudgetTokens int) int {
	if contextBudgetTokens > 0 {
		return contextBudgetTokens
	}
	return cfg.MaxPositionEmbeddings
}

func fitServeGGUFPathOnDevice(ggufPath string, be compute.Backend, f32Resident bool, contextBudgetTokens int) error {
	if ggufPath == "" || be == nil {
		return nil
	}
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return err
	}
	defer ws.Close()
	return fitServeGGUFOnDevice(ws, be, f32Resident, contextBudgetTokens)
}

func fitServeGGUFCPUOffloadPathOnDevice(ggufPath string, be compute.Backend, contextBudgetTokens int) error {
	if ggufPath == "" || be == nil {
		return nil
	}
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return err
	}
	defer ws.Close()
	return fitServeGGUFCPUOffloadOnDevice(ws, be, contextBudgetTokens)
}

func loadServeInKernelModel(ggufPath string, backend compute.Backend, cpuOffloadExperts bool, contextBudgetTokens int) (inKernelModel *fakmodel.Model, inKernelQ4K bool, loadProfile *gateway.ModelLoadProfile, phase gateway.StartupPhase) {
	if ggufPath == "" {
		return nil, false, nil, gateway.StartupPhase{}
	}
	tLoad := time.Now()
	switch {
	case backend != nil && cpuOffloadExperts:
		if !backend.Caps().UploadDtype {
			must(fmt.Errorf("fak serve: --cpu-offload-experts requires backend %q to advertise quantized UploadDtype (Q8_0 upload); use a quantized-upload backend or omit --cpu-offload-experts", backend.Name()))
		}
		fmt.Printf("fak: GGUF device load -> mixed precision on backend %q (Q8 resident dense/device weights, f32 activations/KV, experts host-resident)\n", backend.Name())
		// Device backend + CPU expert-offload: the memory-lean Q8 quantize-at-load path (the
		// SAME representation cmd/modelbench's `-lean -backend cuda` produces). The big matmul
		// weights are dropped from f32 and kept Q8-resident, which the cuda HAL now consumes
		// directly at H2D (internal/compute/cuda.go Upload(t, Q8_0) / uploadQ8Resident, #485) —
		// the old "the HAL only narrows F32" limitation is gone. This is the only load that fits
		// a model whose experts dwarf VRAM (GLM-5.2 Q4 experts ~424GB): with --cpu-offload-experts
		// the per-request session keeps the expert GEMMs on host RAM while dense projections +
		// router + attention run on the device (internal/model glmDsaMatKernel split).
		// The fit check must use the dense-vs-expert split, not total GGUF bytes: experts are
		// host-scoped offload demands, while dense/router/attention weights and KV consume the
		// device budget. That keeps the valid "experts dwarf VRAM" use case alive while still
		// refusing a dense side that cannot fit.
		must(fitServeGGUFCPUOffloadPathOnDevice(ggufPath, backend, contextBudgetTokens))
		prof := ggufload.NewLoadProfiler()
		prof.Progress = os.Stderr // stream load % to stderr so a large multi-minute load is not silent
		mm, err := ggufload.LoadModelQuantProfile(ggufPath, prof)
		must(err)
		modelengine.Preload(mm)
		loadNanos := time.Since(tLoad).Nanoseconds()
		profile := toGatewayLoadProfile(prof.Snapshot("gguf-lean-q8-device", ggufPath, loadNanos))
		return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	case backend != nil:
		if backend.Caps().UploadDtype {
			// A device backend that can consume Q8_0 uploads should not be forced through
			// the f32 resident path. The served planner runs Session.Quant=true, so this
			// is the memory-lean representation it will actually execute.
			fmt.Printf("fak: GGUF device load -> mixed precision on backend %q (Q8 resident weights, f32 activations/KV)\n", backend.Name())
			must(fitServeGGUFPathOnDevice(ggufPath, backend, false, contextBudgetTokens))
			prof := ggufload.NewLoadProfiler()
			mm, err := ggufload.LoadModelQuantProfile(ggufPath, prof)
			must(err)
			modelengine.Preload(mm)
			loadNanos := time.Since(tLoad).Nanoseconds()
			profile := toGatewayLoadProfile(prof.Snapshot("gguf-lean-q8-device", ggufPath, loadNanos))
			return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
		}
		// Backends without quantized upload still need f32-resident weights; a lean-Q8
		// model would drop the f32 matmul weights they fall back to.
		fmt.Printf("fak: GGUF device load -> f32 resident weights on backend %q (backend has no quantized UploadDtype)\n", backend.Name())
		must(fitServeGGUFPathOnDevice(ggufPath, backend, true, contextBudgetTokens))
		mm, err := ggufload.LoadModel(ggufPath)
		must(err)
		modelengine.Preload(mm)
		loadNanos := time.Since(tLoad).Nanoseconds()
		profile := toGatewayLoadProfile(&ggufload.LoadProfile{
			Mode:       "gguf-f32-device",
			Source:     ggufPath,
			TotalNanos: loadNanos,
			TotalMS:    float64(loadNanos) / 1e6,
			Phases:     []ggufload.LoadPhaseStat{{Phase: "f32-load", Calls: 1, Nanos: loadNanos, MS: float64(loadNanos) / 1e6, TimePct: 100}},
			Bottleneck: "f32-load",
		})
		return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	case os.Getenv("FAK_Q4K") != "":
		mm, err := ggufload.LoadModelQ4K(ggufPath)
		must(err)
		modelengine.PreloadQ4K(mm)
		loadNanos := time.Since(tLoad).Nanoseconds()
		// LoadModelQ4K does not thread a LoadProfiler (the direct-q4 path has no
		// dequant/re-quant phases to break down), so surface the load as a single
		// measured phase rather than an empty profile.
		profile := toGatewayLoadProfile(&ggufload.LoadProfile{
			Mode:       "gguf-resident-q4k",
			Source:     ggufPath,
			TotalNanos: loadNanos,
			TotalMS:    float64(loadNanos) / 1e6,
			Phases:     []ggufload.LoadPhaseStat{{Phase: "q4k-direct-load", Calls: 1, Nanos: loadNanos, MS: float64(loadNanos) / 1e6, TimePct: 100}},
			Bottleneck: "q4k-direct-load",
		})
		return mm, true, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	default:
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

// resolveServeTokenizer picks the in-kernel chat planner's tokenizer: an explicit
// --tokenizer (a tokenizer.json or its directory, matching cmd/fakchat's resolution) wins;
// otherwise, with --gguf set, the GGUF's EMBEDDED tokenizer is used so /v1/chat/completions
// and /v1/messages serve real in-kernel chat by default (like cmd/simpledemo); otherwise it
// returns nil, leaving the gateway's offline MockPlanner fallback. The bool reports whether
// a tokenizer-load startup phase should be recorded.
func resolveServeTokenizer(tokPath, ggufPath string) (*tokenizer.Tokenizer, bool) {
	if tokPath != "" {
		tokFile := tokPath
		if info, err := os.Stat(tokFile); err == nil && info.IsDir() {
			tokFile = filepath.Join(tokFile, "tokenizer.json")
		}
		tok, err := tokenizer.LoadJSON(tokFile)
		must(err)
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
