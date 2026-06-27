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
	"github.com/anthony-chaudhary/fak/internal/snapshot"
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
	cudaGraph := fs.Bool("cuda-graph", false, "with --backend cuda: capture each decode token's whole op stream into a CUDA graph and replay it as ONE launch instead of N kernel launches (#483), the per-token launch-overhead lever for large single-stream decode (e.g. Qwen3.6-27B on an A100). OFF by default (a measured no-win on a tiny 0.5B/L4 where launch overhead is already small); witness tok/s before/after on YOUR node before relying on it. Equivalent to FAK_CUDA_GRAPH=1; inert on a non-cuda build or CPU backend.")
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
	elideResultBytes := fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes, "OFF by default: shrink oversized tool_result bodies outside the active working set to a bounded head+tail form once they exceed this byte threshold. 0 disables. The documented candidate is gateway.DocumentedElideResultBytes; flip only after reading the tradeoff witness.")
	sessionID := fs.String("session-id", "", "default trace/session id for callers that omit X-Trace-Id or MCP trace_id (empty = mint gw-N per request unless --context-budget-tokens is set)")
	sessionStatePath := fs.String("session-state", "", "COLD-RESUME the per-session DRIVE state across a process restart (#629): a fleet-snapshot file this `fak serve` RESTORES at boot — re-attaching every session at the budget/priority/run-state/pace it held, not its defaults (a STOPPED session reloads STOPPED with its reason, never silently RUNNING) — and REWRITES on a clean shutdown. Empty (default) = off, byte-for-byte today's path. Distinct from the live Paused→Running resume the /v1/fak/session control verbs already do.")
	contextBudgetTokens := fs.Int("context-budget-tokens", 0, "seed the default session with this prompt/context-token budget; exhaustion returns a reset directive with continuation_id (0 = off)")
	resetOnBudget := fs.Bool("reset-on-budget", false, "on context-budget exhaustion, re-arm the continuation trace with a carryover seed and continue transparently instead of returning 409 (requires --context-budget-tokens)")
	cpuOffloadExperts := fs.Bool("cpu-offload-experts", false, "with --gguf --backend: keep the MoE expert GEMMs on host RAM while dense projections + router + attention run on the device — the `--n-cpu-moe` hybrid that lets a model whose experts dwarf VRAM (e.g. GLM-5.2 Q4 ~424GB experts) serve at all on a smaller VRAM pool. The device load uses the memory-lean Q8 quantize-at-load path when the backend advertises quantized upload; otherwise it falls back to F32 weights until that backend implements UploadDtype.")
	budgetWebhook := fs.String("budget-webhook", "", "POST a JSON event to this URL when a served session's context budget crosses the warning threshold (--budget-warn-fraction) or is exhausted (the reset trigger), so an operator/monitor is notified before exhaustion (#743). Empty = off. Needs --context-budget-tokens to have a budget to watch.")
	budgetWarnFraction := fs.Float64("budget-warn-fraction", 0.8, "consumed share (0..1) of the context budget at which --budget-webhook fires its pre-exhaustion warning (default 0.8 = 80%); <=0 or >=1 disables the warning while the exhaustion event still fires")
	notifyNative := fs.Bool("notify-native", true, "emit a one-line native notification to stderr when a served session hits a PAUSED/DRAINING/STOPPED or budget boundary, carrying the closed stop-reason token — the SIGCHLD-equivalent so a waiting agent is never silent (#761); default on")
	notifyWebhook := fs.String("notify-webhook", "", "POST a JSON StopEvent to this URL on each served-session terminal/paused/budget boundary (#761), carrying the closed reason token; empty = off. Extends the #743 budget webhook to the full stop-reason vocabulary.")
	notifySlack := fs.String("notify-slack", "", "POST a Slack incoming-webhook payload ({\"text\":…}) on each served-session boundary (#761); empty = off")
	debugStats := fs.Bool("debug-stats", false, "print ONE compact, payload-free line per served turn to stderr: request/cache_read/cache_creation tokens, the compaction action, and the resetScore SHADOW health (healthy_cache|cache_decay|stale_prefix|cooldown|unknown_provider). Independent of --log (#793); default off.")
	dojoMode := fs.Bool("dojo", false, "enable live dojo mode: record each serve session as a dojo episode for later scoring with `fak dojo run`. Episodes are written to a dojo corpus directory under the workspace root.")
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
	// --cuda-graph flips the (init-time, FAK_CUDA_GRAPH-gated) graph-replay decode path on
	// from a parsed flag. graphEnabled is consulted per token at GraphBegin, so this post-init
	// flip cleanly activates the fully-wired HAL capture/replay path. No-op on a non-cuda build.
	if *cudaGraph {
		compute.EnableCUDAGraph()
		// Size the fixed device-KV prealloc to the served context so a real prompt never grows
		// the cache mid-capture (a cudaMalloc during capture is illegal — #932). Off-budget (0)
		// leaves the decode-bench default (1024). The prealloc is real VRAM (3 buffers × KV-heads
		// × head-dim × positions × 4B/layer), so an operator who wants a large graph context must
		// budget VRAM for it (or pair with the Q4_K weight lever to free room).
		compute.SetCUDAGraphKVCapacity(*contextBudgetTokens)
		fmt.Printf("fak: CUDA-graph decode replay enabled (#483), KV graph capacity=%d positions — witness tok/s before relying on it\n", max(*contextBudgetTokens, 1024))
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
	// COLD resume (#629): re-attach the persisted drive state of every session BEFORE the
	// per-boot default-budget seed, so a restart resumes each session at the budget/
	// priority/run-state/pace it held — not its defaults — while an explicit
	// --context-budget-tokens on THIS boot still re-seeds the default trace. A STOPPED
	// session reloads STOPPED with its reason (session.Table.Restore), never silently
	// resurrected as RUNNING. A missing file is a clean first boot; a present-but-corrupt
	// file fails loud (a tampered drive record is worse than none).
	if err := restoreServeSessions(serveSessions, *sessionStatePath); err != nil {
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(1)
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
		ElideResultBytes:            *elideResultBytes,
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

	// Stream every drive-state revision on /v1/fak/session/changes (#630). Wired
	// AFTER gateway.New so srv exists: each Rev bump of the process-local table
	// (a control verb, a debit, a continuation) is projected to the wire DTO and
	// pushed onto the gateway's bounded revision ring, where an operator drains it
	// by cursor — the live "what is every session doing right now" tail. The sink is
	// a cheap ring append and never re-enters the table (see session.RevisionObserver).
	serveSessions.WatchRevisions(func(s session.State) {
		srv.PublishSessionRevision(toGatewaySessionState(s))
	})

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

	// If --dojo is enabled, log the start of a live dojo episode.
	if *dojoMode {
		if err := logDojoEpisodeStart("serve"); err != nil {
			fmt.Fprintf(os.Stderr, "fak: --dojo episode logging failed: %v (continuing without dojo)\n", err)
		}
	}
	}

	if *stdio {
		// MCP over stdio: stdout carries the protocol; the log package writes to
		// stderr, so diagnostics never corrupt the frames.
		if err := srv.ServeStdio(ctx, os.Stdin, os.Stdout); err != nil && !errors.Is(err, context.Canceled) {
			must(err)
		}
		dumpServeSessions(serveSessions, *sessionStatePath) // #629: persist drive state for the next cold resume
		return
	}
	if *addr == "" {
		fmt.Fprintln(os.Stderr, "fak serve: --addr is required (or pass --stdio)")
		os.Exit(2)
	}
	if err := srv.ListenAndServe(ctx, *addr); err != nil && !errors.Is(err, http.ErrServerClosed) {
		must(err)
	}
	dumpServeSessions(serveSessions, *sessionStatePath) // #629: persist drive state for the next cold resume
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

const serveGGUFDeviceHeadroom = 0.15

// serveGGUFHostHeadroom reserves a fraction of the process host's allocatable RAM (MemAvailable)
// for the pure-CPU reference serve path's costs NOT in the header estimate: the resident-Q4K
// struct overshoot over the raw-payload estimate (~458 GiB resident vs ~433 GiB on-disk on
// GLM-5.2 UD-Q4_K_M, #974), gateway and KV init, and MemAvailable jitter as clean page cache is
// evicted during the multi-minute load. Matched to serveGGUFDeviceHeadroom for parity with the
// device fit plan, and comfortably above the observed ~6% resident overshoot.
const serveGGUFHostHeadroom = 0.15

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
	return appendServeGGUFDevicePlan(ws, plan, contextBudgetTokens), nil
}

func serveGGUFCPUOffloadMemoryPlan(ws *ggufload.WeightSource, contextBudgetTokens int) (compute.MemoryPlan, error) {
	if ws == nil {
		return nil, nil
	}
	plan, err := ws.EstimateCPUOffloadExpertsMemoryPlan()
	if err != nil {
		return nil, err
	}
	return appendServeGGUFDevicePlan(ws, plan, contextBudgetTokens), nil
}

func appendServeGGUFDevicePlan(ws *ggufload.WeightSource, plan compute.MemoryPlan, contextBudgetTokens int) compute.MemoryPlan {
	if cfg, err := ws.File.Config(); err == nil {
		if tokens := serveGGUFContextPlanTokens(cfg, contextBudgetTokens); tokens > 0 {
			plan = append(plan, compute.EstimateKVStoreMemoryPlan(compute.KVConfig{
				NumLayers:  cfg.NumLayers,
				NumKVHeads: cfg.NumKVHeads,
				HeadDim:    cfg.HeadDim,
				RopeTheta:  cfg.RopeTheta,
			}, tokens)...)
		}
		plan = append(plan, compute.EstimateHALTransientMemoryPlan(compute.TransformerScratchConfig{
			HiddenSize:       cfg.HiddenSize,
			IntermediateSize: cfg.IntermediateSize,
			VocabSize:        cfg.VocabSize,
			NumLayers:        cfg.NumLayers,
			NumHeads:         cfg.NumHeads,
			NumKVHeads:       cfg.NumKVHeads,
			HeadDim:          cfg.HeadDim,
			IncludeLogits:    true,
		})...)
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
	_, err := fitAndPlanServeGGUFPathOnDevice(ggufPath, be, f32Resident, contextBudgetTokens)
	return err
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
	plan, err := serveGGUFPathMemoryPlan(ggufPath, f32Resident, contextBudgetTokens)
	if err != nil {
		return err
	}
	return compute.RefuseMemoryPlanIfTooBigForHost(plan, serveGGUFHostHeadroom)
}

func fitAndPlanServeGGUFPathOnDevice(ggufPath string, be compute.Backend, f32Resident bool, contextBudgetTokens int) (compute.MemoryPlan, error) {
	plan, err := serveGGUFPathMemoryPlan(ggufPath, f32Resident, contextBudgetTokens)
	if err != nil {
		return nil, err
	}
	if be == nil {
		return plan, nil
	}
	return plan, compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

func fitServeGGUFCPUOffloadPathOnDevice(ggufPath string, be compute.Backend, contextBudgetTokens int) error {
	if ggufPath == "" || be == nil {
		return nil
	}
	_, err := fitAndPlanServeGGUFCPUOffloadPathOnDevice(ggufPath, be, contextBudgetTokens)
	return err
}

func fitAndPlanServeGGUFCPUOffloadPathOnDevice(ggufPath string, be compute.Backend, contextBudgetTokens int) (compute.MemoryPlan, error) {
	plan, err := serveGGUFCPUOffloadPathMemoryPlan(ggufPath, contextBudgetTokens)
	if err != nil {
		return nil, err
	}
	if be == nil {
		return plan, nil
	}
	return plan, compute.RefuseMemoryPlanIfTooBig(be, plan, serveGGUFDeviceHeadroom)
}

func serveGGUFPathMemoryPlan(ggufPath string, f32Resident bool, contextBudgetTokens int) (compute.MemoryPlan, error) {
	if ggufPath == "" {
		return nil, nil
	}
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return serveGGUFMemoryPlan(ws, f32Resident, contextBudgetTokens)
}

func serveGGUFCPUOffloadPathMemoryPlan(ggufPath string, contextBudgetTokens int) (compute.MemoryPlan, error) {
	if ggufPath == "" {
		return nil, nil
	}
	ws, err := ggufload.OpenWeights(ggufPath)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return serveGGUFCPUOffloadMemoryPlan(ws, contextBudgetTokens)
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
		prof := ggufload.NewLoadProfiler()
		prof.Progress = os.Stderr // stream load % to stderr so a large multi-minute load is not silent
		mm, err := ggufload.LoadModelQ4KProfile(ggufPath, prof)
		must(err)
		modelengine.PreloadQ4K(mm)
		// Operator visibility: the post-load resident split (raw Q4_K + raw Q5/6_K experts vs
		// the Q8 dequant minority) so a glance confirms the mixed-quant expert bulk loaded
		// resident — the slow f32 round-trip avoided — alongside the per-quant-type load-path
		// summary the profiler already streamed.
		fmt.Fprintln(os.Stderr, "fak: "+fakmodel.FormatResidentReport(mm.ResidentReport()))
		loadNanos := time.Since(tLoad).Nanoseconds()
		profile := withServeGGUFMemoryProfile(toGatewayLoadProfile(prof.Snapshot("gguf-resident-q4k-device", ggufPath, loadNanos)), memPlan, backend)
		return mm, true, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
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
		prof := ggufload.NewLoadProfiler()
		prof.Progress = os.Stderr // stream load % to stderr so a large multi-minute load is not silent
		mm, err := ggufload.LoadModelQ4KProfile(ggufPath, prof)
		must(err)
		modelengine.PreloadQ4K(mm)
		fmt.Fprintln(os.Stderr, "fak: "+fakmodel.FormatResidentReport(mm.ResidentReport()))
		loadNanos := time.Since(tLoad).Nanoseconds()
		profile := withServeGGUFMemoryProfile(toGatewayLoadProfile(prof.Snapshot("gguf-resident-q4k-device", ggufPath, loadNanos)), memPlan, backend)
		return mm, true, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
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
		prof := ggufload.NewLoadProfiler()
		prof.Progress = os.Stderr // stream load % + the load-path summary to stderr so the load is not silent
		mm, err := ggufload.LoadModelQ4KProfile(ggufPath, prof)
		must(err)
		modelengine.PreloadQ4K(mm)
		// Operator visibility: the post-load resident split confirms at a glance that the
		// mixed-quant expert bulk loaded resident (the slow f32 round-trip avoided), alongside
		// the per-quant-type load-path summary the profiler already streamed.
		fmt.Fprintln(os.Stderr, "fak: "+fakmodel.FormatResidentReport(mm.ResidentReport()))
		loadNanos := time.Since(tLoad).Nanoseconds()
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
