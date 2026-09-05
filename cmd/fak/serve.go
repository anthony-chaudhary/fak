package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/snapshot"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
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

// streamProgressTimeoutOff is the gateway.Config/HTTPPlanner encoding for "no
// content-progress deadline at all": a NEGATIVE duration, which
// agent.(*HTTPPlanner).streamProgressWindow resolves to a disabled window. It is deliberately
// NOT the zero value — on that field zero means "unconfigured, take
// agent.DefaultStreamProgressTimeout", so the operator's off switch cannot be passed through
// as-is and has to be translated here.
const streamProgressTimeoutOff = -1 * time.Second

// serveStreamProgressTimeout maps --stream-progress-timeout onto that encoding.
//
// The flag spells "off" as 0 because that is what every other serve knob with an off switch
// spells it as (--ctx-view-budget 0, --compact-history-budget 0, --elide-result-bytes 0,
// --assume-session-turns 0, --metrics-snapshot 0): an operator who wants no deadline types a
// zero, never a negative duration. The flag DEFAULTS to agent.DefaultStreamProgressTimeout
// rather than 0, so a zero here is always something the operator typed — the ambiguity the
// raw config field has (where 0 is the unconfigured state) does not exist at the front door.
// Every other value rides through verbatim for streamProgressWindow to band-check.
func serveStreamProgressTimeout(d time.Duration) time.Duration {
	if d <= 0 {
		return streamProgressTimeoutOff
	}
	return d
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
	configPath                   *string
	printEffectiveConfig         *bool
	addr                         *string
	stdio                        *bool
	provider                     *string
	baseURL                      *string
	replicaBaseURLs              repeatedStringFlag
	model                        *string
	apiKeyEnv                    *string
	streamProgressTimeout        *time.Duration
	engineCacheEngine            *string
	engineCacheBaseURL           *string
	engineCacheAdminKeyEnv       *string
	engineCacheIdleTimeout       *time.Duration
	engineCacheRequireExactSpan  *bool
	remoteKVMode                 *string
	remoteKVBackend              *string
	remoteKVURL                  *string
	remoteKVToken                *string
	remoteKVTimeout              *time.Duration
	engineID                     *string
	backendName                  *string
	qwen38Runtime                *string
	llamaServer                  *string
	llamaStartupTimeout          *time.Duration
	cudaGraph                    *bool
	policyPath                   *string
	profile                      *string
	policyCanaryTurns            *int
	policyCheck                  *bool
	sizingJSON                   *bool
	expose                       repeatedStringFlag
	vdso                         *bool
	invalidation                 *string
	requireKeyEnv                *string
	keyPrincipal                 repeatedStringFlag
	unsafeUnauthedBind           *bool
	routeManifest                *string
	routeAccounts                *string
	ggufPath                     *string
	tokPath                      *string
	ctxViewBudget                *int
	compactHistoryBudget         *int
	positiveResidualSubstitution *bool
	compactAnchorHead            *bool
	vcacheAnchor                 *bool
	deferColdTools               *bool
	deferTools                   *bool
	toolCeiling                  *int
	assumeSessionTurns           *int
	elideResultBytes             *int
	elideStaleReads              *bool
	sessionID                    *string
	sessionStatePath             *string
	sessionRegistry              *string
	contextBudgetTokens          *int
	resetOnBudget                *bool
	cpuOffloadExperts            *bool
	nCPUMoE                      *string
	nativeGPULayers              *int
	nativeQwenQ4KPrefillChunk    *int
	nativeQwen35MetalGDNSequence *bool
	nativeQ4KGateUpOutputSlab    *bool
	nativePrefixProfile          *string
	vulkanQ4KProfile             *bool
	vulkanStageQ4K               *bool
	metal                        *bool
	gpudirectOverflow            *bool
	expertParallel               *int
	tensorParallel               *int
	budgetWebhook                *string
	budgetWarnFraction           *float64
	spendCap                     repeatedStringFlag
	spendScopeTrace              *string
	notifyNative                 *bool
	notifyWebhook                *string
	notifySlack                  *string
	debugStats                   *bool
	otlpEndpoint                 *string
	dojoMode                     *bool
	native                       *bool
	nativeMaxTurns               *int
	nativeAdmissionTokenBudget   *int
	nativeCodeWorkspace          *string
	nativeCodeTools              *bool
	nativeSpeculate              *bool
	vdsoProxyFill                *bool
	metricsSnapshot              *time.Duration
	fleetBus                     *bool
	fleetBusDir                  *string
	fleetBusID                   *string
	fleetBusInterval             *time.Duration
	keepAwake                    *string
}

// newServeFlagSet defines the full `fak serve` flag surface and returns the set
// plus the struct the boot stages read the parsed values through.
func newServeFlagSet() (*flag.FlagSet, *serveFlags) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	verbFlagUsage(fs, "serve")
	configureServeHelp(fs)
	sf := &serveFlags{}
	sf.configPath = fs.String("config", "", "load reviewable deployment defaults from fak.toml (explicit flags override; no implicit ambient lookup)")
	sf.printEffectiveConfig = fs.Bool("print-effective-config", false, "print supported effective serve configuration with value provenance, then exit without binding a listener")
	sf.addr = fs.String("addr", "127.0.0.1:8080", "HTTP listen address (OpenAI + fak + /mcp surface); ignored with --stdio")
	sf.stdio = fs.Bool("stdio", false, "serve MCP over stdin/stdout (newline-delimited JSON-RPC) instead of HTTP")
	sf.provider = fs.String("provider", "openai", "upstream provider transcript wire: openai, anthropic, gemini, or xai")
	sf.baseURL = fs.String("base-url", "", "upstream provider base URL for the /v1/chat/completions proxy (empty = offline mock planner)")
	fs.Var(&sf.replicaBaseURLs, "replica-base-url", "additional upstream provider base URL for a static round-robin replica fleet; repeat for N replicas. If --base-url is set, it is replica 1. Each replica's identity defaults to a stable endpoint-derived id (replica-<digest>) so the same upstream keeps its metric/residency labels regardless of flag order or a dropped peer; pass name=URL to pin an operator-chosen id.")
	sf.model = fs.String("model", "mock", "model id (advertised by /v1/models; used for the upstream call)")
	sf.apiKeyEnv = fs.String("api-key-env", "", "env var holding the upstream API key (proxy mode)")
	sf.streamProgressTimeout = fs.Duration("stream-progress-timeout", agent.DefaultStreamProgressTimeout, "proxy mode: end a STREAMING upstream turn that has stayed warm this long without a single frame that advances it (#5486). Keepalive frames (a ping, an SSE comment, an empty-delta chunk) re-arm the inter-byte deadline but are NOT progress, so a generation wedged behind a live socket otherwise rides the 600s whole-request ceiling. DEFAULT-ON at agent.DefaultStreamProgressTimeout (300s), which sits above the worst prefill-to-first-token gap on a large cached prompt and above any extended-thinking pause (thinking streams content deltas, which do count as progress). Pass 0 to DISABLE the deadline — the escape hatch when a provider's prefill legitimately outlasts the window. A positive value outside [5s, 600s] is not honored as a real window: the default is used instead, so a typo never silently becomes a different deadline. Inert on the non-streaming path and on the offline mock planner.")
	sf.engineCacheEngine = fs.String("engine-cache-engine", "", "self-hosted upstream cache reset engine for quarantined provider-bound tool results: sglang|vllm (empty disables)")
	sf.engineCacheBaseURL = fs.String("engine-cache-base-url", "", "serving-engine control/base URL for cache reset (default: --base-url when --engine-cache-engine is set)")
	sf.engineCacheAdminKeyEnv = fs.String("engine-cache-admin-key-env", "", "env var holding the serving-engine admin API key for cache reset")
	sf.engineCacheIdleTimeout = fs.Duration("engine-cache-idle-timeout", 0, "SGLang /flush_cache idle timeout, e.g. 30s (0 fails fast)")
	sf.engineCacheRequireExactSpan = fs.Bool("engine-cache-require-exact-span", false, "require exact remote K/V/index span eviction; fail closed if the selected engine only supports whole-cache reset")
	sf.remoteKVMode = fs.String("remote-kv-mode", string(l3kv.RemoteKVModeOptional), "remote KV store availability mode: optional|required|disabled (default: optional)")
	sf.remoteKVBackend = fs.String("remote-kv-backend", "l3kv-blobhttp", "remote KV backend: l3kv-blobhttp|none")
	sf.remoteKVURL = fs.String("remote-kv-url", "", "remote KV store URL (default: FAK_REMOTE_KV_URL or FAK_BLOB_HTTP_URL)")
	sf.remoteKVToken = fs.String("remote-kv-token", "", "remote KV store auth token (default: FAK_REMOTE_KV_TOKEN or FAK_BLOB_HTTP_TOKEN)")
	sf.remoteKVTimeout = fs.Duration("remote-kv-timeout", l3kv.DefaultRemoteKVTimeout, "remote KV connectivity probe timeout")
	sf.engineID = fs.String("engine", "mock", "registered engine id that fak_syscall dispatches an allowed call to: mock, inkernel, vllm, sglang, llm-d, dynamo, or another registered driver (default: mock; select inkernel explicitly for fak-native model execution)")
	sf.backendName = fs.String("backend", "", "compute backend for the in-kernel chat decode (with --gguf, no --base-url): empty = the CPU reference path; a registered device name like 'cuda' runs prefill+decode through the GPU HAL. Requires a `-tags cuda` build AND a reachable GPU at runtime; fails loud if named but unavailable so a typo never silently runs on CPU.")
	nativeControls := registerNativeControlFlags(fs)
	sf.nativeQwenQ4KPrefillChunk = nativeControls.prefillChunk
	sf.nativeQwen35MetalGDNSequence = nativeControls.qwen35GDNSequence
	sf.nativeQ4KGateUpOutputSlab = nativeControls.q4kGateUpSlab
	sf.nativePrefixProfile = nativeControls.prefixProfile
	sf.vulkanQ4KProfile = nativeControls.vulkanQ4KProfile
	sf.vulkanStageQ4K = nativeControls.vulkanStageQ4K
	sf.qwen38Runtime = fs.String("qwen38-runtime", qwen38RuntimeNative, "Qwen3.8 GGUF execution: native (default) keeps fak in-kernel execution; llama-mtp explicitly delegates to a capability-proven llama-server for benchmark/reference interoperability. There is no external-runtime fallback, and the removed auto value is rejected.")
	sf.llamaServer = fs.String("llama-server", "llama-server", "versioned llama-server binary used only by explicit --qwen38-runtime llama-mtp benchmark/reference interoperability")
	sf.llamaStartupTimeout = fs.Duration("llama-startup-timeout", 2*time.Minute, "bounded readiness timeout for a fak-owned llama-server child")
	sf.cudaGraph = fs.Bool("cuda-graph", false, "with --backend cuda: capture each decode token's whole op stream into a CUDA graph and replay it as ONE launch instead of N kernel launches (#483), the per-token launch-overhead lever for large single-stream decode (e.g. Qwen3.6-27B on an A100). OFF by default (a measured no-win on a tiny 0.5B/L4 where launch overhead is already small); witness tok/s before/after on YOUR node before relying on it. Equivalent to FAK_CUDA_GRAPH=1; inert on a non-cuda build or CPU backend.")
	sf.policyPath = fs.String("policy", "", "capability-floor manifest to load (default: the built-in adjudicator floor — the tau2 airline-demo tools, NOT the `fak guard` coding floor; see `fak policy --dump`)")
	sf.profile = fs.String("profile", "", "permission profile: dev|prod|audit (env: FAK_PROFILE)")
	sf.policyCanaryTurns = fs.Int("policy-canary-turns", 0, "after a policy reload, roll back when this many consecutive requests are denied; 0 disables the canary")
	sf.policyCheck = fs.Bool("policy-check", false, "validate --policy and exit without binding a listener")
	sf.sizingJSON = fs.Bool("plan-json", false, "with --gguf: print the versioned header-derived memory sizing artifact (classed demands, disk/ram/vram tier rollup, per-pool usable bytes after headroom, warnings incl. would-be fit refusals) as JSON on stdout and exit BEFORE any load — nothing is allocated, no listener binds (#4361). The numbers are the same ones the selected serve arm's fit check admits against; a demand set a live boot would refuse still emits, with the refusal in warnings[].")
	fs.Var(&sf.expose, "expose", "ALLOWLIST of MCP tool-name glob patterns to advertise AND allow — everything else is neither listed by tools/list nor callable (an attempt answers \"unknown tool\", so hiding a tool never leaks that it exists). Patterns are path.Match globs over the bare tool name; one value may be comma-separated (--expose 'fak_memory_*,fak_capabilities') and the flag may repeat. Empty (default) exposes the full surface. A malformed glob or a pattern matching NO known tool fails startup loud (a typo must not silently shrink the surface). Cuts prompt-prefix token cost by advertising only the tools a given harness uses; for the leanest surface expose just the discovery tools (--expose 'fak_tools_search,fak_capabilities') and let the agent page the rest in on demand.")
	sf.vdso = fs.Bool("vdso", true, "enable the vDSO dedup fast path")
	sf.invalidation = fs.String("invalidation", "global", "vDSO tier-2 invalidation granularity for the live fleet: global|namespace|resource")
	sf.requireKeyEnv = fs.String("require-key-env", "", "env var holding a bearer token to REQUIRE on every request (default: no auth)")
	fs.Var(&sf.keyPrincipal, "key-principal", "bind an ADDITIONAL api key to an org/project PRINCIPAL as PRINCIPAL=ENV_VAR, so N corporate keys authenticate against one `fak serve` and every turn is attributed to the tenant that made it (#5332). The value names the env var HOLDING that tenant's key, never the key itself — the raw secret never lands in a unit file, a plist, or shell history, and gateway.New hashes it to a SHA-256 digest and drops the plaintext at construction. Repeat for N tenants (--key-principal acme=ACME_KEY --key-principal beta=BETA_KEY); binding one principal to several env vars is key ROTATION and is allowed. A matching inbound key (x-api-key or Authorization: Bearer) both AUTHENTICATES the caller and stamps its principal onto the access log and /v1/fak/events, joinable by X-Trace-Id. ADDITIVE to --require-key-env: that single bearer still authenticates the anonymous single-tenant caller and an empty keyset leaves that path byte-for-byte unchanged. Fails startup LOUD on a malformed spec, an unset/empty env var, or two tenants sharing a key — a keyset that silently forgot a binding looks armed and is not.")
	sf.unsafeUnauthedBind = fs.Bool(serveUnsafeBindFlag, false, "proceed with a bind that is reachable from OFF THIS HOST even though no inbound token door is configured (#5373, a child of #3279). Default false: `fak serve --addr 0.0.0.0:8080` with neither --require-key-env nor --key-principal is REFUSED at startup with the "+serveBindRefusalToken+" reason, because every request such a listener serves is unauthenticated and the internet-wide scan #3279 cites found 175,108 local-model servers in exactly that shape. Loopback binds (the 127.0.0.1:8080 default, localhost, ::1) are never affected, nor is --stdio, nor is an off-host bind that DOES name a token door — so this flag is only ever needed for the deliberate case: an isolated lab segment, or a host firewall doing the work instead. Passing it prints a loud stderr warning every boot; it is named to be impossible to set by accident and is not a substitute for --require-key-env.")
	sf.routeManifest = fs.String("route-manifest", "", "model-routing policy to install: each fak_syscall call is classified into a modelroute.Subject and a single-model (PICK) plan binds abi.ToolCall.Engine before Submit, so the residency PDP adjudicates the real route (#601). Empty (default) leaves Engine unset → the kernel default engine, byte-for-byte the pre-routing behavior. A malformed manifest fails startup loud (a mis-routed model is a security boundary, never a silent default). The installed file is HOT-RELOADED: an edit is picked up without a restart and swapped atomically (a request classifies against the whole old or whole new policy, never a torn read); a malformed edit is rejected and the last-good policy stays installed (#842).")
	sf.routeAccounts = fs.String("route-accounts", "", "model-ACCOUNT roster (fak-accounts/v1) to install ALONGSIDE --route-manifest (#2528): after the manifest PICKs an abstract model id, the roster BINDS it to a concrete provider account + upstream wire model, and the account-resolved EngineRoute (openai:acct/model, local:acct/model) — not the bare plan-member string — is written to abi.ToolCall.Engine before Submit, so the residency PDP adjudicates the ACCOUNT-resolved route and an ensemble member each binds independently. A route to a provider with no registered adapter fails LOUD at dispatch (no silent fallback to the default engine). Credentials are env-var NAMES in the roster, never secrets. Empty (default) leaves the plan-member string as the route, byte-for-byte the pre-#2528 behavior. A malformed roster fails startup loud. Preflight it no-spend first with `fak api-host acceptance --from-model-accounts FILE`.")
	sf.ggufPath = fs.String("gguf", "", "load these GGUF weights into the in-kernel engine at boot; the load is part of the measured startup sequence and its phase breakdown is exposed on /metrics. Default path is lean-Q8 (Q4→f32→Q8 round-trip); set FAK_Q4K=1 for the direct-resident-Q4_K path (Qwen3.6-27B q4_k_m, the P1/P2 decode lever)")
	sf.tokPath = fs.String("tokenizer", "", "OPTIONAL override for the in-kernel CHAT planner's tokenizer. With --gguf and no --base-url, /v1/chat/completions AND /v1/messages already serve the in-kernel model (real ChatML chat) using the GGUF's EMBEDDED tokenizer; pass this only to override it (e.g. an SPM-only checkpoint with no embedded BPE tokenizer, or a custom vocab). Accepts a tokenizer.json or its directory. e.g. ~/.cache/fak-models/tokenizers/qwen3.6")
	sf.ctxViewBudget = fs.Int("ctx-view-budget", agent.DefaultCtxViewBudget, "wire the ctxplan context PLANNER into the live serve loop: each buffered turn, re-materialize the forwarded history as an O(1) planned VIEW under this resident-token budget (a planned view in place of appending the whole transcript, #555). DEFAULT-ON at a conservative 8000 resident tokens; pass 0 to disable (leaves the existing path byte-for-byte unchanged). The planner only ever SHORTENS and falls open to the full history on any doubt; on the Anthropic passthrough it keeps the cached prefix byte-identical (witness: docs/notes/CTXVIEW-DEFAULT-ON-WITNESS-2026-06-28.md). The streaming fast-path bypasses this; the buffered turn path is what gets planned.")
	sf.compactHistoryBudget = fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "on the Anthropic PASSTHROUGH (an upstream --base-url anthropic), compact OLD conversation turns in the OUTBOUND request body down to this resident-token budget while keeping the cache_control prefix BYTE-IDENTICAL, so the upstream cache hit survives. This reaches the flagship passthrough the streaming ctxplan view cannot (#555). DEFAULT-ON: once a conversation sprawls past ~48k resident tokens the cut fires and sheds the un-cacheable middle the provider re-bills every turn; a typical short session stays untouched. Pass 0 to disable (body forwarded byte-for-byte). No effect on non-passthrough wires.")
	sf.positiveResidualSubstitution = fs.Bool("positive-residual-substitution", false, "replace compacted negated/stale history with its positive residual; original bytes remain available through ctxrestore (default off)")
	sf.compactAnchorHead = fs.Bool("compact-anchor-head", true, "re-anchor --compact-history-budget's protected prefix on the stable system/tools head instead of the first-breakpoint anchor, fixing the anchor-starved trap (#1407) where real Claude Code traffic's recent cache_control breakpoint protects almost the whole conversation so the budget can never shed anything. DEFAULT-ON, and every fire stays gated on the burst economics (CacheBurstPaysBack, #1408): a WARM session with no bounded turns budget never bursts — it fires only when a wired session-turn horizon repays the one-time burst, or when the trace OBSERVABLY idled past the message-breakpoint cache TTL since its last served turn (a penalty-free cut, the long-session firing path). Pass =false to pin the old warm-only first-breakpoint anchor.")
	sf.assumeSessionTurns = fs.Int("assume-session-turns", gateway.DefaultAssumedSessionTurns, "the session length the head-anchored burst gate (--compact-anchor-head) ASSUMES when no bounded turn horizon is wired, so a WARM continuously-active long session sheds early instead of waiting to idle past the message-span cache TTL. The gate maps the trace's real served-turn depth to CurrentTurn and this value to TotalTurns: it fires early (many repaying turns left) and refuses near the presumed end — the same one-time-burst break-even economics (agent.CacheBurstPaysBack, #1408), just given a history-based length instead of refusing outright. DEFAULT-ON at gateway.DefaultAssumedSessionTurns; a genuine wired Budget.TurnsLeft horizon always WINS over this prior, and a large invalidated suffix still refuses regardless. Pass 0 to disable (byte-for-byte the conservative no-horizon behavior — no fire unless the burst is zero-penalty). Consulted only when --compact-anchor-head is on and the head re-anchor engages; inert on every other path.")
	sf.elideResultBytes = fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes, "ON by default at gateway.DefaultElideResultBytes (the reviewed gateway.DocumentedElideResultBytes threshold): shrink oversized tool_result bodies outside the active working set to a bounded head+tail form once they exceed this byte threshold. 0 disables.")
	sf.elideStaleReads = fs.Bool("elide-stale-reads", gateway.DefaultElideStaleReads, "ON by default (gateway.DefaultElideStaleReads): replace a Read tool_result whose file was Edited/Written in a LATER in-session turn (a stale, superseded snapshot no longer reflecting disk) with a compact fak_context_restore marker, in the SAME cache-safe working-set band as --elide-result-bytes and stashing the pre-edit body behind a restore handle. The safer, restorable sibling of --elide-result-bytes: strictly more conservative predicate (superseded, not merely big), fail-safe identity on any ambiguity, protected cache prefix proven byte-identical. Size-independent; lossy but restorable. Pass =false to opt out. No effect on non-passthrough wires.")
	sf.vcacheAnchor = fs.Bool("vcache-anchor", gateway.DefaultVCacheAnchor, "M2 star-anchor pre-flight gate (#1493): on the Anthropic passthrough (an upstream --base-url anthropic), APPLY cachemeta.RecommendLayout before send — hoist volatile system blocks behind a byte-stable cacheable anchor and splice a cache_control breakpoint onto the stable head a no-breakpoint caller did NOT send, so the first natural request warms provider prefix caching and later siblings read it. DEFAULT-ON, DECOUPLED from --compact-history-budget (that path only placed the anchor while its own budget was >0, so --compact-history-budget=0 silently took anchoring down with it). Fail-safe identity on any ambiguity — a hoist that would change the model-visible prefix is REFUSED, not applied — and idempotent with the compaction/TTL placements (a body already carrying a breakpoint bails already_set). Pass =false to opt out. No effect on non-passthrough wires.")
	sf.deferColdTools = fs.Bool("defer-cold-tools", gateway.DefaultDeferColdTools, "the 10x floor lever (#3232, epic #3229): on the OUTBOUND Anthropic body, mark every allowed-but-COLD custom tool `defer_loading:true` and inject one `tool_search_tool`, so the provider loads only the HOT core into context and faults a cold schema in on demand. Deterministic + cache-safe (byte-stable tools[] turn-over-turn) and fail-safe identity on any ambiguity; every deferred def stays byte-complete in tools[], so a first real use still resolves. DEFAULT ON (gateway.DefaultDeferColdTools, the #3537 flip; the A/B and #3200 fault-in gates reported PASS). Pass =false to opt out; ablate an A/B arm with FAK_ABLATE_DEFER_TOOLS=1 (FAK_DEFER_COLD_TOOLS=1 still forces it on). Anthropic passthrough only.")
	sf.deferTools = fs.Bool("defer-tools", true, "defer cold MCP tools on tools/list (default true); pass --defer-tools=false to advertise the full tool registry on tools/list (required for harnesses like OpenCode that do not support dynamic runtime tool paging)")
	sf.toolCeiling = fs.Int("tool-ceiling", gateway.DefaultMCPToolAdvertisementCeiling, "ceiling on advertised MCP tools on tools/list when deferral is disabled (default 10); clamps the advertised set to a curated active set to limit token overhead; 0 for unbounded")
	sf.sessionID = fs.String("session-id", "", "default trace/session id for callers that omit X-Trace-Id or MCP trace_id (empty = mint gw-N per request unless --context-budget-tokens is set)")
	sf.sessionStatePath = fs.String("session-state", "", "COLD-RESUME the per-session DRIVE state across a process restart (#629): a fleet-snapshot file this `fak serve` RESTORES at boot — re-attaching every session at the budget/priority/run-state/pace it held, not its defaults (a STOPPED session reloads STOPPED with its reason, never silently RUNNING) — and REWRITES on a clean shutdown. Empty (default) = off, byte-for-byte today's path. Distinct from the live Paused→Running resume the /v1/fak/session control verbs already do.")
	sf.sessionRegistry = fs.String("session-registry", "", "SCOPE THE SESSIONS THIS SERVE CAN REACH (#5825). The session table is hydrated from this registry, and that table is what a fanned lifecycle op writes through — so this path, not --fleet-bus-dir, is what decides whose sessions `fak fleet control send --op pause --all` touches. --fleet-bus-dir scopes only the BUS; a serve pointed at a private bus directory still adopts every session in this registry, which is how a rehearsal once paused 12 sessions belonging to other workers. Empty keeps today's behaviour: the shared per-user default (FAK_SESSION_REGISTRY, else <UserConfigDir>/fak/session-registry.json), the correct reach for a real fleet. Name a path to adopt and persist only your own sessions — the safe way to rehearse on a shared host. Use 'off' for a pure in-memory table that adopts nothing and persists nothing.")
	sf.contextBudgetTokens = fs.Int("context-budget-tokens", 0, "seed the default session with this prompt/context-token budget; exhaustion returns a reset directive with continuation_id (0 = off)")
	sf.resetOnBudget = fs.Bool("reset-on-budget", false, "on context-budget exhaustion, re-arm the continuation trace with a carryover seed and continue transparently instead of returning 409 (requires --context-budget-tokens)")
	sf.cpuOffloadExperts = fs.Bool("cpu-offload-experts", false, "with --gguf --backend: keep the MoE expert GEMMs on host RAM while dense projections + router + attention run on the device — the `--n-cpu-moe` hybrid that lets a model whose experts dwarf VRAM (e.g. GLM-5.2 Q4 ~424GB experts) serve at all on a smaller VRAM pool. The device load uses the memory-lean Q8 quantize-at-load path when the backend advertises quantized upload; otherwise it falls back to F32 weights until that backend implements UploadDtype.")
	sf.nCPUMoE = fs.String(serveNCPUMoEFlag, "", "with --gguf --backend: GRADE the expert spill instead of taking --cpu-offload-experts' all-or-nothing split (#5628, epic #5606). `auto` sizes the number of host-spilled MoE layers against the device budget compute.DeviceMemoryInfo measures, keeping the rest device-resident behind a bounded expert ring; `N` spills exactly N MoE layers; `off` (the default) is the ungraded placement --cpu-offload-experts alone makes, byte-for-byte. Spelled as llama.cpp spells it, so a working --n-cpu-moe number carries over. A grade that is not auto/off/a count >= 0 REFUSES the launch here, before the multi-minute load — a misspelled grade must never fall back to a placement the operator did not choose. Equivalent to "+agent.ExpertSpillEnv+"; passing the flag WINS over that env var, including an explicit `off`.")
	sf.nativeGPULayers = fs.Int("gpu-layers", 0, "number of contiguous layers [0, N) to place on the GPU (Backend), with the remainder executing on host CPU; alias for --native-gpu-layers")
	fs.IntVar(sf.nativeGPULayers, "native-gpu-layers", 0, "number of contiguous layers [0, N) to place on the GPU (Backend), with the remainder executing on host CPU (alias for --gpu-layers)")
	sf.metal = fs.Bool("metal", false, "with --gguf (no --base-url), require the Apple-Silicon Metal GPU forward — GPU prefill + GPU-resident Q8 decode (#67, ~0.99x of llama.cpp-Metal on dense Qwen2.5-7B Q8). Apple-Silicon+cgo builds auto-select Metal when a usable device is present; this flag/FAK_METAL=1 makes absence fail loud instead of falling back to CPU. Mutually exclusive with --backend (Metal is the CPU-session seam, not a compute HAL device). Dense Qwen-class Q8 GGUFs only — a MoE/hybrid model (GLM-5.2, GDN) self-declines to CPU decode.")
	sf.gpudirectOverflow = fs.Bool("gpudirect-overflow", true, "enable AMD GPU Direct / NVMe P2PDMA zero-copy storage for KV cache and layer overflow handling (bypasses CPU bounce buffers on VRAM saturation; default on)")
	sf.expertParallel = fs.Int("expert-parallel", 1, "with --gguf: shard the routed MoE experts of a glm_moe_dsa model (GLM-5.2) across N expert-parallel ranks — the lever to move supported expert GEMMs off the host (the `--cpu-offload-experts` wall) onto resident GPUs (#971). Mixed k-quant expert formats without backend kernels (for example Q5_K/Q6_K today) still use the host k-quant fallback; set FAK_KQ_INT8=1 to use its production int8 path. The per-rank residual partials are reduced by one AllReduceSum through the wired Collective. 1 (default) = the unchanged monolith forward. N>1 requires an initialized non-cpu-ref compute.CollectiveBackend; CUDA builds provide that only with -tags cuda,nccl (build_cuda.sh: FAK_CUDA_NCCL=1) on a box with enough visible GPUs.")
	sf.tensorParallel = fs.Int("tensor-parallel", 1, "with --gguf: tensor-parallel rank count for the dense projections (the Megatron column/row split, tensor_parallel.go). 1 (default) = no split. N>1 uses the same initialized device-collective gate as --expert-parallel; CUDA builds require -tags cuda,nccl (build_cuda.sh: FAK_CUDA_NCCL=1).")
	sf.budgetWebhook = fs.String("budget-webhook", "", "POST a JSON event to this URL when a served session's context budget crosses the warning threshold (--budget-warn-fraction) or is exhausted (the reset trigger), so an operator/monitor is notified before exhaustion (#743). Also carries the --spend-cap breach event (kind:\"spend_breach\", #4859) when a scope crosses its token/USD budget. Empty = off. Needs --context-budget-tokens to have a budget to watch.")
	sf.budgetWarnFraction = fs.Float64("budget-warn-fraction", 0.8, "consumed share (0..1) of the context budget at which --budget-webhook fires its pre-exhaustion warning (default 0.8 = 80%); <=0 or >=1 disables the warning while the exhaustion event still fires")
	fs.Var(&sf.spendCap, "spend-cap", "arm the control-plane SPEND CAP (#3273) on a budget scope: SCOPE[:ID]=SPEC, repeatable. SCOPE is tenant|team|agent|session; :ID caps one instance (omitted = the scope default for every id); SPEC is a bare token count, or comma-separated tokens=N,usd=N (micro-USD),action=pause|kill (default pause). A scope past its budget is hard-stopped BY THE KERNEL at the next turn (409 BUDGET_SPEND_EXCEEDED, before the model is consulted) and POSTs to --budget-webhook. e.g. --spend-cap session=200000 --spend-cap tenant:acme=tokens=5000000,action=kill. Empty (default) = no cap, request path unchanged.")
	sf.spendScopeTrace = fs.String("spend-scope-trace", "", "how a request's trace id maps onto the --spend-cap scope ladder: a \"/\"-separated template naming what each trace segment carries, e.g. \"tenant/team/session\" reads the trace \"acme/core/s-17\" as tenant=acme team=core session=s-17. Empty (default) = session-only, the whole trace id is the session. A --spend-cap on a scope this template never populates fails startup rather than booting an uncapped server that looks capped.")
	sf.notifyNative = fs.Bool("notify-native", true, "emit a one-line native notification to stderr when a served session hits a PAUSED/DRAINING/STOPPED or budget boundary, carrying the closed stop-reason token — the SIGCHLD-equivalent so a waiting agent is never silent (#761); default on")
	sf.notifyWebhook = fs.String("notify-webhook", "", "POST a JSON StopEvent to this URL on each served-session terminal/paused/budget boundary (#761), carrying the closed reason token; empty = off. Extends the #743 budget webhook to the full stop-reason vocabulary.")
	sf.notifySlack = fs.String("notify-slack", "", "POST a Slack incoming-webhook payload ({\"text\":…}) on each served-session boundary (#761); empty = off")
	sf.otlpEndpoint = fs.String("otlp-traces-endpoint", "", "optional OTLP/HTTP endpoint; appends /v1/traces")
	sf.debugStats = fs.Bool("debug-stats", false, "print ONE compact, payload-free line per served turn to stderr: request/cache_read/cache_creation tokens plus current/previous/average/median/high/low cache savings, the compaction action, and the resetScore SHADOW health (healthy_cache|cache_decay|stale_prefix|cooldown|unknown_provider). Independent of --log (#793); default off.")
	sf.dojoMode = fs.Bool("dojo", false, "enable live dojo mode: write a start-marker for each serve session into the live-episode corpus (.dojo/live-episodes/ under the workspace root) for issue #956. NOTE: live-episode scoring is not yet wired into `fak dojo run` (which today scores Claude Code transcripts passed via --corpus), so this records the boundary but does not yet feed the scorer.")
	sf.native = fs.Bool("native", false, "NATIVE HARNESS (#1316/#1837): drive fak's OWN agent loop for every /v1/messages turn instead of the single-shot proxy turn. Both buffered and `stream: true` requests stay on the owned native path; streaming drives agent.RunArmStream and renders its text deltas plus typed lifecycle progress as Anthropic SSE and does not fall through to the proxy. If streaming cannot be safely emitted â a response writer that cannot flush, a planner that does not support streaming, or an armed answer stop-gate (a rejected answer must never leak as a delta) â the request degrades to the buffered native handler — the same owned loop, one response instead of deltas. The in-kernel syscall boundary remains the sole tool path, and ArmMetrics ride on the response `fak.native_arm` extension. Off by default (the proxy path is byte-for-byte unchanged).")
	sf.nativeMaxTurns = fs.Int("native-max-turns", gateway.DefaultNativeMaxTurns, "with --native: cap the owned loop's model round-trips per served request (<=0 uses the built-in default)")
	sf.nativeAdmissionTokenBudget = fs.Int("native-admission-token-budget", gateway.DefaultAdmissionPolicy().TokenBudget, "with a fak-native in-kernel model: cap the total token footprint admitted by the request scheduler; must be positive (default 8192)")
	sf.nativeCodeWorkspace = fs.String("native-code-workspace", "", "override the workspace root for default-on kernel Read/Write/Edit/Bash/Grep/Glob (requires --native)")
	sf.nativeCodeTools = fs.Bool("native-code-tools", true, "with --native, arm bounded kernel Read/Write/Edit/Bash/Grep/Glob in the current workspace; use --native-code-tools=false to disable")
	sf.nativeSpeculate = fs.Bool("native-speculate", false, "enable effect-free coding speculation (requires --native-code-workspace)")
	sf.vdsoProxyFill = fs.Bool("vdso-proxy-fill", false, "warm the vDSO from ADMITTED inbound tool_result blocks on the proxy path: an allowed, read-only-shaped result the client sends back fills (tool,args)->result so a LATER identical read is served inline (no client re-execution). Off by default — sound only when the principal is named and writes that touch the same resource reach fak (a proxy-closed world), so it is an explicit operator opt-in. Scoped per-principal; never fills a Shareable or write-shaped tool.")
	sf.metricsSnapshot = fs.Duration("metrics-snapshot", 0, "periodically append an interim gateway-usage counter snapshot (internal/gatewayusageledger, .fak/nightrun/gateway-usage.jsonl) while this long-lived `fak serve` is up, so a crash before a clean exit still leaves a trail (#1610). 0 (default) disables periodic snapshots; the exit-time snapshot is always written regardless of this flag.")
	sf.fleetBus = fs.Bool("fleet-bus", false, "JOIN THE FLEET CONTROL BUS (#5600, epic #5599): announce this serve as a control-plane instance on the shared bus and drain directives from it every --fleet-bus-interval, so one `fak fleet control send` reaches every live instance at once — the cross-PROCESS fan-out the per-gateway `sessionctl` broadcast and the display-only `fleetspine` each stop short of. Each drained directive is applied through the SAME writes the single-session verbs ride (steer ⇒ the a2achan operator turn, needs --native; pause/resume/cancel/terminate/throttle ⇒ session.Table.Transition) and every one draws an ACK carrying what this process OBSERVED change — the return path that lets the control point say \"3 of 4 applied, 1 refused STEER_NO_OWNED_LOOP\" instead of \"sent\". Exactly-once under at-least-once redelivery is keyed by the instance identity: the default fixed HTTP listen address survives a restart and remains distinct from simultaneous serves on other addresses; --stdio, --addr :0, or an unusable address fall back to a process-local identity and therefore do not promise restart dedup. Off by default (arming a control plane must be stated); the bus directory is --fleet-bus-dir.")
	sf.fleetBusDir = fs.String("fleet-bus-dir", "", "with --fleet-bus: the shared bus directory (default: FAK_FLEET_BUS, else <FLEET_STATE_DIR>/bus, else beside the fleet registry). On one machine a directory IS a real cross-process control plane; it is an honest cross-HOST one only where the directory itself is shared (a UNC path, an SMB/NFS mount) — which is what FLEET_STATE_DIR already exists to point at.")
	sf.fleetBusID = fs.String("fleet-bus-id", "", "with --fleet-bus: explicit stable bus identity override. Empty derives a restart-stable id from this machine plus the fixed HTTP listen address; --stdio, --addr :0, or an unusable address use a process-local fallback. A configured name is preserved (after bus-token sanitization) and must be unique among simultaneously live serves unless deliberately sharing one apply claim.")
	sf.fleetBusInterval = fs.Duration("fleet-bus-interval", DefaultFleetBusInterval, "with --fleet-bus: how often this instance re-announces presence and drains pending directives. Must stay well under fleetbus.DefaultInstanceTTL (90s) or a live instance flickers out of the roster and silently shrinks the denominator a control point measures \"everyone acked\" against. <=0 uses the default.")
	sf.keepAwake = fs.String("keep-awake", KeepAwakeOff, "prevent OS sleep during execution: off|while-active|always (default off)")
	return fs, sf
}

func (sf *serveFlags) effectiveAdmissionTokenBudget() int {
	if sf == nil {
		return 0
	}
	if sf.contextBudgetTokens != nil && *sf.contextBudgetTokens > 0 {
		return *sf.contextBudgetTokens
	}
	if sf.nativeAdmissionTokenBudget != nil && *sf.nativeAdmissionTokenBudget > 0 {
		return *sf.nativeAdmissionTokenBudget
	}
	return 0
}

func effectiveServeConfigWithQwen38Runtime(sf *serveFlags, m deploymanifest.Manifest, hasManifest bool, explicit map[string]bool) effectiveServeConfigReport {
	report := effectiveServeConfig(sf, m, hasManifest, explicit)
	source := "built-in"
	if explicit["qwen38-runtime"] {
		source = "flag"
	}
	report.Values["qwen38_runtime"] = effectiveConfigValue{Value: *sf.qwen38Runtime, Source: source}
	report.Values["qwen38_runtime_identity"] = effectiveConfigValue{Value: qwen38RuntimeIdentity(*sf.qwen38Runtime), Source: source}
	return report
}

func cmdServe(argv []string) {
	// t0 anchors the boot timeline exposed as fak_gateway_time_to_ready_seconds; it
	// must be the FIRST statement so flag parse + policy + weight load are accounted.
	t0 := time.Now()
	fs, sf := newServeFlagSet()
	if serveHelpRequested(fs, argv) {
		os.Exit(0)
	}
	configPath, err := serveConfigPath(argv)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: %v\n", err)
		os.Exit(2)
	}
	var manifest deploymanifest.Manifest
	manifestPresent := false
	if configPath != "" {
		manifest, err = deploymanifest.Load(configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: config %s: %v\n", configPath, err)
			os.Exit(2)
		}
		applyServeManifestDefaults(sf, manifest)
		manifestPresent = true
	}
	tParse := time.Now()
	_ = fs.Parse(argv)
	parseDur := time.Since(tParse)
	durability := resolveServeSessionState(*sf.sessionStatePath, os.Getenv)
	*sf.sessionStatePath = durability.Path

	explicit := explicitFlagNames(fs)
	toolPlugins, toolPreferences, err := compileToolPluginConfig(manifest)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: config %s: %v\n", configPath, err)
		os.Exit(2)
	}
	qwen38Runtime, err := normalizeQwen38Runtime(*sf.qwen38Runtime)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: %v\n", err)
		os.Exit(2)
	}
	*sf.qwen38Runtime = qwen38Runtime
	keepAwakeMode, err := validateKeepAwake(*sf.keepAwake)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: %v\n", err)
		os.Exit(2)
	}
	*sf.keepAwake = keepAwakeMode
	keepAwakeReleaser, err := acquireProcessKeepAwake(*sf.keepAwake, "fak serve (always)")
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: keep-awake: %v\n", err)
	}
	if keepAwakeReleaser != nil {
		defer keepAwakeReleaser.Release()
	}
	if err := validateNativeQwenQ4KPrefillChunk(*sf.nativeQwenQ4KPrefillChunk); err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: %v\n", err)
		os.Exit(2)
	}
	if _, err := serveNativeAdmissionPolicy(sf); err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: %v\n", err)
		os.Exit(2)
	}
	if *sf.printEffectiveConfig {
		if err := json.NewEncoder(os.Stdout).Encode(effectiveServeConfigWithQwen38Runtime(sf, manifest, manifestPresent, explicit)); err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: print effective config: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if manifestPresent {
		if err := validateServeManifestOpinions(manifest); err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: config %s: %v\n", configPath, err)
			os.Exit(2)
		}
	}

	rt := &serveRuntime{t0: t0, toolPlugins: toolPlugins, toolPreferences: toolPreferences, startupPhases: []gateway.StartupPhase{
		{Name: "flag-parse", Dur: parseDur},
	}}
	rt.resolveServeModelSources(sf)
	// --policy-check: validate the manifest and exit, binding no listener.
	if *sf.policyCheck {
		runServePolicyCheck(*sf.policyPath)
		return
	}

	// --plan-json (#4361): emit the versioned header-derived memory sizing artifact
	// and exit before load — the colibri-inspired pre-load inspection dry-run,
	// mirroring the --policy-check early-exit. Nothing allocates, no listener binds.
	if *sf.sizingJSON {
		runServeSizingJSON(sf)
		return
	}

	// Advisory (#3094): a serve launched from a non-fak cwd silently indexes whatever
	// tree it was dropped into (dojo corpus, devindex, session state all resolve against
	// the workspace root — cwd by default). That mis-binding is how a `/goal` run in a
	// SIBLING repo left the fak substrate pointed at the wrong tree and contributing no
	// value. Emit a loud, one-line advisory when no dos.toml is found upward from cwd, but
	// fail OPEN — serve anyway (an operator may deliberately run outside a fak workspace).
	warnIfNotFakWorkspace(os.Stderr)
	warnIfDeferToolsDisabled(os.Stderr, *sf.deferTools)

	// Bind-safety default (#5373, a child of #3279): refuse to boot a listener that is
	// reachable from off this host while no inbound token door is configured. Placed
	// HERE — after the --policy-check/--plan-json dry-run exits, before the capability
	// floor and every expensive stage — so an operator learns the address is refused in
	// milliseconds rather than after a multi-minute weight load. It reads only parsed
	// flags and binds nothing (serve_bind_safety.go).
	if !admitServeBind(sf, os.Stderr) {
		os.Exit(2)
	}

	// Prompt-shrink lever WIRE admission (#5493): --compact-history-budget,
	// --elide-stale-reads and --defer-cold-tools are each gated, inside the gateway, on
	// the Anthropic passthrough, so on any other upstream wire all three stand down to
	// identity. Refuse by name when the operator EXPLICITLY enabled one here, and name
	// the default-on ones that are merely inert, so "enabled but inert" is never silent
	// and a ~0-saving A/B on this wire cannot be read as a verdict on the kernel. Placed
	// beside the bind rule for the same reason: it reads parsed flags only and binds
	// nothing, so it answers in milliseconds rather than after a weight load
	// (shrink_lever_wire.go).
	if !admitServeShrinkLevers(fs, sf, os.Stderr) {
		os.Exit(2)
	}

	resolvedCodeWorkspace, err := resolveNativeCodeWorkspace(*sf.native, *sf.nativeCodeTools, *sf.nativeCodeWorkspace)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: %v\n", err)
		os.Exit(1)
	}
	*sf.nativeCodeWorkspace = resolvedCodeWorkspace

	// Install the capability floor fail-loud: a bad manifest aborts startup rather
	// than silently falling back to a more permissive default. Time it as the first
	// startup phase.
	tPolicy := time.Now()
	applyFloorWithProfile(*sf.policyPath, *sf.profile)
	rt.startupPhases = append(rt.startupPhases, gateway.StartupPhase{Name: "policy-load", Dur: time.Since(tPolicy)})
	configureServeToolEngines()

	// Start the measured child only after every validation step that can terminate
	// startup, and immediately before compute selection consumes the model source.
	if err := rt.maybeStartQwen38Delegation(sf); err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: %v\n", err)
		os.Exit(2)
	}
	defer rt.stopQwen38Delegation()

	// Boot stages (serve_stages.go). The order is load-bearing: compute before the
	// weight load, the session plane before the gateway, the observer seams resolved
	// before the gateway exists but installed only after it does.
	rt.resolveCompute(sf)
	defer rt.closeEPGroup()
	if isRemoteKVConfigured(sf) {
		if receipt, err := checkServeRemoteKV(context.Background(), sf, nil); err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: remote kv preflight: %v\n", err)
			os.Exit(2)
		} else {
			rt.addStartupMessage(serveRemoteKVStartupMessage(receipt))
		}
	}
	releaseMetalResidency, err := loadLocalLauncherModelWithMetalLease(rt.useMetal, *sf.ggufPath, gpulease.Options{}, func() {
		rt.loadModel(sf)
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	defer releaseMetalResidency()
	rt.configureEPDecode()
	rt.resolveSessionPlane(sf)
	rt.resolveObservers(sf)
	rt.buildGateway(sf)
	rt.wireGateway(sf)
	rt.addStartupMessage(serveDurabilityStartupMessage(durability))
	rt.run(sf)
}

// warnIfNotFakWorkspace emits a loud stderr advisory when the serve cwd is not inside a
// fak workspace (no dos.toml found walking upward). It never blocks startup — the caller
// serves regardless. This is the #3094 mis-binding guard: fak serve's dojo corpus,
// devindex, and session-state planes all resolve against the workspace root, which
// defaults to cwd; launched from a sibling repo they silently bind the wrong tree.
func warnIfNotFakWorkspace(stderr io.Writer) {
	if _, err := branchrole.FindRoot(""); err == nil {
		return // inside a fak workspace (dos.toml found) — nothing to warn about.
	}
	wd, err := os.Getwd()
	if err != nil {
		wd = "(unknown cwd)"
	}
	// A warning rather than a refusal, which is exactly why it needs the same shape
	// as a bail: nothing stops, so an operator who skims this line serves a
	// silently mis-bound tree for the rest of the session.
	writeConfigBail(stderr, configBail{
		Verb:    "fak serve",
		Reason:  bailNotAWorkspace,
		Summary: "WARNING — this is not a fak workspace; the dojo corpus, devindex, and session-state planes will bind THIS cwd, not a fak repo",
		Knobs: []bailKnob{
			bailCWD(wd, "no dos.toml found walking upward").want("a directory inside your fak checkout"),
		},
		// Nothing stops here, so the fix has to stay INLINE — a warning that makes
		// you run a second command to learn the remedy is one you scroll past.
		Check: "git rev-parse --show-toplevel   # the repo this cwd actually resolves to\n          launch fak serve from a fak checkout, or set the MCP server entry's \"cwd\" to your fak workspace root",
	})
}

const deferToolsDisabledWarning = "fak serve: WARNING: --defer-tools=false is set; advertising the full tool registry carries significant token overhead (~5,500+ tokens / ~22KB per turn) and increases per-turn latency. Consider keeping --defer-tools=true (the default) with progressive disclosure."

func warnIfDeferToolsDisabled(stderr io.Writer, deferTools bool) {
	if !deferTools {
		fmt.Fprintln(stderr, deferToolsDisabledWarning)
	}
}

// serveKeyPrincipals resolves the repeated `--key-principal PRINCIPAL=ENV_VAR` specs into
// the gateway.Config.KeyPrincipals map (#5332), reporting a refusal on stderr rather than
// returning it — the same (value, ok) shape resolveRequiredKey uses, so both auth flags
// fail closed identically. ok=false means the caller MUST NOT boot: a gateway serving with
// a keyset the operator believes is armed, but which never resolved, does not merely lose
// attribution — with no keyset matched, principalFor falls through to the CALLER-supplied
// X-Fak-Principal header, and that value is what the account allowlist adjudicates. So a
// half-resolved keyset is a tenant-isolation hole, not a cosmetic one.
//
// No specs is not a failure: it returns a nil map, which gateway.New turns into a nil
// keyset, leaving the --require-key-env single-bearer path byte-for-byte unchanged.
func serveKeyPrincipals(specs []string, lookupEnv func(string) string, stderr io.Writer) (map[string]string, bool) {
	keyPrincipals, err := gateway.ParseKeyPrincipals(specs, lookupEnv)
	if err != nil {
		// The summary keeps the original sentence intact, X-Fak-Principal clause and
		// all: it is what says WHY this fails closed rather than warning, and the
		// block is meant to add a next step, never to cost the reason.
		writeConfigBail(stderr, configBail{
			Verb:    "fak serve",
			Reason:  bailKeyPrincipalUnresolved,
			Summary: fmt.Sprintf("--key-principal %v — refusing to start a gateway whose tenant keyset did not fully resolve (an unresolved binding leaves that tenant attributed by the caller-supplied X-Fak-Principal header instead of by its key)", err),
			Knobs: []bailKnob{
				bailFlag("key-principal", strings.Join(specs, " ")).want("PRINCIPAL=ENV_VAR per tenant, each naming a set, distinct env var"),
			},
		})
		return nil, false
	}
	return keyPrincipals, true
}

func loadServeRouteFile[T any](flagName, path, want string, load func(string) (T, error)) *T {
	loaded, err := load(path)
	if err != nil {
		writeConfigBail(os.Stderr, configBail{
			Verb: "fak serve", Reason: bailRouteManifestInvalid,
			Summary: fmt.Sprintf("--%s did not load: %v", flagName, err),
			Knobs:   []bailKnob{bailFlag(flagName, path), bailFile(path, "did not load").want(want)},
			Bind:    []string{"path=" + path},
		})
		os.Exit(1)
	}
	return &loaded
}

// buildGateway loads the optional model-routing policy, constructs the gateway
// server from the resolved planes, and arms the admission controller for a pure
// in-kernel serve.
func (rt *serveRuntime) buildGateway(sf *serveFlags) {
	startupMessages := append([]gateway.StartupMessage(nil), rt.startupMessages...)
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

	srv, err := gateway.New(gateway.Config{
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
		InKernelModel:                rt.inKernelModel,
		Tokenizer:                    rt.inKernelTok,
		InKernelQ4K:                  rt.inKernelQ4K,
		InKernelPlanner:              serveNativePlannerConfig(sf),
		Backend:                      rt.chatBackend,
		CPUOffloadExperts:            *sf.cpuOffloadExperts,
		Metal:                        rt.useMetal,
		ExpertParallelRanks:          *sf.expertParallel,
		RequireKey:                   rt.requireKey,
		KeyPrincipals:                keyPrincipals,
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
		Native:              *sf.native,
		NativeMaxTurns:      *sf.nativeMaxTurns,
		NativeCodeWorkspace: *sf.nativeCodeWorkspace,
		NativeSpeculate:     *sf.nativeSpeculate,
		VDSOProxyFill:       *sf.vdsoProxyFill,
		// Streaming CONTENT-progress deadline (#5486, --stream-progress-timeout): the
		// window a warm-but-unadvancing proxied stream is given before the turn is ended.
		// gateway.newConfiguredHTTPPlanner carries it onto every proxy planner, where
		// agent.(*HTTPPlanner).streamProgressWindow band-checks it. The flag's 0 means OFF
		// and is translated here into that resolver's negative encoding; every other value
		// (including the 300s default) passes through untouched.
		StreamProgressTimeout: serveStreamProgressTimeout(*sf.streamProgressTimeout),
	})
	must(err)
	srv.AddStartupMessages(startupMessages...)
	srv.SetModelLoadProfile(rt.loadProfile)
	if rt.inKernelModel != nil && rt.inKernelTok != nil && strings.TrimSpace(*sf.baseURL) == "" && len(sf.replicaBaseURLs.Values()) == 0 {
		controller, message, err := newServeNativeAdmissionController(sf)
		must(err)
		srv.SetAdmissionController(controller)
		srv.AddStartupMessages(message)
	}
	// Control-plane SPEND CAP (#4859, the CLI half of #3273): --spend-cap builds the
	// governor, --spend-scope-trace the trace->ScopeKey resolver, and --budget-webhook is
	// wired as the breach sink. No --spend-cap ⇒ a nil governor and no SetSpendGovernor
	// call at all, so the served request path is byte-for-byte historical. A malformed or
	// unenforceable cap fails startup loud (serve_spend.go) — a budget that silently does
	// not bind is worse than a refused boot.
	gov, scopeOf, err := buildSpendGovernor(sf.spendCap.Values(), *sf.spendScopeTrace, *sf.budgetWebhook)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(1)
	}
	if gov != nil {
		srv.SetSpendGovernor(gov, scopeOf)
		srv.AddStartupMessages(gateway.StartupMessage{Source: "serve", Kind: "spend-cap", Level: "info", Text: fmt.Sprintf("spend cap armed on %d scope budget(s)", len(sf.spendCap.Values()))})
	}
	rt.srv = srv
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
		fmt.Fprintf(os.Stderr, "fak: cold resume (#629) — re-attached %d session(s) drive state from %s\n", n, path)
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
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "fak: create session-state parent for %s failed: %v\n", path, err)
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
	fmt.Fprintf(os.Stderr, "fak: persisted live session drive state → %s (#629)\n", path)
}
