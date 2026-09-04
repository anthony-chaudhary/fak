package main

// fak serve-wiring: the durable wiring-status surface for `fak serve`. It answers the
// one question a green `make ci` cannot: of every feature the gateway advertises, which
// ones are actually REACHED from the live serve entrypoint, and which are scaffolded-but-
// dead (a gateway.Config field that is set on the struct but whose flag the operator can
// never reach, or a field serve.go never sets at all).
//
// The verdicts come from an audited baseline (servewiringData below): each row was traced
// flag -> gateway.Config field -> the load-bearing runtime read, and adversarially verified.
// What this verb adds on top of a static doc is DRIFT DETECTION that cannot rot: it reads
// the real cmd/fak/serve.go and internal/gateway/gateway.go on each run and cross-checks two
// machine-derivable facts per row:
//
//   1. the gateway.Config field named in the row still EXISTS in the Config struct, and
//   2. serve.go actually SETS that field in its gateway.New(Config{...}) literal.
//
// A row whose field serve.go stops setting (the dead-wiring regression this verb exists to
// catch) flips to UNWIRED and reds `--check`. A Config field with no row at all is reported
// as UNAUDITED so a newly-added feature cannot slip in unexamined. This is the serve-path
// twin of the scorecard family: a deterministic, tree-cross-checked status the trunk keeps
// honest, regenerated on git via `fak serve-wiring --md > the table in docs/serve-config.md`.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

func cmdServeWiring(argv []string) { os.Exit(runServeWiring(os.Stdout, os.Stderr, argv)) }

// wiringVerdict is the closed set of wiring states a feature can be in.
type wiringVerdict string

const (
	// verdictWired: a load-bearing runtime read consumes the field on a request/turn path,
	// and the operator can reach it by default (no opt-in flag gate).
	verdictWired wiringVerdict = "WIRED"
	// verdictOffByDefault: fully wired, but inert until a non-default flag value arms it
	// (e.g. a budget > 0). A deliberate guard, not a defect.
	verdictOffByDefault wiringVerdict = "OFF_BY_DEFAULT_BUT_WIRED"
	// verdictPartial: the producer side is wired and reachable, but a documented consumer
	// step is deferred, so the end-to-end effect is incomplete.
	verdictPartial wiringVerdict = "PARTIAL"
	// verdictDead: the gateway reads the field, but serve.go never feeds it; the feature
	// is unreachable on the shipped binary. The defect this verb exists to catch.
	verdictDead wiringVerdict = "DEAD_WIRED"
)

// wiringRow is one audited feature: the operator flag, the gateway.Config field it feeds,
// the verdict, and the load-bearing call site that produced (or would produce) the effect.
type wiringRow struct {
	Feature  string
	Flag     string
	Field    string // the gateway.Config field name, or "" for a non-Config seam (e.g. an observer)
	Verdict  wiringVerdict
	CallSite string
	Note     string
}

// servewiringData is the audited baseline (workflow wiring-audit, every row skeptic-verified;
// routemanifest + the #761 notifier were DEAD_WIRED and wired in the same pass that added this
// verb). Update a row's Verdict/CallSite when the wiring changes; `--check` re-derives the
// machine-checkable half (field exists + serve.go sets it) so a stale row cannot hide a
// regression. A "" Field marks a seam wired through a session.Table observer, not Config;
// the serve.go-sets check is skipped for those (tracked by Flag presence instead).
var servewiringData = []wiringRow{
	{"reloadcanary", "--policy-canary-turns", "PolicyCanaryTurns", verdictOffByDefault, "internal/gateway/policy_canary.go:7", "after a reload, rolls back on the configured consecutive deny-all streak; zero disables the canary"},
	{"otlp", "--otlp-traces-endpoint", "OTLPEndpoint", verdictOffByDefault, "internal/gateway/gateway.go:2081", "enables bounded asynchronous OTLP/HTTP JSON trace export; empty disables it"},
	{"orgaudit", "(organization audit config)", "OrgAudit", verdictOffByDefault, "internal/gateway/gateway.go:2085", "enables enrolled privacy-screened adjudication receipts; zero config disables it"},
	{"trajctlmetrics", "(trajectory metrics observer)", "TrajctlMetrics", verdictOffByDefault, "internal/gateway/metrics.go", "projects bounded objective health onto /metrics when configured"},
	{"vcachecalibration", "(provider calibration)", "VCacheCalibration", verdictOffByDefault, "internal/gateway/gateway.go:2145", "applies fresh measured provider cacheability floors when available"},
	{"inkernelchat", "--gguf / --tokenizer", "InKernelModel", verdictWired, "internal/gateway/gateway.go:861", "with model+tokenizer and no --base-url, /v1/chat/completions and /v1/messages serve the in-kernel model"},
	{"inkernelplannerconfig", "--native-qwen-q4k-prefill-chunk-tokens / --native-qwen35-metal-gdn-sequence / --native-q4k-gateup-slab", "InKernelPlanner", verdictOffByDefault, "internal/gateway/gateway.go:newInKernelChatPlanner", "carries explicit native planner/session settings from the serve CLI into every in-kernel planner; defaults preserve prior behavior"},
	{"replica", "--replica-base-url", "ReplicaBaseURLs", verdictWired, "internal/gateway/gateway.go:715", "2+ endpoints -> ReplicaRouter round-robin"},
	{"vdso", "--vdso / --invalidation", "VDSO", verdictWired, "internal/kernel/kernel.go:348", "dedup fast path + tier-2 invalidation granularity"},
	{"vdsoproxyfill", "--vdso-proxy-fill", "VDSOProxyFill", verdictOffByDefault, "internal/gateway/gateway.go:1868", "warms the vDSO tier-2 cache from admitted inbound tool_result blocks; off by default"},
	{"toolfloor", "(adjudicator.Default.NeverAdmits)", "ToolFloorDenies", verdictWired, "internal/gateway/messages.go:392", "prunes provably-unreachable tool defs from the Anthropic passthrough; default-on, fail-safe"},
	{"expose", "--expose", "ExposeTools", verdictOffByDefault, "internal/gateway/mcp.go:exposedToolDescriptors", "allowlist of tool-name globs that narrows BOTH tools/list discovery AND tools/call invocation to the named tools (a hidden tool answers \"unknown tool\", no existence leak); a malformed or zero-match pattern fails startup loud; empty (default) exposes the full surface"},
	{"decidesession", "(host func, default-on)", "DecideSession", verdictWired, "internal/gateway/session_admit.go:57", "run-state refusal + TurnsLeft debit + budget + pace, before the model turn"},
	{"debitsession", "(host func, default-on)", "DebitSession", verdictWired, "internal/gateway/session_admit.go:157", "debits TokensLeft + context budget after the planner returns"},
	{"nativeserve", "--native", "Native", verdictOffByDefault, "internal/gateway/messages.go:153", "routes non-streaming /v1/messages through fak's owned agent.RunArm loop; off by default"},
	{"nativeserveturns", "--native-max-turns", "NativeMaxTurns", verdictOffByDefault, "internal/gateway/native_serve.go:33", "caps the owned native serve loop's model round-trips per request when --native is enabled"},
	{"routemanifest", "--route-manifest", "RouteManifest", verdictWired, "internal/gateway/gateway.go:1127", "binds ToolCall.Engine before Submit; flag wired (was DEAD_WIRED before this pass)"},
	{"toolplugins", "policy manifest tool_plugins", "ToolPlugins", verdictWired, "internal/gateway/gateway.go:2041", "compiled from the loaded policy manifest and installed into the monotone tool-extension host"},
	{"toolpreferences", "policy manifest tool_preferences", "ToolPreferences", verdictWired, "internal/gateway/gateway.go:2042", "compiled preference layers accompany tool plugins into the same extension host"},
	{"nativecodeworkspace", "--native-code-tools / --native-code-workspace", "NativeCodeWorkspace", verdictWired, "cmd/fak/serve_native_code_defaults.go:9", "with --native, arms the bounded kernel coding-tool catalog at the current workspace by default; boolean opt-out and root override remain explicit"},
	{"nativespeculate", "--native-speculate", "NativeSpeculate", verdictOffByDefault, "internal/gateway/native_serve.go:345", "enables effect-free coding speculation only when the native code catalog is armed"},
	{"routeaccounts", "--route-accounts", "RouteAccounts", verdictOffByDefault, "internal/gateway/gateway.go:resolveRoute", "binds the routed model id through the account roster to Target.EngineRoute() before Submit, so the residency PDP adjudicates the account-resolved route (#2528); off when no roster file is named"},
	{"ctxview", "--ctx-view-budget", "CtxViewBudget", verdictWired, "internal/gateway/gateway.go:788", "re-materializes history as an O(1) planned ctxplan view under the budget; DEFAULT-ON at 8000 resident tokens (fail-open, Anthropic cache prefix byte-identical), pass 0 to disable"},
	{"compacthistory", "--compact-history-budget", "CompactHistoryBudget", verdictWired, "internal/gateway/messages.go:compactAnthropicRawWithReason", "compacts old turns in the Anthropic outbound body once it sprawls past the budget, cache prefix byte-identical; DEFAULT-ON at ~48k (gateway.DefaultCompactHistoryBudget), pass 0 to disable"},
	{"compactanchorhead", "--compact-anchor-head", "CompactAnchorHead", verdictWired, "internal/gateway/messages.go:compactAnthropicRawWithReason", "re-anchors compacthistory's protected prefix on the stable system/tools head (agent.CompactAnchorHead) instead of the first cache_control breakpoint, the #1407 anchor-starved fix; DEFAULT-ON with every fire still gated on agent.CacheBurstPaysBack (#1408): fires when the live session's Budget.TurnsLeft horizon repays the one-time burst, or horizon-free when the trace OBSERVABLY idled past the message-breakpoint cache TTL (coldMessageSpanCache — the suffix re-bills cold that turn anyway); a warm un-budgeted session never bursts, pass =false to pin the first-breakpoint anchor"},
	{"assumesessionturns", "--assume-session-turns", "AssumeSessionTurns", verdictWired, "internal/gateway/messages.go:headSessionPrior", "the head-anchored burst gate's presumed session length when NO bounded Budget.TurnsLeft horizon is wired (the `fak guard -- claude` case): headSessionPrior maps the trace's served-turn depth to CurrentTurn and this value to TotalTurns, so a WARM un-budgeted long session fires the #1407 head re-anchor shed early (agent.CacheBurstPaysBack, #1408) and refuses near the presumed end; DEFAULT-ON at gateway.DefaultAssumedSessionTurns, a genuine wired Budget.TurnsLeft always wins, pass 0 for the byte-for-byte conservative no-horizon behavior. Inert unless --compact-anchor-head engages the head re-anchor"},
	{"elideresult", "--elide-result-bytes", "ElideResultBytes", verdictWired, "internal/gateway/messages.go:maybeElideAnthropicRaw", "shrinks an old oversized tool_result body to a bounded head+tail on BOTH wires — the Anthropic passthrough (req.Raw byte-splice, cache head byte-identical) and the decoded local-model path (req.Messages, for GLM-5.2/Qwen-3.6 served by fak); DEFAULT-ON at gateway.DefaultElideResultBytes (16KB), pass 0 to disable"},
	{"debugstats", "--debug-stats", "DebugStatsf", verdictOffByDefault, "internal/gateway/metrics.go:404", "emits one compact payload-free per-turn cache/compaction/resetScore line to stderr; off by default"},
	{"resetonbudget", "--reset-on-budget", "ResetOnBudget", verdictOffByDefault, "internal/gateway/session_admit.go:108", "distills a carryover seed and continues transparently on budget exhaustion; needs --context-budget-tokens"},
	{"budgetwebhook", "--budget-webhook", "", verdictOffByDefault, "internal/session/usage.go:73", "POSTs a pre-exhaustion warning + exhaustion event; wired via WatchBudget, off when URL empty"},
	{"notifier", "--notify-native / --notify-webhook / --notify-slack", "", verdictWired, "cmd/fak/serve.go (WatchTransitions)", "#761 stop-reason push notifier; native default-on (was DEAD_WIRED before this pass)"},
	{"enginecache", "--engine-cache-engine", "EngineCacheEngine", verdictOffByDefault, "internal/gateway/gateway.go:1480", "resets the serving-engine cache after a quarantined proxy turn; off when engine empty"},
	{"backend", "--backend", "Backend", verdictOffByDefault, "internal/agent/inkernel_planner.go:271", "decodes the in-kernel chat through the compute HAL device; off when name empty"},
	{"cpuoffloadexperts", "--cpu-offload-experts", "CPUOffloadExperts", verdictOffByDefault, "internal/agent/inkernel_planner.go:282", "with --gguf --backend, keeps MoE expert GEMMs on host RAM while dense/router/attention run on the device; off by default"},
	{"metal", "--metal", "Metal", verdictWired, "internal/agent/inkernel_planner.go:1067", "with --gguf (no --backend), auto-selects the Apple-Silicon metalgemm GPU when Apple-Silicon+cgo+a device are available; --metal/FAK_METAL=1 requires that path fail-loud; dense-Qwen Q8 only; CPU fallback on non-Metal builds or unavailable devices"},
	{"expertparallel", "--expert-parallel", "ExpertParallelRanks", verdictOffByDefault, "internal/gateway/gateway.go:817", "sets expert-parallel MoE ranks on the in-kernel model before planner construction; 0/1 leave the monolith path unchanged"},
	{"steersession", "(host func, default-on)", "SteerSession", verdictPartial, "internal/agent/loop_session.go:297 (drainSteer)", "POST /session/{id}/steer enqueues onto the a2achan Session bus; the native RunArm loop drains it at its turn boundary and folds it into the next turn as a user message (drainSteer, #850 — the consumer half #760 deferred). PARTIAL because only the native serve path owns that loop: the default proxy serve forwards a single upstream turn and owns none, so a steer to a proxy-served session is refused at ingress with the closed STEER_NO_OWNED_LOOP reason (409) rather than falsely acked as delivered (#3528)."},
	{"elidestale", "--elide-stale-reads", "ElideStaleReads", verdictWired, "internal/gateway/messages.go:865 (maybeElideStaleReads)", "the restorable sibling of --elide-result-bytes: on the Anthropic passthrough, replaces a Read tool_result superseded by a LATER in-session Edit/Write with a compact fak_context_restore marker (pre-edit body stashed behind a restore handle), same cache-safe working-set band, cache prefix byte-identical; DEFAULT-ON (gateway.DefaultElideStaleReads), pass =false to opt out"},
	{"positiveresidual", "--positive-residual-substitution", "PositiveResidualSubstitution", verdictWired, "internal/gateway/messages.go (positive residual substitution before provider dispatch)", "opt-in conservative replacement of compacted negated or stale history with a positive residual; original bytes remain restorable through ctxrestore; covered by internal/gateway/positive_residue_test.go"},
	{"vcacheanchor", "--vcache-anchor", "VCacheAnchor", verdictWired, "internal/gateway/messages.go:796", "M2 star-anchor pre-flight gate (#1493): on the Anthropic passthrough, applies cachemeta.RecommendLayout before send — hoists volatile system blocks behind a byte-stable cacheable anchor and splices a cache_control breakpoint onto the stable head so the first request warms provider prefix caching and siblings read it; DEFAULT-ON (gateway.DefaultVCacheAnchor), DECOUPLED from --compact-history-budget, fail-safe identity on any ambiguity, pass =false to opt out"},
	{"defercoldtools", "--defer-cold-tools", "DeferColdTools", verdictOffByDefault, "internal/gateway/messages_tooldefer.go:87", "the 10x floor lever (#3232, epic #3229): on the outbound Anthropic body marks every allowed-but-COLD custom tool defer_loading:true and injects one tool_search_tool, so the provider loads only the HOT core and faults a cold schema in on demand; deterministic + cache-safe, DEFAULT OFF (also FAK_DEFER_COLD_TOOLS=1), Anthropic passthrough only"},
	{"defertools", "--defer-tools", "DisableMCPDefer", verdictWired, "internal/gateway/mcp_defer.go:62", "defer cold MCP tools on tools/list; pass =false to expose the full registry immediately for clients without dynamic tool discovery like OpenCode"},
	{"toolceiling", "--tool-ceiling", "MCPToolCeiling", verdictWired, "internal/gateway/mcp_defer.go", "advertisement ceiling for exposed MCP tools when deferral is bypassed; clamps tool list to top-K curated tools to limit token overhead"},
	{"streamprogress", "--stream-progress-timeout", "StreamProgressTimeout", verdictWired, "internal/agent/stream_stall.go:streamProgressWindow (armed at internal/agent/stream.go:newStallReader and internal/agent/anthropic_stream.go)", "the streaming CONTENT-progress deadline (#5486): how long a proxied stream may stay WARM — keepalive/ping frames re-arming the inter-byte deadline — without one frame that advances the turn, after which the turn ends as a no-progress stall (504 upstream_stalled) instead of riding the 600s whole-request ceiling. newConfiguredHTTPPlanner carries the Config value verbatim onto every proxy planner (lone upstream AND each replica) and streamProgressWindow resolves it: DEFAULT-ON at agent.DefaultStreamProgressTimeout (300s) when unset, a positive value outside [5s, 600s] falls back to that default rather than being clamped, and a NEGATIVE value disables the deadline. `--stream-progress-timeout 0` is the operator's off switch (the house 0-is-off spelling, as with --ctx-view-budget/--elide-result-bytes) and cmd/fak/serve.go:serveStreamProgressTimeout translates that 0 into the negative encoding — the escape hatch for a provider whose prefill legitimately outlasts the window. Streaming proxy path only; inert on the buffered turn and on the offline mock planner"},
	{"keyprincipals", "--key-principal", "KeyPrincipals", verdictOffByDefault, "internal/gateway/http.go:359 (withAuth -> keyset.lookup)", "the multi-tenant KEYSET (#5332): serve resolves each PRINCIPAL=ENV_VAR spec through gateway.ParseKeyPrincipals (env var NAMES only, never a secret at rest) and gateway.New hashes the keys to SHA-256 digests, so a matching inbound x-api-key / Bearer both AUTHENTICATES the caller and stamps its tenant principal via WithPrincipal — which is what makes principalFor AUTHORITATIVE-from-context instead of falling through to the caller-supplied X-Fak-Principal header, and therefore what makes the modelroute Account.Principals allowlist (Target.Admits) a real isolation boundary. OFF by default: no --key-principal leaves the map nil, newKeyset returns a nil keyset, and the --require-key-env single-bearer path is byte-for-byte unchanged. A malformed spec, an unset/empty env var, or two tenants sharing one key REFUSES to boot (serveKeyPrincipals -> exit 2)"},
}

func runServeWiring(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("serve-wiring", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asMD := fs.Bool("md", false, "emit the markdown table for docs/serve-config.md")
	check := fs.Bool("check", false, "CI gate: exit non-zero on wiring drift (a row's Config field is gone, or serve.go stopped setting it, or a Config field has no audited row)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak serve-wiring: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	serveSrc, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "serve.go"))
	if err != nil {
		fmt.Fprintf(stderr, "fak serve-wiring: read serve.go: %v\n", err)
		return 1
	}
	gwSrc, err := os.ReadFile(filepath.Join(root, "internal", "gateway", "config.go"))
	if err != nil {
		fmt.Fprintf(stderr, "fak serve-wiring: read config.go: %v\n", err)
		return 1
	}

	configFields := parseConfigFields(string(gwSrc))
	serveSets := serveConfigAssignments(string(serveSrc))

	var drift []string

	// Per-row drift: a Config-backed row whose field no longer exists, or that serve.go no
	// longer sets, has silently dead-wired; exactly the regression this verb guards.
	for _, r := range servewiringData {
		if r.Field == "" {
			continue // observer seam, not a Config field; tracked by flag presence, checked below
		}
		if !configFields[r.Field] {
			drift = append(drift, fmt.Sprintf("row %q names gateway.Config.%s, which no longer exists in the Config struct", r.Feature, r.Field))
			continue
		}
		if !serveSets[r.Field] && r.Verdict != verdictDead {
			drift = append(drift, fmt.Sprintf("row %q (%s) is %s but serve.go no longer sets Config.%s; it has dead-wired", r.Feature, r.Field, r.Verdict, r.Field))
		}
	}

	// Coverage drift: a Config field serve.go sets but no audited row covers is an unexamined
	// feature. Skip the plumbing fields that are not operator features.
	covered := map[string]bool{}
	for _, r := range servewiringData {
		if r.Field != "" {
			covered[r.Field] = true
		}
	}
	var unaudited []string
	for f := range serveSets {
		if !covered[f] && !plumbingField[f] {
			unaudited = append(unaudited, f)
		}
	}
	sort.Strings(unaudited)
	for _, f := range unaudited {
		drift = append(drift, fmt.Sprintf("gateway.Config.%s is set by serve.go but has no audited wiring row (add it to servewiringData and trace it)", f))
	}

	if *asMD {
		writeWiringMarkdown(stdout, unaudited)
	} else if !*check {
		writeWiringSummary(stdout, drift)
	}

	if *check {
		if len(drift) == 0 {
			fmt.Fprintln(stdout, "OK  serve wiring: all audited rows still fed by serve.go; no unaudited Config feature")
			return 0
		}
		fmt.Fprintf(stdout, "DRIFT  serve wiring: %d issue(s)\n", len(drift))
		for _, d := range drift {
			fmt.Fprintf(stdout, "  - %s\n", d)
		}
		return 1
	}
	return 0
}

// plumbingField names gateway.Config fields that are infrastructure, not operator-facing
// features, so they are not expected to carry a wiring row.
var plumbingField = map[string]bool{
	"EngineID": true, "Model": true, "BaseURL": true, "Provider": true, "APIKey": true,
	"PinUpstreamCredential": true, "EngineCacheBaseURL": true, "EngineCacheAdminKey": true,
	"EngineCacheIdleTimeout": true, "EngineCacheRequireExactSpan": true, "Tokenizer": true,
	"InKernelQ4K": true, "RequireKey": true, "Invalidation": true, "Version": true,
	"ReloadPolicy": true, "ResetTrace": true, "ObserveTrace": true, "ObserveSession": true,
	"ControlSession": true, "ListSessions": true, "OnBudgetExhausted": true,
	"DefaultTraceID": true, "Logf": true, "StartTime": true, "StartupPhases": true,
}

var configFieldRe = regexp.MustCompile(`(?m)^\t([A-Z][A-Za-z0-9]*)\s+[^/]`)

// parseConfigFields returns the set of field names declared in the gateway.Config struct.
func parseConfigFields(src string) map[string]bool {
	return scanFieldSet(src, "type Config struct {", "\n}", configFieldRe)
}

// scanFieldSet returns the set of capitalized field names that re matches inside the
// src region bounded by startMarker (the first occurrence) and the first endMarker
// after it (or end-of-string if absent). It is the shared scanner behind the
// Config-declaration and Config-literal field extractors.
func scanFieldSet(src, startMarker, endMarker string, re *regexp.Regexp) map[string]bool {
	out := map[string]bool{}
	start := strings.Index(src, startMarker)
	if start < 0 {
		return out
	}
	rest := src[start:]
	end := strings.Index(rest, endMarker)
	if end < 0 {
		end = len(rest)
	}
	for _, m := range re.FindAllStringSubmatch(rest[:end], -1) {
		out[m[1]] = true
	}
	return out
}

var assignRe = regexp.MustCompile(`(?m)^\s*([A-Z][A-Za-z0-9]*):\s+`)

// serveConfigAssignments returns the set of gateway.Config field names assigned inside the
// gateway.New(gateway.Config{...}) literal in serve.go.
func serveConfigAssignments(src string) map[string]bool {
	// The literal ends at the matching "})" that closes New(Config{...}.
	return scanFieldSet(src, "gateway.New(gateway.Config{", "\n\t})", assignRe)
}

func verdictGlyph(v wiringVerdict) string {
	switch v {
	case verdictWired:
		return "wired"
	case verdictOffByDefault:
		return "off-by-default (wired)"
	case verdictPartial:
		return "partial"
	case verdictDead:
		return "dead-wired"
	default:
		return string(v)
	}
}

func writeWiringMarkdown(w io.Writer, unaudited []string) {
	fmt.Fprintln(w, "| Feature | Status | Flag | gateway.Config field | Live call site | Note |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|")
	for _, r := range servewiringData {
		field := r.Field
		if field == "" {
			field = "_(observer seam)_"
		} else {
			field = "`" + field + "`"
		}
		fmt.Fprintf(w, "| `%s` | %s | `%s` | %s | `%s` | %s |\n",
			r.Feature, verdictGlyph(r.Verdict), r.Flag, field, r.CallSite, r.Note)
	}
	if len(unaudited) > 0 {
		fmt.Fprintf(w, "\n> WARNING: Unaudited Config feature(s) serve.go sets with no wiring row: %s\n", strings.Join(unaudited, ", "))
	}
}

func writeWiringSummary(w io.Writer, drift []string) {
	var wired, off, partial, dead int
	for _, r := range servewiringData {
		switch r.Verdict {
		case verdictWired:
			wired++
		case verdictOffByDefault:
			off++
		case verdictPartial:
			partial++
		case verdictDead:
			dead++
		}
	}
	fmt.Fprintf(w, "serve wiring: %d features: %d wired, %d off-by-default, %d partial, %d dead\n",
		len(servewiringData), wired, off, partial, dead)
	for _, r := range servewiringData {
		fmt.Fprintf(w, "  %-26s %-16s %s\n", r.Feature, r.Verdict, r.CallSite)
	}
	if len(drift) > 0 {
		fmt.Fprintf(w, "\nDRIFT (%d):\n", len(drift))
		for _, d := range drift {
			fmt.Fprintf(w, "  - %s\n", d)
		}
	}
}
