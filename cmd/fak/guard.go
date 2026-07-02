package main

import (
	"context"
	_ "embed"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/headroom"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

const guardResourceSampleInterval = 2 * time.Second

// guardDefaultPolicyJSON is the day-to-day capability floor `fak guard` enforces when
// the operator names no --policy. It is embedded in the binary so `fak guard` works
// from ANY directory (a repo or not, an installed binary with no source tree). It
// allows the standard coding-agent tool set and denies the genuine-danger classes:
// destructive removal, privilege escalation, disk wipe, fork bomb, RCE pipe, writes
// that escape the working tree, and writes into credential/SSH/secret paths. Print or
// fork it with `fak guard --dump-policy`.
//
// The allow-list also admits the host harness's ORCHESTRATION + deferred-tool-loading +
// read-only-MCP surface (Agent/Task*/SendMessage/Monitor/ScheduleWakeup/EnterPlanMode/
// AskUserQuestion/ToolSearch/Read*McpResource*). These are NOT capability grants: a
// subagent the floor lets the agent SPAWN makes its own tool calls back through this same
// gateway, so every real effect is re-adjudicated downstream — the floor is unchanged, the
// agent just keeps its task/subagent/plan plumbing. ToolSearch in particular is load-bearing
// on harnesses that defer tool schemas: deny it and the agent cannot even reach WebFetch /
// WebSearch / MCP tools, so a bare floor silently bricks the wrapped agent. This was the
// dominant friction a historical-session replay surfaced ("align_policy_with_real_tool_shapes"
// across every audited Claude Code session: the floor was DEFAULT_DENYing Task*/SendMessage/
// ToolSearch — the harness's own tools). The genuine-danger classes above are untouched.
//
// The allow-list also admits the broader ULTRACODE orchestration surface (Workflow,
// EnterWorktree/ExitWorktree, Cron*, PushNotification, RemoteTrigger, DesignSync) and the
// READ-ONLY DOS verbs (mcp__dos__dos_verify/arbitrate/recall/review/status/doctor/answer/
// check_reason/refuse_reasons/commit_audit/citation_resolve, acme_lane_hint). The
// work-spawners (Workflow, EnterWorktree, Cron*) carry the same re-adjudication property as
// Agent/Task*: the agents and future prompts they create make their own tool calls back
// through this gateway, so every real effect still crosses the floor. The DOS verbs are pure
// reads of git/journal state. This means a turn that advertises the full ultracode toolset is
// never left with those names as silent prune-candidates — and matches the operator posture
// that an ultracode/debug session is governed by re-adjudication of EFFECTS, not by withholding
// orchestration plumbing. The genuine-danger classes (destructive Bash args, self-modify globs)
// are still untouched, so widening the orchestration surface never widens the danger floor.
//
//go:embed guard-default-policy.json
var guardDefaultPolicyJSON []byte

// cmdGuard — run any agent command with the kernel adjudicating every tool call it
// proposes. This is the one-command, cross-platform, productized form of the dogfood
// path: it starts the SAME gateway `fak serve` runs (in-process, on a private loopback
// port), points the child agent's provider base URL at it via a child-ONLY env var
// (never the parent shell, never a config file), execs the agent interactively, and on
// exit prints what the kernel allowed vs blocked before tearing the gateway down.
//
// The default upstream is the real Anthropic API in passthrough mode, so
// `fak guard -- claude` wraps your normal Claude Code: your credential and prompt-cache
// breakpoints flow through untouched (the gateway forwards the request bytes verbatim),
// but every tool call Claude proposes crosses the capability floor first. For
// --provider anthropic, fak uses your Claude Pro/Max SUBSCRIPTION by DEFAULT — it
// sources the OAuth token and sends it upstream as a bearer token. This holds even when
// ANTHROPIC_API_KEY is exported (a global SDK key no longer silently switches you to API
// billing); pass --api-key-env ANTHROPIC_API_KEY to opt INTO API billing, or
// --anthropic-oauth to force the subscription path and fail loud if no token is found.
//
// It also turns the durable DECISION JOURNAL on by default: every verdict the kernel
// reaches this session is appended to a tamper-evident, hash-chained file you can
// later replay with `fak audit verify`. fak is the disinterested referee, and the
// journal is the verifiable record of what it allowed vs blocked — a self-report is
// not a witness. Point it with --audit PATH, or turn it off with --no-audit.
// compressActivates reports whether the --compress flag should turn the native
// context-compressor on for this guard process. The flag only fills an UNSET
// FAK_COMPRESSOR, so an explicit env value (including `noop` to opt out) always
// wins — the same flag-defers-to-explicit-env rule as --landlock-hooks.
func compressActivates(flag bool, env string) bool {
	return flag && strings.TrimSpace(env) == ""
}

func cmdGuard(argv []string) {
	t0 := time.Now()
	fs := flag.NewFlagSet("guard", flag.ExitOnError)
	addr := fs.String("addr", "", "gateway listen address (default: a private 127.0.0.1 port the OS picks)")
	provider := fs.String("provider", "", "upstream wire the gateway proxies to: anthropic|openai|gemini|xai (default: auto-detected from the agent name — claude->anthropic, codex/opencode->openai — else anthropic)")
	baseURL := fs.String("base-url", "", "upstream provider base URL (default: the provider's public API, e.g. anthropic -> https://api.anthropic.com)")
	remoteServe := fs.String("remote-serve", "", "point the guarded turn's INFERENCE at a remote `fak serve` running on a lab box you chose (HOST or HOST:PORT, default port 8080). Forces the OpenAI-compatible wire and upstream base http://HOST:PORT/v1 (the /v1 fak serve serves its chat route under), so the dev turn runs on the lab GPU while the kernel still adjudicates locally. Mutually exclusive with --base-url; preflights GET /healthz AND /v1/models and fails loud if the box is not serving the /v1 surface.")
	model := fs.String("model", "", "upstream model id override (default: forward the client's own model id)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the UPSTREAM API key. For --provider anthropic this is the explicit opt-IN to API billing (e.g. --api-key-env ANTHROPIC_API_KEY); the default is your Claude Pro/Max subscription via OAuth, even when ANTHROPIC_API_KEY is exported. For other providers the default forwards the client's own key (passthrough).")
	anthropicOAuth := fs.Bool("anthropic-oauth", false, "force the Claude Pro/Max SUBSCRIPTION OAuth token upstream (sourced, in precedence order, from CLAUDE_CODE_OAUTH_TOKEN, then <claude-config>/.credentials.json, then <claude-config>/.oauth-token) sent as Authorization: Bearer + the oauth beta. This is ALREADY the default for --provider anthropic (even when ANTHROPIC_API_KEY is set); the flag forces it and fails loud if no token is found.")
	oauthTokenEnv := fs.String("oauth-token-env", "CLAUDE_CODE_OAUTH_TOKEN", "env var to read the subscription OAuth token from first")
	policyPath := fs.String("policy", "", "capability-floor manifest to enforce (default: the built-in guard floor; see --dump-policy)")
	envName := fs.String("env", "", "env var to inject the gateway URL into the child (default: chosen by --provider)")
	requireKeyEnv := fs.String("require-key-env", "", "require this env var's bearer token on the gateway (loopback rarely needs it)")
	logPath := fs.String("log", "", "write the gateway's per-request + per-verdict structured logs to this file (or '-' for stderr); default off to keep the agent's terminal clean")
	auditPath := fs.String("audit", "", "write the durable, hash-chained DECISION JOURNAL to this file (default: .dispatch-runs/guard-audit/interactive-<pid>-<hash>.jsonl; pass 'off' to disable). Every kernel verdict this session is appended as a tamper-evident JSONL row you can later replay with `fak audit verify`.")
	noAudit := fs.Bool("no-audit", false, "disable the durable decision journal for this session (it is ON by default — fak guard is the referee, and the journal is the verifiable record of what it allowed vs blocked)")
	dumpPolicy := fs.Bool("dump-policy", false, "print the built-in guard capability floor (an editable manifest) and exit")
	quiet := fs.Bool("quiet", false, "suppress the startup banner and the exit audit summary")
	bannerFlag := fs.String("banner", guardBannerAuto, "how much of the startup report to print before handing the terminal to the agent: auto|full|compact|off. AUTO (default): the compact 3-line banner for an attended interactive launch (the full report is a wall of text the agent's full-screen UI paints over seconds later), and the FULL report for headless/piped/scripted launches (a captured log wants the detail; byte-for-byte the pre-flag output). The full report is always recorded on the in-process gateway regardless — read it any time during the session with `fak info --startup` (it is the startup_report field of /debug/vars). --quiet still silences everything.")
	resourceStats := fs.Bool("resource-stats", true, "ON by default — track the HARNESS's own hardware-resource use this session (CPU, memory/RSS, disk-I/O) for BOTH halves: the kernel (this guard process + the in-process gateway, sampled continuously) and the agent (the wrapped child, folded from its exit state). Reported as one line in the exit summary and appended as a durable row to docs/nightrun/harness-resources.jsonl. Pass --resource-stats=false to disable (epic #2044).")
	debugStats := fs.Bool("debug-stats", true, "ON by default — the observable debug layer: print ONE compact, payload-free line per served turn to stderr with the turn's cache + token-value economy (request_tokens/cache_read/cache_creation, cache_hit, cache_rebate_tokens), the SAFETY half (blocked:/repaired:/quarantined: with the dominant reason whenever the kernel refused, rewrote, or paged out a call THIS turn — so a refused rm -rf or a quarantined secret is visible the moment it happens, not only in the exit summary), the compaction action, and the resetScore SHADOW health (healthy_cache|cache_decay|stale_prefix|cooldown|unknown_provider). These counts are the provider's own usage numbers, so it works natively over your Claude subscription OAuth. Independent of --log; pass --debug-stats=false or --quiet to silence it (#793).")
	preCompactHook := fs.String("precompact-hook", guardPreCompactModeShadow, "Claude Code PreCompact hook actuator for auto-compaction: off|shadow|enforce. shadow logs would-block/would-allow while exiting 0; enforce returns the compactcohere posture exit code.")
	denyAllContinue := fs.String("deny-all-continue", guardPreCompactModeEnforce, "Claude Code Stop hook that auto-RESUMES the agent after a deny-all turn (the floor refused EVERY proposed tool call, which the wire reports as end_turn — a stop the agent did not choose): off|shadow|enforce. ENFORCE by default (the false-stop fix), bounded by --deny-all-max consecutive continues; shadow logs the would-continue while letting the turn end; off disables. Claude children only.")
	denyAllMax := fs.Int("deny-all-max", guardStopHookDefaultMax, "with --deny-all-continue=enforce: the hard give-up — the maximum number of CONSECUTIVE deny-all turns to auto-continue past (with escalating guidance) before letting the turn end, so a model that keeps re-proposing refused calls cannot loop forever. The give-up is LOGGED so it is not a silent false-stop.")
	denyAllWarn := fs.Int("deny-all-warn", guardStopHookDefaultWarn, "with --deny-all-continue=enforce: at this many CONSECUTIVE deny-all turns the auto-continue guidance escalates from a gentle nudge to a relevance-decision WARNING (asks the agent whether the remaining work is reachable under the floor, and to declare BLOCKED and stop cleanly if not). Clamped to <= --deny-all-final <= --deny-all-max.")
	toolprocHooks := fs.String("toolproc-hooks", guardToolprocModeObserve, "Claude Code tool-process observation hooks (off|observe, observe by default): PreToolUse/PostToolUse/SessionEnd append spawn/exit/session_end rows to the workspace toolproc journal (fail-open; `fak toolproc ps --events .fak/toolproc/journal.jsonl` reads the live table). Claude children only.")
	denyAllFinal := fs.Int("deny-all-final", guardStopHookDefaultFinal, "with --deny-all-continue=enforce: at this many CONSECUTIVE deny-all turns the guidance escalates to a FINAL warning, the last attempts before the hard give-up at --deny-all-max.")
	taskHandoffMode := fs.String("task-handoff", guardPreCompactModeEnforce, "Claude Code Stop hook completion handoff gate: off|shadow|enforce. ENFORCE by default: on a clean stop, require a valid fak.task-handoff.v1 JSON with witnessed done + current state + 1-2 next steps or no-next-step reason. The path is exposed as FAK_TASK_HANDOFF_FILE.")
	taskHandoffFile := fs.String("task-handoff-file", "", "path the wrapped agent must write with fak.task-handoff.v1 before a clean stop (default: a private temp file for this guard session)")
	taskHandoffRepo := fs.String("task-handoff-repo", "", "owner/repo for optional live handoff issue sync (passed to fak task handoff --live)")
	taskHandoffLive := fs.Bool("task-handoff-live", false, "after a valid handoff with next_steps, the Stop hook runs fak task handoff --live before allowing the clean stop")
	splitMode := fs.String("split", "auto", "the default-launch UI: open a 20% `fak info` pane BESIDE the 80% interactive agent pane so the live cache/token economy + the kernel floor's safety counters stay on screen (a bare `fak guard -- claude` hands the whole terminal to Claude, hiding fak). auto|on|off. AUTO (default): enable ONLY for an attended interactive launch inside a terminal multiplexer (tmux, or Windows Terminal via $WT_SESSION); no-op for headless/piped/CI/plain-terminal launches (zero behavior change there). on forces it (prints a recipe if no multiplexer is found); off disables. The pane polls THIS guard's own loopback gateway (auth-exempt on loopback); the bearer is never placed on a pane command line.")
	splitWhere := fs.String("split-where", "bottom", "with --split: place the 20% fak-info pane as a \"bottom\" strip or a \"right\" column")
	splitInterval := fs.Duration("split-interval", 2*time.Second, "with --split: refresh interval for the fak-info pane")
	splitDryRun := fs.Bool("split-dry-run", false, "preview the --split 80/20 plan (resolved multiplexer, geometry, and the exact `fak info` pane command) and EXIT, without bringing up the gateway, spawning a pane, or launching the agent. Use it to see what --split will do before handing the terminal to the agent.")
	ctxViewBudget := fs.Int("ctx-view-budget", 8000, "wire the ctxplan context PLANNER into the live guard loop: each buffered turn, re-materialize the forwarded history as an O(1) planned VIEW under this resident-token budget (a planned view in place of appending the whole transcript, #555). DEFAULT-ON at a conservative 8000 resident tokens; pass 0 to disable (leaves the existing path byte-for-byte unchanged). The planner only ever SHORTENS and falls open to the full history on any doubt; on the Anthropic passthrough it keeps the cached prefix byte-identical (witness: docs/notes/CTXVIEW-DEFAULT-ON-WITNESS-2026-06-28.md). The streaming fast-path bypasses this; the buffered turn path is what gets planned.")
	compactHistoryBudget := fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "compact OLD conversation turns in the OUTBOUND Anthropic request body down to this resident-token budget while keeping the cache_control prefix BYTE-IDENTICAL, so the upstream prompt-cache hit survives. This reaches the flagship `fak guard -- claude` passthrough (where the body is forwarded verbatim, #555). DEFAULT-ON: once a wrapped conversation sprawls past ~48k resident tokens the cut fires and sheds the un-cacheable middle the provider re-bills every turn; a typical short session stays untouched. Pass 0 to disable (body forwarded byte-for-byte). Anthropic passthrough only.")
	compactAnchorHead := fs.Bool("compact-anchor-head", false, "re-anchor --compact-history-budget's protected prefix on the stable system/tools head instead of the default first-breakpoint anchor, fixing the anchor-starved trap (#1407) where real Claude Code traffic's recent cache_control breakpoint protects almost the whole conversation so the budget can never shed anything (see the 'anchor-starved' diagnostic). OPT-IN, not default-on: re-anchoring bursts the recent breakpoint's cached suffix once, so it only fires when the burst repays (CacheBurstPaysBack, #1408) — without a wired session-turn horizon it only fires zero-penalty bursts.")
	elideResultBytes := fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes, "ON by default at gateway.DefaultElideResultBytes (the reviewed gateway.DocumentedElideResultBytes threshold): shrink oversized tool_result bodies outside the active working set to a bounded head+tail form once they exceed this byte threshold. 0 disables.")
	sessionID := fs.String("session-id", "", "default trace/session id for wrapped agents that omit X-Trace-Id or MCP trace_id (default: derived from host, git HEAD, cwd, and wrapped argv)")
	contextBudgetTokens := fs.Int("context-budget-tokens", 0, "seed the guard session with this prompt/context-token budget; exhaustion returns a reset directive with continuation_id (0 = off)")
	maxDuration := fs.Duration("max-duration", 0, "govern this guard session to at most this much REAL WALL-CLOCK time (issue #1584), tracked independently of --context-budget-tokens and surviving a --restart-on-budget hidden restart (the elapsed total carries forward, it does not reset to zero). 0 = unbounded (still tracked for `fak session status`, just never stops the run). Query/inspect anytime with `fak session status <id>`; the time budget drains the session to Draining/Stopped with reason TIME_BUDGET_EXHAUSTED exactly like a token-budget exhaustion.")
	budgetEnvelopeSpec := fs.String("budget-envelope", "", "managed-context budget envelope (#1573): turns=20,tokens=200000,context=64000,wall=2h,spend=$25,throughput=40/s,max-tokens=1024,gap=250ms. Seeds this guard session's budget/pace/wall axes; explicit --context-budget-tokens and --max-duration override those envelope axes.")
	resetOnBudget := fs.Bool("reset-on-budget", false, "on context-budget exhaustion, re-arm the continuation trace with a carryover seed and continue transparently instead of returning 409 (requires --context-budget-tokens)")
	restartOnBudget := fs.Bool("restart-on-budget", false, "on context-budget exhaustion, stop and relaunch the wrapped child under the continuation trace, writing a carryover seed JSON and exposing it via FAK_RESET_* env vars (requires --context-budget-tokens)")
	restartLimit := fs.Int("restart-limit", 0, "maximum child relaunches for --restart-on-budget; 0 means unlimited")
	restartSeedDir := fs.String("restart-seed-dir", "", "directory for --restart-on-budget carryover seed JSON files (default: OS temp dir, one private directory per reset)")
	landlockHooks := fs.Bool("landlock-hooks", false, "LINUX-ONLY defense-in-depth: run the spawned agent under a Landlock profile that makes the git hook surface (.git/hooks + core.hooksPath) READ-ONLY while the rest of the tree stays writable, so a laundered write cannot drop an executable hook. OFF by default; fails OPEN (logs + spawns unrestricted) on a kernel without Landlock or on a non-Linux host. Also settable via "+guard.EnvOptIn+"=1.")
	dojoMode := fs.Bool("dojo", false, "enable live dojo mode: write a start-marker for this guard session, then persist a scored vcache live row at shutdown when provider-cache telemetry exists.")
	ggufPath := fs.String("gguf", "", "run a SMALL MODEL IN-KERNEL as the local upstream — no API key, no network, no second server. fak loads these GGUF weights into its OWN engine and serves them to the wrapped agent, so the whole `local model + your coding harness + kernel floor` stack is ONE command (`fak guard --gguf qwen2.5:7b -- claude`). Accepts a model alias (`fak ls`), an hf://owner/repo/file.gguf URI (downloaded on demand), or a local .gguf path. Every tool call the agent proposes is still adjudicated by the same capability floor and recorded in the same audit journal — only the inference moves onto YOUR box. Alone, the local model IS the upstream (mutually exclusive with --remote-serve); with --alongside or an explicit --base-url it serves ALONGSIDE the API upstream instead (see --alongside).")
	alongside := fs.Bool("alongside", false, "with --gguf: serve the small local model ALONGSIDE the API upstream instead of REPLACING it (the dual planner). The wrapped agent's normal turns proxy to the provider exactly as a plain `fak guard` session (same OAuth/passthrough, same prompt-cache preservation), while any request addressed to the --gguf model's alias — or the literal model id \"local\" — decodes in-kernel on your box with no upstream call and no tokens billed (e.g. point a cheap subagent tier at it). Implied by --gguf + an explicit --base-url.")
	localAuto := fs.Bool("local", false, "auto-detect a local OpenAI-compatible model server you are ALREADY running (Ollama, LM Studio, or a llama.cpp server) and wire guard's upstream to it with zero flags — `fak guard --local -- codex` becomes a governed local coding loop with no base-URL hunting. Probes, fail-soft (~300ms each), Ollama (127.0.0.1:11434, honors OLLAMA_HOST), then LM Studio (127.0.0.1:1234), then llama.cpp (127.0.0.1:8080); the first live one wins and a coding-tuned served model is preferred. If --gguf is ALSO passed it wins (that is the no-server in-kernel path); if nothing is detected and no --gguf, fak fails loud with how to start a server. Mutually exclusive with --base-url / --remote-serve.")
	gpuBackend := fs.String("backend", "", "with --gguf: compute backend for the in-kernel decode — empty = the CPU reference path; a registered device like 'cuda' runs prefill+decode through the GPU HAL (needs a -tags cuda build AND a reachable GPU). Fails loud if named but unavailable, so a typo never silently runs on CPU.")
	tokPath := fs.String("tokenizer", "", "with --gguf: OPTIONAL tokenizer override (a tokenizer.json or its directory); default uses the GGUF's EMBEDDED tokenizer. Pass this only for a checkpoint with no embedded BPE tokenizer or a custom vocab.")
	replayTrace := fs.String("replay-trace", "", "DON'T wrap a live agent — instead REPLAY a recorded trace fixture through the real guard end to end and watch the floor fire. Stands up the gateway against a built-in fake upstream that emits the fixture's tool_use + token-usage turns, posts each turn through the SAME adjudication path `fak guard -- claude` uses, and prints per-turn what was allowed vs denied (with the deny reason), the turn's token/cache economy, and the journal rows recorded — then the exit summary + the verify command. No API key, no GPU, no child process. Use it to understand exactly what the guard does to a trace that leads to token work, and to demo the floor. See internal/gateway/testdata/guard-trace-e2e.json for the fixture shape.")
	replayWire := fs.String("replay-wire", "anthropic", "with --replay-trace: the provider wire to replay over (anthropic = the `fak guard -- claude` flagship /v1/messages path; openai = the codex/opencode /v1/chat/completions path).")
	codexConfig := fs.Bool("codex-config", true, "when wrapping Codex, inject per-run -c model_provider/model_providers.fak overrides so Codex talks to the in-process gateway over the Responses wire. Codex-only; pass --codex-config=false if you already configured the fak provider yourself.")
	mcpRegister := fs.Bool("mcp-register", true, "register fak's own MCP self-query surface (fak_index_*, fak_memory_*, fak_tools_search) into the wrapped Claude Code child by default, via a session-scoped --mcp-config pointing at this gateway's /mcp endpoint. Claude-only; ADDS to any project/user MCP config the child already loads, never replaces it. Every call is still re-adjudicated by the guard floor — this widens discovery, not the danger floor. Pass --mcp-register=false if you already supply your own MCP config.")
	compress := fs.Bool("compress", false, "activate the native context-compressor for this session: shrink benign tool results (ANSI/control strip, CR-redraw collapse, duplicate-line fold, JSON minify) before they enter model context, only when the saving clears the worth-it floor and never on poison, with the original preserved (reversible). Equivalent to FAK_COMPRESSOR=native for this process; an explicit FAK_COMPRESSOR wins. See `fak headroom bench` for the savings and `fak headroom status` for the live decision breakdown.")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak guard [flags] -- <agent command...>")
		fmt.Fprintln(os.Stderr, "  e.g. fak guard -- claude")
		fmt.Fprintln(os.Stderr, "       fak guard --provider openai -- codex")
		fmt.Fprintln(os.Stderr, "       fak guard --policy my-floor.json -- claude")
		fs.PrintDefaults()
	}
	_ = fs.Parse(argv)
	// Boot-timeline instrumentation: mirror serve.go's StartupPhases (internal/gateway/startup.go)
	// so a slow `fak guard` launch is diagnosable from THIS session's own boot timeline instead of
	// only fak_gateway_startup_phase_duration_seconds on an ephemeral port that closes with the
	// session. Populated as each phase completes below; wired into gateway.Config near the bind.
	parseDur := time.Since(t0)
	var (
		localDetectDur     time.Duration
		remotePreflightDur time.Duration
		upstreamResolveDur time.Duration
		pathLookupDur      time.Duration
		tokenizerLoadDur   time.Duration
	)

	// Which flags did the operator set EXPLICITLY (vs leave at their default)? Used below so
	// an explicit --debug-stats can win over the interactive auto-suppress that keeps the
	// per-turn economy line out of an attended agent's full-screen UI.
	guardSetFlags := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { guardSetFlags[f.Name] = true })
	guardTraceID := strings.TrimSpace(*sessionID)
	if guardTraceID == "" {
		guardTraceID = "guard"
	}
	var guardBudgetEnvelope session.BudgetEnvelope
	hasGuardBudgetEnvelope := strings.TrimSpace(*budgetEnvelopeSpec) != ""
	if hasGuardBudgetEnvelope {
		var err error
		guardBudgetEnvelope, err = session.ParseBudgetEnvelope(*budgetEnvelopeSpec)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak guard: --budget-envelope: %v\n", err)
			os.Exit(2)
		}
	}
	contextBudgetLimit := *contextBudgetTokens
	if hasGuardBudgetEnvelope && !guardSetFlags["context-budget-tokens"] && guardBudgetEnvelope.Budget.ContextTokensLeft > 0 {
		contextBudgetLimit = guardBudgetEnvelope.Budget.ContextTokensLeft
	}
	maxDurationLimit := *maxDuration
	if hasGuardBudgetEnvelope && !guardSetFlags["max-duration"] && guardBudgetEnvelope.WallClockLimit() > 0 {
		maxDurationLimit = guardBudgetEnvelope.WallClockLimit()
	}

	// --split-dry-run is a pure PREVIEW: render the resolved 80/20 split plan and exit BEFORE
	// any gateway bind, pane spawn, or agent launch. The live gateway URL is not known yet (the
	// OS picks the port at bind time), so the preview shows a placeholder loopback URL — the
	// resolved multiplexer, geometry, and `fak info` argv shape are what the operator is
	// previewing, and those do not depend on the port.
	if *splitDryRun {
		out, code := renderGuardInfoPaneDryRun(os.Getenv, *splitWhere, "http://127.0.0.1:<gateway-port>", *splitInterval)
		fmt.Fprint(os.Stdout, out)
		os.Exit(code)
	}

	// The --landlock-hooks flag and FAK_GUARD_LANDLOCK env are equivalent; normalize the
	// flag into the env so buildGuardChild (called from two paths) consults one source.
	if *landlockHooks {
		_ = os.Setenv(guard.EnvOptIn, "1")
	}

	// --compress activates the native context-compressor for THIS guard process: the
	// result-admit gate (already registered, but no-op while noop is selected) starts
	// shrinking benign tool results before they enter model context. The gate keeps
	// its own "when not" discipline — never compress poison, only past the worth-it
	// floor, original preserved in the CAS — so activation is safe and reversible. An
	// explicit FAK_COMPRESSOR (incl. =noop to opt out) always wins; the flag only
	// fills an unset default, mirroring the --landlock-hooks/env normalization above.
	if compressActivates(*compress, os.Getenv("FAK_COMPRESSOR")) {
		headroom.Select(headroom.NativeName)
	}

	// Expand a leading ~ in the --gguf / --tokenizer paths up front (PowerShell and most
	// quoting pass ~ through literally and Go never expands it), so `--gguf ~/models/x.gguf`
	// opens. The alias/URI resolution + download is deferred until AFTER the flag-conflict
	// check below, so a `--gguf foo --base-url bar` typo fails loud before any multi-GB pull.
	*ggufPath = pathutil.ExpandTilde(*ggufPath)
	*tokPath = pathutil.ExpandTilde(*tokPath)

	// Raise the gateway's HTTP write/planner timeout floors for the wrapped session. A
	// frontier Claude turn with extended thinking can run well past fak serve's 90 s
	// WriteTimeout / 60 s planner default, which would cut the stream off mid-turn and
	// surface to the worker as a "context canceled" upstream error. Guard binds its own
	// listener and calls Serve() directly, so it must set these BEFORE the server reads
	// them (gateway.Serve consults FAK_HTTP_WRITE_TIMEOUT_S via durEnv). An explicit
	// operator value always wins — guardEnsureTimeoutFloor never clobbers a set var.
	guardEnsureTimeoutFloor("FAK_HTTP_WRITE_TIMEOUT_S", guardTimeoutFloorS)
	guardEnsureTimeoutFloor("FAK_PLANNER_TIMEOUT_S", guardTimeoutFloorS)
	// Pin the streaming IDLE-read deadline too — but deliberately SMALL, the opposite of the
	// 600s write/planner floors above. Those are RAISED so a long but healthy turn is not cut
	// off mid-stream; the stall timeout must stay short so a SILENT upstream (a mid-stream API
	// stall) fails in ~a minute instead of hanging for the whole 600s write window. Reusing
	// guardTimeoutFloorS here would re-introduce exactly that hang. The agent default is
	// already 60s; this makes the value explicit in the wrapped child's env beside the other
	// two floors, and (like them) never clobbers an operator-set value.
	guardEnsureTimeoutFloor("FAK_STREAM_STALL_TIMEOUT_S", guardStallFloorS)

	if *dumpPolicy {
		os.Stdout.Write(guardDefaultPolicyJSON)
		return
	}

	// --replay-trace runs the guard end to end over a recorded fixture instead of
	// wrapping a live agent: it is the observable, no-API-key way to watch the floor
	// fire on a trace that leads to token work. It shares the SAME floor + gateway +
	// journal + summary as the live path (see guard_replay.go), so what it shows is what
	// a real session would do.
	if *replayTrace != "" {
		os.Exit(runGuardReplay(*replayTrace, *replayWire, *policyPath, os.Stdout))
		return
	}

	command := fs.Args() // everything after the flags (and after `--`) is the wrapped agent.
	if len(command) == 0 {
		fs.Usage()
		os.Exit(2)
	}

	// Decide whether the per-turn `fak-turn …` economy line streams to the SHARED terminal
	// stderr. On an attended interactive launch the wrapped agent (Claude Code) paints a
	// full-screen alternate-screen TUI over THIS terminal, so a per-turn stderr write lands
	// on top of it and corrupts the session view; there the economy belongs in the `fak info`
	// split pane (the dedicated fak section) + the exit summary, not the agent pane. An
	// explicit --debug-stats still streams here; headless/piped runs keep it (no TUI to
	// corrupt). See guardDebugStatsToSharedStderr.
	debugStatsStderr := guardDebugStatsToSharedStderr(
		*debugStats, *quiet, guardSetFlags["debug-stats"],
		cmdGuardStdinInteractive(), guardChildInteractive(command))

	// Startup-banner verbosity: resolve --banner now, fail-loud on a bad value before
	// any gateway binds. AUTO compacts only the attended interactive launch — the same
	// attended-vs-headless split as the debug-stats auto-suppress above — so headless/
	// piped launches keep the full startup report byte-for-byte. See guard_banner.go.
	bannerMode, bannerErr := guardBannerModeDecision(*bannerFlag, *quiet, cmdGuardStdinInteractive(), guardChildInteractive(command))
	if bannerErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: %v\n", bannerErr)
		os.Exit(2)
	}

	// Observability sink for the gateway's structured per-request + per-verdict logs
	// (event=gateway_http_request / event=gateway_operation, each carrying the trace_id).
	// Default OFF (a no-op) so the wrapped agent's terminal stays clean; --log FILE (or
	// '-' for stderr) turns the full stream back on. /metrics, /debug/vars, and the
	// FAK_AUDIT_JOURNAL durable audit trail are independent of this — see the banner.
	gwLogf, logCloser, logLabel := guardLogSink(*logPath, os.Stderr)
	if logCloser != nil {
		defer func() { _ = logCloser.Close() }()
	}

	// 1. Install the capability floor. An explicit --policy file wins; otherwise the
	//    embedded guard floor. With NO floor the kernel default-denies every tool and
	//    the wrapped agent can do nothing — so guard ALWAYS loads one, fail-loud.
	var (
		rt           policy.Runtime
		err          error
		floorSource  string
		policyBytes  []byte
		policyDigest string
	)
	tPolicy := time.Now()
	if *policyPath != "" {
		policyBytes, err = os.ReadFile(*policyPath)
		if err == nil {
			rt, err = policy.ParseRuntime(policyBytes)
			if err != nil {
				err = fmt.Errorf("policy %s: %w", *policyPath, err)
			}
		} else {
			err = fmt.Errorf("policy: %w", err)
		}
		floorSource = *policyPath
	} else {
		policyBytes = guardDefaultPolicyJSON
		rt, err = policy.ParseRuntime(guardDefaultPolicyJSON)
		floorSource = "built-in guard floor (--dump-policy to see it)"
	}
	must(err)
	policyDigest = guardPolicyDigest(policyBytes)
	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)
	policyDur := time.Since(tPolicy)

	// 1b. Default the durable DECISION JOURNAL on. fak guard is the disinterested
	//     referee; a tamper-evident, hash-chained record of every verdict is what
	//     lets an auditor confirm after the fact what the kernel allowed and blocked
	//     — the witness that makes the refereeing checkable rather than self-reported.
	//     So it is ON by default (announced in the banner, not silent). The kernel's
	//     EvDecide/EvDeny events on the proxy adjudication path are exactly what the
	//     journal records, so a guard session produces a populated ledger. Precedence:
	//     FAK_AUDIT_JOURNAL honored at boot wins; --no-audit / --audit off disables;
	//     --audit PATH or a repo-local per-session path otherwise. Enable BEFORE serving so
	//     the emitter is registered before the first decision crosses the floor.
	auditLabel, auditJournal := guardEnableAudit(*auditPath, *noAudit)
	var auditSeq0 uint64
	var refusalCarryForward []guardRefusalCarry
	if auditJournal != nil {
		auditSeq0, _, _ = auditJournal.Stats()
		refusalCarryForward = guardReadRefusalCarryForward(auditJournal.Path(), guardTraceID, guardFindReasonRoot())
	}

	// 2. --remote-serve sugar: run the guarded turn's INFERENCE on a lab box you chose.
	//    It is a one-name shorthand for the informal "treat a remote fak serve as a
	//    provider URL" chain (`--provider openai --base-url http://HOST:PORT/v1`): it
	//    forces the OpenAI-compatible wire and sets the base URL to the box, so the kernel
	//    still adjudicates locally while the model runs on the lab GPU. Resolve + validate
	//    it BEFORE binding anything so a typo or a down box fails loud, not mid-session.
	remoteBase, remoteErr := normalizeRemoteServe(*remoteServe)
	if remoteErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: --remote-serve: %v\n", remoteErr)
		os.Exit(2)
	}

	// --gguf turns the in-process gateway into a LOCAL in-kernel model server (fak runs
	// the model itself). Alone, the local model IS the upstream; with --alongside (or an
	// explicit --base-url) it serves ALONGSIDE the API upstream instead — the gateway's
	// dual planner routes requests addressed to the local model id in-kernel and proxies
	// everything else. Decide + validate up front, before binding or pulling weights.
	localModel, localAlongside, localConflict := guardLocalModelDecision(*ggufPath, *baseURL, remoteBase, *alongside)
	if localConflict != "" {
		fmt.Fprintln(os.Stderr, "fak guard:", localConflict)
		os.Exit(2)
	}
	// The alias the operator typed (before resolution rewrites *ggufPath to a file path)
	// is the model id a client asks for to reach the local side in alongside mode.
	localAlias := strings.TrimSpace(*ggufPath)

	// --local: auto-detect a running local OpenAI-compatible server (Ollama/LM Studio/
	// llama.cpp) and wire the upstream to it. This is a PROXY path (the server is external),
	// so on detection we set provider=openai + base-URL=<detected>/v1 exactly as if the user
	// had typed those flags, and the standard resolution flow below handles it. Precedence:
	//   - --gguf wins (it is the no-server in-kernel path); --local is then a no-op.
	//   - --base-url / --remote-serve conflict (the detected server IS the upstream).
	//   - nothing detected + no --gguf -> fail loud with how to start a server.
	if *localAuto && !localModel {
		if strings.TrimSpace(*baseURL) != "" || remoteBase != "" {
			fmt.Fprintln(os.Stderr, "fak guard: --local auto-detects the upstream server, so it is mutually exclusive with --base-url / --remote-serve — pass only one")
			os.Exit(2)
		}
		tLocal := time.Now()
		detBase, detModel, detLabel, found := guardDetectLocalBackend()
		localDetectDur = time.Since(tLocal)
		if !found {
			fmt.Fprintln(os.Stderr, guardLocalNothingDetectedMessage())
			os.Exit(2)
		}
		*provider, *baseURL = "openai", detBase
		if strings.TrimSpace(*model) == "" {
			*model = detModel
		}
		if !*quiet {
			fmt.Fprintln(os.Stderr, guardLocalDetectedBanner(detLabel, detBase, detModel))
		}
	} else if *localAuto && localModel && !*quiet {
		fmt.Fprintln(os.Stderr, "fak guard: --gguf is set, so --local is ignored (the in-kernel model is the upstream)")
	}

	if remoteBase != "" {
		if strings.TrimSpace(*baseURL) != "" && strings.TrimSpace(*baseURL) != remoteBase {
			fmt.Fprintf(os.Stderr, "fak guard: --remote-serve and --base-url disagree (%s vs %s) — pass only one\n", remoteBase, strings.TrimSpace(*baseURL))
			os.Exit(2)
		}
		if p := strings.ToLower(strings.TrimSpace(*provider)); p == "anthropic" {
			fmt.Fprintln(os.Stderr, "fak guard: --remote-serve uses the OpenAI-compatible wire fak serve exposes; drop --provider anthropic")
			os.Exit(2)
		}
		// Preflight: a remote serve that is not answering is the most common failure here
		// (box not started, wrong port). Fail loud with the next step, mirroring the
		// exec.LookPath check above, rather than binding a gateway that 502s on first call.
		tRemote := time.Now()
		preflightErr := guardPreflightRemoteServe(remoteBase)
		remotePreflightDur = time.Since(tRemote)
		if preflightErr != nil {
			fmt.Fprintf(os.Stderr, "fak guard: --remote-serve %s is not reachable: %v\n  start it on the box with `fak serve --gguf <weights> --backend cuda --addr 0.0.0.0:8080`, or check the host/port.\n", remoteBase, preflightErr)
			os.Exit(2)
		}
	}

	// 3. Resolve the upstream wire + credential posture. Two worlds:
	//
	//    LOCAL-ONLY (--gguf, no --alongside): fak runs the model itself in-kernel, so
	//    there is NO upstream API, no API key, and no OAuth. Resolve ONLY the wire
	//    (anthropic for claude, openai for codex/…) — that still selects which base-URL
	//    env var points the child at the gateway and labels the banner — and leave the
	//    credential posture empty.
	//
	//    PROXY (default): resolveGuardUpstream picks the provider, base URL, API key, and
	//    the Claude subscription-OAuth default. --remote-serve, when set, pins provider=openai
	//    + base=the box inside the resolver. ALONGSIDE (--gguf --alongside / --gguf
	//    --base-url) takes THIS world too — the API upstream keeps its full credential
	//    posture unchanged, and the loaded local model rides beside it (dual planner).
	var (
		up                   string
		providerAutodetected bool
		resolvedBase         string
		apiKey               string
		pinUpstream          bool
		oauthSource          string
		// credPath is the on-disk .credentials.json path fak is pinning upstream, populated
		// only when pinUpstream is true. It is threaded through to the post-crash auth-recovery
		// check (guardMaybeRecoverAuthCrash) so a wrapped-agent exit caused by an expired
		// subscription token can be diagnosed and, if a fresh login lands, auto-resumed —
		// without re-deriving the config-dir/credentials-file join at every call site.
		credPath string
		// apiKeyFunc re-resolves the upstream credential per request when set. On the
		// pinned Claude subscription path it re-reads the short-lived OAuth access token
		// from disk, so a long guarded session (which outlives the ~1h token) always sends
		// the live token the client has since rotated — never the frozen boot-time one that
		// would 401 even after a fresh /login.
		apiKeyFunc func() string
	)
	tUpstream := time.Now()
	if localModel && !localAlongside {
		up, providerAutodetected = resolveGuardProvider(*provider, command[0])
	} else {
		us := resolveGuardUpstream(*provider, command[0], *baseURL, remoteBase, *apiKeyEnv, *anthropicOAuth, *oauthTokenEnv)
		up, providerAutodetected, resolvedBase = us.provider, us.autodetected, us.baseURL
		apiKey, pinUpstream, oauthSource = us.apiKey, us.pinUpstream, us.oauthSource
		if pinUpstream {
			credPath = filepath.Join(us.claudeConfigDir, ".credentials.json")
		}
		// No subscription token anywhere AND the child has no key of its own: a headless spawn
		// would block on a /login the wrapped agent can never complete (the unrecoverable end of
		// the 'stuck on login' class — distinct from the rotation race, which the pin-on-intent
		// branch handles). Fail loud with the setup guidance BEFORE spawning, but ONLY when stdin
		// is not interactive: an attended terminal can complete the login, so it keeps today's
		// behavior.
		if us.noTokenAnywhere && !cmdGuardStdinInteractive() {
			fmt.Fprintf(os.Stderr, "fak guard: no Claude subscription token found and no ANTHROPIC_API_KEY set, and stdin is not a terminal — refusing to spawn a headless agent that would hang on an interactive login it cannot complete.%s\n", guardLoginStatusNote(us))
			fmt.Fprintln(os.Stderr, "  fix: run `claude` once to log in, or `claude setup-token` for a long-lived token, or export CLAUDE_CODE_OAUTH_TOKEN, or set ANTHROPIC_API_KEY for API billing.")
			os.Exit(2)
		}
		if us.passthroughFallback && !*quiet {
			fmt.Fprintf(os.Stderr, "fak guard: no Claude subscription OAuth token found; falling back to passthrough — the wrapped agent's own credential (a subscription login or ANTHROPIC_API_KEY) is forwarded upstream.%s If you hit a 401, run `claude` once or `claude setup-token`.\n", guardLoginStatusNote(us))
		}
		if us.ambientKeyOverridden && !*quiet {
			fmt.Fprintln(os.Stderr, "fak guard: ANTHROPIC_API_KEY is set but fak defaults to your Claude Pro/Max subscription (OAuth); the key is ignored upstream. Pass --api-key-env ANTHROPIC_API_KEY to use API billing instead.")
		}
		// Pinned Claude subscription: the OAuth access token fak holds upstream is
		// short-lived (the provider rotates it ~hourly, and Claude Code rewrites the
		// refreshed value into the same credential file). Resolving it ONCE at startup
		// pins the boot-time token for the whole session, so a session that outlives the
		// token 401s — and re-logging in does not help, because the refreshed token lands
		// in the file the frozen string never re-reads. So on this path we hand the gateway
		// a credential FUNC that re-reads the live token per request. It falls back to the
		// boot-time apiKey on a transient read miss (the planner's effectiveAPIKey contract).
		if pinUpstream {
			tokenEnv := *oauthTokenEnv
			apiKeyFunc = func() string {
				// Quiet resolve: this runs on EVERY turn to pick up the rotated token, so a
				// genuinely-expired credential must not reprint the expiry WARNING per request
				// (it fired once at boot via resolveGuardUpstream). io.Discard silences only the
				// warning; the token routing/precedence is identical.
				tok, _, err := resolveAnthropicOAuthTokenWarn(tokenEnv, io.Discard)
				if err != nil {
					return ""
				}
				return tok
			}
		}
		// #1834: PROACTIVE, not passive. A headless launch has no interactive `claude` process
		// rewriting .credentials.json, so the reactive 401 self-heal (a 3s-default poll,
		// internal/agent's authRefreshWindow) never has anything rewrite the file for it to
		// notice — it always times out and the upstream 401 surfaces raw. Wire the #1183
		// StaleCred rung (accounts.NewRehydrateCredRung, unwired until now) in HERE, before the
		// child is spawned and before the first upstream request: on a headless
		// pinned-subscription launch, force the freshness check (and, if stale, an active wait
		// for a rotation) now. A refusal means the credential is expired AND could not refresh
		// within the window — fail loud with the same re-auth guidance the noTokenAnywhere gate
		// above uses, naming STALE_CRED so the operator/CI can route on it, instead of letting
		// the child hit a raw upstream_unauthorized. An interactive launch, or a launch not
		// pinning the subscription, is left alone (Ran=false) — see guardRunHeadlessRehydrate's
		// doc for why.
		if pinUpstream {
			if v := guardRunHeadlessRehydrate(cmdGuardStdinInteractive(), pinUpstream, credPath); v.Refused {
				fmt.Fprintf(os.Stderr, "fak guard: STALE_CRED — the Claude subscription OAuth token in %s is expired and did not refresh within the wait window, and stdin is not a terminal — refusing to spawn a headless agent that would only hit a raw upstream 401.%s\n", v.CredPath, guardLoginStatusNote(us))
				fmt.Fprintln(os.Stderr, "  fix: run `claude` once to log in (refreshes the token), or `claude setup-token` for a long-lived token, or export CLAUDE_CODE_OAUTH_TOKEN, or raise FAK_AUTH_REFRESH_WINDOW if a refresh is just slow.")
				os.Exit(2)
			}
		}
	}
	upstreamResolveDur = time.Since(tUpstream)

	// Fail loud BEFORE binding the gateway if the wrapped agent is not on PATH — a cold
	// adopter who installed only fak (curl|sh) and ran `fak guard -- claude` without Claude
	// Code gets an actionable next step instead of a raw exec error after the gateway
	// already started (issue #835, failure 1). Keep this after the headless no-token gate:
	// in automation, the credential refusal is the actionable failure even on hosts whose
	// test image does not have the wrapped binary installed. A command given as an explicit
	// path is left to exec to resolve.
	tPath := time.Now()
	if !strings.ContainsAny(command[0], "/\\") {
		if _, lookErr := exec.LookPath(command[0]); lookErr != nil {
			fmt.Fprintf(os.Stderr, "fak guard: %q is not on your PATH. Install it (Claude Code: https://claude.com/claude-code), or pass the full path / a different agent after `--`.\n", command[0])
			os.Exit(2)
		}
	}
	pathLookupDur = time.Since(tPath)

	requireKey, ok := resolveRequiredKey(*requireKeyEnv, os.Getenv)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak guard: --require-key-env %s is set but empty — refusing to start a gateway with NO authentication (set it or drop the flag)\n", *requireKeyEnv)
		os.Exit(2)
	}
	if *contextBudgetTokens < 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --context-budget-tokens must be non-negative")
		os.Exit(2)
	}
	if *resetOnBudget && contextBudgetLimit <= 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --reset-on-budget requires --context-budget-tokens N")
		os.Exit(2)
	}
	if *restartOnBudget && contextBudgetLimit <= 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --restart-on-budget requires --context-budget-tokens N")
		os.Exit(2)
	}
	if *restartLimit < 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --restart-limit must be non-negative")
		os.Exit(2)
	}
	if maxDurationLimit < 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --max-duration must be non-negative")
		os.Exit(2)
	}
	// Session durability (the file-backed registry restore + the git-backed leaseref
	// publish) is only useful for RESUME/DISPATCH of THIS session later — a plain
	// attended `fak guard -- claude` never reads it back. So GATE the whole block on an
	// actual signal that durability is wanted (#1833): an explicit --session-id (the
	// caller named a stable id to resume against) or --context-budget-tokens > 0 (budget
	// tracking implies the caller cares about this session's persisted drive state).
	// Neither set: skip it entirely — sessionDescriptorMeta/configureServeSessionDurability/
	// registerServeSessionDurability never run, so a default launch spawns zero git
	// subprocesses for this. guardTraceID itself never needs git: an explicit --session-id
	// is used verbatim, and the no-flag default is the fixed "guard" id (identical to
	// defaultSessionIDFromMeta's own zero-cache-key fallback) rather than a git-SHA-derived
	// cache key nothing will read back.
	guardDurabilityWanted := guardSetFlags["session-id"] || contextBudgetLimit > 0 || maxDurationLimit > 0 || hasGuardBudgetEnvelope
	// Wall-clock budget (issue #1584): an INDEPENDENT axis from --context-budget-tokens
	// above — a managed run may be fine on tokens but out of real time, or vice versa.
	// StartTimeBudget both configures the envelope and arms the clock at the current
	// instant, so `fak session status` can report remaining wall-clock time from the very
	// first turn. This governs the SAME guardTraceID the token budget above does; a hidden
	// restart driven by --restart-on-budget re-arms this trace's clock via the ordinary
	// Recontinue path (RecontinueAt), which carries the accumulated elapsed time forward
	// rather than resetting it to zero — see internal/session/timebudget.go.
	var contextOverride *int
	if guardSetFlags["context-budget-tokens"] {
		contextOverride = contextBudgetTokens
	}
	applyGuardSessionBudgetEnvelope(serveSessions, guardTraceID, guardBudgetEnvelope, hasGuardBudgetEnvelope, contextOverride, contextBudgetLimit, maxDurationLimit, time.Now())
	// DEFER the durability setup's git spawns (sessionStartSHA's `git rev-parse HEAD` and
	// PublishSession's `git hash-object -w` + `git update-ref`) until AFTER the gateway is
	// bound and MarkReady()'d (see the goroutine below, right after srv.MarkReady()) rather
	// than blocking the critical path between flag-parse and the agent exec. The register/
	// publish path is already best-effort (sessionDurability.publishBestEffort logs and
	// continues on failure), so running it a few hundred ms late is safe; guardTraceID
	// above is fixed synchronously so the deferred registration publishes under the exact
	// id the gateway is already using as DefaultTraceID.
	restarter := newGuardBudgetRestarter(*restartOnBudget, contextBudgetLimit, *restartLimit, *restartSeedDir, os.Stderr)

	// 3b. LOCAL in-kernel model (--gguf): resolve the alias/URI (downloading on demand),
	//     pick the decode backend, and load the weights + tokenizer through the SAME serve
	//     loaders `fak serve --gguf` uses — so a name works here exactly as it does there.
	//     Done BEFORE binding so a load failure (or a download) is a clean fail-loud, not a
	//     bound-but-broken gateway. nil/false in the proxy path leaves gateway.New
	//     byte-for-byte the pre-existing behavior.
	var (
		inKernelModel *fakmodel.Model
		inKernelTok   *tokenizer.Tokenizer
		inKernelQ4K   bool
		chatBackend   compute.Backend
		loadProfile   *gateway.ModelLoadProfile
		loadPhase     gateway.StartupPhase
	)
	if localModel {
		// Alias (`qwen2.5:7b`) → target ref, then an hf:// URI → a locally cached file.
		if resolved, expanded := modelreg.Resolve(*ggufPath); expanded {
			fmt.Fprintf(os.Stderr, "fak guard: --gguf %s → %s\n", *ggufPath, resolved)
			*ggufPath = resolved
		}
		if hfhub.IsURI(*ggufPath) {
			fctx, fstop := signal.NotifyContext(context.Background(), os.Interrupt)
			resolved, ferr := hfhub.FetchURI(fctx, *ggufPath, os.Stderr)
			fstop()
			if ferr != nil {
				fmt.Fprintf(os.Stderr, "fak guard: --gguf %v\n", ferr)
				os.Exit(1)
			}
			*ggufPath = resolved
		}
		var berr error
		chatBackend, berr = resolveServeChatBackend(*gpuBackend)
		if berr != nil {
			fmt.Fprintln(os.Stderr, "fak guard:", berr)
			os.Exit(2)
		}
		if chatBackend != nil {
			fmt.Fprintf(os.Stderr, "fak guard: in-kernel decode → device backend %q\n", chatBackend.Name())
		}
		inKernelModel, inKernelQ4K, loadProfile, loadPhase = loadServeInKernelModel(*ggufPath, chatBackend, false, contextBudgetLimit, nil)
		if inKernelModel == nil {
			fmt.Fprintf(os.Stderr, "fak guard: failed to load %q into the in-kernel engine\n", *ggufPath)
			os.Exit(1)
		}
		tTok := time.Now()
		var tokOK bool
		inKernelTok, tokOK = resolveServeTokenizer(*tokPath, *ggufPath)
		tokenizerLoadDur = time.Since(tTok)
		if !tokOK || inKernelTok == nil {
			fmt.Fprintf(os.Stderr, "fak guard: %q has no usable tokenizer; pass --tokenizer or use a GGUF with an embedded tokenizer\n", *ggufPath)
			os.Exit(1)
		}
		if localAlongside && !*quiet {
			fmt.Fprintf(os.Stderr, "fak guard: ALONGSIDE mode — model id %q (or \"local\") decodes in-kernel on this box; every other model id proxies to the API upstream as usual\n", localAlias)
		}
	}

	// 4. Bind the listener up front so the real port is known BEFORE we wire the child,
	//    and so there is no bind race between serving and exec. Serve(ctx, ln) accepts
	//    immediately on the goroutine below.
	listenAddr := strings.TrimSpace(*addr)
	if listenAddr == "" {
		listenAddr = "127.0.0.1:0" // an OS-picked free loopback port.
	}
	tListen := time.Now()
	ln, err := net.Listen("tcp", listenAddr)
	must(err)
	// Harness network tracking (#2049): wrap the gateway listener so the wire bytes it
	// SERVES (the child↔gateway traffic the proxy carries, plus any /metrics scrape) are
	// tallied for the kernel half's network axis — WITNESSED in-process, cross-platform,
	// no privileged per-process socket accounting. Only when resource stats are on, so the
	// default path keeps its listener byte-for-byte. Addr/Close delegate via embedding.
	var netCounter *harnessres.CountingListener
	if *resourceStats {
		netCounter = harnessres.NewCountingListener(ln)
		ln = netCounter
	}
	listenDur := time.Since(tListen)
	gwURL := "http://" + ln.Addr().String()

	// A gateway bound BEYOND loopback with no required key is an UNAUTHENTICATED kernel
	// reachable off-host. `fak serve` warns about this in ListenAndServe, but guard binds
	// its own listener and calls Serve() directly (to know the port up front), which skips
	// that check — so re-assert it here rather than let the warning silently vanish.
	if requireKey == "" && !guardLoopbackOnly(ln.Addr().String()) {
		fmt.Fprintf(os.Stderr, "fak guard: WARNING — binding %s with no --require-key-env: the kernel gateway is reachable off-host with NO authentication. Bind a loopback --addr or set --require-key-env.\n", ln.Addr().String())
	}

	// Boot timeline for THIS guard process (mirrors fak serve's StartupPhases,
	// internal/gateway/startup.go): flag-parse and policy-load always fire; the rest are
	// zero-and-omitted when their flag wasn't used, so a plain `fak guard -- claude` launch
	// reports a short, honest phase list rather than a wall of zero-duration rows.
	startupPhases := []gateway.StartupPhase{
		{Name: "flag-parse", Dur: parseDur},
		{Name: "policy-load", Dur: policyDur},
	}
	if localDetectDur > 0 {
		startupPhases = append(startupPhases, gateway.StartupPhase{Name: "local-detect", Dur: localDetectDur})
	}
	if remotePreflightDur > 0 {
		startupPhases = append(startupPhases, gateway.StartupPhase{Name: "remote-serve-preflight", Dur: remotePreflightDur})
	}
	startupPhases = append(startupPhases, gateway.StartupPhase{Name: "upstream-resolve", Dur: upstreamResolveDur})
	startupPhases = append(startupPhases, gateway.StartupPhase{Name: "path-lookup", Dur: pathLookupDur})
	if loadPhase.Name != "" {
		startupPhases = append(startupPhases, loadPhase)
	}
	if tokenizerLoadDur > 0 {
		startupPhases = append(startupPhases, gateway.StartupPhase{Name: "tokenizer-load", Dur: tokenizerLoadDur})
	}
	startupPhases = append(startupPhases, gateway.StartupPhase{Name: "listener-bind", Dur: listenDur})

	srv, err := gateway.New(gateway.Config{
		EngineID: "inkernel",
		Model:    *model,
		BaseURL:  resolvedBase,
		Provider: up,
		APIKey:   apiKey,
		// Re-resolve the pinned subscription OAuth token per request so a long session
		// never sends the stale boot-time bearer (the 401-after-relogin bug). nil in every
		// non-pinned path leaves the static-APIKey behavior byte-for-byte unchanged.
		APIKeyFunc: apiKeyFunc,
		// LOCAL in-kernel model (--gguf): a loaded model + tokenizer with an EMPTY BaseURL
		// makes the gateway serve BOTH /v1/messages (claude) and /v1/chat/completions (codex)
		// from fak's own engine — no upstream call. With --alongside (BaseURL ALSO set) the
		// gateway instead builds its dual planner: requests addressed to LocalModelID (the
		// --gguf alias as typed, or the literal "local") decode in-kernel and everything
		// else proxies upstream unchanged. nil/false in the proxy path, so the default
		// `fak guard -- claude` upstream behavior is unchanged.
		InKernelModel:         inKernelModel,
		Tokenizer:             inKernelTok,
		InKernelQ4K:           inKernelQ4K,
		LocalModelID:          localAlias,
		Backend:               chatBackend,
		PinUpstreamCredential: pinUpstream,
		RequireKey:            requireKey,
		VDSO:                  true,
		Invalidation:          "global",
		Version:               appversion.Current(),
		ReloadPolicy:          policyReloader(*policyPath),
		ResetTrace:            resetTrace,
		ObserveTrace:          observeTrace,
		ObserveSession:        observeSession,
		ControlSession:        controlSession,
		SteerSession:          steerSession,
		ListSessions:          listSessions,
		DecideSession:         decideSession,
		DebitSession:          debitSession,
		ResetOnBudget:         resetOnBudgetHook(*resetOnBudget, contextBudgetLimit),
		OnBudgetExhausted:     restarter.OnBudgetExhausted,
		DefaultTraceID:        guardTraceID,
		StartTime:             t0,
		StartupPhases:         startupPhases,
		// Default OFF (clean terminal); --log routes the full structured stream to a file
		// or stderr. /metrics + /debug/vars + the audit journal carry the record regardless.
		Logf: gwLogf,
		// The observable debug layer (#793) is ON by default so the cache + token-value
		// economy of every turn is visible without a flag; --debug-stats=false or --quiet
		// silences it. The full JSON --log stream stays separate (and off by default).
		DebugStatsf:          debugStatsSink(debugStatsStderr),
		CtxViewBudget:        *ctxViewBudget,
		CompactHistoryBudget: *compactHistoryBudget,
		CompactAnchorHead:    *compactAnchorHead,
		ElideResultBytes:     *elideResultBytes,
		// Inbound twin of #555: prune tool DEFINITIONS the floor can never admit from the
		// Anthropic passthrough's tools[], cache-prefix-preserving. Default-ON because it is
		// behavior-preserving by construction (a pruned tool stays DEFAULT_DENY at the kernel),
		// so it only ever shrinks uncached tool-def tokens. The predicate is a pure read of the
		// installed floor (rt.Adjudicator.NeverAdmits): true only for a name no argument could
		// make Allowed. nil would disable it; we always supply it.
		ToolFloorDenies: rt.Adjudicator.NeverAdmits,
	})
	must(err)
	if loadProfile != nil {
		srv.SetModelLoadProfile(loadProfile)
	}

	// 4. Serve in the background. The gateway lives EXACTLY as long as the child: its
	//    context is cancelled when the agent exits. We deliberately do NOT tear it down
	//    on Ctrl-C — that interrupt belongs to the interactive child (it cancels a turn),
	//    so the parent IGNORES it and stays alive (which is what keeps the gateway up).
	//    Cross-platform: on Unix the freshly exec'd child resets to SIG_DFL and installs
	//    its own SIGINT handler; on Windows the console delivers CTRL_C_EVENT to every
	//    process in the group, so the child receives and handles its own either way.
	signal.Ignore(os.Interrupt)
	ctx, cancel := context.WithCancel(context.Background())
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ctx, ln) }()

	if err, consumed := guardWaitHealthy(gwURL, serveErr, 5*time.Second); err != nil {
		cancel()
		if !consumed {
			<-serveErr // Serve returns once cancel() fires; drain it so no goroutine leaks.
		}
		fmt.Fprintf(os.Stderr, "fak guard: gateway did not become ready: %v\n", err)
		os.Exit(1)
	}
	srv.MarkReady()

	// First-class harness resource tracking (epic #2044): start sampling THIS process's
	// own hardware-resource use (CPU/RSS/IO — the guard process hosts the in-process
	// gateway on the same PID, so this covers the whole kernel half) now that the gateway
	// is ready and before the agent takes the terminal. The wrapped child (the agent half)
	// is folded from its exit state in finishGuardChildAndReport. nil when disabled, which
	// every downstream call tolerates.
	var resSampler *harnessres.Sampler
	if *resourceStats {
		resSampler = harnessres.New()
		// Feed the kernel half's network axis from the listener counter installed at bind
		// time (#2049). Set BEFORE Start so the first sample already carries it.
		if netCounter != nil {
			resSampler.SetNetworkProvider(func() (rx, tx uint64, ok bool) {
				rx, tx = netCounter.Bytes()
				return rx, tx, true
			})
		}
		// Feed the GPU/accelerator VRAM axis when a model runs IN-KERNEL (--gguf/--backend):
		// the harness's hardware footprint then includes the device. The default proxy path
		// has no local GPU, so the provider reports ok=false and the axis stays honestly n/a
		// (#2052). Sourced from the same compute HAL the serve capacity checks use.
		if chatBackend != nil {
			resSampler.SetGPUProvider(func() (used, total uint64, ok bool) {
				t, free, known := compute.DeviceMemoryInfo(chatBackend)
				if !known || t <= 0 {
					return 0, 0, false
				}
				u := t - free
				if free < 0 || u < 0 {
					u = 0
				}
				if u > t {
					u = t
				}
				return uint64(u), uint64(t), true
			})
		}
		resSampler.Start(guardResourceSampleInterval)
		// Expose the live harness resource snapshot on the gateway's /metrics as the
		// fak_harness_* family, so a running session's CPU/mem/IO is scrapeable — not
		// only printed at exit (epic #2044 / #2047). Pull-only: rendered per scrape.
		srv.SetHarnessMetricsProvider(func() string { return resSampler.Snapshot().PrometheusText() })
	}

	// Deferred session durability (#1833): only now — after the gateway is bound and
	// ready, off the critical path to the agent exec — do the git-spawning setup for an
	// opted-in durable session (guardDurabilityWanted, decided above from --session-id /
	// --context-budget-tokens). sessionDescriptorMeta() shells out to `git rev-parse HEAD`
	// and registerServeSessionDurability's PublishSession shells out to `git hash-object -w`
	// + `git update-ref` — three subprocess spawns that used to sit unconditionally between
	// flag-parse and the child exec. Running them in a background goroutine here means a
	// slow or failing git (a huge repo, a detached worktree, no git on PATH) can never delay
	// the agent's first byte; every failure path already routes through stderr warnings
	// (configureServeSessionDurability/registerServeSessionDurability) or
	// publishBestEffort's warnf, so a late or failed write is observable but never fatal.
	if guardDurabilityWanted {
		go func(traceID string, command []string) {
			meta := sessionDescriptorMeta(command)
			if err := configureServeSessionDurability(serveSessions, "", os.Stderr, meta); err != nil {
				fmt.Fprintln(os.Stderr, "fak guard:", err)
				return
			}
			if err := registerServeSessionDurability(context.Background(), traceID); err != nil {
				fmt.Fprintln(os.Stderr, "fak guard:", err)
			}
		}(guardTraceID, command)
	}

	// Default-launch UI: open the 20% `fak info` pane beside the (inline) 80% agent pane, so
	// fak's live cache economy + floor safety counters stay visible the whole session instead
	// of Claude's full-screen repaint hiding them. AUTO fires only for an attended interactive
	// launch inside a multiplexer and is a pure no-op everywhere else, so a bad value is the
	// only failure here. The gateway is up (MarkReady), so gwURL is live for the pane to poll;
	// the pane is opened BEFORE the agent takes the terminal. FAK_GUARD_SPLIT marks the spawned
	// pane + child so a nested guard never re-splits.
	if splitOn, splitErr := guardSplitEnabled(*splitMode, os.Getenv, cmdGuardStdinInteractive(), guardChildInteractive(command)); splitErr != nil {
		cancel()
		fmt.Fprintf(os.Stderr, "fak guard: %v\n", splitErr)
		os.Exit(2)
	} else if splitOn {
		os.Setenv("FAK_GUARD_SPLIT", "1")
		openGuardInfoPane(os.Stderr, os.Getenv, *splitWhere, gwURL, *splitInterval)
	}

	// If --dojo is enabled, log the start of a live dojo episode.
	if *dojoMode {
		if err := logDojoEpisodeStart("guard"); err != nil {
			fmt.Fprintf(os.Stderr, "fak guard: --dojo episode logging failed: %v (continuing without dojo)\n", err)
		}
	}

	// 5. Wire the child: inject ONLY the gateway URL into the child's environment —
	//    never the parent shell, never settings.json. A `claude` in another terminal is
	//    untouched.
	injected := guardInjectedEnv(up, *envName, gwURL)
	var preCompactInstall guardPreCompactInstall
	var preCompactEnv [][2]string
	command, preCompactEnv, preCompactInstall, err = installGuardPreCompactHook(command, *preCompactHook, gwURL)
	if err != nil {
		cancel()
		fmt.Fprintf(os.Stderr, "fak guard: Claude PreCompact hook setup failed: %v\n", err)
		os.Exit(1)
	}
	injected = append(injected, preCompactEnv...)
	// Install the deny-all auto-continue Stop hook, MERGING it into the SAME --settings file the
	// PreCompact hook wrote (preCompactInstall.SettingsPath; "" when PreCompact is off, in which
	// case the Stop hook writes + injects its own). This is the harness half of the deny-all
	// false-stop fix: it resumes the agent past a turn the floor refused entirely. See guard_stophook.go.
	// The task-handoff gate (ENFORCE by default) demands a fak.task-handoff.v1 JSON on every clean
	// Stop and blocks the stop until one is written — right for an unattended `-p` fleet worker,
	// but on an ATTENDED interactive `fak guard -- claude` it spams the TUI and refuses to hand
	// control back every turn. So auto-OFF it for an interactive child the operator did not gate
	// explicitly, while keeping enforce for headless/fleet runs. See guard_handoff_mode.go.
	handoffMode, err := normalizeGuardTaskHandoffMode(
		guardTaskHandoffEffectiveMode(*taskHandoffMode, guardSetFlags["task-handoff"], guardChildInteractive(command)),
	)
	if err != nil {
		cancel()
		fmt.Fprintf(os.Stderr, "fak guard: task handoff setup failed: %v\n", err)
		os.Exit(2)
	}
	handoffFile := strings.TrimSpace(*taskHandoffFile)
	if handoffMode != guardPreCompactModeOff && handoffFile == "" {
		dir, err := os.MkdirTemp("", "fak-guard-handoff-*")
		if err != nil {
			cancel()
			fmt.Fprintf(os.Stderr, "fak guard: task handoff setup failed: %v\n", err)
			os.Exit(1)
		}
		handoffFile = filepath.Join(dir, "task-handoff.json")
	}
	handoffCfg := guardTaskHandoffConfig{Mode: handoffMode, File: handoffFile, Repo: *taskHandoffRepo, Live: *taskHandoffLive}
	var stopHookInstall guardStopHookInstall
	var stopHookEnv [][2]string
	command, stopHookEnv, stopHookInstall, err = installGuardStopHook(command, *denyAllContinue, gwURL, preCompactInstall.SettingsPath, *denyAllWarn, *denyAllFinal, *denyAllMax, handoffCfg)
	if err != nil {
		cancel()
		fmt.Fprintf(os.Stderr, "fak guard: Claude Stop hook setup failed: %v\n", err)
		os.Exit(1)
	}
	injected = append(injected, stopHookEnv...)
	// Seam 4 of the tool process table: observation hooks (PreToolUse/PostToolUse/
	// SessionEnd -> the toolproc journal), merged into the SAME --settings file the
	// PreCompact/Stop installers wrote. Observe-by-default and fail-open: the hook
	// adapter always exits 0, so this can never wedge the child. See guard_toolproc_hooks.go.
	toolprocSettings := stopHookInstall.SettingsPath
	if toolprocSettings == "" {
		toolprocSettings = preCompactInstall.SettingsPath
	}
	var toolprocHookEnv [][2]string
	command, toolprocHookEnv, _, err = installGuardToolprocHooks(command, *toolprocHooks, toolprocSettings)
	if err != nil {
		cancel()
		fmt.Fprintf(os.Stderr, "fak guard: toolproc hook setup failed: %v\n", err)
		os.Exit(1)
	}
	injected = append(injected, toolprocHookEnv...)
	// First-class `fak guard -- codex`: Codex reads custom upstreams from `-c`
	// provider overrides, not OPENAI_BASE_URL. Repoint only Codex children, after the
	// Claude-specific hook installers have had a chance to no-op.
	command, codexInstall := installGuardCodexConfig(command, *codexConfig, gwURL, *apiKeyEnv)
	injected = append(injected, guardClaudeAutoCompactWindowInjection(up, *model, command)...)
	// Live discovery (#1499): register fak's fak_index_*/fak_memory_*/fak_tools_search
	// MCP tools into the wrapped Claude child by default, so a default `fak guard --
	// claude` session can reach them with no manual .mcp.json setup.
	command, mcpInstall, err := installGuardMCPRegistration(command, *mcpRegister, gwURL)
	if err != nil {
		cancel()
		fmt.Fprintf(os.Stderr, "fak guard: Claude MCP registration setup failed: %v\n", err)
		os.Exit(1)
	}

	// Render the FULL startup report to a buffer — always, even under --quiet or the
	// compact banner — and register it on the gateway, so the session serves it back
	// on demand for its whole life (`fak info --startup` / startup_report on
	// /debug/vars). What reaches the terminal RIGHT NOW is bannerMode's call below.
	var startupReport strings.Builder
	{
		if providerAutodetected {
			fmt.Fprintf(&startupReport, "fak guard: detected agent %q -> --provider %s (pass --provider to override)\n", strings.ToLower(filepath.Base(command[0])), up)
		}
		injectNames := injected[0][0]
		for _, kv := range injected[1:] {
			injectNames += ", " + kv[0]
		}
		localLabel := ""
		if localModel {
			localLabel = filepath.Base(*ggufPath)
		}
		printGuardBanner(&startupReport, guardBannerVersion(), guardBannerBuildStamp(), gwURL, up, resolvedBase, floorSource, injectNames, injected[0][1], logLabel, auditLabel, refusalCarryForward, remoteBase != "", localModel, localLabel, command)
		if preCompactInstall.Applied {
			fmt.Fprintf(&startupReport, "fak guard: Claude PreCompact hook: %s (settings %s)\n", preCompactInstall.Mode, preCompactInstall.SettingsPath)
		}
		if stopHookInstall.Applied {
			fmt.Fprintf(&startupReport, "fak guard: Claude Stop hook (deny-all auto-continue): %s — graduated nudge→warn(%d)→final(%d)→give-up(>%d consecutive); a floor-refused-everything turn is reported as end_turn and this resumes the agent past it with escalating guidance, the give-up logged (--deny-all-continue off to disable)\n", stopHookInstall.Mode, stopHookInstall.WarnAt, stopHookInstall.FinalAt, stopHookInstall.Max)
		}
		if len(guardTaskHandoffEnv(handoffCfg)) > 0 {
			live := "validate-only"
			if handoffCfg.Live {
				live = "live-issue-sync"
			}
			fmt.Fprintf(&startupReport, "fak guard: task handoff Stop gate: %s (%s) — clean stops require %s; child sees $%s\n", handoffCfg.Mode, live, handoffCfg.File, guardTaskHandoffFileEnv)
		}
		printGuardCodexNote(&startupReport, codexInstall)
		printGuardMCPNote(&startupReport, mcpInstall)
		printGuardCapabilitiesNote(&startupReport, mcpInstall)
		switch {
		case debugStatsStderr:
			fmt.Fprintln(&startupReport, "  debug      : observable layer ON — one cache/token-value line per turn to stderr (request_tokens/cache_read/cache_creation/cache_hit/cache_rebate_tokens/compact/health); --debug-stats=false or --quiet to silence")
		case *debugStats && !*quiet:
			fmt.Fprintln(&startupReport, "  debug      : observable layer ON — the per-turn cache/token-value economy is kept OUT of the agent's full-screen UI to avoid corrupting it; read it live in the `fak info` pane and in the exit summary. Pass --debug-stats to also stream it here, --debug-stats=false to disable")
		}
		// A LOCAL in-kernel model has no upstream credential to report; the proxy-path auth
		// note (subscription OAuth vs passthrough) only applies when fak proxies an API.
		if !localModel {
			switch {
			case pinUpstream:
				fmt.Fprintf(&startupReport, "fak guard: upstream auth — Claude Pro/Max subscription (provider-reported identity; OAuth token from %s, sent as a bearer token)\n", oauthSource)
			case up == "anthropic":
				fmt.Fprintln(&startupReport, "fak guard: upstream auth — passthrough (Claude Code forwards its own credential through the gateway)")
			}
		}
		if contextBudgetLimit > 0 {
			fmt.Fprintf(&startupReport, "fak guard: session budget — trace_id=%s context_tokens=%d\n", guardTraceID, contextBudgetLimit)
			if *resetOnBudget {
				fmt.Fprintln(&startupReport, "fak guard: session reset — transparent carryover enabled")
			}
			if *restartOnBudget {
				fmt.Fprintln(&startupReport, "fak guard: session restart — child relaunch on budget exhaustion enabled")
			}
		}
		if maxDurationLimit > 0 {
			fmt.Fprintf(&startupReport, "fak guard: session time budget — trace_id=%s max_duration=%s\n", guardTraceID, maxDurationLimit.String())
		}
	}

	srv.SetStartupReport(startupReport.String())
	switch bannerMode {
	case guardBannerFull:
		fmt.Fprint(os.Stderr, startupReport.String())
	case guardBannerCompact:
		printGuardCompactBanner(os.Stderr, guardBannerVersion(), guardShortBuildID(), gwURL, command, refusalCarryForward)
	}

	// 6. Run the wrapped agent, then tear the gateway down and report the session.
	spawnMeta := newGuardChildSpawnMetadata(guardTraceID, policyDigest, up, rt, command)
	if restarter.Enabled() {
		runGuardChildSupervisedAndReport(command, injected, pinUpstream, credPath, spawnMeta, restarter, srv, cancel, serveErr, *quiet, auditJournal, auditSeq0, guardTraceID, command[0], up, *dojoMode, resSampler)
		return
	}
	runGuardChildAndReport(command, injected, pinUpstream, credPath, spawnMeta, srv, cancel, serveErr, *quiet, auditJournal, auditSeq0, guardTraceID, command[0], up, *dojoMode, resSampler)
}
