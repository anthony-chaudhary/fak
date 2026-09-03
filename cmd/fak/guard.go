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
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountobs"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/fleetspine"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/goalpark"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/headroom"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/logvault"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/toolcallcontrol"
)

// guardLaunchPlan separates the wrapped harness's admitted identity from the argv that
// launch adapters are free to rewrite. Its fields stay private and every slice accessor
// clones, so a broker, trampoline, or exec-boundary transform cannot mutate the semantic
// command or the resolved profile in place.

func cmdGuard(argv []string) {
	cmdManageCommand("guard", argv)
}

func cmdManageCommand(commandName string, argv []string) {
	guardUsageStart = time.Now()
	guardUsageOnce = new(sync.Once)
	if routeGuardOperatorSubcommand(commandName, argv) {
		return
	}
	t0 := time.Now()
	// --core-lock-all (#5423, epic #5170 Track C) is the session-wide RATCHET posture:
	// for the life of this launch no channel — operator allow/deny overlay reload, the
	// allow watcher, POST /v1/fak/policy/reload, or a self-authored overlay — may WIDEN
	// the capability floor; only tighten-only and no-op amendments are installed. It is
	// PEELED here rather than registered on the FlagSet below because fs is
	// flag.ExitOnError: the posture has to be recorded before any parse, and the peel
	// keeps the flag off the wrapped child's argv (see guardLaunchCoreLockAll). Recorded
	// before the verb peels' successors and long before the gateway binds, so every
	// amendment site that reads guardCoreLockAllActive() sees the final value.
	var guardCoreLock bool
	guardCoreLock, argv = guardLaunchCoreLockAll(argv)
	setGuardCoreLockAll(guardCoreLock)
	fs := flag.NewFlagSet(commandName, flag.ExitOnError)
	hostRecovery := fs.Bool("host-recovery", false, "opt this interactive session into automatic terminal-host recovery")
	rotateMode := fs.String("rotate", "", "account rotation: auto|off|<seat> (default auto headless, off interactive)")
	verbFlagUsage(fs, "guard")
	addr := fs.String("addr", "", "gateway listen address (default: a private 127.0.0.1 port the OS picks)")
	provider := fs.String("provider", "", "upstream wire the gateway proxies to: anthropic|openai|gemini|xai (default: auto-detected from the agent name — claude->anthropic, codex/opencode->openai — else anthropic)")
	outputProfile := fs.String("output-profile", agentDefaultOutputStyle, "response profile for the witnessed Claude instruction seam; defaults to caveman:medium, full disables")
	workProfile := fs.String("work-profile", agentDefaultWorkProfile, "work policy for the witnessed Claude instruction seam; defaults to ponytail:medium, standard disables")
	baseURL := fs.String("base-url", "", "upstream provider base URL (default: the provider's public API, e.g. anthropic -> https://api.anthropic.com)")
	remoteServe := fs.String("remote-serve", "", "point the guarded turn's INFERENCE at a remote `fak serve` running on a lab box you chose (HOST or HOST:PORT, default port 8080), or at a public-safe local alias like @lab/glm-5.2 resolved from the user's lab target config. Forces the OpenAI-compatible wire and upstream base http://HOST:PORT/v1 (the /v1 fak serve serves its chat route under), so the dev turn runs on the lab GPU while the kernel still adjudicates locally. Mutually exclusive with --base-url; preflights GET /healthz AND /v1/models and fails loud if the box is not serving the /v1 surface.")
	model := fs.String("model", "", "upstream model id override (default: forward the client's own model id)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the UPSTREAM API key. For --provider anthropic this is the explicit opt-IN to API billing (e.g. --api-key-env ANTHROPIC_API_KEY); the default is your Claude Pro/Max subscription via OAuth, even when ANTHROPIC_API_KEY is exported. For other providers the default forwards the client's own key (passthrough).")
	anthropicOAuth := fs.Bool("anthropic-oauth", false, "force the Claude Pro/Max SUBSCRIPTION OAuth token upstream (sourced, in precedence order, from CLAUDE_CODE_OAUTH_TOKEN, then <claude-config>/.credentials.json, then <claude-config>/.oauth-token) sent as Authorization: Bearer + the oauth beta. This is ALREADY the default for --provider anthropic (even when ANTHROPIC_API_KEY is set); the flag forces it and fails loud if no token is found.")
	oauthTokenEnv := fs.String("oauth-token-env", "CLAUDE_CODE_OAUTH_TOKEN", "env var to read the subscription OAuth token from first")
	policyPath := fs.String("policy", "", "capability-floor manifest to enforce (default: the built-in guard floor; see --dump-policy)")
	var allowTools launchToolFlag
	fs.Var(&allowTools, "allow-tool", "grant one exact tool name for THIS guard process only (repeatable). The grant re-admits DEFAULT_DENY tools but cannot bypass explicit denies, dangerous-argument rules, self-modification, or later tightening.")
	envName := fs.String("env", "", "env var to inject the gateway URL into the child (default: chosen by --provider)")
	requireKeyEnv := fs.String("require-key-env", "", "require this env var's bearer token on the gateway (loopback rarely needs it)")
	logPath := fs.String("log", "", "write the gateway's per-request + per-verdict structured logs to this file (or '-' for stderr); default off to keep the agent's terminal clean")
	auditPath := fs.String("audit", "", "write the durable, hash-chained DECISION JOURNAL to this file (default: .dispatch-runs/guard-audit/interactive-<pid>-<hash>.jsonl; pass 'off' to disable). Every kernel verdict this session is appended as a tamper-evident JSONL row you can later replay with `fak audit verify`.")
	// --no-audit was a pure alias for `--audit off` (both landed on the same branch in
	// guardAuditPlan), so it cost the front door a flag and bought nothing a caller could
	// not already spell. Folded into --audit, which documents 'off' in its own help text.
	// The internal noAudit seam stays — guardAuditPlan/runGuardReplay are still exercised
	// with it directly — but nothing on the CLI sets it any more.
	dumpPolicy := fs.Bool("dump-policy", false, "print the built-in guard capability floor (an editable manifest) and exit")
	probeMode := fs.Bool("probe", false, "local one-shot smoke mode: keep the normal guarded gateway but default the task-handoff Stop gate OFF, so `fak guard --probe -- claude -p \"say pong\"` proves the wire without demanding a fleet handoff. Explicit --task-handoff still wins.")
	quiet := fs.Bool("quiet", false, "suppress the startup banner and the exit audit summary")
	bannerFlag := fs.String("banner", guardBannerAuto, "startup surface before handing the terminal to the agent: auto|full|compact|animate|off. AUTO (default) emits only delayed loading progress for healthy interactive and noninteractive launches: no guard report, profiles, identity/configuration, animation, or persistent settle lines. Explicit full, compact, and animate retain those displays; off suppresses the startup surface. The full report is always recorded on the in-process gateway regardless — read it any time during the session with `fak info --startup` (it is the startup_report field of /debug/vars), and it is spilled to the terminal in full when the launch itself fails. --quiet still silences everything.")
	resourceStats := fs.Bool("resource-stats", true, "ON by default — track the HARNESS's own hardware-resource use this session (CPU, memory/RSS, disk-I/O) for BOTH halves: the kernel (this guard process + the in-process gateway, sampled continuously) and the agent (the wrapped child, folded from its exit state). Reported as one line in the exit summary and appended to .fak/nightrun/harness-resources.jsonl. Pass --resource-stats=false to disable (epic #2044).")
	childMaxMemoryMB := fs.Uint64("child-max-memory-mb", 0, "maximum wrapped-child process-tree memory in MiB (0 uses the host-sized default)")
	childResourcePoll := fs.Duration("child-resource-poll", guardResourcePollDefault, "wrapped-child resource sampling interval (minimum 100ms)")
	childResourceJournal := fs.String("child-resource-journal", "", "child-resource receipt JSONL path (default: user config directory)")
	debugStats := fs.Bool("debug-stats", true, "ON by default — the observable debug layer: print ONE compact, payload-free line per served turn to stderr with the turn's cache + token-value economy (request_tokens/cache_read/cache_creation, cache_hit, cache_rebate_tokens, and session-to-date current/previous/average/median/high/low cache savings), the SAFETY half (blocked:/repaired:/quarantined: with the dominant reason whenever the kernel refused, rewrote, or paged out a call THIS turn — so a refused rm -rf or a quarantined secret is visible the moment it happens, not only in the exit summary), the compaction action, and the resetScore SHADOW health (healthy_cache|cache_decay|stale_prefix|cooldown|unknown_provider). These counts are the provider's own usage numbers, so it works natively over your Claude subscription OAuth. Independent of --log; pass --debug-stats=false or --quiet to silence it (#793).")
	preCompactHook := fs.String("precompact-hook", guardPreCompactModeShadow, "Claude Code PreCompact hook actuator for auto-compaction: off|shadow|enforce. shadow logs would-block/would-allow while exiting 0; enforce returns the compactcohere posture exit code.")
	arbitrateConfig := guardArbitrateConfig{Mode: guardArbitrateModeShadow}
	fs.Var(guardArbitrateFlagValue{cfg: &arbitrateConfig}, "lease", "guard launch lease admission as comma-separated mode=off|shadow|enforce,lane=<lane>,tree=<glob>,force=true (tree repeatable within one value; defaults to **/*)")
	denyAllSettings := newDenyAllRetrySettings()
	fs.Var(&denyAllSettings, "deny-all-continue", "Claude Code Stop hook that auto-RESUMES the agent after a deny-all turn: off|shadow|enforce, optionally followed by warn=N,final=N,max=N,same-stop=N. ENFORCE by default; the limits form one bounded retry policy. Example: --deny-all-continue=enforce,warn=2,final=4,max=6,same-stop=6. Claude children only.")
	toolprocHooks := fs.String("toolproc-hooks", guardToolprocModeObserve, "Claude Code tool-process observation hooks (off|observe, observe by default): PreToolUse/PostToolUse/SessionEnd append spawn/exit/session_end rows to the workspace toolproc journal (fail-open; `fak toolproc ps --events .fak/toolproc/journal.jsonl` reads the live table). Claude children only.")
	toolcallControl := fs.String("toolcall-control", "shadow", "avoidable tool-call control: off|shadow|enforce (default shadow)")
	taskHandoffMode := fs.String("task-handoff", guardPreCompactModeEnforce, "Claude Code Stop hook completion handoff gate: off|shadow|enforce. ENFORCE by default: on a clean stop, require a valid fak.task-handoff.v1 JSON with witnessed done + current state + 1-2 next steps or no-next-step reason. The path is exposed as FAK_TASK_HANDOFF_FILE.")
	taskHandoffFile := fs.String("task-handoff-file", "", "path the wrapped agent must write with fak.task-handoff.v1 before a clean stop (default: a private temp file for this guard session)")
	taskHandoffRepo := fs.String("task-handoff-repo", "", "owner/repo for optional live handoff issue sync (passed to fak task handoff --live)")
	taskHandoffLive := fs.Bool("task-handoff-live", false, "after a valid handoff with next_steps, the Stop hook runs fak task handoff --live before allowing the clean stop")
	operatorDirected := fs.String("operator-directed", guardOperatorDirectedModeWarn, "Claude Code Stop hook that catches a HEADLESS turn ending by asking a human (\"do you want me to push?\", \"waiting for your approval\") — a question no one is there to answer, so the work stalls: off|shadow|warn|enforce. WARN by default (soak: prints the choicetriage remediation, allows the stop); enforce BLOCKS a resolvable ask and feeds the remediation back so the agent acts, while routing a HUMAN_RESIDUAL wall as a typed escalation. Auto-OFF for an attended interactive child the operator did not gate explicitly (a human can always ask); never blocks an interactive session.")
	splitMode := fs.String("split", "auto", "the default-launch UI: open a 20% `fak info` pane BESIDE the 80% interactive agent pane so the live cache/token economy + the kernel floor's safety counters stay on screen (a bare `fak guard -- claude` hands the whole terminal to Claude, hiding fak). auto|on|off. AUTO (default): enable ONLY for an attended interactive launch inside a splittable terminal context (tmux; Windows Terminal via $WT_SESSION; on macOS, iTerm2 gets an inline split and Apple Terminal — which has no split panes — a companion fak-info window, both via osascript); no-op for headless/piped/CI/plain-terminal launches (zero behavior change there). on forces it (prints a recipe if no host is found); off disables. The pane polls THIS guard's own loopback gateway (auth-exempt on loopback); the bearer is never placed on a pane command line.")
	splitWhere := fs.String("split-where", "bottom", "with --split: place the 20% fak-info pane as a \"bottom\" strip or a \"right\" column")
	splitInterval := fs.Duration("split-interval", 2*time.Second, "with --split: refresh interval for the fak-info pane")
	splitDryRun := fs.Bool("split-dry-run", false, "preview the --split 80/20 plan (resolved multiplexer, geometry, and the exact `fak info` pane command) and EXIT, without bringing up the gateway, spawning a pane, or launching the agent. Use it to see what --split will do before handing the terminal to the agent.")
	ctxViewBudget := fs.Int("ctx-view-budget", agent.DefaultCtxViewBudget, "wire the ctxplan context PLANNER into the live guard loop: each buffered turn, re-materialize the forwarded history as an O(1) planned VIEW under this resident-token budget (a planned view in place of appending the whole transcript, #555). DEFAULT-ON at a conservative 8000 resident tokens; pass 0 to disable (leaves the existing path byte-for-byte unchanged). The planner only ever SHORTENS and falls open to the full history on any doubt; on the Anthropic passthrough it keeps the cached prefix byte-identical (witness: docs/notes/CTXVIEW-DEFAULT-ON-WITNESS-2026-06-28.md). The streaming fast-path bypasses this; the buffered turn path is what gets planned.")
	compactHistoryBudget := fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "compact OLD conversation turns in the OUTBOUND Anthropic request body down to this resident-token budget while keeping the cache_control prefix BYTE-IDENTICAL, so the upstream prompt-cache hit survives. This reaches the flagship `fak guard -- claude` passthrough (where the body is forwarded verbatim, #555). DEFAULT-ON, but NOTE the resolved default for `fak guard` is NOT the value printed here: unless you pass this flag explicitly, every guard launch resolves gateway.HeadlessCompactHistoryBudget (96000) instead, because a guard always fronts Claude Code and its fixed system+tools floor (resolveGuardCompactBudget, #4888). The number compared against it is the COMPACTIBLE messages[] span — the system/tools head is a separate top-level block and is NOT counted — so read the live figure off /debug/vars metrics.compaction (budget, last_suffix_tokens, peak_suffix_tokens) rather than inferring it from this line. Once that span sprawls past the resolved budget the cut fires and sheds the un-cacheable middle the provider re-bills every turn; a typical short session stays untouched. Pass 0 to disable (body forwarded byte-for-byte). Anthropic passthrough only.")
	compactAnchorHead := fs.Bool("compact-anchor-head", true, "re-anchor --compact-history-budget's protected prefix on the stable system/tools head instead of the first-breakpoint anchor, fixing the anchor-starved trap (#1407) where real Claude Code traffic's recent cache_control breakpoint protects almost the whole conversation so the budget can never shed anything (see the 'anchor-starved' diagnostic). DEFAULT-ON, and every fire stays gated on the burst economics (CacheBurstPaysBack, #1408): a WARM session with no bounded turns budget never bursts — it fires only when a wired session-turn horizon repays the one-time burst, or when the trace OBSERVABLY idled past the message-breakpoint cache TTL since its last served turn (the suffix re-bills cold that turn anyway, so the cut is penalty-free — the long-session firing path a plain `fak guard -- claude` actually hits). Pass =false to pin the old warm-only first-breakpoint anchor.")
	compactSolvencyFloor := fs.Int("compact-solvency-floor", 0, "resident-token occupancy at or above which CONTEXT SOLVENCY overrides the head-anchored burst gate's cache economics: once this trace's OBSERVED peak resident window reaches it, --compact-history-budget fires even when the burst does not repay (CacheBurstPaysBack says no). It fixes the measured inversion where the pure-economics gate refuses HARDEST exactly where refusing is most expensive — over 3191 real served turns the fire rate ran 33% at 96-125k, 14% at 140-155k, 3% at 155-170k and 0% above 170k, and 100% of traces that ever fired never fired again, drifting a median +33.8k further into the window before the run ended. Above the floor the question stops being 'does this burst pay?' and becomes 'can we afford not to?': the burst penalty is one-time and bounded, hitting the context wall costs the whole session. Set it to a fraction (~0.85) of (model context window − output reserve); the gateway cannot derive it because it never sees a window size. 0 (the default) keeps the gate on pure economics, byte-for-byte. Forced fires are counted apart as 'solvency-forced' in the exit summary so they are never booked as cache wins. Consulted only when --compact-anchor-head is on and the head re-anchor engages; it can only ever turn a burst_unprofitable BAIL into a fire, never the reverse.")
	assumeSessionTurns := fs.Int("assume-session-turns", gateway.DefaultAssumedSessionTurns, "the session length the head-anchored burst gate (--compact-anchor-head) ASSUMES when no bounded turn horizon is wired — the common `fak guard -- claude` case, where the wrapped harness owns the turn loop and hands the gateway no Budget.TurnsLeft. It lets a WARM continuously-active long session shed early instead of waiting to OBSERVABLY idle past the message-span cache TTL: the gate maps the trace's real served-turn depth to CurrentTurn and this value to TotalTurns, fires early (many repaying turns left) and refuses near the presumed end — the same one-time-burst break-even economics (CacheBurstPaysBack, #1408), just given a history-based length instead of refusing outright. DEFAULT-ON at gateway.DefaultAssumedSessionTurns; a genuine wired Budget.TurnsLeft horizon always WINS over this prior, and a large invalidated suffix still refuses regardless. Pass 0 to disable (byte-for-byte the conservative no-horizon behavior). Consulted only when --compact-anchor-head is on and the head re-anchor engages; inert on every other path.")
	elideResultBytes := fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes, "ON by default at gateway.DefaultElideResultBytes (the reviewed gateway.DocumentedElideResultBytes threshold): shrink oversized tool_result bodies outside the active working set to a bounded head+tail form once they exceed this byte threshold. 0 disables.")
	elideStaleReads := fs.Bool("elide-stale-reads", gateway.DefaultElideStaleReads, "ON by default (gateway.DefaultElideStaleReads): replace a Read tool_result whose file was Edited/Written in a LATER in-session turn (a stale, superseded snapshot no longer reflecting disk) with a compact fak_context_restore marker, in the SAME cache-safe working-set band as --elide-result-bytes and stashing the pre-edit body behind a restore handle. The safer, restorable sibling of --elide-result-bytes: strictly more conservative predicate (superseded, not merely big), fail-safe identity on any ambiguity, protected cache prefix proven byte-identical. Size-independent; lossy but restorable. Pass =false to opt out. Anthropic passthrough only.")
	vcacheAnchor := fs.Bool("vcache-anchor", gateway.DefaultVCacheAnchor, "M2 star-anchor pre-flight gate (#1493): on the Anthropic passthrough, APPLY cachemeta.RecommendLayout before send — hoist volatile system blocks behind a byte-stable cacheable anchor and splice a cache_control breakpoint onto the stable head a no-breakpoint caller did NOT send, so the first natural request warms provider prefix caching and later siblings read it. DEFAULT-ON, DECOUPLED from --compact-history-budget (that path only placed the anchor while its own budget was >0, so --compact-history-budget=0 silently took anchoring down with it). Fail-safe identity on any ambiguity — a hoist that would change the model-visible prefix is REFUSED, not applied — and idempotent with the compaction/TTL placements (a body already carrying a breakpoint bails already_set). Pass =false to opt out. Anthropic passthrough only.")
	deferColdTools := fs.Bool("defer-cold-tools", gateway.DefaultDeferColdTools, "the 10x floor lever (#3232, epic #3229): on the OUTBOUND Anthropic body, mark every allowed-but-COLD custom tool `defer_loading:true` and inject one 	ool_search_tool`, so the provider loads only the HOT core (the floor's built-ins Read/Edit/Write/Bash/Grep/Glob/Task/TodoWrite + web, plus the search tool) into context and faults a cold schema in on demand. The systemic tool-schema slice is ~35.8k of the ~41k fresh-session floor, and to fak's gateway it is all just req.Tools — this is the one seam that reaches it. Deterministic + cache-safe (byte-stable tools[] turn-over-turn, so the provider prompt-cache prefix survives) and fail-safe identity on any ambiguity (non-JSON, no tools, only hot tools); every deferred def stays byte-complete in tools[], so a first real use still resolves — nothing goes silently missing. DEFAULT ON (gateway.DefaultDeferColdTools, the #3537 flip; the A/B token-delta x held-accuracy x poison gates reported PASS, the #3200 pin/quarantine guards the fault-in). Pass =false to opt out; ablate an A/B arm with FAK_ABLATE_DEFER_TOOLS=1 (FAK_DEFER_COLD_TOOLS=1 still forces it on). Anthropic passthrough only.")
	exposeProfile := fs.String("expose-profile", "", "in-kernel fak_* MCP tool-surface profile (#3607): \"\" (full registry, default) | \"headless\" — a curated allowlist for a single-issue dispatch worker (fak_capabilities, fak_admit, fak_adjudicate, fak_memory_run, fak_tools_search), pruning the ~9.9k-token full-registry schema floor every worker otherwise pays each turn; the rest page in on demand through the still-exposed fak_tools_search. `fak dispatch` launches workers with =headless. The FAK_GUARD_EXPOSE_PROFILE env OVERRIDES this flag (the fleet opt-out: set it to `full`/`off` to restore the whole registry). Any value other than \"headless\" keeps the full registry.")
	sessionID := fs.String("session-id", "", "default trace/session id for wrapped agents that omit X-Trace-Id or MCP trace_id (default: a fresh launch id derived from host, cwd, and wrapped argv; pass this flag for a stable resumable id)")
	sessionPressureGate := fs.String("session-pressure-gate", "", "before launching the wrapped agent, audit recent sessions for Opus-cost / long-context pressure and refuse when actions at or above this severity exist. Spec: THRESHOLD[,days=N][,max=N][,report=PATH][,justify=TEXT] — THRESHOLD is high|medium|none|off (off by default, so a bare `--session-pressure-gate high` is the common form); days (default 7) and max (default 40) size the audit window over this workspace's transcript namespace; report=PATH writes the fak.session_audit.actions.v1 launch-gate report before allowing or refusing; justify=TEXT, with an explicit Opus --model, is the justification that allows the launch while still recording that report. justify= takes the REST of the spec so prose may contain commas — put it last. e.g. --session-pressure-gate high,days=3,report=pressure.json")
	contextBudgetTokens := fs.Int("context-budget-tokens", 0, "seed the guard session with this prompt/context-token budget; exhaustion returns a reset directive with continuation_id (0 = off)")
	maxDuration := fs.Duration("max-duration", 0, "govern this guard session to at most this much REAL WALL-CLOCK time (issue #1584), tracked independently of --context-budget-tokens and surviving a --restart-on-budget hidden restart (the elapsed total carries forward, it does not reset to zero). 0 = unbounded (still tracked for `fak session status`, just never stops the run). Query/inspect anytime with `fak session status <id>`; the time budget drains the session to Draining/Stopped with reason TIME_BUDGET_EXHAUSTED exactly like a token-budget exhaustion.")
	budgetEnvelopeSpec := fs.String("budget-envelope", "", "managed-context budget envelope (#1573): turns=20,tokens=200000,context=64000,wall=2h,spend=$25,throughput=40/s,max-tokens=1024,gap=250ms. Seeds this guard session's budget/pace/wall axes; explicit --context-budget-tokens and --max-duration override those envelope axes.")
	resetOnBudget := fs.Bool("reset-on-budget", false, "on context-budget exhaustion, re-arm the continuation trace with a carryover seed and continue transparently instead of returning 409 (requires --context-budget-tokens)")
	restartOnBudget := fs.Bool("restart-on-budget", false, "on context-budget exhaustion, stop and relaunch the wrapped child under the continuation trace, writing a carryover seed JSON and exposing it via FAK_RESET_* env vars (requires --context-budget-tokens)")
	restartLimit := fs.Int("restart-limit", 0, "maximum child relaunches for --restart-on-budget; 0 means unlimited")
	restartSeedDir := fs.String("restart-seed-dir", "", "directory for --restart-on-budget carryover seed JSON files (default: OS temp dir, one private directory per reset)")
	restartSeedHandback := fs.Bool("restart-seed-handback", false, "with --restart-on-budget: for a recognized headless/no-continue child (e.g. a deliberately fresh-session `claude -p`), inject the carryover seed_text as the relaunch's initial prompt via --append-system-prompt INSTEAD of reattaching the prior transcript with --continue. The seed is bounded to a documented token budget and any truncation is logged (no silent drop). An unrecognized agent stays a no-op with its seed left on disk (#3056).")
	landlockHooks := fs.Bool("landlock-hooks", false, "LINUX-ONLY defense-in-depth: run the spawned agent under a Landlock profile that makes the git hook surface (.git/hooks + core.hooksPath) READ-ONLY while the rest of the tree stays writable, so a laundered write cannot drop an executable hook. OFF by default; fails OPEN (logs + spawns unrestricted) on a kernel without Landlock or on a non-Linux host. Also settable via "+guard.EnvOptIn+"=1.")
	dojoMode := fs.Bool("dojo", false, "enable live dojo mode: write a start-marker for this guard session, then persist a scored vcache live row at shutdown when provider-cache telemetry exists.")
	ggufPath := fs.String("gguf", "", "run a SMALL MODEL IN-KERNEL as the local upstream — no API key, no network, no second server. fak loads these GGUF weights into its OWN engine and serves them to the wrapped agent, so the whole `local model + your coding harness + kernel floor` stack is ONE command (`fak guard --gguf qwen2.5:7b -- claude`). Accepts a model alias (`fak ls`), an hf://owner/repo/file.gguf URI (downloaded on demand), or a local .gguf path. Every tool call the agent proposes is still adjudicated by the same capability floor and recorded in the same audit journal — only the inference moves onto YOUR box. Alone, the local model IS the upstream (mutually exclusive with --remote-serve); with --alongside or an explicit --base-url it serves ALONGSIDE the API upstream instead (see --alongside).")
	alongside := fs.Bool("alongside", false, "with --gguf: serve the small local model ALONGSIDE the API upstream instead of REPLACING it (the dual planner). The wrapped agent's normal turns proxy to the provider exactly as a plain `fak guard` session (same OAuth/passthrough, same prompt-cache preservation), while any request addressed to the --gguf model's alias — or the literal model id \"local\" — decodes in-kernel on your box with no upstream call and no tokens billed (e.g. point a cheap subagent tier at it). Implied by --gguf + an explicit --base-url.")
	localAuto := fs.Bool("local", false, "auto-detect a local OpenAI-compatible model server you are ALREADY running (Ollama, LM Studio, Qwen3.6 dogfood, or llama.cpp) and wire guard's upstream to it with zero flags — `fak guard --local -- codex` becomes a governed local coding loop with no base-URL hunting. Probes, fail-soft (~300ms each), Ollama (127.0.0.1:11434, honors OLLAMA_HOST), then LM Studio (127.0.0.1:1234), then Qwen3.6 dogfood (127.0.0.1:8131), then llama.cpp (127.0.0.1:8080); the first live one wins and a coding-tuned served model is preferred. If --gguf is ALSO passed it wins (that is the no-server in-kernel path); if nothing is detected and no --gguf, fak fails loud with how to start a server. Mutually exclusive with --base-url / --remote-serve.")
	gpuBackend := fs.String("backend", "", "with --gguf: compute backend for the in-kernel decode — empty = the CPU reference path; a registered device like 'cuda' runs prefill+decode through the GPU HAL (needs a -tags cuda build AND a reachable GPU). Fails loud if named but unavailable, so a typo never silently runs on CPU.")
	guardNativeFlags := registerGuardNativeControlFlags(fs)
	tokPath := fs.String("tokenizer", "", "with --gguf: OPTIONAL tokenizer override (a tokenizer.json or its directory); default uses the GGUF's EMBEDDED tokenizer. Pass this only for a checkpoint with no embedded BPE tokenizer or a custom vocab.")
	nativeAdmissionTokenBudget := registerGuardAdmissionFlag(fs)
	replayTrace := fs.String("replay-trace", "", "DON'T wrap a live agent — instead REPLAY a recorded trace fixture through the real guard end to end and watch the floor fire. Stands up the gateway against a built-in fake upstream that emits the fixture's tool_use + token-usage turns, posts each turn through the SAME adjudication path `fak guard -- claude` uses, and prints per-turn what was allowed vs denied (with the deny reason), the turn's token/cache economy, and the journal rows recorded — then the exit summary + the verify command. No API key, no GPU, no child process. Use it to understand exactly what the guard does to a trace that leads to token work, and to demo the floor. See internal/gateway/testdata/guard-trace-e2e.json for the fixture shape.")
	replayWire := fs.String("replay-wire", "anthropic", "with --replay-trace: the provider wire to replay over (anthropic = the `fak guard -- claude` flagship /v1/messages path; openai = the codex/opencode /v1/chat/completions path).")
	codexConfig := fs.Bool("codex-config", true, "when wrapping Codex, inject per-run -c model_provider/model_providers.fak overrides so Codex talks to the in-process gateway over the Responses wire. Codex-only; pass --codex-config=false if you already configured the fak provider yourself.")
	codexHome := fs.String("codex-home", "", "Codex home directory for `codex login` auth.json when wrapping Codex (default: $CODEX_HOME or ~/.codex). Used only for ChatGPT-subscription OAuth; --api-key-env keeps API billing explicit.")
	codexLoopGate := fs.String("codex-loop-gate", dispatchCodexLoopGateDefaultThreshold(), "Codex-only opt-in launch gate: audit recent Codex sessions before wrapping a Codex child and refuse at threshold loop|action, or use off (default: $FLEET_CODEX_LOOP_GATE, else off)")
	codexLoopGateSinceHours := fs.Float64("codex-loop-gate-since-hours", 24, "with --codex-loop-gate, only scan Codex sessions modified within N hours (0 = all)")
	codexLoopGateLimit := fs.Int("codex-loop-gate-limit", 20, "with --codex-loop-gate, maximum newest Codex sessions to scan")
	mcpRegister := fs.Bool("mcp-register", true, "register fak's own runtime MCP self-query surface (fak_capabilities, fak_feature_query, fak_memory_*, fak_tools_search) into the wrapped Claude Code child by default, via a session-scoped --mcp-config pointing at this gateway's /mcp endpoint. Claude-only; ADDS to any project/user MCP config the child already loads, never replaces it. Every call is still re-adjudicated by the guard floor — this widens discovery, not the danger floor. Pass --mcp-register=false if you already supply your own MCP config.")
	piExtension := fs.Bool("pi-extension", true, "when wrapping Pi (earendil-works), prepend a session-scoped -e extension that calls pi.registerProvider(\"anthropic\", {baseUrl}) so Pi talks to the in-process gateway. Pi-only; Pi's Anthropic client reads baseUrl from provider config, not ANTHROPIC_BASE_URL, so the env repoint alone cannot route it. Pass --pi-extension=false if you already registered the fak provider yourself.")
	managedCacheMode := fs.String("managed-cache", guardManagedCacheAuto, "actively manage the provider prompt-cache on the outbound Anthropic wire: auto|on|off (epic #1844 C6). ACTIVE upgrades the stable-prefix cache_control breakpoint to Anthropic's 1h TTL tier, so a long session that idles past the default 5m cache window (a human stepping away, a slow tool, a rate-limit stall) re-enters on a 0.1x cache READ instead of re-writing the whole prefix; the upgrade is byte-safe (only an existing stable system/tools-head breakpoint is extended, volatile heads refused) and witnessed on /metrics as fak_gateway_cache_ttl_upgrade_total. AUTO (default) activates ONLY when this session provably bills an API key (--api-key-env resolved a key on the Anthropic wire) — there the 2x one-time 1h write premium vs repeated 1.25x prefix re-writes is the operator's own dollars; a subscription-OAuth or passthrough session stays passive. on forces it; off disables.")
	compress := fs.Bool("compress", false, "activate the native context-compressor for this session: shrink benign tool results (ANSI/control strip, CR-redraw collapse, duplicate-line fold, JSON minify) before they enter model context, only when the saving clears the worth-it floor and never on poison, with the original preserved (reversible). Equivalent to FAK_COMPRESSOR=native for this process; an explicit FAK_COMPRESSOR wins. See `fak headroom bench` for the savings and `fak headroom status` for the live decision breakdown.")
	fleetBus := newGuardFleetBusFlag()
	fs.Var(&fleetBus, "fleet-bus", "JOIN THE FLEET CONTROL BUS (#5953, epic #5599), on by default. Use off for a total opt-out, or compose its bounded settings as on,dir=DIR,id=NAME,interval=DURATION. A guard can apply pause/resume/cancel/terminate/throttle and seat-refresh; steer remains unsupported because guard owns no session loop.")
	guardHelpAll := guardArgvHasAll(argv)
	fs.Usage = func() { printGuardUsage(os.Stderr, fs, commandName, guardHelpAll) }
	argv = rewriteLegacyDenyAllArgs(argv)
	var fleetBusRewriteErr error
	argv, fleetBusRewriteErr = rewriteLegacyGuardFleetBusArgs(argv)
	if fleetBusRewriteErr != nil {
		fmt.Fprintln(os.Stderr, "fak guard:", fleetBusRewriteErr)
		os.Exit(2)
	}
	_ = fs.Parse(argv)
	if err := validateNativeQwenQ4KPrefillChunk(*guardNativeFlags.prefillChunk); err != nil {
		fmt.Fprintln(os.Stderr, "fak guard:", err)
		os.Exit(2)
	}
	guardNativeConfig := guardNativeFlags.config()
	if *childResourcePoll < 100*time.Millisecond {
		fmt.Fprintln(os.Stderr, "fak guard: --child-resource-poll must be at least 100ms")
		os.Exit(2)
	}
	setGuardResourceConfig(guardResourceConfig{MaxMemoryMB: *childMaxMemoryMB, PollInterval: *childResourcePoll, ReceiptPath: *childResourceJournal})
	launchPlan := newGuardLaunchPlan(fs.Args())
	setLaunchToolGrant(allowTools)
	rotateSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "rotate" {
			rotateSet = true
		}
	})
	resolvedRotateMode, rotateErr := normalizeGuardRotateMode(*rotateMode, rotateSet, launchPlan.interactive())
	if rotateErr != nil {
		fmt.Fprintln(os.Stderr, rotateErr)
		os.Exit(2)
	}
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

	// Floor-aware compaction budget for EVERY guard launch (resolveGuardCompactBudget): a
	// Claude-Code system+tools floor sits at/above the lean 48k default, so TOTAL resident
	// (floor + conversation) is permanently past the budget — tripping the per-turn inversion
	// nudge (debug_stats) and the fleet's compact-runaway spawn-hold. On a headless worker the
	// cut then correctly bails under_budget (only the post-floor suffix is sheddable); on the
	// head-anchored interactive path the whole array is compactible, so it instead re-fires every
	// turn for a trivial shed and emits a `[fak] compacted` stub each time (#4888). Sizing keys
	// off the floor the launch carries, NOT the expose profile — an explicit operator budget wins.
	*compactHistoryBudget = resolveGuardCompactBudget(
		*compactHistoryBudget, guardSetFlags["compact-history-budget"])
	guardTraceID := strings.TrimSpace(*sessionID)
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
		os.Exit(runGuardReplay(*replayTrace, *replayWire, *policyPath, *auditPath, false, os.Stdout))
		return
	}

	command := launchPlan.executableCommand() // everything after the flags (and after `--`) is the wrapped agent.
	profilesExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "output-profile" || f.Name == "work-profile" {
			profilesExplicit = true
		}
	})
	command, responseProfileCapture, profileErr := injectGuardProfiles(command, *outputProfile, *workProfile, profilesExplicit)
	if profileErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: %v\n", profileErr)
		os.Exit(2)
	}
	if len(command) == 0 {
		fs.Usage()
		os.Exit(2)
	}
	launchPlan = launchPlan.withExecutableCommand(command)
	if code := runGuardStoragePressureGate(os.Stderr, defaultGuardStoragePressureDeps()); code != 0 {
		os.Exit(code)
	}
	agentName := launchPlan.agentName()
	if cfg, ok := guardCodexLoopGateConfigForProfile(launchPlan.harnessProfile(), launchPlan.executableCommand(), *codexLoopGate, *codexHome, *codexLoopGateSinceHours, *codexLoopGateLimit, *quiet); ok {
		if code := runCodexLoopGate(os.Stderr, cfg); code != 0 {
			os.Exit(code)
		}
	}
	sessionPressure, specErr := parseGuardSessionPressureSpec(*sessionPressureGate)
	if specErr != nil {
		fmt.Fprintf(os.Stderr, "fak guard: --session-pressure-gate %q: %v\n", *sessionPressureGate, specErr)
		os.Exit(2)
	}
	if code := runGuardSessionPressureGate(os.Stderr, guardSessionPressureGateConfig{
		Threshold:     sessionPressure.Threshold,
		SinceDays:     sessionPressure.SinceDays,
		Max:           sessionPressure.Max,
		Quiet:         *quiet,
		ReportPath:    sessionPressure.ReportPath,
		LaunchModel:   *model,
		Justification: sessionPressure.Justification,
	}); code != 0 {
		os.Exit(code)
	}

	// Cooldown-aware seat selection: a bare `fak guard -- claude` (no --rotate) resolves its
	// account purely from the environment and, unlike `fak accounts launch --rotate`, never
	// consults the fleet-shared cooldown store — so it would launch against an account the
	// launcher just watched bounce off its own weekly/usage cap, burning a turn on a walled
	// seat. Only meaningful on the subscription-OAuth Anthropic path: an explicit --api-key-env
	// is API billing (one key, no rotation) and a non-Claude child has no Claude seat to rotate.
	// When the currently-resolved seat is actively cooled and a live alternate exists, redirect
	// $CLAUDE_CONFIG_DIR to it BEFORE resolveGuardUpstream and the child spawn, so every
	// downstream consumer (fak's own OAuth read, the failover seed, the cap-recovery transcript
	// path, and the child's inherited env — all of which re-read the env var) follows to the live
	// seat. Fail-open: any doubt leaves the resolved dir untouched (see guardRotateOffCooldown).
	if provResolved, _ := launchPlan.resolveProvider(*provider); provResolved == "anthropic" && strings.TrimSpace(*apiKeyEnv) == "" {
		guardHomeDir, _ := os.UserHomeDir()
		if newDir, rotated := guardRotateOffCooldown(guardHomeDir, guardDefaultAccountsRegistryPath(guardHomeDir), time.Now(), guardRotateWarnWriter(os.Stderr, *quiet)); rotated {
			_ = os.Setenv("CLAUDE_CONFIG_DIR", newDir)
		}
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
		cmdGuardStdinInteractive(), launchPlan.interactive())

	// Startup-banner verbosity: resolve --banner now, fail-loud on a bad value before
	// any gateway binds. AUTO/empty selects the private delayed-progress-only mode for
	// both interactive and noninteractive launches. See guard_banner.go.
	bannerMode, bannerErr := guardBannerModeDecision(*bannerFlag, *quiet, cmdGuardStdinInteractive(), launchPlan.interactive())
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

	// Key the session-scope allow overlay to THIS launch (#5417), then arm its drop (#5180) —
	// both BEFORE the floor reads the overlay layers. The order is load-bearing three ways:
	// key before arm, or the armed path is the shared fallback every other guard also resolves;
	// arm before the floor, or the armed path is not the one the floor read; and BOTH before
	// loadGuardCapabilityFloor, because that is where protectGuardPolicyConfig write-protects
	// the layer paths — protecting a path resolved under a different id than the one read would
	// hand the wrapped agent a writable overlay over its own permissions. guardTraceID here is
	// only the explicit --session-id (or ""), used as a legible file-name base; it is NOT the
	// resolved trace, which is finalized ~220 lines below and collapses to the constant "guard"
	// on an ordinary launch. The matching drop runs in finishGuardChildAndReport's terminal
	// region, and the id reaches the child via $FAK_GUARD_SESSION_ID — see guard_allow_scope.go.
	setGuardAllowSessionScopeID(guardAllowSessionScopeLaunchID(guardTraceID))
	armGuardAllowSessionScopeTeardown()

	// 1. Install the capability floor: an explicit --policy file wins; otherwise the embedded
	//    guard floor, unioned with the operator allow overlay. With NO floor the kernel
	//    default-denies every tool, so guard ALWAYS loads one, fail-loud. See guard_startup.go.
	effectiveResponseProfile, effectiveWorkProfile := "full", "standard"
	if responseProfileCapture != nil {
		effectiveResponseProfile = responseProfileCapture.OutputProfile
		effectiveWorkProfile = responseProfileCapture.WorkProfile
	}

	rt, floorSource, policyDigest, policyDur := loadGuardCapabilityFloor(*policyPath)
	configureGuardPromotionLedger(rt.Adjudicator.Complain, guardPromotionDefaultThreshold)
	var err error

	// 1b. Default the durable DECISION JOURNAL on. fak guard is the disinterested
	//     referee; a tamper-evident, hash-chained record of every verdict is what
	//     lets an auditor confirm after the fact what the kernel allowed and blocked
	//     — the witness that makes the refereeing checkable rather than self-reported.
	//     So it is ON by default (announced in the banner, not silent). The kernel's
	//     EvDecide/EvDeny events on the proxy adjudication path are exactly what the
	//     journal records, so a guard session produces a populated ledger. Precedence:
	//     FAK_AUDIT_JOURNAL honored at boot wins; --audit off disables;
	//     --audit PATH or a repo-local per-session path otherwise. Enable BEFORE serving so
	//     the emitter is registered before the first decision crosses the floor.
	auditLabel, auditJournal := guardEnableAudit(*auditPath, false)
	var auditSeq0 uint64
	var refusalCarryForward []guardRefusalCarry
	if auditJournal != nil {
		auditSeq0, _, _ = auditJournal.Stats()
		refusalCarryForward = guardReadPriorRefusalCarryForward(auditJournal.Path(), guardTraceID, guardFindReasonRoot())
	}

	// 2. --remote-serve sugar: run the guarded turn's INFERENCE on a lab box you chose.
	//    It is a one-name shorthand for the informal "treat a remote fak serve as a
	//    provider URL" chain (`--provider openai --base-url http://HOST:PORT/v1`): it
	//    forces the OpenAI-compatible wire and sets the base URL to the box, so the kernel
	//    still adjudicates locally while the model runs on the lab GPU. Resolve + validate
	//    it BEFORE binding anything so a typo or a down box fails loud, not mid-session.
	remoteBase, remoteErr := resolveGuardRemoteServe(*remoteServe)
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
	// Qwen3.6 dogfood/llama.cpp) and wire the upstream to it. This is a PROXY path (the server
	// is external), so on detection we set provider=openai + base-URL=<detected>/v1 exactly
	// as if the user had typed those flags, and the standard resolution flow below handles it.
	// Precedence:
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
		extraApplied, extraAlreadySet, _, extraErr := guardApplyLocalProviderExtraBody(detLabel, *model, os.Getenv, os.Setenv)
		if extraErr != nil {
			fmt.Fprintf(os.Stderr, "fak guard: --local could not apply Qwen3.6 provider tuning: %v\n", extraErr)
			os.Exit(2)
		}
		if !*quiet {
			fmt.Fprintln(os.Stderr, guardLocalDetectedBanner(detLabel, detBase, detModel))
			switch {
			case extraApplied:
				fmt.Fprintln(os.Stderr, "-> local tuning: Qwen3.6 provider extra body enabled (top_k=20, preserve_thinking=true)")
			case extraAlreadySet:
				fmt.Fprintln(os.Stderr, "-> local tuning: using existing FAK_PROVIDER_EXTRA_BODY_JSON")
			}
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

	// 3. Resolve the upstream wire + credential posture: LOCAL-ONLY (--gguf without
	//    --alongside, where fak IS the upstream and there is no credential at all) vs the
	//    PROXY default (provider + base URL + API key, the Claude subscription-OAuth
	//    default, the per-request token re-resolver a long session needs, and the
	//    account-failover controls the gateway and the live status area consult later).
	//    The whole two-world resolution lives in guard_upstream_posture.go; a credential
	//    the wrapped agent could never repair still fails the launch there, before any
	//    spawn, exactly as it did inline.
	tUpstream := time.Now()
	launchProvider, launchProviderAutodetected := launchPlan.resolveProvider(*provider)
	if remoteBase != "" {
		launchProviderAutodetected = false
	}
	posture := resolveGuardUpstreamPosture(guardUpstreamPostureInputs{
		command:        command,
		profile:        launchPlan.harnessProfile(),
		provider:       launchProvider,
		baseURL:        *baseURL,
		remoteBase:     remoteBase,
		apiKeyEnv:      *apiKeyEnv,
		anthropicOAuth: *anthropicOAuth,
		oauthTokenEnv:  *oauthTokenEnv,
		model:          *model,
		codexHome:      *codexHome,
		quiet:          *quiet,
		localModel:     localModel,
		localAlongside: localAlongside,
	})
	upstreamResolveDur = time.Since(tUpstream)
	up, providerAutodetected, resolvedBase := posture.up, launchProviderAutodetected, posture.resolvedBase
	apiKey, pinUpstream, oauthSource := posture.apiKey, posture.pinUpstream, posture.oauthSource
	keychainAPIKey, credPath := posture.keychainAPIKey, posture.credPath
	apiKeyFunc, extraHeaders, extraHeadersFunc := posture.apiKeyFunc, posture.extraHeaders, posture.extraHeadersFunc
	accountFailoverFunc := posture.accountFailoverFunc
	transientTargetFunc := posture.transientTargetFunc
	guardActiveAccountDir, guardWalledAccounts := posture.activeAccountDir, posture.walledAccounts
	guardAccountRehome := posture.accountRehome

	// Prompt-shrink lever WIRE admission (#5538, the `fak guard` sibling of #5493):
	// --compact-history-budget, --elide-stale-reads and --defer-cold-tools are each gated,
	// inside the gateway, on the Anthropic passthrough (Server.anthropicPassthroughFor), so
	// on any other upstream wire all three stand down to identity — and all three ship
	// default-ON, which makes silently-inert the DEFAULT experience for a self-hosted guard
	// (--local forces provider=openai; --remote-serve forces the OpenAI-compatible wire; a
	// bare --gguf makes the in-kernel planner the upstream). Refuse by name when the operator
	// EXPLICITLY enabled one here, and name the default-on ones that are merely inert, so
	// "enabled but inert" is never silent and a ~0-saving A/B on this wire cannot be read as
	// a verdict on the kernel.
	//
	// Placed HERE — not beside the flag parse, where serve.go puts its twin — because guard's
	// raw --provider/--base-url are not its wire: they are auto-detected from the agent name
	// and rewritten by --local/--remote-serve, and an empty --base-url means the provider's
	// public API rather than serve's mock. Only the just-resolved posture (up, resolvedBase)
	// names the wire the gateway will actually build. Still ahead of every expensive stage:
	// the GGUF resolve/download and weight load are below, and nothing has bound yet
	// (shrink_lever_wire.go).
	if !admitGuardShrinkLevers(guardShrinkLeverInputs{
		SetFlags:             guardSetFlags,
		Provider:             up,
		BaseURL:              resolvedBase,
		GGUFPath:             *ggufPath,
		CompactHistoryBudget: *compactHistoryBudget,
		ElideStaleReads:      *elideStaleReads,
		DeferColdTools:       *deferColdTools,
	}, os.Stderr) {
		os.Exit(2)
	}

	// Managed-cache posture (epic #1844 C6): decide from the JUST-resolved upstream whether
	// this session actively manages the provider prompt-cache (the stable-prefix 1h TTL
	// upgrade). AUTO keys on provable API-key billing — the OAuth branch above rewrote
	// apiKey to the subscription token but stamped oauthSource, so the pair distinguishes
	// the two credentials. Fail loud on an unknown mode before any gateway binds.
	mcIn := guardManagedCacheInputs{
		// ALONGSIDE mode still has a real provider wire on the proxy side, so only the
		// PURE local branch (no upstream at all) turns the cache posture off.
		localModel:     localModel && !localAlongside,
		provider:       up,
		apiKey:         apiKey,
		oauthSource:    oauthSource,
		keychainAPIKey: keychainAPIKey,
	}
	mcache, mcErr := resolveGuardManagedCache(*managedCacheMode, mcIn)
	if mcErr != nil {
		fmt.Fprintln(os.Stderr, "fak guard:", mcErr)
		os.Exit(2)
	}
	// Publish the billing seat off the SAME resolved credential (#3664) so the Track-2
	// savings rows this session writes at exit record whether their list-priced dollars
	// were billed per token (API-key, real) or against a flat-rate subscription
	// (OAuth, notional). Without it the fleet reduction blends both seats into one
	// API-key-equivalent headline that the ledger cannot decompose.
	recordBillingMode(billingModeFrom(mcIn))

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
	// Same wording serve refuses a non-positive --native-admission-token-budget with:
	// a typo must fail loud at launch, never silently boot a seat whose scheduler
	// budget was not the one the operator declared.
	if *nativeAdmissionTokenBudget <= 0 {
		fmt.Fprintf(os.Stderr, "fak guard: --native-admission-token-budget must be positive (got %d)\n", *nativeAdmissionTokenBudget)
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
	// is used verbatim; an ordinary non-durable launch keeps the process-local "guard" id;
	// and an implicitly named durable launch gets a fresh host/cwd/argv-derived id. The fresh
	// suffix is load-bearing: reusing "guard" would let the registry restore a previous run's
	// STOPPED/TIME_BUDGET_EXHAUSTED state into a brand-new --max-duration launch.
	guardDurabilityWanted := guardSetFlags["session-id"] || contextBudgetLimit > 0 || maxDurationLimit > 0 || hasGuardBudgetEnvelope
	guardTraceID = resolveGuardSessionID(guardTraceID, guardDurabilityWanted, session.DescriptorMeta{
		CacheKey: sessionCacheKey(sessionDurabilityHost(), sessionWorkingDir(), "", command),
	}, newGuardLaunchNonce())
	if err := installGuardDevAttestation(guardTraceID, *policyPath); err != nil {
		fmt.Fprintln(os.Stderr, "fak guard: development lease attestation refused:", err)
		os.Exit(2)
	}
	maybeRecordGuardSessionIndex(auditJournal, guardTraceID, command, time.Now())
	// Record the operator-owned terminal session before its child starts. Normal
	// return appends a tombstone; WT/RDP teardown cannot run the defer, leaving an
	// actuator-ready crash row independent of the terminal host.
	if guardOwnsInteractiveTerminal() {
		cwd, _ := os.Getwd()
		row := guardsessions.NewInteractiveRow(guardTraceID, agentName, os.Getpid(), cwd, auditJournal.Path(), "", time.Now(), command, *hostRecovery)
		if err := recordInteractiveSessionRows(row); err != nil && !*quiet {
			fmt.Fprintf(os.Stderr, "fak guard: interactive session registry start: %v\n", err)
		}
		// #5400: this row is recorded BEFORE the listener binds, so it cannot carry the
		// gateway address yet. Register it for the post-bind stamp below; the republish goes
		// through the SAME recorder, so the machine control-plane mirror gets it too, and it
		// updates `row` in place so the deferred tombstone keeps the same fields.
		trackGuardSessionGatewayPublish(&row, recordInteractiveSessionRows)
		defer func() {
			if err := recordInteractiveSessionRows(row.Ended(time.Now())); err != nil && !*quiet {
				fmt.Fprintf(os.Stderr, "fak guard: interactive session registry exit: %v\n", err)
			}
		}()
	}
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
	restarter.seedHandback = *restartSeedHandback

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
		if err := applyNativeControls(chatBackend, guardNativeConfig); err != nil {
			fmt.Fprintln(os.Stderr, "fak guard:", err)
			os.Exit(2)
		}
		inKernelModel, inKernelQ4K, loadProfile, loadPhase = loadServeInKernelModel(*ggufPath, chatBackend, false, contextBudgetLimit, nil, 1)
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
	// #5400: mint THIS session's read-scoped observability bearer here, before the gateway is
	// built, because it is a gateway.Config field AND the token published with gwURL into the
	// guard-session index. The gateway honors it only on /healthz, /debug/vars and /metrics,
	// so the published credential can read this session's status and can never steer it.
	guardReadBearer := newGuardReadBearer()

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
	gatewayModel := guardCodexGatewayModelForProfile(launchPlan.harnessProfile(), *model, up)

	wireErrors := &guardWireErrorGauge{}
	parkStore := goalpark.Store{Dir: filepath.Join(repoRoot(), ".fak", "goal-park")}
	quotaStore := accountobs.Store{Dir: filepath.Join(repoRoot(), ".fak", "account-observations")}
	quotaKey := strings.TrimSpace(os.Getenv("FAK_ACCOUNT_ADMISSION_KEY"))
	if quotaKey == "" {
		quotaKey = "default"
	}
	quotaHarvester := accountobs.NewHarvester(quotaStore, quotaKey)
	parkGoal := strings.TrimSpace(os.Getenv("DISPATCH_GOAL"))
	if parkGoal == "" {
		parkGoal = strings.TrimSpace(os.Getenv("DISPATCH_LANE"))
	}
	parkTemplate := goalpark.Record{
		Goal: parkGoal, Lane: os.Getenv("DISPATCH_LANE"), Account: os.Getenv("DISPATCH_ACCOUNT"),
		Pool: os.Getenv("DISPATCH_POOL"), Lease: os.Getenv("DISPATCH_LEASE"),
		Witness: os.Getenv("DISPATCH_WITNESS_REQUIREMENT"), Command: append([]string(nil), command...),
	}
	observeUpstreamResponse := guardUpstreamObserver(quotaHarvester, parkStore, parkGoal, parkTemplate, os.Stderr)

	// Pin the compaction anchor mode this launch runs under for the durable per-session
	// compaction-health witness (#3152). Stamped here, next to the CompactAnchorHead the
	// gateway is about to be built with, so the row can never disagree with the server; the
	// exit funnel reads it when it writes the witness.
	setGuardCompactionAnchorMode(*compactAnchorHead)

	srv, err := gateway.New(gateway.Config{
		EngineID: "mock",
		Model:    gatewayModel,
		BaseURL:  resolvedBase,
		Provider: up,
		APIKey:   apiKey,
		// The child is the only caller of a normal guard's loopback gateway, so surface
		// the upstream's scrubbed, bounded 400 detail there. This turns an opaque fakc
		// "check model/roles/ranges" failure into the provider's exact rejected field
		// (the call_id-vs-item-id bug reported input[3].id). A guard explicitly bound
		// beyond loopback keeps the no-leak default, matching fak serve.
		ExposeUpstreamErrorDetail:      guardLoopbackOnly(ln.Addr().String()),
		UpstreamBadRequestNotify:       guardUpstreamBadRequestAuditNotify(auditJournal, guardTraceID),
		UpstreamResponseObserver:       observeUpstreamResponse,
		UpstreamTransportErrorObserver: func(err error) { wireErrors.Observe(time.Now(), err) },
		// Re-resolve the pinned subscription OAuth token per request so a long session
		// never sends the stale boot-time bearer (the 401-after-relogin bug). nil in every
		// non-pinned path leaves the static-APIKey behavior byte-for-byte unchanged.
		APIKeyFunc:       apiKeyFunc,
		ExtraHeaders:     extraHeaders,
		ExtraHeadersFunc: extraHeadersFunc,
		ForceResponsesStream: pinUpstream && up == "openai-responses" &&
			strings.TrimRight(resolvedBase, "/") == guardCodexChatGPTBackendBaseURL,
		// On an ACCOUNT-SCOPED 403 wall (org OAuth disabled / region / billing), fail over to a
		// permitted sibling account instead of surfacing a 403 that no re-login can fix. nil on
		// every non-pinned path (and when the home root is unresolvable), preserving the
		// historical terminal-on-account-403 behavior exactly.
		AccountFailoverFunc: accountFailoverFunc,
		TransientTargetFunc: transientTargetFunc,
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
		InKernelPlanner:       guardNativeConfig.Planner,
		LocalModelID:          localAlias,
		Backend:               chatBackend,
		PinUpstreamCredential: pinUpstream,
		RequireKey:            requireKey,
		// The read-scoped twin of RequireKey (#5400): accepted ONLY on the observability
		// reads, and published into the guard-session index so `fak session status <handle>`
		// can fetch this session's /debug/vars from another process with no port or key
		// knowledge — including on a guard bound beyond loopback, where the loopback
		// exemption does not apply and an operator would otherwise need the control key.
		ReadBearer:          guardReadBearer,
		VDSO:                true,
		Invalidation:        "global",
		Version:             appversion.Current(),
		ReloadPolicy:        guardPolicyReloader(*policyPath),
		ResetTrace:          resetTrace,
		ObserveTrace:        observeTrace,
		ObserveSession:      observeSession,
		ControlSession:      controlSession,
		SteerSession:        steerSession,
		ListSessions:        listSessions,
		DecideSession:       decideSession,
		DebitSession:        debitSession,
		ResetOnBudget:       resetOnBudgetHook(*resetOnBudget, contextBudgetLimit),
		OnBudgetExhausted:   restarter.OnBudgetExhausted,
		DefaultTraceID:      guardTraceID,
		GuardRecoveryPrompt: guardRecoveryPrompt(refusalCarryForward),
		StartTime:           t0,
		StartupPhases:       startupPhases,
		// Default OFF (clean terminal); --log routes the full structured stream to a file
		// or stderr. /metrics + /debug/vars + the audit journal carry the record regardless.
		Logf: gwLogf,
		// The observable debug layer (#793) is ON by default so the cache + token-value
		// economy of every turn is visible without a flag; --debug-stats=false or --quiet
		// silences it. The full JSON --log stream stays separate (and off by default).
		DebugStatsf:          debugStatsSink(debugStatsStderr),
		CtxViewBudget:        *ctxViewBudget,
		CompactHistoryBudget: *compactHistoryBudget,
		AutoCheckpoint: func(session, reason string) {
			_ = runWipAutoCheckpoint(io.Discard, io.Discard, []string{"--session", session, "--reason", reason})
		},
		CompactAnchorHead:          *compactAnchorHead,
		AssumeSessionTurns:         *assumeSessionTurns,
		CompactSolvencyFloorTokens: *compactSolvencyFloor,
		ElideResultBytes:           *elideResultBytes,
		ElideStaleReads:            *elideStaleReads,
		// Managed-cache posture (--managed-cache, epic #1844 C6): when active, the gateway
		// upgrades the stable-prefix cache_control breakpoint to the 1h TTL tier on the
		// outbound Anthropic wire (maybeUpgradeAnthropicCacheTTL1H). Resolved above from
		// the session's billing posture; witnessed on /metrics per turn.
		CacheTTL1H: mcache.active,
		// M2 star-anchor pre-flight gate (#1493): DEFAULT-ON (--vcache-anchor). On the
		// Anthropic passthrough, APPLY cachemeta.RecommendLayout before send — hoist volatile
		// system blocks behind a byte-stable anchor and splice a cache_control breakpoint the
		// caller did NOT send — DECOUPLED from CompactHistoryBudget so --compact-history-budget=0
		// no longer takes anchoring down with it. Fail-safe identity on any ambiguity.
		VCacheAnchor:      *vcacheAnchor,
		VCacheCalibration: loadVCacheRuntimeCalibration(up, gatewayModel),
		// Inbound twin of #555: prune tool DEFINITIONS the floor can never admit from the
		// Anthropic passthrough's tools[], cache-prefix-preserving. Default-ON because it is
		// behavior-preserving by construction (a pruned tool stays DEFAULT_DENY at the kernel),
		// so it only ever shrinks uncached tool-def tokens. The predicate is a pure read of the
		// installed floor (rt.Adjudicator.NeverAdmits): true only for a name no argument could
		// make Allowed. nil would disable it; we always supply it.
		ToolFloorDenies: rt.Adjudicator.NeverAdmits,
		// The 10x floor lever (--defer-cold-tools, #3232): defer the COLD tool tail
		// (defer_loading:true) and inject a tool_search_tool on the outbound Anthropic body,
		// so the provider loads only the hot core into context. DEFAULT ON
		// (gateway.DefaultDeferColdTools, the #3537 flip); --defer-cold-tools=false opts out,
		// FAK_ABLATE_DEFER_TOOLS=1 ablates the live seam, and gateway.New still ORs in
		// FAK_DEFER_COLD_TOOLS. Deterministic, cache-safe, fail-safe identity on any ambiguity.
		DeferColdTools: *deferColdTools,
		// Curated headless tool surface (#3607): a dispatch worker launches with --expose-profile
		// headless, so the in-kernel fak_* registry is pruned to the allowlist it actually uses
		// (the rest page in via fak_tools_search), trimming the ~9.9k-token full-registry floor.
		// nil for an interactive launch or the FAK_GUARD_EXPOSE_PROFILE=full/off opt-out → full.
		ExposeTools: resolveGuardExposeTools(*exposeProfile),
	})
	must(err)
	if loadProfile != nil {
		srv.SetModelLoadProfile(loadProfile)
	}
	// LOCAL in-kernel model (--gguf, #10597): install the operator's
	// --native-admission-token-budget on the managed gateway exactly as `fak serve`
	// does, so a harness whose first prompt exceeds the default 8192 (opencode 1.18's
	// ~45k-token floor) is admitted instead of 429-shed every turn. The condition
	// mirrors gateway.New's auto-attach (in-kernel model + tokenizer, no proxy
	// upstream): alongside/remote/base-url seats keep the prior behavior byte-for-byte.
	if inKernelModel != nil && inKernelTok != nil && strings.TrimSpace(resolvedBase) == "" {
		controller, admissionNote, err := newGuardNativeAdmissionController(commandName, *nativeAdmissionTokenBudget)
		must(err)
		srv.SetAdmissionController(controller)
		srv.AddStartupMessages(admissionNote)
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
	startGuardAllowWatcher(ctx, guardPolicyReloader(*policyPath), *quiet)
	fleetLogf := fleetspine.Logf(nil)
	// Compact/animated attended startup owns a hard pre-child output budget. Fleet
	// discovery is best-effort operational detail and remains visible in the full
	// startup report/logs; it must not independently scroll the child UI.
	if !*quiet && bannerMode == guardBannerFull {
		fleetLogf = func(format string, args ...any) {
			fmt.Fprintf(os.Stderr, "fak guard: "+format+"\n", args...)
		}
	}
	installGuardFleetProvider(srv, ctx, fleetLogf)
	// Arm the CONTROL bus (guard_fleetbus.go) — distinct from the display pane above:
	// the provider answers "what does this box look like", this answers "and here is
	// something that can be told to change". Armed here, beside the pane and BEFORE
	// Serve, because nothing this guard can apply depends on the gateway being healthy:
	// the lifecycle ops write the process-wide session table, and seat-refresh is
	// registry-and-disk work. Waiting for MarkReady would therefore hide a guard that
	// is genuinely applyable, and a control point that refuses FLEETBUS_NO_TARGET
	// against a booting fleet is correct about the roster and wrong about the world.
	// The other direction is bounded and honest: a guard that announces and then dies
	// before ready ages out of the roster within one TTL, and until it does it folds as
	// OUTSTANDING — "addressed, never answered" — never as an apply.
	// --fleet-bus=false makes this a total no-op.
	stopGuardFleetBus := startGuardFleetBus(ctx, fleetBus.enabled, fleetBus.dir, fleetBus.id, fleetBus.interval, bannerMode != guardBannerFull)
	defer stopGuardFleetBus()
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

	// PUBLISH this session's gateway into the guard-session index (#5400) — the producer
	// half of cross-process discovery. Placed HERE, after guardWaitHealthy + MarkReady,
	// because that is the first instant the address is both known and actually answering:
	// publishing at bind time would advertise a URL that 503s, and publishing later would
	// leave `fak session status` blind for the whole startup window. Best-effort by the same
	// contract as the launch append — a failed republish leaves the launch row intact (the
	// session simply stays unreachable cross-process) and never blocks the agent.
	if _, perrs := publishGuardSessionGateway(gwURL, guardReadBearer); len(perrs) > 0 && !*quiet {
		for _, perr := range perrs {
			fmt.Fprintf(os.Stderr, "fak guard: guard-session gateway publish: %v\n", perr)
		}
	}

	// Feed the live accounts+nodes status area (`fak info` / /debug/vars): which Claude
	// seats and which serving nodes THIS session uses. The provider is a pull source
	// (re-read per scrape) so a mid-session account failover's active/walled marks stay
	// live; the serving-node list is resolved once from the boot upstream posture. See
	// guard_endpoints.go. guardActiveAccountDir is nil on a non-subscription session, so
	// the accounts half is simply absent there while the nodes still render.
	srv.SetSessionEndpointsProvider(newGuardEndpointsProvider(guardActiveAccountDir, guardWalledAccounts, guardEndpointNodes{
		provider:     up,
		resolvedBase: resolvedBase,
		remoteServe:  remoteBase != "",
		localModel:   localModel,
		localAlong:   localAlongside,
		localAlias:   localAlias,
	}))

	// Operator seat switch (POST /v1/fak/account/rehome, `fak accounts rehome`): the
	// on-demand form of the account failover. Wired only when the failover state exists
	// (pinned Claude subscription with a resolvable home root); everywhere else the
	// route stays inert (404). See guard_account_failover.go / gateway/account_rehome.go.
	if guardAccountRehome != nil {
		srv.SetAccountRehomeFunc(guardAccountRehome)
	}

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
		// (#2052). VRAM PREFERS the same compute HAL the serve capacity checks use (the
		// in-kernel backend's own device handle); it falls back to nvidia-smi only on a host
		// where the handle cannot report — the fail-soft fallback the issue names.
		if chatBackend != nil {
			resSampler.SetGPUProvider(func() (used, total uint64, ok bool) {
				t, free, known := compute.DeviceMemoryInfo(chatBackend)
				var smi []compute.GPUStat
				if !known || t <= 0 {
					smi, _ = compute.NvidiaGPUStats() // fail-soft; nil → axis stays n/a
				}
				return compute.HarnessGPUVRAM(t, free, known, smi)
			})
			// Feed the GPU utilization axis. The in-kernel device-handle seam
			// (DeviceMemoryInfo) reports memory only — there is no utilization on it — so
			// this is the nvidia-smi fallback the issue names: per-device VRAM+util folded
			// to the busiest device's percent. Fail-soft (no nvidia-smi / timeout /
			// unparseable → ok=false), so the util axis stays honestly n/a rather than a
			// fabricated 0 on a host that lacks the tool (#2052).
			resSampler.SetGPUUtilProvider(func() (pct float64, ok bool) {
				stats, present := compute.NvidiaGPUStats()
				if !present {
					return 0, false
				}
				_, _, util, aok := compute.AggregateGPUStats(stats)
				return util, aok
			})
		}
		resSampler.Start(guardResourceSampleInterval)
		// Expose the live harness resource snapshot on the gateway's /metrics as the
		// fak_harness_* family, so a running session's CPU/mem/IO is scrapeable — not
		// only printed at exit (epic #2044 / #2047). Pull-only: rendered per scrape.
		srv.SetHarnessMetricsProvider(func() string { return resSampler.Snapshot().PrometheusText() })
		// Structured twin of the /metrics harness family, on /debug/vars, so the live `fak
		// info` pane can show the kernel CPU/RSS/IO the exit summary prints instead of only
		// scraping Prometheus text. Same pull sampler, converted to the gateway's shape.
		srv.SetSessionHarnessProvider(func() gateway.SessionHarness { return guardHarnessToSession(resSampler.Snapshot()) })
	}

	// Vault observability (#2455): expose the three fak_logvault_* gauges on the
	// gateway's /metrics when a capture vault exists on this box, so an operator can
	// see last-capture age, footprint, and verify mismatches where they already
	// scrape — and a forced verify mismatch surfaces within one capture cycle. Each
	// value is WITNESSED (folded from logvault's own hash-chained manifest + a
	// bounded mirror re-hash); pull-only, rendered per scrape. Wired only when a
	// vault manifest is present so boxes without a vault emit no phantom family.
	if vaultDir := resolveLogvaultDir(repoRoot()); vaultDir != "" {
		if _, statErr := os.Stat(filepath.Join(vaultDir, logvault.ManifestName)); statErr == nil {
			lv := &logvault.Vault{Dir: vaultDir}
			srv.SetLogvaultMetricsProvider(func() string {
				text, err := lv.MetricsText(logvaultMetricsVerifySample, time.Now().UnixNano())
				if err != nil {
					return "" // unreadable manifest: emit nothing rather than a broken family
				}
				return text
			})
		}
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
	if splitOn, splitErr := guardSplitEnabled(*splitMode, os.Getenv, cmdGuardStdinInteractive(), launchPlan.interactive()); splitErr != nil {
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
	lifecycleIPC, lifecycleErr := startGuardLifecycleServer(srv)
	if lifecycleErr == nil {
		defer lifecycleIPC.Close()
		injected = append(injected, lifecycleIPC.Env()...)
	} else {
		fmt.Fprintf(os.Stderr, "fak guard: lifecycle IPC unavailable; external hooks will use /metrics fallback: %v\n", lifecycleErr)
	}
	var preCompactInstall guardPreCompactInstall
	var preCompactEnv [][2]string
	installersStarted := time.Now()
	command, preCompactEnv, preCompactInstall, err = installGuardPreCompactHook(command, *preCompactHook, gwURL)
	if err != nil {
		abortChildWiring(cancel, "Claude PreCompact hook setup", err, 1)
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
		guardTaskHandoffEffectiveMode(*taskHandoffMode, guardSetFlags["task-handoff"], launchPlan.interactive(), *probeMode),
	)
	if err != nil {
		abortChildWiring(cancel, "task handoff setup", err, 2)
	}
	handoffFile := strings.TrimSpace(*taskHandoffFile)
	if handoffMode != guardPreCompactModeOff && handoffFile == "" {
		// Allocate through the shared creation seam so the dir carries this guard's PID
		// (#5527). "handoff" was already listed in guardTempDirHooks, but this call site
		// used a raw os.MkdirTemp whose name had no <pid> segment, so guardTempDirOwner
		// refused to claim it and the reaper buried none of them. Lifetime is unchanged:
		// the child reads this file for its whole run, so there is no setup-time defer to
		// remove it — a LATER guard's dead-owner sweep is what reclaims it.
		dir, err := guardSessionTempDir("handoff")
		if err != nil {
			abortChildWiring(cancel, "task handoff setup", err, 1)
		}
		handoffFile = filepath.Join(dir, "task-handoff.json")
	}
	handoffCfg := guardTaskHandoffConfig{Mode: handoffMode, File: handoffFile, Repo: *taskHandoffRepo, Live: *taskHandoffLive}
	// Resolve the operator-directed gate mode with the operator-absent cap: an attended interactive
	// child is capped so a human's genuine question can always end the turn; a headless child — or an
	// interactive child an orchestrator marked unattended (guardOperatorUnattended, #4951) — reaches
	// enforce. Same guardChildInteractive signal the task-handoff gate leans on above, plus the
	// operator-presence axis that tells an operator-driven interactive session from an attended one.
	operatorDirectedMode := guardOperatorDirectedEffectiveMode(*operatorDirected, guardSetFlags["operator-directed"], launchPlan.interactive(), guardOperatorUnattended())
	var stopHookInstall guardStopHookInstall
	var stopHookEnv [][2]string
	command, stopHookEnv, stopHookInstall, err = installGuardStopHook(command, denyAllSettings.Mode, gwURL, preCompactInstall.SettingsPath, denyAllSettings.Warn, denyAllSettings.Final, denyAllSettings.Max, denyAllSettings.SameStop, operatorDirectedMode, handoffCfg)
	if err != nil {
		abortChildWiring(cancel, "Claude Stop hook setup", err, 1)
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
	var toolprocInstall guardToolprocInstall
	command, toolprocHookEnv, toolprocInstall, err = installGuardToolprocHooks(command, *toolprocHooks, toolprocSettings)
	if err != nil {
		abortChildWiring(cancel, "toolproc hook setup", err, 1)
	}
	injected = append(injected, toolprocHookEnv...)
	if mode := toolcallcontrol.ParseMode(*toolcallControl); mode != toolcallcontrol.ModeOff {
		injected = append(injected, [2]string{"FAK_TOOLCALL_CONTROL_MODE", string(mode)})
	}
	// Discoverability affordance (#3092): a SessionStart hook that injects a one-line hint
	// naming fak's MCP entry verbs into the first turn, so the agent reaches past Claude
	// Code's deferred-tool wall instead of running as a generic coder. Merged into the SAME
	// --settings file the hooks above wrote. On by default; FAK_GUARD_AFFORDANCE_MODE=off opts
	// out for a lean harness.
	sessionStartSettings := guardSharedHookSettingsPath(toolprocInstall, stopHookInstall, preCompactInstall)
	// A headless/fleet worker (a `-p` child, not an attended TUI) is admitted onto the
	// long-horizon MANAGED posture: its SessionStart injection carries the persistence +
	// managed-context rule (#3512), where keep-going-past-a-long-window matters most and no
	// human is present to drive it. An attended interactive session gets the base affordance
	// only. Same headless signal the task-handoff gate leans on above (guardChildInteractive).
	sessionStartManaged := !launchPlan.interactive()
	// Thread the guard trace id into the hook argv so the running SessionStart hook holds both
	// ids and can record the A1 uuid<->trace identity join (#4112/#4113).
	var sessionStartInstall guardSessionStartInstall
	command, sessionStartInstall, err = installGuardSessionStartHookForProfile(command, launchPlan.harnessProfile(), os.Getenv(guardSessionStartEnvMode), sessionStartManaged, sessionStartSettings, guardTraceID)
	if err != nil {
		abortChildWiring(cancel, "provider SessionStart hook setup", err, 1)
	}
	// First-class `fak guard -- codex`: Codex reads custom upstreams from `-c`
	// provider overrides, not OPENAI_BASE_URL. Repoint only Codex children, after the
	// Claude-specific hook installers have had a chance to no-op.
	command, codexInstall := installGuardCodexConfigForProfile(command, launchPlan.harnessProfile(), *codexConfig, gwURL, *apiKeyEnv, *codexHome)
	launchPlan = launchPlan.withExecutableCommand(command)
	if codexInstall.Applied && pinUpstream && up == "openai-responses" && strings.TrimSpace(oauthSource) != "" {
		codexInstall.AuthMode = "chatgpt"
		codexInstall.AuthSource = oauthSource
	}
	codexAuthEnv, codexAuthErr := guardCodexAuthEnv(codexInstall, apiKey, localModel && !localAlongside, os.Getenv)
	if codexAuthErr != nil {
		cancel()
		fmt.Fprintln(os.Stderr, "fak guard:", codexAuthErr)
		os.Exit(2)
	}
	injected = append(injected, codexAuthEnv...)
	// First-class `fak guard -- pi`: Pi (earendil-works) speaks the Anthropic wire but its
	// client reads baseUrl from provider config, not ANTHROPIC_BASE_URL, so the env repoint
	// cannot route it. Prepend a session-scoped `-e` extension that registers the anthropic
	// provider at the gateway. Pi-only; no-ops for every other child. See guard_pi.go.
	command, piInstall, err := installGuardPiExtension(command, *piExtension, gwURL)
	if err != nil {
		abortChildWiring(cancel, "Pi extension setup", err, 1)
	}
	injected = append(injected, guardClaudeAutoCompactWindowInjection(up, *model, command)...)
	// Headless workers: make editor/pager-opening git forms (a `git commit` with no message
	// source, incl. a `-m` after `--`; `git rebase -i`) fail fast instead of hanging on a
	// TTY-less $EDITOR — the top recurring stall in the trajectory audit (#2365). No-op for an
	// attended interactive child; never overrides an inherited value. See guard_provider.go.
	injected = append(injected, guardGitNonInteractiveEnv(command, os.Getenv)...)
	// Carry THIS launch's session-scope allow id into the child (#5417), so an in-session
	// `fak guard allow --session <tool>` writes the overlay this guard actually reads and drops
	// — and so the child cannot inherit an outer guard's id and write into a live peer
	// session's layer instead. Appended late on purpose: both spawn paths take the last value
	// bound to a name, so this overwrites any ambient one. See guard_allow_scope.go.
	injected = append(injected, guardAllowSessionScopeChildEnv())
	// Live discovery (#1499): register fak's runtime self-query/memory/search surface
	// MCP tools into the wrapped Claude child by default, so a default `fak guard --
	// claude` session can reach them with no manual .mcp.json setup.
	srv.RecordStartupPhase("guard-installers", time.Since(installersStarted), "measured")
	mcpStarted := time.Now()
	command, mcpInstall, err := installGuardMCPRegistration(command, *mcpRegister, gwURL)
	srv.RecordStartupPhase("mcp-registration", time.Since(mcpStarted), "measured")
	if err != nil {
		cancel()
		fmt.Fprintf(os.Stderr, "fak guard: Claude MCP registration setup failed: %v\n", err)
		os.Exit(1)
	}

	// Render the FULL startup report and register it on the gateway so the session serves it
	// back on demand for its whole life (`fak info --startup` / startup_report on /debug/vars),
	// then spill to the terminal RIGHT NOW only what bannerMode asks for. See guard_startup.go.
	view := guardStartupView{
		providerAutodetected: providerAutodetected,
		up:                   up,
		command:              command,
		gwURL:                gwURL,
		resolvedBase:         resolvedBase,
		remoteBase:           remoteBase,
		floorSource:          floorSource,
		policyDigest:         policyDigest,
		injected:             injected,
		responseProfile:      effectiveResponseProfile,
		workProfile:          effectiveWorkProfile,
		logLabel:             logLabel,
		auditLabel:           auditLabel,
		refusalCarryForward:  refusalCarryForward,
		localModel:           localModel,
		ggufPath:             *ggufPath,
		preCompactInstall:    preCompactInstall,
		stopHookInstall:      stopHookInstall,
		handoffCfg:           handoffCfg,
		codexInstall:         codexInstall,
		piInstall:            piInstall,
		mcpInstall:           mcpInstall,
		debugStatsStderr:     debugStatsStderr,
		debugStats:           *debugStats,
		quiet:                *quiet,
		pinUpstream:          pinUpstream,
		apiKey:               apiKey,
		apiKeyEnv:            *apiKeyEnv,
		keychainAPIKey:       keychainAPIKey,
		oauthSource:          oauthSource,
		mcache:               mcache,
		contextBudgetLimit:   contextBudgetLimit,
		guardTraceID:         guardTraceID,
		resetOnBudget:        *resetOnBudget,
		restartOnBudget:      *restartOnBudget,
		maxDurationLimit:     maxDurationLimit,
		auditJournal:         auditJournal,
		bannerMode:           bannerMode,
		upstreamTrustNote:    posture.upstreamTrustNote,
		cloudRouteWaived:     posture.cloudRouteWaived,
	}
	startupReport := renderGuardStartupReport(view)
	srv.SetStartupReport(startupReport)
	emitGuardStartupBanner(view, startupReport)
	startupProgress := newGuardStartupProgress(os.Stderr, !*quiet && bannerMode != guardBannerOff, guardFdIsTerminal(int(os.Stderr.Fd())), guardStartupProgressDelay)
	defer startupProgress.Stop()

	// Seed the guard child into the opt-in process task manager at origin, with the
	// durable policy/budget/Stop-ledger paths that already exist before launch.
	policyEvidence := guardPolicyOriginEvidencePath(guardTraceID, *policyPath)
	budgetEvidence := writeGuardBudgetEnvelopeEvidence(guardTraceID, contextBudgetLimit, maxDurationLimit.String())
	registerGuardChildOriginTask(guardTraceID, agentName, policyEvidence, resume.IdentityLedgerPath(resolveSweepRegDir("")), budgetEvidence, stopHookInstall.StopsLedger)
	startupProgress.Phase("native-hook install")
	command, restoreNativeHooks, err := installManagedNativeHooksForProfile(command, launchPlan.harnessProfile())
	if err != nil {
		startupProgress.Abort()
		fmt.Fprintf(os.Stderr, "fak %s: install native hooks: %v\n", commandName, err)
		cancel()
		return
	}
	defer restoreNativeHooks()
	launchPlan = launchPlan.withExecutableCommand(command)

	// 6. Run the wrapped agent, then tear the gateway down and report the session.
	rotationRuntime := guardRotationRuntimeForProfile(launchPlan.harnessProfile(), resolvedRotateMode)
	spawnMeta := newGuardChildSpawnMetadata(guardTraceID, policyDigest, up, rt, launchPlan)
	// On a genuine launch FAILURE, spill the full startup report to stderr — except under
	// --banner=full, which already streamed it at boot (avoid printing it twice).
	dumpStartupOnLaunchFail := guardDumpStartupOnLaunchFail(bannerMode)
	startupProgress.Phase("lease admission")
	var arbitrateStderr strings.Builder
	arbitrateLease, arbitrateErr := guardArbitrateAcquire(context.Background(), &arbitrateStderr, guardArbitrateConfig{
		Mode: arbitrateConfig.Mode, Lane: arbitrateConfig.Lane, Tree: arbitrateConfig.Tree, Force: arbitrateConfig.Force,
	})
	if arbitrateStderr.Len() > 0 {
		startupProgress.EndLine()
		fmt.Fprint(os.Stderr, arbitrateStderr.String())
	}
	if arbitrateErr != nil {
		startupProgress.Abort()
		fmt.Fprintf(os.Stderr, "fak guard: %v\n", arbitrateErr)
		os.Exit(1)
	}
	if arbitrateLease != nil {
		defer arbitrateLease.Close()
	}
	startupProgress.Phase("broker/preparing child")

	// The supervised loop is required whenever the child must be interrupted mid-run:
	// a --restart-on-budget context restart OR a --max-duration wall-clock envelope that
	// must be ENFORCED (#2229). A --max-duration-only run routes here with a disabled
	// restarter (its events channel never fires), gaining only the time-budget ticker.
	if restarter.Enabled() || maxDurationLimit > 0 {
		runGuardChildSupervisedAndReport(command, injected, pinUpstream, credPath, &rotationRuntime, spawnMeta, sessionStartInstall.StatePath, restarter, wireErrors, srv, cancel, serveErr, *quiet, auditJournal, auditSeq0, guardTraceID, agentName, up, *dojoMode, resSampler, dumpStartupOnLaunchFail, startupProgress)
		return
	}
	runGuardChildAndReport(command, injected, pinUpstream, credPath, &rotationRuntime, spawnMeta, sessionStartInstall.StatePath, wireErrors, srv, cancel, serveErr, *quiet, auditJournal, auditSeq0, guardTraceID, agentName, up, *dojoMode, resSampler, dumpStartupOnLaunchFail, startupProgress)
}
