package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/callavoid"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/headroom"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/secretload"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

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
	anthropicOAuth := fs.Bool("anthropic-oauth", false, "force the Claude Pro/Max SUBSCRIPTION OAuth token upstream (sourced from CLAUDE_CODE_OAUTH_TOKEN, <claude-config>/.oauth-token, or ~/.claude/.credentials.json) sent as Authorization: Bearer + the oauth beta. This is ALREADY the default for --provider anthropic (even when ANTHROPIC_API_KEY is set); the flag forces it and fails loud if no token is found.")
	oauthTokenEnv := fs.String("oauth-token-env", "CLAUDE_CODE_OAUTH_TOKEN", "env var to read the subscription OAuth token from first")
	policyPath := fs.String("policy", "", "capability-floor manifest to enforce (default: the built-in guard floor; see --dump-policy)")
	envName := fs.String("env", "", "env var to inject the gateway URL into the child (default: chosen by --provider)")
	requireKeyEnv := fs.String("require-key-env", "", "require this env var's bearer token on the gateway (loopback rarely needs it)")
	logPath := fs.String("log", "", "write the gateway's per-request + per-verdict structured logs to this file (or '-' for stderr); default off to keep the agent's terminal clean")
	auditPath := fs.String("audit", "", "write the durable, hash-chained DECISION JOURNAL to this file (default: a per-user path under your config dir; pass 'off' to disable). Every kernel verdict this session is appended as a tamper-evident JSONL row you can later replay with `fak audit verify`.")
	noAudit := fs.Bool("no-audit", false, "disable the durable decision journal for this session (it is ON by default — fak guard is the referee, and the journal is the verifiable record of what it allowed vs blocked)")
	dumpPolicy := fs.Bool("dump-policy", false, "print the built-in guard capability floor (an editable manifest) and exit")
	quiet := fs.Bool("quiet", false, "suppress the startup banner and the exit audit summary")
	debugStats := fs.Bool("debug-stats", true, "ON by default — the observable debug layer: print ONE compact, payload-free line per served turn to stderr with the turn's cache + token-value economy (request_tokens/cache_read/cache_creation, cache_hit, cache_rebate_tokens), the SAFETY half (blocked:/repaired:/quarantined: with the dominant reason whenever the kernel refused, rewrote, or paged out a call THIS turn — so a refused rm -rf or a quarantined secret is visible the moment it happens, not only in the exit summary), the compaction action, and the resetScore SHADOW health (healthy_cache|cache_decay|stale_prefix|cooldown|unknown_provider). These counts are the provider's own usage numbers, so it works natively over your Claude subscription OAuth. Independent of --log; pass --debug-stats=false or --quiet to silence it (#793).")
	preCompactHook := fs.String("precompact-hook", guardPreCompactModeShadow, "Claude Code PreCompact hook actuator for auto-compaction: off|shadow|enforce. shadow logs would-block/would-allow while exiting 0; enforce returns the compactcohere posture exit code.")
	denyAllContinue := fs.String("deny-all-continue", guardPreCompactModeEnforce, "Claude Code Stop hook that auto-RESUMES the agent after a deny-all turn (the floor refused EVERY proposed tool call, which the wire reports as end_turn — a stop the agent did not choose): off|shadow|enforce. ENFORCE by default (the false-stop fix), bounded by --deny-all-max consecutive continues; shadow logs the would-continue while letting the turn end; off disables. Claude children only.")
	denyAllMax := fs.Int("deny-all-max", guardStopHookDefaultMax, "with --deny-all-continue=enforce: the maximum number of CONSECUTIVE deny-all turns to auto-continue past before giving up and letting the turn end, so a model that keeps re-proposing refused calls cannot loop forever.")
	ctxViewBudget := fs.Int("ctx-view-budget", 8000, "wire the ctxplan context PLANNER into the live guard loop: each buffered turn, re-materialize the forwarded history as an O(1) planned VIEW under this resident-token budget (a planned view in place of appending the whole transcript, #555). DEFAULT-ON at a conservative 8000 resident tokens; pass 0 to disable (leaves the existing path byte-for-byte unchanged). The planner only ever SHORTENS and falls open to the full history on any doubt; on the Anthropic passthrough it keeps the cached prefix byte-identical (witness: docs/notes/CTXVIEW-DEFAULT-ON-WITNESS-2026-06-28.md). The streaming fast-path bypasses this; the buffered turn path is what gets planned.")
	compactHistoryBudget := fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "compact OLD conversation turns in the OUTBOUND Anthropic request body down to this resident-token budget while keeping the cache_control prefix BYTE-IDENTICAL, so the upstream prompt-cache hit survives. This reaches the flagship `fak guard -- claude` passthrough (where the body is forwarded verbatim, #555). DEFAULT-ON: once a wrapped conversation sprawls past ~48k resident tokens the cut fires and sheds the un-cacheable middle the provider re-bills every turn; a typical short session stays untouched. Pass 0 to disable (body forwarded byte-for-byte). Anthropic passthrough only.")
	elideResultBytes := fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes, "ON by default at gateway.DefaultElideResultBytes (the reviewed gateway.DocumentedElideResultBytes threshold): shrink oversized tool_result bodies outside the active working set to a bounded head+tail form once they exceed this byte threshold. 0 disables.")
	sessionID := fs.String("session-id", "guard", "default trace/session id for wrapped agents that omit X-Trace-Id or MCP trace_id")
	contextBudgetTokens := fs.Int("context-budget-tokens", 0, "seed the guard session with this prompt/context-token budget; exhaustion returns a reset directive with continuation_id (0 = off)")
	resetOnBudget := fs.Bool("reset-on-budget", false, "on context-budget exhaustion, re-arm the continuation trace with a carryover seed and continue transparently instead of returning 409 (requires --context-budget-tokens)")
	restartOnBudget := fs.Bool("restart-on-budget", false, "on context-budget exhaustion, stop and relaunch the wrapped child under the continuation trace, writing a carryover seed JSON and exposing it via FAK_RESET_* env vars (requires --context-budget-tokens)")
	restartLimit := fs.Int("restart-limit", 0, "maximum child relaunches for --restart-on-budget; 0 means unlimited")
	restartSeedDir := fs.String("restart-seed-dir", "", "directory for --restart-on-budget carryover seed JSON files (default: OS temp dir, one private directory per reset)")
	landlockHooks := fs.Bool("landlock-hooks", false, "LINUX-ONLY defense-in-depth: run the spawned agent under a Landlock profile that makes the git hook surface (.git/hooks + core.hooksPath) READ-ONLY while the rest of the tree stays writable, so a laundered write cannot drop an executable hook. OFF by default; fails OPEN (logs + spawns unrestricted) on a kernel without Landlock or on a non-Linux host. Also settable via "+guard.EnvOptIn+"=1.")
	dojoMode := fs.Bool("dojo", false, "enable live dojo mode: write a start-marker for this guard session into the live-episode corpus (.dojo/live-episodes/ under the workspace root) for issue #956. NOTE: live-episode scoring is not yet wired into `fak dojo run` (which today scores Claude Code transcripts passed via --corpus), so this records the boundary but does not yet feed the scorer.")
	ggufPath := fs.String("gguf", "", "run a SMALL MODEL IN-KERNEL as the local upstream — no API key, no network, no second server. fak loads these GGUF weights into its OWN engine and serves them to the wrapped agent, so the whole `local model + your coding harness + kernel floor` stack is ONE command (`fak guard --gguf qwen2.5:7b -- claude`). Accepts a model alias (`fak ls`), an hf://owner/repo/file.gguf URI (downloaded on demand), or a local .gguf path. Every tool call the agent proposes is still adjudicated by the same capability floor and recorded in the same audit journal — only the inference moves onto YOUR box. Mutually exclusive with --base-url / --remote-serve.")
	localAuto := fs.Bool("local", false, "auto-detect a local OpenAI-compatible model server you are ALREADY running (Ollama, LM Studio, or a llama.cpp server) and wire guard's upstream to it with zero flags — `fak guard --local -- codex` becomes a governed local coding loop with no base-URL hunting. Probes, fail-soft (~300ms each), Ollama (127.0.0.1:11434, honors OLLAMA_HOST), then LM Studio (127.0.0.1:1234), then llama.cpp (127.0.0.1:8080); the first live one wins and a coding-tuned served model is preferred. If --gguf is ALSO passed it wins (that is the no-server in-kernel path); if nothing is detected and no --gguf, fak fails loud with how to start a server. Mutually exclusive with --base-url / --remote-serve.")
	gpuBackend := fs.String("backend", "", "with --gguf: compute backend for the in-kernel decode — empty = the CPU reference path; a registered device like 'cuda' runs prefill+decode through the GPU HAL (needs a -tags cuda build AND a reachable GPU). Fails loud if named but unavailable, so a typo never silently runs on CPU.")
	tokPath := fs.String("tokenizer", "", "with --gguf: OPTIONAL tokenizer override (a tokenizer.json or its directory); default uses the GGUF's EMBEDDED tokenizer. Pass this only for a checkpoint with no embedded BPE tokenizer or a custom vocab.")
	replayTrace := fs.String("replay-trace", "", "DON'T wrap a live agent — instead REPLAY a recorded trace fixture through the real guard end to end and watch the floor fire. Stands up the gateway against a built-in fake upstream that emits the fixture's tool_use + token-usage turns, posts each turn through the SAME adjudication path `fak guard -- claude` uses, and prints per-turn what was allowed vs denied (with the deny reason), the turn's token/cache economy, and the journal rows recorded — then the exit summary + the verify command. No API key, no GPU, no child process. Use it to understand exactly what the guard does to a trace that leads to token work, and to demo the floor. See internal/gateway/testdata/guard-trace-e2e.json for the fixture shape.")
	replayWire := fs.String("replay-wire", "anthropic", "with --replay-trace: the provider wire to replay over (anthropic = the `fak guard -- claude` flagship /v1/messages path; openai = the codex/opencode /v1/chat/completions path).")
	compress := fs.Bool("compress", false, "activate the native context-compressor for this session: shrink benign tool results (ANSI/control strip, CR-redraw collapse, duplicate-line fold, JSON minify) before they enter model context, only when the saving clears the worth-it floor and never on poison, with the original preserved (reversible). Equivalent to FAK_COMPRESSOR=native for this process; an explicit FAK_COMPRESSOR wins. See `fak headroom bench` for the savings and `fak headroom status` for the live decision breakdown.")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak guard [flags] -- <agent command...>")
		fmt.Fprintln(os.Stderr, "  e.g. fak guard -- claude")
		fmt.Fprintln(os.Stderr, "       fak guard --provider openai -- codex")
		fmt.Fprintln(os.Stderr, "       fak guard --policy my-floor.json -- claude")
		fs.PrintDefaults()
	}
	_ = fs.Parse(argv)

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

	// Fail loud BEFORE binding the gateway if the wrapped agent is not on PATH — a cold
	// adopter who installed only fak (curl|sh) and ran `fak guard -- claude` without Claude
	// Code gets an actionable next step instead of a raw exec error after the gateway
	// already started (issue #835, failure 1). A command given as an explicit path is left
	// to exec to resolve.
	if !strings.ContainsAny(command[0], "/\\") {
		if _, lookErr := exec.LookPath(command[0]); lookErr != nil {
			fmt.Fprintf(os.Stderr, "fak guard: %q is not on your PATH. Install it (Claude Code: https://claude.com/claude-code), or pass the full path / a different agent after `--`.\n", command[0])
			os.Exit(2)
		}
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
		rt          policy.Runtime
		err         error
		floorSource string
	)
	if *policyPath != "" {
		rt, err = policy.LoadRuntime(*policyPath)
		floorSource = *policyPath
	} else {
		rt, err = policy.ParseRuntime(guardDefaultPolicyJSON)
		floorSource = "built-in guard floor (--dump-policy to see it)"
	}
	must(err)
	adjudicator.Default.SetPolicy(rt.Adjudicator)
	applyRuntime(rt)

	// 1b. Default the durable DECISION JOURNAL on. fak guard is the disinterested
	//     referee; a tamper-evident, hash-chained record of every verdict is what
	//     lets an auditor confirm after the fact what the kernel allowed and blocked
	//     — the witness that makes the refereeing checkable rather than self-reported.
	//     So it is ON by default (announced in the banner, not silent). The kernel's
	//     EvDecide/EvDeny events on the proxy adjudication path are exactly what the
	//     journal records, so a guard session produces a populated ledger. Precedence:
	//     FAK_AUDIT_JOURNAL honored at boot wins; --no-audit / --audit off disables;
	//     --audit PATH or a per-user default path otherwise. Enable BEFORE serving so
	//     the emitter is registered before the first decision crosses the floor.
	auditLabel, auditJournal := guardEnableAudit(*auditPath, *noAudit)
	var auditSeq0 uint64
	if auditJournal != nil {
		auditSeq0, _, _ = auditJournal.Stats()
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

	// --gguf turns the in-process gateway into a LOCAL in-kernel model server (fak runs the
	// model itself), so it is mutually exclusive with the upstream-proxy flags — the local
	// model IS the upstream. Decide + validate up front, before binding or pulling weights.
	localModel, localConflict := guardLocalModelDecision(*ggufPath, *baseURL, remoteBase)
	if localConflict != "" {
		fmt.Fprintln(os.Stderr, "fak guard:", localConflict)
		os.Exit(2)
	}

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
		detBase, detModel, detLabel, found := guardDetectLocalBackend()
		if !found {
			fmt.Fprintln(os.Stderr, guardLocalNothingDetectedMessage())
			os.Exit(2)
		}
		*provider, *baseURL = "openai", detBase
		if strings.TrimSpace(*model) == "" {
			*model = detModel
		}
		if !*quiet {
			modelNote := detModel
			if modelNote == "" {
				modelNote = "(server default)"
			}
			fmt.Fprintf(os.Stderr, "fak guard --local: detected %s at %s, using model %s\n", detLabel, detBase, modelNote)
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
		if err := guardPreflightRemoteServe(remoteBase); err != nil {
			fmt.Fprintf(os.Stderr, "fak guard: --remote-serve %s is not reachable: %v\n  start it on the box with `fak serve --gguf <weights> --backend cuda --addr 0.0.0.0:8080`, or check the host/port.\n", remoteBase, err)
			os.Exit(2)
		}
	}

	// 3. Resolve the upstream wire + credential posture. Two worlds:
	//
	//    LOCAL (--gguf): fak runs the model itself in-kernel, so there is NO upstream API,
	//    no API key, and no OAuth. Resolve ONLY the wire (anthropic for claude, openai for
	//    codex/…) — that still selects which base-URL env var points the child at the
	//    gateway and labels the banner — and leave the credential posture empty.
	//
	//    PROXY (default): resolveGuardUpstream picks the provider, base URL, API key, and
	//    the Claude subscription-OAuth default. --remote-serve, when set, pins provider=openai
	//    + base=the box inside the resolver.
	var (
		up                   string
		providerAutodetected bool
		resolvedBase         string
		apiKey               string
		pinUpstream          bool
		oauthSource          string
		// apiKeyFunc re-resolves the upstream credential per request when set. On the
		// pinned Claude subscription path it re-reads the short-lived OAuth access token
		// from disk, so a long guarded session (which outlives the ~1h token) always sends
		// the live token the client has since rotated — never the frozen boot-time one that
		// would 401 even after a fresh /login.
		apiKeyFunc func() string
	)
	if localModel {
		up, providerAutodetected = resolveGuardProvider(*provider, command[0])
	} else {
		us := resolveGuardUpstream(*provider, command[0], *baseURL, remoteBase, *apiKeyEnv, *anthropicOAuth, *oauthTokenEnv)
		up, providerAutodetected, resolvedBase = us.provider, us.autodetected, us.baseURL
		apiKey, pinUpstream, oauthSource = us.apiKey, us.pinUpstream, us.oauthSource
		if us.passthroughFallback && !*quiet {
			fmt.Fprintln(os.Stderr, "fak guard: no Claude subscription OAuth token found; falling back to passthrough — the wrapped agent's own credential (a subscription login or ANTHROPIC_API_KEY) is forwarded upstream. If you hit a 401, run `claude` once or `claude setup-token`.")
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
				tok, _, err := resolveAnthropicOAuthToken(tokenEnv)
				if err != nil {
					return ""
				}
				return tok
			}
		}
	}

	requireKey, ok := resolveRequiredKey(*requireKeyEnv, os.Getenv)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak guard: --require-key-env %s is set but empty — refusing to start a gateway with NO authentication (set it or drop the flag)\n", *requireKeyEnv)
		os.Exit(2)
	}
	if *contextBudgetTokens < 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --context-budget-tokens must be non-negative")
		os.Exit(2)
	}
	if *resetOnBudget && *contextBudgetTokens <= 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --reset-on-budget requires --context-budget-tokens N")
		os.Exit(2)
	}
	if *restartOnBudget && *contextBudgetTokens <= 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --restart-on-budget requires --context-budget-tokens N")
		os.Exit(2)
	}
	if *restartLimit < 0 {
		fmt.Fprintln(os.Stderr, "fak guard: --restart-limit must be non-negative")
		os.Exit(2)
	}
	guardTraceID := strings.TrimSpace(*sessionID)
	if guardTraceID == "" {
		guardTraceID = "guard"
	}
	if *contextBudgetTokens > 0 {
		serveSessions.SetBudget(guardTraceID, session.Budget{
			TurnsLeft:         session.Unbounded,
			TokensLeft:        session.Unbounded,
			ContextTokensLeft: *contextBudgetTokens,
		})
	}
	restarter := newGuardBudgetRestarter(*restartOnBudget, *contextBudgetTokens, *restartLimit, *restartSeedDir, os.Stderr)

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
		inKernelModel, inKernelQ4K, _, _ = loadServeInKernelModel(*ggufPath, chatBackend, false, *contextBudgetTokens)
		if inKernelModel == nil {
			fmt.Fprintf(os.Stderr, "fak guard: failed to load %q into the in-kernel engine\n", *ggufPath)
			os.Exit(1)
		}
		var tokOK bool
		inKernelTok, tokOK = resolveServeTokenizer(*tokPath, *ggufPath)
		if !tokOK || inKernelTok == nil {
			fmt.Fprintf(os.Stderr, "fak guard: %q has no usable tokenizer; pass --tokenizer or use a GGUF with an embedded tokenizer\n", *ggufPath)
			os.Exit(1)
		}
	}

	// 4. Bind the listener up front so the real port is known BEFORE we wire the child,
	//    and so there is no bind race between serving and exec. Serve(ctx, ln) accepts
	//    immediately on the goroutine below.
	listenAddr := strings.TrimSpace(*addr)
	if listenAddr == "" {
		listenAddr = "127.0.0.1:0" // an OS-picked free loopback port.
	}
	ln, err := net.Listen("tcp", listenAddr)
	must(err)
	gwURL := "http://" + ln.Addr().String()

	// A gateway bound BEYOND loopback with no required key is an UNAUTHENTICATED kernel
	// reachable off-host. `fak serve` warns about this in ListenAndServe, but guard binds
	// its own listener and calls Serve() directly (to know the port up front), which skips
	// that check — so re-assert it here rather than let the warning silently vanish.
	if requireKey == "" && !guardLoopbackOnly(ln.Addr().String()) {
		fmt.Fprintf(os.Stderr, "fak guard: WARNING — binding %s with no --require-key-env: the kernel gateway is reachable off-host with NO authentication. Bind a loopback --addr or set --require-key-env.\n", ln.Addr().String())
	}

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
		// from fak's own engine — no upstream call. nil/false in the proxy path, so the
		// default `fak guard -- claude` upstream behavior is unchanged.
		InKernelModel:         inKernelModel,
		Tokenizer:             inKernelTok,
		InKernelQ4K:           inKernelQ4K,
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
		ResetOnBudget:         resetOnBudgetHook(*resetOnBudget, *contextBudgetTokens),
		OnBudgetExhausted:     restarter.OnBudgetExhausted,
		DefaultTraceID:        guardTraceID,
		StartTime:             t0,
		// Default OFF (clean terminal); --log routes the full structured stream to a file
		// or stderr. /metrics + /debug/vars + the audit journal carry the record regardless.
		Logf: gwLogf,
		// The observable debug layer (#793) is ON by default so the cache + token-value
		// economy of every turn is visible without a flag; --debug-stats=false or --quiet
		// silences it. The full JSON --log stream stays separate (and off by default).
		DebugStatsf:          debugStatsSink(*debugStats && !*quiet),
		CtxViewBudget:        *ctxViewBudget,
		CompactHistoryBudget: *compactHistoryBudget,
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
	var stopHookInstall guardStopHookInstall
	var stopHookEnv [][2]string
	command, stopHookEnv, stopHookInstall, err = installGuardStopHook(command, *denyAllContinue, gwURL, preCompactInstall.SettingsPath, *denyAllMax)
	if err != nil {
		cancel()
		fmt.Fprintf(os.Stderr, "fak guard: Claude Stop hook setup failed: %v\n", err)
		os.Exit(1)
	}
	injected = append(injected, stopHookEnv...)
	child := buildGuardChild(command, injected, pinUpstream)

	if !*quiet {
		if providerAutodetected {
			fmt.Fprintf(os.Stderr, "fak guard: detected agent %q -> --provider %s (pass --provider to override)\n", strings.ToLower(filepath.Base(command[0])), up)
		}
		injectNames := injected[0][0]
		for _, kv := range injected[1:] {
			injectNames += ", " + kv[0]
		}
		localLabel := ""
		if localModel {
			localLabel = filepath.Base(*ggufPath)
		}
		printGuardBanner(os.Stderr, gwURL, up, resolvedBase, floorSource, injectNames, injected[0][1], logLabel, auditLabel, remoteBase != "", localModel, localLabel, command)
		if preCompactInstall.Applied {
			fmt.Fprintf(os.Stderr, "fak guard: Claude PreCompact hook: %s (settings %s)\n", preCompactInstall.Mode, preCompactInstall.SettingsPath)
		}
		if stopHookInstall.Applied {
			fmt.Fprintf(os.Stderr, "fak guard: Claude Stop hook (deny-all auto-continue): %s, max %d consecutive (the floor-refused-everything turn is reported as end_turn; this resumes the agent past it — --deny-all-continue off to disable)\n", stopHookInstall.Mode, stopHookInstall.Max)
		}
		if *debugStats {
			fmt.Fprintln(os.Stderr, "  debug      : observable layer ON — one cache/token-value line per turn to stderr (request_tokens/cache_read/cache_creation/cache_hit/cache_rebate_tokens/compact/health); --debug-stats=false or --quiet to silence")
		}
		// A LOCAL in-kernel model has no upstream credential to report; the proxy-path auth
		// note (subscription OAuth vs passthrough) only applies when fak proxies an API.
		if !localModel {
			switch {
			case pinUpstream:
				fmt.Fprintf(os.Stderr, "fak guard: upstream auth — Claude Pro/Max subscription (OAuth token from %s, sent as a bearer token)\n", oauthSource)
			case up == "anthropic":
				fmt.Fprintln(os.Stderr, "fak guard: upstream auth — passthrough (Claude Code forwards its own credential through the gateway)")
			}
		}
		if *contextBudgetTokens > 0 {
			fmt.Fprintf(os.Stderr, "fak guard: session budget — trace_id=%s context_tokens=%d\n", guardTraceID, *contextBudgetTokens)
			if *resetOnBudget {
				fmt.Fprintln(os.Stderr, "fak guard: session reset — transparent carryover enabled")
			}
			if *restartOnBudget {
				fmt.Fprintln(os.Stderr, "fak guard: session restart — child relaunch on budget exhaustion enabled")
			}
		}
	}

	// 6. Run the wrapped agent, then tear the gateway down and report the session.
	if restarter.Enabled() {
		runGuardChildSupervisedAndReport(command, injected, pinUpstream, restarter, srv, cancel, serveErr, *quiet, auditJournal, auditSeq0, command[0])
		return
	}
	runGuardChildAndReport(child, srv, cancel, serveErr, *quiet, auditJournal, auditSeq0, command[0])
}

const guardAnthropicOAuthSecretKey = "CLAUDE_SUBSCRIPTION_OAUTH_TOKEN"

// logDojoEpisodeStart records the start of a live dojo episode when --dojo is enabled.
// This is the minimal implementation for issue #956: create the .dojo/live-episodes
// corpus directory and write a start-marker with basic metadata (mode, command, started,
// cwd, workspace).
//
// SCOPE (honest boundary): this writes ONLY the start-marker. It does NOT yet capture the
// session's turn transcript or AdjudicationSummary, and `fak dojo run` does NOT yet read
// .dojo/live-episodes (it scores Claude Code transcripts passed via --corpus). So the
// marker is not yet consumed by the scorer — closing that loop (capture the full episode
// in dojo's scored format + teach `fak dojo run` to discover this corpus) is the rest of
// #956. The flag help says the same so it does not over-promise a wired scoring path.
func logDojoEpisodeStart(mode string) error {
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("getcwd: %w", err)
	}

	workspaceRoot := findRepoRoot(cwd)
	dojoDir := filepath.Join(workspaceRoot, ".dojo", "live-episodes")
	if err := os.MkdirAll(dojoDir, 0755); err != nil {
		return fmt.Errorf("mkdir dojo corpus: %w", err)
	}

	episodeFile := filepath.Join(dojoDir, fmt.Sprintf("episode_%s.jsonl", time.Now().Format("20060102_150405")))
	f, err := os.OpenFile(episodeFile, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return fmt.Errorf("open episode file: %w", err)
	}
	defer f.Close()

	ep := map[string]any{
		"mode":      "live",
		"command":   mode,
		"started":   time.Now().UTC().Format(time.RFC3339),
		"cwd":       cwd,
		"workspace": workspaceRoot,
	}
	if err := json.NewEncoder(f).Encode(ep); err != nil {
		return fmt.Errorf("encode episode: %w", err)
	}

	return nil
}

// findRepoRoot walks up from start to the nearest dir containing .git; falls back to start.
func findRepoRoot(start string) string {
	cur := filepath.Clean(start)
	for {
		if _, err := os.Stat(filepath.Join(cur, ".git")); err == nil {
			return cur
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return start
		}
		cur = parent
	}
}

type guardOAuthEnvSource struct {
	key string
	env string
}

func (s guardOAuthEnvSource) Name() string { return "$" + s.env }

func (s guardOAuthEnvSource) Lookup(key string) (string, bool) {
	if key != s.key || s.env == "" {
		return "", false
	}
	v := strings.TrimSpace(os.Getenv(s.env))
	return v, v != ""
}

type guardOAuthCredentialsSource struct {
	key  string
	path string
	now  func() time.Time
	warn io.Writer
}

func (s guardOAuthCredentialsSource) Name() string { return s.path }

func (s guardOAuthCredentialsSource) Lookup(key string) (string, bool) {
	if key != s.key {
		return "", false
	}
	now := time.Now
	if s.now != nil {
		now = s.now
	}
	// Claude Code refreshes this file ~hourly by rewriting it, so a read can race the
	// rewrite and catch a torn/empty/partial body that fails to parse. The window closes
	// in microseconds, so when the file EXISTS but the current read does not yield a token,
	// retry a few times over a few ms before giving up — that keeps a transient torn read
	// from being reported as "no active login" and falling through to the sibling
	// .oauth-token (a DIFFERENT, possibly-stale setup token). A genuinely-absent file (the
	// first os.Stat error) still misses immediately.
	const tornReadRetries = 3
	for attempt := 0; ; attempt++ {
		v, ok, transient := s.readOnce(now)
		if ok || !transient || attempt >= tornReadRetries {
			return v, ok
		}
		time.Sleep(15 * time.Millisecond)
	}
}

// readOnce performs a single resolve of the credentials file. It returns the token and
// ok=true on success; on failure, transient reports whether the failure looks like a
// torn/racing read of an EXISTING file (worth a brief retry) versus a definitive miss —
// an absent file, or a present-but-expired token (which must NOT be sent: an expired
// bearer 401s, so it is treated as absent so a fresher source or the per-request 401
// refresh can take over).
func (s guardOAuthCredentialsSource) readOnce(now func() time.Time) (tok string, ok bool, transient bool) {
	b, err := os.ReadFile(s.path)
	if err != nil {
		// A missing file is a definitive miss; any other read error (a momentary
		// permission/lock blip during the rewrite) is worth a short retry.
		return "", false, !os.IsNotExist(err)
	}
	var doc struct {
		ClaudeAIOauth struct {
			AccessToken string `json:"accessToken"`
			ExpiresAt   int64  `json:"expiresAt"`
		} `json:"claudeAiOauth"`
	}
	if json.Unmarshal(b, &doc) != nil {
		// File exists but does not parse — a torn read mid-rewrite. Retry.
		return "", false, true
	}
	v := strings.TrimSpace(doc.ClaudeAIOauth.AccessToken)
	if v == "" {
		// Parsed but no token (truncated-but-valid JSON, or a transitional empty write).
		return "", false, true
	}
	// An expired access token is a KNOWN-BAD bearer (the upstream 401s on it). Treat it as
	// absent rather than send it: a higher-priority source already lost, so falling through
	// lets the long-lived .oauth-token setup token answer, and the per-request 401 refresh
	// (agent.HTTPPlanner) re-reads the file once Claude Code rewrites the rotated token in.
	if exp := doc.ClaudeAIOauth.ExpiresAt; exp > 0 && exp < now().UnixMilli() {
		if s.warn != nil {
			fmt.Fprintf(s.warn, "fak guard: WARNING — the OAuth token in %s expired; Claude Code normally refreshes it. Re-run `claude` once, or use `claude setup-token` for a long-lived token.\n", s.path)
		}
		return "", false, false
	}
	return v, true, false
}

type guardOAuthFileSource struct {
	key  string
	path string
}

func (s guardOAuthFileSource) Name() string { return s.path }

func (s guardOAuthFileSource) Lookup(key string) (string, bool) {
	if key != s.key {
		return "", false
	}
	b, err := os.ReadFile(s.path)
	if err != nil {
		return "", false
	}
	v := strings.TrimSpace(string(b))
	return v, v != ""
}

func guardAnthropicOAuthLoader(tokenEnv, cfgDir string, now func() time.Time, warn io.Writer) (*secretload.Loader, []string) {
	tried := make([]string, 0, 3)
	sources := make([]secretload.SecretSource, 0, 3)

	if tokenEnv != "" {
		tried = append(tried, "$"+tokenEnv)
		sources = append(sources, guardOAuthEnvSource{key: guardAnthropicOAuthSecretKey, env: tokenEnv})
	}

	credPath := filepath.Join(cfgDir, ".credentials.json")
	tried = append(tried, credPath)
	sources = append(sources, guardOAuthCredentialsSource{
		key:  guardAnthropicOAuthSecretKey,
		path: credPath,
		now:  now,
		warn: warn,
	})

	setupPath := filepath.Join(cfgDir, ".oauth-token")
	tried = append(tried, setupPath)
	sources = append(sources, guardOAuthFileSource{key: guardAnthropicOAuthSecretKey, path: setupPath})

	l := secretload.New(sources...)
	l.Require(guardAnthropicOAuthSecretKey, "Claude Pro/Max subscription OAuth token", nil)
	return l, tried
}

// resolveAnthropicOAuthToken finds a Claude Pro/Max SUBSCRIPTION OAuth token to
// authenticate the upstream with, in priority order:
//  1. the named env var (default CLAUDE_CODE_OAUTH_TOKEN) — the explicit
//     headless/automation override;
//  2. <claude-config>/.credentials.json -> claudeAiOauth.accessToken — the active
//     Claude Code login token. This mirrors the credential direct Claude Code is
//     using right now, which matters because a stale or org-disallowed setup token
//     can exist beside a working interactive login;
//  3. <claude-config>/.oauth-token — a long-lived setup-token file. This remains
//     the fallback for headless homes with no interactive login, and callers that
//     want to force a specific setup token can still put it in tokenEnv.
//
// <claude-config> is $CLAUDE_CONFIG_DIR (first entry if it is a list) when set,
// else ~/.claude. Returns the token and a human source label, or an error that
// names every place it looked so the operator can fix the setup.
func resolveAnthropicOAuthToken(tokenEnv string) (token, source string, err error) {
	loader, tried := guardAnthropicOAuthLoader(tokenEnv, guardClaudeConfigDir(), time.Now, os.Stderr)
	if v, src, ok := loader.LookupSource(guardAnthropicOAuthSecretKey); ok {
		return v, src, nil
	}

	return "", "", fmt.Errorf("no Claude subscription OAuth token found (looked in: %s). Log into Claude Code (`claude`), or create a long-lived one with `claude setup-token` and export it as %s", strings.Join(tried, ", "), tokenEnv)
}

// guardSubscriptionLoginPresent reports whether a Claude subscription login EXISTS on disk,
// independent of whether its token is readable RIGHT NOW. Claude Code rewrites
// <claude-config>/.credentials.json roughly hourly and the OAuth access token it holds is
// short-lived, so a single boot-time read can legitimately catch the file mid-rotation (or
// holding a just-expired token) and miss — even though a live login is there and the token
// will rotate back in within seconds. resolveAnthropicOAuthToken correctly returns "absent"
// in that window (a torn/expired read must NOT be sent as a bearer), but the guard's
// pin-vs-passthrough boot decision must NOT read that transient miss as "no subscription":
// demoting to passthrough strips the placeholder that keeps the wrapped agent out of its own
// /login, so the agent hangs on a login prompt for a session that would have recovered on the
// first per-request token re-resolve (the 'stuck on login sometimes' race). This is the cheap
// disk witness that separates "a login is present, the token is just briefly unreadable" (pin
// on intent; the per-request APIKeyFunc recovers the fresh token) from "no subscription at
// all" (genuinely fall back to passthrough). The named env override (CLAUDE_CODE_OAUTH_TOKEN)
// counts as present when set. Existence only — it never reads or validates the token.
func guardSubscriptionLoginPresent(tokenEnv string) bool {
	if tokenEnv != "" && strings.TrimSpace(os.Getenv(tokenEnv)) != "" {
		return true
	}
	cfgDir := guardClaudeConfigDir()
	for _, name := range []string{".credentials.json", ".oauth-token"} {
		if fi, err := os.Stat(filepath.Join(cfgDir, name)); err == nil && !fi.IsDir() {
			return true
		}
	}
	return false
}

// guardClaudeConfigDir resolves the directory that holds Claude Code's per-account
// credentials: $CLAUDE_CONFIG_DIR (first path if it is an OS-list) when set, else
// ~/.claude. A home that cannot be resolved degrades to the literal ".claude".
func guardClaudeConfigDir() string {
	if v := strings.TrimSpace(os.Getenv("CLAUDE_CONFIG_DIR")); v != "" {
		if i := strings.IndexByte(v, os.PathListSeparator); i >= 0 {
			v = v[:i]
		}
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ".claude"
	}
	return filepath.Join(home, ".claude")
}

// printGuardBanner explains exactly what is now in front of the agent: where the
// gateway is, what it proxies to, which floor is loaded, the single env var injected
// into the child, and WHERE TO WATCH IT — the live metrics/debug endpoints, the durable
// audit journal, and the structured log stream. It goes to stderr so it never pollutes a
// `-p` JSON run the child writes to stdout.
func printGuardBanner(w io.Writer, gwURL, provider, baseURL, floorSource, injectVar, injectVal, logLabel, auditLabel string, remoteServe, local bool, localLabel string, command []string) {
	fmt.Fprintf(w, "fak guard — kernel-adjudicated: %s\n", strings.Join(command, " "))
	fmt.Fprintf(w, "  gateway    : %s   (in-process; torn down when the command exits)\n", gwURL)
	switch {
	case local:
		// The model runs IN-KERNEL on this box; there is no upstream API call at all. Say so
		// plainly — it is the headline of `fak guard --gguf`: a local model + your harness.
		fmt.Fprintf(w, "  upstream   : in-kernel %s   (LOCAL — fak runs the model itself; no API key, no network) via the %s wire\n", localLabel, provider)
	case remoteServe:
		// Tell the operator the dev turn's INFERENCE is on the lab box they chose, not a
		// public API — the whole point of --remote-serve.
		fmt.Fprintf(w, "  upstream   : %s   (remote fak serve on a lab box, %s wire)\n", baseURL, provider)
	default:
		fmt.Fprintf(w, "  upstream   : %s   (via the %s wire)\n", baseURL, provider)
	}
	fmt.Fprintf(w, "  floor      : %s\n", floorSource)
	fmt.Fprintf(w, "  wired via  : %s=%s   (child only — your shell is untouched)\n", injectVar, injectVal)
	// Observability: the live scrape surfaces are on the gateway URL above (unauth on
	// loopback); the audit journal is ON by default (auditLabel says where), the log
	// stream survives the session only if asked for.
	fmt.Fprintf(w, "  metrics    : %s/metrics  ·  %s/debug/vars  ·  %s/v1/fak/events\n", gwURL, gwURL, gwURL)
	// Point operators at the cache-value metric family by name — it lives on /metrics
	// above, but nothing told them to scrape for it (#1077, epic #1072). These are the
	// numbers that answer "what did fak's owned KV cache actually save this session?".
	fmt.Fprintf(w, "  cache value: scrape %s/metrics for the fak_vcache_* family (saved_token_equiv, hit_rate, multiplier, proven)\n", gwURL)
	fmt.Fprintf(w, "  audit log  : %s\n", auditLabel)
	fmt.Fprintf(w, "  gateway log: %s\n", logLabel)
	fmt.Fprintln(w, "  every tool call the agent proposes crosses the capability floor before it runs.")
}

// guardLogSink builds the gateway's structured-log destination from the --log value.
// "" (default) mutes it (a no-op) to keep the wrapped agent's terminal clean; "-" or
// "stderr" streams it to stderr; any other value appends to that file. It returns the
// log function, an optional closer (the opened file), and a human label for the banner.
// A file that cannot be opened is fatal — an operator who asked for a log and silently
// got none is worse than a loud failure.
func guardLogSink(logPath string, stderr io.Writer) (logf func(string, ...any), closer io.Closer, label string) {
	switch strings.TrimSpace(logPath) {
	case "":
		return func(string, ...any) {}, nil, "off (--log FILE or --log - to enable)"
	case "-", "stderr":
		lg := log.New(stderr, "fak-gateway ", log.LstdFlags|log.Lmsgprefix)
		return lg.Printf, nil, "stderr"
	default:
		f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		must(err)
		lg := log.New(f, "", log.LstdFlags)
		return lg.Printf, f, logPath
	}
}

// guardAuditPlan is the PURE decision behind guard's default-on audit journal: it
// returns the path to enable (""=> do not enable) and whether the off was an
// explicit opt-out, given the flags and whether a journal is already active at
// boot (FAK_AUDIT_JOURNAL). Kept side-effect-free so the precedence — boot env
// wins, then --no-audit / --audit off, then --audit PATH, then the per-user
// default — is unit-tested without touching the process-global journal.
func guardAuditPlan(auditPath string, noAudit, bootActive bool) (enablePath string, optedOut bool) {
	if bootActive {
		return "", false // FAK_AUDIT_JOURNAL already registered an emitter; nothing to enable
	}
	if noAudit || strings.EqualFold(strings.TrimSpace(auditPath), "off") {
		return "", true
	}
	p := strings.TrimSpace(auditPath)
	if p == "" {
		p = guardDefaultAuditPath()
	}
	return p, false
}

// guardDefaultAuditPath is where fak guard writes its durable decision journal
// when the operator names none: <user-config>/fak/guard-audit.jsonl — a stable,
// per-user, cross-platform location appended across sessions so the tamper-evident
// chain CONTINUES rather than forking each run. Falls back to ".fak/guard-audit.jsonl"
// under the working directory if no user config dir resolves.
func guardDefaultAuditPath() string {
	if dir, err := os.UserConfigDir(); err == nil && strings.TrimSpace(dir) != "" {
		return filepath.Join(dir, "fak", "guard-audit.jsonl")
	}
	return filepath.Join(".fak", "guard-audit.jsonl")
}

// guardEnableAudit turns the durable, hash-chained decision journal ON for the
// session per guardAuditPlan and returns a human label for the banner plus the
// active journal (nil when disabled). A failure to open a REQUESTED path is fatal
// (must) — an operator who asked for an audit trail and silently got none is worse
// than a loud failure, mirroring guardLogSink's file-sink contract.
func guardEnableAudit(auditPath string, noAudit bool) (label string, active *journal.Journal) {
	// A boot-time FAK_AUDIT_JOURNAL already registered an emitter we cannot
	// unregister; respect it (and note --no-audit cannot turn it off).
	if j := journal.Active(); j != nil {
		if p := j.Path(); p != "" {
			return p + "  (durable, hash-chained; from FAK_AUDIT_JOURNAL)", j
		}
		return "active  (durable, hash-chained; from FAK_AUDIT_JOURNAL)", j
	}
	path, optedOut := guardAuditPlan(auditPath, noAudit, false)
	if path == "" {
		if optedOut {
			return "off  (default-on; disabled by --no-audit / --audit off)", nil
		}
		return "off", nil
	}
	j, err := journal.Enable(path)
	must(err)
	return path + "  (durable, hash-chained — verify with: fak audit verify <path>)", j
}

// formatJournalSummary is the exit-roll-up line for the durable trail: how many
// hash-chained rows this session appended, where, and the command to re-verify the
// chain. Empty when no journal ran, so a --no-audit session stays quiet.
func formatJournalSummary(j *journal.Journal, seq0 uint64) string {
	if j == nil {
		return ""
	}
	path := j.Path()
	if path == "" {
		return ""
	}
	if err := j.Flush(); err != nil {
		return fmt.Sprintf("fak guard: audit journal — flush error: %v\n", err)
	}
	seq, _, writeErr := j.Stats()
	var b strings.Builder
	fmt.Fprintf(&b, "fak guard: audit journal — %d decision(s) appended this session; chain now holds %d hash-chained row(s) at %s",
		seq-seq0, seq, path)
	if writeErr > 0 {
		fmt.Fprintf(&b, " (%d write error(s))", writeErr)
	}
	b.WriteByte('\n')
	fmt.Fprintf(&b, "  verify the tamper-evident chain: fak audit verify %s\n", path)
	return b.String()
}

// formatAuditSummary renders the exit roll-up of what the kernel decided while the
// agent ran. "kernel decision(s)" — not "tool calls" — because the tally folds BOTH
// proposed-call adjudications AND inbound tool-result admissions (a quarantined result
// is a kernel decision about a result the agent already ran, not a proposed call). It
// is one honest count: every number came from the same operation counters /metrics
// exposes, so the line can never overstate the protection.
func formatAuditSummary(sum gateway.AdjudicationSummary) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak guard: %d kernel decision(s) — %d allowed, %d denied, %d repaired, %d quarantined",
		sum.Total, sum.Allowed, sum.Denied, sum.Transformed, sum.Quarantined)
	// Deferred (a non-blocking admit, e.g. a tool result let through) and escalated
	// (held pending a witness) are normal, non-error outcomes — show them only when
	// they happened so the common clean line stays short, and never under "errored".
	if sum.Deferred > 0 {
		fmt.Fprintf(&b, ", %d deferred", sum.Deferred)
	}
	if sum.Escalated > 0 {
		fmt.Fprintf(&b, ", %d escalated", sum.Escalated)
	}
	if sum.Errored > 0 {
		fmt.Fprintf(&b, ", %d errored", sum.Errored)
	}
	b.WriteByte('\n')
	// Make the provider prompt-cache reuse legible AND honest. This is ANTHROPIC's cache:
	// `fak guard` forwards the client's cache_control prefix byte-for-byte, so the rebate
	// would land with or without fak — fak's contribution is PRESERVING it (not bursting the
	// prefix) plus the separate tool-floor-prune line, never AUTHORING the saving. So we name
	// the baseline explicitly on the line — the same turns billed with NO cache, every prompt
	// token at the full input price — which answers "saved vs WHAT?" instead of leaving the
	// reader to know "uncached cost" by heart. NET of the write premium (a cold cache write
	// bills ABOVE an uncached read), so a write-dominated session reads "not yet repaid" until
	// the reads catch up. Same engine /metrics (fak_vcache_saved_token_equiv) and `fak vcache
	// observe` report; raw cache_read/cache_creation counts stay on /metrics and --log.
	if sum.CachedPromptTokens > 0 || sum.CacheCreationTokens > 0 {
		net := sum.ProviderCacheNetSavings()
		billed := gateway.HumanTokenEquiv(net.ActualTokenEquiv)
		baseline := gateway.HumanTokenEquiv(net.BaselineTokenEquiv)
		if net.SavedTokenEquiv > 0 {
			fmt.Fprintf(&b, "fak guard: prompt-cache saving — input billed ~%s of ~%s token-equiv (the no-cache price) → the provider cache saved ~%s (%.0f%% off), net of the cache-write premium, across %d cached turn(s). fak forwarded the cache_control prefix intact; it relays this provider-reported value and did not author this saving.\n",
				billed, baseline, gateway.HumanTokenEquiv(net.SavedTokenEquiv), net.SavedPct, sum.CachedTurns)
		} else {
			fmt.Fprintf(&b, "fak guard: prompt-cache saving — input billed ~%s of ~%s token-equiv (the no-cache price); the cache has not yet repaid its write premium (net ~%s so far — a cold write the later reads have not paid back), across %d cached turn(s). fak forwarded the cache_control prefix intact.\n",
				billed, baseline, gateway.HumanTokenEquiv(net.SavedTokenEquiv), sum.CachedTurns)
		}
	}
	if sum.CompactionFired > 0 || sum.CompactionBailed > 0 || sum.CompactionOff > 0 {
		// WITNESSED half only: what fak attempted and removed. The OBSERVED post-fire cache_read
		// is a provider counter (it lives on /metrics) and is noise here — a low value with no
		// prefix_mismatch bail is a provider-side miss fak does not control, not a fak failure.
		// Lead with whether the lever is ENABLED so "0 fired" can't read as "disabled": budget>0
		// with all-under_budget bails is compaction ON and correctly idle (nothing sprawled past
		// the cut), the opposite of OFF.
		status := fmt.Sprintf("ENABLED, budget %d tok", sum.CompactionBudget)
		if sum.CompactionBudget <= 0 {
			status = "DISABLED (budget 0; body forwarded byte-for-byte)"
		} else if sum.CompactionFired == 0 && sum.CompactionShedTokens == 0 {
			status = fmt.Sprintf("ENABLED but idle, budget %d tok — nothing sprawled past the cut", sum.CompactionBudget)
		}
		fmt.Fprintf(&b, "fak guard: compaction [%s] — %d fired, %d bailed, %d off; shed %d token(s)\n",
			status,
			sum.CompactionFired,
			sum.CompactionBailed,
			sum.CompactionOff,
			sum.CompactionShedTokens)
		// Break the bailed lump out by reason (same shape as the deny "blocked:" loop below):
		// without this, N bailed conflates under_budget (benign, working-as-designed) with
		// no_breakpoint (can't fire) and prefix_mismatch (the ONLY fak-fault cache signal — call
		// it out explicitly when nonzero so a real regression can never hide in the lump).
		if len(sum.CompactionBailReasons) > 0 {
			reasons := make([]string, 0, len(sum.CompactionBailReasons))
			for r := range sum.CompactionBailReasons {
				reasons = append(reasons, r)
			}
			sort.Strings(reasons)
			for _, r := range reasons {
				note := ""
				if r == "prefix_mismatch" || r == "splice_failed" || r == "redecode_failed" {
					note = "  ⚠ fak-fault: a fired rewrite would have burst the cache — must stay 0"
				}
				fmt.Fprintf(&b, "  bailed: %-16s x%d%s\n", r, sum.CompactionBailReasons[r], note)
			}
		}
	}
	// Tool-floor prune (the INBOUND tools[] lever): how many unreachable tool DEFINITIONS fak
	// dropped from the advertised surface this session — a pure uncached-token saving that
	// never bursts the cache (the pruner only drops tools after the cache_control breakpoint and
	// re-proves the protected prefix is byte-identical). WITNESSED. Printed only when it actually
	// fired, so the common run — and the dominant Claude Code path, whose single breakpoint sits on
	// the LAST tool so nothing is droppable — stays quiet rather than printing a vacuous 0.
	if sum.ToolPruneCount > 0 {
		fmt.Fprintf(&b, "fak guard: tool-floor prune — dropped %d unreachable tool def(s) from tools[] across %d turn(s) (uncached-token saving; cache prefix byte-identical)\n",
			sum.ToolPruneCount, sum.ToolPruneTurns)
	}
	// Deny-all stops: turns the floor refused ENTIRELY, which the wire reports to the client as
	// end_turn so it does not hang hunting for a dropped tool_use block (the v0.15.0 contract).
	// That end_turn halts the agent though the model wanted to act — a STOP the agent did not
	// choose, and the false-stop this audit surfaces. Print it only when it happened, and name the
	// Stop-hook lever that auto-resumes the agent past it, so a session that hit the false stop
	// tells the operator both that it happened and how to keep the loop moving next time.
	if sum.DenyAllStops > 0 {
		fmt.Fprintf(&b, "fak guard: deny-all stops — %d turn(s) had EVERY proposed tool call refused, reported to the client as end_turn (a stop the agent did not choose; the model wanted to act, the floor blocked all of it). Keep the agent moving past these with --deny-all-continue=enforce (auto-resumes the agent with 'choose an allowed alternative', bounded).\n",
			sum.DenyAllStops)
	}
	if len(sum.ByReason) > 0 {
		reasons := make([]string, 0, len(sum.ByReason))
		for r := range sum.ByReason {
			reasons = append(reasons, r)
		}
		sort.Strings(reasons)
		for _, r := range reasons {
			fmt.Fprintf(&b, "  blocked: %-16s x%d\n", r, sum.ByReason[r])
		}
	}
	return b.String()
}

// formatAmplification renders the avoided-call amplification headline for the guard
// exit summary — the realized answer to "how much further did the agent get per unit
// of real work?" It folds the session's kernel call-path counters (engine dispatches,
// vDSO hits, in-syscall repairs, fast-reject denies) through internal/callavoid.Account,
// the SAME pure economics the `fak callavoid account` CLI computes, so the line can
// never disagree with that tool. This closes the callavoid leaf's "Next milestone (not
// yet wired)": the tier-4 caller that reads a live guard session's kernel.Counters into
// Account for the exit summary.
//
// It returns the empty string when there was no avoidance to report — a session whose
// vDSO never hit and whose kernel repaired nothing has nothing to amplify (Execute-only
// work is 1:1), so the common clean run stays quiet rather than printing a vacuous 1.0×.
//
// kc is the in-kernel call-path axis (vDSO memo hits + in-syscall repairs), which only moves
// on the Submit/Reap path `fak serve` drives. On the flagship `fak guard -- claude` PROXY the
// kernel adjudicates with Decide, which increments none of those counters — so kc is empty
// every guard session and the kernel-axis line would never fire there. sum carries the
// Decide-path verdicts that DO move on the proxy (grammar repairs = Transformed, fast-reject
// denies = Denied); when kc is empty but the proxy repaired/denied real calls, we print a
// proxy-honest line about what the floor DID, framed as "repairs/denies applied" rather than
// "calls avoided" (a Decide-only proxy avoids no calls — the client still executes each tool).
func formatAmplification(kc kernel.Counters, sum gateway.AdjudicationSummary) string {
	// Map the live kernel counters onto the tier-1 callavoid mirror (a total, behaviour-
	// free field copy — the field names mirror kernel.Counters on purpose) and fold.
	rep := callavoid.Account(callavoid.TallyFromCounters(callavoid.Counters{
		EngineCalls: int(kc.EngineCalls),
		VDSOHits:    int(kc.VDSOHits),
		Transforms:  int(kc.Transforms),
		Denies:      int(kc.Denies),
	}))
	// Nothing was avoided on the kernel axis. Before staying silent, check the PROXY axis:
	// on `fak guard -- claude` the kernel counters are structurally 0 (Decide increments none),
	// but the floor may have repaired or denied real proposed calls — work the agent would
	// otherwise have paid a failed round-trip for. Surface THAT so the dominant path is not
	// silently blank when the floor was actually doing its job.
	if rep.MemoHits == 0 && rep.Repairs == 0 {
		if sum.Transformed > 0 || sum.Denied > 0 {
			return fmt.Sprintf("fak guard: floor effect — %d call(s) repaired in-flight, %d denied before a wasted round-trip (proxy path: the kernel adjudicates with Decide, so the in-kernel vDSO/amplification axis does not apply)\n",
				sum.Transformed, sum.Denied)
		}
		return ""
	}
	var b strings.Builder
	// Lead with the realized amplification ratio and the turns it spared, then the
	// breakdown of WHERE the avoidance came from (vDSO cache hits + in-syscall repairs).
	// A memo hit always pays callavoid.ValidateFloor (never free), so a pure-cache window
	// is capped at 1/ValidateFloor (=100×), not +Inf — Amplification is always finite on
	// this path. The only +Inf case is zero executed work, which means zero memo hits and
	// zero repairs, which the guard above has already returned the empty string for.
	fmt.Fprintf(&b, "fak guard: avoided-call amplification — %.2f× (%s); spared ~%.0f naive round-trip(s) of %d proposed",
		rep.Amplification, rep.Status, rep.AvoidedTurns, rep.RawTurns)
	parts := make([]string, 0, 2)
	if rep.MemoHits > 0 {
		parts = append(parts, fmt.Sprintf("%d served from the vDSO cache", rep.MemoHits))
	}
	if rep.Repairs > 0 {
		parts = append(parts, fmt.Sprintf("%d repaired in-syscall", rep.Repairs))
	}
	if len(parts) > 0 {
		fmt.Fprintf(&b, " — %s", strings.Join(parts, ", "))
	}
	b.WriteByte('\n')
	return b.String()
}
