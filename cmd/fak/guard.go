package main

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/secretload"
	"github.com/anthony-chaudhary/fak/internal/session"
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
func cmdGuard(argv []string) {
	t0 := time.Now()
	fs := flag.NewFlagSet("guard", flag.ExitOnError)
	addr := fs.String("addr", "", "gateway listen address (default: a private 127.0.0.1 port the OS picks)")
	provider := fs.String("provider", "", "upstream wire the gateway proxies to: anthropic|openai|gemini|xai (default: auto-detected from the agent name — claude->anthropic, codex/opencode->openai — else anthropic)")
	baseURL := fs.String("base-url", "", "upstream provider base URL (default: the provider's public API, e.g. anthropic -> https://api.anthropic.com)")
	remoteServe := fs.String("remote-serve", "", "point the guarded turn's INFERENCE at a remote `fak serve` running on a lab box you chose (HOST or HOST:PORT, default port 8080). Forces the OpenAI-compatible wire and base URL http://HOST:PORT, so the dev turn runs on the lab GPU while the kernel still adjudicates locally. Mutually exclusive with --base-url; preflights GET /healthz and fails loud if the box is not serving.")
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
	debugStats := fs.Bool("debug-stats", false, "print ONE compact, payload-free line per served turn to stderr: request/cache_read/cache_creation tokens, the compaction action, and the resetScore SHADOW health (healthy_cache|cache_decay|stale_prefix|cooldown|unknown_provider). Independent of --log; default off to keep the wrapped agent's terminal clean (#793).")
	ctxViewBudget := fs.Int("ctx-view-budget", 0, "wire the ctxplan context PLANNER into the live guard loop: each buffered turn, re-materialize the forwarded history as an O(1) planned VIEW under this resident-token budget (a planned view in place of appending the whole transcript, #555). 0 (default) leaves the existing path byte-for-byte unchanged. OFF by default: it rewrites in-flight turn history, so gate it until you have watched a wrapped session. The streaming fast-path bypasses this; the buffered turn path is what gets planned.")
	compactHistoryBudget := fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget, "compact OLD conversation turns in the OUTBOUND Anthropic request body down to this resident-token budget while keeping the cache_control prefix BYTE-IDENTICAL, so the upstream prompt-cache hit survives. This reaches the flagship `fak guard -- claude` passthrough (where the body is forwarded verbatim, #555). DEFAULT-ON: once a wrapped conversation sprawls past ~48k resident tokens the cut fires and sheds the un-cacheable middle the provider re-bills every turn; a typical short session stays untouched. Pass 0 to disable (body forwarded byte-for-byte). Anthropic passthrough only.")
	elideResultBytes := fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes, "OFF by default: shrink oversized tool_result bodies outside the active working set to a bounded head+tail form once they exceed this byte threshold. 0 disables. The documented candidate is gateway.DocumentedElideResultBytes; flip only after reading the tradeoff witness.")
	sessionID := fs.String("session-id", "guard", "default trace/session id for wrapped agents that omit X-Trace-Id or MCP trace_id")
	contextBudgetTokens := fs.Int("context-budget-tokens", 0, "seed the guard session with this prompt/context-token budget; exhaustion returns a reset directive with continuation_id (0 = off)")
	resetOnBudget := fs.Bool("reset-on-budget", false, "on context-budget exhaustion, re-arm the continuation trace with a carryover seed and continue transparently instead of returning 409 (requires --context-budget-tokens)")
	restartOnBudget := fs.Bool("restart-on-budget", false, "on context-budget exhaustion, stop and relaunch the wrapped child under the continuation trace, writing a carryover seed JSON and exposing it via FAK_RESET_* env vars (requires --context-budget-tokens)")
	restartLimit := fs.Int("restart-limit", 0, "maximum child relaunches for --restart-on-budget; 0 means unlimited")
	restartSeedDir := fs.String("restart-seed-dir", "", "directory for --restart-on-budget carryover seed JSON files (default: OS temp dir, one private directory per reset)")
	landlockHooks := fs.Bool("landlock-hooks", false, "LINUX-ONLY defense-in-depth: run the spawned agent under a Landlock profile that makes the git hook surface (.git/hooks + core.hooksPath) READ-ONLY while the rest of the tree stays writable, so a laundered write cannot drop an executable hook. OFF by default; fails OPEN (logs + spawns unrestricted) on a kernel without Landlock or on a non-Linux host. Also settable via "+guard.EnvOptIn+"=1.")
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

	// Raise the gateway's HTTP write/planner timeout floors for the wrapped session. A
	// frontier Claude turn with extended thinking can run well past fak serve's 90 s
	// WriteTimeout / 60 s planner default, which would cut the stream off mid-turn and
	// surface to the worker as a "context canceled" upstream error. Guard binds its own
	// listener and calls Serve() directly, so it must set these BEFORE the server reads
	// them (gateway.Serve consults FAK_HTTP_WRITE_TIMEOUT_S via durEnv). An explicit
	// operator value always wins — guardEnsureTimeoutFloor never clobbers a set var.
	guardEnsureTimeoutFloor("FAK_HTTP_WRITE_TIMEOUT_S", guardTimeoutFloorS)
	guardEnsureTimeoutFloor("FAK_PLANNER_TIMEOUT_S", guardTimeoutFloorS)

	if *dumpPolicy {
		os.Stdout.Write(guardDefaultPolicyJSON)
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

	// 3. Resolve the upstream wire + credential posture (provider autodetect, base URL,
	//    API key, and the Claude subscription-OAuth default); see resolveGuardUpstream.
	//    --remote-serve, when set, pins provider=openai + base=the box inside the resolver.
	us := resolveGuardUpstream(*provider, command[0], *baseURL, remoteBase, *apiKeyEnv, *anthropicOAuth, *oauthTokenEnv)
	up, providerAutodetected, resolvedBase := us.provider, us.autodetected, us.baseURL
	apiKey, pinUpstream, oauthSource := us.apiKey, us.pinUpstream, us.oauthSource
	if us.passthroughFallback && !*quiet {
		fmt.Fprintln(os.Stderr, "fak guard: no Claude subscription OAuth token found; falling back to passthrough — the wrapped agent's own credential (a subscription login or ANTHROPIC_API_KEY) is forwarded upstream. If you hit a 401, run `claude` once or `claude setup-token`.")
	}
	if us.ambientKeyOverridden && !*quiet {
		fmt.Fprintln(os.Stderr, "fak guard: ANTHROPIC_API_KEY is set but fak defaults to your Claude Pro/Max subscription (OAuth); the key is ignored upstream. Pass --api-key-env ANTHROPIC_API_KEY to use API billing instead.")
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

	// 3. Bind the listener up front so the real port is known BEFORE we wire the child,
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
		EngineID:              "inkernel",
		Model:                 *model,
		BaseURL:               resolvedBase,
		Provider:              up,
		APIKey:                apiKey,
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
		Logf:                 gwLogf,
		DebugStatsf:          debugStatsSink(*debugStats),
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

	// 5. Wire the child: inject ONLY the gateway URL into the child's environment —
	//    never the parent shell, never settings.json. A `claude` in another terminal is
	//    untouched.
	injected := guardInjectedEnv(up, *envName, gwURL)
	child := buildGuardChild(command, injected, pinUpstream)

	if !*quiet {
		if providerAutodetected {
			fmt.Fprintf(os.Stderr, "fak guard: detected agent %q -> --provider %s (pass --provider to override)\n", strings.ToLower(filepath.Base(command[0])), up)
		}
		injectNames := injected[0][0]
		for _, kv := range injected[1:] {
			injectNames += ", " + kv[0]
		}
		printGuardBanner(os.Stderr, gwURL, up, resolvedBase, floorSource, injectNames, injected[0][1], logLabel, auditLabel, us.remoteServe, command)
		switch {
		case pinUpstream:
			fmt.Fprintf(os.Stderr, "fak guard: upstream auth — Claude Pro/Max subscription (OAuth token from %s, sent as a bearer token)\n", oauthSource)
		case up == "anthropic":
			fmt.Fprintln(os.Stderr, "fak guard: upstream auth — passthrough (Claude Code forwards its own credential through the gateway)")
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
	b, err := os.ReadFile(s.path)
	if err != nil {
		return "", false
	}
	var doc struct {
		ClaudeAIOauth struct {
			AccessToken string `json:"accessToken"`
			ExpiresAt   int64  `json:"expiresAt"`
		} `json:"claudeAiOauth"`
	}
	if json.Unmarshal(b, &doc) != nil {
		return "", false
	}
	v := strings.TrimSpace(doc.ClaudeAIOauth.AccessToken)
	if v == "" {
		return "", false
	}
	now := time.Now
	if s.now != nil {
		now = s.now
	}
	if exp := doc.ClaudeAIOauth.ExpiresAt; exp > 0 && exp < now().UnixMilli() && s.warn != nil {
		fmt.Fprintf(s.warn, "fak guard: WARNING — the OAuth token in %s expired; Claude Code normally refreshes it. Re-run `claude` once, or use `claude setup-token` for a long-lived token.\n", s.path)
	}
	return v, true
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

// resolveGuardProvider picks the upstream wire for a guard session. An explicit
// --provider value (normalized) always wins. An empty value is inferred from the wrapped
// agent's name via guardDetectProvider, and an unrecognized agent falls back to anthropic
// (Claude Code, the historical default) — so an existing `fak guard -- claude` is
// unchanged while `fak guard -- codex` now picks the OpenAI wire on its own. The bool
// reports whether the wire was inferred, for the banner.
func resolveGuardProvider(flagValue, command string) (provider string, autodetected bool) {
	if v := strings.ToLower(strings.TrimSpace(flagValue)); v != "" {
		return v, false
	}
	if detected, ok := guardDetectProvider(command); ok {
		return detected, true
	}
	return "anthropic", false
}

// guardDetectProvider infers the upstream wire from the wrapped agent's command when the
// operator passes no --provider, so naming a known agent (`fak guard -- codex`) Just
// Works without also having to say `--provider openai`. The table lists agents that read
// a base-URL variable guard injects: ANTHROPIC_BASE_URL for the Anthropic wire, and
// OPENAI_BASE_URL plus OPENAI_API_BASE for the OpenAI wire (guard sets both, so Aider,
// which reads OPENAI_API_BASE rather than OPENAI_BASE_URL, connects too). An agent that
// reads neither (Goose's split OPENAI_HOST + OPENAI_BASE_PATH, an IDE-extension settings
// panel) is left to an explicit --provider/--env on purpose, rather than autodetected into
// a base URL it ignores. Matching is on the executable's base name, lowercased, with any
// directory and a Windows .exe/.cmd/.bat/.ps1/.com launcher suffix stripped, so an
// absolute path or a wrapped launcher still matches.
func guardDetectProvider(command string) (provider string, recognized bool) {
	base := strings.ToLower(strings.TrimSpace(command))
	// Strip any directory component handling BOTH path separators regardless of the
	// host OS. filepath.Base is host-specific — on the Linux CI runner it does not
	// split a Windows backslash path, so a launcher like `C:\…\claude.exe` would
	// fail to match there even though it works on a Windows dev box. LastIndexAny
	// over `/` and `\` makes the match cross-platform.
	if i := strings.LastIndexAny(base, `/\`); i >= 0 {
		base = base[i+1:]
	}
	switch filepath.Ext(base) {
	case ".exe", ".cmd", ".bat", ".ps1", ".com":
		base = strings.TrimSuffix(base, filepath.Ext(base))
	}
	switch base {
	case "claude", "claude-code":
		return "anthropic", true
	case "codex", "opencode", "aider":
		return "openai", true
	default:
		return "", false
	}
}

// guardDefaultBaseURL maps a provider to its public API base URL. The anthropic host
// is given WITHOUT a /v1 suffix (the gateway's Anthropic client appends the Messages
// path), matching the witnessed `fak serve --provider anthropic --base-url
// https://api.anthropic.com`. An unknown provider returns "" so the caller can require
// an explicit --base-url instead of guessing.
func guardDefaultBaseURL(provider string) string {
	switch provider {
	case "anthropic":
		return "https://api.anthropic.com"
	case "openai":
		return "https://api.openai.com/v1"
	default:
		return ""
	}
}

// normalizeRemoteServe turns a --remote-serve operand (HOST or HOST:PORT, with or
// without a scheme) into the canonical base URL "http://HOST:PORT" the OpenAI-compatible
// wire expects, defaulting the port to 8080 (the documented `fak serve` addr). It is the
// one place the operand grammar lives, so the resolver, the preflight, and the banner all
// agree. "" in -> "" out (feature off). A malformed operand returns an error so cmdGuard
// fails loud before binding, rather than constructing a base URL that 404s mid-session.
func normalizeRemoteServe(operand string) (string, error) {
	s := strings.TrimSpace(operand)
	if s == "" {
		return "", nil
	}
	// Strip a scheme if the operator typed one; we always emit http:// (a remote fak serve
	// on a lab tailnet is plain HTTP — TLS is the gateway's job, not assumed here).
	s = strings.TrimPrefix(strings.TrimPrefix(s, "http://"), "https://")
	s = strings.TrimRight(s, "/")
	if s == "" {
		return "", fmt.Errorf("empty host in %q", operand)
	}
	host, port := s, "8080"
	// A bracketed IPv6 host always needs SplitHostPort; a plain host has a port only if it
	// contains a single colon (an unbracketed "::1" is an IPv6 literal, not host:port).
	if strings.HasPrefix(s, "[") || strings.Count(s, ":") == 1 {
		h, p, err := net.SplitHostPort(s)
		if err != nil {
			return "", fmt.Errorf("invalid host:port %q: %w", operand, err)
		}
		if strings.TrimSpace(h) == "" {
			return "", fmt.Errorf("empty host in %q", operand)
		}
		if n, perr := strconv.Atoi(p); perr != nil || n < 1 || n > 65535 {
			return "", fmt.Errorf("invalid port %q in %q", p, operand)
		}
		host, port = h, p
	}
	return "http://" + net.JoinHostPort(host, port), nil
}

// guardPreflightRemoteServe confirms a remote `fak serve` is actually answering before
// guard binds its own gateway and execs the agent. A down box (not started, wrong port)
// is the most common --remote-serve failure, and a 502 on the first real turn is a far
// worse place to discover it than a one-line fail-loud here. It probes GET <base>/healthz
// (the endpoint `fak serve` exposes) with a short timeout; any 2xx/3xx/4xx response means
// the box is up (even a 401 proves a live gateway, like the witnessed /v1/models probe),
// so only a connection-level failure is fatal.
func guardPreflightRemoteServe(baseURL string) error {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(strings.TrimRight(baseURL, "/") + "/healthz")
	if err != nil {
		return err
	}
	_ = resp.Body.Close()
	return nil
}

// guardEnvVar picks the env var that points the child agent at the gateway. An
// explicit --env override always wins; otherwise it is the provider's conventional
// base-URL variable: Anthropic clients (Claude Code, the Anthropic SDKs) read
// ANTHROPIC_BASE_URL; OpenAI-compatible clients read OPENAI_BASE_URL (gemini/xai are
// proxied on the OpenAI-compatible surface here).
func guardEnvVar(provider, override string) string {
	if v := strings.TrimSpace(override); v != "" {
		return v
	}
	switch provider {
	case "anthropic":
		return "ANTHROPIC_BASE_URL"
	default:
		return "OPENAI_BASE_URL"
	}
}

// guardTimeoutFloorS is the generous HTTP write / planner timeout (in seconds) a
// guarded session raises the gateway floors to, so a long frontier turn is never cut
// off mid-stream. It matches the value the always-on dogfood server doc documents.
const guardTimeoutFloorS = 600

// guardEnsureTimeoutFloor sets env var `name` to `floorS` seconds ONLY when the
// operator has not already set it. An explicit value — including an explicit "0"
// (Go's no-timeout opt-out) — always wins, so this raises the default without ever
// clobbering a deliberate choice. It is the wiring behind the doc's promise that
// `fak guard` lifts the gateway's 90 s WriteTimeout floor for a wrapped session.
func guardEnsureTimeoutFloor(name string, floorS int) {
	if strings.TrimSpace(os.Getenv(name)) != "" {
		return // operator already chose a value; never clobber it.
	}
	_ = os.Setenv(name, strconv.Itoa(floorS))
}

// guardEnvValue is the base-URL VALUE injected into the child — and the two wires
// disagree on the /v1 suffix, which is the difference between a working session and a
// 404. Anthropic clients (Claude Code) append "/v1/messages" to ANTHROPIC_BASE_URL, so
// it must be the bare host. OpenAI-compatible clients (OpenCode, Codex, the OpenAI SDK,
// the Vercel AI SDK) treat OPENAI_BASE_URL as ending in "/v1" and append
// "/chat/completions" — so the value MUST carry the /v1 the gateway serves its OpenAI
// routes under. Without it the client calls "<host>/chat/completions" and the gateway
// (which exposes "/v1/chat/completions") 404s.
func guardEnvValue(provider, gwURL string) string {
	if provider == "anthropic" {
		return gwURL
	}
	return strings.TrimRight(gwURL, "/") + "/v1"
}

// guardInjectedEnv lists the environment variables guard sets in the child to point it at
// the gateway. An explicit --env override yields exactly that one var (value follows the
// wire's /v1 convention). The Anthropic wire is ANTHROPIC_BASE_URL. The OpenAI-compatible
// wire gets BOTH conventional base-URL variables a client might read: OPENAI_BASE_URL (the
// OpenAI SDK, Codex, OpenCode, the Vercel AI SDK) and OPENAI_API_BASE (LiteLLM-backed
// clients and Aider). Setting both is harmless to a client that reads only one, and it
// means more agents work under `fak guard` with no extra flag. Both pairs share one value
// (guardEnvValue), so the gateway URL is injected once under two names.
func guardInjectedEnv(provider, override, gwURL string) [][2]string {
	val := guardEnvValue(provider, gwURL)
	primary := guardEnvVar(provider, override)
	pairs := [][2]string{{primary, val}}
	if strings.TrimSpace(override) == "" && primary == "OPENAI_BASE_URL" {
		pairs = append(pairs, [2]string{"OPENAI_API_BASE", val})
	}
	return pairs
}

// guardWaitHealthy blocks until the gateway answers 200 on /healthz, the Serve
// goroutine returns early (its result is delivered on serveErr — e.g. a bound listener
// that fails before /healthz can answer), or the deadline passes. The listener is
// already bound (Serve got it pre-bound), so this normally returns on the first poll;
// the loop covers the goroutine-start gap WITHOUT waiting the full timeout on a dead
// gateway. The consumed return is true iff it drained serveErr (the early-exit case),
// so the caller knows not to drain it again.
func guardWaitHealthy(gwURL string, serveErr <-chan error, timeout time.Duration) (err error, consumed bool) {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 500 * time.Millisecond}
	var lastErr error
	for time.Now().Before(deadline) {
		// Did Serve already stop? Then the gateway is dead — fail now, do not poll a
		// corpse for the rest of the timeout.
		select {
		case se := <-serveErr:
			if se == nil {
				se = errors.New("gateway stopped before it became ready")
			}
			return se, true
		default:
		}
		resp, getErr := client.Get(gwURL + "/healthz")
		if getErr == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil, false
			}
			lastErr = fmt.Errorf("healthz returned %d", resp.StatusCode)
		} else {
			lastErr = getErr
		}
		time.Sleep(50 * time.Millisecond)
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("timed out after %s", timeout)
	}
	return lastErr, false
}

// guardLoopbackOnly reports whether addr binds only the loopback interface. It mirrors
// the gateway's own (unexported) loopbackOnly so guard can re-assert the no-auth warning
// the gateway skips when handed a pre-bound listener. A bare ":port" (all interfaces)
// and any routable IP are NOT loopback.
func guardLoopbackOnly(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr // no port present
	}
	host = strings.Trim(host, "[]")
	if host == "" {
		return false // ":port" => all interfaces
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// printGuardBanner explains exactly what is now in front of the agent: where the
// gateway is, what it proxies to, which floor is loaded, the single env var injected
// into the child, and WHERE TO WATCH IT — the live metrics/debug endpoints, the durable
// audit journal, and the structured log stream. It goes to stderr so it never pollutes a
// `-p` JSON run the child writes to stdout.
func printGuardBanner(w io.Writer, gwURL, provider, baseURL, floorSource, injectVar, injectVal, logLabel, auditLabel string, remoteServe bool, command []string) {
	fmt.Fprintf(w, "fak guard — kernel-adjudicated: %s\n", strings.Join(command, " "))
	fmt.Fprintf(w, "  gateway    : %s   (in-process; torn down when the command exits)\n", gwURL)
	if remoteServe {
		// Tell the operator the dev turn's INFERENCE is on the lab box they chose, not a
		// public API — the whole point of --remote-serve.
		fmt.Fprintf(w, "  upstream   : %s   (remote fak serve on a lab box, %s wire)\n", baseURL, provider)
	} else {
		fmt.Fprintf(w, "  upstream   : %s   (via the %s wire)\n", baseURL, provider)
	}
	fmt.Fprintf(w, "  floor      : %s\n", floorSource)
	fmt.Fprintf(w, "  wired via  : %s=%s   (child only — your shell is untouched)\n", injectVar, injectVal)
	// Observability: the live scrape surfaces are on the gateway URL above (unauth on
	// loopback); the audit journal is ON by default (auditLabel says where), the log
	// stream survives the session only if asked for.
	fmt.Fprintf(w, "  metrics    : %s/metrics  ·  %s/debug/vars  ·  %s/v1/fak/events\n", gwURL, gwURL, gwURL)
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
	// Make the provider prompt-cache reuse legible: with passthrough the client's
	// cache_control prefix survives the kernel hop byte-for-byte, so a daily `fak guard`
	// session reads most of its prompt from Anthropic's cache. Show it when it happened
	// so the operator sees the saving rather than assuming the hop re-bills the prefix.
	if sum.CachedPromptTokens > 0 {
		fmt.Fprintf(&b, "fak guard: provider cache - %d prompt token(s) the provider reported serving from its cache across %d turn(s) (OBSERVED, relayed; fak's part was preserving cache_control byte-for-byte through the hop, not the hit itself)\n",
			sum.CachedPromptTokens, sum.CachedTurns)
	}
	if sum.CompactionFired > 0 || sum.CompactionBailed > 0 || sum.CompactionOff > 0 {
		fmt.Fprintf(&b, "fak guard: compaction — %d fired, %d bailed, %d off; shed %d token(s), cache_read %d token(s) total, last post-fire cache_read %.0f\n",
			sum.CompactionFired,
			sum.CompactionBailed,
			sum.CompactionOff,
			sum.CompactionShedTokens,
			sum.CompactionCacheReadTokens,
			sum.LastCompactionCacheRead)
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

// guardUpstream is the resolved upstream wire + credential posture for `fak guard`: which
// provider the gateway proxies to, its base URL, the API key (if any), and — for Claude —
// whether to hold a Pro/Max subscription OAuth token upstream (pinUpstream) and where it
// came from (oauthSource).
type guardUpstream struct {
	provider     string
	autodetected bool
	baseURL      string
	apiKey       string
	pinUpstream  bool
	oauthSource  string
	// remoteServe is set when the base URL came from --remote-serve (a remote `fak serve`
	// on a lab box), so the banner can say "remote fak serve" instead of a generic
	// provider — the operator's signal that the dev turn's inference is on the lab GPU.
	remoteServe bool
	// passthroughFallback is set when the Anthropic subscription-OAuth auto-lookup found
	// no token and guard fell back to plain passthrough. That path works ONLY if the
	// wrapped agent (Claude Code) is itself logged in; cmdGuard surfaces a one-line note
	// so a cold agent that is ALSO not logged in gets a pointer home instead of an opaque
	// upstream 401 (issue #835, failure 2).
	passthroughFallback bool
	// ambientKeyOverridden is set when guard held the Pro/Max subscription OAuth token
	// upstream even though a bare ANTHROPIC_API_KEY was present in the environment. The
	// subscription is the default now regardless of that key — a global SDK key must not
	// silently bill the API account — so cmdGuard surfaces a one-line note pointing at the
	// explicit API-billing opt-in (--api-key-env ANTHROPIC_API_KEY) for discoverability.
	ambientKeyOverridden bool
}

// resolveGuardUpstream picks the upstream wire and credential posture: an explicit
// --provider wins, else the wire is inferred from the wrapped agent's name (anthropic as
// the fallback); the base URL defaults to the provider's public API. Subscription is the
// DEFAULT for Claude — when the upstream is Anthropic and no API key was EXPLICITLY named
// (--api-key-env), it sources the Pro/Max OAuth token and pins it upstream regardless of a
// bare ambient ANTHROPIC_API_KEY; --anthropic-oauth forces that and fails loud if no token
// is found. It exits(2) on an unresolvable base URL or OAuth misuse.
func resolveGuardUpstream(providerFlag, agentName, baseURLFlag, remoteServeBase, apiKeyEnv string, forceOAuth bool, oauthTokenEnv string) guardUpstream {
	// --remote-serve pins the OpenAI-compatible wire and the box's base URL: a remote
	// `fak serve` speaks the OpenAI routes the gateway proxies, and the caller has already
	// validated that it does not conflict with --provider/--base-url.
	remote := strings.TrimSpace(remoteServeBase) != ""
	if remote {
		providerFlag = "openai"
		baseURLFlag = strings.TrimSpace(remoteServeBase)
	}
	up, autodetected := resolveGuardProvider(providerFlag, agentName)
	resolvedBase := strings.TrimSpace(baseURLFlag)
	if resolvedBase == "" {
		resolvedBase = guardDefaultBaseURL(up)
	}
	if resolvedBase == "" {
		fmt.Fprintf(os.Stderr, "fak guard: provider %q has no public default base URL — pass --base-url\n", up)
		os.Exit(2)
	}
	apiKey := ""
	if apiKeyEnv != "" {
		apiKey = os.Getenv(apiKeyEnv)
	}

	// Subscription is the DEFAULT for Claude: whenever the upstream is Anthropic and no
	// API key was EXPLICITLY configured (--api-key-env), fak sources the Claude Pro/Max
	// OAuth token and sends it upstream as Authorization: Bearer + the oauth beta (the
	// scheme api.anthropic.com accepts an sk-ant-oat token under), holding the token
	// itself and ignoring the client's credential. A bare ANTHROPIC_API_KEY in the
	// environment NO LONGER flips this — a global SDK key must not silently bill your API
	// account when you hold a subscription. To opt INTO API billing, name the key
	// explicitly: --api-key-env ANTHROPIC_API_KEY. --anthropic-oauth forces the
	// subscription path and fails loud if no token is found.
	pinUpstream := false
	oauthSource := ""
	passthroughFallback := false
	ambientKeyOverridden := false
	if forceOAuth && up != "anthropic" {
		fmt.Fprintf(os.Stderr, "fak guard: --anthropic-oauth applies only to --provider anthropic (got %q)\n", up)
		os.Exit(2)
	}
	autoOAuth := up == "anthropic" && apiKey == ""
	if forceOAuth || autoOAuth {
		tok, src, terr := resolveAnthropicOAuthToken(oauthTokenEnv)
		switch {
		case terr == nil:
			apiKey = tok
			pinUpstream = true
			oauthSource = src
			// Held the subscription token despite a bare ANTHROPIC_API_KEY in the
			// environment: flag it so cmdGuard can make the override discoverable (the
			// user may have expected that key to bill their API account).
			ambientKeyOverridden = autoOAuth && os.Getenv("ANTHROPIC_API_KEY") != ""
		case forceOAuth:
			// Explicitly requested but nothing to use — fail loud.
			fmt.Fprintf(os.Stderr, "fak guard: --anthropic-oauth: %v\n", terr)
			os.Exit(2)
		default:
			// Auto attempt found no token: fall back to plain passthrough — the wrapped
			// agent's own credential (a subscription login OR ANTHROPIC_API_KEY) flows
			// upstream, so a pure API-billing user is unaffected.
			passthroughFallback = true
		}
	}
	return guardUpstream{
		provider: up, autodetected: autodetected, baseURL: resolvedBase,
		apiKey: apiKey, pinUpstream: pinUpstream, oauthSource: oauthSource,
		passthroughFallback: passthroughFallback, ambientKeyOverridden: ambientKeyOverridden,
		remoteServe: remote,
	}
}

// buildGuardChild constructs the wrapped-agent command with ONLY the gateway URL injected
// into its environment (never the parent shell). In pinned subscription mode it also hands
// the client a placeholder ANTHROPIC_API_KEY (when it has none) so it talks x-api-key to the
// gateway, which ignores the placeholder and authenticates upstream with the held token.
func buildGuardChild(command []string, injected [][2]string, pinUpstream bool, extraEnv ...[2]string) *exec.Cmd {
	// Landlock hook-floor (opt-in, Linux): rewrite the agent argv so the child is launched
	// through the fak re-exec trampoline, which applies the read-only-.git/hooks ruleset to
	// itself before exec'ing the agent. Off by default, no-op on non-Linux or when the hook
	// dirs cannot be resolved — the original command is used unchanged.
	command = maybeLandlockCommand(command)
	child := exec.Command(command[0], command[1:]...)
	child.Stdin, child.Stdout, child.Stderr = os.Stdin, os.Stdout, os.Stderr
	child.Env = os.Environ()
	for _, kv := range injected {
		child.Env = append(child.Env, kv[0]+"="+kv[1])
	}
	for _, kv := range extraEnv {
		if strings.TrimSpace(kv[0]) != "" {
			child.Env = append(child.Env, kv[0]+"="+kv[1])
		}
	}
	// Subscription mode: hand the client a PLACEHOLDER api key (only if it has none) so
	// it talks to the gateway in x-api-key mode; the gateway IGNORES the placeholder
	// (pinUpstream) and authenticates upstream with the real held OAuth token. Without it
	// the client may forward its own subscription bearer — also ignored in pinned mode —
	// so either way the held token is what reaches Anthropic.
	if pinUpstream && os.Getenv("ANTHROPIC_API_KEY") == "" {
		child.Env = append(child.Env, "ANTHROPIC_API_KEY=fak-guard-oauth-placeholder")
	}
	return child
}

// maybeLandlockCommand rewrites the agent argv to run through the fak Landlock trampoline
// when the hook-floor is opted in (guard.OptedIn) AND the host is Linux. It resolves the
// repo's git dir, work-tree root, and hooks dir with git's OWN resolution — never by string-
// concatenating "<root>/.git/hooks", which would break linked worktrees and submodules where
// .git is a file. On any miss — not opted in, not Linux, fak's own path unresolvable, no git
// dir, no hook dir to protect — it returns command unchanged (the floor degrades to today's
// behavior, never blocking the spawn). The trampoline itself fails open at runtime on a
// kernel without Landlock.
func maybeLandlockCommand(command []string) []string {
	if runtime.GOOS != "linux" || !guard.OptedIn(os.Getenv) {
		return command
	}
	fakBin, err := os.Executable()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak guard: landlock hook-floor not applied — cannot resolve fak binary (%v); spawning agent unrestricted\n", err)
		return command
	}
	gitOut := func(args ...string) string {
		out, err := exec.Command("git", args...).Output()
		if err != nil {
			return ""
		}
		return strings.TrimSpace(string(out))
	}
	gitDir := gitOut("rev-parse", "--absolute-git-dir")
	if gitDir == "" {
		fmt.Fprintln(os.Stderr, "fak guard: landlock hook-floor not applied — not in a git repo; spawning agent unrestricted")
		return command
	}
	repoRoot := gitOut("rev-parse", "--show-toplevel")
	hooksPath := gitOut("rev-parse", "--git-path", "hooks")
	bare := gitOut("rev-parse", "--is-bare-repository") == "true"

	spec := guard.ResolveSpec(repoRoot, gitDir, hooksPath, bare)
	if len(spec.ReadOnlyDirs) == 0 {
		fmt.Fprintln(os.Stderr, "fak guard: landlock hook-floor not applied — no hook dir resolved; spawning agent unrestricted")
		return command
	}
	return guard.TrampolineArgv(fakBin, spec, command)
}

type guardBudgetRestartEvent struct {
	Schema      string          `json:"schema"`
	FromTraceID string          `json:"from_trace_id"`
	ToTraceID   string          `json:"to_trace_id"`
	Reason      string          `json:"reason,omitempty"`
	SeedFile    string          `json:"seed_file,omitempty"`
	Seed        []agent.Message `json:"seed_messages,omitempty"`
	SeedText    string          `json:"seed_text,omitempty"`
	Note        string          `json:"note"`
}

type guardBudgetRestarter struct {
	enabled            bool
	freshContextTokens int
	limit              int
	seedDir            string
	stderr             io.Writer
	events             chan guardBudgetRestartEvent
}

func newGuardBudgetRestarter(enabled bool, freshContextTokens, limit int, seedDir string, stderr io.Writer) *guardBudgetRestarter {
	return &guardBudgetRestarter{
		enabled:            enabled,
		freshContextTokens: freshContextTokens,
		limit:              limit,
		seedDir:            strings.TrimSpace(seedDir),
		stderr:             stderr,
		events:             make(chan guardBudgetRestartEvent, 1),
	}
}

func (r *guardBudgetRestarter) Enabled() bool { return r != nil && r.enabled }

func (r *guardBudgetRestarter) OnBudgetExhausted(ctx context.Context, st gateway.SessionState, messages []agent.Message) {
	if !r.Enabled() || strings.TrimSpace(st.TraceID) == "" || strings.TrimSpace(st.ContinuationID) == "" {
		return
	}
	reset := resetServedSessionOnBudget(r.freshContextTokens)
	if reset == nil {
		return
	}
	nextTrace, seed, ok := reset(ctx, st.TraceID, messages)
	if !ok || strings.TrimSpace(nextTrace) == "" {
		return
	}
	ev := guardBudgetRestartEvent{
		Schema:      "fak.guard.budget_restart.v1",
		FromTraceID: st.TraceID,
		ToTraceID:   nextTrace,
		Reason:      st.Reason,
		Seed:        seed,
		SeedText:    guardSeedText(seed),
		Note:        "context budget exhausted; fak guard is relaunching the child under the continuation trace",
	}
	if path, err := writeGuardRestartSeedFile(r.seedDir, ev); err == nil {
		ev.SeedFile = path
	} else if r.stderr != nil {
		fmt.Fprintf(r.stderr, "fak guard: budget restart seed write failed: %v\n", err)
	}
	select {
	case r.events <- ev:
	default:
		if r.stderr != nil {
			fmt.Fprintf(r.stderr, "fak guard: budget restart event for %s dropped; restart already pending\n", st.TraceID)
		}
	}
}

func guardSeedText(seed []agent.Message) string {
	var parts []string
	for _, m := range seed {
		if c := strings.TrimSpace(m.Content); c != "" {
			parts = append(parts, c)
		}
	}
	return strings.Join(parts, "\n\n")
}

func writeGuardRestartSeedFile(dir string, ev guardBudgetRestartEvent) (string, error) {
	if strings.TrimSpace(dir) == "" {
		var err error
		dir, err = os.MkdirTemp("", "fak-guard-reset-*")
		if err != nil {
			return "", err
		}
	} else if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	name := "reset-" + guardSafeFilePart(ev.FromTraceID) + "-to-" + guardSafeFilePart(ev.ToTraceID) + ".json"
	path := filepath.Join(dir, name)
	ev.SeedFile = path
	raw, err := json.MarshalIndent(ev, "", "  ")
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, append(raw, '\n'), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

func guardSafeFilePart(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "trace"
	}
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	out := b.String()
	if out == "" {
		return "trace"
	}
	if len(out) > 80 {
		out = out[:80]
	}
	return out
}

func guardRestartEnv(ev guardBudgetRestartEvent) [][2]string {
	env := [][2]string{
		{"FAK_RESET_FROM_TRACE", ev.FromTraceID},
		{"FAK_RESET_TRACE_ID", ev.ToTraceID},
		{"FAK_SESSION_ID", ev.ToTraceID},
		{"FAK_RESET_REASON", ev.Reason},
	}
	if ev.SeedFile != "" {
		env = append(env, [2]string{"FAK_RESET_SEED_FILE", ev.SeedFile})
	}
	return env
}

// runGuardChildAndReport runs the wrapped agent to completion, tears the gateway down,
// prints the session's adjudication + journal summary (unless quiet), flushes the durable
// trail, and exits with the child's own code — surfacing a gateway-mid-session failure as
// a non-silent error so a clean child exit never hides a downed adjudication boundary.
func runGuardChildAndReport(child *exec.Cmd, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, agentName string) {
	runErr := child.Run()
	finishGuardChildAndReport(runErr, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName)
}

func runGuardChildSupervisedAndReport(command []string, injected [][2]string, pinUpstream bool, restarter *guardBudgetRestarter, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, agentName string) {
	var extraEnv [][2]string
	restarts := 0
	for {
		child := buildGuardChild(command, injected, pinUpstream, extraEnv...)
		wait := make(chan error, 1)
		if err := child.Start(); err != nil {
			finishGuardChildAndReport(err, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName)
			return
		}
		go func() { wait <- child.Wait() }()
		select {
		case runErr := <-wait:
			finishGuardChildAndReport(runErr, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName)
			return
		case ev := <-restarter.events:
			if restarter.limit > 0 && restarts >= restarter.limit {
				if restarter.stderr != nil {
					fmt.Fprintf(restarter.stderr, "fak guard: restart limit %d reached; leaving child on drained session %s\n", restarter.limit, ev.FromTraceID)
				}
				runErr := <-wait
				finishGuardChildAndReport(runErr, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName)
				return
			}
			restarts++
			if restarter.stderr != nil {
				fmt.Fprintf(restarter.stderr, "fak guard: context budget exhausted for %s; restarting child as %s\n", ev.FromTraceID, ev.ToTraceID)
				if ev.SeedFile != "" {
					fmt.Fprintf(restarter.stderr, "fak guard: carryover seed written to %s\n", ev.SeedFile)
				}
			}
			srv.SetDefaultTraceID(ev.ToTraceID)
			extraEnv = guardRestartEnv(ev)
			// Let the triggering response finish flushing to the wrapped client before
			// stopping the process that initiated it.
			time.Sleep(750 * time.Millisecond)
			stopGuardChild(child, wait, 2*time.Second)
		}
	}
}

func stopGuardChild(child *exec.Cmd, wait <-chan error, grace time.Duration) {
	if child == nil || child.Process == nil {
		return
	}
	_ = child.Process.Signal(os.Interrupt)
	select {
	case <-wait:
		return
	case <-time.After(grace):
		_ = child.Process.Kill()
		<-wait
	}
}

func finishGuardChildAndReport(runErr error, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, agentName string) {

	// Tear the gateway down and report what the kernel decided this session.
	cancel()
	serr := <-serveErr
	if !quiet {
		fmt.Fprintln(os.Stderr)
		fmt.Fprint(os.Stderr, formatAuditSummary(srv.AdjudicationSummary()))
		fmt.Fprint(os.Stderr, formatJournalSummary(auditJournal, auditSeq0))
	}
	// Flush + fsync the durable trail before exit so a row returned to the agent is
	// never lost to a buffered write (Close is safe on a nil/in-memory journal).
	if auditJournal != nil {
		_ = auditJournal.Close()
	}
	// Faithfully surface the child's exit code first (so `fak guard -- claude -p …`
	// scripts see what the agent returned).
	if runErr != nil {
		if ee, isExit := runErr.(*exec.ExitError); isExit {
			os.Exit(ee.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "fak guard: could not run %q: %v\n", agentName, runErr)
		os.Exit(1)
	}
	// The child succeeded — but if the gateway itself failed mid-session (Serve returned
	// something other than a clean shutdown), the adjudication boundary was down for part
	// of the run, so do not report a silent success. A clean teardown returns nil.
	if serr != nil && !errors.Is(serr, http.ErrServerClosed) && !errors.Is(serr, context.Canceled) {
		fmt.Fprintf(os.Stderr, "fak guard: gateway error during the session: %v\n", serr)
		os.Exit(1)
	}
}
