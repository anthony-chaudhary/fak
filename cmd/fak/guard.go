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
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// guardDefaultPolicyJSON is the day-to-day capability floor `fak guard` enforces when
// the operator names no --policy. It is embedded in the binary so `fak guard` works
// from ANY directory (a repo or not, an installed binary with no source tree). It
// allows the standard coding-agent tool set and denies the genuine-danger classes:
// destructive removal, privilege escalation, disk wipe, fork bomb, RCE pipe, writes
// that escape the working tree, and writes into credential/SSH/secret paths. Print or
// fork it with `fak guard --dump-policy`.
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
// --provider anthropic with no API key set, fak uses your Claude Pro/Max SUBSCRIPTION
// by default — it sources the OAuth token and sends it upstream as a bearer token
// (set ANTHROPIC_API_KEY to use API billing instead; --anthropic-oauth forces it).
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
	model := fs.String("model", "", "upstream model id override (default: forward the client's own model id)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the UPSTREAM API key (default: forward the client's own key — passthrough)")
	anthropicOAuth := fs.Bool("anthropic-oauth", false, "force the Claude Pro/Max SUBSCRIPTION OAuth token upstream (sourced from CLAUDE_CODE_OAUTH_TOKEN, <claude-config>/.oauth-token, or ~/.claude/.credentials.json) sent as Authorization: Bearer + the oauth beta. This is ALREADY the default for --provider anthropic when no API key is set; the flag forces it and fails loud if no token is found.")
	oauthTokenEnv := fs.String("oauth-token-env", "CLAUDE_CODE_OAUTH_TOKEN", "env var to read the subscription OAuth token from first")
	policyPath := fs.String("policy", "", "capability-floor manifest to enforce (default: the built-in guard floor; see --dump-policy)")
	envName := fs.String("env", "", "env var to inject the gateway URL into the child (default: chosen by --provider)")
	requireKeyEnv := fs.String("require-key-env", "", "require this env var's bearer token on the gateway (loopback rarely needs it)")
	logPath := fs.String("log", "", "write the gateway's per-request + per-verdict structured logs to this file (or '-' for stderr); default off to keep the agent's terminal clean")
	auditPath := fs.String("audit", "", "write the durable, hash-chained DECISION JOURNAL to this file (default: a per-user path under your config dir; pass 'off' to disable). Every kernel verdict this session is appended as a tamper-evident JSONL row you can later replay with `fak audit verify`.")
	noAudit := fs.Bool("no-audit", false, "disable the durable decision journal for this session (it is ON by default — fak guard is the referee, and the journal is the verifiable record of what it allowed vs blocked)")
	dumpPolicy := fs.Bool("dump-policy", false, "print the built-in guard capability floor (an editable manifest) and exit")
	quiet := fs.Bool("quiet", false, "suppress the startup banner and the exit audit summary")
	ctxViewBudget := fs.Int("ctx-view-budget", 0, "wire the ctxplan context PLANNER into the live guard loop: each buffered turn, re-materialize the forwarded history as an O(1) planned VIEW under this resident-token budget (a planned view in place of appending the whole transcript, #555). 0 (default) leaves the existing path byte-for-byte unchanged. OFF by default: it rewrites in-flight turn history, so gate it until you have watched a wrapped session. The streaming fast-path bypasses this; the buffered turn path is what gets planned.")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak guard [flags] -- <agent command...>")
		fmt.Fprintln(os.Stderr, "  e.g. fak guard -- claude")
		fmt.Fprintln(os.Stderr, "       fak guard --provider openai -- codex")
		fmt.Fprintln(os.Stderr, "       fak guard --policy my-floor.json -- claude")
		fs.PrintDefaults()
	}
	_ = fs.Parse(argv)

	if *dumpPolicy {
		os.Stdout.Write(guardDefaultPolicyJSON)
		return
	}

	command := fs.Args() // everything after the flags (and after `--`) is the wrapped agent.
	if len(command) == 0 {
		fs.Usage()
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

	// 2. Resolve the upstream the gateway proxies to (and the key handling). An explicit
	//    --provider wins; otherwise the wire is inferred from the wrapped agent's name so
	//    naming a known agent Just Works, with anthropic (Claude Code) as the fallback.
	up, providerAutodetected := resolveGuardProvider(*provider, command[0])
	resolvedBase := strings.TrimSpace(*baseURL)
	if resolvedBase == "" {
		resolvedBase = guardDefaultBaseURL(up)
	}
	if resolvedBase == "" {
		fmt.Fprintf(os.Stderr, "fak guard: provider %q has no public default base URL — pass --base-url\n", up)
		os.Exit(2)
	}
	apiKey := ""
	if *apiKeyEnv != "" {
		apiKey = os.Getenv(*apiKeyEnv)
	}

	// Subscription is the DEFAULT for Claude: when the upstream is Anthropic and no
	// API key is configured, fak sources the Claude Pro/Max OAuth token and sends it
	// upstream as Authorization: Bearer + the oauth beta (the scheme api.anthropic.com
	// accepts an sk-ant-oat token under), holding the token itself and ignoring the
	// client's credential. A configured API key (--api-key-env / ANTHROPIC_API_KEY)
	// opts out (use API billing); --anthropic-oauth forces the subscription path and
	// fails loud if no token is found.
	pinUpstream := false
	oauthSource := ""
	if *anthropicOAuth && up != "anthropic" {
		fmt.Fprintf(os.Stderr, "fak guard: --anthropic-oauth applies only to --provider anthropic (got %q)\n", up)
		os.Exit(2)
	}
	autoOAuth := up == "anthropic" && apiKey == "" && os.Getenv("ANTHROPIC_API_KEY") == ""
	if *anthropicOAuth || autoOAuth {
		tok, src, terr := resolveAnthropicOAuthToken(*oauthTokenEnv)
		switch {
		case terr == nil:
			apiKey = tok
			pinUpstream = true
			oauthSource = src
		case *anthropicOAuth:
			// Explicitly requested but nothing to use — fail loud.
			fmt.Fprintf(os.Stderr, "fak guard: --anthropic-oauth: %v\n", terr)
			os.Exit(2)
		default:
			// Auto attempt found no token (normal for an API-key user): fall back to
			// plain passthrough — Claude Code forwards its own bearer if it is logged in.
		}
	}

	requireKey, ok := resolveRequiredKey(*requireKeyEnv, os.Getenv)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak guard: --require-key-env %s is set but empty — refusing to start a gateway with NO authentication (set it or drop the flag)\n", *requireKeyEnv)
		os.Exit(2)
	}

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
		StartTime:             t0,
		// Default OFF (clean terminal); --log routes the full structured stream to a file
		// or stderr. /metrics + /debug/vars + the audit journal carry the record regardless.
		Logf:          gwLogf,
		CtxViewBudget: *ctxViewBudget,
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
	child := exec.Command(command[0], command[1:]...)
	child.Stdin, child.Stdout, child.Stderr = os.Stdin, os.Stdout, os.Stderr
	child.Env = os.Environ()
	for _, kv := range injected {
		child.Env = append(child.Env, kv[0]+"="+kv[1])
	}

	// Subscription mode: put the wrapped client in a deterministic state by handing it
	// a PLACEHOLDER api key (only if it has none). Claude Code then talks to the gateway
	// in x-api-key mode; the gateway IGNORES that placeholder (pinUpstream) and
	// authenticates upstream with the real held OAuth token. Without this the client may
	// instead forward its own subscription bearer — which the gateway also ignores in
	// pinned mode — so either way the held token is what reaches Anthropic.
	if pinUpstream && os.Getenv("ANTHROPIC_API_KEY") == "" {
		child.Env = append(child.Env, "ANTHROPIC_API_KEY=fak-guard-oauth-placeholder")
	}

	if !*quiet {
		if providerAutodetected {
			fmt.Fprintf(os.Stderr, "fak guard: detected agent %q -> --provider %s (pass --provider to override)\n", strings.ToLower(filepath.Base(command[0])), up)
		}
		injectNames := injected[0][0]
		for _, kv := range injected[1:] {
			injectNames += ", " + kv[0]
		}
		printGuardBanner(os.Stderr, gwURL, up, resolvedBase, floorSource, injectNames, injected[0][1], logLabel, auditLabel, command)
		switch {
		case pinUpstream:
			fmt.Fprintf(os.Stderr, "fak guard: upstream auth — Claude Pro/Max subscription (OAuth token from %s, sent as a bearer token)\n", oauthSource)
		case up == "anthropic":
			fmt.Fprintln(os.Stderr, "fak guard: upstream auth — passthrough (Claude Code forwards its own credential through the gateway)")
		}
	}

	runErr := child.Run()

	// 6. Tear the gateway down and report what the kernel decided this session.
	cancel()
	serr := <-serveErr
	if !*quiet {
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
		fmt.Fprintf(os.Stderr, "fak guard: could not run %q: %v\n", command[0], runErr)
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

// resolveAnthropicOAuthToken finds a Claude Pro/Max SUBSCRIPTION OAuth token to
// authenticate the upstream with, in priority order:
//  1. the named env var (default CLAUDE_CODE_OAUTH_TOKEN) — a long-lived
//     `claude setup-token` credential, the headless-friendly source;
//  2. <claude-config>/.oauth-token — a long-lived setup-token file (what the fleet
//     pins; preferred over the expiring interactive creds);
//  3. <claude-config>/.credentials.json -> claudeAiOauth.accessToken — the
//     interactive login token. This one EXPIRES; Claude Code refreshes it
//     out-of-band (directly against Anthropic, not through the gateway), so a
//     long session may outlive the snapshot fak read at startup — prefer a setup
//     token for anything long-running.
//
// <claude-config> is $CLAUDE_CONFIG_DIR (first entry if it is a list) when set,
// else ~/.claude. Returns the token and a human source label, or an error that
// names every place it looked so the operator can fix the setup.
func resolveAnthropicOAuthToken(tokenEnv string) (token, source string, err error) {
	tried := make([]string, 0, 3)

	if tokenEnv != "" {
		tried = append(tried, "$"+tokenEnv)
		if v := strings.TrimSpace(os.Getenv(tokenEnv)); v != "" {
			return v, "$" + tokenEnv, nil
		}
	}

	cfgDir := guardClaudeConfigDir()

	setupPath := filepath.Join(cfgDir, ".oauth-token")
	tried = append(tried, setupPath)
	if b, rerr := os.ReadFile(setupPath); rerr == nil {
		if v := strings.TrimSpace(string(b)); v != "" {
			return v, setupPath, nil
		}
	}

	credPath := filepath.Join(cfgDir, ".credentials.json")
	tried = append(tried, credPath)
	if b, rerr := os.ReadFile(credPath); rerr == nil {
		var doc struct {
			ClaudeAIOauth struct {
				AccessToken string `json:"accessToken"`
				ExpiresAt   int64  `json:"expiresAt"`
			} `json:"claudeAiOauth"`
		}
		if json.Unmarshal(b, &doc) == nil {
			if v := strings.TrimSpace(doc.ClaudeAIOauth.AccessToken); v != "" {
				label := credPath
				if exp := doc.ClaudeAIOauth.ExpiresAt; exp > 0 && exp < time.Now().UnixMilli() {
					fmt.Fprintf(os.Stderr, "fak guard: WARNING — the OAuth token in %s expired; Claude Code normally refreshes it. Re-run `claude` once, or use `claude setup-token` for a long-lived token.\n", credPath)
				}
				return v, label, nil
			}
		}
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
func printGuardBanner(w io.Writer, gwURL, provider, baseURL, floorSource, injectVar, injectVal, logLabel, auditLabel string, command []string) {
	fmt.Fprintf(w, "fak guard — kernel-adjudicated: %s\n", strings.Join(command, " "))
	fmt.Fprintf(w, "  gateway    : %s   (in-process; torn down when the command exits)\n", gwURL)
	fmt.Fprintf(w, "  upstream   : %s   (via the %s wire)\n", baseURL, provider)
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
		fmt.Fprintf(&b, "fak guard: provider cache — %d prompt token(s) served from cache across %d turn(s) (cache_control preserved through the kernel hop)\n",
			sum.CachedPromptTokens, sum.CachedTurns)
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
