package main

import (
	"context"
	_ "embed"
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
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/gateway"
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
// `fak guard -- claude` wraps your normal Claude Code: your own key and prompt-cache
// breakpoints flow through untouched (the gateway forwards the request bytes verbatim),
// but every tool call Claude proposes crosses the capability floor first.
func cmdGuard(argv []string) {
	t0 := time.Now()
	fs := flag.NewFlagSet("guard", flag.ExitOnError)
	addr := fs.String("addr", "", "gateway listen address (default: a private 127.0.0.1 port the OS picks)")
	provider := fs.String("provider", "anthropic", "upstream wire the gateway proxies to: anthropic|openai|gemini|xai")
	baseURL := fs.String("base-url", "", "upstream provider base URL (default: the provider's public API, e.g. anthropic -> https://api.anthropic.com)")
	model := fs.String("model", "", "upstream model id override (default: forward the client's own model id)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the UPSTREAM API key (default: forward the client's own key — passthrough)")
	policyPath := fs.String("policy", "", "capability-floor manifest to enforce (default: the built-in guard floor; see --dump-policy)")
	envName := fs.String("env", "", "env var to inject the gateway URL into the child (default: chosen by --provider)")
	requireKeyEnv := fs.String("require-key-env", "", "require this env var's bearer token on the gateway (loopback rarely needs it)")
	logPath := fs.String("log", "", "write the gateway's per-request + per-verdict structured logs to this file (or '-' for stderr); default off to keep the agent's terminal clean")
	dumpPolicy := fs.Bool("dump-policy", false, "print the built-in guard capability floor (an editable manifest) and exit")
	quiet := fs.Bool("quiet", false, "suppress the startup banner and the exit audit summary")
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

	// 2. Resolve the upstream the gateway proxies to (and the key handling).
	up := strings.ToLower(strings.TrimSpace(*provider))
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
		EngineID:     "inkernel",
		Model:        *model,
		BaseURL:      resolvedBase,
		Provider:     up,
		APIKey:       apiKey,
		RequireKey:   requireKey,
		VDSO:         true,
		Invalidation: "global",
		Version:      appversion.Current(),
		ReloadPolicy: policyReloader(*policyPath),
		ResetTrace:   resetTrace,
		ObserveTrace: observeTrace,
		StartTime:    t0,
		// Default OFF (clean terminal); --log routes the full structured stream to a file
		// or stderr. /metrics + /debug/vars + the audit journal carry the record regardless.
		Logf: gwLogf,
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
	injectVar := guardEnvVar(up, *envName)
	injectVal := guardEnvValue(up, gwURL)
	child := exec.Command(command[0], command[1:]...)
	child.Stdin, child.Stdout, child.Stderr = os.Stdin, os.Stdout, os.Stderr
	child.Env = append(os.Environ(), injectVar+"="+injectVal)

	if !*quiet {
		printGuardBanner(os.Stderr, gwURL, up, resolvedBase, floorSource, injectVar, injectVal, logLabel, os.Getenv("FAK_AUDIT_JOURNAL"), command)
		if up == "anthropic" && *apiKeyEnv == "" && os.Getenv("ANTHROPIC_API_KEY") == "" {
			fmt.Fprintln(os.Stderr, "fak guard: note — ANTHROPIC_API_KEY is unset; Claude Code needs an API key when pointed at a custom")
			fmt.Fprintln(os.Stderr, "           base URL (subscription OAuth is not forwarded). Export your key, or pass --policy/--provider.")
		}
	}

	runErr := child.Run()

	// 6. Tear the gateway down and report what the kernel decided this session.
	cancel()
	serr := <-serveErr
	if !*quiet {
		fmt.Fprintln(os.Stderr)
		fmt.Fprint(os.Stderr, formatAuditSummary(srv.AdjudicationSummary()))
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
func printGuardBanner(w io.Writer, gwURL, provider, baseURL, floorSource, injectVar, injectVal, logLabel, journalPath string, command []string) {
	fmt.Fprintf(w, "fak guard — kernel-adjudicated: %s\n", strings.Join(command, " "))
	fmt.Fprintf(w, "  gateway    : %s   (in-process; torn down when the command exits)\n", gwURL)
	fmt.Fprintf(w, "  upstream   : %s   (via the %s wire)\n", baseURL, provider)
	fmt.Fprintf(w, "  floor      : %s\n", floorSource)
	fmt.Fprintf(w, "  wired via  : %s=%s   (child only — your shell is untouched)\n", injectVar, injectVal)
	// Observability: the live scrape surfaces are on the gateway URL above (unauth on
	// loopback); the audit journal and log stream survive the session only if asked for.
	fmt.Fprintf(w, "  metrics    : %s/metrics  ·  %s/debug/vars  ·  %s/v1/fak/events\n", gwURL, gwURL, gwURL)
	journal := journalPath
	if strings.TrimSpace(journal) == "" {
		journal = "off (set FAK_AUDIT_JOURNAL=/path/audit.jsonl for a durable hash-chained trail)"
	}
	fmt.Fprintf(w, "  audit log  : %s\n", journal)
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
	if sum.Errored > 0 {
		fmt.Fprintf(&b, ", %d errored", sum.Errored)
	}
	b.WriteByte('\n')
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
