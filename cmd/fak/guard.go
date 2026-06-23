package main

import (
	"context"
	_ "embed"
	"flag"
	"fmt"
	"io"
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
		// Keep the child's terminal clean: gateway request logs are not the UX of an
		// interactive wrap. /metrics + /debug/vars still carry the full record.
		Logf: func(string, ...any) {},
	})
	must(err)

	// 4. Serve in the background. The gateway lives EXACTLY as long as the child: its
	//    context is cancelled when the agent exits. We deliberately do NOT tear it down
	//    on SIGINT — Ctrl-C belongs to the interactive child (it interrupts a turn), so
	//    the parent IGNORES SIGINT and lets the freshly-exec'd child own it.
	signal.Ignore(os.Interrupt)
	ctx, cancel := context.WithCancel(context.Background())
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ctx, ln) }()

	if err := guardWaitHealthy(gwURL, 5*time.Second); err != nil {
		cancel()
		<-serveErr
		fmt.Fprintf(os.Stderr, "fak guard: gateway did not become ready: %v\n", err)
		os.Exit(1)
	}
	srv.MarkReady()

	// 5. Wire the child: inject ONLY the gateway URL into the child's environment —
	//    never the parent shell, never settings.json. A `claude` in another terminal is
	//    untouched.
	injectVar := guardEnvVar(up, *envName)
	child := exec.Command(command[0], command[1:]...)
	child.Stdin, child.Stdout, child.Stderr = os.Stdin, os.Stdout, os.Stderr
	child.Env = append(os.Environ(), injectVar+"="+gwURL)

	if !*quiet {
		printGuardBanner(os.Stderr, gwURL, up, resolvedBase, floorSource, injectVar, command)
		if up == "anthropic" && *apiKeyEnv == "" && os.Getenv("ANTHROPIC_API_KEY") == "" {
			fmt.Fprintln(os.Stderr, "fak guard: note — ANTHROPIC_API_KEY is unset; Claude Code needs an API key when pointed at a custom")
			fmt.Fprintln(os.Stderr, "           base URL (subscription OAuth is not forwarded). Export your key, or pass --policy/--provider.")
		}
	}

	runErr := child.Run()

	// 6. Tear the gateway down and report what the kernel decided this session.
	cancel()
	<-serveErr
	if !*quiet {
		fmt.Fprintln(os.Stderr)
		fmt.Fprint(os.Stderr, formatAuditSummary(srv.AdjudicationSummary()))
	}
	// Faithfully surface the child's exit code (so `fak guard -- claude -p …` scripts).
	if runErr != nil {
		if ee, isExit := runErr.(*exec.ExitError); isExit {
			os.Exit(ee.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "fak guard: could not run %q: %v\n", command[0], runErr)
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

// guardWaitHealthy polls the gateway's unauthenticated /healthz until it answers 200
// or the deadline passes. The listener is already bound (Serve got it pre-bound), so
// this normally returns on the first poll; the loop just covers the goroutine-start
// gap.
func guardWaitHealthy(gwURL string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 500 * time.Millisecond}
	var lastErr error
	for time.Now().Before(deadline) {
		resp, err := client.Get(gwURL + "/healthz")
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
			lastErr = fmt.Errorf("healthz returned %d", resp.StatusCode)
		} else {
			lastErr = err
		}
		time.Sleep(50 * time.Millisecond)
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("timed out after %s", timeout)
	}
	return lastErr
}

// printGuardBanner explains, in five lines, exactly what is now in front of the agent:
// where the gateway is, what it proxies to, which floor is loaded, and the single env
// var injected into the child. It goes to stderr so it never pollutes a `-p` JSON run
// the child writes to stdout.
func printGuardBanner(w io.Writer, gwURL, provider, baseURL, floorSource, injectVar string, command []string) {
	fmt.Fprintf(w, "fak guard — kernel-adjudicated: %s\n", strings.Join(command, " "))
	fmt.Fprintf(w, "  gateway    : %s   (in-process; torn down when the command exits)\n", gwURL)
	fmt.Fprintf(w, "  upstream   : %s   (via the %s wire)\n", baseURL, provider)
	fmt.Fprintf(w, "  floor      : %s\n", floorSource)
	fmt.Fprintf(w, "  wired via  : %s   (child only — your shell is untouched)\n", injectVar)
	fmt.Fprintln(w, "  every tool call the agent proposes crosses the capability floor before it runs.")
}

// formatAuditSummary renders the exit roll-up of what the kernel decided while the
// agent ran. It is one honest tally: every count came from the same operation counters
// /metrics exposes, so the line can never overstate the protection.
func formatAuditSummary(sum gateway.AdjudicationSummary) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak guard: %d tool-call decision(s) — %d allowed, %d denied, %d repaired, %d quarantined",
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
