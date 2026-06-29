package main

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// guard_provider.go — upstream provider + base-URL resolution and the child
// environment wiring (env-var injection, timeout floors, loopback/health waits)
// for `fak guard`. Split out of guard.go to keep the dispatch surface readable.

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

// guardLocalModelDecision decides whether `fak guard` should run a LOCAL in-kernel model
// (fak loading the weights into its own engine) and validates that the local-model flag
// does not collide with an upstream-proxy flag. ggufRef is the raw --gguf value (a
// non-empty value requests local mode); baseURL is --base-url; remoteServe is the
// normalized --remote-serve base. It returns local=true when local mode is requested, and
// a non-empty conflict message (local still true) when the operator ALSO named an upstream —
// the two are mutually exclusive because a local in-kernel model IS the upstream. Pure (no
// I/O, no exit) so the precedence is unit-tested without standing a model up.
func guardLocalModelDecision(ggufRef, baseURL, remoteServe string) (local bool, conflict string) {
	if strings.TrimSpace(ggufRef) == "" {
		return false, ""
	}
	if strings.TrimSpace(remoteServe) != "" {
		return true, "--gguf (local in-kernel model) and --remote-serve are mutually exclusive — the local model IS the upstream; pass only one"
	}
	if strings.TrimSpace(baseURL) != "" {
		return true, "--gguf (local in-kernel model) and --base-url are mutually exclusive — the local model IS the upstream; pass only one"
	}
	return true, ""
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

// guardOpenAIV1Base appends the "/v1" the OpenAI-compatible wire expects to a bare remote
// base (http://HOST:PORT -> http://HOST:PORT/v1), so the proxy planner's
// adapter.Endpoint(base, model) lands on the remote `fak serve`'s registered
// /v1/chat/completions route instead of a 404'ing /chat/completions. It is the
// upstream-base twin of guardEnvValue (which adds /v1 to the CHILD's OPENAI_BASE_URL).
// Idempotent: a base that already ends in /v1 (e.g. an operator who typed the long-form
// --base-url http://HOST:PORT/v1 — though --remote-serve and --base-url are mutually
// exclusive, this keeps the helper safe to reuse) is returned unchanged. A trailing slash
// is trimmed first so "http://host:8080/" yields "http://host:8080/v1", not ".../v1".
func guardOpenAIV1Base(base string) string {
	b := strings.TrimRight(strings.TrimSpace(base), "/")
	if b == "" || strings.HasSuffix(b, "/v1") {
		return b
	}
	return b + "/v1"
}

// guardPreflightRemoteServe confirms a remote `fak serve` is actually answering — and that
// it exposes the OpenAI /v1 SURFACE the proxy hop will use — before guard binds its own
// gateway and execs the agent. A down box (not started, wrong port) is the most common
// --remote-serve failure, and a 404/502 on the first real turn is a far worse place to
// discover it than a one-line fail-loud here.
//
// It probes two routes off the BARE base (http://HOST:PORT, no /v1):
//
//  1. GET <base>/healthz — the root liveness route `fak serve` exposes. Any HTTP response
//     (even a 401) proves a live gateway; only a connection-level failure is fatal.
//  2. GET <base>/v1/models — the OpenAI route `fak serve` registers ALONGSIDE
//     /v1/chat/completions. This is the witness that the /v1 surface the proxy POSTs to is
//     actually mounted: a clean 404 here means the box answers health but does NOT serve
//     the /v1 routes (an older/mis-started serve, or not a fak serve at all), which is
//     exactly the silent "health passes, chat 404s mid-session" trap. Any non-404 response
//     (200, 401, 405) proves the route exists, so only a 404 — or a connection failure — is
//     fatal.
func guardPreflightRemoteServe(baseURL string) error {
	base := strings.TrimRight(baseURL, "/")
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(base + "/healthz")
	if err != nil {
		return err
	}
	_ = resp.Body.Close()
	// Confirm the /v1 surface the proxy hop will use is mounted, not just root health.
	mresp, merr := client.Get(base + "/v1/models")
	if merr != nil {
		return merr
	}
	_ = mresp.Body.Close()
	if mresp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("box answers /healthz but /v1/models is 404 — it is not serving the OpenAI /v1 surface (start it with `fak serve --gguf <weights>`, or check it is a fak serve)")
	}
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

// guardStallFloorS is the streaming IDLE-read deadline (in seconds) a guarded session pins
// — deliberately small and INDEPENDENT of guardTimeoutFloorS. It bounds how long a streamed
// upstream may go SILENT mid-turn before the read is aborted, so a stalled API fails fast
// instead of riding the 600s whole-request floor to a ten-minute hang. It must stay well
// under guardTimeoutFloorS and comfortably above the provider's ping/keepalive cadence.
const guardStallFloorS = 60

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
