package main

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"time"
)

// guard_router.go — the OPT-IN `fak guard --router` upstream: point a guarded Claude
// session's model traffic at the fak router (the local `fak serve` / fak-server routing
// gateway, default http://127.0.0.1:8080) instead of api.anthropic.com. The router serves
// the Anthropic Messages wire (/v1/messages) and authenticates with the gateway key as
// `Authorization: Bearer`, so this path:
//
//   - keeps the Anthropic wire (Claude Code is unchanged; the gateway still adjudicates
//     every proposed tool call locally),
//   - PINS the upstream credential so the wrapped agent's own Claude login is never
//     forwarded to the router, and never sources the Pro/Max subscription token at all,
//   - presents the router key only as the bearer the router checks.
//
// Without --router none of this runs: the default `fak guard -- claude` resolution
// (subscription OAuth to api.anthropic.com) is untouched.

// guardRouterKeyEnv is the router's gateway-key variable, shared with `fak agent`'s
// router auto-connect (resolveRouterAPIKey).
const guardRouterKeyEnv = "FAK_GATEWAY_KEY"

// guardRouterUpstream is the resolved --router target.
type guardRouterUpstream struct {
	origin    string // router origin, no trailing slash or /v1 (the Anthropic adapter appends /v1/messages)
	key       string // router gateway key; "" when the router runs keyless
	keySource string // env var the key came from, for the banner ("" when keyless)
}

// guardRouterInputs are the flag values --router composes with.
type guardRouterInputs struct {
	provider       string // resolved wire (after agent-name inference)
	baseURL        string // --base-url: an explicit router origin
	remoteServe    string // --remote-serve (conflict)
	apiKeyEnv      string // --api-key-env: explicit router key variable
	anthropicOAuth bool   // --anthropic-oauth (conflict)
	localAuto      bool   // --local (conflict)
	ggufPath       string // --gguf (conflict)
}

// resolveGuardRouterUpstream validates the --router flag combination and resolves the
// router origin and key. Pure apart from getenv, so the precedence is unit-tested:
// origin = --base-url, else routerOrigin() (FAK_AGENT_ROUTER_ORIGIN, else the default
// local router); key = the --api-key-env variable (which must be non-empty when named),
// else FAK_GATEWAY_KEY, else keyless.
func resolveGuardRouterUpstream(in guardRouterInputs, getenv func(string) string) (guardRouterUpstream, error) {
	switch {
	case strings.TrimSpace(in.remoteServe) != "":
		return guardRouterUpstream{}, errors.New("--router and --remote-serve both pick the upstream — pass only one")
	case in.localAuto:
		return guardRouterUpstream{}, errors.New("--router and --local both pick the upstream — pass only one")
	case strings.TrimSpace(in.ggufPath) != "":
		return guardRouterUpstream{}, errors.New("--router and --gguf both pick the upstream — pass only one")
	case in.anthropicOAuth:
		return guardRouterUpstream{}, errors.New("--router never sends the Claude subscription token to the router; drop --anthropic-oauth")
	case in.provider != "anthropic":
		return guardRouterUpstream{}, fmt.Errorf("--router currently supports the Anthropic wire (claude) only, got provider %q", in.provider)
	}
	origin := strings.TrimSpace(in.baseURL)
	if origin == "" {
		origin = routerOrigin()
	}
	origin = strings.TrimRight(strings.TrimSuffix(strings.TrimRight(origin, "/"), "/v1"), "/")
	if !strings.HasPrefix(origin, "http://") && !strings.HasPrefix(origin, "https://") {
		return guardRouterUpstream{}, fmt.Errorf("router origin %q must be an http:// or https:// URL", origin)
	}
	up := guardRouterUpstream{origin: origin}
	if env := strings.TrimSpace(in.apiKeyEnv); env != "" {
		up.key, up.keySource = strings.TrimSpace(getenv(env)), env
		if up.key == "" {
			return guardRouterUpstream{}, fmt.Errorf("--api-key-env %s is set but that env var is empty — export the router key or drop the flag", env)
		}
		return up, nil
	}
	if key := strings.TrimSpace(getenv(guardRouterKeyEnv)); key != "" {
		up.key, up.keySource = key, guardRouterKeyEnv
	}
	return up, nil
}

// headers returns the upstream headers carrying the router key: the router accepts only
// `Authorization: Bearer`. nil when keyless, so no auth header is sent at all.
func (u guardRouterUpstream) headers() map[string]string {
	if u.key == "" {
		return nil
	}
	return map[string]string{"Authorization": "Bearer " + u.key}
}

// posture is the upstream posture for --router: the Anthropic wire at the router origin,
// the credential PINNED (the child's own Claude login is ignored and never forwarded),
// no static API key (the router key rides only in the bearer header), and none of the
// subscription token / account-failover machinery.
func (u guardRouterUpstream) posture(p guardUpstreamPosture) guardUpstreamPosture {
	p.up = "anthropic"
	p.resolvedBase = u.origin
	p.pinUpstream = true
	p.extraHeaders = u.headers()
	return p
}

// guardPreflightRouter fails loud before the gateway binds when the router is not
// answering /healthz, instead of 502ing on the first real turn.
func guardPreflightRouter(origin string) error {
	_, err := claudeStatusFetcher(origin, 2*time.Second)
	return err
}

// guardRouterBanner is the one-line launch note naming where model traffic goes.
func guardRouterBanner(u guardRouterUpstream) string {
	return fmt.Sprintf("fak guard: --router: model traffic goes to the fak router at %s (Anthropic /v1/messages; %s)", u.origin, u.authPhrase())
}

// guardRouterAuthLine is the startup report's upstream-auth line for --router, in place
// of the subscription line the pinned Anthropic posture would otherwise print.
func guardRouterAuthLine(u guardRouterUpstream) string {
	return "fak guard: upstream auth — fak router at " + u.origin + " (" + u.authPhrase() + ")"
}

func (u guardRouterUpstream) authPhrase() string {
	if u.keySource == "" {
		return "keyless router; the Claude subscription token is not sent"
	}
	return "router key from $" + u.keySource + " as a bearer token; the Claude subscription token is not sent"
}

// guardRouterFail prints a --router error and exits 2, matching the other upstream
// flag conflicts in cmdGuard.
func guardRouterFail(err error) {
	fmt.Fprintf(os.Stderr, "fak guard: %v\n", err)
	os.Exit(2)
}
