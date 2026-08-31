package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

// guardUpstreamPostureInputs is the launch-time input to step 3 of a `fak guard` launch:
// the flag values that select the upstream wire plus the already-decided in-kernel model
// posture (--gguf / --alongside) and the already-resolved --remote-serve base. Grouped in
// one struct so the resolution reads as a single concern instead of a dozen positional
// parameters threaded through cmdGuard.
type guardUpstreamPostureInputs struct {
	command        []string
	profile        harnessprofile.HarnessProfile
	provider       string
	baseURL        string
	remoteBase     string
	apiKeyEnv      string
	anthropicOAuth bool
	oauthTokenEnv  string
	model          string
	codexHome      string
	quiet          bool
	localModel     bool
	localAlongside bool
}

// guardUpstreamPosture is what the upstream resolution hands back to the rest of the
// launch: the resolved wire and base URL, the credential (static, and/or re-resolved per
// request), and the account-failover controls the gateway and the live status area consult
// later. Every field is zero on the LOCAL-ONLY path, which has no upstream at all.
type guardUpstreamPosture struct {
	up                   string
	providerAutodetected bool
	resolvedBase         string
	apiKey               string
	pinUpstream          bool
	oauthSource          string
	// keychainAPIKey marks apiKey as Claude Code's saved API key adopted from the
	// macOS Keychain (#5363) — not an --api-key-env value — so the startup report's
	// auth line and the managed-cache reason can name the real source.
	keychainAPIKey bool
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
	apiKeyFunc       func() string
	extraHeaders     map[string]string
	extraHeadersFunc func() map[string]string
	// accountFailoverFunc, when set on the pinned path, supplies a permitted sibling
	// account's live token when the current account hits an ACCOUNT-SCOPED 403 wall (org
	// OAuth disabled / region / billing). It also stickily advances the config dir apiKeyFunc
	// reads from, so a walled session heals onto a working account and stays there.
	accountFailoverFunc func(reason string) (string, bool)
	// transientTargetFunc supplies a sibling for temporary 5xx/529 failover without
	// adding the current account to the permanent walled set.
	transientTargetFunc func(status int) (string, bool)
	// activeAccountDir/walledAccounts feed the live accounts+nodes status
	// area (guard_endpoints.go): the config dir of the seat currently serving turns
	// (it follows a failover) and the seats an account-scoped 403 walled this session.
	// Set only on the pinned Claude subscription path; nil elsewhere (a non-subscription
	// session has no seat "in use", so the status area shows nodes only).
	activeAccountDir func() string
	walledAccounts   func() map[string]bool
	// accountRehome, when set on the pinned path, is the operator "switch seat
	// now" function the gateway serves at POST /v1/fak/account/rehome (`fak accounts
	// rehome`) — the on-demand form of the failover above.
	accountRehome func(reason string) (gateway.AccountRehome, error)
	// upstreamTrustNote is the startup-report line naming the trust store this session
	// validates with, empty when no corporate CA bundle is declared (#8172). Carried on
	// the posture rather than printed at gate time so it lands in the durable startup
	// report and obeys --banner.
	upstreamTrustNote string
	// cloudRouteWaived records that a request-signed cloud route was detected and
	// the operator waived the UPSTREAM_UNSUPPORTED refusal (#8172). The session runs
	// with the hook floor, tool brokering, transcript, and sandbox intact but fak
	// sees NONE of its model traffic, so the startup report must not claim upstream
	// adjudication it is not performing.
	cloudRouteWaived bool
}

// resolveGuardUpstreamPosture resolves the upstream wire + credential posture. Two worlds:
//
//	LOCAL-ONLY (--gguf, no --alongside): fak runs the model itself in-kernel, so
//	there is NO upstream API, no API key, and no OAuth. Resolve ONLY the wire
//	(anthropic for claude, openai for codex/…) — that still selects which base-URL
//	env var points the child at the gateway and labels the banner — and leave the
//	credential posture empty.
//
//	PROXY (default): resolveGuardUpstream picks the provider, base URL, API key, and
//	the Claude subscription-OAuth default. --remote-serve, when set, pins provider=openai
//	+ base=the box inside the resolver. ALONGSIDE (--gguf --alongside / --gguf
//	--base-url) takes THIS world too — the API upstream keeps its full credential
//	posture unchanged, and the loaded local model rides beside it (dual planner).
//
// A credential the wrapped agent could never repair is launch-FATAL here (os.Exit(2)),
// exactly as it was inline in cmdGuard: refusing before the spawn beats a headless child
// hanging on a login it cannot complete or hitting a raw upstream 401.
func resolveGuardUpstreamPosture(in guardUpstreamPostureInputs) guardUpstreamPosture {
	var p guardUpstreamPosture
	command := in.command
	// #8172, FIRST: the two enterprise-posture gates. Both run before every
	// credential gate below — including the local-only short-circuit, because a
	// cloud-routed child ignores the base-URL repoint whether the model behind the
	// gateway is remote or in-kernel — so a managed host is refused for its REAL
	// cause instead of a subscription credential it does not lack, and never parks
	// 24h for a re-login that cannot change which credential is sent. Both are
	// no-ops when no CA bundle is declared and no cloud selector is set, which is
	// every non-enterprise host. See guard_upstream_trust.go.
	p.upstreamTrustNote = guardUpstreamTrustGate(in.quiet)
	p.cloudRouteWaived = guardCloudRouteGate(in.quiet)
	if in.localModel && !in.localAlongside {
		p.up, p.providerAutodetected = resolveGuardProvider(in.provider, command[0])
		return p
	}
	us := resolveGuardUpstream(in.provider, command[0], in.baseURL, in.remoteBase, in.apiKeyEnv, in.anthropicOAuth, in.oauthTokenEnv)
	p.up, p.providerAutodetected, p.resolvedBase = us.provider, us.autodetected, us.baseURL
	p.apiKey, p.pinUpstream, p.oauthSource = us.apiKey, us.pinUpstream, us.oauthSource
	p.keychainAPIKey = us.keychainAPIKey
	// resolveGuardUpstream armed the spend meter with the agent name (Opus
	// default). Re-arm with the statically-known upstream model so a non-default
	// tier prices at its own rate — e.g. a claude-fable-5 session bills 2x Opus
	// instead of being under-booked as claude. Falls back to the agent name for
	// an unknown model, so this only corrects a known misprice (see
	// guardSpendPricingContext).
	armServedSpendPricing(p.up, guardSpendPricingContext(p.up, in.model, command))
	if p.pinUpstream && p.up == "anthropic" {
		p.credPath = filepath.Join(us.claudeConfigDir, ".credentials.json")
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
	if us.passthroughFallback && !in.quiet {
		fmt.Fprintf(os.Stderr, "fak guard: no Claude subscription OAuth token found; falling back to passthrough — the wrapped agent's own credential (a subscription login or ANTHROPIC_API_KEY) is forwarded upstream.%s If you hit a 401, run `claude` once or `claude setup-token`.\n", guardLoginStatusNote(us))
	}
	if us.ambientKeyOverridden && !in.quiet {
		fmt.Fprintln(os.Stderr, "fak guard: ANTHROPIC_API_KEY is set but fak defaults to your Claude Pro/Max subscription (OAuth); the key is ignored upstream. Pass --api-key-env ANTHROPIC_API_KEY to use API billing instead.")
	}
	if us.keychainAPIKey && !in.quiet {
		fmt.Fprintln(os.Stderr, "fak guard: no Claude subscription login found; using Claude Code's saved API key from the macOS Keychain upstream (API billing — the same key the wrapped agent itself authenticates with, so the billed account is unchanged).")
	}
	// Pinned Claude subscription: the OAuth access token fak holds upstream is
	// short-lived (the provider rotates it ~hourly, and Claude Code rewrites the
	// refreshed value into the same credential file). Resolving it ONCE at startup
	// pins the boot-time token for the whole session, so a session that outlives the
	// token 401s — and re-logging in does not help, because the refreshed token lands
	// in the file the frozen string never re-reads. So on this path we hand the gateway
	// a credential FUNC that re-reads the live token per request. It falls back to the
	// boot-time apiKey on a transient read miss (the planner's effectiveAPIKey contract).
	if p.pinUpstream && p.up == "anthropic" {
		tokenEnv := in.oauthTokenEnv
		// The account-failover state is seeded with the pinned config dir. Until a failover
		// fires it is inert (currentConfigDir == the pinned dir, walled set empty), so the
		// token path below is byte-for-byte the historical one. On an account-scoped 403 the
		// planner's hook calls af.failover, which advances currentConfigDir to a permitted
		// sibling — and apiKeyFunc then reads THAT account's rotating token, making the swap
		// sticky across turns. A homeRoot we cannot resolve leaves af nil and disables failover
		// (the historical terminal-on-account-403 behavior), never a crash.
		var af *accountFailover
		if homeRoot, hErr := os.UserHomeDir(); hErr == nil && strings.TrimSpace(homeRoot) != "" {
			af = newAccountFailover(homeRoot, us.claudeConfigDir, nil)
			p.accountFailoverFunc = af.failover
			p.transientTargetFunc = af.transientTarget
			p.accountRehome = af.forceRehome
		}
		// Feed the live accounts+nodes status area: the ACTIVE seat follows a failover
		// (af.currentConfigDir), and af.walledKeys marks the seats an account-scoped 403
		// skipped. With af nil (home root unresolvable) the active seat is the pinned
		// config dir and nothing is walled — a stable single-seat view.
		if af != nil {
			p.activeAccountDir = af.currentConfigDir
			p.walledAccounts = af.walledKeys
		} else {
			pinnedDir := us.claudeConfigDir
			p.activeAccountDir = func() string { return pinnedDir }
		}
		p.apiKeyFunc = func() string {
			// After a failover, read the ADOPTED sibling account's live token directly from its
			// config dir; that dir differs from the pinned one only once af.failover has run.
			if af != nil {
				if dir := af.currentConfigDir(); dir != "" && dir != us.claudeConfigDir {
					if tok, live := readLiveAccessToken(dir, time.Now()); live {
						return tok
					}
					// The adopted account's token is momentarily unreadable/expired — fall through
					// to the default resolve rather than dropping auth entirely.
				}
			}
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
	if p.pinUpstream && p.up == "anthropic" {
		if v := guardRunHeadlessRehydrate(cmdGuardStdinInteractive(), p.pinUpstream, p.credPath, in.oauthTokenEnv); v.Refused {
			fmt.Fprintf(os.Stderr, "fak guard: STALE_CRED — the Claude subscription OAuth token in %s is expired and did not refresh within the wait window, and stdin is not a terminal — refusing to spawn a headless agent that would only hit a raw upstream 401.%s\n", v.CredPath, guardLoginStatusNote(us))
			fmt.Fprintln(os.Stderr, "  fix: run `claude` once to log in (refreshes the token), or `claude setup-token` for a long-lived token, or export CLAUDE_CODE_OAUTH_TOKEN, or raise FAK_AUTH_REFRESH_WINDOW if a refresh is just slow.")
			os.Exit(2)
		}
	}
	if !guardCodexAuthManagementCommand(command) &&
		guardCodexSubscriptionEligibleForProfile(in.profile, p.up, in.baseURL, in.remoteBase, in.apiKeyEnv) {
		if cred, err := resolveCodexSubscriptionCredential(in.codexHome); err == nil {
			p.apiKey = cred.AccessToken
			p.pinUpstream = true
			p.oauthSource = cred.Source
			p.resolvedBase = guardCodexChatGPTBackendBaseURL
			p.extraHeaders = guardCodexSubscriptionHeaders(cred)

			// Codex subscription seats need the same default account-wall recovery as Claude:
			// the planner rotates once and resends the still-owned request in place. Keep the
			// access token and ChatGPT-Account-Id on one moving credential source so a switch
			// can never pair one account's bearer with another account's routing header.
			homeRoot, _ := os.UserHomeDir()
			installCodexAccountRefresh(&p, in.codexHome, homeRoot, cred)
		} else if strings.TrimSpace(os.Getenv(guardCodexEnvKey(in.apiKeyEnv))) == "" && !in.quiet {
			fmt.Fprintf(os.Stderr, "fak guard: Codex ChatGPT subscription unavailable: %v\n", err)
		}
	}
	return p
}

func installCodexAccountRefresh(p *guardUpstreamPosture, explicitHome, homeRoot string, cred codexSubscriptionCredential) {
	// An explicit Codex home is a process-level account pin. Do not install
	// cross-home failover: two concurrent launches must never migrate onto
	// each other's seat after one account returns an auth or capacity wall.
	if strings.TrimSpace(explicitHome) != "" {
		p.apiKeyFunc, p.extraHeadersFunc = newCodexSubscriptionRefreshers(explicitHome, cred)
		return
	}
	if strings.TrimSpace(homeRoot) != "" {
		af := newCodexAccountFailover(homeRoot, filepath.Dir(cred.Source))
		p.accountFailoverFunc = af.failover
		p.transientTargetFunc = af.transientTarget
		p.activeAccountDir = af.currentConfigDir
		p.walledAccounts = af.walledKeys
		p.apiKeyFunc, p.extraHeadersFunc = newCodexFailoverRefreshers(af, cred)
		return
	}
	p.apiKeyFunc, p.extraHeadersFunc = newCodexSubscriptionRefreshers(explicitHome, cred)
}
