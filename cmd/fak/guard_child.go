package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/cloudroute"
	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// guard_child.go — the resolved upstream wire/credential posture, building and
// supervising the wrapped child process (incl. the budget-driven fresh-context
// restart loop), and tearing the gateway down with a faithful exit report.
// Split out of guard.go to keep the dispatch surface readable.

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
	// noTokenAnywhere is set when guard is on the Anthropic auto-OAuth path and found NO
	// subscription token anywhere AND nothing to recover (no env token, no .credentials.json to
	// rotate, no .oauth-token). There is nothing to pin and nothing to refresh, so cmdGuard
	// fails loud before spawning a headless child that would block on a login it can never
	// complete. Distinct from passthroughFallback, which still has a path (the child's own key).
	noTokenAnywhere bool
	// keychainAPIKey is set when the auto-OAuth path found no subscription anywhere but
	// adopted Claude Code's SAVED API KEY from the macOS Keychain as the upstream credential
	// (#5363) — the credential a Mac on API billing actually authenticates with. Billing is
	// unchanged by the adoption (the wrapped agent itself bills this same key); holding it in
	// guard means the managed-cache auto posture can see API billing and go ACTIVE. cmdGuard
	// surfaces a one-line note for discoverability, mirroring ambientKeyOverridden.
	keychainAPIKey bool
	// Claude config-home login posture for the CLAUDE_CONFIG_DIR guard will hand to the
	// child. This is credential-safe observability only; token routing still follows the
	// explicit source precedence in resolveAnthropicOAuthToken.
	claudeConfigDir string
	loginStatus     accounts.LoginStatus
	canServe        bool
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
	//
	// The base MUST carry the /v1 suffix the OpenAI wire appends "/chat/completions" to:
	// the proxy planner POSTs adapter.Endpoint(BaseURL, model) = <base>/chat/completions,
	// while `fak serve` registers its route at /v1/chat/completions. normalizeRemoteServe
	// returns a bare http://HOST:PORT (so the /healthz preflight probes the ROOT health
	// route, which is NOT under /v1), so we add /v1 HERE — symmetric with guardEnvValue,
	// which adds /v1 only to the CHILD's OPENAI_BASE_URL. Without this the upstream proxy
	// hop 404s on every real turn (the /healthz preflight passes regardless, so it would
	// surface only mid-session). Idempotent: an operator base already ending in /v1 is
	// left as-is.
	remote := strings.TrimSpace(remoteServeBase) != ""
	if remote {
		providerFlag = "openai"
		baseURLFlag = guardOpenAIV1Base(strings.TrimSpace(remoteServeBase))
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

	// An explicitly-named --api-key-env that is EMPTY on the Anthropic wire is almost
	// certainly an accident the operator wants to hear about, not silently absorb: naming
	// the key is the explicit opt-IN to API billing, so an empty value (a typo, a sudo-
	// stripped env, a CI secret that did not inject) would otherwise collapse to apiKey=""
	// and fall straight into subscription OAuth below — billing the WRONG account with no
	// signal. Fail loud here, mirroring the --require-key-env gate (guard.go), UNLESS
	// --anthropic-oauth was passed (that flag means "force the subscription regardless", so
	// an empty named key is not a contradiction there). Scoped to anthropic on purpose: for
	// the OpenAI-compatible wires an empty named key is documented passthrough convention
	// (the client's own key flows upstream), so it must NOT exit there.
	if guardEmptyNamedKeyIsError(up, apiKeyEnv, apiKey, forceOAuth) {
		fmt.Fprintf(os.Stderr, "fak guard: --api-key-env %s is set but that env var is empty — export it for API billing, drop the flag to use your Claude Pro/Max subscription, or pass --anthropic-oauth to force the subscription.\n", apiKeyEnv)
		os.Exit(2)
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
	noTokenAnywhere := false
	keychainAPIKey := false
	claudeConfigDir := ""
	loginStatus := accounts.LoginStatus("")
	canServe := false
	if forceOAuth && up != "anthropic" {
		fmt.Fprintf(os.Stderr, "fak guard: --anthropic-oauth applies only to --provider anthropic (got %q)\n", up)
		os.Exit(2)
	}
	autoOAuth := up == "anthropic" && apiKey == ""
	if forceOAuth || autoOAuth {
		claudeConfigDir, loginStatus, canServe = guardClaudeLoginPosture()
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
		case guardSubscriptionLoginPresent(oauthTokenEnv):
			// A subscription login EXISTS on disk but its token was unreadable this instant —
			// Claude Code rewrites .credentials.json ~hourly and the OAuth access token is
			// short-lived, so a boot read can catch the file mid-rotation (or holding a
			// just-expired token, which resolveAnthropicOAuthToken correctly drops rather than
			// send). Demoting to passthrough HERE would strip the placeholder ANTHROPIC_API_KEY
			// that keeps the wrapped agent from falling into its OWN /login — the 'stuck on
			// login sometimes' hang. So PIN ON INTENT with an empty boot apiKey: pinUpstream
			// stays true and the per-request APIKeyFunc (guard.go) re-reads the freshly-rotated
			// token on the first turn. effectiveAPIKey already tolerates an empty boot key
			// (func result wins; the 401 path self-heals once), so the first turn waits for the
			// rotation instead of dropping the agent into a login prompt.
			pinUpstream = true
			oauthSource = "subscription login (token rotating; resolved per request)"
		default:
			// Auto attempt found no token AND no subscription login is present at all. There is
			// nothing to pin and nothing for the per-request refresh to recover. Before falling
			// back, ask the macOS Keychain for Claude Code's SAVED API KEY (#5363): a Mac whose
			// Claude Code runs on API billing has no subscription to find — its real credential
			// is that key, so guard adopts it upstream. Billing is identical either way (the
			// wrapped agent itself authenticates with this same key), and holding it means the
			// managed-cache auto posture sees API billing instead of "billing unknown". This
			// stays subordinate to the subscription default: any OAuth token above wins, and the
			// ambient-ANTHROPIC_API_KEY-must-not-override-a-subscription rule is untouched.
			if key, ok := guardKeychainAPIKey(claudeConfigDir); ok {
				apiKey = key
				keychainAPIKey = true
				break
			}
			// Two remaining sub-cases:
			//   - the child carries its own ANTHROPIC_API_KEY → legitimate API-billing
			//     passthrough (its key flows upstream); keep spawning.
			//   - the child has no key either → a headless spawn would block on a /login the
			//     wrapped agent can never complete. Flag noTokenAnywhere so cmdGuard can fail
			//     loud BEFORE spawning rather than hang. (An attended terminal still gets the
			//     interactive login — cmdGuard gates the hard exit on a non-interactive stdin.)
			passthroughFallback = true
			noTokenAnywhere = os.Getenv("ANTHROPIC_API_KEY") == ""
		}
	}
	// Arm the served-session spend meter for this upstream: the guard boot is the
	// one point where the wrapped provider identity is known, and the session
	// table's spend axis (Budget.SpendMicroCentsLeft) only debits priced turns —
	// see session_spend.go. Unpriced pairs stay dollar-blind (no debit), honestly.
	armServedSpendPricing(up, agentName)
	return guardUpstream{
		provider: up, autodetected: autodetected, baseURL: resolvedBase,
		apiKey: apiKey, pinUpstream: pinUpstream, oauthSource: oauthSource,
		passthroughFallback: passthroughFallback, ambientKeyOverridden: ambientKeyOverridden,
		noTokenAnywhere: noTokenAnywhere, keychainAPIKey: keychainAPIKey,
		claudeConfigDir: claudeConfigDir, loginStatus: loginStatus, canServe: canServe,
		remoteServe: remote,
	}
}

// guardHeadlessRehydrateWindow is how long the proactive wake-time StaleCred check (#1834)
// polls disk for the credential file to rotate BEFORE the first upstream request goes out,
// under a headless launch where no interactive `claude` process is running to rewrite
// .credentials.json on its own. It intentionally matches internal/agent's
// maxAuthRefreshWindow (the ceiling FAK_AUTH_REFRESH_WINDOW clamps to — that constant is
// unexported, so this is a deliberately duplicated literal, not an independent budget): this
// check is what the reactive 401 poll was always hoping to catch, just moved BEFORE the
// request instead of after a 401 already happened, so there is no reason for its ceiling to
// differ. FAK_AUTH_REFRESH_WINDOW also governs this proactive wait (see
// guardHeadlessRehydrateWindowDuration) so one operator knob tunes both the proactive and
// reactive paths together.
const guardHeadlessRehydrateWindow = 30 * time.Second

// guardHeadlessRehydratePollInterval mirrors internal/agent's authRefreshPollInterval (also
// unexported) so the proactive pre-request poll puts no more disk pressure on
// .credentials.json than the existing reactive 401 poll already does.
const guardHeadlessRehydratePollInterval = 150 * time.Millisecond

// guardHeadlessRehydrateWindowDuration resolves the proactive wait's budget the same way
// internal/agent's authRefreshWindow does: FAK_AUTH_REFRESH_WINDOW (any Go duration) when
// set and valid, clamped to [0, guardHeadlessRehydrateWindow], else the default. Honoring the
// SAME env var as the reactive path means an operator who raises the reactive window (to ride
// out a slower refresh) raises the proactive one too, with one knob instead of two.
func guardHeadlessRehydrateWindowDuration() time.Duration {
	if v := strings.TrimSpace(os.Getenv("FAK_AUTH_REFRESH_WINDOW")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			if d > guardHeadlessRehydrateWindow {
				return guardHeadlessRehydrateWindow
			}
			return d
		}
	}
	return guardHeadlessRehydrateWindow
}

// guardHeadlessCredCheck builds the accounts.CredCheck the #1834 proactive rehydrate rung
// runs on a headless launch: it reads credPath's accessToken expiry once, and — if it is
// already expired — polls the file for up to guardHeadlessRehydrateWindowDuration() for a
// REWRITTEN, still-live expiry (a concurrent `claude` refresh landing, or an operator's cron
// re-auth), exactly mirroring the reactive 401-path's poll (internal/agent/stream.go
// refreshAPIKeyWait/refreshAPIKey) but run proactively before the first request instead of
// after a 401. now defaults to time.Now; sleep defaults to time.Sleep (both overridable so a
// test never sleeps wall-clock time).
func guardHeadlessCredCheck(credPath string, now func() time.Time, sleep func(time.Duration)) accounts.CredCheck {
	return guardCredCheckWithWindow(credPath, guardHeadlessRehydrateWindowDuration(), now, sleep)
}

// guardCredCheckWithWindow is guardHeadlessCredCheck generalized over an explicit wait window,
// so the pre-spawn #1834 rehydrate rung (a short, rotation-in-progress window) and the
// post-crash auth-recovery path (guardMaybeRecoverAuthCrash — a much longer, human-paced
// window) share ONE poll implementation instead of two copies that could drift.
//
// When the credential is found expired, the check first TRIGGERS a refresh actively (spawn one
// `claude -p` against the config dir, which makes Claude Code rotate its own credential file)
// rather than only WAITING for something else to rewrite it — closing the headless-expiry gap
// where an idle seat's token died with no interactive `claude` around to refresh it. The trigger
// is gated on FAK_GUARD_AUTO_REFRESH (default on) and, when it advances the on-disk expiry, the
// poll/wait below is skipped entirely. Only if the trigger is disabled or does not rotate the
// token do we fall through to the original poll — the human-relogin backstop stays intact.
func guardCredCheckWithWindow(credPath string, window time.Duration, now func() time.Time, sleep func(time.Duration)) accounts.CredCheck {
	return guardCredCheckWithRefresh(credPath, window, now, sleep, nil)
}

// guardCredCheckWithRefresh is guardCredCheckWithWindow with an injectable refresh spawn, so a
// test can drive the active-refresh branch deterministically without exec'ing a real `claude`.
// A nil spawn uses accounts.DefaultRefreshSpawn in production; the FAK_GUARD_AUTO_REFRESH knob
// (not the spawn being nil) is what disables the branch, so a test can also assert the
// disabled-path behavior is byte-for-byte the old poll.
func guardCredCheckWithRefresh(credPath string, window time.Duration, now func() time.Time, sleep func(time.Duration), spawn accounts.RefreshSpawn) accounts.CredCheck {
	if now == nil {
		now = time.Now
	}
	if sleep == nil {
		sleep = time.Sleep
	}
	// credLive reports whether the credential currently on disk has an expiry strictly after
	// t — the same "is this bearer still good" question CredFreshness answers for a
	// last-refresh+window pair, expressed directly over an absolute expiresAt instant.
	credLive := func(t time.Time) bool {
		expiresAt, ok := credExpiresAt(credPath)
		return ok && expiresAt.After(t)
	}
	skew := guardRefreshSkew()
	return func(ctx context.Context) (fresh bool, refreshed bool) {
		if _, ok := credExpiresAt(credPath); !ok {
			// No parseable credential on disk at all (missing/torn/no token) — nothing this
			// rung can vouch for or refresh; fail closed to the caller's STALE_CRED refusal.
			return false, false
		}
		// The initial gate looks ahead by skew: a token that expires within the skew window is
		// treated as needing refresh NOW, so the rotation lands before a request races expiry
		// (matching Claude Code's own refresh-early behavior). The post-refresh and post-poll
		// confirmations below use bare now() — a token live at this instant is a real success.
		if credLive(now().Add(skew)) {
			return true, false // comfortably live: no wait needed, first request goes out immediately
		}
		// Expired: try to CAUSE a refresh in place before falling back to waiting for one. A
		// successful rotation returns immediately; a no-op (refresh token dead, or the branch
		// disabled) drops through to the poll + the caller's human-relogin backstop. The trigger
		// runs under its own bounded sub-context so a hung `claude` can never wedge this check.
		if guardAutoRefreshEnabled() {
			cfgDir := filepath.Dir(credPath)
			rctx, cancel := context.WithTimeout(ctx, guardAutoRefreshTimeout())
			did, _ := accounts.TriggerRefresh(rctx, cfgDir, spawn, now)
			cancel()
			if did && credLive(now()) {
				return false, true
			}
		}
		// A refresh was wanted but did not land. If the token is nonetheless still live at this
		// instant (we were inside the proactive skew window, not actually expired), let the first
		// request go out now rather than blocking on a wait for a rotation that is not yet due —
		// the skew is a nudge, never a stall for a still-usable token.
		if credLive(now()) {
			return true, false
		}
		deadline := now().Add(window)
		for {
			select {
			case <-ctx.Done():
				return false, false
			default:
			}
			if !now().Before(deadline) {
				return false, false // window exhausted with no rotation observed — refresh walled
			}
			sleep(guardHeadlessRehydratePollInterval)
			if credLive(now()) {
				return false, true // a fresher token landed mid-poll: refreshed in place
			}
		}
	}
}

// guardAutoRefreshEnabled reports whether the active-refresh branch (spawning `claude -p` to make
// Claude Code rotate its own credential) is on. It defaults ON — the whole point is to self-heal a
// headless seat's expiry without a human — and is disabled only by an explicit falsey
// FAK_GUARD_AUTO_REFRESH ("0"/"false"/"off"/"no"), which restores the pure wait-for-rotation poll.
func guardAutoRefreshEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_GUARD_AUTO_REFRESH"))) {
	case "0", "false", "off", "no":
		return false
	default:
		return true
	}
}

// guardAutoRefreshTimeout bounds a single active-refresh spawn. It defaults to
// guardHeadlessRehydrateWindow (the same 30s ceiling the wait path uses) so a refresh turn that
// never returns cannot outlast the window it is meant to avoid; FAK_GUARD_AUTO_REFRESH_TIMEOUT
// overrides it (any Go duration), clamped to a sane floor/ceiling.
func guardAutoRefreshTimeout() time.Duration {
	const def = guardHeadlessRehydrateWindow
	if v := strings.TrimSpace(os.Getenv("FAK_GUARD_AUTO_REFRESH_TIMEOUT")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			if d > 5*time.Minute {
				return 5 * time.Minute
			}
			return d
		}
	}
	return def
}

// guardRefreshSkew is how far BEFORE a token's expiry the proactive check treats it as needing
// refresh, so the rotation lands before a request races the lapse (Claude Code refreshes ~5min
// early; we match that). Applied only to the initial refresh-decision gate, never to the
// confirmations that a token is live right now. Defaults to 5m; FAK_GUARD_REFRESH_SKEW overrides
// it (any Go duration), clamped to a non-negative value under a sane ceiling. 0 disables the
// look-ahead, restoring the strict "only refresh once actually expired" behavior.
func guardRefreshSkew() time.Duration {
	const def = 5 * time.Minute
	if v := strings.TrimSpace(os.Getenv("FAK_GUARD_REFRESH_SKEW")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			if d > time.Hour {
				return time.Hour
			}
			return d
		}
	}
	return def
}

// guardHeadlessRehydrateVerdict is cmdGuard's outcome from running the #1834 proactive
// StaleCred rung: Ran is false when the rung was not applicable (not a headless pinned-OAuth
// launch, or no credentials file to check at all — resolveGuardUpstream's own resolution
// already covers that case), so cmdGuard's caller can tell "didn't run" from "ran and
// cleared". Refused is true only on a genuine STALE_CRED refusal (expired credential, refresh
// walled within the window) — the exact case that used to fall through to a raw upstream 401.
type guardHeadlessRehydrateVerdict struct {
	Ran      bool
	Refused  bool
	Detail   string
	CredPath string
}

// guardRunHeadlessRehydrate wires accounts.NewRehydrateCredRung (#1183) into the guard launch
// path (#1834): on a HEADLESS launch (stdinInteractive false) that is pinning the Claude
// subscription OAuth token upstream (pinUpstream true), it forces the credential-freshness
// check — and, if needed, an active wait for a rotation — to run BEFORE cmdGuard spawns the
// child and the first upstream request goes out, instead of discovering staleness reactively
// on a 401 after the fact (internal/agent's 3s-then-configurable authRefreshWindow). It uses
// rehydrate.NewGate/Admit at the canonical StaleCred band (dormancy.Cold) so this composes
// with the SAME staged-gate vocabulary internal/sessionimage.Rehydrate uses for a resumed
// session — this call site just has no session image to resume, so it runs the one applicable
// rung directly rather than staging a whole dormancy-banded gate.
//
// An INTERACTIVE launch is left alone (Ran=false): there a live `claude` process is already
// the thing rewriting .credentials.json, so the existing reactive per-request re-read is
// sufficient and a blocking pre-spawn wait would only delay an attended terminal for no
// benefit. A launch that is not pinning the subscription OAuth token (API-key billing, a
// non-Anthropic wire, local --gguf) has no credential file this rung understands, so it is
// also skipped (Ran=false) — resolveGuardUpstream's own noTokenAnywhere/passthroughFallback
// handling already covers those postures.
//
// An explicit env token (tokenEnv, default CLAUDE_CODE_OAUTH_TOKEN) is ALSO left alone
// (Ran=false) — see the #3267 note at the guard below.
func guardRunHeadlessRehydrate(stdinInteractive, pinUpstream bool, credPath, tokenEnv string) guardHeadlessRehydrateVerdict {
	if stdinInteractive || !pinUpstream || strings.TrimSpace(credPath) == "" {
		return guardHeadlessRehydrateVerdict{}
	}
	// #3267: the env token OUTRANKS the credential file — resolveAnthropicOAuthToken's
	// precedence is tokenEnv, THEN <claude-config>/.credentials.json, THEN .oauth-token — so
	// when it is set, the file this rung inspects is not the credential the upstream will be
	// authenticated with. Checking it can only produce a false STALE_CRED, and the #2260 park
	// below then waits for a re-login that cannot change which token is sent. Server-side that
	// is the whole failure: a container running `fak guard -- <harness>` with
	// CLAUDE_CODE_OAUTH_TOKEN exported has no interactive `claude` to re-login and usually no
	// credential file at all, so the launch parks for the full 24h budget — before binding the
	// gateway or spawning the child — and an unattended worker hangs silently instead of
	// running or failing loud. Defer to the env token: it is the explicit headless/automation
	// override, and a bad one still surfaces through the reactive per-request 401 path.
	if strings.TrimSpace(tokenEnv) != "" && strings.TrimSpace(os.Getenv(tokenEnv)) != "" {
		return guardHeadlessRehydrateVerdict{}
	}
	check := guardHeadlessCredCheck(credPath, nil, nil)
	gate := rehydrate.NewGate(accounts.NewRehydrateCredRung(check))
	adm := gate.Admit(context.Background(), dormancy.Cold)
	if adm.Admitted {
		return guardHeadlessRehydrateVerdict{Ran: true, CredPath: credPath}
	}
	// #2260: the ≤30s window above rides out a rotation ALREADY in progress; an expired
	// credential on a headless host needs a HUMAN-paced re-login (minutes-to-hours). Park
	// on a few-minute poll — bounded (FAK_GUARD_PARK_BUDGET, default 24h; 0 restores the
	// immediate refusal), observable (one park line + one outcome line on stderr) — before
	// failing loud, so the fleet self-heals the moment `claude` runs once anywhere on the
	// host instead of every queued spawn dying inside half a minute. See guard_park.go.
	if park := guardParkForRelogin(credPath, guardParkBudget(), guardParkPoll(), nil, nil, os.Stderr); park.Recovered {
		return guardHeadlessRehydrateVerdict{Ran: true, CredPath: credPath}
	}
	return guardHeadlessRehydrateVerdict{Ran: true, Refused: true, Detail: adm.Detail, CredPath: credPath}
}

// guardAuthCrashRecoverWindow is the default bound for guardAuthCrashRecoverWindowDuration —
// see its doc for why this is a SEPARATE, much longer knob than the reactive
// authRefreshWindow/guardHeadlessRehydrateWindow pair.
const guardAuthCrashRecoverWindow = 5 * time.Minute

// maxGuardAuthCrashRecoverWindow bounds FAK_GUARD_AUTH_RECOVER_WINDOW so a fat-fingered value
// cannot wedge the guard process indefinitely after a crash; the operator is expected to notice
// within this ceiling or fall back to the printed manual-resume guidance.
const maxGuardAuthCrashRecoverWindow = 30 * time.Minute

// guardAuthCrashRecoverWindowDuration resolves how long fak guard actively waits, AFTER the
// wrapped agent has already exited on what looks like an expired subscription token, for a
// fresh login to land before giving up and falling back to the manual formatGuardResumeGuidance
// path. This is deliberately a SEPARATE, much longer budget than authRefreshWindow (10s,
// internal/agent) and guardHeadlessRehydrateWindow (30s, the pre-spawn proactive check): those
// two are riding out a ROTATION ALREADY IN PROGRESS (an interactive `claude` elsewhere rewrites
// the file within seconds); this one is riding out a crash that has ALREADY happened and now
// needs a HUMAN to notice and re-authenticate — five minutes is a realistic "someone is watching
// an alert" budget, not a network-hiccup budget. FAK_GUARD_AUTH_RECOVER_WINDOW overrides it (any
// Go duration), clamped to [0, maxGuardAuthCrashRecoverWindow]; 0 disables the wait entirely (an
// auth-caused crash is still diagnosed in the exit message, but never auto-relaunched).
func guardAuthCrashRecoverWindowDuration() time.Duration {
	if v := strings.TrimSpace(os.Getenv("FAK_GUARD_AUTH_RECOVER_WINDOW")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d >= 0 {
			if d > maxGuardAuthCrashRecoverWindow {
				return maxGuardAuthCrashRecoverWindow
			}
			return d
		}
	}
	return guardAuthCrashRecoverWindow
}

// guardContinueFlagForAgent returns the resume/continue flag fak knows is SAFE to auto-inject
// for a recognized wrapped agent, keyed off the same command[0] basename resolveGuardProvider
// uses. Currently only Claude Code (`claude`) is recognized — its `--continue` flag is exactly
// the resume path formatGuardResumeGuidance already tells an operator to run BY HAND, so
// auto-injecting it here only automates the one case already proven safe. Any other/unrecognized
// binary returns ok=false: fak cannot safely guess a foreign tool's continuation syntax, so it
// never auto-relaunches for it — the crash falls through to today's manual guidance instead of
// risking a silent, context-dropping relaunch.
func guardContinueFlagForAgent(agentName string) (flag string, ok bool) {
	// guardAgentBaseName (not filepath.Base) so a Windows launcher path like
	// `C:\tools\claude.exe` is recognized on every host OS — filepath.Base is
	// host-specific and does not split backslash paths on the Linux CI runner.
	if guardAgentBaseName(agentName) == "claude" {
		return "--continue", true
	}
	return "", false
}

// guardAppendContinueFlag returns command with flag appended, unless flag is already present
// anywhere in command[1:] — so a second auth-crash-and-recover cycle in the same guarded
// session never stacks the flag twice. The input is never mutated in place.
func guardAppendContinueFlag(command []string, flag string) []string {
	for _, a := range command[1:] {
		if a == flag {
			return command
		}
	}
	out := make([]string, len(command), len(command)+1)
	copy(out, command)
	return append(out, flag)
}

// guardStripContinueFlag removes the agent's resume/--continue flag (and only that flag) from
// command, leaving the binary (command[0]) and every other argument intact. It is the inverse of
// guardAppendContinueFlag: when a relaunch boots FRESH on a distilled carryover seed, any --continue
// left on the command from a prior fallback restart would ALSO reattach — and re-inflate — the
// exhausted transcript underneath the seed, defeating the shrink. A no-op for an unrecognized agent
// (no known resume flag) or a command that never carried the flag. The input is never mutated in
// place. --continue is a boolean flag (no value), so only the token itself is dropped.
func guardStripContinueFlag(command []string, agentName string) []string {
	flag, ok := guardContinueFlagForAgent(agentName)
	if !ok || len(command) == 0 {
		return command
	}
	out := make([]string, 0, len(command))
	out = append(out, command[0])
	for _, a := range command[1:] {
		if a == flag {
			continue
		}
		out = append(out, a)
	}
	return out
}

// guardSeedPromptFlagForAgent returns the prompt entrypoint fak knows is SAFE to inject a carryover
// seed_text through as a fresh initial prompt on a headless/no-continue relaunch (#3056), keyed off
// the SAME recognized-agent allowlist as guardContinueFlagForAgent (#3055 / "#A"). Claude Code is
// the one recognized agent: --append-system-prompt reaches the child in both interactive and
// headless `-p` modes and is additive (it never collides with a positional prompt already on the
// command line), so the seed rides along as extra orientation rather than clobbering the invocation.
// Any other/unrecognized binary returns ok=false — fak never guesses a foreign tool's prompt syntax,
// so its seed is left on disk unread (a no-op relaunch) instead of being mis-injected.
func guardSeedPromptFlagForAgent(agentName string) (flag string, ok bool) {
	if guardAgentBaseName(agentName) == "claude" {
		return "--append-system-prompt", true
	}
	return "", false
}

// guardClassifyAuthCrash decides whether a completed credential check correlates a non-zero
// child exit with an expired subscription token. hasCredential must come from a caller-side
// credExpiresAt(credPath) probe — check's own (fresh, refreshed) result cannot distinguish "no
// parseable credential on disk at all" from "a credential that stayed expired for the whole
// wait window" (both return false, false), and only the FORMER is a genuine "nothing to
// correlate against" case that must never be misreported as an auth crash. correlated is true
// only when there IS a credential to judge AND it was not already live at check time — a crash
// with a perfectly live token is something else entirely (a bad flag, an OOM) and must not be
// mislabeled. recovered mirrors check's own refreshed result: a fresh login landed within the
// window check was built with.
func guardClassifyAuthCrash(ctx context.Context, hasCredential bool, check accounts.CredCheck) (correlated, recovered bool) {
	if !hasCredential || check == nil {
		return false, false
	}
	fresh, refreshed := check(ctx)
	if fresh {
		return false, false
	}
	return true, refreshed
}

// guardMaybeRecoverAuthCrash is the mid-session counterpart to guardRunHeadlessRehydrate
// (#1834): where that rung heads off a STALE_CRED refusal BEFORE the child ever spawns, this
// one runs AFTER the wrapped agent has already exited abnormally, asking "did this crash happen
// because the Claude subscription token expired mid-session, and if so, has a fresh login landed
// since?" A crash the wrapped agent's OWN 401-handling causes (dropping into its own /login, or
// exiting outright) is exactly the failure class formatGuardResumeGuidance's manual "re-run with
// --continue" note already exists to route around — this closes that loop automatically for the
// one wrapped agent (Claude Code) fak knows a safe resume flag for. runErr is the child's
// completed exec.Cmd.Run/Wait error (nil/success and non-*exec.ExitError never match); credPath
// is the credential fak was pinning upstream (empty when not pinning, e.g. API-key billing or a
// local model — this never fires there). On a match, it BLOCKS for up to
// guardAuthCrashRecoverWindowDuration() polling the credential file, then returns the command
// with the resume flag appended and ok=true only if a fresh login actually landed; otherwise it
// returns ok=false and the caller's existing exit/report path proceeds unchanged (a
// non-auth-caused crash never even reaches the blocking poll, since correlated is checked first).
func guardMaybeRecoverAuthCrash(runErr error, command []string, credPath, agentName string, quiet bool, stderr io.Writer) (relaunch []string, ok bool) {
	if runErr == nil || strings.TrimSpace(credPath) == "" {
		return nil, false
	}
	ee, isExit := runErr.(*exec.ExitError)
	if !isExit || ee.ExitCode() == 0 {
		return nil, false
	}
	flag, known := guardContinueFlagForAgent(agentName)
	if !known {
		return nil, false
	}
	_, hasCred := credExpiresAt(credPath)
	window := guardAuthCrashRecoverWindowDuration()
	check := guardCredCheckWithWindow(credPath, window, nil, nil)
	correlated, recovered := guardClassifyAuthCrash(context.Background(), hasCred, check)
	if !correlated {
		return nil, false
	}
	next := guardAppendContinueFlag(command, flag)
	if !recovered {
		if !quiet && stderr != nil {
			fmt.Fprintf(stderr, "fak guard: %s exited (code %d) with an expired subscription token that did not recover within %s — resume manually once re-authenticated (`fak guard -- %s`)\n", agentName, ee.ExitCode(), window, strings.Join(next, " "))
		}
		return nil, false
	}
	if !quiet && stderr != nil {
		fmt.Fprintf(stderr, "fak guard: %s crashed on an expired subscription token; a fresh login landed within %s — auto-relaunching `%s` to resume this session\n", agentName, window, strings.Join(next, " "))
	}
	return next, true
}

// guardEmptyNamedKeyIsError is the pure decision behind the empty-`--api-key-env` fail-loud
// gate: it is an error ONLY when the upstream is the Anthropic wire, an api-key env var was
// EXPLICITLY named (apiKeyEnv != ""), that var resolved EMPTY (after trimming), and
// --anthropic-oauth was NOT passed. Naming the key is the explicit opt-in to API billing, so
// an empty value is an accident worth refusing rather than silently demoting to subscription
// OAuth (which would bill the wrong account). forceOAuth short-circuits to false: that flag
// means "force the subscription", so an empty named key beside it is not a contradiction. The
// non-Anthropic wires treat an empty named key as documented passthrough, so they are never an
// error here. Pure (no I/O, no exit) so the precedence is unit-tested without standing guard up.
func guardEmptyNamedKeyIsError(provider, apiKeyEnv, apiKeyValue string, forceOAuth bool) bool {
	if forceOAuth || provider != "anthropic" {
		return false
	}
	return strings.TrimSpace(apiKeyEnv) != "" && strings.TrimSpace(apiKeyValue) == ""
}

func guardClaudeLoginPosture() (string, accounts.LoginStatus, bool) {
	dir := guardClaudeConfigDir()
	h := accounts.Home{
		Name:     filepath.Base(strings.TrimRight(dir, string(os.PathSeparator))),
		Dir:      dir,
		Identity: accounts.DeriveIdentity(dir),
	}
	status := h.LoginStatus()
	return dir, status, h.CanServe()
}

func guardLoginStatusNote(us guardUpstream) string {
	if us.loginStatus == "" {
		return ""
	}
	return fmt.Sprintf(" CLAUDE_CONFIG_DIR=%s login=%s can_serve=%t",
		us.claudeConfigDir, us.loginStatus, us.canServe)
}

type guardChildSpawnMetadata struct {
	AgentRunID   string
	ParentRunID  string
	ToolCallID   string
	PolicyDigest string
	Backend      string
	Envelope     toolprocgate.CapabilityEnvelope
	RegistryPath string
	LaunchPlan   guardLaunchPlan
}

type guardChildLauncher func(toolprocgate.SpawnGrant) (*exec.Cmd, error)

const (
	guardCodexTerminalRestorePulseDuration = 8 * time.Second
	guardCodexTerminalRestorePulseInterval = 500 * time.Millisecond
)

var startGuardChildTerminalRestorePulse = windowgate.StartTerminalRestorePulse

func maybeStartGuardChildTerminalRestorePulse(command []string) {
	maybeStartGuardChildTerminalRestorePulseForPlan(newGuardLaunchPlan(command))
}

func maybeStartGuardChildTerminalRestorePulseForPlan(plan guardLaunchPlan) {
	if !plan.harnessProfile().HasRepoint(harnessprofile.RepointCLIConfig) {
		return
	}
	startGuardChildTerminalRestorePulse(guardCodexTerminalRestorePulseDuration, guardCodexTerminalRestorePulseInterval)
}

// maybeStartGuardChildHarnessTerminalRestorePulse repairs the parent terminal after a
// wrapped interactive harness starts. Codex already needed the pulse because its launch
// can perturb the console window; Claude can leave the same stale/hidden terminal state,
// so production launch paths cover both harnesses through the same restore seam.
func maybeStartGuardChildHarnessTerminalRestorePulse(command []string) {
	maybeStartGuardChildHarnessTerminalRestorePulseForPlan(newGuardLaunchPlan(command))
}

func maybeStartGuardChildHarnessTerminalRestorePulseForPlan(plan guardLaunchPlan) {
	profile := plan.harnessProfile()
	if !plan.recognized() {
		return
	}
	if profile.Name == "claude" {
		startGuardChildTerminalRestorePulse(guardCodexTerminalRestorePulseDuration, guardCodexTerminalRestorePulseInterval)
		return
	}
	maybeStartGuardChildTerminalRestorePulseForPlan(plan)
}

func newGuardChildSpawnMetadata(agentRunID, policyDigest, backend string, rt policy.Runtime, launchPlan guardLaunchPlan) guardChildSpawnMetadata {
	agentRunID = strings.TrimSpace(agentRunID)
	if agentRunID == "" {
		agentRunID = "guard"
	}
	env := toolprocgate.CapabilityEnvelope{
		Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn},
	}
	runtimeName := launchPlan.agentBaseName()
	if runtimeName != "" && rt.ToolRuntime != nil {
		if r, ok := rt.ToolRuntime.EnvelopeFor(runtimeName); ok {
			env.DeadlineMS = r.DeadlineMS
			env.HeartbeatEveryMS = r.HeartbeatEveryMS
		}
	}
	registryPath := strings.TrimSpace(os.Getenv("FAK_CHILD_REGISTRY"))
	if registryPath == "" {
		registryPath = sessionregistry.DefaultPath()
		if strings.TrimSpace(os.Getenv("FAK_SESSION_REGISTRY")) != "" {
			registryPath += ".children"
		}
	}
	return guardChildSpawnMetadata{
		AgentRunID:   agentRunID,
		ToolCallID:   "guard-child:" + agentRunID,
		PolicyDigest: strings.TrimSpace(policyDigest),
		Backend:      strings.TrimSpace(backend),
		Envelope:     env,
		RegistryPath: registryPath,
		LaunchPlan:   launchPlan,
	}
}

// buildGuardChild constructs the wrapped-agent command with ONLY the gateway URL injected
// into its environment (never the parent shell). In pinned subscription mode it also hands
// the client a provider-shaped placeholder API key (when it has none) so it talks to the
// gateway, which ignores the placeholder and authenticates upstream with the held token.
const guardActiveEnv = "FAK_GUARD_ACTIVE"

func buildGuardChild(command []string, injected [][2]string, pinUpstream bool, extraEnv ...[2]string) *exec.Cmd {
	plan, env := guardChildPlanCommandEnv(newGuardLaunchPlan(command), injected, pinUpstream, extraEnv...)
	child := newResolvedExecCommand(plan.executableCommand())
	child.Stdin, child.Stdout, child.Stderr = os.Stdin, os.Stdout, os.Stderr
	child.Env = env
	return child
}

func newResolvedExecCommand(command []string) *exec.Cmd {
	command = resolveWindowsBatchCommand(command)
	return exec.Command(command[0], command[1:]...)
}

// resolveWindowsBatchCommand preserves Windows PATH semantics for npm-installed agent
// shims. Keep this at the exec boundary: replacing the harness with node earlier erases
// the identity that provider wiring, native hooks, and session reporting consume.
func resolveWindowsBatchCommand(command []string) []string {
	if runtime.GOOS != "windows" || len(command) == 0 || strings.ContainsAny(command[0], `/\`) || filepath.Ext(command[0]) != "" {
		return command
	}
	entrypointRel := ""
	switch strings.ToLower(strings.TrimSpace(command[0])) {
	case "codex":
		entrypointRel = filepath.Join("node_modules", "@openai", "codex", "bin", "codex.js")
	case "gemini":
		entrypointRel = filepath.Join("node_modules", "@google", "gemini-cli", "bundle", "gemini.js")
	default:
		return command
	}
	// CreateProcess cannot execute a batch file directly. npm's shim sits beside
	// the package entrypoint. Search every PATH directory rather than trusting the
	// first shim: fak's managed launcher may intentionally precede npm's directory
	// and does not own the package entrypoint itself.
	node, nodeErr := exec.LookPath("node.exe")
	if nodeErr != nil {
		return command
	}
	return resolveNodeBatchCommandFromPath(command, entrypointRel, node, os.Getenv("PATH"))
}

func resolveNodeBatchCommandFromPath(command []string, entrypointRel, node, pathEnv string) []string {
	for _, dir := range filepath.SplitList(pathEnv) {
		dir = strings.TrimSpace(strings.Trim(dir, `"`))
		if dir == "" {
			continue
		}
		shim := filepath.Join(dir, command[0]+".cmd")
		if info, statErr := os.Stat(shim); statErr != nil || info.IsDir() {
			continue
		}
		entrypoint := filepath.Join(dir, entrypointRel)
		if info, statErr := os.Stat(entrypoint); statErr == nil && !info.IsDir() {
			out := []string{node, entrypoint}
			return append(out, command[1:]...)
		}
	}
	return command
}

func guardChildCommandEnv(command []string, injected [][2]string, pinUpstream bool, extraEnv ...[2]string) ([]string, []string) {
	plan, env := guardChildPlanCommandEnv(newGuardLaunchPlan(command), injected, pinUpstream, extraEnv...)
	return plan.executableCommand(), env
}

func guardChildPlanCommandEnv(plan guardLaunchPlan, injected [][2]string, pinUpstream bool, extraEnv ...[2]string) (guardLaunchPlan, []string) {
	// Landlock hook-floor (opt-in, Linux): rewrite the agent argv so the child is launched
	// through the fak re-exec trampoline, which applies the read-only-.git/hooks ruleset to
	// itself before exec'ing the agent. Off by default, no-op on non-Linux or when the hook
	// dirs cannot be resolved — the original command is used unchanged.
	plan = plan.withExecutableCommand(maybeLandlockCommand(plan.executableCommand()))
	// Apply the always-on #2358 secret floor to the AMBIENT parent environment
	// before it is inherited by the wrapped agent: a spawned child (and anything
	// it spawns) must not receive inherited credentials it never needed. Only the
	// ambient os.Environ() portion is stripped — the injected gateway wiring, the
	// caller's extraEnv, and the placeholder key below are guard's EXPLICIT grants,
	// appended after, and always survive. This is safe for every guard auth
	// posture: guard already resolved and captured its own upstream OAuth token in
	// THIS (parent) process before spawning, and the provider API keys an
	// API-billing child needs are spared by StripInheritedSecrets.
	ambient := os.Environ()
	// #8172: a request-signed cloud route (Bedrock SigV4 / Vertex ADC) resolves its
	// credential through the cloud SDK chain, and the credential-shaped members of
	// that chain (AWS_SESSION_TOKEN, GOOGLE_APPLICATION_CREDENTIALS, …) are exactly
	// what the floor above strips by name and value shape. Stripping them is right
	// by default and wrong here: this IS the child's configured credential, so it is
	// declared through the floor's own keep-set rather than by widening the floor.
	// Narrow on both axes — only when the route selector is actually set, and only
	// for names already present in the ambient environment (declaring is permission
	// for a variable to cross, never an instruction to invent one).
	var declaredCloudCreds []string
	if r, routed := cloudroute.Detect(ambient); routed {
		declaredCloudCreds = r.CredentialNames(ambient)
	}
	env, _ := policy.StripInheritedSecretsExcept(ambient, declaredCloudCreds)
	// #8172: derive the per-runtime trust variables from the ONE declared CA bundle
	// so the wrapped agent (Node), the AWS SDKs, curl, git, and Python all validate
	// against the same root fak does. Fills only the names the parent left unset —
	// an operator who pointed NODE_EXTRA_CA_CERTS at a fuller bundle keeps it. Empty
	// when no bundle is declared, so an unconfigured box is unchanged.
	env = append(env, guardChildTrustEnv(env)...)
	for _, kv := range injected {
		env = append(env, kv[0]+"="+kv[1])
	}
	for _, kv := range extraEnv {
		if strings.TrimSpace(kv[0]) != "" {
			env = append(env, kv[0]+"="+kv[1])
		}
	}
	// Durable child marker consumed by the Codex continuation hook. This is
	// injected by guard itself, so direct/unwrapped sessions cannot spoof it via
	// ambient inheritance after StripInheritedSecrets.
	env = append(env, guardActiveEnv+"=1")
	// Subscription mode: hand the client a PLACEHOLDER api key (only if it has none) so
	// it talks to the gateway; the gateway IGNORES the placeholder (pinUpstream) and
	// authenticates upstream with the real held OAuth token.
	isCodex := plan.harnessProfile().HasRepoint(harnessprofile.RepointCLIConfig)
	if pinUpstream && !isCodex && os.Getenv("ANTHROPIC_API_KEY") == "" {
		env = append(env, "ANTHROPIC_API_KEY=fak-guard-oauth-placeholder")
	}
	if pinUpstream && isCodex && os.Getenv("OPENAI_API_KEY") == "" {
		env = append(env, "OPENAI_API_KEY="+guardCodexOAuthPlaceholderAPIKey)
	}
	return plan, env
}

func guardChildSpawnAttempt(plan guardLaunchPlan, injected [][2]string, pinUpstream bool, meta guardChildSpawnMetadata, extraEnv ...[2]string) (guardLaunchPlan, toolprocgate.SpawnAttempt, error) {
	plan, envStrings := guardChildPlanCommandEnv(plan, injected, pinUpstream, extraEnv...)
	env, err := toolprocgate.EnvFromStrings(envStrings)
	if err != nil {
		return guardLaunchPlan{}, toolprocgate.SpawnAttempt{}, err
	}
	cwd, err := os.Getwd()
	if err != nil {
		return guardLaunchPlan{}, toolprocgate.SpawnAttempt{}, err
	}
	return plan, toolprocgate.SpawnAttempt{
		AgentRunID:   meta.AgentRunID,
		ParentRunID:  meta.ParentRunID,
		ToolCallID:   meta.ToolCallID,
		PolicyDigest: meta.PolicyDigest,
		Argv:         plan.executableCommand(),
		Env:          env,
		CWD:          cwd,
		Backend:      meta.Backend,
		Envelope:     meta.Envelope,
	}, nil
}

var guardPromptTransportOS = runtime.GOOS

// configureManagedHiddenConsole gives a managed console-subsystem root a hidden
// console on Windows. It is indirected so the launch gate stays unit-testable on
// non-Windows CI, where the underlying windowgate hook is a no-op.
var configureManagedHiddenConsole = windowgate.ConfigureBackgroundCommand

// managedLaunchNeedsHiddenConsole keeps #3597's windowless posture for headless
// workers only. An attended child needs the inherited console itself, not merely
// os.File values pointing at terminal handles: on Windows CREATE_NO_WINDOW left
// attended Codex alive but unable to initialize its TUI (no terminal-ready byte
// and no session file after 40s in #9656). Popup containment for attended Codex
// must therefore use a mechanism that preserves console usability; applying the
// background-process flag here is not a valid containment trade.
func managedLaunchNeedsHiddenConsole(plan guardLaunchPlan) bool {
	return !plan.interactive()
}

func launchGuardChildWithBroker(command []string, injected [][2]string, pinUpstream bool, meta guardChildSpawnMetadata, broker *toolprocgate.SpawnBroker, launcher guardChildLauncher, extraEnv ...[2]string) (toolprocgate.SpawnGrant, *exec.Cmd, error) {
	plan := meta.LaunchPlan
	if !plan.initialized {
		plan = newGuardLaunchPlan(command)
	}
	plan = plan.withExecutableCommand(command)
	// Decide from the semantic command before prompt transport can move a `-p`
	// prompt off argv (#4852). Headless runs remain windowless; attended roots
	// retain a console capable of hosting their TUI.
	hiddenConsoleChild := managedLaunchNeedsHiddenConsole(plan)
	executable, stdinPrompt, promptOnStdin := guardPromptStdinTransportForOS(plan.executableCommand(), guardPromptTransportOS)
	plan = plan.withExecutableCommand(executable)
	if broker == nil {
		broker = toolprocgate.NewSpawnBroker()
	}
	plan, attempt, err := guardChildSpawnAttempt(plan, injected, pinUpstream, meta, extraEnv...)
	if err != nil {
		return toolprocgate.SpawnGrant{}, nil, err
	}
	grant, err := broker.Admit(attempt)
	if err != nil {
		return toolprocgate.SpawnGrant{}, nil, err
	}
	reg, err := prepareGuardChildRegistration(meta, grant)
	if err != nil {
		return toolprocgate.SpawnGrant{}, nil, err
	}
	grant = withGuardRegistrationEnv(grant, reg)
	if err := guardWindowsArgvPreflight(grant.Argv, guardPromptTransportOS); err != nil {
		return toolprocgate.SpawnGrant{}, nil, err
	}
	if launcher == nil {
		launcher = guardExecLauncher
	}
	child, err := launcher(grant)
	if err != nil {
		terminalGuardRegistration(reg, sessionregistry.StateFailed, "launch_prepare_failed", "")
		return toolprocgate.SpawnGrant{}, nil, err
	}
	bindGuardRegistration(child, reg)
	if promptOnStdin {
		child.Stdin = strings.NewReader(stdinPrompt)
	}
	// #3597: headless workers must not allocate an unattended pane. Do not infer
	// console usability from the inherited os.File pointers here: #9656 proved
	// attended Codex can retain those pointers yet produce no TUI when Windows
	// CREATE_NO_WINDOW is set. StartInNewJob may add CREATE_SUSPENDED later, but an
	// attended child reaches it without the background console flags.
	if hiddenConsoleChild {
		configureManagedHiddenConsole(child)
	}
	return grant, child, nil
}

func guardExecLauncher(grant toolprocgate.SpawnGrant) (*exec.Cmd, error) {
	if len(grant.Argv) == 0 {
		return nil, fmt.Errorf("empty brokered argv")
	}
	child := newResolvedExecCommand(grant.Argv)
	child.Stdin, child.Stdout, child.Stderr = os.Stdin, os.Stdout, os.Stderr
	child.Env = toolprocgate.EnvStrings(grant.Env)
	child.Dir = grant.CWD
	return child, nil
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
		cmd := exec.Command("git", args...)
		windowgate.ConfigureBackgroundCommand(cmd)
		out, err := cmd.Output()
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
