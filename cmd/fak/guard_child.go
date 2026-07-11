package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
	"github.com/anthony-chaudhary/fak/internal/session"
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
			// nothing to pin and nothing for the per-request refresh to recover. Two sub-cases:
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
		noTokenAnywhere: noTokenAnywhere,
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
func guardRunHeadlessRehydrate(stdinInteractive, pinUpstream bool, credPath string) guardHeadlessRehydrateVerdict {
	if stdinInteractive || !pinUpstream || strings.TrimSpace(credPath) == "" {
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
}

type guardChildLauncher func(toolprocgate.SpawnGrant) (*exec.Cmd, error)

const (
	guardCodexTerminalRestorePulseDuration = 8 * time.Second
	guardCodexTerminalRestorePulseInterval = 500 * time.Millisecond
)

var startGuardChildTerminalRestorePulse = windowgate.StartTerminalRestorePulse

func maybeStartGuardChildTerminalRestorePulse(command []string) {
	if len(command) == 0 || !guardIsCodex(command[0]) {
		return
	}
	startGuardChildTerminalRestorePulse(guardCodexTerminalRestorePulseDuration, guardCodexTerminalRestorePulseInterval)
}

// maybeStartGuardChildHarnessTerminalRestorePulse repairs the parent terminal after a
// wrapped interactive harness starts. Codex already needed the pulse because its launch
// can perturb the console window; Claude can leave the same stale/hidden terminal state,
// so production launch paths cover both harnesses through the same restore seam.
func maybeStartGuardChildHarnessTerminalRestorePulse(command []string) {
	if len(command) == 0 {
		return
	}
	if guardAgentBaseName(command[0]) == "claude" {
		startGuardChildTerminalRestorePulse(guardCodexTerminalRestorePulseDuration, guardCodexTerminalRestorePulseInterval)
		return
	}
	maybeStartGuardChildTerminalRestorePulse(command)
}

func newGuardChildSpawnMetadata(agentRunID, policyDigest, backend string, rt policy.Runtime, command []string) guardChildSpawnMetadata {
	agentRunID = strings.TrimSpace(agentRunID)
	if agentRunID == "" {
		agentRunID = "guard"
	}
	env := toolprocgate.CapabilityEnvelope{
		Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn},
	}
	if len(command) > 0 && rt.ToolRuntime != nil {
		if r, ok := rt.ToolRuntime.EnvelopeFor(guardAgentBaseName(command[0])); ok {
			env.DeadlineMS = r.DeadlineMS
			env.HeartbeatEveryMS = r.HeartbeatEveryMS
		}
	}
	return guardChildSpawnMetadata{
		AgentRunID:   agentRunID,
		ToolCallID:   "guard-child:" + agentRunID,
		PolicyDigest: strings.TrimSpace(policyDigest),
		Backend:      strings.TrimSpace(backend),
		Envelope:     env,
	}
}

// buildGuardChild constructs the wrapped-agent command with ONLY the gateway URL injected
// into its environment (never the parent shell). In pinned subscription mode it also hands
// the client a provider-shaped placeholder API key (when it has none) so it talks to the
// gateway, which ignores the placeholder and authenticates upstream with the held token.
const guardActiveEnv = "FAK_GUARD_ACTIVE"

func buildGuardChild(command []string, injected [][2]string, pinUpstream bool, extraEnv ...[2]string) *exec.Cmd {
	command, env := guardChildCommandEnv(command, injected, pinUpstream, extraEnv...)
	child := exec.Command(command[0], command[1:]...)
	child.Stdin, child.Stdout, child.Stderr = os.Stdin, os.Stdout, os.Stderr
	child.Env = env
	return child
}

func guardChildCommandEnv(command []string, injected [][2]string, pinUpstream bool, extraEnv ...[2]string) ([]string, []string) {
	// Landlock hook-floor (opt-in, Linux): rewrite the agent argv so the child is launched
	// through the fak re-exec trampoline, which applies the read-only-.git/hooks ruleset to
	// itself before exec'ing the agent. Off by default, no-op on non-Linux or when the hook
	// dirs cannot be resolved — the original command is used unchanged.
	command = maybeLandlockCommand(command)
	// Apply the always-on #2358 secret floor to the AMBIENT parent environment
	// before it is inherited by the wrapped agent: a spawned child (and anything
	// it spawns) must not receive inherited credentials it never needed. Only the
	// ambient os.Environ() portion is stripped — the injected gateway wiring, the
	// caller's extraEnv, and the placeholder key below are guard's EXPLICIT grants,
	// appended after, and always survive. This is safe for every guard auth
	// posture: guard already resolved and captured its own upstream OAuth token in
	// THIS (parent) process before spawning, and the provider API keys an
	// API-billing child needs are spared by StripInheritedSecrets.
	env, _ := policy.StripInheritedSecrets(os.Environ())
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
	isCodex := len(command) > 0 && guardIsCodex(command[0])
	if pinUpstream && !isCodex && os.Getenv("ANTHROPIC_API_KEY") == "" {
		env = append(env, "ANTHROPIC_API_KEY=fak-guard-oauth-placeholder")
	}
	if pinUpstream && isCodex && os.Getenv("OPENAI_API_KEY") == "" {
		env = append(env, "OPENAI_API_KEY="+guardCodexOAuthPlaceholderAPIKey)
	}
	return command, env
}

func guardChildSpawnAttempt(command []string, injected [][2]string, pinUpstream bool, meta guardChildSpawnMetadata, extraEnv ...[2]string) (toolprocgate.SpawnAttempt, error) {
	command, envStrings := guardChildCommandEnv(command, injected, pinUpstream, extraEnv...)
	env, err := toolprocgate.EnvFromStrings(envStrings)
	if err != nil {
		return toolprocgate.SpawnAttempt{}, err
	}
	cwd, err := os.Getwd()
	if err != nil {
		return toolprocgate.SpawnAttempt{}, err
	}
	return toolprocgate.SpawnAttempt{
		AgentRunID:   meta.AgentRunID,
		ParentRunID:  meta.ParentRunID,
		ToolCallID:   meta.ToolCallID,
		PolicyDigest: meta.PolicyDigest,
		Argv:         command,
		Env:          env,
		CWD:          cwd,
		Backend:      meta.Backend,
		Envelope:     meta.Envelope,
	}, nil
}

func launchGuardChildWithBroker(command []string, injected [][2]string, pinUpstream bool, meta guardChildSpawnMetadata, broker *toolprocgate.SpawnBroker, launcher guardChildLauncher, extraEnv ...[2]string) (toolprocgate.SpawnGrant, *exec.Cmd, error) {
	if broker == nil {
		broker = toolprocgate.NewSpawnBroker()
	}
	attempt, err := guardChildSpawnAttempt(command, injected, pinUpstream, meta, extraEnv...)
	if err != nil {
		return toolprocgate.SpawnGrant{}, nil, err
	}
	grant, err := broker.Admit(attempt)
	if err != nil {
		return toolprocgate.SpawnGrant{}, nil, err
	}
	if launcher == nil {
		launcher = guardExecLauncher
	}
	child, err := launcher(grant)
	if err != nil {
		return toolprocgate.SpawnGrant{}, nil, err
	}
	return grant, child, nil
}

func guardExecLauncher(grant toolprocgate.SpawnGrant) (*exec.Cmd, error) {
	if len(grant.Argv) == 0 {
		return nil, fmt.Errorf("empty brokered argv")
	}
	child := exec.Command(grant.Argv[0], grant.Argv[1:]...)
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
	// seedHandback selects the #3056 headless/no-continue handback: inject the carryover
	// seed_text as the recognized child's initial prompt on relaunch instead of the default
	// #3055 --continue transcript reattach. Set from the --restart-seed-handback knob.
	seedHandback bool
	stderr       io.Writer
	events       chan guardBudgetRestartEvent
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

// guardRestartRelaunchCommand returns the command to relaunch the wrapped child with after a
// budget restart. For a recognized agent (Claude Code) it REATTACHES the existing transcript by
// appending the agent's resume flag (`--continue`), so the relaunched child resumes the same
// conversation the carryover seed was captured from instead of booting a cold, empty session and
// reporting "I don't have the task" (#3055). The FAK_RESET_* env vars guardRestartEnv sets are
// advisory only — no in-child reader consumes them — so continuity must come from the agent's own
// resume path: the exact flag formatGuardResumeGuidance already tells operators to run by hand, and
// the one guardMaybeRecoverAuthCrash already auto-injects on the auth-crash path. Idempotent via
// guardAppendContinueFlag: a second restart in the same session never stacks the flag. For an
// unrecognized agent fak cannot guess a safe resume syntax, so command is returned unchanged and
// the relaunch falls back to today's cold behavior (the headless/no-continue seed-prompt handback
// is the separate #3056 rung).
func guardRestartRelaunchCommand(command []string, agentName string) []string {
	if flag, ok := guardContinueFlagForAgent(agentName); ok {
		return guardAppendContinueFlag(command, flag)
	}
	return command
}

// guardSeedPromptTokenBudget is the documented ceiling on a carryover seed re-injected as a
// relaunch prompt (#3056). Measured in guardApproxTokens' ~4-bytes/token gauge, so ~8 KB of seed
// prose. A relaunch prompt is a task RE-ORIENTATION — the load-bearing "what were you doing"
// carryover — not the whole transcript (reattaching that is the --continue path's job, #3055), so a
// few thousand tokens is the ceiling; anything past it is truncated AND logged, never silently.
const guardSeedPromptTokenBudget = 2000

// guardBoundSeedPrompt truncates seed to at most tokenBudget approx-tokens (guardApproxTokens'
// 4-bytes/token gauge), cutting on a UTF-8 rune boundary so a multi-byte rune is never split. It
// returns the bounded text and the number of dropped approx-tokens — 0 when the seed already fit.
// A non-zero drop is the caller's cue to LOG what was dropped: the bound is never silent (#3056).
func guardBoundSeedPrompt(seed string, tokenBudget int) (bounded string, droppedTokens int) {
	if tokenBudget <= 0 {
		return seed, 0
	}
	total := guardApproxTokens(seed)
	if total <= tokenBudget {
		return seed, 0
	}
	keep := tokenBudget * 4 // approx-tokens back to a byte budget (guardApproxTokens is ceil(len/4))
	if keep >= len(seed) {
		return seed, 0
	}
	// Back up off any UTF-8 continuation byte (0b10xxxxxx) so the cut lands on a rune start and a
	// multi-byte rune is never split mid-sequence.
	for keep > 0 && seed[keep]&0xC0 == 0x80 {
		keep--
	}
	bounded = seed[:keep]
	return bounded, total - guardApproxTokens(bounded)
}

// guardSeedPromptRelaunchCommand injects the bounded carryover seed_text as the recognized child's
// initial prompt on a headless/no-continue relaunch (#3056) — the handback the operator selects with
// --restart-seed-handback for a deliberately fresh-session child (e.g. `claude -p`) that the #3055
// --continue reattach does not serve. On success it returns the augmented command, the "seed-prompt"
// handback mode, and injected=true; it LOGS the dropped approx-token/byte count whenever the seed is
// truncated past guardSeedPromptTokenBudget. It is a NO-OP — (command, "", false) — for an
// unrecognized agent (fak never guesses a foreign tool's prompt syntax; the seed stays on disk
// unread) or an empty seed. Idempotent across repeated restarts: a prior injected seed VALUE is
// replaced with the fresher one rather than stacking a second flag. The input command is never
// mutated in place.
func guardSeedPromptRelaunchCommand(command []string, agentName, seedText string, log io.Writer) (out []string, handback string, injected bool) {
	seed := strings.TrimSpace(seedText)
	flag, ok := guardSeedPromptFlagForAgent(agentName)
	if !ok || seed == "" {
		return command, "", false
	}
	bounded, droppedTokens := guardBoundSeedPrompt(seed, guardSeedPromptTokenBudget)
	if droppedTokens > 0 && log != nil {
		fmt.Fprintf(log, "fak guard: seed-prompt handback bounded carryover for %s to %d approx-tokens (budget %d); dropped %d approx-tokens / %d bytes — no silent truncation\n",
			guardAgentBaseName(agentName), guardApproxTokens(bounded), guardSeedPromptTokenBudget, droppedTokens, len(seed)-len(bounded))
	}
	out = make([]string, len(command), len(command)+2)
	copy(out, command)
	// A prior restart in the same session may already carry an injected seed flag; replace its
	// value with the fresher seed rather than appending a second --append-system-prompt.
	for i := 1; i+1 < len(out); i++ {
		if out[i] == flag {
			out[i+1] = bounded
			return out, guardRestartHandbackSeedPrompt, true
		}
	}
	out = append(out, flag, bounded)
	return out, guardRestartHandbackSeedPrompt, true
}

func guardRestartLimitStatus(limit int, ev guardBudgetRestartEvent) string {
	reason := strings.TrimSpace(ev.Reason)
	if reason == "" {
		reason = "BUDGET_CONTEXT_EXHAUSTED"
	}
	continuity := "degraded"
	if strings.TrimSpace(ev.ToTraceID) == "" && strings.TrimSpace(ev.SeedFile) == "" && strings.TrimSpace(ev.SeedText) == "" {
		continuity = "blocked"
	}
	next := "raise --restart-limit or restart manually after the budget window clears"
	if trace := strings.TrimSpace(ev.ToTraceID); trace != "" {
		next = "raise --restart-limit or restart the child with FAK_RESET_TRACE_ID=" + trace
	}
	if seed := strings.TrimSpace(ev.SeedFile); seed != "" {
		// ToSlash: %q below escapes backslashes, so an unconverted Windows path
		// (filepath.Join's native separator) would render as seeds\\reset.json —
		// doubled backslashes a plain-substring check (or a human) never expects.
		// Forward-slash normalization keeps the seed path byte-identical in the
		// %q-quoted next_action field on every OS.
		seed = strings.ReplaceAll(filepath.ToSlash(seed), "\\", "/")
		next += " and FAK_RESET_SEED_FILE=" + seed
	}
	return fmt.Sprintf("fak guard: managed-context status reset_limit limit=%d reason=%s continuity=%s next_action=%q",
		limit, reason, continuity, next)
}

// guardChildIsLaunchFailure reports whether runErr is a FAILURE TO LAUNCH (a spawn/exec error
// — the binary is missing, not executable, a bad path) rather than a normal run that exited
// non-zero. An *exec.ExitError means the child DID start and then exited, so it is never a
// launch failure; a nil error is a clean run. Everything else (exec.Error, a PathError from
// exec.Command().Run()) is the child never starting — the one case the compact/animate launch
// spills the full startup report for. Pure, so the classification is unit-tested.
func guardChildIsLaunchFailure(runErr error) bool {
	if runErr == nil {
		return false
	}
	_, isExit := runErr.(*exec.ExitError)
	return !isExit
}

// guardDumpStartupReportOnLaunchFail spills the full recorded startup report to w when the
// child failed to launch. This is the one case the compact/animate banner deliberately
// withholds the wall of text for: on a launch failure no agent TUI ever took the terminal, so
// the full floor/hook/auth detail is exactly what the operator needs, co-located with the
// error. enabled is false when the full report already streamed at boot (--banner=full) so the
// text is never printed twice; a nil Server or an unrecorded report is a silent no-op. It reads
// the report off the gateway and hands the formatting to the pure guardWriteLaunchFailReport.
func guardDumpStartupReportOnLaunchFail(w io.Writer, srv *gateway.Server, enabled bool) {
	if srv == nil {
		return
	}
	guardWriteLaunchFailReport(w, srv.StartupReport(), enabled)
}

// guardWriteLaunchFailReport writes the report under a "launch failed" header when enabled and
// the report is non-empty, and is a no-op otherwise. Split from the Server read so the exact
// header + body format is unit-tested without standing up a gateway.
func guardWriteLaunchFailReport(w io.Writer, report string, enabled bool) {
	if !enabled || strings.TrimSpace(report) == "" {
		return
	}
	report = strings.TrimRight(report, "\n")
	fmt.Fprintln(w, "fak guard: launch failed — full startup report (the detail an attended launch keeps in `fak info --startup`):")
	fmt.Fprintln(w, report)
}

// runGuardChildAndReport runs the wrapped agent to completion, tears the gateway down,
// prints the session's adjudication + journal summary (unless quiet), flushes the durable
// trail, and exits with the child's own code — surfacing a gateway-mid-session failure as
// a non-silent error so a clean child exit never hides a downed adjudication boundary.
//
// Before reporting a non-zero exit, it gives guardMaybeRecoverAuthCrash (the mid-session
// counterpart to the #1834 pre-spawn rehydrate rung) a chance to diagnose an expired
// subscription token and, if a fresh login lands within the recovery window, relaunch the SAME
// command with a resume flag appended — so a crash caused by auth expiry self-heals within this
// guarded session instead of always needing a manual re-run. credPath is empty when guard is not
// pinning the Claude subscription upstream, which makes the check an unconditional no-op there.
//
// dumpStartupOnLaunchFail spills the full startup report to stderr if the child never starts
// (guardChildIsLaunchFailure) — set by the caller for every banner mode except --banner=full,
// which already streamed it at boot.
func runGuardChildAndReport(command []string, injected [][2]string, pinUpstream bool, credPath string, rotation *guardRotationRuntime, spawnMeta guardChildSpawnMetadata, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, guardTraceID, agentName, provider string, dojoMode bool, sampler *harnessres.Sampler, dumpStartupOnLaunchFail bool) {
	spawnBroker := toolprocgate.NewSpawnBroker()
	for {
		_, child, err := launchGuardChildWithBroker(command, injected, pinUpstream, spawnMeta, spawnBroker, rotation.launcher())
		if err != nil {
			guardDumpStartupReportOnLaunchFail(os.Stderr, srv, dumpStartupOnLaunchFail)
			finishGuardChildAndReport(err, nil, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		maybeStartGuardChildHarnessTerminalRestorePulse(command)
		childStarted := time.Now()
		rotationEvidenceBefore := srv.RotationEvidenceSnapshot()
		runErr := child.Run()
		if next, ok := guardMaybeRecoverAuthCrash(runErr, command, credPath, agentName, quiet, os.Stderr); ok {
			command = next
			continue
		}
		if nextCommand, nextInjected, ok := rotation.rotateAfterExit(runErr, guardRotationEvidenceSince(rotationEvidenceBefore, srv.RotationEvidenceSnapshot()), command, injected, auditJournal, guardTraceID, os.Stderr); ok {
			command, injected = nextCommand, nextInjected
			continue
		}
		if next, ok := guardMaybeRecoverCapCrash(runErr, command, agentName, childStarted, quiet, nil, nil, os.Stderr); ok {
			command = next
			continue
		}
		if guardChildIsLaunchFailure(runErr) {
			guardDumpStartupReportOnLaunchFail(os.Stderr, srv, dumpStartupOnLaunchFail)
		}
		finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
		return
	}
}

// guardTimeBudgetTickInterval is the coarse cadence at which the supervision loop
// polls the session wall-clock envelope. 15s is fine-grained enough that a
// --max-duration overshoot is bounded to one tick, and coarse enough to add no
// measurable overhead to a guarded run.
const guardTimeBudgetTickInterval = 15 * time.Second

// guardTimeBudgetExhausted is the production caller of the session wall-clock gate
// that issue #2229 found missing: the supervision ticker asks it, once per tick,
// whether traceID's --max-duration envelope has elapsed as of now. It returns
// (true, TIME_BUDGET_EXHAUSTED) only on a genuine wall-clock exhaustion — an
// unbounded (--max-duration 0), still-within-budget, paused, or terminal session
// reports (false, "") and is left untouched, so an unconfigured envelope never
// stops a run. The transition to Draining/Stopped and the final elapsed-time fold
// are done by DecideTimeBudget itself (see internal/session/timebudget.go); this
// wrapper only classifies its verdict for the loop, and is the seam #2229's
// enforcement test drives.
func guardTimeBudgetExhausted(sessions *session.Table, traceID string, now time.Time) (bool, string) {
	if sessions == nil || strings.TrimSpace(traceID) == "" {
		return false, ""
	}
	v := sessions.DecideTimeBudget(traceID, now)
	if v.Stop && v.Reason == session.ReasonTimeBudgetExhausted {
		return true, v.Reason
	}
	return false, ""
}

func runGuardChildSupervisedAndReport(command []string, injected [][2]string, pinUpstream bool, credPath string, rotation *guardRotationRuntime, spawnMeta guardChildSpawnMetadata, restarter *guardBudgetRestarter, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, guardTraceID, agentName, provider string, dojoMode bool, sampler *harnessres.Sampler, dumpStartupOnLaunchFail bool) {
	spawnBroker := toolprocgate.NewSpawnBroker()
	var extraEnv [][2]string
	restarts := 0
	// Wall-clock enforcement (#2229): poll the session time budget on a coarse ticker so a
	// --max-duration envelope is actually ENFORCED here, not merely armed/persisted/displayed.
	// guardTimeBudgetExhausted is a no-op for an unbounded/paused/still-fine session, so a run
	// without a configured envelope is untouched; on a TIME_BUDGET_EXHAUSTED verdict the child
	// is stopped through the existing stopGuardChild path and the session is reported.
	budgetTicker := time.NewTicker(guardTimeBudgetTickInterval)
	defer budgetTicker.Stop()
	for {
		_, child, err := launchGuardChildWithBroker(command, injected, pinUpstream, spawnMeta, spawnBroker, nil, extraEnv...)
		wait := make(chan error, 1)
		if err != nil {
			guardDumpStartupReportOnLaunchFail(os.Stderr, srv, dumpStartupOnLaunchFail)
			finishGuardChildAndReport(err, nil, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		maybeStartGuardChildHarnessTerminalRestorePulse(command)
		childStarted := time.Now()
		rotationEvidenceBefore := srv.RotationEvidenceSnapshot()
		if err := child.Start(); err != nil {
			// child.Start() failing IS a launch failure (the child never ran); spill the detail.
			guardDumpStartupReportOnLaunchFail(os.Stderr, srv, dumpStartupOnLaunchFail)
			finishGuardChildAndReport(err, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		go func() { wait <- child.Wait() }()
		event := waitGuardChild(wait, restarter.events, budgetTicker.C, func(now time.Time) (bool, string) {
			return guardTimeBudgetExhausted(serveSessions, guardTraceID, now)
		})
		switch event.Kind {
		case guardChildCompleted:
			runErr := event.RunErr
			if next, ok := guardMaybeRecoverAuthCrash(runErr, command, credPath, agentName, quiet, os.Stderr); ok {
				command = next
				continue
			}
			if nextCommand, nextInjected, ok := rotation.rotateAfterExit(runErr, guardRotationEvidenceSince(rotationEvidenceBefore, srv.RotationEvidenceSnapshot()), command, injected, auditJournal, guardTraceID, os.Stderr); ok {
				command, injected = nextCommand, nextInjected
				continue
			}
			if next, ok := guardMaybeRecoverCapCrash(runErr, command, agentName, childStarted, quiet, nil, nil, os.Stderr); ok {
				command = next
				continue
			}
			finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		case guardChildRestart:
			ev := event.Restart
			if restarter.limit > 0 && restarts >= restarter.limit {
				if restarter.stderr != nil {
					fmt.Fprintln(restarter.stderr, guardRestartLimitStatus(restarter.limit, ev))
				}
				runErr := <-wait
				finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
				return
			}
			restarts++
			// Decide the relaunch command AND the handback mode it represents, so the
			// hop recorded below matches the command actually launched.
			handback := ""
			if restarter.seedHandback {
				// #3056: headless/no-continue handback — inject the bounded carryover seed as the
				// recognized child's initial prompt (--append-system-prompt) instead of reattaching
				// the transcript. A no-op for an unrecognized agent or an empty seed, which falls
				// through to the #3055 default below.
				if next, hb, injected := guardSeedPromptRelaunchCommand(command, agentName, ev.SeedText, restarter.stderr); injected {
					command, handback = next, hb
				}
			}
			if handback == "" {
				// #3055: reattach the existing transcript on relaunch. The FAK_RESET_* env vars
				// set below are advisory only (Claude Code reads none of them), so continuity comes
				// from the wrapped agent's own resume flag — a recognized child resumes the captured
				// conversation instead of booting cold and losing the task. Idempotent across repeated
				// restarts; an unrecognized agent is relaunched unchanged (handback derives to
				// continue/ORPHANED in the hop below).
				command = guardRestartRelaunchCommand(command, agentName)
			}
			// #3057: ONE correlated record per restart — a RESTART_HOP row in the
			// guard audit journal plus a single stderr one-liner carrying the same
			// fields (from/to trace, seed size + file, handback mode, child session,
			// continuity status) — replacing the two uncorrelated lines that used to
			// be the only evidence a hidden relaunch ever happened. Queryable later
			// via `fak guard restart-audit` and `fak session status <id>`.
			guardEmitRestartHop(auditJournal, restarter.stderr, agentName, guardTraceID,
				guardRestartHopFromEventHandback(ev, restarts, agentName, handback))
			srv.SetDefaultTraceID(ev.ToTraceID)
			extraEnv = guardRestartEnv(ev)
			// Let the triggering response finish flushing to the wrapped client before
			// stopping the process that initiated it.
			time.Sleep(750 * time.Millisecond)
			stopGuardChild(child, wait, 2*time.Second)
		case guardChildTimeBudget:
			// The wall-clock envelope elapsed: stop the wrapped agent and report, rather
			// than let it keep burning tokens past its --max-duration (the #2229 gap).
			if !quiet {
				fmt.Fprintf(os.Stderr, "fak guard: %s — wall-clock --max-duration envelope elapsed for %s; stopping the wrapped agent\n", event.Reason, guardTraceID)
			}
			stopGuardChild(child, wait, 2*time.Second)
			finishGuardChildAndReport(nil, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
	}
}

// guardChildTreeKill is the destructive escalation used when a wrapped child does not exit
// within the graceful-interrupt window. It defaults to procguard.KillPID — a process-TREE
// kill (native job termination / taskkill /T on Windows, process-group/descendant SIGKILL on
// POSIX) — so the child's own descendants (the node runtime, MCP/tool subprocesses a `claude`
// spawns) are reaped too. A bare child.Process.Kill() only reaps the immediate PID and leaves
// that subtree orphaned; across a budget restart those orphans accumulate under one guard
// parent (#2989: three live claude children from two restarts, poisoning dispatch preflight
// with unattributed_live). Injectable for tests. Mirrors fleetKillPID in fleet.go.
var guardChildTreeKill = procguard.KillPID

// stopGuardChild stops the wrapped child on restart/timeout. It first asks politely
// (os.Interrupt) and waits out the grace window; if the child is still alive it escalates to a
// TREE kill of the child's PID so no descendant subtree survives the transition. Returning
// only after <-wait guarantees the previous child is fully reaped before the caller relaunches,
// so a budget restart is single-child (#2989).
func stopGuardChild(child *exec.Cmd, wait <-chan error, grace time.Duration) {
	if child == nil || child.Process == nil {
		return
	}
	_ = child.Process.Signal(os.Interrupt)
	select {
	case <-wait:
		return
	case <-time.After(grace):
		// Tree-kill: reap the child AND its descendants, not just the immediate PID.
		guardChildTreeKill(child.Process.Pid)
		<-wait
	}
}

// formatGuardResumeGuidance is the concise, actionable note printed when the wrapped agent
// exits abnormally (a non-zero code — a crash, an OOM, or a terminal upstream error). The
// guard process holds no agent conversation state itself — the wrapped tool owns that — so
// "resume" means re-running the same `fak guard -- <command>` with the agent's own
// resume/continue flag. The last line encodes a hard-won recovery: a guarded resume that
// dies IMMEDIATELY with "upstream model error" (a malformed-request rejection that can
// follow a mid-conversation quarantine) usually clears if that one resume is retried WITHOUT
// fak guard, then re-wrapped. Returned as a string (not printed) so it is unit-testable.
func formatGuardResumeGuidance(agentName string, code int) string {
	return formatGuardResumeGuidanceWithRefusals(agentName, code, nil)
}

func formatGuardResumeGuidanceWithRefusals(agentName string, code int, refusals []guardRefusalCarry) string {
	guardActivity := ""
	if len(refusals) > 0 {
		var b strings.Builder
		b.WriteString("  guard activity: the last guarded run hit floor refusal(s); treat the resume as recovery/debugging, not a blind retry.\n")
		for _, item := range refusals {
			if strings.TrimSpace(item.Reason) == "" {
				continue
			}
			fmt.Fprintf(&b, "    - %s x%d", item.Reason, item.Count)
			if fix := strings.TrimSpace(item.Fix); fix != "" {
				fmt.Fprintf(&b, " — fix: %s", fix)
			}
			b.WriteByte('\n')
		}
		b.WriteString("  guard resume: keep fak guard wrapped after clearing the refusal; do not retry the same refused call unchanged.\n")
		guardActivity = b.String()
	}
	return fmt.Sprintf(
		"\nfak guard: %s exited abnormally (code %d).\n"+
			"  resume: re-run the same `fak guard -- %s …` — launch the agent with its own resume/continue flag (e.g. `claude --continue`) so it picks the conversation back up.\n"+
			"%s"+
			"  this session's decision journal is replayable with `fak audit verify` (path in the audit summary above).\n"+
			"  if a guarded resume dies IMMEDIATELY with \"upstream model error\", retry that one resume WITHOUT fak guard to recover, then re-wrap.\n",
		agentName, code, agentName, guardActivity)
}

// formatVCacheSnapshotPointer is the exit pointer that closes the loop between the LIVE
// guard cache summary and the OFFLINE `fak vcache` family. After a session persists its
// OBSERVED provider-cache window (vcachesnapshot.Write), this line tells the operator the
// window is on disk AND names the one command that re-derives THIS session's REALIZED cache
// multiplier from it: `fak vcache score` reads the well-known snapshot path with no flags
// (it folds the same turns through the same vcacheobserve engine the summary line priced),
// and `fak vcache observe` renders the per-sub-concept panels. Without this, the snapshot is
// written silently and the related vcache tools look like they only have a synthetic forecast
// to chew on — the operator never learns the real session data is right there to replay.
//
// Empty when no turns were recorded (a run that never saw a cached turn writes an empty
// snapshot, which the score correctly treats as "no observed window" and falls open to the
// forecast), so a no-cache session stays quiet rather than printing a vacuous 0-turn pointer.
func formatVCacheSnapshotPointer(turns int, path string) string {
	if turns <= 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString(guardSection("cache window"))
	b.WriteString(guardRow("recorded", fmt.Sprintf("%d turn(s)", turns)))
	b.WriteString(guardRow("to", path))
	b.WriteString(guardRow("replay realized multiplier", "fak vcache score"))
	b.WriteString(guardNote("no flags reads this snapshot; `fak vcache observe` renders the per-sub-concept panels"))
	return b.String()
}

// guardSummaryResetPrefix is the terminal escape the exit summary emits before its first line
// so it never inherits a dangling SGR style or a hidden cursor the wrapped agent's torn-down
// alt-screen left. "\x1b[0m" resets all SGR attributes (color, bold, reverse), "\x1b[?25h"
// re-shows the cursor a TUI may have hidden. It is emitted ONLY to a real terminal (isTTY): a
// summary piped to a file or a `-p` JSON capture must stay byte-clean, so a non-TTY sink gets
// the empty string. Pure (string in, string out) so the TTY-gated behavior is unit-tested.
func guardSummaryResetPrefix(isTTY bool) string {
	if !isTTY {
		return ""
	}
	return "\x1b[0m\x1b[?25h"
}

// guardClassifyChildCrash maps a completed child's exit into the closed crash
// vocabulary for a CHILD_CRASH journal row. It is pure and portable (no Unix-only
// syscall.WaitStatus — it reads only the exit code and the ProcessState's portable
// String(), so the SAME classification runs on the Windows dev host and the Unix
// fleet). isCrash is false for a clean exit (nil runErr, or a zero exit code) and
// for a run failure that never produced a child (a spawn error is reported by the
// caller's existing path, not journaled as a crash). When isCrash is true, class
// is one of the journal.Crash* constants and exitCode is the child's code (-1 when
// the platform reports a signaled kill).
//
//   - A signaled kill that looks like an OOM (exit 137 = 128+SIGKILL, or a state
//     string naming "killed") -> OOM: a resource exhaustion the loop tracks apart
//     from a logic fault.
//   - Any other signaled kill (ExitCode -1, or a state string naming "signal") ->
//     SIGNAL_CRASH: SIGSEGV/SIGABRT/an external kill.
//   - A plain non-zero exit (a Go panic the runtime turned into exit 2, an
//     unrecovered error) -> NONZERO_EXIT.
func guardClassifyChildCrash(runErr error, childState *os.ProcessState) (class string, exitCode int, isCrash bool) {
	ee, isExit := runErr.(*exec.ExitError)
	if !isExit {
		return "", 0, false // nil, or a spawn failure the caller reports directly
	}
	code := ee.ExitCode()
	if code == 0 {
		return "", 0, false // an ExitError that somehow carries a clean code is not a crash
	}
	stateStr := ""
	if childState != nil {
		stateStr = strings.ToLower(childState.String())
	}
	signaled := code < 0 || strings.Contains(stateStr, "signal")
	oom := code == 137 || strings.Contains(stateStr, "killed")
	switch {
	case oom:
		return journal.CrashOOM, code, true
	case signaled:
		return journal.CrashSignal, code, true
	default:
		return journal.CrashNonzeroExit, code, true
	}
}

func finishGuardChildAndReport(runErr error, childState *os.ProcessState, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, guardTraceID, agentName, provider string, dojoMode bool, sampler *harnessres.Sampler) {
	var currentRefusals []guardRefusalCarry

	// Tear the gateway down and report what the kernel decided this session.
	if sampler != nil {
		sampler.FoldChildExit(childState)
	}
	cancel()
	serr := <-serveErr
	// The session's wall-clock window, when the resource sampler tracked one — the
	// honest scope for the hook-latency exit line below (0 = no anchor, fold all-time).
	var sessionWindow time.Duration
	if sampler != nil {
		snap := sampler.Stop()
		snap.GuardStops = foldGuardStopCounts()
		sessionWindow = snap.Elapsed
		appendHarnessResources("guard", provider, agentName, snap)
		if !quiet {
			fmt.Fprintln(os.Stderr, "fak guard: "+snap.Report())
		}
	}
	if !quiet {
		// The wrapped agent (Claude Code) paints a full-screen alternate-screen TUI over this
		// same terminal and, on a crash or an abnormal exit, can tear it down mid-escape-sequence
		// — leaving a dangling SGR color/style or a hidden cursor. The exit summary then renders
		// mis-colored or invisible. Emit a soft reset (SGR reset + show-cursor) onto a clean
		// baseline FIRST, but only when stderr is a real terminal: piping the summary to a file or
		// a JSON capture must stay byte-clean, so a non-TTY stderr gets no escape bytes.
		summaryTTY := guardFdIsTerminal(int(os.Stderr.Fd()))
		fmt.Fprint(os.Stderr, guardSummaryResetPrefix(summaryTTY))
		fmt.Fprintln(os.Stderr)
		sum := srv.AdjudicationSummary()
		kc := srv.KernelCounters()
		// Each formatter returns byte-clean text (no escape bytes — pure and unit-tested);
		// color is layered on HERE, gated on a real terminal, so the section rules read as
		// headings and the demoted notes recede while the value rows stand out, yet a piped
		// or JSON-captured summary stays exactly the plain text the tests assert.
		emit := func(text string) { fmt.Fprint(os.Stderr, guardColorizeSummary(text, summaryTTY)) }
		emit(formatAuditSummary(sum, kc))
		emit(formatAmplification(kc, sum))
		emit(formatTurnsTimeSaved(kc, sum))
		emit(formatJournalSummary(auditJournal, auditSeq0))
		// The guard-hook wall-clock tax (#1993): what the pre+post adjudication hooks
		// cost per tool call this session. Best-effort — a hook-less session prints
		// nothing rather than a vacuous zero row.
		emit(guardHookLatencySummaryLine(sessionWindow, time.Now()))
		// The tool process table (#2445): any event-stream monitor that went silent
		// past its cadence this session folded to a killed TOOL_HEARTBEAT_STALLED —
		// surface that count here (silence is not success). Best-effort, empty when
		// nothing is outstanding.
		emit(guardToolprocSummary(time.Now()))
		emit(guardTrajectoryWarningLine())
		// The context-health verdict (#3099): fold this session's LIVE trajectory
		// corpus through the #3098 HEALTHY/STALL/DRIFT/DETOUR_OVERRUN scorer + the
		// #3096 shed-span use-after-free detector onto the same guard status channel,
		// so a repeat-failure loop or a shed-then-reference is visible here, not only
		// in a post-hoc `fak traj score`. Empty (silent) unless trajectory recording
		// is on, exactly like the sibling lines stay quiet when their signal is absent.
		emit(guardContextHealthLine())
	}
	// Append cache-value observation to ledger (epic #1072, issue #1075) AND surface it.
	// Persist both tracks, then — for a non-quiet (interactive) session — print the
	// dollar-aware evidence summary so the operator SEES the savings this session earned
	// instead of it landing silently in the ledger. The summary formatter had been built
	// and tested but wired to nothing (detection-without-enforcement); this is the felt
	// "usage savings moment" for a `fak c` / `fak guard` session. A headless --quiet
	// worker still persists silently, exactly as before.
	cvReport := buildCacheValuePersistenceReport(srv, "guard", agentName, provider, cachevalueledger.DefaultLedgerRel, time.Now())
	if !quiet {
		fmt.Fprint(os.Stderr, guardColorizeSummary(formatCacheValuePersistenceSummary("fak guard", cvReport), guardFdIsTerminal(int(os.Stderr.Fd()))))
	}
	// Append the gateway-usage exit row (#1610), same writer as the serve exits, so a
	// guard session's full served-turn counter family — compaction fired/bailed/shed
	// among them — survives the process instead of dying with the console summary.
	// Until now only `fak serve` exits reached this ledger, so the per-session WHY
	// behind a zero fak-slice (burst_unprofitable vs anchor-starved vs under_budget)
	// was unrecoverable after exit on the flagship guard path (epic #1601 gap).
	persistGatewayUsageObservation(srv, "guard", agentName, sessionWindow)
	if dojoMode {
		_ = persistLiveDojoEpisode("guard", srv)
	}
	// Persist this session's OBSERVED provider-cache window so a later `fak vcache score`
	// (a separate process) reports the REALIZED multiplier from real traffic instead of the
	// synthetic-Zipf forecast (#1090). Best-effort: a write failure never fails the session,
	// and an empty window leaves the snapshot empty so the score falls open to the forecast.
	// On a clean write, point the operator at the offline vcache tools that now hold this
	// session's data — otherwise the snapshot is silent and the related vcache items look
	// like they only have a synthetic forecast to score.
	if turns, _ := srv.VCacheTurnsSnapshot(); len(turns) > 0 {
		if snapPath, ok, err := writeConfiguredVCacheSnapshot(turns); err == nil && ok && !quiet {
			fmt.Fprint(os.Stderr, formatVCacheSnapshotPointer(len(turns), snapPath))
		}
	}
	// Flush + fsync the durable trail before exit so a row returned to the agent is
	// never lost to a buffered write (Close is safe on a nil/in-memory journal).
	if auditJournal != nil {
		// If the wrapped child died abnormally, record a durable CHILD_CRASH row BEFORE
		// close so the guard-RSI loop — which folds only this journal — can see the
		// crash that verdict rows structurally cannot carry. Written directly through
		// the chain (a crash is not a kernel decision); a clean exit writes nothing.
		if class, code, isCrash := guardClassifyChildCrash(runErr, childState); isCrash {
			row := auditJournal.AppendCrash(agentName, guardTraceID, class, code)
			if !quiet {
				fmt.Fprintf(os.Stderr, "fak guard: recorded CHILD_CRASH (%s, exit %d) to the audit journal at seq %d\n", class, code, row.Seq)
			}
		}
		var err error
		currentRefusals, err = guardWriteRefusalCarryForwardAndReturn(auditJournal, auditSeq0, guardTraceID, guardFindReasonRoot())
		if err != nil && !quiet {
			fmt.Fprintf(os.Stderr, "fak guard: refusal carry-forward unavailable: %v\n", err)
		}
		_ = auditJournal.Close()
	}
	// Terminal edit for the live session card (#2259): classify the exit exactly as the
	// surfacing logic below does, then stop the updater and write the final outcome line. The
	// card handle is nil unless this session queued a Slack root, so finalizeOutcome is a no-op
	// otherwise; it flushes synchronously because an os.Exit follows immediately below.
	guardExitCode := 0
	if runErr != nil {
		if ee, isExit := runErr.(*exec.ExitError); isExit {
			guardExitCode = ee.ExitCode()
		} else {
			guardExitCode = 1
		}
	} else if serr != nil && !errors.Is(serr, http.ErrServerClosed) && !errors.Is(serr, context.Canceled) {
		guardExitCode = 1
	}
	guardSessionCardHandle.finalizeOutcome(guardExitCode, srv.AdjudicationSummary())
	// Faithfully surface the child's exit code first (so `fak guard -- claude -p …`
	// scripts see what the agent returned).
	if runErr != nil {
		if ee, isExit := runErr.(*exec.ExitError); isExit {
			// An abnormal (non-zero) exit is a crash / OOM / terminal upstream error — print
			// the actionable resume note so the operator isn't left with a bare exit code.
			// Suppressed under --quiet (scripted `-p` runs) and skipped on a clean 0 exit.
			if code := ee.ExitCode(); code != 0 && !quiet {
				fmt.Fprint(os.Stderr, formatGuardResumeGuidanceWithRefusals(agentName, code, currentRefusals))
			}
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
