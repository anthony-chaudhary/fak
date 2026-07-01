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

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
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
func guardCredCheckWithWindow(credPath string, window time.Duration, now func() time.Time, sleep func(time.Duration)) accounts.CredCheck {
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
	return func(ctx context.Context) (fresh bool, refreshed bool) {
		if _, ok := credExpiresAt(credPath); !ok {
			// No parseable credential on disk at all (missing/torn/no token) — nothing this
			// rung can vouch for or refresh; fail closed to the caller's STALE_CRED refusal.
			return false, false
		}
		if credLive(now()) {
			return true, false // still live: no wait needed, first request goes out immediately
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
	base := strings.ToLower(filepath.Base(agentName))
	base = strings.TrimSuffix(base, ".exe")
	base = strings.TrimSuffix(base, ".cmd")
	if base == "claude" {
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
		next += " and FAK_RESET_SEED_FILE=" + seed
	}
	return fmt.Sprintf("fak guard: managed-context status reset_limit limit=%d reason=%s continuity=%s next_action=%q",
		limit, reason, continuity, next)
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
func runGuardChildAndReport(command []string, injected [][2]string, pinUpstream bool, credPath string, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, agentName, provider string) {
	for {
		child := buildGuardChild(command, injected, pinUpstream)
		runErr := child.Run()
		if next, ok := guardMaybeRecoverAuthCrash(runErr, command, credPath, agentName, quiet, os.Stderr); ok {
			command = next
			continue
		}
		finishGuardChildAndReport(runErr, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName, provider)
		return
	}
}

func runGuardChildSupervisedAndReport(command []string, injected [][2]string, pinUpstream bool, credPath string, restarter *guardBudgetRestarter, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, agentName, provider string) {
	var extraEnv [][2]string
	restarts := 0
	for {
		child := buildGuardChild(command, injected, pinUpstream, extraEnv...)
		wait := make(chan error, 1)
		if err := child.Start(); err != nil {
			finishGuardChildAndReport(err, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName, provider)
			return
		}
		go func() { wait <- child.Wait() }()
		select {
		case runErr := <-wait:
			if next, ok := guardMaybeRecoverAuthCrash(runErr, command, credPath, agentName, quiet, os.Stderr); ok {
				command = next
				continue
			}
			finishGuardChildAndReport(runErr, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName, provider)
			return
		case ev := <-restarter.events:
			if restarter.limit > 0 && restarts >= restarter.limit {
				if restarter.stderr != nil {
					fmt.Fprintln(restarter.stderr, guardRestartLimitStatus(restarter.limit, ev))
				}
				runErr := <-wait
				finishGuardChildAndReport(runErr, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, agentName, provider)
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

// formatGuardResumeGuidance is the concise, actionable note printed when the wrapped agent
// exits abnormally (a non-zero code — a crash, an OOM, or a terminal upstream error). The
// guard process holds no agent conversation state itself — the wrapped tool owns that — so
// "resume" means re-running the same `fak guard -- <command>` with the agent's own
// resume/continue flag. The last line encodes a hard-won recovery: a guarded resume that
// dies IMMEDIATELY with "upstream model error" (a malformed-request rejection that can
// follow a mid-conversation quarantine) usually clears if that one resume is retried WITHOUT
// fak guard, then re-wrapped. Returned as a string (not printed) so it is unit-testable.
func formatGuardResumeGuidance(agentName string, code int) string {
	return fmt.Sprintf(
		"\nfak guard: %s exited abnormally (code %d).\n"+
			"  resume: re-run the same `fak guard -- %s …` — launch the agent with its own resume/continue flag (e.g. `claude --continue`) so it picks the conversation back up.\n"+
			"  this session's decision journal is replayable with `fak audit verify` (path in the audit summary above).\n"+
			"  if a guarded resume dies IMMEDIATELY with \"upstream model error\", retry that one resume WITHOUT fak guard to recover, then re-wrap.\n",
		agentName, code, agentName)
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
	return fmt.Sprintf(
		"fak guard: cache window — recorded %d turn(s) to %s; replay THIS session's realized cache multiplier offline with `fak vcache score` (no flags reads this snapshot), or `fak vcache observe` for the per-sub-concept panels.\n",
		turns, path)
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

func finishGuardChildAndReport(runErr error, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, agentName, provider string) {

	// Tear the gateway down and report what the kernel decided this session.
	cancel()
	serr := <-serveErr
	if !quiet {
		// The wrapped agent (Claude Code) paints a full-screen alternate-screen TUI over this
		// same terminal and, on a crash or an abnormal exit, can tear it down mid-escape-sequence
		// — leaving a dangling SGR color/style or a hidden cursor. The exit summary then renders
		// mis-colored or invisible. Emit a soft reset (SGR reset + show-cursor) onto a clean
		// baseline FIRST, but only when stderr is a real terminal: piping the summary to a file or
		// a JSON capture must stay byte-clean, so a non-TTY stderr gets no escape bytes.
		fmt.Fprint(os.Stderr, guardSummaryResetPrefix(guardFdIsTerminal(int(os.Stderr.Fd()))))
		fmt.Fprintln(os.Stderr)
		sum := srv.AdjudicationSummary()
		kc := srv.KernelCounters()
		fmt.Fprint(os.Stderr, formatAuditSummary(sum, kc))
		fmt.Fprint(os.Stderr, formatAmplification(kc, sum))
		fmt.Fprint(os.Stderr, formatJournalSummary(auditJournal, auditSeq0))
	}
	// Append cache-value observation to ledger (epic #1072, issue #1075).
	stats := cacheobs.Default.Snapshot()
	if stats.Turns > 0 {
		_ = cachevalueledger.Append("guard", agentName, cachevalueledger.DefaultLedgerRel, stats)
	}
	appendObservedCacheSavings("guard", provider, agentName, srv.AdjudicationSummary())
	// Persist this session's OBSERVED provider-cache window so a later `fak vcache score`
	// (a separate process) reports the REALIZED multiplier from real traffic instead of the
	// synthetic-Zipf forecast (#1090). Best-effort: a write failure never fails the session,
	// and an empty window leaves the snapshot empty so the score falls open to the forecast.
	// On a clean write, point the operator at the offline vcache tools that now hold this
	// session's data — otherwise the snapshot is silent and the related vcache items look
	// like they only have a synthetic forecast to score.
	if turns, _ := srv.VCacheTurnsSnapshot(); len(turns) > 0 {
		snapPath := vcachesnapshot.DefaultPath()
		if err := vcachesnapshot.Write(snapPath, turns); err == nil && !quiet {
			fmt.Fprint(os.Stderr, formatVCacheSnapshotPointer(len(turns), snapPath))
		}
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
			// An abnormal (non-zero) exit is a crash / OOM / terminal upstream error — print
			// the actionable resume note so the operator isn't left with a bare exit code.
			// Suppressed under --quiet (scripted `-p` runs) and skipped on a clean 0 exit.
			if code := ee.ExitCode(); code != 0 && !quiet {
				fmt.Fprint(os.Stderr, formatGuardResumeGuidance(agentName, code))
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
