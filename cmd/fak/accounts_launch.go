package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// `fak accounts launch` — the account-switcher LAUNCHER. It resolves a seat (the active
// role by default, or --name <seat>), rehoming a dead/unservable seat forward exactly as
// `resolve` does, then starts the agent UNDER `fak guard` with that seat's
// CLAUDE_CONFIG_DIR — so the launch is, by default:
//
//   - cache/vCache ON — guard's prompt-cache-preserving compaction + the kernel vDSO are
//     on, and guard sources THAT seat's Claude Pro/Max subscription OAuth upstream (it reads
//     CLAUDE_CONFIG_DIR), so each switched account bills its own subscription;
//   - fak permissions, not Claude's prompts — the agent is launched with
//     --dangerously-skip-permissions so Claude does NOT prompt per tool, while every tool
//     call still crosses fak's capability floor first. The kernel is the permission system.
//
// Both defaults are opt-out: --guard=false launches the agent directly (still seat-switched,
// no kernel/cache hop), and --skip-permissions=false lets Claude prompt normally.
//
// This is the durable Go front door the shell shortcuts (`c` / `claude-as <name>`) call,
// replacing a hand-rolled `CLAUDE_CONFIG_DIR=… claude` line with one that defaults to the
// guarded, cache-on, kernel-adjudicated path.

const launchModelFallbackMaxDuration = 30 * time.Second

// launchOpts captures the launch knobs after the seat is resolved.
type launchOpts struct {
	command         string   // agent command to start (default "claude")
	useGuard        bool     // wrap the agent in `fak guard` (kernel adjudication + vCache)
	skipPermissions bool     // pass --dangerously-skip-permissions to the agent
	ultracode       bool     // pass --settings '{"ultracode":true}' to Claude (workflow mode)
	model           string   // pass --model <id> to Claude (default Fable); empty => the seat's own default
	guardCacheArgs  []string // managed-cache posture flags spliced into `fak guard` (--api-key-env / --managed-cache); nil => guard's own auto
	passthrough     []string // extra args appended to the agent command (everything after `--`)
}

// ultracodeSettingsArg is the session-only knob the `f` shortcut injects to put Claude in
// ultracode: xhigh per-message reasoning PLUS dynamic multi-agent workflow orchestration. It is
// NOT a persisted settings.json value (writing it there would make it sticky, and it is not how
// Claude Code models the mode) — it must be handed per-launch via --settings. buildLaunchArgv
// only emits it for Claude, since --settings is Claude-specific.
const ultracodeSettingsArg = `{"ultracode":true}`

// defaultLaunchModel is the Fable 5 model alias an account-switched Claude launch pins by
// default. The switcher passes it explicitly via --model so every seat a launch lands on
// starts on the same model regardless of that seat's OWN saved default. `--model ""` opts
// out and lets the seat's saved default stand. Like ultracode it is emitted for Claude only:
// --model is a Claude-specific flag and the id names a Claude model, meaningless to other
// agents.
const defaultLaunchModel = "fable"

// defaultLaunchFallbackModel is the default fallback CHAIN (`--fallback-model`, comma-separated)
// tried in order when the default Fable 5 launch is refused before a session starts because the
// model is unavailable — unknown/invalid OR a usage/rate limit (e.g. Fable's weekly cap). It
// lands on the explicit Opus 4.8 id; ultracode is preserved by reusing the same launchOpts. A
// caller can widen it, e.g. `--fallback-model claude-opus-4-8,claude-sonnet-5`.
const defaultLaunchFallbackModel = "claude-opus-4-8"

// launchSkipPermsFlag returns the agent-specific flag that hands permission authority to
// fak's capability floor — i.e. suppresses the agent's OWN per-call approval prompts, because
// under `fak guard` the kernel is the permission system. The flag is agent-specific, so a
// hardcoded value is a footgun: Claude Code's `--dangerously-skip-permissions` is an
// "unexpected argument" to Codex, which is why `fak accounts launch --command codex` used to
// fail before it ever started. The mappings match the flags the repo's own non-interactive
// codex dispatch already uses (tools/issue_resolve_dispatch.py): Claude Code takes
// `--dangerously-skip-permissions`; the Codex CLI takes `--dangerously-bypass-approvals-and-sandbox`
// (a root-level global flag, so it composes with the guard `-c` provider overrides
// installGuardCodexConfig prepends). An agent fak doesn't know the flag for returns "" — the
// kernel floor still adjudicates every call under guard, that agent just keeps its own
// prompting rather than being handed a wrong flag. Matching reuses guardAgentBaseName, so a
// path, a Windows launcher suffix, or odd casing still resolves.
func launchSkipPermsFlag(command string) string {
	switch guardAgentBaseName(command) {
	case "claude", "claude-code":
		return "--dangerously-skip-permissions"
	case "codex":
		return "--dangerously-bypass-approvals-and-sandbox"
	default:
		return ""
	}
}

// buildLaunchArgv constructs the process argv for an account-switched launch. With useGuard
// the agent runs UNDER `fak guard` (this same binary), so the kernel adjudicates every tool
// call and the prompt-cache/compaction (vCache) layer is on; the agent itself is started with
// its permission-bypass flag (when skipPermissions) so fak's capability floor — not the
// agent's own prompts — is the permission system. The flag is resolved PER AGENT
// (launchSkipPermsFlag), so a Claude launch gets --dangerously-skip-permissions while a Codex
// launch gets --dangerously-bypass-approvals-and-sandbox; an agent with no known flag simply
// gets none. fakBin is the path to this binary (os.Executable), used only for the guard wrap.
// It is pure (no I/O) so the wiring is unit-tested without spawning anything.
func buildLaunchArgv(fakBin string, o launchOpts) []string {
	agentCmd := []string{o.command}
	if o.skipPermissions {
		if flag := launchSkipPermsFlag(o.command); flag != "" {
			agentCmd = append(agentCmd, flag)
		}
	}
	// Default model is Claude-specific and gated exactly as ultracode is: --model is a
	// Claude flag and the id names a Claude model, so only a Claude launch gets it. An empty model
	// opts out (launch with the seat's own saved default). It is emitted BEFORE --settings and any
	// passthrough, so an explicit `-- --model <x>` a caller adds after `--` still comes later.
	if o.model != "" {
		switch guardAgentBaseName(o.command) {
		case "claude", "claude-code":
			agentCmd = append(agentCmd, "--model", o.model)
		}
	}
	// Ultracode (workflow mode) is Claude-only and session-only: emit --settings for a Claude
	// launch so a fak launch defaults to the same workflow-on posture the `f` shortcut sets.
	// Gated on the agent being Claude exactly as launchSkipPermsFlag gates, since --settings is
	// a Claude-specific flag; other agents get nothing.
	if o.ultracode {
		switch guardAgentBaseName(o.command) {
		case "claude", "claude-code":
			agentCmd = append(agentCmd, "--settings", ultracodeSettingsArg)
		}
	}
	agentCmd = append(agentCmd, o.passthrough...)
	if !o.useGuard {
		return agentCmd
	}
	// `fak guard [posture] -- <agent ...>`: guard binds the in-process gateway, installs the
	// capability floor, and execs the agent with the gateway URL injected into the child only.
	// The managed-cache posture flags (guardCacheArgs) go BEFORE `--` so guard parses them; nil
	// keeps the argv byte-identical to a launch that never configured a posture.
	argv := []string{fakBin, "guard"}
	argv = append(argv, o.guardCacheArgs...)
	argv = append(argv, "--")
	return append(argv, agentCmd...)
}

// launchParams are the resolved inputs to runAccountsLaunch.
type launchParams struct {
	name    string // seat to launch (empty => the active role / a sensible default)
	command string // agent command (default "claude")
	// rotate launches the NEXT account in the rotation instead of the active/named seat —
	// the round-robin that lets an operator hop off a walled account onto a fresh bucket.
	// after is the anchor it rotates OFF of (empty => the named seat, else the active seat).
	rotate        bool
	after         string
	useHeadroom   bool   // default true — order the rotation by the live runtime headroom signal
	useGuard      bool   // default true
	skipPerms     bool   // default true
	ultracode     bool   // default true — put Claude in ultracode (workflow) mode via --settings
	model         string // default Fable — the model a switched Claude launch pins via --model ("" => seat default)
	modelExplicit bool
	fallbackModel string // default Opus 4.8 — comma-separated fallback CHAIN tried when the default Fable startup is unavailable
	managedCache  string // managed-cache posture: auto|on|off (default $FAK_MANAGED_CACHE, else auto)
	dryRun        bool   // print the plan, do not exec
	passthrough   []string
	registryPath  string
	homeDir       string
}

// launchRunResult is the exec seam result. Stderr carries a bounded tail only, so the
// fallback classifier can inspect startup failures without retaining a whole agent session.
type launchRunResult struct {
	Code     int
	Stderr   string
	Duration time.Duration
}

// accountsLaunchRun is the exec seam: it spawns the resolved argv with the seat's
// CLAUDE_CONFIG_DIR in the environment and returns the child's exit code plus a stderr tail.
// A test overrides it to capture the plan without spawning a real agent. Production uses
// execLaunchChild.
var accountsLaunchRun = execLaunchChild

// accountsLaunchStamp/accountsLaunchHeadRev are freshness seams. A stale launcher is a
// special footgun because the default guard path re-execs this same binary; without a warning,
// a user can keep starting the old guard forever even while the checkout is newer.
var (
	accountsLaunchStamp   = binstamp.Self
	accountsLaunchHeadRev = accountsLaunchGitHeadRev
)

// runAccountsLaunch resolves the seat, builds the (guard-wrapped, skip-permissions) launch
// argv, and execs it under that seat's CLAUDE_CONFIG_DIR. With dryRun it prints the plan and
// returns without launching.
func runAccountsLaunch(stdout, stderr io.Writer, p launchParams) int {
	reg, err := loadOrDiscover(p.registryPath, p.homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	// Serve needs disk-derived identity (a seat that can't serve falls forward), so refresh.
	reg = reg.Refresh()
	fixes := accountFixSummary(p.registryPath, reg)

	name := strings.TrimSpace(p.name)
	if p.rotate {
		// Rotate onto the next DISTINCT account bucket after the anchor (an explicit --after,
		// else the named seat, else the active seat), so a walled account is hopped off of
		// rather than re-launched. NextInRotation skips the anchor's own bucket and every
		// reserved/disabled/tombstoned/duplicate seat. By default it also folds in the live
		// runtime headroom signal, so the rotate lands on the account with room and never on a
		// walled/capped one when an account with headroom exists.
		var hr accounts.RotationHeadroom
		if p.useHeadroom {
			hr = rotationHeadroom(p.homeDir)
		}
		anchor := firstNonEmpty(strings.TrimSpace(p.after), name)
		if anchor == "" {
			if picked, ok := activeLaunchSeatName(reg); ok {
				anchor = picked
			}
		}
		seat, ok := reg.NextInRotationWithHeadroom(anchor, hr)
		if !ok {
			plan := reg.RotationPlanWithHeadroom(hr)
			if len(plan.Pool) == 0 {
				fmt.Fprintln(stderr, "fak accounts launch --rotate: no eligible accounts in rotation "+
					"(every seat is reserved, disabled, tombstoned, or has no live credentials)")
			} else {
				fmt.Fprintf(stderr, "fak accounts launch --rotate: only one account bucket in rotation (%s) — "+
					"nowhere else to rotate; enroll another with `fak accounts add`\n", plan.Pool[0].Name)
			}
			printAccountFixSummary(stderr, fixes, "account fixes")
			return 1
		}
		if anchor != "" {
			rotNote := fmt.Sprintf("fak accounts launch: rotating off %q -> %q", anchor, seat.Name)
			if seat.Headroom != nil {
				rotNote += fmt.Sprintf(" (headroom=%s)", headroomLabel(*seat.Headroom))
			}
			fmt.Fprintln(stderr, rotNote)
		}
		name = seat.Name
	} else if name == "" {
		picked, ok := activeLaunchSeatName(reg)
		if !ok {
			fmt.Fprintln(stderr, "fak accounts launch: no --name and no active seat to default to — "+
				"set one with `fak accounts set-default --name <seat>`, or pass --name <seat>")
			printAccountFixSummary(stderr, fixes, "account fixes")
			return 2
		}
		name = picked
	}

	// Rehome by default (a tombstoned/unservable seat falls forward to a live one), exactly
	// as `fak accounts resolve` does without --pin.
	home, chain, err := reg.Serve(name)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts launch: %v\n", err)
		return 1
	}
	id := accountsReportHome(stderr, home, chain)

	command := strings.TrimSpace(p.command)
	if command == "" {
		command = "claude"
	}
	fakBin, err := os.Executable()
	if err != nil || strings.TrimSpace(fakBin) == "" {
		fakBin = "fak" // fall back to PATH resolution if the binary path can't be read
	}
	warnIfAccountsLaunchStaleBinary(stderr, fakBin, p.useGuard)
	// Managed-cache posture (epic #1844 C6): a launched seat's guard session used to inherit
	// guard's own auto default, which stays PASSIVE on a subscription-OAuth seat. Resolve the
	// posture from --managed-cache (defaulted to $FAK_MANAGED_CACHE) plus $FAK_GUARD_API_KEY_ENV
	// so an API-key-billed seat can reach ACTIVE without hand-editing this launcher. Fail loud on
	// a bad mode. The flags ride the guard argv (guardCacheArgs), so a resumed child keeps them.
	mcMode, mcErr := normalizeManagedCacheMode(p.managedCache)
	if mcErr != nil {
		fmt.Fprintf(stderr, "fak accounts launch: %v\n", mcErr)
		return 2
	}
	guardCacheArgs := guardCachePostureArgs(mcMode, os.Getenv(fleetGuardAPIKeyEnvEnv))
	argv := buildLaunchArgv(fakBin, launchOpts{
		command:         command,
		useGuard:        p.useGuard,
		skipPermissions: p.skipPerms,
		ultracode:       p.ultracode,
		model:           p.model,
		guardCacheArgs:  guardCacheArgs,
		passthrough:     p.passthrough,
	})
	env := append(os.Environ(), "CLAUDE_CONFIG_DIR="+home.Dir)
	grant := launchSpawnBroker(newLaunchBrokerAttempt("accounts_launch", guardAgentBaseName(command), argv, envMap(env), home.Dir))

	guardWord := "off (--guard=false; launching the agent directly, no kernel/cache hop)"
	if p.useGuard {
		guardWord = "on (fak guard — kernel adjudicates every tool call; prompt-cache/compaction vCache layer on)"
	}
	permWord := command + " prompts per tool (--skip-permissions=false)"
	if p.skipPerms {
		if flag := launchSkipPermsFlag(command); flag != "" {
			permWord = fmt.Sprintf("fak floor is the permission system (%s passed to %s)", flag, command)
		} else {
			permWord = fmt.Sprintf("fak floor is the permission system; %s keeps its own prompts (no known kernel-bypass flag)", command)
		}
	}
	fmt.Fprintf(stderr, "fak accounts launch — seat %q\n", home.Name)
	fmt.Fprintf(stderr, "  CLAUDE_CONFIG_DIR = <account-dir>\n")
	if id.Email != "" {
		fmt.Fprintf(stderr, "  identity          = %s\n", id.Email)
	}
	fmt.Fprintf(stderr, "  login             = %s (can_serve=%t)\n", home.LoginStatus(), home.CanServe())
	ultracodeWord := "off (--ultracode=false)"
	if p.ultracode {
		switch guardAgentBaseName(command) {
		case "claude", "claude-code":
			ultracodeWord = `on (--settings '{"ultracode":true}' — xhigh reasoning + workflow orchestration)`
		default:
			ultracodeWord = fmt.Sprintf("n/a (%s is not Claude; --settings not applied)", command)
		}
	}
	modelWord := "seat default (--model '')"
	if strings.TrimSpace(p.model) != "" {
		switch guardAgentBaseName(command) {
		case "claude", "claude-code":
			modelWord = p.model
		default:
			modelWord = fmt.Sprintf("n/a (%s is not Claude; --model not applied)", command)
		}
	}
	if chain, ok := modelFallbackChain(command, p); ok {
		modelWord += fmt.Sprintf(" (fallback chain: %s — on unknown-model or usage/rate-limit startup)", strings.Join(chain, " -> "))
	}
	fmt.Fprintf(stderr, "  guard             = %s\n", guardWord)
	fmt.Fprintf(stderr, "  permissions       = %s\n", permWord)
	fmt.Fprintf(stderr, "  ultracode         = %s\n", ultracodeWord)
	fmt.Fprintf(stderr, "  model             = %s\n", modelWord)
	if p.useGuard {
		fmt.Fprintf(stderr, "  managed-cache     = %s\n", accountsLaunchManagedCacheWord(mcMode, os.Getenv(fleetGuardAPIKeyEnvEnv)))
	}
	fmt.Fprintf(stderr, "  command           = %s\n", strings.Join(grant.SanitizedArgv, " "))
	fmt.Fprintf(stderr, "  agent_run         = %s policy_digest=%s broker=%s\n",
		grant.Metadata.AgentRunID, grant.Metadata.PolicyDigest, grant.Reason)
	printAccountFixSummary(stderr, fixes, "account fixes")

	if !grant.Allow {
		fmt.Fprintf(stderr, "fak accounts launch: spawn broker denied launch: %s\n", grant.Reason)
		return 1
	}
	launchArgv := grant.Argv
	launchEnv := envSliceFromMap(grant.Env)

	if p.dryRun {
		fmt.Fprintln(stderr, "  (dry-run — not launching)")
		// Also echo the launch command to stdout so it is scriptable (eval/wrappers).
		fmt.Fprintln(stdout, strings.Join(launchArgv, " "))
		return 0
	}
	res := accountsLaunchRun(stdout, stderr, launchArgv, launchEnv)
	if chain, ok := modelFallbackChain(command, p); ok {
		// Walk the fallback chain: after each unavailable startup, try the next model until one
		// starts (exit 0), the chain is exhausted, or a failure a model switch cannot fix appears
		// (auth wall, a long-running crash). `tried` tracks the model of the current failed launch
		// so the unknown-model gate keys off the right id as we descend.
		tried := p.model
		for _, fallback := range chain {
			if !shouldRetryLaunchWithFallback(res, tried) {
				break
			}
			kind := classifyLaunchModelUnavailable(res.Stderr, tried)
			fmt.Fprintf(stderr, "fak accounts launch: model %q was unavailable at startup (%s); falling back to %q.\n",
				tried, kind, fallback)
			fallbackArgv := buildLaunchArgv(fakBin, launchOpts{
				command:         command,
				useGuard:        p.useGuard,
				skipPermissions: p.skipPerms,
				ultracode:       p.ultracode,
				model:           fallback,
				guardCacheArgs:  guardCacheArgs,
				passthrough:     p.passthrough,
			})
			fallbackGrant := launchSpawnBroker(newLaunchBrokerAttempt("accounts_launch", guardAgentBaseName(command), fallbackArgv, envMap(env), home.Dir))
			fmt.Fprintf(stderr, "  fallback command  = %s\n", strings.Join(fallbackGrant.SanitizedArgv, " "))
			fmt.Fprintf(stderr, "  fallback agent_run = %s policy_digest=%s broker=%s\n",
				fallbackGrant.Metadata.AgentRunID, fallbackGrant.Metadata.PolicyDigest, fallbackGrant.Reason)
			if !fallbackGrant.Allow {
				fmt.Fprintf(stderr, "fak accounts launch: spawn broker denied fallback launch: %s\n", fallbackGrant.Reason)
				return 1
			}
			res = accountsLaunchRun(stdout, stderr, fallbackGrant.Argv, envSliceFromMap(fallbackGrant.Env))
			tried = fallback
			if res.Code == 0 {
				break
			}
		}
	}
	return res.Code
}

// modelFallbackChain returns the ordered list of Claude model ids to try, in order, when the
// default startup model is unavailable — the "Fable 5 -> Opus 4.8 -> ..." chain. It reads the
// (comma-separated) --fallback-model list, dropping blanks, the primary model itself, and any
// duplicate so the chain never re-launches a model already attempted. ok is false when auto
// fallback does not apply at all: a non-Claude agent (the id is a Claude model, meaningless
// elsewhere), an explicit --model (the operator named a specific model — respect it, don't
// second-guess), an empty chain, or a passthrough `-- --model x` override.
func modelFallbackChain(command string, p launchParams) ([]string, bool) {
	switch guardAgentBaseName(command) {
	case "claude", "claude-code":
	default:
		return nil, false
	}
	primary := strings.TrimSpace(p.model)
	if primary == "" {
		return nil, false
	}
	if p.modelExplicit || !strings.EqualFold(primary, defaultLaunchModel) {
		return nil, false
	}
	if passthroughOverridesClaudeModel(p.passthrough) {
		return nil, false
	}
	seen := map[string]bool{strings.ToLower(primary): true}
	var chain []string
	for _, m := range strings.Split(p.fallbackModel, ",") {
		m = strings.TrimSpace(m)
		if m == "" {
			continue
		}
		key := strings.ToLower(m)
		if seen[key] {
			continue
		}
		seen[key] = true
		chain = append(chain, m)
	}
	if len(chain) == 0 {
		return nil, false
	}
	return chain, true
}

func passthroughOverridesClaudeModel(args []string) bool {
	for i, a := range args {
		if a == "--model" && i+1 < len(args) {
			return true
		}
		if strings.HasPrefix(a, "--model=") {
			return true
		}
	}
	return false
}

func shouldRetryLaunchWithFallback(res launchRunResult, tried string) bool {
	if res.Code == 0 {
		return false
	}
	// Only a FAST failure is a startup refusal. A non-zero exit after a long session is the
	// user quitting or a mid-run crash, not "this model can't start" — never burn the chain on it.
	if res.Duration > launchModelFallbackMaxDuration {
		return false
	}
	return classifyLaunchModelUnavailable(res.Stderr, tried) != launchModelAvailable
}

// launchModelUnavailKind is the closed set of startup-failure reasons that a fallback to a
// DIFFERENT model can address. It is the launch-time (stderr-text) sibling of the gateway's
// richer, header-aware upstream taxonomy (internal/agent/upstream_remedy.go's UpstreamRemedy);
// unifying the two vocabularies is tracked as a follow-on so both layers name a limit the same
// way. Auth/login walls and plain crashes are deliberately NOT in this set: a model switch on a
// walled account still fails, so they map to launchModelAvailable and never trigger a fallback.
type launchModelUnavailKind int

const (
	launchModelAvailable  launchModelUnavailKind = iota // no model-unavailability signal in the text
	launchModelUnknown                                  // the model id was refused: unknown / invalid / not entitled
	launchModelUsageLimit                               // a usage / weekly / session cap — a different model has its own bucket
	launchModelRateLimit                                // a transient 429 / overload — another model may be clear right now
)

func (k launchModelUnavailKind) String() string {
	switch k {
	case launchModelUnknown:
		return "unknown-model"
	case launchModelUsageLimit:
		return "usage-limit"
	case launchModelRateLimit:
		return "rate-limit"
	default:
		return "available"
	}
}

// launchModelUsageLimitSignals name a self-recovering USAGE/OVERAGE cap the tried model hit
// (session/weekly/usage limits, or a reset window). This is the class the old unknown-model-only
// detector MISSED — the "Fable 5 hit its weekly limit, fall back to Opus" the switcher exists for.
// Kept aligned with the account-side taxonomy in internal/fleetaccounts/authsignals.go so a limit
// reads the same wherever fak classifies one.
var launchModelUsageLimitSignals = []string{
	"usage limit", "weekly limit", "session limit", "usage cap",
	"5-hour limit", "5 hour limit", "limit reached", "limit exceeded",
	"quota exceeded", "out of usage", "overage",
	"resets at", "resets in", "try again at", "try again in", "try again after",
	"/usage-credits",
}

// launchModelRateLimitSignals name a TRANSIENT throttle/overload of the tried model — a
// different model (a separate capacity pool) may serve right now.
var launchModelRateLimitSignals = []string{
	"rate limit", "rate_limit", "too many requests", "429",
	"overloaded", "server is overloaded",
}

// launchModelUnknownSignals name an UNKNOWN / INVALID / UNENTITLED model refusal.
var launchModelUnknownSignals = []string{
	"not available", "unavailable", "not found", "does not exist",
	"unknown model", "invalid model", "unsupported model", "model_not_found",
	"no access to model", "not have access", "not entitled",
}

// classifyLaunchModelUnavailable inspects an agent's startup stderr and decides whether the model
// just tried is unavailable in a way a fallback to a DIFFERENT model can fix — and of what kind.
// `tried` is the model id handed to the failed launch. Usage/rate limits are matched FIRST and
// WITHOUT a model-name gate: a weekly/usage cap is account- and bucket-scoped, so its text rarely
// names the model id, and the whole point of the fallback is to reach a model with a separate
// allocation. An unknown/invalid MODEL refusal, by contrast, must name the model dimension and
// must not be about some OTHER model than the one we tried (so `opus is not available` while we
// launched fable does not fire a fable fallback). Everything else — including auth/login walls and
// ordinary crashes — returns launchModelAvailable so the chain is never burned on a failure a
// model switch cannot fix.
func classifyLaunchModelUnavailable(stderr, tried string) launchModelUnavailKind {
	text := strings.ToLower(stderr)
	tried = strings.ToLower(strings.TrimSpace(tried))
	for _, sig := range launchModelUsageLimitSignals {
		if strings.Contains(text, sig) {
			return launchModelUsageLimit
		}
	}
	for _, sig := range launchModelRateLimitSignals {
		if strings.Contains(text, sig) {
			return launchModelRateLimit
		}
	}
	if !strings.Contains(text, "model") {
		return launchModelAvailable
	}
	// The default id surfaces as the friendly alias "fable" as often as its canonical form, so
	// accept either for the first hop; a later hop (opus, sonnet, ...) matches on its own id.
	if tried != "" && !strings.Contains(text, tried) && !strings.Contains(text, "fable") {
		return launchModelAvailable
	}
	for _, sig := range launchModelUnknownSignals {
		if strings.Contains(text, sig) {
			return launchModelUnknown
		}
	}
	return launchModelAvailable
}

// launchModelUnavailable is the boolean form retained for callers/tests that only need
// "is this a model-unavailability the chain should act on?".
func launchModelUnavailable(stderr, tried string) bool {
	return classifyLaunchModelUnavailable(stderr, tried) != launchModelAvailable
}

func warnIfAccountsLaunchStaleBinary(stderr io.Writer, fakBin string, useGuard bool) {
	stamp := accountsLaunchStamp()
	headRev := accountsLaunchHeadRev()
	if binstamp.Compare(stamp, headRev) != binstamp.Stale {
		return
	}
	reexecNote := "before launching"
	if useGuard {
		reexecNote = "before launching; otherwise fak guard will re-exec the same stale file"
	}
	fmt.Fprintf(stderr, "fak accounts launch: WARNING: running fak binary %q was built from %s, but this checkout is at %s; run `fak self-update` or rebuild/install fak %s.\n",
		fakBin, shortLaunchRev(stamp.Revision), shortLaunchRev(headRev), reexecNote)
}

func accountsLaunchGitHeadRev() string {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func shortLaunchRev(rev string) string {
	rev = strings.TrimSpace(rev)
	if len(rev) > 12 {
		return rev[:12]
	}
	if rev == "" {
		return "(unknown)"
	}
	return rev
}

// activeLaunchSeatName picks the seat a bare `fak accounts launch` (no --name) starts: the
// active-role seat if set, else a seat literally named "default" (the bare ~/.claude home a
// discovered registry surfaces), else the sole active seat when there is exactly one. ok is
// false when none of those uniquely identify a seat, so the caller can fail loud with a hint
// instead of guessing.
func activeLaunchSeatName(reg accounts.Registry) (string, bool) {
	if h, ok := reg.Role(accounts.RoleActive); ok {
		return h.Name, true
	}
	for _, h := range reg.Homes {
		if h.Active() && strings.EqualFold(h.Name, "default") {
			return h.Name, true
		}
	}
	only, n := "", 0
	for _, h := range reg.Homes {
		if h.Active() {
			only, n = h.Name, n+1
		}
	}
	if n == 1 {
		return only, true
	}
	return "", false
}

// execLaunchChild spawns argv[0] with argv[1:] under env, wiring the child to the real
// terminal (an interactive agent owns stdin/stdout/stderr), and returns its exit code.
// A non-exec error (binary not found, etc.) is surfaced and mapped to 1.
func execLaunchChild(_, stderr io.Writer, argv, env []string) launchRunResult {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak accounts launch: empty command")
		return launchRunResult{Code: 2}
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = env
	var errTail cappedBuffer
	errTail.max = 64 << 10
	cmd.Stdin, cmd.Stdout = os.Stdin, os.Stdout
	cmd.Stderr = io.MultiWriter(os.Stderr, &errTail)
	start := time.Now()
	if err := cmd.Run(); err != nil {
		dur := time.Since(start)
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return launchRunResult{Code: ee.ExitCode(), Stderr: errTail.String(), Duration: dur}
		}
		fmt.Fprintf(stderr, "fak accounts launch: %v\n", err)
		return launchRunResult{Code: 1, Stderr: errTail.String(), Duration: dur}
	}
	return launchRunResult{Code: 0, Stderr: errTail.String(), Duration: time.Since(start)}
}

type cappedBuffer struct {
	bytes.Buffer
	max int
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	if b.max <= 0 {
		return n, nil
	}
	if len(p) >= b.max {
		b.Buffer.Reset()
		_, _ = b.Buffer.Write(p[len(p)-b.max:])
		return n, nil
	}
	_, _ = b.Buffer.Write(p)
	if b.Buffer.Len() > b.max {
		data := append([]byte(nil), b.Buffer.Bytes()[b.Buffer.Len()-b.max:]...)
		b.Buffer.Reset()
		_, _ = b.Buffer.Write(data)
	}
	return n, nil
}
