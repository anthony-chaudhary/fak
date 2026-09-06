package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

// (usage-limit cooldown write/resolve wiring lives in accounts_cooldown.go)

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
	model           string   // pass --model <id> to Claude (default Opus 4.8); empty => the seat's own default
	guardCacheArgs  []string // managed-cache posture flags spliced into `fak guard` (--api-key-env / --managed-cache); nil only for an explicit `auto` (guard's own billing-gated default) — the unconfigured default now resolves to on
	codexHome       string   // explicit Codex config home; hard-pinned in guard and child env
	passthrough     []string // extra args appended to the agent command (everything after `--`)
}

// ultracodeSettingsArg is the session-only knob the `f` shortcut injects to put Claude in
// ultracode: xhigh per-message reasoning PLUS dynamic multi-agent workflow orchestration. It is
// NOT a persisted settings.json value (writing it there would make it sticky, and it is not how
// Claude Code models the mode) — it must be handed per-launch via --settings. buildLaunchArgv
// only emits it for Claude, since --settings is Claude-specific.
const ultracodeSettingsArg = `{"ultracode":true}`

// ultracodeWorkKind is the coarse work class `--ultracode=auto` routes on. It is deliberately a
// small launcher-local vocabulary rather than the fleet dispatch tier table (the
// FLEET_TIER_LAUNCH tier→profile resolver), so the account-switcher launcher stays
// self-contained and the two seams can evolve independently.
type ultracodeWorkKind int

const (
	// ultracodeKindUnknown is an untagged launch — the interactive `fak accounts launch` case,
	// which carries no work class at all.
	ultracodeKindUnknown ultracodeWorkKind = iota
	// ultracodeKindRigor is work whose output must be VERIFIED before it is relayed: design,
	// audit, security, benchmark-claims. Ultracode's fan-out and self-verification pay for
	// themselves here.
	ultracodeKindRigor
	// ultracodeKindGrind is mechanical work: hygiene sweeps, doc sync, high-tool-count autonomous
	// loops. Ultracode is pure wall-clock overhead here.
	ultracodeKindGrind
)

// resolveUltracodePosture folds the --ultracode knob and the work class into the single boolean
// buildLaunchArgv needs: whether to emit --settings '{"ultracode":true}'. This is the
// posture→ultracode table (#5016) that replaced a blanket default-on bool:
//
//	posture  work kind  ultracode  why
//	-------  ---------  ---------  -------------------------------------------------------
//	on       (any)      ON         operator forced it; an explicit posture always wins
//	off      (any)      OFF        operator disabled it; an explicit posture always wins
//	auto     rigor      ON         verified-before-relayed work earns the orchestration tax
//	auto     grind      OFF        mechanical work never recovers the wall-clock cost
//	auto     unknown    OFF        conservative: unclassifiable work probably needs no rigor
//
// The FLAG default is `on`, not `auto`: a new instance should already BE in ultracode when it
// starts, so nobody has to type `/effort ultracode` into a fresh session — the posture a launch
// is born with is the one that survives, since a session-only --settings cannot be retrofitted
// onto an already-running agent. `auto` is still the work-class-aware posture and stays one flag
// away (`--ultracode=auto`) for a caller that CAN classify its work; the interactive launcher
// carries no work class, so auto→unknown→OFF remains the lean/fast escape hatch #5016 wanted, and
// `--ultracode=off` suppresses it outright. `true`/`false` stay accepted as on/off aliases so a
// script written against the old bool flag keeps working. An unrecognized posture is a loud error
// rather than a silent default — the same fail-on-bad-mode discipline normalizeManagedCacheMode uses.
func resolveUltracodePosture(posture string, kind ultracodeWorkKind) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(posture)) {
	case "on", "true":
		return true, nil
	case "off", "false":
		return false, nil
	case "", "auto":
		return kind == ultracodeKindRigor, nil
	default:
		return false, fmt.Errorf("invalid --ultracode %q: want auto|on|off", posture)
	}
}

// ultracodePostureWord normalizes a posture for the launch plan readout, so an empty knob still
// reads as the `auto` it resolves to.
func ultracodePostureWord(posture string) string {
	if p := strings.ToLower(strings.TrimSpace(posture)); p != "" {
		return p
	}
	return "auto"
}

// defaultLaunchModel is the Opus 5 model id an account-switched Claude launch pins by
// default. The switcher passes it explicitly via --model so every seat a launch lands on
// starts on the same model regardless of that seat's OWN saved default. `--model ""` opts
// out and lets the seat's saved default stand. Like ultracode it is emitted for Claude only:
// --model is a Claude-specific flag and the id names a Claude model, meaningless to other
// agents. Pinned to Opus 5 as the fleet primary (was Opus 4.8, and Fable 5 before that);
// the PREVIOUS Opus generation is now the first fallback rung, so "default to Opus 5" is
// safe to ship fleet-wide — a CLI build or seat that does not yet know the id degrades
// WITHIN the Opus class instead of dropping a whole tier on every launch.
const defaultLaunchModel = "claude-opus-5"

// defaultLaunchFallbackModel is the default fallback CHAIN (`--fallback-model`, comma-separated)
// tried in order when the default Opus 5 launch is refused before a session starts because the
// model is unavailable — unknown/invalid OR a usage/rate limit (e.g. an Opus weekly cap). The two
// rungs answer the two DIFFERENT walls, in the order they are cheap to rule out:
//
//   - claude-opus-4-8 — the previous Opus generation, for the UNKNOWN-MODEL wall. A CLI build or a
//     seat not yet entitled to Opus 5 refuses the id at startup, and the honest degrade there is
//     the same-class model that is known-good, not a tier change. (It shares the Opus allocation
//     bucket, so it does NOT answer a weekly cap — that is the next rung's job.)
//   - claude-fable-5 — reached for its SEPARATE allocation bucket (a capped Opus window does not
//     cap Fable), NOT because Fable is cheaper: under the repo's canonical taxonomy Fable 5 is the
//     restricted, PRICIEST apex model (internal/modelroute/cost.go prices it ~2x the Opus/frontier
//     baseline; internal/fleetaccounts/apextier.go gates it explicit-only — #3927).
//
// Both rungs are spelled with the VERSIONED id, never a bare alias like `fable`. This same chain
// is what dispatchWorkerFallbackModel hands headless workers as `claude -p --fallback-model`, and
// a bare alias 400-crashes a headless worker — the fleet crash-loop internal/dispatchtick's
// WorkerModel* constants already warn about. ultracode is preserved by reusing the same
// launchOpts. A caller can widen the chain, e.g. `--fallback-model claude-opus-4-8,claude-sonnet-5`.
const defaultLaunchFallbackModel = "claude-opus-4-8,claude-fable-5"

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

// resolveAccountsLaunchSkipPermissions preserves the shared launcher's historical Claude
// posture while making interactive Codex bypass an explicit operator choice. requested is
// the parsed boolean value; explicit reports whether --skip-permissions appeared on argv.
// Low-level and headless callers that already pass launchOpts remain unchanged.
func resolveAccountsLaunchSkipPermissions(command string, requested, explicit bool) bool {
	if explicit {
		return requested
	}
	if guardAgentBaseName(command) == "codex" {
		return false
	}
	return requested
}

// buildLaunchArgv constructs the process argv for an account-switched launch. With useGuard
// the agent runs UNDER `fak guard` (this same binary), so the kernel adjudicates every tool
// call and the prompt-cache/compaction (vCache) layer is on; the agent itself is started with
// its permission-bypass flag only when skipPermissions was resolved on. The flag is resolved PER AGENT
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
	if guardAgentBaseName(o.command) == "codex" && strings.TrimSpace(o.codexHome) != "" {
		argv = append(argv, "--codex-home", o.codexHome)
	}
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
	rotate           bool
	after            string
	useHeadroom      bool   // default true — order the rotation by the live runtime headroom signal
	useGuard         bool   // default true
	skipPerms        bool   // resolved per harness: Claude defaults true; Codex defaults false
	ultracodePosture string // ultracode posture: auto|on|off (default on — resolved by resolveUltracodePosture)
	model            string // default Opus 4.8 — the model a switched Claude launch pins via --model ("" => seat default)
	modelExplicit    bool
	fallbackModel    string // default Fable 5 — comma-separated fallback CHAIN tried when the default Opus 4.8 startup is unavailable
	managedCache     string // managed-cache posture: auto|on|off (default $FAK_MANAGED_CACHE, else on — best-effort; explicit "auto" restores guard's billing-gated auto)
	dryRun           bool   // print the plan, do not exec
	passthrough      []string
	registryPath     string
	homeDir          string
}

// launchRunResult is the exec seam result. Stderr carries a bounded tail only, so the
// fallback classifier can inspect startup failures without retaining a whole agent session.
type launchRunResult struct {
	Code               int
	Stderr             string
	Duration           time.Duration
	RegistrationID     string
	AttemptID          string
	RootRegistrationID string
}

// accountsLaunchRun is the exec seam: it spawns the resolved argv with the seat's
// CLAUDE_CONFIG_DIR in the environment and returns the child's exit code plus a stderr tail.
// A test overrides it to capture the plan without spawning a real agent. Production uses
// execLaunchChild.
var accountsLaunchRun = execLaunchChild

// accountsLaunchStamp/accountsLaunchAssess are freshness seams. A stale launcher is a special
// footgun because the default guard path re-execs this same binary; without a warning, a user
// can keep starting the old guard forever even while origin/main is newer. accountsLaunchAssess
// classifies THIS launcher's build against origin/main by git ANCESTRY (versionskew), so a
// binary that is provably BEHIND (Skewed) or OFF the trunk line (Diverged) is called out
// distinctly — while a developer's dirty/ahead local build stays quiet.
var (
	accountsLaunchStamp  = binstamp.Self
	accountsLaunchAssess = defaultAccountsLaunchAssess
)

// defaultAccountsLaunchAssess compares the running launcher's stamp to origin/main by ancestry.
// It runs git in the process cwd and does NOT fetch — launch must stay fast, and a stamp that is
// a strict ANCESTOR of even a slightly-old local origin/main ref is still a genuine Skewed. When
// origin/main cannot be resolved (fresh clone, no remote) the verdict is the honest Unknown and
// no warning fires.
func defaultAccountsLaunchAssess() versionskew.Assessment {
	return versionskew.AssessStamp(context.Background(), versionskew.RealRunner, "", "origin/main", accountsLaunchStamp())
}

// runAccountsLaunch resolves the seat and builds the guard-wrapped launch
// argv, and execs it under that seat's CLAUDE_CONFIG_DIR. With dryRun it prints the plan and
// returns without launching.
func runAccountsLaunch(stdout, stderr io.Writer, p launchParams) int {
	state := accountsLaunchState{params: p}
	if code := resolveAccountsLaunch(stdout, stderr, &state); code != 0 {
		return code
	}
	if code := prepareAccountsLaunch(stderr, &state); code != 0 {
		return code
	}
	return executeAccountsLaunch(stdout, stderr, &state)
}

// launchSeatAPIKeyEnv resolves the --api-key-env reference an account-switched launch
// fronts `fak guard` with. An API-KEY seat (CredKindAPIKey, #5331) carries its OWN env-var
// reference in the registry — the seat's credential IS that key, so guard must bill it
// (and managed cache resolves ACTIVE on the Anthropic wire) — and it wins over the
// fleet-wide $FAK_GUARD_API_KEY_ENV knob. Every other seat keeps the historical behavior:
// the fleet knob, possibly empty (omit --api-key-env; subscription OAuth bills the seat).
func launchSeatAPIKeyEnv(home accounts.Home) string {
	if home.CredentialKind() == accounts.CredKindAPIKey {
		if env := strings.TrimSpace(home.APIKeyEnv); env != "" {
			return env
		}
	}
	return strings.TrimSpace(os.Getenv(fleetGuardAPIKeyEnvEnv))
}

// launchGuardAPIKeyEnvRef returns the env-var NAME the guard argv's `--api-key-env` references,
// or "" when the launch names none. Only the GUARD half of the argv is scanned — everything
// after the `--` separator belongs to the wrapped agent, and a passthrough flag that happens to
// spell `--api-key-env` is the child's own concern, not this launch's contradiction.
func launchGuardAPIKeyEnvRef(argv []string) string {
	for i, arg := range argv {
		if arg == "--" {
			return ""
		}
		switch {
		case arg == "--api-key-env" || arg == "-api-key-env":
			if i+1 < len(argv) {
				return strings.TrimSpace(argv[i+1])
			}
		case strings.HasPrefix(arg, "--api-key-env="):
			return strings.TrimSpace(strings.TrimPrefix(arg, "--api-key-env="))
		case strings.HasPrefix(arg, "-api-key-env="):
			return strings.TrimSpace(strings.TrimPrefix(arg, "-api-key-env="))
		}
	}
	return ""
}

// launchStrippedAPIKeyEnvRefusal names the one contradiction a launch can build against itself
// (#5503): the argv references `--api-key-env NAME`, and the SAME launch has already removed
// NAME from the environment it is about to hand that child. An api-key seat carries its own
// env-var reference (launchSeatAPIKeyEnv), the shaper splices it into the guard argv, and then
// the always-on #2358 inherited-secret floor — policy.StripInheritedSecrets, applied to every
// brokered spawn by sanitizeLaunchEnv — drops NAME because it is credential-shaped. Both halves
// are working as designed; only their COMPOSITION is broken, and nothing downstream can see it:
// the guard child reads an empty NAME and reports it as "set but that env var is empty — export
// it", which misdiagnoses an operator who did export it, and on the passthrough wires it goes
// unremarked entirely.
//
// The parent is the only place that can tell the difference, because it alone saw the variable
// present BEFORE the floor swept it — that is exactly what grant.Metadata.StrippedSecretEnv
// records (NAMES only, never values). So the refusal fires only on the stripped terminal: a
// variable the operator simply never exported keeps its existing handling (the seat-servability
// gate for an api-key seat, guard's own empty-named-key gate for the fleet knob), and the
// passthrough wires are not newly refused.
//
// This permits nothing. It relaxes no floor, re-admits no variable, and reveals no value — it
// only converts a silent/misattributed downstream failure into a named refusal at the boundary
// that caused it. Returns "" when there is nothing to refuse; otherwise a ready-to-print block.
func launchStrippedAPIKeyEnvRefusal(argv []string, grant launchBrokerGrant) string {
	name := launchGuardAPIKeyEnvRef(argv)
	if name == "" {
		return ""
	}
	if strings.TrimSpace(grant.Env[name]) != "" {
		return "" // the child can read it — nothing contradictory about this launch
	}
	if !slices.Contains(grant.Metadata.StrippedSecretEnv, name) {
		return "" // never present to begin with; not this floor's doing, so do not blame it
	}
	var b strings.Builder
	fmt.Fprintf(&b, "fak accounts launch: refusing to launch — this launch would hand the child "+
		"`--api-key-env %s` while removing %s from that child's environment, so the agent would "+
		"start with no upstream key.\n", name, name)
	fmt.Fprintf(&b, "  cause: the always-on #2358 inherited-secret floor (policy.StripInheritedSecrets, "+
		"applied to every brokered spawn) stripped %s on the way to the child because the variable is "+
		"credential-shaped — its NAME matches a secret marker (TOKEN/SECRET/PASSWORD/CREDENTIAL/COOKIE/…) "+
		"or its VALUE is secret-shaped under a name the floor does not spare. %s IS set in this shell; "+
		"the floor is why the child cannot see it.\n", name, name)
	fmt.Fprintf(&b, "  fix: hold the key in a variable the floor spares and re-point the seat at it — "+
		"`fak accounts add --name <seat> --api-key-env ANTHROPIC_API_KEY` (or OPENAI_API_KEY) — and export "+
		"the key under THAT name. fak records and prints the NAME only, never the key.\n")
	return b.String()
}

// printAccountsLaunchPlan renders the human-readable launch plan summary to stderr: the resolved
// seat, identity, login, the guard/permissions/ultracode/model posture (Claude-only words gated
// off for other agents), the managed-cache word, and the broker-sanitized command + agent_run
// provenance. Pure output — extracted from runAccountsLaunch verbatim.
func printAccountsLaunchPlan(stderr io.Writer, p launchParams, command string, home accounts.Home, id accounts.Identity, grant launchBrokerGrant, mcMode string, ultracodeOn bool) {
	guardWord := "off (--guard=false; launching the agent directly, no kernel/cache hop)"
	if p.useGuard {
		guardWord = "on (fak guard — kernel adjudicates every tool call; prompt-cache/compaction vCache layer on)"
	}
	permWord := command + " keeps its native permission prompts"
	if guardAgentBaseName(command) == "codex" {
		permWord = "Codex native approvals + sandbox (default); fak gates remain active"
	}
	if p.skipPerms {
		if flag := launchSkipPermsFlag(command); flag != "" {
			if guardAgentBaseName(command) == "codex" {
				permWord = fmt.Sprintf("Codex full approval/sandbox bypass explicitly requested (%s passed); fak gates remain active", flag)
			} else {
				permWord = fmt.Sprintf("fak floor is the permission system (%s passed to %s)", flag, command)
			}
		} else {
			permWord = fmt.Sprintf("fak floor is the permission system; %s keeps its own prompts (no known kernel-bypass flag)", command)
		}
	}
	ver := appversion.Current()
	if id := guardShortBuildID(); id != "" {
		ver += " (" + id + ")"
	}
	fmt.Fprintf(stderr, "fak %s · accounts launch — seat %q\n", ver, home.Name)
	if guardAgentBaseName(command) == "codex" {
		fmt.Fprintln(stderr, "  CODEX_HOME        = <account-dir> (child-local; guard hard-pinned)")
	} else {
		fmt.Fprintln(stderr, "  CLAUDE_CONFIG_DIR = <account-dir>")
	}
	if id.Email != "" {
		fmt.Fprintf(stderr, "  identity          = %s\n", id.Email)
	} else if home.CredentialKind() == accounts.CredKindAPIKey && home.APIKeyEnv != "" {
		// An API-key seat (#5331) has no OAuth email; its identity is the key held in the env
		// var the registry references (never the secret itself).
		fmt.Fprintf(stderr, "  identity          = api-key seat ($%s — env-var reference, key never stored)\n", home.APIKeyEnv)
	}
	fmt.Fprintf(stderr, "  login             = %s (can_serve=%t)\n", home.LoginStatus(), home.CanServe())
	// A third-party seat's endpoint and overlay are the two things that make its launch differ
	// from every other seat's, so name both. Keys only, never values: this plan is what an
	// operator pastes into an issue, and an overlay value can be sensitive even when the
	// variable name is not credential-shaped (ANTHROPIC_CUSTOM_HEADERS carries header text).
	if home.ThirdParty() {
		fmt.Fprintf(stderr, "  endpoint          = %s (third-party Anthropic-compatible; base_url)\n", home.BaseURL)
	}
	if keys := home.EnvOverlayKeys(); len(keys) > 0 {
		fmt.Fprintf(stderr, "  seat env overlay  = %s (values not shown)\n", strings.Join(keys, ", "))
	}
	// Name the posture that produced the verdict, not just the verdict: under the default `auto`
	// the operator should be able to see WHY ultracode is on or off for this launch (#5016).
	posture := ultracodePostureWord(p.ultracodePosture)
	ultracodeWord := fmt.Sprintf("off (--ultracode=%s)", posture)
	if ultracodeOn {
		switch guardAgentBaseName(command) {
		case "claude", "claude-code":
			ultracodeWord = fmt.Sprintf(`on (--ultracode=%s; --settings '{"ultracode":true}' — xhigh reasoning + workflow orchestration)`, posture)
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
		// Name the same api-key-env resolution the real guard argv used (an api-key seat's own
		// reference wins over the fleet knob, #5331) so the summary matches the launch.
		fmt.Fprintf(stderr, "  managed-cache     = %s\n", accountsLaunchManagedCacheWord(mcMode, launchSeatAPIKeyEnv(home)))
	}
	fmt.Fprintf(stderr, "  command           = %s\n", strings.Join(grant.SanitizedArgv, " "))
	fmt.Fprintf(stderr, "  agent_run         = %s policy_digest=%s broker=%s\n",
		grant.Metadata.AgentRunID, grant.Metadata.PolicyDigest, grant.Reason)
}

// modelFallbackChain returns the ordered list of Claude model ids to try, in order, when the
// default startup model is unavailable — the "Opus 4.8 -> Fable 5 -> ..." chain. It reads the
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
// launchModelAlias returns the friendly alias for a known model id (and the canonical id for an
// alias), so classifyLaunchModelUnavailable matches a stderr that names the tried model in EITHER
// form — the CLI prints "opus" as often as "claude-opus-4-8". Returns a sentinel that can't occur
// in real stderr for an unmapped id, so the "names a different model" guard stays strict rather
// than matching on an empty substring.
func launchModelAlias(id string) string {
	switch strings.ToLower(strings.TrimSpace(id)) {
	case "claude-opus-4-8", "opus":
		return "opus"
	case "claude-sonnet-5", "sonnet":
		return "sonnet"
	case "claude-haiku-4-5-20251001", "haiku":
		return "haiku"
	case "claude-fable-5", "fable":
		return "fable"
	}
	return "\x00" // no alias known; never a substring of real stderr
}

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
	// A model id surfaces as its friendly alias ("opus", "fable", "sonnet") as often as its
	// canonical form ("claude-opus-4-8"), so accept the alias of whatever id was tried on this hop
	// as naming the same model. A stderr that names a DIFFERENT model than the one we tried is not
	// a signal to fall the chain forward.
	if tried != "" && !strings.Contains(text, tried) && !strings.Contains(text, launchModelAlias(tried)) {
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
	a := accountsLaunchAssess()
	reexecNote := "before launching"
	if useGuard {
		reexecNote = "before launching; otherwise fak guard will re-exec the same stale file"
	}
	switch a.Verdict {
	case versionskew.Skewed:
		fmt.Fprintf(stderr, "fak accounts launch: WARNING: running fak binary %q was built from %s, but origin/main is at %s (provably BEHIND); run `fak self-update` or rebuild/install fak %s.\n",
			fakBin, shortLaunchRev(a.Running), shortLaunchRev(a.TrunkTip), reexecNote)
	case versionskew.Diverged:
		// Off the trunk line entirely: neither an ancestor nor a descendant of origin/main. A
		// fleet meant to converge on trunk should not keep re-launching it.
		fmt.Fprintf(stderr, "fak accounts launch: WARNING: running fak binary %q was built from %s, which is OFF the trunk line (origin/main is at %s); rebuild/install fak from origin/main %s.\n",
			fakBin, shortLaunchRev(a.Running), shortLaunchRev(a.TrunkTip), reexecNote)
	case versionskew.Unstamped:
		// An UNSTAMPED launcher is its own footgun: it cannot attest which commit it is, so
		// staleness is UNVERIFIABLE and a possibly-old binary would otherwise pass silently —
		// while the default guard path re-execs THIS same file (#3306). Distinct from a dev's
		// Dirty/Ahead local build, which stay quiet; only the no-provenance case warns here.
		fmt.Fprintf(stderr, "fak accounts launch: WARNING: running fak binary %q carries NO VCS stamp — it cannot confirm which commit it is, so staleness is UNVERIFIABLE; rebuild in-repo with `go build ./cmd/fak` or run `fak self-update --force` %s.\n",
			fakBin, reexecNote)
	}
	// Dirty (dev build), Ahead (unpushed local build), Fresh, and the honest Unknown stay quiet:
	// none is a stale binary a launch warning should nag about.
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
	name, ok, _ := activeLaunchSeatNameAt(reg, time.Now())
	return name, ok
}

// activeLaunchSeatNameAt treats RoleActive as a preference, not a hard pin. A bare launch
// stays on the active seat while it can serve; when that seat is cooled, tombstoned, or has
// a known login failure, it falls forward through the declared rotation pool. The anchor role
// is deliberately not consulted or changed: rehome anchoring and interactive launch choice are
// separate contracts.
func activeLaunchSeatNameAt(reg accounts.Registry, now time.Time) (name string, ok, fellForward bool) {
	if h, found := reg.Role(accounts.RoleActive); found {
		if launchSeatServable(h, now) {
			return h.Name, true, false
		}
		for _, c := range reg.Homes {
			if !strings.EqualFold(c.Name, h.Name) && launchSeatServable(c, now) {
				return c.Name, true, true
			}
		}
		// Preserve the old diagnostic path when the entire pool is walled: Resolve will name
		// why the active seat cannot launch instead of silently choosing an arbitrary home.
		return h.Name, true, false
	}
	for _, h := range reg.Homes {
		if launchSeatServable(h, now) && strings.EqualFold(h.Name, "default") {
			return h.Name, true, false
		}
	}
	only, n := "", 0
	for _, h := range reg.Homes {
		if launchSeatServable(h, now) {
			only, n = h.Name, n+1
		}
	}
	if n == 1 {
		return only, true, false
	}
	return "", false, false
}

func launchSeatServable(h accounts.Home, _ time.Time) bool {
	return h.Active() && h.EnabledOrDefault()
}

func runRegisteredAccountsChild(stdout, stderr io.Writer, argv, env []string, resumeOf string) launchRunResult {
	runtime, agent := accountsAgentRuntime(argv)
	if !agent {
		return accountsLaunchRun(stdout, stderr, argv, env)
	}
	m := envMapFromStrings(env)
	attempt := strings.TrimSpace(m["FAK_CHILD_ATTEMPT_ID"])
	if attempt == "" {
		attempt = "accounts-" + runtime + "-" + strconv.FormatInt(time.Now().UTC().UnixNano(), 36)
	}
	parent := firstGuardEnv(m, "FAK_REGISTRATION_ID", "FAK_PARENT_REGISTRATION_ID")
	parentAttempt := firstGuardEnv(m, "FAK_ATTEMPT_ID", "FAK_PARENT_ATTEMPT_ID")
	root := firstGuardEnv(m, "FAK_ROOT_REGISTRATION_ID")
	if parent != "" && root == "" {
		root = parent
	}
	rec, err := sessionregistry.New(sessionregistry.NewInput{RegistrationID: m["FAK_CHILD_REGISTRATION_ID"], ParentRegistrationID: parent, ParentAttemptID: parentAttempt, RootRegistrationID: root, RootOutcome: m["FAK_ROOT_OUTCOME"], RootIssue: firstGuardEnv(m, "FAK_ROOT_ISSUE", "DISPATCH_ISSUE"), TaskID: firstGuardEnv(m, "FAK_TASK_ID", "DISPATCH_ISSUE"), GoalID: firstGuardEnv(m, "FAK_GOAL_ID"), AttemptID: attempt, ResumeOfAttemptID: resumeOf, LaunchKind: "external_account_launch", Scope: []string{m["PWD"]}, Lane: firstGuardEnv(m, "FAK_LANE", "DISPATCH_LANE"), LeaseID: m["FAK_LEASE_ID"], Runtime: runtime, SessionID: m["FAK_SESSION_ID"], ThreadID: m["FAK_THREAD_ID"], HostID: firstGuardEnv(m, "COMPUTERNAME", "HOSTNAME")})
	if err != nil {
		return launchRunResult{Code: 2, Stderr: "child registration refused: " + err.Error()}
	}
	store := sessionregistry.Store{Path: accountsRegistryPath(m)}
	if err = store.Register(rec); err != nil {
		return launchRunResult{Code: 2, Stderr: "child registration persist failed (child not started): " + err.Error()}
	}
	m["FAK_SESSION_REGISTRY"] = store.Path
	m["FAK_REGISTRATION_ID"] = rec.RegistrationID
	m["FAK_ATTEMPT_ID"] = rec.AttemptID
	m["FAK_PARENT_REGISTRATION_ID"] = rec.ParentRegistrationID
	m["FAK_PARENT_ATTEMPT_ID"] = rec.ParentAttemptID
	m["FAK_ROOT_REGISTRATION_ID"] = rec.RootRegistrationID
	m["FAK_ROOT_OUTCOME"] = rec.RootOutcome
	m["FAK_ROOT_ISSUE"] = rec.RootIssue
	m["FAK_TASK_ID"] = rec.TaskID
	m["FAK_GOAL_ID"] = rec.GoalID
	res := accountsLaunchRun(stdout, stderr, argv, envSliceFromMap(m))
	res.RegistrationID = rec.RegistrationID
	res.AttemptID = rec.AttemptID
	res.RootRegistrationID = rec.RootRegistrationID
	state, reason := sessionregistry.StateCompleted, ""
	if res.Code != 0 {
		state, reason = sessionregistry.StateFailed, fmt.Sprintf("worker_exit_%d", res.Code)
	}
	_, _ = store.Terminal(rec.RegistrationID, state, reason, m["FAK_WITNESS_REF"], time.Now().UTC())
	return res
}
func accountsAgentRuntime(argv []string) (string, bool) {
	if len(argv) == 0 {
		return "", false
	}
	for _, a := range argv {
		n := strings.ToLower(strings.TrimSuffix(filepath.Base(a), filepath.Ext(a)))
		switch n {
		case "codex", "claude", "opencode":
			return n, true
		}
	}
	return "", false
}
func accountsRegistryPath(env map[string]string) string {
	if p := strings.TrimSpace(env["FAK_SESSION_REGISTRY"]); p != "" {
		return p
	}
	return sessionregistry.DefaultPath()
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
	start := time.Now().UTC()
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(stderr, "fak accounts launch: %v\n", err)
		return launchRunResult{Code: 1, Stderr: errTail.String(), Duration: time.Since(start)}
	}
	if m := envMapFromStrings(env); strings.TrimSpace(m["FAK_REGISTRATION_ID"]) != "" {
		store := sessionregistry.Store{Path: accountsRegistryPath(m)}
		if _, err := store.Start(m["FAK_REGISTRATION_ID"], cmd.Process.Pid, start); err != nil {
			_ = cmd.Process.Kill()
			_, _ = cmd.Process.Wait()
			fmt.Fprintf(stderr, "fak accounts launch: child start read-back failed: %v\n", err)
			return launchRunResult{Code: 2, Stderr: errTail.String(), Duration: time.Since(start), RegistrationID: m["FAK_REGISTRATION_ID"]}
		}
	}
	if err := cmd.Wait(); err != nil {
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
