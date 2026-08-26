package main

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

type accountsLaunchState struct {
	params                           launchParams
	registry                         accounts.Registry
	fixes                            acctFixSummary
	command, fakBin, managedCache    string
	home                             accounts.Home
	identity                         accounts.Identity
	ultracode                        bool
	guardCacheArgs                   []string
	environment, declaredCredentials []string
	grant                            launchBrokerGrant
	launchArgv, launchEnv            []string
}

func resolveAccountsLaunch(stdout, stderr io.Writer, state *accountsLaunchState) int {
	p := state.params
	reg, err := loadOrDiscover(p.registryPath, p.homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	// Serve needs disk-derived identity (a seat that can't serve falls forward), so refresh.
	reg = reg.Refresh()
	command := strings.TrimSpace(p.command)
	if command == "" {
		command = "claude"
	}
	if guardAgentBaseName(command) == "codex" {
		codexHomes := codexLaunchAlternatives(reg)
		reg.Homes = append(reg.Homes, codexHomes...)
		if strings.TrimSpace(p.name) == "" || p.rotate {
			fmt.Fprint(stderr, codexExplicitNameGuidance(codexHomes))
			return 2
		}
	}
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
		dec := reg.NextRotationDecision(anchor, hr)
		if !dec.OK {
			printRotationNoCandidate(stderr, "fak accounts launch --rotate", dec)
			printAccountFixSummary(stderr, fixes, "account fixes")
			return 1
		}
		seat := dec.Seat
		if anchor != "" {
			rotNote := fmt.Sprintf("fak accounts launch: rotating off %q -> %q", anchor, seat.Name)
			if seat.Headroom != nil {
				rotNote += fmt.Sprintf(" (headroom=%s)", headroomLabel(*seat.Headroom))
			}
			fmt.Fprintln(stderr, rotNote)
		}
		name = seat.Name
	} else if name == "" {
		picked, ok, fellForward := activeLaunchSeatNameAt(reg, time.Now())
		if !ok {
			fmt.Fprintln(stderr, "fak accounts launch: no --name and no active seat to default to — "+
				"set one with `fak accounts set-default --name <seat>`, or pass --name <seat>")
			printAccountFixSummary(stderr, fixes, "account fixes")
			return 2
		}
		if fellForward {
			active, _ := reg.Role(accounts.RoleActive)
			fmt.Fprintf(stderr, "fak accounts launch: active %s is walled; launching %s with room\n", active.Name, picked)
		}
		name = picked
	}

	// Rehome by default (a tombstoned/unservable seat falls forward to a live one), exactly
	// as `fak accounts resolve` does without --pin — and cooldown-aware (#4675): the walk
	// consults the fleet-shared cooldown store, so a launch skips seats whose account is
	// inside an active usage-limit window instead of stranding the session on a throttled
	// seat. In the all-cooled terminal (non-nil cdEntry) the launch proceeds on the
	// soonest-reset seat with a warning: a seat that resets soon still beats refusing to
	// launch at all.
	home, chain, cdEntry, err := reg.ServeAt(name, loadCooldownStoreFailOpen("fak accounts launch", stderr), time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts launch: %v\n", err)
		return 1
	}
	id := accountsReportHome(stderr, home, chain)
	warnAllCooled(stderr, home, cdEntry)

	state.registry, state.fixes, state.command = reg, fixes, command
	state.home, state.identity = home, id
	return 0
}

func prepareAccountsLaunch(stderr io.Writer, state *accountsLaunchState) int {
	p, command, home := state.params, state.command, state.home
	fixes, id := state.fixes, state.identity
	fakBin, err := os.Executable()
	if err != nil || strings.TrimSpace(fakBin) == "" {
		fakBin = "fak" // fall back to PATH resolution if the binary path can't be read
	}
	warnIfAccountsLaunchStaleBinary(stderr, fakBin, p.useGuard)
	// Managed-cache posture (epic #1844 C6; on-by-default 2026-07-10): resolve the posture from
	// --managed-cache (defaulted to $FAK_MANAGED_CACHE). An UNSET knob now normalizes to `on`
	// (operator policy: best-effort managed cache everywhere), so a launched seat forces the
	// stable-prefix 1h-TTL upgrade instead of inheriting guard's own PASSIVE-on-subscription auto.
	// $FAK_GUARD_API_KEY_ENV still lets an explicit `auto` reach ACTIVE by billing a key. Fail loud
	// on a bad mode. The flags ride the guard argv (guardCacheArgs), so a resumed child keeps them.
	mcMode, mcErr := normalizeManagedCacheMode(p.managedCache)
	if mcErr != nil {
		fmt.Fprintf(stderr, "fak accounts launch: %v\n", mcErr)
		return 2
	}
	// Ultracode posture: auto|on|off, default ON — an instance this launcher starts is born in
	// ultracode rather than needing an operator to type /effort ultracode into it. The interactive
	// launcher carries no work class, so the #5016 work-class table is reached only by an explicit
	// `--ultracode=auto` (which resolves through ultracodeKindUnknown to OFF); `--ultracode=off`
	// is the direct opt-out. Fail loud on a bad posture, exactly as the managed-cache mode does above.
	ultracodeOn, ucErr := resolveUltracodePosture(p.ultracodePosture, ultracodeKindUnknown)
	if ucErr != nil {
		fmt.Fprintf(stderr, "fak accounts launch: %v\n", ucErr)
		return 2
	}
	// A third-party seat's models live in the vendor's namespace, so the first-party default
	// --model (and the fallback chain behind it) would name ids the endpoint does not serve.
	// Defer to the seat's own $ANTHROPIC_MODEL unless the operator named a model explicitly.
	// Resolved before buildLaunchArgv so the argv, the plan summary, and modelFallbackChain all
	// read the same posture.
	if resolved, why, changed := thirdPartySeatModel(home, p.model, p.modelExplicit); changed {
		fmt.Fprintf(stderr, "fak accounts launch: %s\n", why)
		p.model = resolved
	}
	guardCacheArgs := guardCachePostureArgs(mcMode, launchSeatAPIKeyEnv(home))
	argv := buildLaunchArgv(fakBin, launchOpts{
		command:         command,
		useGuard:        p.useGuard,
		skipPermissions: p.skipPerms,
		ultracode:       ultracodeOn,
		model:           p.model,
		guardCacheArgs:  guardCacheArgs,
		codexHome:       codexHomeForCommand(command, home),
		passthrough:     p.passthrough,
	})
	if why, conflict := thirdPartyGuardConflict(home, p.useGuard); conflict {
		fmt.Fprintf(stderr, "fak accounts launch: %s\n", why)
		return 2
	}
	// Validation is enforced HERE as well as at write time: a registry is a plaintext file an
	// operator can hand-edit, so the launch is the last point that can refuse to hand a
	// credential-shaped variable to a child process.
	if err := accounts.ValidateExtraEnv(home.ExtraEnv); err != nil {
		fmt.Fprintf(stderr, "fak accounts launch: seat %q: %v\n", home.Name, err)
		return 2
	}
	env, scrubbed := launchSeatEnv(os.Environ(), home)
	if guardAgentBaseName(command) == "codex" {
		env = launchCodexEnv(env, home.Dir)
	}
	if len(scrubbed) > 0 {
		// Say it out loud: an operator who exported a key expects it to be used, so silently
		// dropping it would be its own surprise.
		fmt.Fprintf(stderr, "fak accounts launch: seat %q has its own endpoint; dropped inherited credential(s) %s "+
			"so the vendor gateway is not sent another account's token (the seat's $%s is kept)\n",
			home.Name, strings.Join(scrubbed, ", "), home.APIKeyEnv)
	}
	// Declare the seat's own credential variable to the spawn broker's secret floor, so the
	// variable the argv already references with --api-key-env actually reaches the child. See
	// seatDeclaredCredentialEnv: without it a TOKEN-named credential is stripped, and the launch
	// authenticates with nothing while looking correctly configured.
	declaredCreds := seatDeclaredCredentialEnv(home)
	grant := launchSpawnBroker(newLaunchBrokerAttemptDeclaring("accounts_launch", guardAgentBaseName(command), argv, envMap(env), home.Dir, declaredCreds))

	printAccountsLaunchPlan(stderr, p, command, home, id, grant, mcMode, ultracodeOn)
	printAccountFixSummary(stderr, fixes, "account fixes")

	if !grant.Allow {
		fmt.Fprintf(stderr, "fak accounts launch: spawn broker denied launch: %s\n", grant.Reason)
		return 1
	}
	launchArgv := grant.Argv
	launchEnv := envSliceFromMap(grant.Env)

	state.params, state.fakBin, state.managedCache = p, fakBin, mcMode
	state.ultracode, state.guardCacheArgs = ultracodeOn, guardCacheArgs
	state.environment, state.declaredCredentials = env, declaredCreds
	state.grant, state.launchArgv, state.launchEnv = grant, launchArgv, launchEnv
	return 0
}

func executeAccountsLaunch(stdout, stderr io.Writer, state *accountsLaunchState) int {
	p, command, home := state.params, state.command, state.home
	fakBin, ultracodeOn := state.fakBin, state.ultracode
	guardCacheArgs, env, declaredCreds := state.guardCacheArgs, state.environment, state.declaredCredentials
	grant, launchArgv, launchEnv := state.grant, state.launchArgv, state.launchEnv
	// #5503 (diagnosability half): never hand a child an argv that references a variable the
	// same launch has already removed from that child's environment. See
	// launchStrippedAPIKeyEnvRefusal — this REFUSES, it never re-admits the stripped variable.
	if refusal := launchStrippedAPIKeyEnvRefusal(launchArgv, grant); refusal != "" {
		fmt.Fprint(stderr, refusal)
		return 2
	}

	if p.dryRun {
		fmt.Fprintln(stderr, "  (dry-run — not launching)")
		// Also echo the launch command to stdout so it is scriptable (eval/wrappers).
		fmt.Fprintln(stdout, strings.Join(launchArgv, " "))
		return 0
	}
	if err := persistAccountsUltracodeActivation(p.homeDir, grant.Metadata.AgentRunID, command, p.ultracodePosture, ultracodeOn); err != nil {
		fmt.Fprintf(stderr, "fak accounts launch: persist pre-spawn Ultracode activation: %v\n", err)
		return 1
	}
	res := runRegisteredAccountsChild(stdout, stderr, launchArgv, launchEnv, "")
	if res.Code == 2 && res.Stderr != "" {
		fmt.Fprintln(stderr, "fak accounts launch:", res.Stderr)
	}
	lastTried := p.model
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
				ultracode:       ultracodeOn,
				model:           fallback,
				guardCacheArgs:  guardCacheArgs,
				passthrough:     p.passthrough,
			})
			fallbackGrant := launchSpawnBroker(newLaunchBrokerAttemptDeclaring("accounts_launch", guardAgentBaseName(command), fallbackArgv, envMap(env), home.Dir, declaredCreds))
			fmt.Fprintf(stderr, "  fallback command  = %s\n", strings.Join(fallbackGrant.SanitizedArgv, " "))
			fmt.Fprintf(stderr, "  fallback agent_run = %s policy_digest=%s broker=%s\n",
				fallbackGrant.Metadata.AgentRunID, fallbackGrant.Metadata.PolicyDigest, fallbackGrant.Reason)
			if !fallbackGrant.Allow {
				fmt.Fprintf(stderr, "fak accounts launch: spawn broker denied fallback launch: %s\n", fallbackGrant.Reason)
				return 1
			}
			fallbackEnv := envSliceFromMap(fallbackGrant.Env)
			if err := persistAccountsUltracodeActivation(p.homeDir, fallbackGrant.Metadata.AgentRunID, command, p.ultracodePosture, ultracodeOn); err != nil {
				fmt.Fprintf(stderr, "fak accounts launch: persist fallback pre-spawn Ultracode activation: %v\n", err)
				return 1
			}
			res = runRegisteredAccountsChild(stdout, stderr, fallbackGrant.Argv, fallbackEnv, res.AttemptID)
			tried = fallback
			if res.Code == 0 {
				break
			}
		}
		lastTried = tried
	}
	// Cooldown gate, applied to the FINAL outcome. If every model we could try —
	// the primary and any fallbacks — still ended on this account's usage/rate cap,
	// the account itself is walled (a model switch cannot fix it), so record a
	// cooldown and drop every seat on the account from the servable pool until the
	// window elapses. A launch that ultimately started (res.Code == 0) records
	// nothing: the account served. Fail-open, best-effort.
	if res.Code != 0 {
		kind := classifyLaunchModelUnavailable(res.Stderr, lastTried)
		recordLaunchCooldown(stderr, home.Identity.AccountKey(), res.Stderr, kind, time.Now())
	}
	return res.Code

}
