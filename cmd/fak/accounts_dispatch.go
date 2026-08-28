package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// accountsRunner owns parsed command state while dispatching one accounts subcommand.
// Keeping that state together lets command adapters stay cohesive without long argument lists.
type accountsRunner struct {
	stdout io.Writer
	stderr io.Writer
	sub    string
	rest   []string
	cmd    accountsCmd
}

func (r accountsRunner) run() int {
	switch r.sub {
	case "list":
		return r.runList()
	case "status":
		return accountsStatus(r.stdout, r.stderr, *r.cmd.registryPath, *r.cmd.homeDir, *r.cmd.asJSON, *r.cmd.probeIdent, *r.cmd.listAll)
	case "cooldown":
		return r.runCooldown()
	case "doctor":
		return r.runDoctor()
	case "resolve":
		return accountsResolve(r.stdout, r.stderr, r.cmd.positional, *r.cmd.addName, *r.cmd.registryPath, *r.cmd.homeDir, *r.cmd.pin, *r.cmd.asEnv)
	case "next":
		return r.runNext()
	case "rotation":
		return r.runRotation()
	case "rehome":
		return r.runRehome()
	case "pull":
		return accountsPull(r.stdout, r.stderr, r.cmd.positional, *r.cmd.addName, *r.cmd.registryPath, *r.cmd.homeDir, *r.cmd.dryRun)
	case "discover":
		return accountsDiscover(r.stdout, r.stderr, *r.cmd.registryPath, *r.cmd.homeDir, *r.cmd.write)
	case "validate":
		return r.runValidate()
	case "check-twins":
		return accountsCheckTwins(r.stdout, r.stderr, *r.cmd.homeDir, *r.cmd.asJSON)
	case "gate-write":
		return accountsGateWrite(r.stdout, r.stderr, *r.cmd.gateDir, *r.cmd.homeDir, *r.cmd.asJSON)
	case "add":
		return r.runAdd()
	case "enroll-current":
		return r.runEnrollCurrent()
	case "remove":
		return r.runRemove()
	case "restore":
		return r.runRestore()
	case "refresh":
		return r.runRefresh()
	case "backup":
		return r.runBackup()
	case "restore-credential":
		return r.runRestoreCredential()
	case "set-role":
		return r.runSetRole()
	case "set-default":
		return r.runSetDefault()
	case "launch":
		return r.runLaunch()
	case "sync":
		return r.runSync()
	case "check":
		return accountsCheck(r.stdout, r.stderr, *r.cmd.registryPath, *r.cmd.dosView, *r.cmd.jobView, *r.cmd.checkDiff)
	case "version":
		return accountsVersion(r.stdout, *r.cmd.asJSON)
	default:
		fmt.Fprintf(r.stderr, "fak accounts: unknown subcommand %q (want add|enroll-current|remove|restore|backup|restore-credential|set-role|set-default|launch|next|rotation|list|status|resolve|pull|discover|sync|check|validate|version|check-twins|gate-write)\n", r.sub)
		return 2
	}
}

func (r accountsRunner) runList() int {
	reg, err := loadOrDiscover(*r.cmd.registryPath, *r.cmd.homeDir)
	if err != nil {
		fmt.Fprintf(r.stderr, "fak accounts: %v\n", err)
		return 1
	}
	reg = appendDiscoveredCodexHomes(reg.Refresh())
	if *r.cmd.asJSON {
		// Emit the per-seat LoginReport roster (schema+summary+seats[]) that the
		// sibling `status --json` path produces, not the raw registry persistence
		// wrapper whose seats hide under .homes with empty scalar fields: a machine
		// consumer that iterates the top level then gets the real seat roster with
		// per-seat can_serve/status/warnings, not one empty-fielded object. (#4593)
		report := loginReportWithCooldown(r.stderr, reg)
		if !*r.cmd.listAll {
			report = report.WithoutTombstoned()
		}
		r.stdout.Write(mustJSON(report))
		fmt.Fprintln(r.stdout)
		return 0
	}
	printAccountsTable(r.stdout, reg, *r.cmd.listAll)
	return 0
}

// runCooldown exposes the fleet-shared usage-limit cooldown store: list accounts currently
// walled off a cap, or clear one that is actually free again.
func (r accountsRunner) runCooldown() int {
	return accountsCooldown(r.stdout, r.stderr, strings.TrimSpace(*r.cmd.cooldownClear), *r.cmd.asJSON)
}

// runDoctor reports one closed recovery action per seat and optionally applies deterministic
// repairs through the same audited path as remove.
func (r accountsRunner) runDoctor() int {
	return accountsDoctor(r.stdout, r.stderr, *r.cmd.registryPath, *r.cmd.dosView, *r.cmd.jobView, *r.cmd.asJSON, *r.cmd.write)
}

// runNext prints the next distinct eligible rate-limit bucket after the requested seat,
// wrapping around the live rotation.
func (r accountsRunner) runNext() int {
	after := strings.TrimSpace(*r.cmd.afterSeat)
	if after == "" && len(r.cmd.positional) > 0 {
		after = strings.TrimSpace(r.cmd.positional[0])
	}
	return accountsNext(r.stdout, r.stderr, *r.cmd.registryPath, *r.cmd.homeDir, after, *r.cmd.asJSON, *r.cmd.asEnv, !*r.cmd.noHeadroom)
}

// runRotation reports the full witnessed rotation decision, including why each seat is in or out.
func (r accountsRunner) runRotation() int {
	return accountsRotation(r.stdout, r.stderr, *r.cmd.registryPath, *r.cmd.homeDir, *r.cmd.asJSON, !*r.cmd.noHeadroom)
}

// runRehome switches a live guard session; registry-side rehoming remains part of remove.
func (r accountsRunner) runRehome() int {
	return runAccountsRehome(r.stdout, r.stderr, *r.cmd.rehomeAddr, *r.cmd.rehomeKey, *r.cmd.rmReason, *r.cmd.asJSON)
}

func (r accountsRunner) runValidate() int {
	reg, err := accounts.LoadRegistry(*r.cmd.registryPath)
	if err != nil {
		fmt.Fprintf(r.stderr, "fak accounts: %v\n", err)
		return 1
	}
	// Keep this human-facing confirmation stable for scripts and operators.
	fmt.Fprintf(r.stdout, "ok: %d homes, registry valid (%s)\n", len(reg.Homes), *r.cmd.registryPath)
	return 0
}

// runAdd enrolls a login in an isolated config directory, records its identity, updates the
// canonical registry, and regenerates roster views.
func (r accountsRunner) runAdd() int {
	p := commonAccountsAddParams(r.cmd)
	p.chrome, p.baseURL, p.extraEnv = *r.cmd.addChrome, *r.cmd.addBaseURL, *r.cmd.addEnv
	p.noLogin, p.token, p.adopt = *r.cmd.addNoLogin, *r.cmd.addToken, *r.cmd.addAdopt
	p.apiKeyEnv, p.probeIdentity, p.noProbeIdentity = *r.cmd.addAPIKeyEnv, *r.cmd.addProbeIdentity, *r.cmd.addNoProbeIdentity
	return runAccountsAdd(r.stdout, r.stderr, p)
}

// runEnrollCurrent promotes the current session login into a rotation seat and always probes
// the credential's live identity rather than trusting potentially stale on-disk metadata.
func (r accountsRunner) runEnrollCurrent() int {
	p := commonAccountsAddParams(r.cmd)
	return runAccountsEnrollCurrent(r.stdout, r.stderr, enrollParams{
		name: p.name, from: p.from, reserved: p.reserved, force: p.force,
		suffix: p.suffix, noSync: p.noSync, noDivorce: p.noDivorce, probeURL: p.probeURL,
		dryRun: p.dryRun, homeDir: p.homeDir, registryPath: p.registryPath, dosView: p.dosView, jobView: p.jobView,
	})
}

// runRemove builds removal policy once, then selects either one seat or every active seat in
// an account bucket so both retirement forms share rehome, archive, audit, and sync behavior.
func (r accountsRunner) runRemove() int {
	rm := removeParams{
		rehomeTo:     *r.cmd.rmRehome,
		reason:       *r.cmd.rmReason,
		archive:      *r.cmd.rmArchive,
		terminal:     *r.cmd.rmTerminal,
		registryPath: *r.cmd.registryPath,
		dosView:      *r.cmd.dosView,
		jobView:      *r.cmd.jobView,
		noSync:       *r.cmd.addNoSync,
	}
	if *r.cmd.rmByAccount != "" {
		rm.byAccount = *r.cmd.rmByAccount
		return runAccountsRemoveByAccount(r.stdout, r.stderr, rm)
	}
	rm.name = *r.cmd.addName
	return runAccountsRemove(r.stdout, r.stderr, rm)
}

// runRestore reverses the reversible half of remove --archive and repairs generated views.
func (r accountsRunner) runRestore() int {
	return runAccountsRestore(r.stdout, r.stderr, restoreParams{
		name:         *r.cmd.addName,
		registryPath: *r.cmd.registryPath,
		dosView:      *r.cmd.dosView,
		jobView:      *r.cmd.jobView,
		noSync:       *r.cmd.addNoSync,
	})
}

// runRefresh proactively rotates OAuth credentials, grading the result from recorded files
// and requiring an explicit acknowledgement before invalidating the current session's family.
func (r accountsRunner) runRefresh() int {
	return runAccountsRefresh(r.stdout, r.stderr, refreshParams{
		name:         *r.cmd.addName,
		timeout:      *r.cmd.refreshTimeout,
		force:        *r.cmd.addForce,
		ackLogout:    *r.cmd.refreshAckLogout,
		registryPath: *r.cmd.registryPath,
		homeDir:      *r.cmd.homeDir,
		asJSON:       *r.cmd.asJSON,
	})
}

// runBackup snapshots credential blobs into the content-addressed home-tree store.
func (r accountsRunner) runBackup() int {
	return runAccountsBackup(r.stdout, r.stderr, backupParams{
		name:         *r.cmd.addName,
		list:         *r.cmd.backupList,
		keep:         *r.cmd.backupKeep,
		registryPath: *r.cmd.registryPath,
		homeDir:      *r.cmd.homeDir,
		asJSON:       *r.cmd.asJSON,
	})
}

// runRestoreCredential restores a selected credential snapshot after first preserving the current blob.
func (r accountsRunner) runRestoreCredential() int {
	return runAccountsRestoreCredential(r.stdout, r.stderr, restoreCredParams{
		name:         *r.cmd.addName,
		at:           *r.cmd.backupAt,
		file:         *r.cmd.backupFile,
		registryPath: *r.cmd.registryPath,
		homeDir:      *r.cmd.homeDir,
		asJSON:       *r.cmd.asJSON,
	})
}

// runSetRole points a well-known role at a seat without coupling active and anchor roles.
func (r accountsRunner) runSetRole() int {
	role := *r.cmd.roleFlag
	if role == "" && len(r.cmd.positional) > 0 {
		role = r.cmd.positional[0]
	}
	return runAccountsSetRole(r.stdout, r.stderr, setRoleParams{
		role:         role,
		name:         *r.cmd.addName,
		registryPath: *r.cmd.registryPath,
		dosView:      *r.cmd.dosView,
		jobView:      *r.cmd.jobView,
		noSync:       *r.cmd.addNoSync,
	})
}

// runSetDefault preserves the back-compatible alias that changes only the active role.
func (r accountsRunner) runSetDefault() int {
	return runAccountsSetRole(r.stdout, r.stderr, setRoleParams{
		role:         accounts.RoleActive,
		name:         *r.cmd.addName,
		registryPath: *r.cmd.registryPath,
		dosView:      *r.cmd.dosView,
		jobView:      *r.cmd.jobView,
		noSync:       *r.cmd.addNoSync,
	})
}

// runLaunch resolves a seat and starts the selected agent, preserving command-specific
// permission defaults and passing arguments after -- through unchanged.
func (r accountsRunner) runLaunch() int {
	seat := strings.TrimSpace(*r.cmd.addName)
	if seat == "" && r.cmd.lead > 0 {
		seat = strings.TrimSpace(r.rest[0])
	}
	skipPerms := resolveAccountsLaunchSkipPermissions(*r.cmd.launchCommand, *r.cmd.launchSkipPerms, flagSet(r.cmd.fs, "skip-permissions"))
	return runAccountsLaunch(r.stdout, r.stderr, launchParams{
		name:             seat,
		command:          *r.cmd.launchCommand,
		rotate:           *r.cmd.rotateFlag,
		after:            strings.TrimSpace(*r.cmd.afterSeat),
		useHeadroom:      !*r.cmd.noHeadroom,
		useGuard:         *r.cmd.launchGuard,
		skipPerms:        skipPerms,
		ultracodePosture: strings.TrimSpace(*r.cmd.launchUltracode),
		model:            strings.TrimSpace(*r.cmd.launchModel),
		modelExplicit:    flagSet(r.cmd.fs, "model"),
		fallbackModel:    strings.TrimSpace(*r.cmd.launchFallbackModel),
		managedCache:     strings.TrimSpace(*r.cmd.launchManagedCache),
		dryRun:           *r.cmd.dryRun,
		passthrough:      r.cmd.fs.Args(),
		registryPath:     *r.cmd.registryPath,
		homeDir:          *r.cmd.homeDir,
	})
}

// runSync projects the canonical registry into configured generated roster views.
func (r accountsRunner) runSync() int {
	wrote, code := syncViews(r.stdout, r.stderr, *r.cmd.registryPath, *r.cmd.dosView, *r.cmd.jobView)
	if code != 0 {
		return code
	}
	if wrote == 0 {
		fmt.Fprintln(r.stderr, "fak accounts: no view targets (set --dos-view/--job-view or FAK_DOS_ROSTER/FAK_JOB_ROSTER)")
		return 2
	}
	return 0
}
