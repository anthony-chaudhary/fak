package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// `fak accounts` — the durable, identity-true registry of Claude config homes
// (CLAUDE_CONFIG_DIR seats), with tombstone + auto-rehome. It answers two questions a
// directory name alone cannot: WHO is this seat actually logged in as (disk truth, so
// a name that lies is flagged), and WHERE does a retired seat send anything still
// pinned to it (a tombstone's rehome target, followed transitively). See
// internal/accounts for the model.
//
// registry.json is the SINGLE SOURCE OF TRUTH (identity + policy attributes per account); the
// dos roster (~/.claude/accounts.yaml) and the job roster (job/config/claude_accounts.yaml)
// are GENERATED VIEWS of it — `sync` writes them, `check` flags drift, never hand-edit them.
//
// Subcommands:
//
//	fak accounts add <name> [--reserved] [--chrome-profile P] [--no-login --token -]
//	                                   enroll a NEW account end-to-end: isolated-dir login (never
//	                                   ~/.claude), identity probe, twin-check, registry + views
//	fak accounts remove --name <n> [--archive]  tombstone an account in the registry + regenerate views;
//	                                   --archive ALSO renames the dir to .DELETED-<date> + repoints the registry, in one go
//	fak accounts set-role <role> --name <n> point a role (active|anchor) at <n> + regenerate views
//	fak accounts set-default --name <n> alias for `set-role active` (the launch/active seat)
//	fak accounts launch [--name <n>]   start claude UNDER `fak guard` on a seat (the active role by
//	                                   default): cache/vCache ON + the kernel as the permission system
//	                                   (--dangerously-skip-permissions). Claude launches default to Opus 4.8
//	                                   (--model claude-opus-4-8); --model '' uses the seat's own saved default.
//	                                   --guard=false / --skip-permissions=false opt out
//	fak accounts list                  table of every seat: name, lifecycle, LOGIN status, TRUE identity, creds, rehome, flags
//	fak accounts status [--json]       observable login report: closed status, can_serve, warnings, next action
//	fak accounts resolve <name> [--env] the live config dir serving <name>, following a tombstone's rehome
//	fak accounts discover [--write]    emit (or MERGE-and-write) a registry.json from ~/.claude* (disk truth)
//	fak accounts sync                  project the registry into the dos + job roster views AND
//	                                   deep-merge defaults.settings into each account's settings.json
//	                                   (the in-tree replacement for the external csync chore)
//	fak accounts check                 RED (exit 1) if a generated view drifts from the registry
//	fak accounts validate              load the registry and check every invariant (incl. tombstones resolve)
//	fak accounts version               this binary's build + the registry schema/family it supports + verb set
func cmdAccounts(argv []string) { os.Exit(runAccounts(os.Stdout, os.Stderr, argv)) }

func runAccounts(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak accounts <add|remove|set-role|set-default|launch|next|list|status|resolve|pull|discover|sync|check|validate|version|check-twins|gate-write> [flags]")
		return 2
	}
	sub, rest := argv[0], argv[1:]

	fs := flag.NewFlagSet("accounts "+sub, flag.ContinueOnError)
	fs.SetOutput(stderr)
	defHome, _ := os.UserHomeDir()
	regDefault := os.Getenv("FAK_ACCOUNTS_REGISTRY")
	if regDefault == "" && defHome != "" {
		regDefault = filepath.Join(defHome, ".claude-accounts", "registry.json")
	}
	registryPath := fs.String("registry", regDefault, "path to the config-home registry.json")
	homeDir := fs.String("home", defHome, "home dir to discover ~/.claude* under")
	asJSON := fs.Bool("json", false, "emit JSON instead of a table")
	asEnv := fs.Bool("env", false, "(resolve) print CLAUDE_CONFIG_DIR=<dir> for eval/wrappers")
	pin := fs.Bool("pin", false, "(resolve) PIN to the exact seat (strict); default rehomes to a live seat")
	dryRun := fs.Bool("dry-run", false, "(pull) print what would be pulled without copying; (launch) print the launch plan without starting the agent")
	gateDir := fs.String("dir", "", "(gate-write) target config dir to gate a stdin setup-token write against")
	write := fs.Bool("write", false, "(discover) MERGE the disk scan into the registry and write it back (preserving authored policy), instead of emitting to stdout")
	dosView := fs.String("dos-view", firstNonEmpty(os.Getenv("FAK_DOS_ROSTER"), defaultDosView(defHome)), "(sync/check) path to the generated dos roster view (~/.claude/accounts.yaml)")
	jobView := fs.String("job-view", os.Getenv("FAK_JOB_ROSTER"), "(sync/check) path to the generated job roster view; empty skips the job view")
	addName := fs.String("name", "", "(add) roster name for the new account")
	addReserved := fs.Bool("reserved", false, "(add) hold the new account OUT of routine rotation (last-resort fallback)")
	addChrome := fs.String("chrome-profile", "", "(add) Chrome profile provenance for the new account (informational)")
	addNoLogin := fs.Bool("no-login", false, "(add) do NOT run `claude setup-token`; read the token from --token/stdin instead")
	addToken := fs.String("token", "", "(add) the setup-token (sk-ant-oat…); '-' or empty with --no-login reads stdin")
	addSuffix := fs.String("suffix", firstNonEmpty(os.Getenv("FAK_ACCOUNT_SUFFIX"), "-netra"), "(add) config-dir suffix: dir is ~/.claude-<name> when <name> already ends with it, else ~/.claude-<name><suffix>")
	addNoSync := fs.Bool("no-sync", false, "(add) skip regenerating the roster views after adding (just write the registry)")
	rmRehome := fs.String("rehome-to", "", "(remove) live seat to rehome the tombstoned account to (default: the registry's anchor seat)")
	rmReason := fs.String("reason", "", "(remove) tombstone_reason recorded in the registry")
	rmArchive := fs.Bool("archive", false, "(remove) ALSO rename the config dir to <dir>.DELETED-<date> and repoint the registry (name+dir+rehome refs) in one command; refuses the live CLAUDE_CONFIG_DIR seat")
	roleFlag := fs.String("role", "", "(set-role) the role to point at --name (active|anchor); may also be given as the first positional")
	launchGuard := fs.Bool("guard", true, "(launch) wrap the agent in `fak guard` so the kernel adjudicates every tool call and the prompt-cache/compaction (vCache) layer is on; --guard=false launches the agent directly")
	launchSkipPerms := fs.Bool("skip-permissions", true, "(launch) pass --dangerously-skip-permissions to claude so fak's capability floor — not Claude's own prompts — is the permission system; --skip-permissions=false lets Claude prompt")
	launchCommand := fs.String("command", "claude", "(launch) the agent command to start under the resolved seat")
	launchUltracode := fs.Bool("ultracode", true, "(launch) run Claude in ultracode (xhigh reasoning + dynamic multi-agent workflow orchestration) by default, via --settings '{\"ultracode\":true}'; --ultracode=false launches without it. Claude-only; ignored for other agents")
	launchModel := fs.String("model", defaultLaunchModel, "(launch) model id a switched Claude launch pins via --model; defaults to Opus 4.8 ("+defaultLaunchModel+") so every seat starts on it regardless of its own saved default; --model '' launches with the seat's saved default. Claude-only; ignored for other agents")
	rotateFlag := fs.Bool("rotate", false, "(launch) launch the NEXT account in the rotation instead of the active/named seat — the round-robin off a walled account")
	afterSeat := fs.String("after", "", "(next/launch) rotate to the account bucket AFTER this seat (default: the named seat, else the active seat)")
	noHeadroom := fs.Bool("no-headroom", false, "(next/launch --rotate) ignore the live runtime headroom signal and rotate stable-by-name; by default rotation prefers the account with room and sorts walled/capped accounts last")
	// Allow a leading positional (e.g. `resolve <name> --env`) BEFORE flags — Go's flag
	// package otherwise stops parsing at the first non-flag token, silently dropping the
	// flags. Collect leading non-flag tokens, parse the remainder, then rejoin.
	lead := 0
	for lead < len(rest) && !strings.HasPrefix(rest[lead], "-") {
		lead++
	}
	if err := fs.Parse(rest[lead:]); err != nil {
		return 2
	}
	// Defense-in-depth against a view-clobber footgun: the dos-view default is computed
	// from the process home (os.UserHomeDir) at flag-definition time, so a caller that
	// redirects --home to an isolated tree (every accounts test does) would STILL write the
	// dos roster into the REAL ~/.claude/accounts.yaml — the exact way a `remove`/`add` test
	// once overwrote a live operator's switcher roster with a temp-dir seat. When --home is
	// overridden but --dos-view is left at its default, re-derive the dos view under the
	// chosen home so --home alone makes the whole command hermetic. An explicit --dos-view (or
	// FAK_DOS_ROSTER) still wins.
	if !flagSet(fs, "dos-view") && flagSet(fs, "home") && *homeDir != "" {
		*dosView = defaultDosView(*homeDir)
	}
	*registryPath = pathutil.ExpandTilde(*registryPath)
	*homeDir = pathutil.ExpandTilde(*homeDir)
	*gateDir = pathutil.ExpandTilde(*gateDir)
	*dosView = pathutil.ExpandTilde(*dosView)
	*jobView = pathutil.ExpandTilde(*jobView)
	positional := append(append([]string{}, rest[:lead]...), fs.Args()...)

	switch sub {
	case "list":
		reg, err := loadOrDiscover(*registryPath, *homeDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		reg = reg.Refresh()
		if *asJSON {
			stdout.Write(reg.JSON())
			return 0
		}
		printAccountsTable(stdout, reg)
		return 0

	case "status":
		return accountsStatus(stdout, stderr, *registryPath, *homeDir, *asJSON)

	case "resolve":
		return accountsResolve(stdout, stderr, positional, *registryPath, *homeDir, *pin, *asEnv)

	case "next":
		// The live ROTATION READ: print the next eligible account in the round-robin — the
		// next DISTINCT rate-limit bucket after --after (or a leading positional), wrapping.
		// This is what a launcher/shortcut consults to hop off a walled account instead of
		// re-handing the same seat. --env prints CLAUDE_CONFIG_DIR=<dir> for eval/wrappers.
		after := strings.TrimSpace(*afterSeat)
		if after == "" && len(positional) > 0 {
			after = strings.TrimSpace(positional[0])
		}
		return accountsNext(stdout, stderr, *registryPath, *homeDir, after, *asJSON, *asEnv, !*noHeadroom)

	case "pull":
		return accountsPull(stdout, stderr, positional, *registryPath, *homeDir, *dryRun)

	case "discover":
		return accountsDiscover(stdout, stderr, *registryPath, *homeDir, *write)

	case "validate":
		reg, err := accounts.LoadRegistry(*registryPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "ok: %d homes, registry valid (%s)\n", len(reg.Homes), *registryPath)
		return 0

	case "check-twins":
		return accountsCheckTwins(stdout, stderr, *homeDir, *asJSON)

	case "gate-write":
		return accountsGateWrite(stdout, stderr, *gateDir, *homeDir, *asJSON)

	case "add":
		// The end-to-end "enroll a brand-new account" verb: log in to an ISOLATED config dir
		// (never ~/.claude), probe its identity, upsert the canonical registry, seed the
		// account dir's markers, and regenerate the roster views — one command for what was a
		// multi-file, multi-tool runbook.
		return runAccountsAdd(stdout, stderr, addParams{
			name:         *addName,
			reserved:     *addReserved,
			chrome:       *addChrome,
			noLogin:      *addNoLogin,
			token:        *addToken,
			suffix:       *addSuffix,
			noSync:       *addNoSync,
			homeDir:      *homeDir,
			registryPath: *registryPath,
			dosView:      *dosView,
			jobView:      *jobView,
		})

	case "remove":
		// Tombstone an account in the canonical registry and regenerate the views — the
		// single-source inverse of `add`. The account becomes status=tombstoned with a rehome
		// target + audit fields, drops out of the dos view's active rows, and moves to the job
		// view's tombstoned_accounts block, all from one registry edit.
		return runAccountsRemove(stdout, stderr, removeParams{
			name:         *addName,
			rehomeTo:     *rmRehome,
			reason:       *rmReason,
			archive:      *rmArchive,
			registryPath: *registryPath,
			dosView:      *dosView,
			jobView:      *jobView,
			noSync:       *addNoSync,
		})

	case "set-role":
		// Point a well-known role (active|anchor) at --name — the deterministic one-command way
		// to move the launch seat OR the rehome anchor INDEPENDENTLY. The role is the first
		// positional (`set-role active --name x`) or --role. RoleActive is surfaced as
		// active_default in the dos view; RoleAnchor is the Serve fall-forward target.
		role := *roleFlag
		if role == "" && len(positional) > 0 {
			role = positional[0]
		}
		return runAccountsSetRole(stdout, stderr, setRoleParams{
			role:         role,
			name:         *addName,
			registryPath: *registryPath,
			dosView:      *dosView,
			jobView:      *jobView,
			noSync:       *addNoSync,
		})

	case "set-default":
		// Back-compat alias for `set-role active`: the "default active account" a bare launch /
		// the watchdog uses. Kept because that is the word an operator reaches for; it points
		// the active role ONLY, never the rehome anchor — the separation roles exist to provide.
		return runAccountsSetRole(stdout, stderr, setRoleParams{
			role:         accounts.RoleActive,
			name:         *addName,
			registryPath: *registryPath,
			dosView:      *dosView,
			jobView:      *jobView,
			noSync:       *addNoSync,
		})

	case "launch":
		// The account-switcher LAUNCHER: resolve a seat (the active role by default, or
		// --name <seat>, or a leading positional) and start the agent UNDER `fak guard` with
		// that seat's CLAUDE_CONFIG_DIR — cache/vCache ON and the kernel as the permission
		// system by default. Everything after `--` is passed through to the agent.
		seat := strings.TrimSpace(*addName)
		if seat == "" && lead > 0 {
			seat = strings.TrimSpace(rest[0])
		}
		return runAccountsLaunch(stdout, stderr, launchParams{
			name:         seat,
			command:      *launchCommand,
			rotate:       *rotateFlag,
			after:        strings.TrimSpace(*afterSeat),
			useHeadroom:  !*noHeadroom,
			useGuard:     *launchGuard,
			skipPerms:    *launchSkipPerms,
			ultracode:    *launchUltracode,
			model:        strings.TrimSpace(*launchModel),
			dryRun:       *dryRun,
			passthrough:  fs.Args(),
			registryPath: *registryPath,
			homeDir:      *homeDir,
		})

	case "sync":
		// Project the canonical registry into the generated roster views and write them. The
		// registry is the single source of truth; these files are caches of it, never
		// hand-edited.
		wrote, code := syncViews(stdout, stderr, *registryPath, *dosView, *jobView)
		if code != 0 {
			return code
		}
		if wrote == 0 {
			fmt.Fprintln(stderr, "fak accounts: no view targets (set --dos-view/--job-view or FAK_DOS_ROSTER/FAK_JOB_ROSTER)")
			return 2
		}
		return 0

	case "check":
		return accountsCheck(stdout, stderr, *registryPath, *dosView, *jobView)

	case "version":
		return accountsVersion(stdout, *asJSON)

	default:
		fmt.Fprintf(stderr, "fak accounts: unknown subcommand %q (want add|remove|set-role|set-default|launch|next|list|status|resolve|pull|discover|sync|check|validate|version|check-twins|gate-write)\n", sub)
		return 2
	}
}

// accountsVersion prints the tool-version surface. A stale binary is the trap this closes: it
// silently lacks a newer verb and fails with a raw "flag provided but not defined" instead of
// saying it is behind. Printing the build + registry schema/family + verb set makes staleness
// VISIBLE — compare it against source, or `go install …/cmd/fak@latest`.
func accountsVersion(stdout io.Writer, asJSON bool) int {
	verbs := []string{
		"add", "remove", "set-role", "set-default", "launch", "next", "list", "status", "resolve", "pull",
		"discover", "sync", "check", "validate", "version", "check-twins", "gate-write",
	}
	if asJSON {
		stdout.Write(mustJSON(map[string]any{
			"fak":              appversion.Current(),
			"registry_version": accounts.RegistryVersion,
			"registry_family":  accounts.RegistryFamily + "*",
			"verbs":            verbs,
		}))
		fmt.Fprintln(stdout)
		return 0
	}
	fmt.Fprintf(stdout, "fak %s\n", appversion.Current())
	fmt.Fprintf(stdout, "registry schema: %s (family %s*)\n", accounts.RegistryVersion, accounts.RegistryFamily)
	fmt.Fprintf(stdout, "verbs: %s\n", strings.Join(verbs, " "))
	return 0
}

// accountsResolve prints the config dir that serves <name>: rehoming to a live seat by default
// (Serve), or pinning to the exact seat with --pin (Resolve). With --env it prints
// CLAUDE_CONFIG_DIR=<dir> for eval/wrappers, else the bare dir.
// accountsLoadFor resolves the shared `fak accounts <verb> <name>` prologue: it requires a
// positional name (usage on absence), loads-or-discovers the registry, and refreshes it from
// disk (Serve/Resolve/Pull all need disk-derived identity). It returns (name, refreshed
// registry, code, ok): ok=false means the caller should return code (2 on a missing name, 1
// on a registry load error).
func accountsLoadFor(stderr io.Writer, positional []string, usage, registryPath, homeDir string) (string, accounts.Registry, int, bool) {
	if len(positional) == 0 {
		fmt.Fprintln(stderr, usage)
		return "", accounts.Registry{}, 2, false
	}
	reg, err := loadOrDiscover(registryPath, homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return "", accounts.Registry{}, 1, false
	}
	return positional[0], reg.Refresh(), 0, true
}

func accountsResolve(stdout, stderr io.Writer, positional []string, registryPath, homeDir string, pin, asEnv bool) int {
	// Rehome is the DEFAULT (a seat that can't serve falls forward to a live one); --pin is the
	// strict opt-in. The shared prologue refreshes the registry from disk for that identity.
	name, reg, code, ok := accountsLoadFor(stderr, positional, "usage: fak accounts resolve <name> [--env]", registryPath, homeDir)
	if !ok {
		return code
	}
	var home accounts.Home
	var chain []string
	var err error
	if pin {
		home, chain, err = reg.Resolve(name)
	} else {
		home, chain, err = reg.Serve(name)
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	accountsReportHome(stderr, home, chain)
	if asEnv {
		fmt.Fprintf(stdout, "CLAUDE_CONFIG_DIR=%s\n", home.Dir)
	} else {
		fmt.Fprintln(stdout, home.Dir)
	}
	return 0
}

// accountsNext runs the live rotation read and prints the next eligible account — the next
// DISTINCT rate-limit bucket after `after` (wrapping). It is the queryable surface a launcher
// or shell shortcut consults to rotate off a walled seat. With --json it prints the chosen
// RotationSeat; with --env it prints CLAUDE_CONFIG_DIR=<dir> for eval/wrappers; otherwise a
// human one-liner. A pool with nothing to rotate to fails loud (rc 1) with the reason, so a
// caller never silently re-hands the same exhausted account.
func accountsNext(stdout, stderr io.Writer, registryPath, homeDir, after string, asJSON, asEnv, useHeadroom bool) int {
	reg, err := loadOrDiscover(registryPath, homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	reg = reg.Refresh()
	// By default fold in the live runtime headroom signal so the pool is ordered with the
	// account that has room first and walled/capped accounts last, instead of stable-by-name.
	var hr accounts.RotationHeadroom
	if useHeadroom {
		hr = rotationHeadroom(homeDir)
	}
	seat, ok := reg.NextInRotationWithHeadroom(after, hr)
	if !ok {
		plan := reg.RotationPlanWithHeadroom(hr)
		if len(plan.Pool) == 0 {
			fmt.Fprintln(stderr, "fak accounts next: no eligible accounts in rotation "+
				"(every seat is reserved, disabled, tombstoned, or has no live credentials)")
		} else {
			fmt.Fprintf(stderr, "fak accounts next: only one account bucket in rotation (%s) — "+
				"nowhere else to rotate; enroll another with `fak accounts add`\n", plan.Pool[0].Name)
		}
		return 1
	}
	switch {
	case asJSON:
		stdout.Write(mustJSON(seat))
		fmt.Fprintln(stdout)
	case asEnv:
		fmt.Fprintf(stdout, "CLAUDE_CONFIG_DIR=%s\n", seat.Dir)
	default:
		line := "next: " + seat.Name
		if seat.Dir != "" {
			line += "  " + seat.Dir
		}
		if seat.Email != "" {
			line += "  (" + seat.Email + ")"
		}
		line += fmt.Sprintf("  login=%s can_serve=%t", seat.Login, seat.CanServe)
		if seat.Headroom != nil {
			line += fmt.Sprintf("  headroom=%s", headroomLabel(*seat.Headroom))
		}
		fmt.Fprintln(stdout, line)
	}
	return 0
}

// headroomLabel renders a rotation headroom score as a short, honest word for the one-liner.
// The score is a banded offerability tier (see accounts_headroom.go): the SIGN is the tier and
// the fraction is only a within-tier tie-break (soonest-reset / least-loaded), NOT a quota
// percentage — so the label keys off the sign and reads as room/unknown/walled rather than
// leaking a false-precision number.
func headroomLabel(score float64) string {
	switch {
	case score > 0:
		return "room"
	case score < 0:
		return "walled"
	default:
		return "unknown"
	}
}

// accountsStatus emits the first-class login-status report. It is the machine-readable
// sibling of `accounts list`: closed statuses, can_serve, warnings, and next actions live in
// internal/accounts, not in table-rendering guesses.
func accountsStatus(stdout, stderr io.Writer, registryPath, homeDir string, asJSON bool) int {
	reg, err := loadOrDiscover(registryPath, homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	reg = reg.Refresh()
	report := reg.LoginReport()
	if asJSON {
		stdout.Write(mustJSON(report))
		fmt.Fprintln(stdout)
		return 0
	}
	printAccountsStatus(stdout, report)
	return 0
}

// accountsReportHome prints the rehoming-chain notes for a resolved serving home and
// warns when it carries no live credentials, returning the derived identity for any
// further use. Shared by `accounts resolve`/`serve` and `accounts launch`.
func accountsReportHome(stderr io.Writer, home accounts.Home, chain []string) accounts.Identity {
	for i, hop := range chain {
		to := home.Name
		if i+1 < len(chain) {
			to = chain[i+1]
		}
		fmt.Fprintf(stderr, "note: %q can't serve -> rehoming to %q\n", hop, to)
	}
	id := home.Identity
	if st := home.LoginStatus(); st != accounts.LoginReady {
		reason, action := accounts.LoginReasonAction(st, home)
		if action != "" {
			fmt.Fprintf(stderr, "warning: %q (%s) login=%s — %s; %s\n", home.Name, home.Dir, st, reason, action)
		} else {
			fmt.Fprintf(stderr, "warning: %q (%s) login=%s — %s\n", home.Name, home.Dir, st, reason)
		}
	}
	return id
}

// accountsPull copies the credential bundles a name's seat depends on INTO its serving dir,
// following the registry's pull plan. With dryRun it prints the plan without copying.
func accountsPull(stdout, stderr io.Writer, positional []string, registryPath, homeDir string, dryRun bool) int {
	name, reg, code, ok := accountsLoadFor(stderr, positional, "usage: fak accounts pull <name> [--dry-run]", registryPath, homeDir)
	if !ok {
		return code
	}
	plan, err := reg.Plan(name)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	if len(plan.From) == 0 {
		fmt.Fprintf(stdout, "nothing to pull: %q serves directly from %s\n", name, plan.Into.Dir)
		return 0
	}
	for _, bundle := range plan.From {
		if dryRun {
			fmt.Fprintf(stdout, "would pull %s -> %s\n", bundle, plan.Into.Dir)
			continue
		}
		n, err := copyTree(bundle, plan.Into.Dir)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: pull %s: %v\n", bundle, err)
			return 1
		}
		fmt.Fprintf(stdout, "pulled %d files: %s -> %s\n", n, bundle, plan.Into.Dir)
	}
	return 0
}

// accountsDiscover scans the home dir for config homes. With write it MERGES the scan into the
// canonical registry (preserving authored policy) and saves it; otherwise it emits the scanned
// homes as JSON to stdout.
func accountsDiscover(stdout, stderr io.Writer, registryPath, homeDir string, write bool) int {
	if write {
		// Regenerator mode: load the canonical registry (or start empty), MERGE the disk
		// scan in (refresh identities, add new dirs, PRESERVE authored policy fields), and
		// write it back atomically. This is how the registry becomes the single source of
		// truth without a human re-typing identities — it derives them from disk.
		base := accounts.Registry{}
		if _, err := os.Stat(registryPath); err == nil {
			base, err = accounts.LoadRegistry(registryPath)
			if err != nil {
				fmt.Fprintf(stderr, "fak accounts: %v\n", err)
				return 1
			}
		}
		merged, err := base.MergeDiscovered(homeDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		if err := accounts.SaveRegistry(registryPath, merged); err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "wrote %d home(s) to %s\n", len(merged.Homes), registryPath)
		return 0
	}
	homes, err := accounts.Discover(homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	reg := accounts.Registry{Homes: homes}
	stdout.Write(reg.JSON())
	return 0
}

// accountsCheckTwins is the audit gate for Regression A: RED (exit 1) when two config homes
// logged into DIFFERENT accounts share one setup-token fingerprint (the cross-account smear that
// surfaces as "subscription disabled"). Homes that share a token but resolve to ONE account pass.
func accountsCheckTwins(stdout, stderr io.Writer, homeDir string, asJSON bool) int {
	findings, err := accounts.AuditTokenTwins(homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	if asJSON {
		stdout.Write(mustJSON(map[string]any{"clean": len(findings) == 0, "findings": findings}))
		fmt.Fprintln(stdout)
	}
	if len(findings) == 0 {
		if !asJSON {
			fmt.Fprintln(stdout, "ok: no cross-account token-twins — every shared setup token is one account")
		}
		return 0
	}
	if !asJSON {
		for _, f := range findings {
			fmt.Fprintf(stdout, "TOKEN-TWIN: homes [%s] share one setup token but log into %d accounts [%s]\n",
				strings.Join(f.Homes, ", "), len(f.Accounts), strings.Join(f.Accounts, ", "))
		}
		fmt.Fprintf(stderr, "fak accounts: %d cross-account token-twin(s) — a foreign token will surface as "+
			"\"subscription disabled\". Give each account its OWN setup token in its OWN dir.\n", len(findings))
	}
	return 1
}

// accountsGateWrite is the pre-write gate: decide whether writing a setup token (stdin) into
// gateDir is safe BEFORE any flow persists it. Exit 0 = safe; exit 1 = refused (would create a
// cross-account token-twin). The token is read from stdin only, never argv, and is fingerprinted.
func accountsGateWrite(stdout, stderr io.Writer, gateDir, homeDir string, asJSON bool) int {
	if gateDir == "" {
		fmt.Fprintln(stderr, "usage: fak accounts gate-write --dir <config-dir> < token")
		return 2
	}
	tokBytes, _ := io.ReadAll(os.Stdin)
	verdict := accounts.GateTokenWrite(gateDir, string(tokBytes), homeDir)
	if asJSON {
		stdout.Write(mustJSON(verdict))
		fmt.Fprintln(stdout)
	} else if verdict.Allow {
		fmt.Fprintf(stdout, "ok: safe to write into %s (login: %s)\n", gateDir, verdict.DirAccount)
	} else {
		fmt.Fprintf(stderr, "REFUSED (%s): %s\n", verdict.Reason, verdict.Detail)
	}
	if verdict.Allow {
		return 0
	}
	return 1
}

// accountsCheck is the drift detector: RED (exit 1) if any on-disk view differs from a
// freshly-rendered projection of the registry. The ratchet that keeps the generated views from
// silently diverging from the canonical source.
func accountsCheck(stdout, stderr io.Writer, registryPath, dosView, jobView string) int {
	reg, err := accounts.LoadRegistry(registryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	reg = reg.Refresh()
	drift := 0
	for _, t := range viewTargets(dosView, jobView) {
		want, err := reg.RenderView(t.view)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		got, err := os.ReadFile(t.path)
		if err != nil {
			fmt.Fprintf(stdout, "DRIFT %s: cannot read %s (%v)\n", t.view, t.path, err)
			drift++
			continue
		}
		if string(got) != want {
			fmt.Fprintf(stdout, "DRIFT %s: %s differs from registry projection — run `fak accounts sync`\n", t.view, t.path)
			drift++
			continue
		}
		fmt.Fprintf(stdout, "ok %s: %s matches registry\n", t.view, t.path)
	}
	if drift > 0 {
		return 1
	}
	return 0
}

// syncViews projects the canonical registry (at registryPath) into the named roster views and
// writes them atomically, refreshing identities from disk first so emitted emails are current.
// It returns the number of views written and a process exit code (0 on success). Shared by the
// `sync` verb and `add`'s final step so both regenerate views identically.
func syncViews(stdout, stderr io.Writer, registryPath, dosView, jobView string) (int, int) {
	reg, err := accounts.LoadRegistry(registryPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 0, 1
	}
	reg = reg.Refresh()
	wrote := 0
	for _, t := range viewTargets(dosView, jobView) {
		text, err := reg.RenderView(t.view)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return wrote, 1
		}
		if err := writeViewFile(t.path, text); err != nil {
			fmt.Fprintf(stderr, "fak accounts: write %s: %v\n", t.path, err)
			return wrote, 1
		}
		fmt.Fprintf(stdout, "synced %s view -> %s\n", t.view, t.path)
		wrote++
	}
	// Project the registry's per-account settings defaults (defaults.settings) into every active
	// account's own settings.json — the in-tree replacement for the external csync chore, so a
	// `sync` leaves the whole roster's bypass/permission defaults consistent, not just the roster
	// view files. A registry with no defaults.settings block is a clean no-op.
	if code := projectSettingsForHomes(stdout, stderr, reg, reg.Homes); code != 0 {
		return wrote, code
	}
	return wrote, 0
}

// projectSettingsForHomes deep-merges the registry's defaults.settings block into each home's
// own settings.json (via the atomic writeSettingsFile) and prints the per-account report csync
// used to: one "updated"/"ok (no change)" line per acted-on seat and a trailing count. A
// registry with no defaults.settings block prints one note and returns 0 (nothing to project).
// It returns a process exit code (0 on success, 1 on a write failure). Shared by the `sync`
// verb and `add`'s final step so both seed settings.json identically.
func projectSettingsForHomes(stdout, stderr io.Writer, reg accounts.Registry, homes []accounts.Home) int {
	results, ok, err := reg.ProjectSettings(homes, writeSettingsFile)
	if !ok {
		fmt.Fprintln(stdout, "settings: registry has no defaults.settings block — nothing to project")
		return 0
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: project settings: %v\n", err)
		return 1
	}
	changed := 0
	for _, r := range results {
		switch {
		case r.Skipped != "":
			fmt.Fprintf(stdout, "  settings %-24s skipped (%s)\n", r.Name, r.Skipped)
		case r.Changed:
			fmt.Fprintf(stdout, "  settings %-24s updated -> %s\n", r.Name, r.Path)
			changed++
		default:
			fmt.Fprintf(stdout, "  settings %-24s ok (no change)\n", r.Name)
		}
	}
	fmt.Fprintf(stdout, "settings: %d account(s) changed\n", changed)
	return 0
}

// viewTarget pairs a view name with its on-disk path.
type viewTarget struct {
	view accounts.ViewName
	path string
}

// viewTargets returns the view destinations to sync/check: the dos roster always (it has a
// default path), and the job roster only when a path was given (it lives in a separate repo,
// so it is opt-in via --job-view / FAK_JOB_ROSTER).
func viewTargets(dosPath, jobPath string) []viewTarget {
	var out []viewTarget
	if dosPath != "" {
		out = append(out, viewTarget{accounts.ViewDos, dosPath})
	}
	if jobPath != "" {
		out = append(out, viewTarget{accounts.ViewJob, jobPath})
	}
	return out
}

// writeViewFile writes a generated view atomically (temp + rename) so a reader never sees a
// half-written roster.
func writeViewFile(path, text string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".view-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.WriteString(text); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// writeSettingsFile writes an account's settings.json atomically (temp + rename), creating the
// config dir if absent so a brand-new seat's file lands. It is the []byte sibling of
// writeViewFile — same crash-safe shape, a distinct temp prefix — and is the writeFn the
// settings projection is handed.
func writeSettingsFile(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".settings-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// defaultDosView is the dos roster's conventional path (~/.claude/accounts.yaml).
func defaultDosView(home string) string {
	if home == "" {
		return ""
	}
	return filepath.Join(home, ".claude", "accounts.yaml")
}

// mustJSON marshals v to indented JSON for the --json output paths; on the (unreachable
// for these value types) marshal error it returns a JSON error object rather than panic.
func mustJSON(v any) []byte {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return []byte(fmt.Sprintf("{\"error\":%q}", err.Error()))
	}
	return b
}

// loadOrDiscover reads the registry file if present, else falls back to a fresh
// discovery of ~/.claude* so `fak accounts list` works before a registry is authored.
func loadOrDiscover(registryPath, homeDir string) (accounts.Registry, error) {
	if registryPath != "" {
		if _, err := os.Stat(registryPath); err == nil {
			return accounts.LoadRegistry(registryPath)
		}
	}
	homes, err := accounts.Discover(homeDir)
	if err != nil {
		return accounts.Registry{}, err
	}
	return accounts.Registry{Homes: homes}, nil
}

// copyTree merge-copies the file tree rooted at src into dst (overwriting same-named
// files, creating dirs as needed) and returns the number of files copied. It is how a
// rehome PULLS a tombstoned seat's history bundle from the shared store into the live
// seat's config home.
func copyTree(src, dst string) (int, error) {
	count := 0
	err := filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		in, err := os.Open(path)
		if err != nil {
			return err
		}
		defer in.Close()
		out, err := os.Create(target)
		if err != nil {
			return err
		}
		if _, err := io.Copy(out, in); err != nil {
			out.Close()
			return err
		}
		if err := out.Close(); err != nil {
			return err
		}
		count++
		return nil
	})
	return count, err
}

func printAccountsTable(w io.Writer, reg accounts.Registry) {
	// One provenance line above the table: WHICH fak build rendered this and the registry
	// schema it speaks. It is the cheap visibility half of `fak accounts version` — an operator
	// reading a roster sees the tool version inline, so a stale binary is obvious at a glance.
	fmt.Fprintf(w, "# fak %s · registry %s\n", appversion.Current(), accounts.RegistryVersion)
	// Reconcile groups the seats by the account each truly resolves to, so the table can
	// flag a seat that is really a duplicate of another (one rate-limit bucket presented
	// as several) and a seat whose setup token belongs to a different login than its own.
	rec := reg.Reconcile()
	report := reg.LoginReport()
	obsByName := loginObservationsByName(report)
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tSTATUS\tLOGIN\tIDENTITY\tCREDS\tREHOME\tFLAG")
	dupes, twins := 0, 0
	accountSet := map[string]bool{}
	for _, h := range reg.Homes {
		name := h.Name
		if h.Default {
			name += " *"
		}
		status := string(h.Status)
		if status == "" {
			status = "active"
		}
		login := string(h.LoginStatus())
		ident := h.Identity.Email
		exists := h.Identity.Exists
		hasCreds := h.Identity.HasCreds
		if obs, ok := obsByName[h.Name]; ok {
			login = string(obs.Status)
			ident = obs.Email
			exists = obs.Exists
			hasCreds = obs.HasCreds
		}
		if ident == "" {
			ident = "-"
		}
		creds := "-"
		if exists {
			if hasCreds {
				creds = "yes"
			} else {
				creds = "NO"
			}
		}
		rehome := ""
		if h.RehomeTo != "" {
			rehome = "-> " + h.RehomeTo
		}
		// Flags accumulate (a seat can be both a name-lie AND a duplicate): tombstone,
		// name<>identity, dup->canonical, and the token-twin split-identity warning.
		var flags []string
		if !h.Active() {
			flags = append(flags, "TOMBSTONED")
		}
		if h.NameLie() {
			flags = append(flags, "WARN name<>identity")
		}
		if si, ok := rec[h.Name]; ok {
			if si.Account != "" {
				accountSet[si.Account] = true
			}
			if si.Role == accounts.RoleDuplicate {
				dupes++
				flags = append(flags, "dup -> "+si.Canonical)
			}
			if len(si.TokenTwin) > 0 {
				twins++
				flags = append(flags, "token-twin -> "+strings.Join(si.TokenTwin, ","))
			}
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", name, status, login, ident, creds, rehome, strings.Join(flags, "; "))
	}
	tw.Flush()
	printLoginSummary(w, report, "login")
	// A one-line reconcile summary when there is anything to collapse or warn about, so
	// the operator sees "N seats are really M accounts" instead of inferring it per row.
	if dupes > 0 || twins > 0 {
		fmt.Fprintf(w, "reconcile: %d active seat(s) resolve to %d distinct account(s)",
			len(rec), len(accountSet))
		if dupes > 0 {
			fmt.Fprintf(w, "; %d duplicate seat(s) collapse onto their canonical", dupes)
		}
		if twins > 0 {
			fmt.Fprintf(w, "; %d seat(s) carry another login's setup token (token-twin)", twins)
		}
		fmt.Fprintln(w)
	}
}

func printAccountsStatus(w io.Writer, report accounts.LoginReport) {
	fmt.Fprintf(w, "# %s\n", report.Schema)
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tLOGIN\tCAN_SERVE\tACCOUNT\tIDENTITY\tROLES\tNEXT_ACTION\tWARNING")
	for _, obs := range report.Seats {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			obs.Name,
			obs.Status,
			yesNo(obs.CanServe),
			dash(obs.Account),
			dash(obs.Email),
			dash(strings.Join(obs.Roles, ",")),
			dash(obs.NextAction),
			dash(loginWarningsText(obs.Warnings)),
		)
	}
	tw.Flush()
	printLoginSummary(w, report, "summary")
}

func loginObservationsByName(report accounts.LoginReport) map[string]accounts.LoginObservation {
	out := make(map[string]accounts.LoginObservation, len(report.Seats))
	for _, obs := range report.Seats {
		out[obs.Name] = obs
	}
	return out
}

func printLoginSummary(w io.Writer, report accounts.LoginReport, prefix string) {
	fmt.Fprintf(w, "%s: %d/%d can serve; %d distinct account(s)",
		prefix, report.Summary.CanServe, report.Summary.Total, report.Summary.DistinctAccounts)
	for _, part := range sortedLoginStatusParts(report.Summary.ByStatus) {
		fmt.Fprintf(w, "; %s", part)
	}
	if report.Summary.WarningSeats > 0 {
		fmt.Fprintf(w, "; %d warning seat(s)", report.Summary.WarningSeats)
	}
	fmt.Fprintln(w)
}

func sortedLoginStatusParts(by map[string]int) []string {
	keys := make([]string, 0, len(by))
	for k := range by {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, by[k]))
	}
	return parts
}

func loginWarningsText(ws []accounts.LoginWarning) string {
	if len(ws) == 0 {
		return ""
	}
	out := make([]string, len(ws))
	for i, w := range ws {
		out[i] = string(w)
	}
	sort.Strings(out)
	return strings.Join(out, ",")
}

func dash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func yesNo(v bool) string {
	if v {
		return "yes"
	}
	return "no"
}
