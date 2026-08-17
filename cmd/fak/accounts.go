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
	"time"

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
//	fak accounts remove --name <n> [--archive]  tombstone ONE seat in the registry + regenerate views;
//	                                   --archive ALSO renames the dir to .DELETED-<date> + repoints the registry, in one go
//	                                   --by-account <email|uuid|seat> retires the WHOLE account instead: EVERY
//	                                   active seat resolving to that bucket, under one --rehome-to + --reason,
//	                                   so a duplicate seat can't leave the account live (#4669)
//	fak accounts restore --name <n>    reverse `remove --archive`: rename .DELETED dir back,
//	                                   clear tombstone fields, repair rehome refs, resync views
//	fak accounts set-role <role> --name <n> point a role (active|anchor) at <n> + regenerate views
//	fak accounts set-default --name <n> alias for `set-role active` (the launch/active seat)
//	fak accounts launch [--name <n>]   start claude UNDER `fak guard` on a seat (the active role by
//	                                   default): cache/vCache ON + the kernel as the permission system
//	                                   (--dangerously-skip-permissions). Claude launches default to Opus 4.8
//	                                   (--model claude-opus-4-8) with a startup fallback to Fable 5;
//	                                   --model '' uses the seat's own saved default.
//	                                   --guard=false / --skip-permissions=false opt out
//	fak accounts list                  table of every seat: name, lifecycle, LOGIN status, TRUE identity, creds, rehome, flags
//	fak accounts status [--json]       observable login report: closed status, can_serve, warnings, next action
//	fak accounts doctor [--json] [--write] fold every seat into ONE closed recovery action (none|relogin|
//	                                   wait_reset|top_up|prune|hydrate|enable_or_remove|dedupe) with the exact command;
//	                                   --write applies the deterministic repairs (tombstone+rehome a seat whose
//	                                   config dir vanished; hydrate a canonical home from a ready same-account peer).
//	                                   Exit 1 while actions remain, 0 when clean
//	fak accounts rotation [--json]     the FULL witnessed rotation decision: the pool in launch order
//	                                   (with headroom tiers) AND every excluded seat with the reason it
//	                                   is out (duplicate/reserved/disabled/tombstoned/unservable), plus
//	                                   a registry-drift check against disk truth
//	fak accounts rehome [--addr URL]   OPERATOR seat switch on a LIVE `fak guard` session: force it onto
//	                                   the next available account NOW (POST /v1/fak/account/rehome — the
//	                                   on-demand form of the 403 account failover). Nothing in the
//	                                   registry changes; distinct from a tombstone's rehome target
//	fak accounts resolve <name> [--env] the live config dir serving <name>, following a tombstone's rehome
//	fak accounts discover [--write]    emit (or MERGE-and-write) a registry.json from ~/.claude* (disk truth)
//	fak accounts sync                  project the registry into the dos + job roster views AND
//	                                   deep-merge defaults.settings into each account's settings.json
//	                                   (the in-tree replacement for the external csync chore)
//	fak accounts refresh [--name <n>]  proactively rotate seats' OAuth credentials so an IDLE seat never
//	                                   decays into a human-/login-only state, and split a SHARED token family
//	                                   (what copying a credential leaves behind) apart on demand. Graded from
//	                                   disk — recorded expiry + refresh-token family fingerprint — never the
//	                                   spawn's exit code. Exit 1 while any seat is stale/hollow/blocked.
//	                                   A seat sharing its family with THIS session's config dir is REFUSED
//	                                   without --yes-log-me-out (rotating it ends this session's login)
//	fak accounts check                 RED (exit 1) if a generated view drifts from the registry
//	fak accounts validate              load the registry and check every invariant (incl. tombstones resolve)
//	fak accounts version               this binary's build + the registry schema/family it supports + verb set
func cmdAccounts(argv []string) { os.Exit(runAccounts(os.Stdout, os.Stderr, argv)) }

// accountsCmd carries the parsed `fak accounts` flag set plus the leading-positional split so
// runAccounts can dispatch on it without owning the full flag-definition block.
type accountsCmd struct {
	fs *flag.FlagSet

	registryPath, homeDir, gateDir, dosView, jobView *string
	asJSON, listAll, asEnv, pin, dryRun, write       *bool
	checkDiff                                        *bool

	addName, addChrome, addToken, addSuffix, addFrom           *string
	addAPIKeyEnv, addBaseURL                                   *string
	addEnv                                                     *repeatedString
	addReserved, addNoLogin, addNoSync, addAdopt               *bool
	addForce, addProbeIdentity, addNoProbeIdentity, probeIdent *bool
	addNoDivorce                                               *bool

	refreshTimeout   *time.Duration
	refreshAckLogout *bool

	rmRehome, rmReason, rehomeAddr, rehomeKey, rmByAccount *string
	rmArchive, rmTerminal                                  *bool

	roleFlag, launchCommand, launchModel, launchFallbackModel, launchManagedCache, launchUltracode, afterSeat, cooldownClear *string
	launchGuard, launchSkipPerms, rotateFlag, noHeadroom                                                                     *bool

	backupAt, backupFile *string
	backupKeep           *int
	backupList           *bool

	positional []string
	lead       int
}

// parseAccountsCmd builds the accounts flag set, parses argv (tolerating a leading positional
// before flags), applies the --home/--dos-view hermeticity defense and tilde expansion, and
// returns the parsed command. code is non-zero (2) only on a flag parse error.
func parseAccountsCmd(stderr io.Writer, sub string, rest []string) (accountsCmd, int) {
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
	listAll := fs.Bool("all", false, "(list/status) include tombstoned seats in the table; by default they are hidden behind the summary's tombstoned=N count and a one-line note")
	asEnv := fs.Bool("env", false, "(resolve) print CLAUDE_CONFIG_DIR=<dir> for eval/wrappers")
	pin := fs.Bool("pin", false, "(resolve) PIN to the exact seat (strict); default rehomes to a live seat")
	dryRun := fs.Bool("dry-run", false, "(pull) print what would be pulled without copying; (launch) print the launch plan without starting the agent; (add/enroll-current) print the enrollment plan without creating the dir, copying credentials, probing, or writing the registry")
	gateDir := fs.String("dir", "", "(gate-write) target config dir to gate a stdin setup-token write against")
	write := fs.Bool("write", false, "(discover) MERGE the disk scan into the registry and write it back (preserving authored policy), instead of emitting to stdout; (doctor) APPLY the auto-fixable repairs instead of only reporting them")
	dosView := fs.String("dos-view", firstNonEmpty(os.Getenv("FAK_DOS_ROSTER"), defaultDosView(defHome)), "(sync/check) path to the generated dos roster view (~/.claude/accounts.yaml)")
	jobView := fs.String("job-view", os.Getenv("FAK_JOB_ROSTER"), "(sync/check) path to the generated job roster view; empty skips the job view")
	checkDiff := fs.Bool("diff", false, "(check) show a line-level diff of each drifting view: `- ` lines are on-disk lines `sync` would overwrite, `+ ` lines are projection lines it would add back")
	addName := fs.String("name", "", "(add) roster name for the new account")
	addReserved := fs.Bool("reserved", false, "(add) hold the new account OUT of routine rotation (last-resort fallback)")
	addChrome := fs.String("chrome-profile", "", "(add) Chrome profile provenance for the new account (informational)")
	addBaseURL := fs.String("base-url", "", "(add) point the seat at a THIRD-PARTY Anthropic-compatible endpoint (a vendor gateway speaking the Messages API) instead of first-party api.anthropic.com. Such a seat must be launched with --guard=false, since guard fronts the child with its own base URL and credential")
	var addEnv repeatedString
	// NOT --env: that name is already the `resolve` verb's "print CLAUDE_CONFIG_DIR=<dir>" bool
	// on this same flag set (flag.Var would panic on the redefinition).
	fs.Var(&addEnv, "seat-env", "(add) extra KEY=VALUE environment for the agent this seat launches; repeatable. For the client bootstrap toggles a third-party endpoint needs (ANTHROPIC_MODEL, ANTHROPIC_CUSTOM_HEADERS, CLAUDE_CODE_USE_GATEWAY, …). NON-SECRET values only — the registry is plaintext and `accounts list --json` prints it, so a credential-shaped NAME is refused; pass the credential's env-var name to --api-key-env and export the value instead")
	addNoLogin := fs.Bool("no-login", false, "(add) do NOT run `claude setup-token`; read the token from --token/stdin instead")
	addToken := fs.String("token", "", "(add) the setup-token (sk-ant-oat…); '-' or empty with --no-login reads stdin")
	addSuffix := fs.String("suffix", firstNonEmpty(os.Getenv("FAK_ACCOUNT_SUFFIX"), "-netra"), "(add) config-dir suffix: dir is ~/.claude-<name> when <name> already ends with it, else ~/.claude-<name><suffix>")
	addNoSync := fs.Bool("no-sync", false, "(add) skip regenerating the roster views after adding (just write the registry)")
	addAdopt := fs.Bool("adopt", false, "(add) enroll by ADOPTING an existing login instead of running `claude setup-token`: copy the source seat's live credential bundle (.credentials.json and/or .oauth-token) into the new isolated dir. Turns the current default login into a rotation seat in one command")
	addFrom := fs.String("from", "", "(add --adopt) source seat to copy the login bundle from: a seat name, a config-dir path, or empty for the default ~/.claude seat")
	addAPIKeyEnv := fs.String("api-key-env", "", "(add) enroll an API-KEY seat (#5331): NAME of the env var holding the account's Anthropic API key (e.g. ANTHROPIC_API_KEY). The registry stores only this REFERENCE, never the secret; `launch` fronts guard with --api-key-env + ACTIVE managed cache. Mutually exclusive with --adopt/--no-login/--token")
	addForce := fs.Bool("force", false, "(add --adopt) reconcile an EXISTING target dir/registry row in place (refresh creds + re-derive identity + upsert) instead of refusing; (refresh) rotate even a credential that is not yet due, by backdating only its recorded expiry so the rotation MUST happen — the difference between assuming a seat can still refresh and witnessing it")
	addProbeIdentity := fs.Bool("probe-identity", false, "(add --adopt) reconcile the adopted seat's identity against a live OAuth profile probe of its credential, preferring the credential over stale on-disk .claude.json metadata (always on for enroll-current)")
	addNoProbeIdentity := fs.Bool("no-probe-identity", false, "(add --adopt) opt OUT of the default identity probe: record the adopted seat's on-disk .claude.json metadata as-is and hit no network (the pre-probe disk-only behavior); enroll-current ignores this and always probes")
	addNoDivorce := fs.Bool("no-divorce", false, "(add --adopt / enroll-current) opt OUT of the default post-copy OAuth token-family divorce. An adopt COPIES the source's credential, so both dirs hold ONE refresh token and the first to refresh silently 401s the other (even hours before its expiresAt). By default the enroll immediately refreshes the NEW seat so it owns its own family — which also proves the seat can refresh, and reports that the SOURCE dir now needs a `/login`. Pass this to control that timing yourself; the shared-family hazard then stays armed")
	refreshTimeout := fs.Duration("refresh-timeout", defaultRefreshTimeout, "(refresh) per-seat deadline for the throwaway `claude -p` turn that causes the credential rotation")
	refreshAckLogout := fs.Bool("yes-log-me-out", false, "(refresh) accept that rotating a seat which shares its OAuth token family with the config dir THIS session runs out of will END this session's login. Such a seat is REFUSED without this flag, because a shared family is one login and the first side to refresh silently 401s the other — the operator's own interactive session, mid-task (#5954). With the flag the rotation proceeds and the report names the invalidated dir and its exact `claude /login` recovery")
	probeIdent := fs.Bool("probe", false, "(status) probe each seat's live credential identity and flag identity-metadata-stale when the on-disk .claude.json disagrees with the account the credential actually serves")
	rmRehome := fs.String("rehome-to", "", "(remove) live seat to rehome the tombstoned account to (default: the registry's anchor seat)")
	rmTerminal := fs.Bool("terminal", false, "(remove) retire without a rehome target because no same-harness account remains")
	rmReason := fs.String("reason", "", "(remove) tombstone_reason recorded in the registry; (rehome) reason token recorded on the live seat switch")
	rehomeAddr := fs.String("addr", defaultSessionAddr(), "(rehome) gateway base URL of the LIVE fak guard session (from the guard banner, or $FAK_ADDR)")
	rehomeKey := fs.String("key", defaultGatewayBearerToken(), "(rehome) bearer credential (only if the gateway sets --require-key)")
	rmArchive := fs.Bool("archive", false, "(remove) ALSO rename the config dir to <dir>.DELETED-<date> and repoint the registry (name+dir+rehome refs) in one command; refuses the live CLAUDE_CONFIG_DIR seat")
	rmByAccount := fs.String("by-account", "", "(remove) retire the WHOLE account: tombstone EVERY active seat that resolves to this account bucket (an email, account UUID, raw bucket key, or seat name), with one --rehome-to + --reason. Refuses if --rehome-to resolves back into the account being retired (#4669)")
	roleFlag := fs.String("role", "", "(set-role) the role to point at --name (active|anchor); may also be given as the first positional")
	launchGuard := fs.Bool("guard", true, "(launch) wrap the agent in `fak guard` so the kernel adjudicates every tool call and the prompt-cache/compaction (vCache) layer is on; --guard=false launches the agent directly")
	launchSkipPerms := fs.Bool("skip-permissions", true, "(launch) pass --dangerously-skip-permissions to claude so fak's capability floor — not Claude's own prompts — is the permission system; --skip-permissions=false lets Claude prompt")
	launchCommand := fs.String("command", "claude", "(launch) the agent command to start under the resolved seat")
	launchUltracode := fs.String("ultracode", "on", "(launch) ultracode posture auto|on|off (default on): run Claude in ultracode (xhigh reasoning + dynamic multi-agent workflow orchestration) via --settings '{\"ultracode\":true}'. on is the default so every instance this launcher starts is ALREADY in ultracode and nobody has to type /effort ultracode into a fresh session; off disables it; auto defers to the work class and turns it on only for rigor-class work, so `--ultracode=auto` is the lean/fast posture for a grind or unclassified launch. true/false are accepted as on/off aliases. Claude-only; ignored for other agents")
	launchModel := fs.String("model", defaultLaunchModel, "(launch) model id a switched Claude launch pins via --model; defaults to Opus 5 ("+defaultLaunchModel+") so every seat starts on it regardless of its own saved default; --model '' launches with the seat's saved default. Claude-only; ignored for other agents")
	launchFallbackModel := fs.String("fallback-model", defaultLaunchFallbackModel, "(launch) comma-separated Claude fallback CHAIN, tried in order when the default Opus 5 startup is unavailable — an unknown/invalid model (-> the previous Opus generation) OR a usage/rate limit such as an Opus weekly cap (-> Fable 5, a separate allocation bucket); empty disables. Default: "+defaultLaunchFallbackModel+". Ignored when --model is explicit")
	launchManagedCache := fs.String("managed-cache", os.Getenv(fleetManagedCacheEnv), "(launch) managed-cache posture for the guard session: auto|on|off (default: $"+fleetManagedCacheEnv+", else on — best-effort managed cache everywhere). on forces the stable-prefix 1h-TTL cache upgrade regardless of billing; explicit auto restores guard's billing-gated default (PASSIVE on a subscription-OAuth seat unless $"+fleetGuardAPIKeyEnvEnv+" makes it ACTIVE); off is the express opt-out for a seat where on self-blocks")
	rotateFlag := fs.Bool("rotate", false, "(launch) launch the NEXT account in the rotation instead of the active/named seat — the round-robin off a walled account")
	afterSeat := fs.String("after", "", "(next/launch) rotate to the account bucket AFTER this seat (default: the named seat, else the active seat)")
	noHeadroom := fs.Bool("no-headroom", false, "(next/launch --rotate) ignore the live runtime headroom signal and rotate stable-by-name; by default rotation prefers the account with room and sorts walled/capped accounts last")
	cooldownClear := fs.String("clear", "", "(cooldown) clear the cooldown for this account key so its seats re-enter the servable pool immediately (use once the account is actually free)")
	backupAt := fs.String("at", "", "(restore-credential) select the snapshot to restore by timestamp OR content-sha prefix; empty restores the newest")
	backupFile := fs.String("file", ".credentials.json", "(restore-credential) which credential blob to restore (.credentials.json|.claude.json|.oauth-token)")
	backupKeep := fs.Int("keep", 20, "(backup) keep at most this many snapshots per file per seat, pruning older ones")
	backupList := fs.Bool("list", false, "(backup) list the stored snapshots for --name instead of taking a new one")
	// Allow a leading positional (e.g. `resolve <name> --env`) BEFORE flags — Go's flag
	// package otherwise stops parsing at the first non-flag token, silently dropping the
	// flags. Collect leading non-flag tokens, parse the remainder, then rejoin.
	lead := 0
	for lead < len(rest) && !strings.HasPrefix(rest[lead], "-") {
		lead++
	}
	if err := fs.Parse(rest[lead:]); err != nil {
		return accountsCmd{}, 2
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

	return accountsCmd{
		fs:                  fs,
		registryPath:        registryPath,
		homeDir:             homeDir,
		gateDir:             gateDir,
		dosView:             dosView,
		jobView:             jobView,
		asJSON:              asJSON,
		listAll:             listAll,
		asEnv:               asEnv,
		pin:                 pin,
		dryRun:              dryRun,
		write:               write,
		checkDiff:           checkDiff,
		addName:             addName,
		addChrome:           addChrome,
		addToken:            addToken,
		addSuffix:           addSuffix,
		addFrom:             addFrom,
		addAPIKeyEnv:        addAPIKeyEnv,
		addBaseURL:          addBaseURL,
		addEnv:              &addEnv,
		addReserved:         addReserved,
		addNoLogin:          addNoLogin,
		addNoSync:           addNoSync,
		addAdopt:            addAdopt,
		addForce:            addForce,
		addProbeIdentity:    addProbeIdentity,
		addNoProbeIdentity:  addNoProbeIdentity,
		addNoDivorce:        addNoDivorce,
		refreshTimeout:      refreshTimeout,
		refreshAckLogout:    refreshAckLogout,
		probeIdent:          probeIdent,
		rmRehome:            rmRehome,
		rmReason:            rmReason,
		rehomeAddr:          rehomeAddr,
		rehomeKey:           rehomeKey,
		rmByAccount:         rmByAccount,
		rmArchive:           rmArchive,
		rmTerminal:          rmTerminal,
		roleFlag:            roleFlag,
		launchCommand:       launchCommand,
		launchModel:         launchModel,
		launchFallbackModel: launchFallbackModel,
		launchManagedCache:  launchManagedCache,
		afterSeat:           afterSeat,
		cooldownClear:       cooldownClear,
		launchGuard:         launchGuard,
		launchSkipPerms:     launchSkipPerms,
		launchUltracode:     launchUltracode,
		rotateFlag:          rotateFlag,
		noHeadroom:          noHeadroom,
		backupAt:            backupAt,
		backupFile:          backupFile,
		backupKeep:          backupKeep,
		backupList:          backupList,
		positional:          positional,
		lead:                lead,
	}, 0
}

func runAccounts(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "headroom" {
		return runAccountsHeadroom(stdout, stderr, argv[1:])
	}
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak accounts <add|enroll-current|remove|restore|backup|restore-credential|set-role|set-default|launch|next|rotation|rehome|list|status|cooldown|doctor|resolve|pull|discover|sync|check|validate|version|check-twins|gate-write> [flags]")
		return 2
	}
	sub, rest := argv[0], argv[1:]

	c, code := parseAccountsCmd(stderr, sub, rest)
	if code != 0 {
		return code
	}
	fs := c.fs
	registryPath, homeDir, gateDir, dosView, jobView := c.registryPath, c.homeDir, c.gateDir, c.dosView, c.jobView
	asJSON, listAll, asEnv, pin, dryRun, write := c.asJSON, c.listAll, c.asEnv, c.pin, c.dryRun, c.write
	checkDiff := c.checkDiff
	addName, addReserved, addChrome, addNoLogin, addToken := c.addName, c.addReserved, c.addChrome, c.addNoLogin, c.addToken
	addSuffix, addNoSync, addAdopt, addFrom, addForce := c.addSuffix, c.addNoSync, c.addAdopt, c.addFrom, c.addForce
	addAPIKeyEnv, addBaseURL, addEnv := c.addAPIKeyEnv, c.addBaseURL, c.addEnv
	addProbeIdentity, addNoProbeIdentity, probeIdent := c.addProbeIdentity, c.addNoProbeIdentity, c.probeIdent
	addNoDivorce, refreshTimeout, refreshAckLogout := c.addNoDivorce, c.refreshTimeout, c.refreshAckLogout
	rmRehome, rmReason, rehomeAddr, rehomeKey, rmArchive := c.rmRehome, c.rmReason, c.rehomeAddr, c.rehomeKey, c.rmArchive
	rmByAccount, rmTerminal := c.rmByAccount, c.rmTerminal
	roleFlag, launchGuard, launchSkipPerms, launchCommand := c.roleFlag, c.launchGuard, c.launchSkipPerms, c.launchCommand
	launchUltracode, launchModel, launchFallbackModel, launchManagedCache := c.launchUltracode, c.launchModel, c.launchFallbackModel, c.launchManagedCache
	rotateFlag, afterSeat, noHeadroom, cooldownClear := c.rotateFlag, c.afterSeat, c.noHeadroom, c.cooldownClear
	positional, lead := c.positional, c.lead

	switch sub {
	case "list":
		reg, err := loadOrDiscover(*registryPath, *homeDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		reg = reg.Refresh()
		if *asJSON {
			// Emit the per-seat LoginReport roster (schema+summary+seats[]) that the
			// sibling `status --json` path produces, not the raw registry persistence
			// wrapper whose seats hide under .homes with empty scalar fields: a machine
			// consumer that iterates the top level then gets the real seat roster with
			// per-seat can_serve/status/warnings, not one empty-fielded object. (#4593)
			report := loginReportWithCooldown(stderr, reg)
			if !*listAll {
				report = report.WithoutTombstoned()
			}
			stdout.Write(mustJSON(report))
			fmt.Fprintln(stdout)
			return 0
		}
		printAccountsTable(stdout, reg, *listAll)
		return 0

	case "status":
		return accountsStatus(stdout, stderr, *registryPath, *homeDir, *asJSON, *probeIdent, *listAll)

	case "cooldown":
		// The usage-limit cooldown surface: list accounts currently walled off a
		// cap (with reset time), or --clear one that is actually free again. The
		// store is the fleet-shared one the launcher writes to, so this reflects
		// what every checkout's login overlay is honoring.
		return accountsCooldown(stdout, stderr, strings.TrimSpace(*cooldownClear), *asJSON)

	case "doctor":
		// The recover/clean fold: one closed action per seat (with the exact command),
		// and --write applies the deterministic repairs through the same audited path
		// as `remove`. See accounts_doctor.go.
		return accountsDoctor(stdout, stderr, *registryPath, *dosView, *jobView, *asJSON, *write)

	case "resolve":
		return accountsResolve(stdout, stderr, positional, *addName, *registryPath, *homeDir, *pin, *asEnv)

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

	case "rotation":
		// The full witnessed rotation decision — the inspect surface `next` is one row of.
		// `next` answers "who's next?"; this answers "why THAT seat, and why is every other
		// seat out?" — the question that costs a source-dive when a bucket silently
		// disappears from rotation (a stale label, a duplicate collapse, a tombstone).
		return accountsRotation(stdout, stderr, *registryPath, *homeDir, *asJSON, !*noHeadroom)

	case "rehome":
		// The LIVE-session seat switch (accounts_rehome.go): talks to a running fak guard
		// gateway, never to the registry — the registry-side rehome is `remove --rehome-to`.
		return runAccountsRehome(stdout, stderr, *rehomeAddr, *rehomeKey, *rmReason, *asJSON)

	case "pull":
		return accountsPull(stdout, stderr, positional, *addName, *registryPath, *homeDir, *dryRun)

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
		// The end-to-end "enroll an account" verb: land a login in an ISOLATED config dir
		// (never ~/.claude), record its identity, upsert the canonical registry, seed the
		// account dir's markers, and regenerate the roster views — one command for what was a
		// multi-file, multi-tool runbook. Two credential sources: the default runs `claude
		// setup-token` interactively for a BRAND-NEW login; `--adopt` copies an EXISTING
		// login's bundle from a source seat (default ~/.claude) so the account you are already
		// logged into becomes a rotation seat with no setup-token and no hand-scripting.
		return runAccountsAdd(stdout, stderr, addParams{
			name:            *addName,
			reserved:        *addReserved,
			chrome:          *addChrome,
			baseURL:         *addBaseURL,
			extraEnv:        *addEnv,
			noLogin:         *addNoLogin,
			token:           *addToken,
			suffix:          *addSuffix,
			noSync:          *addNoSync,
			adopt:           *addAdopt,
			from:            *addFrom,
			force:           *addForce,
			apiKeyEnv:       *addAPIKeyEnv,
			probeIdentity:   *addProbeIdentity,
			noProbeIdentity: *addNoProbeIdentity,
			noDivorce:       *addNoDivorce,
			probeURL:        enrollProfileURL(),
			dryRun:          *dryRun,
			homeDir:         *homeDir,
			registryPath:    *registryPath,
			dosView:         *dosView,
			jobView:         *jobView,
		})

	case "enroll-current":
		// Promote the login the CURRENT session is using into a first-class rotation seat, with
		// an always-on credential-identity probe so the seat is enrolled as the account its live
		// credential actually serves — not whatever the source dir's .claude.json metadata claims
		// (which lies after a /login into a shared dir rewrote only .credentials.json).
		return runAccountsEnrollCurrent(stdout, stderr, enrollParams{
			name:         *addName,
			from:         *addFrom,
			reserved:     *addReserved,
			force:        *addForce,
			suffix:       *addSuffix,
			noSync:       *addNoSync,
			noDivorce:    *addNoDivorce,
			probeURL:     enrollProfileURL(),
			dryRun:       *dryRun,
			homeDir:      *homeDir,
			registryPath: *registryPath,
			dosView:      *dosView,
			jobView:      *jobView,
		})

	case "remove":
		// Tombstone an account in the canonical registry and regenerate the views — the
		// single-source inverse of `add`. The account becomes status=tombstoned with a rehome
		// target + audit fields, drops out of the dos view's active rows, and moves to the job
		// generated views, all from one registry edit.
		//
		// --by-account (#4669) retires the WHOLE account rather than one seat: it tombstones
		// every active seat resolving to the named account bucket in one audited pass, so a
		// duplicate seat identity_mismatched onto the account can't leave it live after the
		// canonical seat is removed.
		//
		// Both retirement forms apply the SAME removal policy — rehome target, audit reason,
		// archive choice, registry and the two generated views — and differ only in the
		// selector that picks what to retire, so the policy is built once here and each form
		// sets just its own selector. That is what keeps a by-account retirement from ever
		// tombstoning under different policy than the by-seat one.
		rm := removeParams{
			rehomeTo:     *rmRehome,
			reason:       *rmReason,
			archive:      *rmArchive,
			terminal:     *rmTerminal,
			registryPath: *registryPath,
			dosView:      *dosView,
			jobView:      *jobView,
			noSync:       *addNoSync,
		}
		if *rmByAccount != "" {
			rm.byAccount = *rmByAccount
			return runAccountsRemoveByAccount(stdout, stderr, rm)
		}
		rm.name = *addName
		return runAccountsRemove(stdout, stderr, rm)

	case "restore":
		// Reverse the reversible half of `remove --archive`: bring the .DELETED config dir
		// back under its live name, clear tombstone policy fields, repair rehome refs that
		// pointed at the archived handle, and regenerate generated roster views.
		return runAccountsRestore(stdout, stderr, restoreParams{
			name:         *addName,
			registryPath: *registryPath,
			dosView:      *dosView,
			jobView:      *jobView,
			noSync:       *addNoSync,
		})

	case "refresh":
		// Proactively rotate seats' OAuth credentials so an IDLE seat never silently decays into a
		// state only a human /login can fix, and so a shared token family (what copying a credential
		// leaves behind) can be split apart on demand. Graded from the FILE — the recorded expiry and
		// the refresh-token family fingerprint — never from the spawn's exit code. Exit 1 while any
		// seat is stale/hollow, so a scheduled sweep can alert on the exit code alone. A seat sharing
		// its family with THIS session's own config dir is refused without --yes-log-me-out: that
		// rotation ends the operator's own login, and it used to do it silently (#5954).
		return runAccountsRefresh(stdout, stderr, refreshParams{
			name:         *addName,
			timeout:      *refreshTimeout,
			force:        *addForce,
			ackLogout:    *refreshAckLogout,
			registryPath: *registryPath,
			homeDir:      *homeDir,
			asJSON:       *asJSON,
		})

	case "backup":
		// The credential safety net (#3987): snapshot every live seat's credential blobs (or just
		// --name) into the content-addressed, gitignored home-tree store before any /login can
		// overwrite them, pruning to --keep per file. --list shows what is recoverable for a seat.
		return runAccountsBackup(stdout, stderr, backupParams{
			name:         *addName,
			list:         *c.backupList,
			keep:         *c.backupKeep,
			registryPath: *registryPath,
			homeDir:      *homeDir,
			asJSON:       *asJSON,
		})

	case "restore-credential":
		// Reverse a credential overwrite (#3987): restore a seat's prior credential blob from the
		// backup store — the newest, or the one named by --at (timestamp/sha prefix). The restore
		// is itself reversible (it snapshots the current blob first).
		return runAccountsRestoreCredential(stdout, stderr, restoreCredParams{
			name:         *addName,
			at:           *c.backupAt,
			file:         *c.backupFile,
			registryPath: *registryPath,
			homeDir:      *homeDir,
			asJSON:       *asJSON,
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
			name:             seat,
			command:          *launchCommand,
			rotate:           *rotateFlag,
			after:            strings.TrimSpace(*afterSeat),
			useHeadroom:      !*noHeadroom,
			useGuard:         *launchGuard,
			skipPerms:        *launchSkipPerms,
			ultracodePosture: strings.TrimSpace(*launchUltracode),
			model:            strings.TrimSpace(*launchModel),
			modelExplicit:    flagSet(fs, "model"),
			fallbackModel:    strings.TrimSpace(*launchFallbackModel),
			managedCache:     strings.TrimSpace(*launchManagedCache),
			dryRun:           *dryRun,
			passthrough:      fs.Args(),
			registryPath:     *registryPath,
			homeDir:          *homeDir,
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
		return accountsCheck(stdout, stderr, *registryPath, *dosView, *jobView, *checkDiff)

	case "version":
		return accountsVersion(stdout, *asJSON)

	default:
		fmt.Fprintf(stderr, "fak accounts: unknown subcommand %q (want add|enroll-current|remove|restore|backup|restore-credential|set-role|set-default|launch|next|rotation|list|status|resolve|pull|discover|sync|check|validate|version|check-twins|gate-write)\n", sub)
		return 2
	}
}

// accountsVersion prints the tool-version surface. A stale binary is the trap this closes: it
// silently lacks a newer verb and fails with a raw "flag provided but not defined" instead of
// saying it is behind. Printing the build + registry schema/family + verb set makes staleness
// VISIBLE — compare it against source, or `go install …/cmd/fak@latest`.
func accountsVersion(stdout io.Writer, asJSON bool) int {
	verbs := []string{
		"add", "enroll-current", "remove", "restore", "backup", "restore-credential", "set-role", "set-default", "launch", "next", "rotation", "list", "status", "resolve", "pull",
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
// (the cooldown-aware ServeAt, #4675 — a seat inside an active usage-limit window is walked
// past), or pinning to the exact seat with --pin (the pure Resolve — the raw static rehome
// pointer). With --env it prints CLAUDE_CONFIG_DIR=<dir> for eval/wrappers, else the bare dir.
// accountsLoadFor resolves the shared `fak accounts <verb> <name>` prologue: it resolves the
// target seat, loads-or-discovers the registry, and refreshes it from disk (Serve/Resolve/Pull
// all need disk-derived identity). The positional is the primary form for back-compat; nameFlag
// (the `--name` value) is the fallback when no positional is given, so `resolve/pull --name X`
// matches the mutating verbs (remove/set-role) that REQUIRE --name — the launch pattern. It
// returns (name, refreshed registry, code, ok): ok=false means the caller should return code
// (2 on a missing name, 1 on a registry load error).
func accountsLoadFor(stderr io.Writer, positional []string, nameFlag, usage, registryPath, homeDir string) (string, accounts.Registry, int, bool) {
	name := ""
	if len(positional) > 0 {
		name = positional[0]
	}
	if name == "" {
		name = strings.TrimSpace(nameFlag)
	}
	if name == "" {
		fmt.Fprintln(stderr, usage)
		return "", accounts.Registry{}, 2, false
	}
	reg, err := loadOrDiscover(registryPath, homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return "", accounts.Registry{}, 1, false
	}
	return name, reg.Refresh(), 0, true
}

func accountsResolve(stdout, stderr io.Writer, positional []string, nameFlag, registryPath, homeDir string, pin, asEnv bool) int {
	// Rehome is the DEFAULT (a seat that can't serve — including one whose account is inside
	// an active cooldown window — falls forward to a live one); --pin is the strict opt-in.
	// The shared prologue refreshes the registry from disk for that identity.
	name, reg, code, ok := accountsLoadFor(stderr, positional, nameFlag, "usage: fak accounts resolve <name>|--name <name> [--env]", registryPath, homeDir)
	if !ok {
		return code
	}
	home, chain, entry, err := accountsResolveServe(reg, name, pin, loadCooldownStoreFailOpen("fak accounts resolve", stderr), time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	accountsReportHome(stderr, home, chain)
	warnAllCooled(stderr, home, entry)
	if asEnv {
		fmt.Fprintf(stdout, "CLAUDE_CONFIG_DIR=%s\n", home.Dir)
	} else {
		fmt.Fprintln(stdout, home.Dir)
	}
	return 0
}

// accountsResolveServe is the pure serve step of `fak accounts resolve` (#4675): the default
// resolution is COOLDOWN-AWARE — ServeAt with the fleet-shared store, so the answer lands on
// a seat that can actually serve now, walking past accounts inside an active usage-limit
// window — while pin keeps the pure, cooldown-blind Resolve for the "where does the static
// rehome pointer go?" question. entry is ServeAt's all-cooled degraded signal (always nil
// from the pin path, which cannot degrade). Split from the I/O wrapper so the
// skip-a-cooled-seat contract is unit-testable without a disk registry or the fleet-shared
// store path.
func accountsResolveServe(reg accounts.Registry, name string, pin bool, cd *accounts.CooldownStore, now time.Time) (accounts.Home, []string, *accounts.CooldownEntry, error) {
	if pin {
		home, chain, err := reg.Resolve(name)
		return home, chain, nil, err
	}
	return reg.ServeAt(name, cd, now)
}

// warnAllCooled surfaces ServeAt's all-cooled terminal on stderr: every reachable,
// otherwise-serveable seat is inside an active cooldown window, and the resolution landed on
// the one with the SOONEST reset rather than failing loud. A nil entry (some seat truly
// serves) prints nothing. Shared by `accounts resolve` and `accounts launch` so both degrade
// with the same explanation.
func warnAllCooled(stderr io.Writer, home accounts.Home, entry *accounts.CooldownEntry) {
	if entry == nil {
		return
	}
	fmt.Fprintf(stderr, "warning: every serveable seat is cooling down — %q has the soonest reset (%s)\n",
		home.Name, entry.ResetAt.UTC().Format(time.RFC3339))
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
	fixes := accountFixSummary(registryPath, reg)
	// By default fold in the live runtime headroom signal so the pool is ordered with the
	// account that has room first and walled/capped accounts last, instead of stable-by-name.
	var hr accounts.RotationHeadroom
	if useHeadroom {
		hr = rotationHeadroom(homeDir)
	}
	dec := reg.NextRotationDecision(after, hr)
	if !dec.OK {
		// Same anchor-aware printer `launch --rotate` uses, so `next` and `launch --rotate` give
		// the IDENTICAL explanation for the identical failure (and name the roomy anchor when the
		// only reason there is nowhere to rotate is that you are already on the account with room).
		printRotationNoCandidate(stderr, "fak accounts next", dec)
		printAccountFixSummary(stderr, fixes, "account fixes")
		return 1
	}
	seat := dec.Seat
	switch {
	case asJSON:
		stdout.Write(mustJSON(seat))
		fmt.Fprintln(stdout)
		printAccountFixSummary(stderr, fixes, "account fixes")
	case asEnv:
		fmt.Fprintf(stdout, "CLAUDE_CONFIG_DIR=%s\n", seat.Dir)
		printAccountFixSummary(stderr, fixes, "account fixes")
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
		printAccountFixSummary(stdout, fixes, "account fixes")
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

func printAccountFixSummary(w io.Writer, sum acctFixSummary, prefix string) {
	if sum.Actionable == 0 {
		return
	}
	if prefix == "" {
		prefix = "account fixes"
	}
	fmt.Fprintf(w, "%s: %d seat(s) need action", prefix, sum.Actionable)
	if by := actionCountsText(sum.ByAction); by != "" {
		fmt.Fprintf(w, " (%s)", by)
	}
	if sum.AutoFixable > 0 {
		fmt.Fprintf(w, "; run `fak accounts doctor --write` for %d auto-fixable repair(s)", sum.AutoFixable)
	} else {
		fmt.Fprint(w, "; run `fak accounts doctor`")
	}
	fmt.Fprintln(w)
	for _, seat := range sum.Seats {
		detail := firstString(seat.Command, seat.Reset, seat.Reason)
		if detail == "" {
			detail = seat.Status
		}
		fmt.Fprintf(w, "  - %s: %s - %s\n", seat.Name, seat.Action, detail)
	}
}

func actionCountsText(by map[string]int) string {
	if len(by) == 0 {
		return ""
	}
	keys := make([]string, 0, len(by))
	for k := range by {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, by[k]))
	}
	return strings.Join(parts, ", ")
}

// accountsRotation prints the FULL witnessed rotation decision: the pool in launch order
// (one seat per DISTINCT account bucket, headroom tiers folded in when available) and every
// excluded seat with the closed reason it is out. It also cross-checks the STORED registry
// identities against disk truth and reports drift — the failure mode where a stale
// .claude.json label (or a re-login that landed on another account) silently reshapes the
// pool while the registry file keeps narrating the old world. Read-only; the healing
// command it points at is `fak accounts discover --write`.
func accountsRotation(stdout, stderr io.Writer, registryPath, homeDir string, asJSON, useHeadroom bool) int {
	stored, err := loadOrDiscover(registryPath, homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	// Snapshot the STORED identities before Refresh: Refresh writes the disk-derived
	// identity into the same backing Homes slice, so the pre-refresh view must be copied
	// out or the drift comparison would compare disk with itself.
	storedIDs := make(map[string]accounts.Identity, len(stored.Homes))
	for _, h := range stored.Homes {
		storedIDs[h.Name] = h.Identity
	}
	live := stored.Refresh()
	var hr accounts.RotationHeadroom
	if useHeadroom {
		hr = rotationHeadroom(homeDir)
	}
	res := live.RotationPlanWithHeadroom(hr)
	drift := identityDrift(storedIDs, live)
	fixes := accountFixSummary(registryPath, live)
	if asJSON {
		stdout.Write(mustJSON(map[string]any{
			"schema":         "fak.accounts.rotation.v1",
			"policy":         res.Policy,
			"order_applied":  res.OrderApplied,
			"pool":           res.Pool,
			"excluded":       res.Excluded,
			"registry_drift": drift,
			"account_fixes":  fixes,
		}))
		fmt.Fprintln(stdout)
		return 0
	}
	fmt.Fprintf(stdout, "# fak %s · rotation over %s · order=%s\n", appversion.Current(), registryPath, res.OrderApplied)
	fmt.Fprintf(stdout, "POOL — %d distinct account bucket(s), launch order:\n", len(res.Pool))
	for i, s := range res.Pool {
		line := fmt.Sprintf("  %d. %-26s %-34s login=%s", i+1, s.Name, s.Email, s.Login)
		if s.Headroom != nil {
			line += "  headroom=" + headroomLabel(*s.Headroom)
		}
		fmt.Fprintln(stdout, line)
	}
	if len(res.Excluded) > 0 {
		fmt.Fprintf(stdout, "EXCLUDED — %d seat(s) out, each with its reason:\n", len(res.Excluded))
		for _, s := range res.Excluded {
			line := fmt.Sprintf("  %-29s %-12s", s.Name, s.Status)
			if s.Status == accounts.RotationDuplicate && s.Canonical != "" {
				line += " -> " + s.Canonical
			}
			if s.Email != "" {
				line += "  (" + s.Email + ")"
			}
			fmt.Fprintln(stdout, line)
		}
	}
	if len(drift.Seats) == 0 {
		fmt.Fprintln(stdout, "registry drift: none — stored identities match disk")
	} else {
		fmt.Fprintf(stdout, "registry drift: %d seat(s) whose stored identity disagrees with disk — heal with `fak accounts discover --write`:\n", len(drift.Seats))
		for _, d := range drift.Seats {
			fmt.Fprintf(stdout, "  %-29s stored %s -> disk %s\n", d.Name, d.Stored, d.Disk)
		}
	}
	printAccountFixSummary(stdout, fixes, "account fixes")
	return 0
}

// rotationDrift is the registry-file staleness report `accounts rotation` carries: each
// seat whose STORED identity (registry.json) no longer matches DISK truth (.claude.json /
// .credentials.json). The live rotation decision is always computed from disk (Refresh),
// so drift never corrupts the pool — but every OTHER reader of the file (a human, an
// external switcher, the job roster) inherits the stale story until it is healed.
type rotationDrift struct {
	Seats []rotationDriftSeat `json:"seats"`
}

// rotationDriftSeat is one drifted seat: the stored vs disk identity, each rendered as
// "<email-or-account-key> creds=<bool>".
type rotationDriftSeat struct {
	Name   string `json:"name"`
	Stored string `json:"stored"`
	Disk   string `json:"disk"`
}

// identityDrift compares each seat's STORED identity (snapshotted before Refresh) against
// its disk-derived refresh and returns the seats that disagree on account (AccountKey) or
// credential presence.
func identityDrift(storedIDs map[string]accounts.Identity, live accounts.Registry) rotationDrift {
	drift := rotationDrift{Seats: []rotationDriftSeat{}}
	for _, l := range live.Homes {
		s, ok := storedIDs[l.Name]
		if !ok {
			continue
		}
		if s.AccountKey() == l.Identity.AccountKey() && s.HasCreds == l.Identity.HasCreds {
			continue
		}
		drift.Seats = append(drift.Seats, rotationDriftSeat{
			Name:   l.Name,
			Stored: driftIdentityLabel(s),
			Disk:   driftIdentityLabel(l.Identity),
		})
	}
	return drift
}

// driftIdentityLabel renders an identity for the drift report: the email when known, else
// the account key, else "-", plus whether live credentials are present.
func driftIdentityLabel(id accounts.Identity) string {
	who := id.Email
	if who == "" {
		who = id.AccountKey()
	}
	if who == "" {
		who = "-"
	}
	return fmt.Sprintf("%s creds=%t", who, id.HasCreds)
}

// accountsStatus emits the first-class login-status report. It is the machine-readable
// sibling of `accounts list`: closed statuses, can_serve, warnings, and next actions live in
// internal/accounts, not in table-rendering guesses.
func accountsStatus(stdout, stderr io.Writer, registryPath, homeDir string, asJSON, probe, showAll bool) int {
	reg, err := loadOrDiscover(registryPath, homeDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak accounts: %v\n", err)
		return 1
	}
	reg = reg.Refresh()
	report := loginReportWithCooldown(stderr, reg)
	if probe {
		// Live credential-identity confirmation: for each seat that serves creds, probe the
		// account the credential ACTUALLY authenticates as and, when it disagrees with the
		// on-disk .claude.json metadata, flag identity_metadata_stale and show the true account.
		// This catches the general mislabel (a named seat's single .claude.json naming A while
		// its credential serves B), which the offline MetadataStale heuristic — scoped to the
		// default home's two-writer conflict — cannot see.
		probeStatusIdentities(stderr, &report, enrollProfileURL())
	}
	if !showAll {
		report = report.WithoutTombstoned()
	}
	if asJSON {
		stdout.Write(mustJSON(report))
		fmt.Fprintln(stdout)
		return 0
	}
	printAccountsStatus(stdout, report, showAll)
	return 0
}

// probeStatusIdentities mutates report in place: for every seat carrying live creds, it probes
// the credential's true identity and — on disagreement with the recorded (disk-derived) account —
// appends the identity_metadata_stale warning and overwrites the seat's shown Account/Email with
// the probed truth. Fail-open per seat: an unprobed or errored seat is left exactly as the
// offline report had it, so the network probe can never make status worse than its disk-only form.
func probeStatusIdentities(stderr io.Writer, report *accounts.LoginReport, profileURL string) {
	probe := func(tok string) (accounts.ProbedIdentity, error) {
		return accounts.ProbeToken(nil, profileURL, tok)
	}
	staleAdded := 0
	for i := range report.Seats {
		s := &report.Seats[i]
		if s.Dir == "" || !s.HasCreds {
			continue
		}
		if s.CredKind == accounts.CredKindAPIKey {
			// An api-key seat's identity is the KEY's org/workspace, not an OAuth profile —
			// there is no disk OAuth credential to resolve, so the profile probe does not
			// apply. TODO(#5331): probe the key's org via the Console/profile endpoint.
			continue
		}
		res := accounts.ResolveCredentialIdentity(s.Dir, probe)
		if res.ProbeErr != nil {
			fmt.Fprintf(stderr, "note: %q (%s): credential identity probe failed (%v) — shown from disk\n", s.Name, s.Dir, res.ProbeErr)
			continue
		}
		if !res.Probed {
			continue
		}
		if res.Resolved.AccountKey() != "" {
			s.Account = res.Resolved.AccountKey()
		}
		s.Email = res.Resolved.Email
		if res.Stale {
			if !hasWarning(s.Warnings, accounts.LoginWarningIdentityStale) {
				s.Warnings = append(s.Warnings, accounts.LoginWarningIdentityStale)
			}
			staleAdded++
			fmt.Fprintf(stderr, "warning: %q (%s): on-disk identity %s but the live credential serves %s — run `fak accounts enroll-current --name %s` (or discover --write) to correct the metadata\n",
				s.Name, s.Dir, identityLabel(res.Disk.Email, res.Disk.AccountUUID), identityLabel(res.Credential.Email, res.Credential.AccountUUID), s.Name)
		}
	}
	if staleAdded > 0 {
		report.Summary.WarningSeats = countWarningSeats(report.Seats)
	}
}

// hasWarning reports whether ws already contains w (so a probe never double-appends a warning the
// offline MetadataStale fold already surfaced).
func hasWarning(ws []accounts.LoginWarning, w accounts.LoginWarning) bool {
	for _, x := range ws {
		if x == w {
			return true
		}
	}
	return false
}

// countWarningSeats recomputes the warning-seat rollup after the probe pass may have added
// identity_metadata_stale flags the offline fold did not carry.
func countWarningSeats(seats []accounts.LoginObservation) int {
	n := 0
	for _, s := range seats {
		if len(s.Warnings) > 0 {
			n++
		}
	}
	return n
}

// loginReportWithCooldown folds the login report with the fleet-shared usage-limit
// cooldown overlay applied: a Ready seat whose upstream account is within an active
// cooldown window shows as cooled_down (can_serve=false), matching what the launcher
// and dispatcher see. Fail-open: an unreadable store yields the plain report so a
// bad state file never blocks a status read.
func loginReportWithCooldown(stderr io.Writer, reg accounts.Registry) accounts.LoginReport {
	store, err := accounts.LoadCooldownStore(defaultCooldownStorePath())
	if err != nil {
		fmt.Fprintf(stderr, "note: cooldown store unreadable (%v) — status shown without the usage-limit overlay\n", err)
		return reg.LoginReport()
	}
	return reg.LoginReportAt(store, time.Now())
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
func accountsPull(stdout, stderr io.Writer, positional []string, nameFlag, registryPath, homeDir string, dryRun bool) int {
	name, reg, code, ok := accountsLoadFor(stderr, positional, nameFlag, "usage: fak accounts pull <name>|--name <name> [--dry-run]", registryPath, homeDir)
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
