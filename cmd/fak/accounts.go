package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/accounts"
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
//	fak accounts remove --name <n>     tombstone an account in the registry + regenerate views
//	fak accounts set-default --name <n> make <n> the single default (active) seat + regenerate views
//	fak accounts list                  table of every seat: name, status, TRUE identity, creds, rehome, flags
//	fak accounts resolve <name> [--env] the live config dir serving <name>, following a tombstone's rehome
//	fak accounts discover [--write]    emit (or MERGE-and-write) a registry.json from ~/.claude* (disk truth)
//	fak accounts sync                  project the registry into the dos + job roster views
//	fak accounts check                 RED (exit 1) if a generated view drifts from the registry
//	fak accounts validate              load the registry and check every invariant (incl. tombstones resolve)
func cmdAccounts(argv []string) { os.Exit(runAccounts(os.Stdout, os.Stderr, argv)) }

func runAccounts(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak accounts <add|remove|set-default|list|resolve|pull|discover|sync|check|validate|check-twins|gate-write> [flags]")
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
	dryRun := fs.Bool("dry-run", false, "(pull) print what would be pulled without copying")
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
	rmRehome := fs.String("rehome-to", "", "(remove) live seat to rehome the tombstoned account to (default: the registry's default seat)")
	rmReason := fs.String("reason", "", "(remove) tombstone_reason recorded in the registry")
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

	case "resolve":
		if len(positional) == 0 {
			fmt.Fprintln(stderr, "usage: fak accounts resolve <name> [--env]")
			return 2
		}
		name := positional[0]
		reg, err := loadOrDiscover(*registryPath, *homeDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		// Rehome is the DEFAULT (a seat that can't serve falls forward to a live one);
		// --pin is the strict opt-in. Serve needs disk-derived identity, so refresh.
		reg = reg.Refresh()
		var home accounts.Home
		var chain []string
		if *pin {
			home, chain, err = reg.Resolve(name)
		} else {
			home, chain, err = reg.Serve(name)
		}
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		for i, hop := range chain {
			to := home.Name
			if i+1 < len(chain) {
				to = chain[i+1]
			}
			fmt.Fprintf(stderr, "note: %q can't serve -> rehoming to %q\n", hop, to)
		}
		id := accounts.DeriveIdentity(home.Dir)
		if !id.HasCreds {
			fmt.Fprintf(stderr, "warning: %q (%s) has no live credentials — claude will prompt for /login\n", home.Name, home.Dir)
		}
		if *asEnv {
			fmt.Fprintf(stdout, "CLAUDE_CONFIG_DIR=%s\n", home.Dir)
		} else {
			fmt.Fprintln(stdout, home.Dir)
		}
		return 0

	case "pull":
		if len(positional) == 0 {
			fmt.Fprintln(stderr, "usage: fak accounts pull <name> [--dry-run]")
			return 2
		}
		name := positional[0]
		reg, err := loadOrDiscover(*registryPath, *homeDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		reg = reg.Refresh()
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
			if *dryRun {
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

	case "discover":
		if *write {
			// Regenerator mode: load the canonical registry (or start empty), MERGE the disk
			// scan in (refresh identities, add new dirs, PRESERVE authored policy fields), and
			// write it back atomically. This is how the registry becomes the single source of
			// truth without a human re-typing identities — it derives them from disk.
			base := accounts.Registry{}
			if _, err := os.Stat(*registryPath); err == nil {
				base, err = accounts.LoadRegistry(*registryPath)
				if err != nil {
					fmt.Fprintf(stderr, "fak accounts: %v\n", err)
					return 1
				}
			}
			merged, err := base.MergeDiscovered(*homeDir)
			if err != nil {
				fmt.Fprintf(stderr, "fak accounts: %v\n", err)
				return 1
			}
			if err := accounts.SaveRegistry(*registryPath, merged); err != nil {
				fmt.Fprintf(stderr, "fak accounts: %v\n", err)
				return 1
			}
			fmt.Fprintf(stdout, "wrote %d home(s) to %s\n", len(merged.Homes), *registryPath)
			return 0
		}
		homes, err := accounts.Discover(*homeDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		reg := accounts.Registry{Homes: homes}
		stdout.Write(reg.JSON())
		return 0

	case "validate":
		reg, err := accounts.LoadRegistry(*registryPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "ok: %d homes, registry valid (%s)\n", len(reg.Homes), *registryPath)
		return 0

	case "check-twins":
		// The audit gate for Regression A: red (exit 1) when two config homes logged into
		// DIFFERENT accounts share one setup-token fingerprint (the cross-account smear
		// that surfaces as "subscription disabled"). Homes that share a token but resolve
		// to ONE account (~/.claude + its named dir) are legitimate and pass.
		findings, err := accounts.AuditTokenTwins(*homeDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		if *asJSON {
			stdout.Write(mustJSON(map[string]any{"clean": len(findings) == 0, "findings": findings}))
			fmt.Fprintln(stdout)
		}
		if len(findings) == 0 {
			if !*asJSON {
				fmt.Fprintln(stdout, "ok: no cross-account token-twins — every shared setup token is one account")
			}
			return 0
		}
		if !*asJSON {
			for _, f := range findings {
				fmt.Fprintf(stdout, "TOKEN-TWIN: homes [%s] share one setup token but log into %d accounts [%s]\n",
					strings.Join(f.Homes, ", "), len(f.Accounts), strings.Join(f.Accounts, ", "))
			}
			fmt.Fprintf(stderr, "fak accounts: %d cross-account token-twin(s) — a foreign token will surface as "+
				"\"subscription disabled\". Give each account its OWN setup token in its OWN dir.\n", len(findings))
		}
		return 1

	case "gate-write":
		// Pre-write gate: decide whether writing a setup token (stdin) into --dir is safe,
		// BEFORE any flow persists it. Exit 0 = safe to write; exit 1 = refused (would
		// create a cross-account token-twin). The token is read from stdin only, never
		// argv, and is fingerprinted, never echoed.
		if *gateDir == "" {
			fmt.Fprintln(stderr, "usage: fak accounts gate-write --dir <config-dir> < token")
			return 2
		}
		tokBytes, _ := io.ReadAll(os.Stdin)
		verdict := accounts.GateTokenWrite(*gateDir, string(tokBytes), *homeDir)
		if *asJSON {
			stdout.Write(mustJSON(verdict))
			fmt.Fprintln(stdout)
		} else if verdict.Allow {
			fmt.Fprintf(stdout, "ok: safe to write into %s (login: %s)\n", *gateDir, verdict.DirAccount)
		} else {
			fmt.Fprintf(stderr, "REFUSED (%s): %s\n", verdict.Reason, verdict.Detail)
		}
		if verdict.Allow {
			return 0
		}
		return 1

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
			registryPath: *registryPath,
			dosView:      *dosView,
			jobView:      *jobView,
			noSync:       *addNoSync,
		})

	case "set-default":
		// Make <name> the single default (active) seat — the deterministic one-command inverse
		// of hand-editing `default: true`. The default is what a bare launch / the watchdog
		// picks; it is surfaced as `active_default` in the dos view.
		return runAccountsSetDefault(stdout, stderr, setDefaultParams{
			name:         *addName,
			registryPath: *registryPath,
			dosView:      *dosView,
			jobView:      *jobView,
			noSync:       *addNoSync,
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
		// Drift detector: RED (exit 1) if any on-disk view differs from a freshly-rendered
		// projection of the registry. This is the ratchet that keeps the generated views from
		// silently diverging from the canonical source again.
		reg, err := accounts.LoadRegistry(*registryPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak accounts: %v\n", err)
			return 1
		}
		reg = reg.Refresh()
		drift := 0
		for _, t := range viewTargets(*dosView, *jobView) {
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

	default:
		fmt.Fprintf(stderr, "fak accounts: unknown subcommand %q (want add|remove|set-default|list|resolve|pull|discover|sync|check|validate|check-twins|gate-write)\n", sub)
		return 2
	}
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
	return wrote, 0
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
	// Reconcile groups the seats by the account each truly resolves to, so the table can
	// flag a seat that is really a duplicate of another (one rate-limit bucket presented
	// as several) and a seat whose setup token belongs to a different login than its own.
	rec := reg.Reconcile()
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "NAME\tSTATUS\tIDENTITY\tCREDS\tREHOME\tFLAG")
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
		ident := h.Identity.Email
		if ident == "" {
			ident = "-"
		}
		creds := "-"
		if h.Identity.Exists {
			if h.Identity.HasCreds {
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
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n", name, status, ident, creds, rehome, strings.Join(flags, "; "))
	}
	tw.Flush()
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
