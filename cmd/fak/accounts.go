package main

import (
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// `fak accounts` — the durable, identity-true registry of Claude config homes
// (CLAUDE_CONFIG_DIR seats), with tombstone + auto-rehome. It answers two questions a
// directory name alone cannot: WHO is this seat actually logged in as (disk truth, so
// a name that lies is flagged), and WHERE does a retired seat send anything still
// pinned to it (a tombstone's rehome target, followed transitively). See
// internal/accounts for the model.
//
// Subcommands:
//
//	fak accounts list                  table of every seat: name, status, TRUE identity, creds, rehome, flags
//	fak accounts resolve <name> [--env] the live config dir serving <name>, following a tombstone's rehome
//	fak accounts discover              emit a starter registry.json from ~/.claude* (disk truth)
//	fak accounts validate              load the registry and check every invariant (incl. tombstones resolve)
func cmdAccounts(argv []string) { os.Exit(runAccounts(os.Stdout, os.Stderr, argv)) }

func runAccounts(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak accounts <list|resolve|pull|discover|validate> [flags]")
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

	default:
		fmt.Fprintf(stderr, "fak accounts: unknown subcommand %q (want list|resolve|pull|discover|validate)\n", sub)
		return 2
	}
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
