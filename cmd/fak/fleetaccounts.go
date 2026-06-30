package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// `fak fleet-accounts` — the native Go port of the READ-ONLY roster/resolve/probe +
// status fold from tools/fleet_accounts.py. It answers "what is an account, and is it
// offered right now?" across BOTH product families (Claude Code + opencode) by:
//   - DISCOVERING config dirs (<home>/.claude*, <config_home>/opencode*),
//   - classifying each by the operator POLICY (accounts_policy.json) into
//     worker | excluded | non-account,
//   - reconciling Claude dirs that share one Anthropic account (duplicate collapse),
//   - folding live runtime status (usage throttle / auth block / live sessions) from the
//     watchdog's sessions.json registry.
//
// It reuses the SAME single sources of truth the Python tool does — the same env-override
// path resolution, the same policy file, the same sessions.json — so it is a drop-in
// read surface, never a second account contract. The --json shapes are byte-compatible
// with the Python `json`/`seats` output.
//
// Subcommands:
//
//	fak fleet-accounts roster|list [--json]   the full classified roster + live status
//	fak fleet-accounts json                   alias for `roster --json` (Python `json` envelope)
//	fak fleet-accounts available              the account dirs safe to offer now (one per line)
//	fak fleet-accounts resolve [--account P] [--work-kind K] [--product P] [--t1|--t2|--t3]
//	                                          ONE flat record: config_dir + oauth_token + tier
//	fak fleet-accounts seats [--product P] [--json]   the explicit seat pool (M distinct seats)
//	fak fleet-accounts status                 the watchdog status fold (roster + availability)
//
// NOT yet ported (documented follow-on, see issue #1415): the ACTIVE network probe
// (`probe`, which delegates to tools/account_probe.py), the probe-LEDGER freshness
// override inside runtime status, and the mutating ops (relogin / top-up / launch). The
// passive registry fold + roster/resolve/seats are the operator hot path and are fully
// ported here with tests; `probe` and the mutators remain on the Python shim.
func cmdFleetAccounts(argv []string) { os.Exit(runFleetAccounts(os.Stdout, os.Stderr, argv)) }

func runFleetAccounts(stdout, stderr io.Writer, argv []string) int {
	mode := "list"
	rest := argv
	if len(argv) > 0 && len(argv[0]) > 0 && argv[0][0] != '-' {
		mode, rest = argv[0], argv[1:]
	}

	fs := flag.NewFlagSet("fleet-accounts "+mode, flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit JSON instead of a human table")
	account := fs.String("account", "", "(resolve) pin to this account tag/basename")
	workKind := fs.String("work-kind", "", "(resolve) operator work-kind (gardening|engineering|...) — wins over tier")
	task := fs.String("task", "", "(resolve) task text for the light/hard heuristic")
	product := fs.String("product", "", "(resolve/seats) scope to a product family (claude|opencode)")
	t1 := fs.Bool("t1", false, "(resolve) pin tier 1")
	t2 := fs.Bool("t2", false, "(resolve) pin tier 2")
	t3 := fs.Bool("t3", false, "(resolve) pin tier 3")
	allowFallback := fs.Bool("allow-tier-fallback", false, "(resolve) allow a tier-1 target to fall back to tier 2")
	faklocalOK := fs.Bool("faklocal-ok", false, "(resolve) synthesize the dogfood .claude-faklocal account when pinned")
	if err := fs.Parse(rest); err != nil {
		return 2
	}

	cwd, _ := os.Getwd()
	toolsDir := filepath.Join(findRepoRoot(cwd), "tools")
	paths := fleetaccounts.ResolvePaths(toolsDir)
	pol := fleetaccounts.LoadPolicy(paths)
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	rows := fleetaccounts.AnnotatedRoster(paths.Home, paths.ConfigHome, pol, reg)

	switch mode {
	case "list", "roster", "status":
		if *asJSON || mode == "json" {
			return emitRosterJSON(stdout, stderr, paths, rows)
		}
		exampleNote := ""
		if !faFileExists(paths.PolicyPath) && faFileExists(paths.ExamplePath) {
			exampleNote = paths.ExamplePath + " (example; copy to _registry/ to customize)"
		}
		fmt.Fprint(stdout, fleetaccounts.RenderList(rows, paths.Home, paths.PolicyPath,
			faFileExists(paths.PolicyPath), exampleNote))
		return 0

	case "json":
		return emitRosterJSON(stdout, stderr, paths, rows)

	case "available":
		for _, r := range fleetaccounts.Available(rows) {
			fmt.Fprintln(stdout, r.Account)
		}
		return 0

	case "resolve":
		taskClass := "auto"
		switch {
		case *t1:
			taskClass = "t1"
		case *t2:
			taskClass = "t2"
		case *t3:
			taskClass = "t3"
		}
		strict := *t1 || *t2 || *t3
		req := fleetaccounts.ResolveRequest{
			Pin: *account, TaskText: *task, TaskClass: taskClass, WorkKind: *workKind,
			Product: *product, AllowTierFallback: *allowFallback, StrictTier: strict,
			FaklocalOK: *faklocalOK,
		}
		resolved := fleetaccounts.Resolve(rows, paths.Home, req, pol)
		out, err := json.MarshalIndent(resolved, "", " ")
		if err != nil {
			fmt.Fprintln(stderr, "fleet-accounts: marshal:", err)
			return 1
		}
		fmt.Fprintln(stdout, string(out))
		if resolved.OK {
			return 0
		}
		return 1

	case "seats":
		pool := fleetaccounts.BuildSeatPool(rows, nil, *product)
		if *asJSON {
			out, err := json.MarshalIndent(pool, "", " ")
			if err != nil {
				fmt.Fprintln(stderr, "fleet-accounts: marshal:", err)
				return 1
			}
			fmt.Fprintln(stdout, string(out))
		} else {
			fmt.Fprint(stdout, fleetaccounts.RenderSeats(pool))
		}
		return 0

	default:
		fmt.Fprintln(stderr, "usage: fak fleet-accounts <roster|list|json|available|resolve|seats|status> [flags]")
		fmt.Fprintln(stderr, "note: the active network probe + mutating ops (relogin/top-up/launch) remain on tools/fleet_accounts.py (issue #1415).")
		return 2
	}
}

// faFileExists reports whether a path exists (used for the policy/registry provenance
// flags in the JSON envelope + list footer).
func faFileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

// emitRosterJSON renders the `json` envelope (the Python json.dumps(indent=1) shape).
func emitRosterJSON(stdout, stderr io.Writer, paths fleetaccounts.Paths,
	rows []fleetaccounts.Account) int {
	env := fleetaccounts.BuildJSONEnvelope(paths.Home, paths.PolicyPath,
		faFileExists(paths.PolicyPath), paths.RegistryPath, faFileExists(paths.RegistryPath), rows)
	out, err := env.MarshalIndent()
	if err != nil {
		fmt.Fprintln(stderr, "fleet-accounts: marshal:", err)
		return 1
	}
	fmt.Fprintln(stdout, string(out))
	return 0
}
