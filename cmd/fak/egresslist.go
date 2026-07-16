package main

// egresslist.go — `fak egresslist`, the maintenance surface for the bundled egress filter
// lists the adjudicator's egress rung compiles (internal/egresslist/lists).
//
//	fak egresslist refresh                       # re-fetch every list that records an upstream
//	fak egresslist refresh --dry-run             # show what would change, write nothing
//	fak egresslist refresh --name stevenblack    # just one list
//	fak egresslist refresh --json                # machine-readable per-list outcome
//
// WHY A VERB AND NOT A CRON. Refreshing rewrites the checked-in artifact and re-pins its
// checksum, then STOPS. The diff is the review, the commit is the audit record. Nothing
// fetches at adjudication time: the decide path compiles embedded bytes and stays offline
// and deterministic. An egress block set that silently updated itself under a live agent
// would be an unreviewed change to what the kernel permits — which is the thing the layer
// exists to prevent.
//
// FAIL CLOSED. A fetch error, a non-200, an oversize body, an upstream that parses to too
// few rules, or one that collapsed against its pinned rule count all leave the previously
// pinned artifact untouched and exit 3. A stale block list still blocks yesterday's
// malware; an empty one blocks nothing at all.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/egressrefresh"
)

func cmdEgresslist(argv []string) { os.Exit(runEgresslist(os.Stdout, os.Stderr, argv)) }

// runEgresslist dispatches the egresslist subcommands. It is the testable core.
func runEgresslist(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprint(stderr, egresslistUsage)
		return 2
	}
	switch argv[0] {
	case "refresh":
		return runEgresslistRefresh(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprint(stdout, egresslistUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak egresslist: unknown subcommand %q\n\n%s", argv[0], egresslistUsage)
		return 2
	}
}

const egresslistUsage = `usage: fak egresslist <subcommand> [flags]

subcommands:
  refresh   re-fetch the bundled filter lists from their recorded provenance URLs,
            re-normalize through the kernel's own ingest path, and rewrite the
            checked-in artifact + its pinned checksum (a reviewable diff)

run "fak egresslist refresh -h" for flags
`

// runEgresslistRefresh is the testable core of `fak egresslist refresh`. Exit 0 when every
// selected list resolved (updated/unchanged/skipped), 1 on a run error, 2 on usage, 3 when
// at least one list FAILED CLOSED — so CI or a script can gate on a refusal instead of
// reading prose.
func runEgresslistRefresh(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("egresslist refresh", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "repository root holding internal/egresslist/lists")
	asJSON := fs.Bool("json", false, "emit the per-list outcome as JSON")
	dryRun := fs.Bool("dry-run", false, "report what would change without writing anything")
	allowShrink := fs.Bool("allow-shrink", false, "accept an upstream whose rule count collapsed (waives the truncation guard)")
	minRules := fs.Int("min-rules", egressrefresh.DefaultMinRules, "refuse an upstream parsing to fewer rules than this")
	timeout := fs.Duration("timeout", egressrefresh.DefaultTimeout, "per-list fetch timeout")
	var names stringList
	fs.Var(&names, "name", "refresh only this list (repeatable; default: every recorded list)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	dir := filepath.Join(*root, "internal", "egresslist", "lists")
	ctx, cancel := context.WithTimeout(context.Background(), *timeout*time.Duration(len(names)+1))
	defer cancel()

	results, err := egressrefresh.Refresh(ctx, egressrefresh.Options{
		Dir:         dir,
		Names:       names,
		Fetcher:     egressrefresh.HTTPFetcher{},
		MinRules:    *minRules,
		AllowShrink: *allowShrink,
		DryRun:      *dryRun,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak egresslist refresh: %v\n", err)
		return 1
	}

	failed := 0
	for _, r := range results {
		if r.Status == egressrefresh.StatusFailed {
			failed++
		}
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(map[string]any{
			"dir":     dir,
			"dry_run": *dryRun,
			"failed":  failed,
			"lists":   results,
		}); err != nil {
			fmt.Fprintf(stderr, "fak egresslist refresh: encode: %v\n", err)
			return 1
		}
		if failed > 0 {
			return 3
		}
		return 0
	}

	if len(results) == 0 {
		fmt.Fprintln(stdout, "no bundled lists recorded")
		return 0
	}
	if *dryRun {
		fmt.Fprintln(stdout, "(dry run - nothing written)")
	}
	for _, r := range results {
		switch r.Status {
		case egressrefresh.StatusUpdated:
			fmt.Fprintf(stdout, "updated   %-20s %d -> %d rules  %s -> %s\n",
				r.Name, r.OldRules, r.NewRules, short(r.OldSHA256), short(r.NewSHA256))
		case egressrefresh.StatusUnchanged:
			fmt.Fprintf(stdout, "unchanged %-20s %d rules  %s\n", r.Name, r.NewRules, short(r.NewSHA256))
		case egressrefresh.StatusSkipped:
			fmt.Fprintf(stdout, "skipped   %-20s %s\n", r.Name, r.Reason)
		case egressrefresh.StatusFailed:
			fmt.Fprintf(stdout, "FAILED    %-20s %s\n", r.Name, r.Reason)
		}
	}
	if failed > 0 {
		fmt.Fprintf(stderr, "\n%d list(s) failed closed; the previously pinned artifact(s) are intact.\n", failed)
		return 3
	}
	if !*dryRun {
		fmt.Fprintln(stdout, "\nReview the diff and commit it: the checked-in artifact is the source of truth.")
	}
	return 0
}

// short (checksum prefixes for the human table) and stringList (the repeatable --name
// flag) are the package's existing helpers, in commit.go and dogfoodissues.go.
