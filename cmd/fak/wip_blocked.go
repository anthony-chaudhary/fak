package main

// wip_blocked.go — `fak wip blocked`, the cost view over the dirty working tree
// (#4320). `fak wip attribute --orphans` answers "which dirty hunks are at risk?";
// this answers "which of them is throttling the fleet, and may I land it?".
//
// It exists because the step-1 measurement in docs/dispatch/cmd-lane-split-plan.md
// found that dispatch concurrency on this ledger is gated by orphan-WIP hygiene, not
// lease geometry: 146 of 172 refusals were DIRTY_PATH_COLLISION, 125 of those named a
// path dirty for >=2 days, and ONE abandoned file carried 87 of them. That ranking was
// produced by hand once; a lever that large should not need a human to rediscover it,
// so this verb recomputes it from the ledger the guard already writes.
//
//	fak wip blocked [--landable] [--stale-days N] [-C <repo>] [--ledger <path>] [--json]
//
// All git/ledger/filesystem I/O lives here; the ranking is the pure wipattr.Rank fold.

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/wipattr"
)

// wipBlockedResult is the JSON/plain payload. The footer counts are carried
// explicitly rather than recomputed by consumers so a scripted caller and the human
// listing can never disagree about how many admissions a sweep would recover.
type wipBlockedResult struct {
	Rows            []wipattr.Blocked `json:"rows"`
	DirtyPaths      int               `json:"dirty_paths"`
	Landable        int               `json:"landable"`
	BlocksRecovered int               `json:"blocks_recovered"`
	RefusalsScanned int               `json:"refusals_scanned"`
	LedgerPath      string            `json:"ledger_path"`
	StaleAfterDays  float64           `json:"stale_after_days"`
}

func runWipBlocked(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip blocked", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	ledger := fs.String("ledger", "", "dispatch loop ledger to read refusals from (default: FAK_LOOP_LEDGER or .fak/loops.jsonl)")
	staleDays := fs.Float64("stale-days", wipattr.DefaultStaleAfterDays, "treat a change set untouched at least this many days as abandoned")
	landableOnly := fs.Bool("landable", false, "print only the LAND rows — the actionable queue; exit 3 if any exist")
	asJSON := fs.Bool("json", false, "emit the ranking as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	res, err := wipBlockedScan(context.Background(), *repo, *ledger, *staleDays, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak wip blocked: %v\n", err)
		return 1
	}

	rows := res.Rows
	if *landableOnly {
		rows = wipattr.Landable(rows)
	}
	if *asJSON {
		out := res
		out.Rows = rows
		if code := encodeJSONOrFail(stdout, stderr, out, "fak wip blocked"); code != 0 {
			return code
		}
	} else {
		wipBlockedRender(stdout, res, rows, *landableOnly)
	}

	// Exit 3 when there is landable work, mirroring `wip attribute --orphans`: a
	// nonzero-but-not-error code lets a hook or sweep branch on "there is a lever
	// here" without parsing output.
	if *landableOnly && res.Landable > 0 {
		return 3
	}
	return 0
}

// wipBlockedScan gathers the three inputs Rank needs — the dirty path set, each
// path's mtime age, and the per-path refusal counts — and folds them.
func wipBlockedScan(ctx context.Context, repo, ledger string, staleDays float64, now time.Time) (wipBlockedResult, error) {
	root, err := gitWipOut(ctx, repo, nil, "rev-parse", "--show-toplevel")
	if err != nil {
		return wipBlockedResult{}, fmt.Errorf("resolve repo root: %w", err)
	}
	root = strings.TrimSpace(root)

	// -uall expands untracked directories to files: an untracked dir has no useful
	// mtime, and its contents are exactly the orphan WIP this verb is looking for.
	status, err := gitWipOut(ctx, repo, nil, "status", "--porcelain", "-uall")
	if err != nil {
		return wipBlockedResult{}, fmt.Errorf("read working-tree status: %w", err)
	}
	paths := parseWipStatusPaths(status)

	ledgerPath := ledger
	if ledgerPath == "" {
		ledgerPath = filepath.Join(root, defaultLoopLedger())
	}
	summaries := wipBlockedRefusalSummaries(ledgerPath)
	blocks := wipattr.CountBlocks(summaries)

	rows := wipattr.Rank(wipBlockers(root, paths, now), blocks, staleDays)
	return wipBlockedResult{
		Rows:            rows,
		DirtyPaths:      len(rows),
		Landable:        len(wipattr.Landable(rows)),
		BlocksRecovered: wipattr.BlocksRecovered(rows),
		RefusalsScanned: len(summaries),
		LedgerPath:      ledgerPath,
		StaleAfterDays:  staleDays,
	}, nil
}

// wipBlockedRefusalSummaries returns the ledger summaries that actually name a dirty
// path. A missing or chain-broken ledger degrades to the recovered prefix (possibly
// none) rather than failing: with no ledger every row simply blocks nothing, which is
// an honest "no measured cost signal" instead of a fabricated one.
func wipBlockedRefusalSummaries(ledgerPath string) []string {
	events, _, err := loopmgr.LoadPrefix(ledgerPath)
	if err != nil {
		return nil
	}
	var out []string
	for _, ev := range events {
		if len(wipattr.ParseBlockedPaths(ev.Summary)) > 0 {
			out = append(out, ev.Summary)
		}
	}
	return out
}

// wipBlockers stats each dirty path into a wipattr.Blocker, grouping by directory as
// the change-set key. The directory is the right default grain for Go: an
// implementation and the test that exercises it share a package, which is exactly the
// pairing that must never be landed half-way.
//
// An unstattable path (a staged deletion, a vanished file) gets age 0 — deliberately
// the FRESHEST possible value, so a path whose staleness cannot be established is
// never recommended for landing.
func wipBlockers(root string, paths []string, now time.Time) []wipattr.Blocker {
	out := make([]wipattr.Blocker, 0, len(paths))
	for _, p := range paths {
		age := 0.0
		if fi, err := os.Stat(filepath.Join(root, filepath.FromSlash(p))); err == nil {
			if d := now.Sub(fi.ModTime()); d > 0 {
				age = d.Hours() / 24
			}
		}
		out = append(out, wipattr.Blocker{Path: p, Set: path.Dir(p), AgeDays: age})
	}
	return out
}

// parseWipStatusPaths extracts the live repo-relative path from each `git status
// --porcelain` line. Rename/copy entries report "old -> new"; the live path is the new
// one. Ignored (!!) entries are skipped, and a path git quoted (non-ASCII, spaces) is
// unquoted so it matches the ledger's spelling.
func parseWipStatusPaths(porcelain string) []string {
	out := make([]string, 0)
	seen := make(map[string]bool)
	for _, line := range strings.Split(porcelain, "\n") {
		line = strings.TrimRight(line, "\r")
		if len(line) < 4 || strings.HasPrefix(line, "!!") {
			continue
		}
		rest := strings.TrimSpace(line[2:])
		if i := strings.Index(rest, " -> "); i >= 0 {
			rest = rest[i+len(" -> "):]
		}
		p := unquoteWipStatusPath(strings.TrimSpace(rest))
		if p == "" || strings.HasSuffix(p, "/") || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	return out
}

// unquoteWipStatusPath undoes git's C-style quoting. An unparseable quoted path falls
// back to the raw text minus its quotes rather than being dropped — a path we spell
// slightly wrong still surfaces as dirty; a path we drop disappears from the ranking.
func unquoteWipStatusPath(s string) string {
	if !strings.HasPrefix(s, `"`) {
		return s
	}
	if unquoted, err := strconv.Unquote(s); err == nil {
		return unquoted
	}
	return strings.Trim(s, `"`)
}

func wipBlockedRender(stdout io.Writer, res wipBlockedResult, rows []wipattr.Blocked, landableOnly bool) {
	switch {
	case res.DirtyPaths == 0:
		fmt.Fprintln(stdout, "clean working tree: nothing is blocking dispatch admission")
		return
	case len(rows) == 0 && landableOnly:
		fmt.Fprintf(stdout, "no landable rows: none of %d dirty paths is both blocking and idle >=%.1fd\n",
			res.DirtyPaths, res.StaleAfterDays)
	default:
		fmt.Fprintln(stdout, "STATE\tBLOCKS\tAGE_D\tSET_D\tPATH")
		for _, r := range rows {
			fmt.Fprintf(stdout, "%s\t%d\t%.1f\t%.1f\t%s\n", r.State, r.Blocks, r.AgeDays, r.SetAgeDays, r.Path)
		}
	}

	// The reasons are the audit trail for the queue — print them for the actionable
	// view, where an operator is about to act on the verdict.
	if landableOnly {
		for _, r := range rows {
			fmt.Fprintf(stdout, "  %s: %s\n", r.Path, r.Reason)
		}
	}
	fmt.Fprintf(stdout, "%d dirty paths · %d landable · %d dispatch admissions recoverable · %d refusals scanned in %s\n",
		res.DirtyPaths, res.Landable, res.BlocksRecovered, res.RefusalsScanned, res.LedgerPath)
}
