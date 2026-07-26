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
//	fak wip blocked [--landable | --residue] [--stale-days N] [-C <repo>] [--ledger <path>] [--json]
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
	// Residue counts the dirty paths carrying no new work, with the admissions they
	// block. Kept apart from Landable/BlocksRecovered because the remedy differs: these
	// come back by CLEARING an entry, and landing them would revert a peer or delete
	// live code. A caller that adds the two totals is reading a destructive number.
	Residue         int     `json:"residue"`
	ResidueBlocks   int     `json:"residue_blocks"`
	RefusalsScanned int     `json:"refusals_scanned"`
	LedgerPath      string  `json:"ledger_path"`
	StaleAfterDays  float64 `json:"stale_after_days"`
}

func runWipBlocked(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip blocked", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	ledger := fs.String("ledger", "", "dispatch loop ledger to read refusals from (default: FAK_LOOP_LEDGER or .fak/loops.jsonl)")
	staleDays := fs.Float64("stale-days", wipattr.DefaultStaleAfterDays, "treat a change set untouched at least this many days as abandoned")
	landableOnly := fs.Bool("landable", false, "print only the LAND rows — the actionable queue; exit 3 if any exist")
	residueOnly := fs.Bool("residue", false, "print only the RESIDUE rows — dirty paths carrying no new work, to CLEAR not land; exit 3 if any exist")
	asJSON := fs.Bool("json", false, "emit the ranking as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *landableOnly && *residueOnly {
		fmt.Fprintln(stderr, "fak wip blocked: --landable and --residue select disjoint queues; pass at most one")
		return 2
	}

	res, err := wipBlockedScan(context.Background(), *repo, *ledger, *staleDays, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak wip blocked: %v\n", err)
		return 1
	}

	rows := res.Rows
	switch {
	case *landableOnly:
		rows = wipattr.Landable(rows)
	case *residueOnly:
		rows = wipattr.Residue(rows)
	}
	if *asJSON {
		out := res
		out.Rows = rows
		if code := encodeJSONOrFail(stdout, stderr, out, "fak wip blocked"); code != 0 {
			return code
		}
	} else {
		wipBlockedRender(stdout, res, rows, *landableOnly || *residueOnly)
	}

	// Exit 3 when the selected queue is non-empty, mirroring `wip attribute --orphans`:
	// a nonzero-but-not-error code lets a hook or sweep branch on "there is a lever
	// here" without parsing output.
	if (*landableOnly && res.Landable > 0) || (*residueOnly && res.Residue > 0) {
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
	entries := parseWipStatusEntries(status)
	paths := parseWipStatusPaths(status)
	content := wipContentProbe(ctx, repo, entries)

	ledgerPath := ledger
	if ledgerPath == "" {
		ledgerPath = filepath.Join(root, defaultLoopLedger())
	}
	summaries := wipBlockedRefusalSummaries(ledgerPath)
	blocks := wipattr.CountBlocks(summaries)

	rows := wipattr.Rank(wipBlockers(root, paths, now, content), blocks, staleDays)
	return wipBlockedResult{
		Rows:            rows,
		DirtyPaths:      len(rows),
		Landable:        len(wipattr.Landable(rows)),
		BlocksRecovered: wipattr.BlocksRecovered(rows),
		Residue:         len(wipattr.Residue(rows)),
		ResidueBlocks:   wipattr.ResidueBlocks(rows),
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
//
// content maps a path to its established wipattr.Content; a path absent from it stays
// ContentUnprobed and ranks exactly as it did before the content probe existed.
func wipBlockers(root string, paths []string, now time.Time, content map[string]wipattr.Content) []wipattr.Blocker {
	out := make([]wipattr.Blocker, 0, len(paths))
	for _, p := range paths {
		age := 0.0
		if fi, err := os.Stat(filepath.Join(root, filepath.FromSlash(p))); err == nil {
			if d := now.Sub(fi.ModTime()); d > 0 {
				age = d.Hours() / 24
			}
		}
		out = append(out, wipattr.Blocker{Path: p, Set: path.Dir(p), AgeDays: age, Content: content[p]})
	}
	return out
}

// wipContentProbe establishes, for every dirty path, whether its working-tree bytes
// actually differ from what is already committed — the difference between real WIP and
// residue that is destructive to land (see the Content note in internal/wipattr).
//
// It costs exactly two whole-tree git reads regardless of how many paths are dirty:
//
//	git diff --name-only HEAD        -> the paths whose worktree differs from HEAD
//	git diff --name-only @{upstream} -> the paths whose worktree differs from the trunk
//
// A path DIRTY but absent from the first set has a worktree equal to HEAD, so whatever
// makes it dirty lives only in the index. A path in the first set but absent from the
// second already matches the published trunk. Neither read reports untracked files, so
// untracked paths are classified before either set is consulted.
//
// FAIL TOWARD THE OLD RANKING: if the HEAD read fails, nothing is probed and every row
// keeps its pre-Content verdict. If only the upstream read fails (no tracking branch,
// never fetched) the stale-index and phantom-delete shapes are still caught and only
// the landed-upstream shape is skipped — a partial probe is still strictly safer than
// none, and pretending otherwise would throw away the two destructive shapes to avoid
// missing the harmless one.
func wipContentProbe(ctx context.Context, repo string, entries []wipStatusEntry) map[string]wipattr.Content {
	vsHEAD, err := gitWipOut(ctx, repo, nil, "diff", "--name-only", "HEAD")
	if err != nil {
		// Unprobed: an empty map leaves every row ContentUnprobed, i.e. exactly its
		// pre-Content verdict. Never a partial map — a half-probed ranking would
		// silently mix two policies.
		return map[string]wipattr.Content{}
	}
	diverged := wipNameOnlySet(vsHEAD)

	// The upstream read is optional; its absence downgrades the probe, never fails it.
	upstreamKnown := false
	var vsUpstream map[string]bool
	if s, uerr := gitWipOut(ctx, repo, nil, "diff", "--name-only", "@{upstream}"); uerr == nil {
		vsUpstream, upstreamKnown = wipNameOnlySet(s), true
	}

	return wipClassifyContent(entries, diverged, vsUpstream, upstreamKnown)
}

// wipClassifyContent is the PURE half of the content probe: given the porcelain entries
// and the two name-only sets, decide each path's Content. Split from the git reads so
// the four shapes are unit-testable without a repo, matching this file's rule that the
// I/O lives at the edge and the decision does not.
//
// upstreamKnown distinguishes "the upstream read returned nothing to report" from "the
// upstream read never ran" — without it an unfetched repo would read as though every
// path already matched the trunk, which is the one misreading that recommends
// discarding real work.
func wipClassifyContent(entries []wipStatusEntry, diverged, vsUpstream map[string]bool, upstreamKnown bool) map[string]wipattr.Content {
	// A path staged as DELETED that ALSO appears as untracked is a phantom delete: the
	// file is still on disk, so the "deletion" would remove live code. git reports the
	// two facts on separate porcelain lines, which is why the raw entries are needed
	// here rather than the deduplicated path list.
	stagedDelete, untracked := map[string]bool{}, map[string]bool{}
	for _, e := range entries {
		if e.Untracked {
			untracked[e.Path] = true
		} else if e.Index == 'D' {
			stagedDelete[e.Path] = true
		}
	}

	out := make(map[string]wipattr.Content, len(entries))
	for _, e := range entries {
		p := e.Path
		switch {
		case stagedDelete[p] && untracked[p]:
			out[p] = wipattr.ContentPhantomDelete
		case untracked[p]:
			// Neither git diff read covers untracked files, so an untracked path is
			// taken as real work. KNOWN GAP: an untracked file whose bytes already
			// exist upstream reads as work here; catching that needs a per-file blob
			// compare, which this two-read probe deliberately does not pay for.
			out[p] = wipattr.ContentDiverged
		case !diverged[p]:
			out[p] = wipattr.ContentMatchesHEAD
		case upstreamKnown && !vsUpstream[p]:
			out[p] = wipattr.ContentMatchesUpstream
		default:
			out[p] = wipattr.ContentDiverged
		}
	}
	return out
}

// wipNameOnlySet reads a `git diff --name-only` listing into a set. Paths git quoted
// are unquoted so they match the spelling parseWipStatusPaths produces.
func wipNameOnlySet(out string) map[string]bool {
	set := make(map[string]bool)
	for _, line := range strings.Split(out, "\n") {
		if p := unquoteWipStatusPath(strings.TrimSpace(strings.TrimRight(line, "\r"))); p != "" {
			set[p] = true
		}
	}
	return set
}

// wipStatusEntry is one `git status --porcelain` line with its two status columns kept
// apart. The columns matter because a single path can produce TWO lines — a staged
// deletion plus an untracked file of the same name — and that pair is precisely the
// phantom-delete shape that must never be landed. Collapsing to a path set loses it.
type wipStatusEntry struct {
	Path      string
	Index     byte // column X: index vs HEAD
	Worktree  byte // column Y: worktree vs index
	Untracked bool // the "??" entry
}

// parseWipStatusEntries parses every `git status --porcelain` line, preserving
// duplicates so a path reported twice keeps both facts. Rename/copy entries report
// "old -> new"; the live path is the new one. Ignored (!!) entries are skipped, and a
// path git quoted (non-ASCII, spaces) is unquoted so it matches the ledger's spelling.
func parseWipStatusEntries(porcelain string) []wipStatusEntry {
	out := make([]wipStatusEntry, 0)
	for _, line := range strings.Split(porcelain, "\n") {
		line = strings.TrimRight(line, "\r")
		if len(line) < 4 || strings.HasPrefix(line, "!!") {
			continue
		}
		x, y := line[0], line[1]
		rest := strings.TrimSpace(line[2:])
		if i := strings.Index(rest, " -> "); i >= 0 {
			rest = rest[i+len(" -> "):]
		}
		p := unquoteWipStatusPath(strings.TrimSpace(rest))
		if p == "" || strings.HasSuffix(p, "/") {
			continue
		}
		out = append(out, wipStatusEntry{Path: p, Index: x, Worktree: y, Untracked: x == '?' && y == '?'})
	}
	return out
}

// parseWipStatusPaths extracts the deduplicated live repo-relative paths from a `git
// status --porcelain` listing — the ranking's one-row-per-path input, in git's order.
func parseWipStatusPaths(porcelain string) []string {
	out := make([]string, 0)
	seen := make(map[string]bool)
	for _, e := range parseWipStatusEntries(porcelain) {
		if seen[e.Path] {
			continue
		}
		seen[e.Path] = true
		out = append(out, e.Path)
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

func wipBlockedRender(stdout io.Writer, res wipBlockedResult, rows []wipattr.Blocked, filtered bool) {
	switch {
	case res.DirtyPaths == 0:
		fmt.Fprintln(stdout, "clean working tree: nothing is blocking dispatch admission")
		return
	case len(rows) == 0 && filtered:
		fmt.Fprintf(stdout, "no rows in the selected queue: of %d dirty paths, %d are landable and %d are residue\n",
			res.DirtyPaths, res.Landable, res.Residue)
	default:
		fmt.Fprintln(stdout, "STATE\tBLOCKS\tAGE_D\tSET_D\tPATH")
		for _, r := range rows {
			fmt.Fprintf(stdout, "%s\t%d\t%.1f\t%.1f\t%s\n", r.State, r.Blocks, r.AgeDays, r.SetAgeDays, r.Path)
		}
	}

	// The reasons are the audit trail for the queue — print them for the actionable
	// view, where an operator is about to act on the verdict.
	if filtered {
		for _, r := range rows {
			fmt.Fprintf(stdout, "  %s: %s\n", r.Path, r.Reason)
		}
	}
	fmt.Fprintf(stdout, "%d dirty paths · %d landable · %d dispatch admissions recoverable · %d refusals scanned in %s\n",
		res.DirtyPaths, res.Landable, res.BlocksRecovered, res.RefusalsScanned, res.LedgerPath)
	// The residue line is separate and never summed into the landable totals: those
	// admissions come back by clearing an entry, and committing one instead reverts a
	// peer or deletes live code. Printed only when there IS residue, so a clean tree's
	// footer keeps its old shape.
	if res.Residue > 0 {
		fmt.Fprintf(stdout, "%d residue path(s) carry no new work · %d further admissions recoverable WITHOUT committing (see `fak wip blocked --residue`)\n",
			res.Residue, res.ResidueBlocks)
	}
}
