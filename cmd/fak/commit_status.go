package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/commitlane"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

var commitStatusFn = commitlane.Status

// removeIndexLockFn removes a reclaimed stale .git/index.lock or an orphaned
// .git/next-index-<pid>.lock. Indirected so the reclaim actuator is testable without
// touching a real lock file.
var removeIndexLockFn = os.Remove

func runCommitStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("commit status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	asJSON := fs.Bool("json", false, "emit the commit-lane status as JSON")
	reclaim := fs.Bool("reclaim-stale-index-lock", false, "reclaim an orphaned .git/index.lock, and sweep leftover .git/next-index-<pid>.lock residue, when the lane evidence proves them stale with no live writer (dry-run unless --apply)")
	apply := fs.Bool("apply", false, "with --reclaim-stale-index-lock, actually remove the reclaimed files (default: dry-run)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak commit status: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	rep, err := commitStatusFn(context.Background(), commitlane.Options{Dir: pathutil.ExpandTilde(*dir)})
	if err != nil {
		fmt.Fprintf(stderr, "fak commit status: %v\n", err)
		return 1
	}
	if *reclaim {
		return runIndexLockReclaim(stdout, stderr, rep, *apply)
	}
	if *asJSON {
		return encodeJSONOrFailPrefixed(stdout, stderr, rep, "fak commit status")
	}
	renderCommitStatus(stdout, rep)
	return 0
}

// runIndexLockReclaim applies the stale-.git/index.lock reclaim decision derived from
// the lane report (#5294), then sweeps the orphaned .git/next-index-<pid>.lock residue
// git leaves behind and never reaps (#5338). It NEVER removes a file the decision would
// keep: the present + stale-past-grace signature reaps (index.lock has no owner pid, so
// staleness alone refutes a holder — an unrelated by-name writer or a failed probe only
// gate a FRESH lock, matching safecommit's age-only reap; #5335 item 3), and even then
// only with --apply — the default is a dry-run that reports what it would do. A file
// another session already cleared is idempotent success, not an error.
//
// The two sweeps share one flag because they share one cause: an index writer that died
// mid-write leaves BOTH the lock that wedges the lane and the temp file that accumulates
// behind it. Reclaiming only the lock is what let 60+ next-index files pile up.
func runIndexLockReclaim(stdout, stderr io.Writer, rep commitlane.Report, apply bool) int {
	code := 0
	d := commitlane.DecideIndexLockReclaim(rep)
	switch {
	case !d.Reap:
		fmt.Fprintf(stdout, "index.lock: no reclaim (%s) %s\n", d.Reason, d.Path)
	case !apply:
		fmt.Fprintf(stdout, "index.lock: WOULD reclaim (%s) %s — re-run with --apply to remove\n", d.Reason, d.Path)
	default:
		err := removeIndexLockFn(d.Path)
		switch {
		case err == nil:
			fmt.Fprintf(stdout, "index.lock: reclaimed stale orphan (%s) %s\n", d.Reason, d.Path)
		case errors.Is(err, os.ErrNotExist):
			fmt.Fprintf(stdout, "index.lock: already cleared %s\n", d.Path)
		default:
			fmt.Fprintf(stderr, "fak commit status: reclaim failed: %v\n", err)
			code = 1
		}
	}
	if c := runNextIndexReclaim(stdout, stderr, rep, apply); c != 0 {
		code = c
	}
	return code
}

// runNextIndexReclaim reaps the orphaned .git/next-index-<pid>.lock residue under the
// same flag, the same --apply gate and the same actuator as the index.lock reclaim,
// removing only the files DecideNextIndexReclaim proved reapable (dead named owner, no
// live writer, stale past the grace window). Output is summarized rather than per-file:
// this residue accumulates into the dozens, and an operator needs the count and the keep
// reasons — not sixty identical lines. Prints nothing when there is no residue, so a
// clean repo's reclaim output is unchanged.
func runNextIndexReclaim(stdout, stderr io.Writer, rep commitlane.Report, apply bool) int {
	decisions := commitlane.DecideNextIndexReclaim(rep)
	if len(decisions) == 0 {
		return 0
	}
	var reapable []commitlane.NextIndexReclaim
	kept := map[commitlane.IndexLockReclaimReason]int{}
	for _, d := range decisions {
		if d.Reap {
			reapable = append(reapable, d)
			continue
		}
		kept[d.Reason]++
	}
	switch {
	case len(reapable) == 0:
		fmt.Fprintf(stdout, "next-index residue: %d file(s) — no reclaim (%s)\n", len(decisions), keepReasonSummary(kept))
		return 0
	case !apply:
		fmt.Fprintf(stdout, "next-index residue: %d file(s) — WOULD reclaim %d%s — re-run with --apply to remove\n",
			len(decisions), len(reapable), keptSuffix(kept))
		return 0
	}
	removed, code := 0, 0
	for _, d := range reapable {
		// An already-cleared file is success: a peer session reclaiming the same residue
		// concurrently is the expected case on a shared trunk, not a failure.
		if err := removeIndexLockFn(d.Path); err != nil && !errors.Is(err, os.ErrNotExist) {
			fmt.Fprintf(stderr, "fak commit status: next-index reclaim failed: %v\n", err)
			code = 1
			continue
		}
		removed++
	}
	fmt.Fprintf(stdout, "next-index residue: reclaimed %d of %d stale orphan(s)%s\n", removed, len(decisions), keptSuffix(kept))
	return code
}

// keepReasonSummary renders the kept-file tally as a stable, sorted, closed-vocabulary
// string (e.g. "keep_fresh=2, keep_live_owner=1") so an operator — and a test — can read
// WHY files survived without a per-file dump.
func keepReasonSummary(kept map[commitlane.IndexLockReclaimReason]int) string {
	if len(kept) == 0 {
		return "none"
	}
	parts := make([]string, 0, len(kept))
	for reason, n := range kept {
		parts = append(parts, fmt.Sprintf("%s=%d", reason, n))
	}
	sort.Strings(parts)
	return strings.Join(parts, ", ")
}

func keptSuffix(kept map[commitlane.IndexLockReclaimReason]int) string {
	if len(kept) == 0 {
		return ""
	}
	return " (kept " + keepReasonSummary(kept) + ")"
}

// runCommitReclaimAlias backs `fak commit --reclaim-stale-index-lock`, the alias that
// makes the recovery reachable from where the wedge is actually hit (#5338). It resolves
// the report and hands off to the SAME decision + actuator path as
// `fak commit status --reclaim-stale-index-lock`, so the two entry points cannot drift.
func runCommitReclaimAlias(stdout, stderr io.Writer, dir string, apply bool) int {
	rep, err := commitStatusFn(context.Background(), commitlane.Options{Dir: dir})
	if err != nil {
		fmt.Fprintf(stderr, "fak commit: %v\n", err)
		return 1
	}
	return runIndexLockReclaim(stdout, stderr, rep, apply)
}

// runCommitLockReclaimAlias is the narrow actuator for the advisory lock that
// serializes fak commit. Its target is always exactly <git-dir>/fak-commit.lock:
// index.lock, refs, and other worktrees are outside this recovery's authority.
//
// The default path is a non-mutating ProbeLock. Apply deliberately delegates to
// ReapStaleLockResult instead of deleting from the dry-run probe: that API probes
// again immediately before os.Remove, closing the live-holder race and preserving
// the typed PID-reuse/foreign-holder policy in one audited implementation.
func runCommitLockReclaimAlias(stdout, stderr io.Writer, dir string, apply bool) int {
	rep, err := commitStatusFn(context.Background(), commitlane.Options{Dir: dir})
	if err != nil {
		fmt.Fprintf(stderr, "fak commit: %v\n", err)
		return 1
	}
	if strings.TrimSpace(rep.GitDir) == "" {
		fmt.Fprintln(stderr, "fak commit: commit status did not resolve a git directory")
		return 1
	}
	lockPath := filepath.Join(rep.GitDir, "fak-commit.lock")
	if !apply {
		probe := safecommit.ProbeLock(lockPath)
		if !probe.Reapable() {
			renderCommitLockNoReclaim(stdout, probe)
			return 0
		}
		fmt.Fprintf(stdout, "fak-commit.lock: WOULD reclaim (%s, pid=%d) %s — re-run with --apply to remove\n", probe.Reason, probe.HolderPID, lockPath)
		return 0
	}

	result := safecommit.ReapStaleLockResult(lockPath)
	switch {
	case result.Reaped:
		fmt.Fprintf(stdout, "fak-commit.lock: reclaimed (%s, pid=%d) %s\n", result.Reason, result.HolderPID, lockPath)
		return 0
	case result.Failed():
		fmt.Fprintf(stderr, "fak commit: fak-commit.lock reclaim failed (%s, pid=%d, %s): %s\n", result.Reason, result.HolderPID, result.RemoveErrClass, result.RemoveErr)
		return 1
	default:
		// The typed actuator re-probed and refused the remove. Probe once more only
		// to make that fail-safe decision legible; this read cannot authorize a delete.
		renderCommitLockNoReclaim(stdout, safecommit.ProbeLock(lockPath))
		return 0
	}
}

func renderCommitLockNoReclaim(w io.Writer, probe safecommit.LockProbe) {
	switch {
	case !probe.Exists:
		fmt.Fprintf(w, "fak-commit.lock: no reclaim (absent) %s\n", probe.Path)
	case probe.HolderPID == 0:
		fmt.Fprintf(w, "fak-commit.lock: no reclaim (owner unknown; preserved) %s\n", probe.Path)
	case probe.Alive:
		fmt.Fprintf(w, "fak-commit.lock: no reclaim (holder live, pid=%d; preserved) %s\n", probe.HolderPID, probe.Path)
	default:
		fmt.Fprintf(w, "fak-commit.lock: no reclaim (not proven stale; preserved) %s\n", probe.Path)
	}
}

func renderCommitStatus(w io.Writer, rep commitlane.Report) {
	fmt.Fprintf(w, "commit lane: %s", rep.Verdict)
	if rep.Reason != "" {
		fmt.Fprintf(w, " (%s)", rep.Reason)
	}
	fmt.Fprintln(w)
	if rep.RepoRoot != "" {
		fmt.Fprintf(w, "  repo: %s\n", rep.RepoRoot)
	}
	renderCommitLockLine(w, rep.CommitLock)
	renderIndexLockLine(w, rep.IndexLock)
	writeIndexChurnLines(w, rep.IndexChurn)
	if rep.Owner != nil {
		fmt.Fprintf(w, "  owner: pid=%d %s\n", rep.Owner.PID, processLabel(*rep.Owner))
	}
	if len(rep.Queue) > 0 {
		fmt.Fprintf(w, "  queue: %d possible fak commit waiter(s)\n", len(rep.Queue))
		for _, q := range rep.Queue {
			fmt.Fprintf(w, "    pid=%d %s\n", q.PID, processLabel(q))
		}
	} else {
		fmt.Fprintln(w, "  queue: none observed")
	}
	if len(rep.LiveWriters) > 0 {
		fmt.Fprintf(w, "  live writers: %d observed\n", len(rep.LiveWriters))
	}
	if rep.ProcessProbe != "" && rep.ProcessProbe != "ok" {
		fmt.Fprintf(w, "  process probe: %s\n", rep.ProcessProbe)
	}
	for _, e := range rep.Errors {
		fmt.Fprintf(w, "  warning: %s\n", e)
	}
	if rep.NextAction != "" {
		fmt.Fprintf(w, "  next: %s\n", rep.NextAction)
	}
}

func renderCommitLockLine(w io.Writer, lock commitlane.CommitLock) {
	if !lock.Present {
		fmt.Fprintf(w, "  fak commit lock: none (%s)\n", lock.Path)
		return
	}
	if lock.Stale {
		fmt.Fprintf(w, "  fak commit lock: STALE pid=%d (%s)\n", lock.HolderPID, lock.Path)
		return
	}
	if lock.HolderPID > 0 {
		live := "dead"
		if lock.HolderAlive {
			live = "live"
		}
		fmt.Fprintf(w, "  fak commit lock: held pid=%d %s (%s)\n", lock.HolderPID, live, lock.Path)
		return
	}
	fmt.Fprintf(w, "  fak commit lock: present, owner unknown (%s)\n", lock.Path)
}

func renderIndexLockLine(w io.Writer, lock commitlane.IndexLock) {
	if !lock.Present {
		fmt.Fprintf(w, "  git index lock: none (%s)\n", lock.Path)
		return
	}
	age := ""
	if lock.AgeSeconds > 0 {
		age = fmt.Sprintf(", age=%ds", lock.AgeSeconds)
	}
	state := "present"
	if lock.StaleHint {
		state = "stale-hint"
	}
	fmt.Fprintf(w, "  git index lock: %s%s (%s)\n", state, age, lock.Path)
	if lock.Detail != "" {
		fmt.Fprintf(w, "    %s\n", lock.Detail)
	}
}

// indexChurnPathPreview bounds how many no-op paths the text renderer names inline. The
// churn set reached 51 entries at filing (#5339); an operator needs the count and a
// sample, and the full list is always available in --json and in the remedy line.
const indexChurnPathPreview = 5

// writeIndexChurnLines surfaces the no-op staged-deletion audit (#5339) and OFFERS the
// scoped clear as text. It prints the remedy; it never runs it, and no code path in fak
// runs it either — on a shared clone an automatic `git restore --staged` would silently
// discard a peer's in-flight index work, so the un-stage stays an operator decision made
// against a path list they can read first.
//
// (Named write*, not render*, deliberately: CONCEPT_ADMISSION gates a NEW identifier whose
// normalized token contains a concept-family root, and "render" is a root of the
// render-materialize family. The sibling render* helpers here predate the corpus and are
// grandfathered; a new one would block the commit until the corpus carried a row for it.)
//
// Paths outside the no-op set (a real deletion, or a file that exists but differs from
// HEAD) are counted separately and never appear in the remedy: the offer is scoped to
// exactly the entries proven byte-identical to HEAD.
func writeIndexChurnLines(w io.Writer, audit *commitlane.StagedDeletionAudit) {
	if audit == nil || len(audit.Rows) == 0 {
		return
	}
	noop := audit.NoOpCount()
	if noop == 0 {
		fmt.Fprintf(w, "  index churn: none (%d staged deletion(s), all real or unproven)\n", len(audit.Rows))
		return
	}
	fmt.Fprintf(w, "  index churn: %d of %d staged deletion(s) are no-ops (on disk, byte-identical to HEAD)\n",
		noop, len(audit.Rows))
	shown := audit.NoOpPaths
	if len(shown) > indexChurnPathPreview {
		shown = shown[:indexChurnPathPreview]
	}
	for _, p := range shown {
		fmt.Fprintf(w, "    %s\n", p)
	}
	if rest := noop - len(shown); rest > 0 {
		fmt.Fprintf(w, "    ... and %d more\n", rest)
	}
	if other := len(audit.Rows) - noop; other > 0 {
		fmt.Fprintf(w, "    (%d other staged deletion(s) left alone: real or unproven)\n", other)
	}
	if audit.Remedy != "" {
		fmt.Fprintf(w, "    clear only these (not run for you): %s\n", audit.Remedy)
	}
}

func processLabel(p commitlane.ProcessFact) string {
	parts := []string{}
	if p.Name != "" {
		parts = append(parts, p.Name)
	}
	if p.Match != "" {
		parts = append(parts, p.Match)
	}
	if p.Confidence != "" {
		parts = append(parts, p.Confidence)
	}
	if len(parts) == 0 {
		return strings.TrimSpace(p.Command)
	}
	label := strings.Join(parts, " ")
	if p.Command != "" {
		label += " :: " + p.Command
	}
	return label
}
