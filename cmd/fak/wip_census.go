package main

// wip_census.go — the READ-ONLY `fak wip reap --census` path (#5340). It classifies
// every refs/fak/wip/* ref into an owner-state census and reports the counts (plus a
// --json per-ref breakdown) WITHOUT deleting anything and WITHOUT touching the reap
// delete path. Every git command it runs is read-only: for-each-ref (via
// wipListRecords), rev-parse/diff (via wipOwnerState + the delta read), and
// `git show HEAD:<file>` for the subsumption oracle. It NEVER calls `git update-ref -d`.
//
// The classification facts are gathered here (git I/O); the pure fold lives in
// internal/wipref (census.go), mirroring how reap keeps its decision pure.

import (
	"context"
	"fmt"
	"io"
	"runtime"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// runWipCensus renders the census. Non-JSON prints the per-class counts; --json emits
// the full report (counts + every per-ref verdict). Read-only: exit 0 on success, 1 on
// a git error. It deletes nothing regardless of --dry-run (the census has no mutate mode).
func runWipCensus(ctx context.Context, stdout, stderr io.Writer, repo string, asJSON bool) int {
	report, err := wipCensus(ctx, repo)
	if err != nil {
		fmt.Fprintf(stderr, "fak wip reap --census: %v\n", err)
		return 1
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, report, "fak wip reap --census")
	}
	c := report.Counts
	fmt.Fprintf(stdout, "wip checkpoint census over %d ref(s) (read-only — nothing deleted):\n", c.Total)
	fmt.Fprintf(stdout, "  LANDED                    %6d  delta already in HEAD (today's reap collects these)\n", c.Landed)
	fmt.Fprintf(stdout, "  LIVE                      %6d  owning session still live (kept)\n", c.Live)
	fmt.Fprintf(stdout, "  CLOSED_CLEAN_ESTIMATE    %6d  dead session, every payload file byte-identical to HEAD\n", c.ClosedCleanEstimate)
	fmt.Fprintf(stdout, "  CLOSED_DIRTY_RECOVERABLE  %6d  recoverable unlanded deliverable (ACTION REQUIRED; kept)\n", c.ClosedDirtyRecoverable)
	fmt.Fprintf(stdout, "  UNKNOWN                   %6d  unresolved (kept, fail-safe)\n", c.Unknown)
	fmt.Fprintf(stdout, "  safely reapable (LANDED only): %d\n", c.Landed)
	if c.ClosedDirtyRecoverable > 0 {
		fmt.Fprintf(stdout, "  recovery backlog (not reapable): %d unlanded deliverable(s) require reconcile --reclaim\n", c.ClosedDirtyRecoverable)
	}
	return 0
}

// wipCensus reads every live checkpoint ref, resolves each into census facts from git,
// and folds them into the pure report. The per-ref classification is git-spawn-bound
// (a couple of diffs plus HEAD reads each), so over thousands of refs it runs a bounded
// pool of workers — the difference between a few minutes and tens of minutes on a busy
// shared host. HEAD's per-file line sets are memoized across refs in a shared,
// concurrency-safe memo (HEAD is fixed for the whole pass), so a hot file touched by
// hundreds of dead sessions is read from HEAD once, not once per ref. Verdicts are
// written by index (race-free) and the pure BuildCensus re-sorts them, so the report is
// deterministic regardless of completion order.
func wipCensus(ctx context.Context, repo string) (wipref.CensusReport, error) {
	recs, err := wipListRecords(ctx, repo)
	if err != nil {
		return wipref.CensusReport{}, err
	}
	live, err := wipLiveSessions(ctx, repo)
	if err != nil {
		return wipref.CensusReport{}, err
	}

	memo := newWipHeadMemo()
	verdicts := make([]wipref.CensusVerdict, len(recs))
	var (
		wg       sync.WaitGroup
		errMu    sync.Mutex
		firstErr error
	)
	jobs := make(chan int)
	for w := 0; w < wipCensusWorkers(); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				facts, ferr := wipCensusFacts(ctx, repo, recs[i], live, memo)
				if ferr != nil {
					errMu.Lock()
					if firstErr == nil {
						firstErr = ferr
					}
					errMu.Unlock()
					continue
				}
				payload := wipref.PayloadCensus{}
				if !facts.Landed && !facts.Live {
					payload = wipref.BuildPayloadCensus(wipPayloadReading(ctx, repo, recs[i]))
					read, files, absent, diverged := payload.Facts()
					facts.PayloadRead, facts.PayloadFiles, facts.PayloadAbsent, facts.PayloadDiverged = read, files, absent, diverged
				}
				class := wipref.Classify(facts)
				verdicts[i] = wipref.CensusVerdict{
					Session: wipSessionOf(recs[i]),
					Ref:     recs[i].Ref,
					Object:  recs[i].Object,
					Class:   class,
					Reason:  wipref.CensusReason(class),
				}.WithPayload(payload)
			}
		}()
	}
	for i := range recs {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	if firstErr != nil {
		return wipref.CensusReport{}, firstErr
	}
	return wipref.BuildCensus(verdicts), nil
}

// wipPayloadReading measures every payload file with checked Git plumbing. The
// path list comes from the checkpoint metadata or is derived from the commit's
// delta / tree when unscoped, so parentless and descendant refs are both measured
// rather than mistaken for empty. The two-dot tree diff works for both
// parentless and descendant checkpoints.
func wipPayloadReading(ctx context.Context, repo string, rec wipref.RefRecord) wipref.PayloadReading {
	paths := rec.Stamp.Scope
	if len(paths) == 0 {
		_, _, code, err := gitWip(ctx, repo, nil, "rev-parse", "--verify", "-q", rec.Object+"^")
		if err != nil {
			return wipref.Unread("verify parent: %v", err)
		}
		var out string
		if code == 0 {
			o, errStr, dcode, derr := gitWip(ctx, repo, nil, "diff", "--name-only", "-z", rec.Object+"^", rec.Object)
			if derr != nil {
				return wipref.Unread("git diff --name-only: %v", derr)
			}
			if dcode != 0 {
				return wipref.Unread("git diff --name-only exited %d: %s", dcode, strings.TrimSpace(errStr))
			}
			out = o
		} else {
			o, errStr, lcode, lerr := gitWip(ctx, repo, nil, "ls-tree", "-r", "--name-only", "-z", rec.Object)
			if lerr != nil {
				return wipref.Unread("git ls-tree: %v", lerr)
			}
			if lcode != 0 {
				return wipref.Unread("git ls-tree exited %d: %s", lcode, strings.TrimSpace(errStr))
			}
			out = o
		}
		var derivedPaths []string
		for _, p := range strings.Split(out, "\x00") {
			if p != "" {
				derivedPaths = append(derivedPaths, p)
			}
		}
		if len(derivedPaths) == 0 {
			return wipref.PayloadReading{Read: true}
		}
		paths = derivedPaths
	}
	base := "HEAD"
	if _, _, code, err := gitWip(ctx, repo, nil, "rev-parse", "--verify", "HEAD^{tree}"); err != nil {
		return wipref.Unread("resolve HEAD tree: %v", err)
	} else if code != 0 {
		// An unborn repository has no HEAD commit, but a parentless checkpoint still
		// has a real tree. Compare it with Git's own empty-tree object rather than
		// treating the missing branch name as an empty payload.
		empty, errStr, emptyCode, emptyErr := gitWipStdin(ctx, repo, "", "mktree")
		if emptyErr != nil {
			return wipref.Unread("create empty tree: %v", emptyErr)
		}
		if emptyCode != 0 {
			return wipref.Unread("create empty tree exited %d: %s", emptyCode, strings.TrimSpace(errStr))
		}
		base = strings.TrimSpace(empty)
	}
	args := []string{"diff", "--name-status", "--no-renames", "-z", base, rec.Object, "--"}
	args = append(args, paths...)
	out, errStr, code, err := gitWip(ctx, repo, nil, args...)
	if err != nil {
		return wipref.Unread("git diff --name-status: %v", err)
	}
	if code != 0 {
		return wipref.Unread("git diff --name-status exited %d: %s", code, strings.TrimSpace(errStr))
	}
	return wipref.PayloadReading{Read: true, Paths: paths, NameStatus: out}
}

// wipCensusWorkers sizes the census worker pool. The work is git-subprocess-bound, not
// CPU-bound, so it oversubscribes cores modestly, clamped to a sane band so a busy
// shared host is not swamped with concurrent git spawns.
func wipCensusWorkers() int {
	n := runtime.NumCPU() * 2
	if n < 4 {
		n = 4
	}
	if n > 16 {
		n = 16
	}
	return n
}

// wipHeadMemo memoizes HEAD's per-file line sets across an entire census pass and is
// safe for the worker pool to share. The git read runs OUTSIDE the lock — so two workers
// racing on the same hot file may read it twice (harmless, idempotent) — but the lock is
// never held across a subprocess, which would serialize the pool it exists to parallelize.
type wipHeadMemo struct {
	mu    sync.Mutex
	lines map[string]map[string]bool
}

func newWipHeadMemo() *wipHeadMemo {
	return &wipHeadMemo{lines: map[string]map[string]bool{}}
}

// lineSet returns HEAD's trimmed non-blank line set for path, reading it once and
// memoizing it. A lost race (a peer memoized it while we read) reuses the peer's set.
func (c *wipHeadMemo) lineSet(ctx context.Context, repo, path string) map[string]bool {
	c.mu.Lock()
	if s, ok := c.lines[path]; ok {
		c.mu.Unlock()
		return s
	}
	c.mu.Unlock()
	set := wipHeadFileLineSet(ctx, repo, path)
	c.mu.Lock()
	if s, ok := c.lines[path]; ok {
		set = s
	} else {
		c.lines[path] = set
	}
	c.mu.Unlock()
	return set
}

// wipCensusFacts resolves one ref into its census facts, short-circuiting on the
// cheapest positive signal first. Liveness is decided first from the SAME live-lease
// set wipReconcile uses (no git per ref). A verbatim landing reuses wipOwnerState — the
// EXACT resolver the reap delete path keys on — so the census's LANDED bucket is
// precisely the set today's `fak wip reap` already collects. Only a dead, unlanded ref
// pays for the delta read + subsumption oracle. It mutates nothing.
func wipCensusFacts(ctx context.Context, repo string, rec wipref.RefRecord, live map[string]bool, headMemo *wipHeadMemo) (wipref.CensusFacts, error) {
	var f wipref.CensusFacts
	if live[wipSessionOf(rec)] {
		f.Live = true
		return f, nil
	}
	// LANDED: reuse the reap delete-path resolver verbatim, so LANDED == what reap deletes.
	st, err := wipOwnerState(ctx, repo, rec)
	if err != nil {
		return f, err
	}
	if st == wipref.OwnerLanded {
		f.Landed = true
		return f, nil
	}
	// Dead + not landed: read the delta ONCE and split empty / subsumed / recoverable.
	patch, _, code, err := gitWip(ctx, repo, nil, "diff", rec.Object+"^", rec.Object)
	if err != nil || code != 0 {
		// No reachable parent / undiffable: leave Resolved=false -> UNKNOWN (fail-safe keep).
		return f, nil
	}
	f.Resolved = true
	if strings.TrimSpace(patch) == "" {
		f.DeltaEmpty = true
		return f, nil
	}
	added, removed := wipParseDeltaLines(patch)
	head := wipHeadLineSets(ctx, repo, headMemo, added, removed)
	f.Subsumed = wipref.DeltaSubsumed(added, removed, head)
	return f, nil
}

// wipParseDeltaLines parses a unified `git diff <obj>^ <obj>` into per-file added and
// removed CONTENT lines (trimmed; blank lines dropped), keyed by the b-side path for
// additions and the a-side path for removals. Attributing removals to the a-side path
// is what keeps the safety invariant honest: a wholly-DELETED file's removed lines are
// checked against the path that still holds them in HEAD (so an unlanded deletion is
// never called subsumed), and a brand-NEW file's added lines are keyed on a path HEAD
// does not have (so they are never counted present). Pure string work.
func wipParseDeltaLines(patch string) (added, removed map[string][]string) {
	added, removed = map[string][]string{}, map[string][]string{}
	var aFile, bFile string
	for _, line := range strings.Split(patch, "\n") {
		switch {
		case strings.HasPrefix(line, "diff --git "):
			aFile, bFile = "", ""
		case strings.HasPrefix(line, "--- "):
			aFile = wipDiffHeaderPath(line, "--- ", "a/")
		case strings.HasPrefix(line, "+++ "):
			bFile = wipDiffHeaderPath(line, "+++ ", "b/")
		case strings.HasPrefix(line, "+"):
			if s := strings.TrimSpace(line[1:]); s != "" && bFile != "" {
				added[bFile] = append(added[bFile], s)
			}
		case strings.HasPrefix(line, "-"):
			if s := strings.TrimSpace(line[1:]); s != "" && aFile != "" {
				removed[aFile] = append(removed[aFile], s)
			}
		}
	}
	return added, removed
}

// wipDiffHeaderPath extracts the path from a '--- a/<path>' / '+++ b/<path>' header,
// stripping the side prefix git adds and any '\t' timestamp suffix. '/dev/null' (the
// create/delete sentinel) yields "" so its lines are attributed to no file.
func wipDiffHeaderPath(line, prefix, side string) string {
	rest := strings.TrimSpace(strings.TrimPrefix(line, prefix))
	if i := strings.IndexByte(rest, '\t'); i >= 0 {
		rest = rest[:i]
	}
	if rest == "/dev/null" {
		return ""
	}
	return strings.TrimPrefix(rest, side)
}

// wipHeadLineSets returns, for every path the delta touches, the set of trimmed
// non-blank lines in HEAD's current version of that path — the subsumption oracle. The
// per-path result is memoized across refs (HEAD is fixed for the whole census
// pass), so a file many dead sessions touched is read from HEAD once. A path HEAD does
// not have memoizes an empty set, so a new file's added lines are never subsumed.
func wipHeadLineSets(ctx context.Context, repo string, memo *wipHeadMemo, added, removed map[string][]string) map[string]map[string]bool {
	out := map[string]map[string]bool{}
	want := func(path string) {
		if path == "" {
			return
		}
		if _, done := out[path]; done {
			return
		}
		out[path] = memo.lineSet(ctx, repo, path)
	}
	for p := range added {
		want(p)
	}
	for p := range removed {
		want(p)
	}
	return out
}

// wipHeadFileLineSet reads HEAD's version of path (`git show HEAD:<path>`, read-only)
// into the set of its trimmed non-blank lines. A path absent from HEAD (a deleted or
// never-committed file) yields an empty set — not an error — so the caller treats its
// added lines as un-subsumed.
func wipHeadFileLineSet(ctx context.Context, repo, path string) map[string]bool {
	set := map[string]bool{}
	content, _, code, err := gitWip(ctx, repo, nil, "show", "HEAD:"+path)
	if err != nil || code != 0 {
		return set
	}
	for _, ln := range strings.Split(content, "\n") {
		if s := strings.TrimSpace(ln); s != "" {
			set[s] = true
		}
	}
	return set
}
