package main

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"flag"
	"os"

	"github.com/anthony-chaudhary/fak/internal/flowmetrics"
	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/wipinventory"
)

// sweep.go — `fak sweep`: drive a dirty multi-session working tree TOWARD zero, honestly.
//
// On the always-on shared trunk the working tree accrues dozens of uncommitted paths spanning
// many lanes. `fak commit` lands ONE explicit-path commit; this verb is the layer above it that
// turns "142 dirty paths" into a per-lane PLAN: every stampable change grouped under the
// `(fak <leaf>)` trailer its paths imply, plus the residual a sweep must NOT silently commit
// (stray scratch/log junk, and root-level files with no inferable lane). It reuses the SAME
// path->lane engine the pre-commit lint binds to (internal/hooks.LintCommitMessage) so the
// grouping tracks dos.toml automatically, and the SAME safe-commit discipline (safecommit) so
// an --apply still refuses OFF_TRUNK / a pathspec race / an off-lane stamp.
//
// It deliberately does NOT invent a subject. A sweep cannot know whether a peer's half-finished
// edit is a feat or a fix, so the default mode REPORTS the groups and the operator (or a loop,
// via --json) supplies an ACCURATE subject per lane through `--apply --lane L -m "..."`. That
// keeps the tool from ever authoring an unwitnessed claim about work it did not do.

// runSweep is the `fak sweep` shim. Default: enumerate the dirty tree, group it by lane, and
// REPORT the plan (text, or --json for a loop). With --clean-junk it removes only freshly
// classified junk files. With --apply --lane L -m S it commits exactly lane L's dirty paths
// (optionally narrowed by --path) through the safe-commit path. Exit codes mirror `fak commit`:
// 0 ok, 2 usage, 3 a pre-commit refusal, 1 a raced/failed commit or cleanup failure.
func runSweep(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sweep", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "sweep")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	asJSON := fs.Bool("json", false, "emit the plan as JSON")
	census := fs.Bool("census", false, "render the rev-pinned working-tree WIP census and preview-only candidate paths")
	cleanJunk := fs.Bool("clean-junk", false, "remove only paths freshly classified as junk files, then report what changed")
	autoArchive := fs.Bool("auto-archive", false, "with --clean-junk: snapshot orphan junk into refs/fak/quarantine/* before removing")
	apply := fs.Bool("apply", false, "commit one lane group (requires --lane and -m); default is plan-only")
	lane := fs.String("lane", "", "with --apply: the lane to commit")
	unit := fs.Int("unit", 0, "with --apply: commit only sub-unit N of the lane (from groups[].units[].index; 0 = the whole lane)")
	msg := fs.String("m", "", "with --apply: the commit subject (a `(fak <lane>)` trailer is appended if absent)")
	push := fs.Bool("push", false, "with --apply: push after a VERIFIED commit through the safe sync path (never --force)")
	fs.Bool("s", false, "with --apply: add DCO sign-off (default: true; git-compatible flag)")
	noOrigin := fs.Bool("no-origin", false, "skip the per-path origin/<trunk> relation probe (NEW/AHEAD/ALREADY); faster, but a stale already-shipped duplicate is no longer flagged")
	var only pathList
	fs.Var(&only, "path", "with --apply: restrict the commit to these repo-relative paths (repeatable; default: every dirty path in the lane)")
	if !parseFlags(fs, argv) {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)

	root := resolveRoot(*dir)
	if strings.TrimSpace(root) == "" {
		fmt.Fprintln(stderr, "fak sweep: could not resolve a git repo root (pass --dir)")
		return 2
	}
	if *cleanJunk && *apply {
		fmt.Fprintln(stderr, "fak sweep: --clean-junk and --apply are separate actions; run one at a time")
		return 2
	}
	if *census && (*cleanJunk || *apply) {
		fmt.Fprintln(stderr, "fak sweep: --census is preview-only and cannot be combined with --clean-junk or --apply")
		return 2
	}

	if *census {
		return runSweepCensus(stdout, stderr, root, *asJSON)
	}

	entries, err := gitStatusDirty(ctx(), root)
	if err != nil {
		fmt.Fprintf(stderr, "fak sweep: %v\n", err)
		return 1
	}
	var origin originProbe
	if !*noOrigin {
		origin = originProbeFor(ctx(), root)
	}
	plan := classifyDirty(entries, hooksLaneResolver(root), origin)
	plan.Parked = collectSweepParked(root)

	if *cleanJunk {
		res := cleanSweepJunk(root, plan, *autoArchive)
		if *asJSON {
			if err := writeIndentedJSON(stdout, res); err != nil {
				fmt.Fprintf(stderr, "fak sweep --clean-junk: %v\n", err)
				return 1
			}
		} else {
			renderSweepCleanJunk(stdout, res)
		}
		if !res.OK {
			return 1
		}
		return 0
	}
	if *apply {
		return runSweepApply(stdout, stderr, root, plan, *lane, *msg, only, *unit, *push)
	}
	if *asJSON {
		return encodeJSONOrFailPrefixed(stdout, stderr, plan, "fak sweep")
	}
	renderSweepPlan(stdout, plan)
	return 0
}

type sweepCleanJunkResult struct {
	OK            bool                    `json:"ok"`
	QuarantineRef string                  `json:"quarantine_ref,omitempty"`
	QuarantineSHA string                  `json:"quarantine_sha,omitempty"`
	ArchivedCount int                     `json:"archived_count,omitempty"`
	ArchivedBytes int64                   `json:"archived_bytes,omitempty"`
	Removed       []string                `json:"removed,omitempty"`
	Skipped       []string                `json:"skipped,omitempty"`
	Refused       []sweepCleanJunkRefusal `json:"refused,omitempty"`
	NextAction    string                  `json:"next_action,omitempty"`
}

type sweepCleanJunkRefusal struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
	Detail string `json:"detail,omitempty"`
}

func cleanSweepJunk(root string, plan sweepPlan, autoArchives ...bool) sweepCleanJunkResult {
	autoArchive := len(autoArchives) > 0 && autoArchives[0]
	res := sweepCleanJunkResult{OK: true}
	if len(plan.Junk) == 0 {
		res.NextAction = "no junk paths classified; run `fak sweep --json` to inspect remaining work"
		return res
	}

	if autoArchive {
		var targets []string
		for _, e := range plan.Junk {
			full, ok, reason := sweepCleanPath(root, e.Path)
			if !ok {
				res.Refused = append(res.Refused, sweepCleanJunkRefusal{Path: e.Path, Reason: "unsafe_path", Detail: reason})
				continue
			}
			st, err := os.Lstat(full)
			if err != nil {
				if os.IsNotExist(err) {
					res.Skipped = append(res.Skipped, e.Path)
					continue
				}
				res.Refused = append(res.Refused, sweepCleanJunkRefusal{Path: e.Path, Reason: "stat_failed", Detail: err.Error()})
				continue
			}
			if st.IsDir() {
				res.Refused = append(res.Refused, sweepCleanJunkRefusal{
					Path:   e.Path,
					Reason: "directory_refused",
					Detail: "sweep only deletes junk files; inspect and remove directories by hand",
				})
				continue
			}
			targets = append(targets, e.Path)
		}
		if len(targets) > 0 {
			qref, err := wipinventory.EvictOrphans(context.Background(), root, wipinventory.GitRunner{}, wipinventory.EvictOptions{
				Targets: targets,
				Reason:  "fak sweep --clean-junk --auto-archive",
			})
			if err != nil {
				res.Refused = append(res.Refused, sweepCleanJunkRefusal{
					Path:   "quarantine",
					Reason: "auto_archive_failed",
					Detail: err.Error(),
				})
			} else if qref != nil {
				res.QuarantineRef = qref.Ref
				res.QuarantineSHA = qref.SHA
				res.ArchivedCount = qref.Count
				res.ArchivedBytes = qref.ByteTotal
				res.Removed = qref.Files
			}
		}
		res.OK = len(res.Refused) == 0
		if res.OK {
			res.NextAction = "rerun `fak sweep --json` to inspect remaining work"
		} else {
			res.NextAction = "inspect refused junk paths, then rerun `fak sweep --json`"
		}
		return res
	}

	for _, e := range plan.Junk {
		full, ok, reason := sweepCleanPath(root, e.Path)
		if !ok {
			res.Refused = append(res.Refused, sweepCleanJunkRefusal{Path: e.Path, Reason: "unsafe_path", Detail: reason})
			continue
		}
		st, err := os.Lstat(full)
		if err != nil {
			if os.IsNotExist(err) {
				res.Skipped = append(res.Skipped, e.Path)
				continue
			}
			res.Refused = append(res.Refused, sweepCleanJunkRefusal{Path: e.Path, Reason: "stat_failed", Detail: err.Error()})
			continue
		}
		if st.IsDir() {
			res.Refused = append(res.Refused, sweepCleanJunkRefusal{
				Path:   e.Path,
				Reason: "directory_refused",
				Detail: "sweep only deletes junk files; inspect and remove directories by hand",
			})
			continue
		}
		if err := os.Remove(full); err != nil {
			res.Refused = append(res.Refused, sweepCleanJunkRefusal{Path: e.Path, Reason: "remove_failed", Detail: err.Error()})
			continue
		}
		res.Removed = append(res.Removed, e.Path)
	}
	res.OK = len(res.Refused) == 0
	if res.OK {
		res.NextAction = "rerun `fak sweep --json` to inspect remaining work"
	} else {
		res.NextAction = "inspect refused junk paths, then rerun `fak sweep --json`"
	}
	return res
}

func sweepCleanPath(root, path string) (string, bool, string) {
	norm := normSweepPath(path)
	if strings.TrimSpace(norm) == "" {
		return "", false, "empty path"
	}
	cleanRel := filepath.Clean(filepath.FromSlash(norm))
	if filepath.IsAbs(cleanRel) || cleanRel == "." || cleanRel == ".." || strings.HasPrefix(cleanRel, ".."+string(filepath.Separator)) {
		return "", false, "path escapes the repository"
	}
	full := filepath.Join(root, cleanRel)
	absRoot, rootErr := filepath.Abs(root)
	absFull, fullErr := filepath.Abs(full)
	if rootErr != nil || fullErr != nil {
		return "", false, "could not resolve absolute path"
	}
	rel, err := filepath.Rel(absRoot, absFull)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", false, "path escapes the repository"
	}
	return full, true, ""
}

func renderSweepCleanJunk(w io.Writer, res sweepCleanJunkResult) {
	if res.QuarantineRef != "" {
		fmt.Fprintf(w, "quarantined %d path(s) (%d bytes) to %s (%s)\n", res.ArchivedCount, res.ArchivedBytes, res.QuarantineRef, res.QuarantineSHA)
	}
	if len(res.Removed) == 0 && len(res.Skipped) == 0 && len(res.Refused) == 0 {
		fmt.Fprintln(w, "no junk paths classified")
	} else {
		if len(res.Removed) > 0 {
			fmt.Fprintf(w, "removed %d junk path(s):\n", len(res.Removed))
			for _, p := range res.Removed {
				fmt.Fprintf(w, "  %s\n", p)
			}
		}
		if len(res.Skipped) > 0 {
			fmt.Fprintf(w, "skipped %d already-gone junk path(s):\n", len(res.Skipped))
			for _, p := range res.Skipped {
				fmt.Fprintf(w, "  %s\n", p)
			}
		}
		if len(res.Refused) > 0 {
			fmt.Fprintf(w, "refused %d junk path(s):\n", len(res.Refused))
			for _, r := range res.Refused {
				detail := ""
				if r.Detail != "" {
					detail = " - " + r.Detail
				}
				fmt.Fprintf(w, "  %s  %s%s\n", r.Path, r.Reason, detail)
			}
		}
	}
	if res.NextAction != "" {
		fmt.Fprintf(w, "next: %s\n", res.NextAction)
	}
}

// runSweepApply commits one lane group — or one directory-coherent sub-unit of it — through the
// safe-commit path. It NEVER invents a subject: --lane and -m are both required, so the caller (a
// human or a loop reading --json) always owns the claim. --unit N narrows the commit to sub-unit N
// of the lane's split (0 = the whole lane); it composes with --path, which then narrows within the
// selected unit. The `(fak <lane>)` trailer is appended when absent, the message is pre-linted (the
// shared trunk has no amend), and safecommit verifies only the requested paths landed.
func runSweepApply(stdout, stderr io.Writer, root string, plan sweepPlan, lane, msg string, only []string, unit int, push bool) int {
	lane = strings.TrimSpace(lane)
	if lane == "" || strings.TrimSpace(msg) == "" {
		fmt.Fprintln(stderr, "fak sweep --apply: --lane L and -m SUBJECT are both required (a sweep never invents a subject for peer work)")
		return 2
	}

	var group *sweepGroup
	for i := range plan.Groups {
		if plan.Groups[i].Lane == lane {
			group = &plan.Groups[i]
			break
		}
	}
	if group == nil {
		fmt.Fprintf(stderr, "fak sweep --apply: no dirty, stampable paths in lane %q\n", lane)
		// A verdict on the lane, not contention (#5505 W4): re-running finds the same
		// empty lane, so exit 4 tells a loop to replan rather than back off.
		return safecommit.ExitRefused
	}

	// --unit N selects one sub-unit of the lane's split BEFORE the --path narrowing. It resolves
	// against the freshly re-derived plan, so a lane that no longer splits (dropped to <= the
	// atomic-unit target since the plan was read) correctly refuses rather than committing a stale
	// slice. An out-of-range or not-applicable unit is a usage error (2), like a missing -m.
	paths := group.Paths
	if unit > 0 {
		if len(group.Units) == 0 {
			fmt.Fprintf(stderr, "fak sweep --apply: lane %q is not split into sub-units (it fits one commit; drop --unit)\n", lane)
			return 2
		}
		var picked *sweepSubUnit
		for i := range group.Units {
			if group.Units[i].Index == unit {
				picked = &group.Units[i]
				break
			}
		}
		if picked == nil {
			fmt.Fprintf(stderr, "fak sweep --apply: lane %q has no sub-unit %d (valid: 1..%d)\n", lane, unit, len(group.Units))
			return 2
		}
		paths = picked.Paths
	}
	if len(only) > 0 {
		paths = intersectPaths(paths, only)
		if len(paths) == 0 {
			fmt.Fprintf(stderr, "fak sweep --apply: none of the --path values are dirty stampable paths in lane %q\n", lane)
			return safecommit.ExitRefused
		}
	}

	message := ensureTrailer(msg, lane)
	// Pre-lint so a bad subject / off-lane stamp is caught BEFORE the commit lands (a sibling
	// may push your local commit first, so there is no amend on the shared trunk).
	rep := hooks.LintCommitMessage(message, paths, root)
	if !rep.OK {
		fmt.Fprintln(stderr, "fak sweep --apply: refused — the subject/stamp did not pass preview:")
		renderPreview(stderr, rep, safecommit.ExpectedTrunk(root, ""))
		return safecommit.ExitRefused
	}

	buildCheckOutcome, buildCheckDetail := safecommit.BuildCheckDisabled, ""
	if os.Getenv("FAK_COMMIT_BUILD_CHECK") != "off" {
		buildCheckOutcome, buildCheckDetail = commitBuildCheckGate(stderr, root, paths)
	}
	buildCheck, admitBuild, buildReason := safecommit.DecideBuildCheck(buildCheckOutcome, buildCheckDetail, os.Getenv("FAK_COMMIT_BUILD_CHECK") == "allow-timeout")
	if !admitBuild {
		fmt.Fprintf(stderr, "fak sweep --apply: %s\n", buildReason)
		if d := strings.TrimSpace(buildCheck.Detail); d != "" {
			fmt.Fprintln(stderr, d)
		}
		fmt.Fprintln(stderr, commitBuildCheckAdvice(buildReason))
		return safecommit.ExitRefused
	}

	res, err := commitFn(ctx(), safecommit.Options{
		Dir:     root,
		Paths:   paths,
		Message: message,
		SignOff: true,
		Push:    push,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak sweep --apply: %v\n", err)
		return 1
	}
	res.BuildCheck = &buildCheck
	renderCommitResult(stdout, res)
	return commitExitCode(res)
}

// ensureTrailer appends a `(fak <lane>)` trailer to the subject line when none is present, so an
// operator/loop need not retype the stamp the lane already implies. A subject that already carries
// any `(fak ...)` / `fak/<leaf>:` stamp is left untouched (the lint then catches a mismatch).
func ensureTrailer(msg, lane string) string {
	if kind, _ := hooks.StampOf(firstCommitLine(msg)); kind == "trailer" || kind == "direct" {
		return msg
	}
	lines := strings.SplitN(msg, "\n", 2)
	lines[0] = strings.TrimRight(lines[0], " ") + " (fak " + lane + ")"
	return strings.Join(lines, "\n")
}

// gitStatusDirty starts with porcelain status for staged and untracked records, then reconciles
// unstaged tracked records against Git's content-aware diff. On Windows, status can retain hundreds
// of stat-cache/CRLF ghosts whose normalized blob is unchanged; presenting those as WIP makes the
// commit queue both slower and misleading.
func gitStatusDirty(ctx context.Context, root string) ([]dirtyEntry, error) {
	out, err := runSweepGit(ctx, root, "status", "status", "--porcelain=v1", "-z", "--no-renames")
	if err != nil {
		return nil, err
	}
	entries := parsePorcelainZ(out)
	if !hasWorktreeTracked(entries) {
		return annotateDirtyAges(root, entries, time.Now()), nil
	}

	diffOut, err := runSweepGit(ctx, root, "diff", "-c", "core.safecrlf=false", "diff", "--name-only", "-z", "--no-renames", "--")
	if err != nil {
		return nil, err
	}
	return annotateDirtyAges(root, filterContentDirty(entries, parseNULPaths(diffOut)), time.Now()), nil
}

func runSweepGit(ctx context.Context, root, operation string, args ...string) (string, error) {
	out, code, err := gitRunner(ctx, root, args...)
	if err != nil {
		return "", err
	}
	if code != 0 {
		return "", fmt.Errorf("git %s exited %d: %s", operation, code, strings.TrimSpace(out))
	}
	return out, nil
}

func hasWorktreeTracked(entries []dirtyEntry) bool {
	for _, entry := range entries {
		if entry.WorktreeDirty && !entry.Untracked {
			return true
		}
	}
	return false
}

func parseNULPaths(out string) map[string]struct{} {
	paths := make(map[string]struct{})
	for _, path := range strings.Split(out, "\x00") {
		if path != "" {
			paths[normSweepPath(path)] = struct{}{}
		}
	}
	return paths
}

func filterContentDirty(entries []dirtyEntry, unstaged map[string]struct{}) []dirtyEntry {
	kept := make([]dirtyEntry, 0, len(entries))
	for _, entry := range entries {
		_, contentDirty := unstaged[normSweepPath(entry.Path)]
		if entry.Untracked || entry.IndexDirty || !entry.WorktreeDirty || contentDirty {
			kept = append(kept, entry)
		}
	}
	return kept
}

func annotateDirtyAges(root string, entries []dirtyEntry, now time.Time) []dirtyEntry {
	for i := range entries {
		p := filepath.Join(root, filepath.FromSlash(normSweepPath(entries[i].Path)))
		st, err := os.Stat(p)
		if err != nil {
			continue
		}
		mtime := st.ModTime().Unix()
		entries[i].MTime = mtime
		if age := int64(now.Sub(st.ModTime()).Seconds()); age > 0 {
			entries[i].AgeSeconds = age
		}
	}
	return entries
}

// hooksLaneResolver derives a path's lane through the SAME engine the pre-commit lint binds to:
// LintCommitMessage computes PathLanes for the given paths off dos.toml, so a single-path call
// yields that path's lane (or "" when none can be inferred).
func hooksLaneResolver(root string) laneResolver {
	return func(path string) string {
		rep := hooks.LintCommitMessage("", []string{path}, root)
		if len(rep.PathLanes) == 0 {
			return ""
		}
		return rep.PathLanes[0]
	}
}

// originProbeFor builds the real origin probe: it places each dirty path against
// origin/<development_branch> so the sweep can flag an already-shipped stale duplicate before an
// agent wastes a commit (or a scarce lock-window) re-landing an unchanged file. The trunk name is
// the same branch-role contract every other verb reads (safecommit.ExpectedTrunk), remote-qualified
// as "origin/<trunk>".
//
// The upstream ref is resolved ONCE up front. If it does not resolve — a fresh clone whose
// installer never fetched a remote-tracking ref — the probe returns originUnknown for every path
// (fail-open ABSTAIN, the same posture safecommit's stale-base guard takes on a missing ref), so a
// sweep on such a clone behaves exactly as before. Each probe call is object-DB only
// (rev-parse/hash-object) — it never touches .git/index.lock, so it is safe under the commit-storm
// contention that makes `git status` itself block.
func originProbeFor(ctx context.Context, root string) originProbe {
	ref := "origin/" + safecommit.ExpectedTrunk(root, "")
	// Resolve the tip once; a non-resolving ref means we cannot compare — abstain for everything.
	if _, code, err := gitRunner(ctx, root, "rev-parse", "--verify", "--quiet", ref); err != nil || code != 0 {
		return func(string) originRelation { return originUnknown }
	}
	return func(path string) originRelation {
		// Upstream blob OID at ref:path. A non-zero exit means the path does not exist upstream.
		upstream, code, err := gitRunner(ctx, root, "rev-parse", ref+":"+path)
		if err != nil || code != 0 {
			return originNew
		}
		upstream = strings.TrimSpace(upstream)
		if upstream == "" {
			return originNew
		}
		// Working-tree blob OID. hash-object needs the file to exist; a staged/working deletion has
		// no file to hash, so it is a real tree change vs origin (ahead), never "already".
		wt, code, err := gitRunner(ctx, root, "hash-object", path)
		if err != nil || code != 0 {
			return originAhead
		}
		if strings.TrimSpace(wt) == upstream {
			return originAlready
		}
		return originAhead
	}
}

func renderSweepPlan(w io.Writer, plan sweepPlan) {
	if plan.TotalDirty == 0 {
		fmt.Fprintln(w, "working tree is clean — nothing to sweep")
		writeSweepParkedText(w, plan.Parked)
		return
	}
	fmt.Fprintf(w, "dirty paths: %d  (%d stampable across %d lane(s), %d no-lane, %d junk)\n",
		plan.TotalDirty, stampableCount(plan), len(plan.Groups), len(plan.NoLane), len(plan.Junk))
	if plan.OldestDirtyPath != "" {
		fmt.Fprintf(w, "oldest dirty: %s at %s\n", sweepAgeLabel(plan.OldestDirtyAgeSeconds), plan.OldestDirtyPath)
	}
	writeSweepParkedText(w, plan.Parked)
	if plan.NextAction != "" {
		fmt.Fprintf(w, "next: %s\n", plan.NextAction)
	}

	if len(plan.Groups) > 0 {
		fmt.Fprintln(w, "\nstampable lane groups — commit each with an ACCURATE subject:")
		for _, g := range plan.Groups {
			already := map[string]bool{}
			for _, p := range g.AlreadyShipped {
				already[p] = true
			}
			fmt.Fprintf(w, "\n  lane %-12s score %3d  %s  (%d path(s))\n", g.Lane, g.Score, g.Trailer, len(g.Paths))
			if g.OldestDirtyPath != "" {
				fmt.Fprintf(w, "    oldest: %s at %s\n", sweepAgeLabel(g.OldestDirtyAgeSeconds), g.OldestDirtyPath)
			}
			if g.AllAlready {
				// The whole lane is byte-identical to the trunk: nothing to commit. This is the line
				// that turns a multi-probe investigation into one glance.
				fmt.Fprintf(w, "    ALREADY on origin — all %d path(s) match the trunk; discard the working copies, nothing to ship\n", len(g.Paths))
			} else if len(g.AlreadyShipped) > 0 {
				fmt.Fprintf(w, "    note: %d of %d path(s) ALREADY match origin (marked below) — a commit would not change them\n", len(g.AlreadyShipped), len(g.Paths))
			}
			if len(g.ScoreReasons) > 0 {
				fmt.Fprintf(w, "    score notes: %s\n", strings.Join(g.ScoreReasons, "; "))
			}
			for _, p := range g.Paths {
				tag := ""
				if already[p] {
					tag = "  [ALREADY on origin]"
				}
				fmt.Fprintf(w, "    %s%s\n", p, tag)
			}
			if !g.AllAlready {
				fmt.Fprintf(w, "    -> fak sweep --apply --lane %s -m \"<type>(%s): <verb> <what>\" [--push]\n", g.Lane, g.Lane)
			}
			// A too-large lane also gets a directory-coherent sub-unit plan: commit these one at a
			// time so the change lands in small, reviewable units instead of one blob. The whole-lane
			// hint above still stands — sub-units are the RECOMMENDED path, not the only one.
			if len(g.Units) > 0 && !g.AllAlready {
				fmt.Fprintf(w, "    LARGE lane (%d paths) — commit in %d directory-coherent sub-unit(s):\n", len(g.Paths), len(g.Units))
				for _, u := range g.Units {
					dir := u.Dir
					if dir == "" {
						dir = "(repo root)"
					}
					fmt.Fprintf(w, "      unit %d  dir %-24s score %3d  (%d path(s))\n", u.Index, dir, u.Score, len(u.Paths))
					if u.OldestDirtyPath != "" {
						fmt.Fprintf(w, "        oldest: %s at %s\n", sweepAgeLabel(u.OldestDirtyAgeSeconds), u.OldestDirtyPath)
					}
					for _, p := range u.Paths {
						tag := ""
						if already[p] {
							tag = "  [ALREADY on origin]"
						}
						fmt.Fprintf(w, "        %s%s\n", p, tag)
					}
					fmt.Fprintf(w, "        -> fak sweep --apply --lane %s --unit %d -m \"<type>(%s): <verb> <what>\" [--push]\n", g.Lane, u.Index, g.Lane)
				}
			}
		}
	}
	if len(plan.NoLane) > 0 {
		fmt.Fprintln(w, "\nno-lane (root-level; no lane could be inferred — pick a stamp by hand with fak commit):")
		for _, e := range plan.NoLane {
			fmt.Fprintf(w, "  %-2s %s\n", e.Status, e.Path)
		}
	}
	if len(plan.Junk) > 0 {
		fmt.Fprintln(w, "\njunk (stray scratch/log output — SURFACED, never committed; remove if you own it):")
		for _, e := range plan.Junk {
			fmt.Fprintf(w, "  %-2s %s\n", e.Status, e.Path)
		}
	}
}

func sweepAgeLabel(seconds int64) string {
	switch {
	case seconds <= 0:
		return "now"
	case seconds < 60:
		return fmt.Sprintf("%ds", seconds)
	case seconds < 3600:
		return fmt.Sprintf("%dm", seconds/60)
	case seconds < 86400:
		return fmt.Sprintf("%dh", seconds/3600)
	default:
		return fmt.Sprintf("%dd", seconds/86400)
	}
}

// sweepCensusResult models the structured JSON payload for `fak sweep --census --json`.
type sweepCensusResult struct {
	Schema        string              `json:"schema"`
	Rev           string              `json:"rev"`
	Census        flowmetrics.TreeWIP `json:"census"`
	LitterPaths   []string            `json:"litter_paths"`
	UnlandedPaths []string            `json:"unlanded_paths"`
	RemovedCount  int                 `json:"removed_count"`
}

// runSweepCensus implements the preview-only rev-pinned working tree WIP census and candidate classification.
func runSweepCensus(stdout, stderr io.Writer, root string, asJSON bool) int {
	treeWIP, err := flowmetrics.GatherTree(context.Background(), root, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak sweep --census: %v\n", err)
		return 1
	}

	out, err := runSweepGit(context.Background(), root, "status", "status", "--porcelain", "-z")
	if err != nil {
		fmt.Fprintf(stderr, "fak sweep --census: %v\n", err)
		return 1
	}

	entries := parsePorcelainZ(out)
	litter, unlanded := classifyCandidatePaths(entries)

	if asJSON {
		res := sweepCensusResult{
			Schema:        "fak-sweep-census/1",
			Rev:           treeWIP.Rev,
			Census:        treeWIP,
			LitterPaths:   litter,
			UnlandedPaths: unlanded,
			RemovedCount:  0,
		}
		if err := writeIndentedJSON(stdout, res); err != nil {
			fmt.Fprintf(stderr, "fak sweep --census --json: %v\n", err)
			return 1
		}
		return 0
	}

	renderSweepCensus(stdout, treeWIP, litter, unlanded)
	return 0
}

func classifyCandidatePaths(entries []dirtyEntry) (litter []string, unlanded []string) {
	for _, e := range entries {
		norm := normSweepPath(e.Path)
		if isCandidateLitter(e) {
			litter = append(litter, norm)
		} else {
			unlanded = append(unlanded, norm)
		}
	}
	return litter, unlanded
}

func isCandidateLitter(e dirtyEntry) bool {
	norm := normSweepPath(e.Path)
	base := filepath.Base(filepath.FromSlash(norm))
	rootLevel := !strings.Contains(norm, "/")

	// Scratch probes zz_-prefixed
	if strings.HasPrefix(base, "zz_") {
		return true
	}
	// Hidden dot-prefixed .go
	if strings.HasPrefix(base, ".") && strings.HasSuffix(base, ".go") {
		return true
	}
	// Root throwaway files
	if rootLevel && isSweepJunk(e) {
		return true
	}
	return false
}

func formatCensusVerdict(count, ceiling int) string {
	if count <= ceiling {
		return "PASS"
	}
	return "DEFECT"
}

func formatUntrackedAge(hours float64) string {
	if hours <= 0 {
		return "0.0h"
	}
	if hours < 24 {
		return fmt.Sprintf("%.1fh", hours)
	}
	days := hours / 24.0
	return fmt.Sprintf("%.1fd (%.1fh)", days, hours)
}

func renderSweepCensus(w io.Writer, tree flowmetrics.TreeWIP, litter, unlanded []string) {
	fmt.Fprintln(w, "Rev-pinned working tree census:")
	if !tree.Measured {
		fmt.Fprintln(w, "  Working-tree WIP: NOT MEASURED")
	} else {
		fmt.Fprintf(w, "  HEAD rev: %s\n", tree.Rev)
		fmt.Fprintf(w, "  Untracked source files: %d (ceiling: %d) -> %s\n",
			tree.UntrackedGo, flowmetrics.UntrackedGoCeiling, formatCensusVerdict(tree.UntrackedGo, flowmetrics.UntrackedGoCeiling))
		fmt.Fprintf(w, "  Scratch probe files: %d (ceiling: %d) -> %s\n",
			tree.ScratchLitter, flowmetrics.ScratchLitterCeiling, formatCensusVerdict(tree.ScratchLitter, flowmetrics.ScratchLitterCeiling))
		fmt.Fprintf(w, "  Recent writers (last 10m): %d (ceiling: %d) -> %s\n",
			tree.RecentWriters, flowmetrics.RecentWritersCeiling, formatCensusVerdict(tree.RecentWriters, flowmetrics.RecentWritersCeiling))
		fmt.Fprintf(w, "  Modified source files: %d\n", tree.ModifiedGo)
		fmt.Fprintf(w, "  Added/Deleted lines churn: +%d/-%d\n", tree.AddedLines, tree.DeletedLines)
		fmt.Fprintf(w, "  Oldest untracked file age: %s\n", formatUntrackedAge(tree.OldestUntrackedHours))
		if tree.StatFailures > 0 {
			fmt.Fprintf(w, "  Stat failures: %d file(s) could not be stat'd (mtimes and ages understated)\n", tree.StatFailures)
		}
	}

	fmt.Fprintln(w, "\nCandidate paths preview:")
	fmt.Fprintf(w, "  Litter candidate paths (%d):\n", len(litter))
	for _, p := range litter {
		fmt.Fprintf(w, "    %s\n", p)
	}
	fmt.Fprintf(w, "  Unlanded candidate paths (%d):\n", len(unlanded))
	for _, p := range unlanded {
		fmt.Fprintf(w, "    %s\n", p)
	}
	fmt.Fprintln(w, "\nPreview only: removed nothing. Run with --clean-junk or explicit fak commit to actuate.")
}
