package main

import (
	"context"
	"fmt"
	"io"
	"strings"

	"flag"
	"os"

	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
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

func cmdSweep(argv []string) { os.Exit(runSweep(os.Stdout, os.Stderr, argv)) }

// runSweep is the `fak sweep` shim. Default: enumerate the dirty tree, group it by lane, and
// REPORT the plan (text, or --json for a loop). With --apply --lane L -m S it commits exactly
// lane L's dirty paths (optionally narrowed by --path) through the safe-commit path. Exit codes
// mirror `fak commit`: 0 ok, 2 usage, 3 a pre-commit refusal, 1 a raced/failed commit.
func runSweep(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sweep", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	asJSON := fs.Bool("json", false, "emit the plan as JSON")
	apply := fs.Bool("apply", false, "commit one lane group (requires --lane and -m); default is plan-only")
	lane := fs.String("lane", "", "with --apply: the lane to commit")
	msg := fs.String("m", "", "with --apply: the commit subject (a `(fak <lane>)` trailer is appended if absent)")
	push := fs.Bool("push", false, "with --apply: push after a VERIFIED commit (plain push, never --force)")
	var only pathList
	fs.Var(&only, "path", "with --apply: restrict the commit to these repo-relative paths (repeatable; default: every dirty path in the lane)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)

	root := resolveRoot(*dir)
	if strings.TrimSpace(root) == "" {
		fmt.Fprintln(stderr, "fak sweep: could not resolve a git repo root (pass --dir)")
		return 2
	}

	entries, err := gitStatusDirty(ctx(), root)
	if err != nil {
		fmt.Fprintf(stderr, "fak sweep: %v\n", err)
		return 1
	}
	plan := classifyDirty(entries, hooksLaneResolver(root))

	if *apply {
		return runSweepApply(stdout, stderr, root, plan, *lane, *msg, only, *push)
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, plan); err != nil {
			fmt.Fprintf(stderr, "fak sweep: %v\n", err)
			return 1
		}
		return 0
	}
	renderSweepPlan(stdout, plan)
	return 0
}

// runSweepApply commits one lane group through the safe-commit path. It NEVER invents a subject:
// --lane and -m are both required, so the caller (a human or a loop reading --json) always owns
// the claim. The `(fak <lane>)` trailer is appended when absent, the message is pre-linted (the
// shared trunk has no amend), and safecommit verifies only the requested paths landed.
func runSweepApply(stdout, stderr io.Writer, root string, plan sweepPlan, lane, msg string, only []string, push bool) int {
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
		return 3
	}

	paths := group.Paths
	if len(only) > 0 {
		paths = intersectPaths(group.Paths, only)
		if len(paths) == 0 {
			fmt.Fprintf(stderr, "fak sweep --apply: none of the --path values are dirty stampable paths in lane %q\n", lane)
			return 3
		}
	}

	message := ensureTrailer(msg, lane)
	// Pre-lint so a bad subject / off-lane stamp is caught BEFORE the commit lands (a sibling
	// may push your local commit first, so there is no amend on the shared trunk).
	rep := hooks.LintCommitMessage(message, paths, root)
	if !rep.OK {
		fmt.Fprintln(stderr, "fak sweep --apply: refused — the subject/stamp did not pass preview:")
		renderPreview(stderr, rep, safecommit.ExpectedTrunk(root, ""))
		return 3
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

// gitStatusDirty runs `git status --porcelain=v1 -z --no-renames` and parses the dirty entries.
// --no-renames keeps every record a single NUL-terminated "XY PATH" so the parse is unambiguous.
func gitStatusDirty(ctx context.Context, root string) ([]dirtyEntry, error) {
	out, code, err := gitRunner(ctx, root, "status", "--porcelain=v1", "-z", "--no-renames")
	if err != nil {
		return nil, err
	}
	if code != 0 {
		return nil, fmt.Errorf("git status exited %d: %s", code, strings.TrimSpace(out))
	}
	return parsePorcelainZ(out), nil
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

func renderSweepPlan(w io.Writer, plan sweepPlan) {
	if plan.TotalDirty == 0 {
		fmt.Fprintln(w, "working tree is clean — nothing to sweep")
		return
	}
	fmt.Fprintf(w, "dirty paths: %d  (%d stampable across %d lane(s), %d no-lane, %d junk)\n",
		plan.TotalDirty, stampableCount(plan), len(plan.Groups), len(plan.NoLane), len(plan.Junk))

	if len(plan.Groups) > 0 {
		fmt.Fprintln(w, "\nstampable lane groups — commit each with an ACCURATE subject:")
		for _, g := range plan.Groups {
			fmt.Fprintf(w, "\n  lane %-12s score %3d  %s  (%d path(s))\n", g.Lane, g.Score, g.Trailer, len(g.Paths))
			if len(g.ScoreReasons) > 0 {
				fmt.Fprintf(w, "    score notes: %s\n", strings.Join(g.ScoreReasons, "; "))
			}
			for _, p := range g.Paths {
				fmt.Fprintf(w, "    %s\n", p)
			}
			fmt.Fprintf(w, "    -> fak sweep --apply --lane %s -m \"<type>(%s): <verb> <what>\" [--push]\n", g.Lane, g.Lane)
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
