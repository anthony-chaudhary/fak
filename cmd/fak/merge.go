package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mergepreview"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdMerge(argv []string) { os.Exit(runMerge(os.Stdout, os.Stderr, argv)) }

func runMerge(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("merge", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "merge")
	dryRun := fs.Bool("dry-run", false, "preview a merge without touching the index/worktree")
	apply := fs.Bool("apply", false, "auto-resolve the trivial superset case (merged tree == HEAD) with a textless `-s ours` merge; defer any real conflict to hand-merge")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	target := fs.String("target", "", "ref to preview merging into HEAD (default: origin/<trunk>)")
	trunk := fs.String("trunk", "main", "trunk branch name used for the default target")
	message := fs.String("message", "", "commit message for an --apply textless merge (default: derived from target)")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	args := fs.Args()
	if len(args) > 1 {
		fmt.Fprintln(stderr, "fak merge: at most one positional target ref is allowed")
		return 2
	}
	if len(args) == 1 {
		if strings.TrimSpace(*target) != "" {
			fmt.Fprintln(stderr, "fak merge: use either --target or a positional target, not both")
			return 2
		}
		*target = args[0]
	}
	if *dryRun && *apply {
		fmt.Fprintln(stderr, "fak merge: --dry-run and --apply are mutually exclusive")
		return 2
	}
	if !*dryRun && !*apply {
		fmt.Fprintln(stderr, "fak merge: pass --dry-run to preview or --apply to auto-resolve the trivial superset case; refusing a bare mutating merge")
		return 2
	}
	if strings.TrimSpace(*target) == "" {
		*target = "origin/" + strings.TrimSpace(*trunk)
	}
	root := resolveRoot(pathutil.ExpandTilde(*dir))
	if root == "" {
		fmt.Fprintln(stderr, "fak merge: could not resolve git repo root")
		return 2
	}
	if *apply {
		return runMergeApply(stdout, stderr, root, *target, *message, *asJSON)
	}
	res, err := mergepreview.Preview(context.Background(), root, *target, mergepreview.RealRunner)
	if err != nil {
		fmt.Fprintf(stderr, "fak merge: %v\n", err)
		return 1
	}
	if code := emitJSONOrRenderPrefixed(stdout, stderr, "fak merge", *asJSON, res, func(w io.Writer) {
		renderMergePreview(w, res)
	}); code != 0 {
		return code
	}
	if res.Outcome == mergepreview.OutcomeConflicts {
		return 3
	}
	return 0
}

// runMergeApply auto-resolves the trivial superset case (#2154): when the 3-way merge tree
// already equals HEAD it commits a textless `-s ours` merge and asserts tree == HEAD; a genuine
// conflict (or a clean merge that changes the tree) mutates nothing and defers to the agent.
func runMergeApply(stdout, stderr io.Writer, root, target, message string, asJSON bool) int {
	res, err := mergepreview.Apply(context.Background(), root, target, message, mergepreview.RealRunner)
	if err != nil {
		fmt.Fprintf(stderr, "fak merge --apply: %v\n", err)
		return 1
	}
	if asJSON {
		if encErr := writeIndentedJSON(stdout, res); encErr != nil {
			fmt.Fprintf(stderr, "fak merge --apply: %v\n", encErr)
			return 1
		}
	} else {
		renderMergeApply(stdout, res)
	}
	switch res.ApplyOutcome {
	case mergepreview.ApplyResolvedSuperset:
		return 0
	case mergepreview.ApplyDeferredConflict:
		return 3 // genuine conflict — agent must hand-merge (mirrors dry-run's conflict exit)
	default:
		return 0 // deferred_clean: informational, agent runs a plain git merge
	}
}

func renderMergeApply(w io.Writer, res mergepreview.ApplyResult) {
	fmt.Fprintf(w, "fak merge --apply: %s\n", res.ApplyOutcome)
	switch res.ApplyOutcome {
	case mergepreview.ApplyResolvedSuperset:
		fmt.Fprintf(w, "  %s\n", res.Detail)
		fmt.Fprintf(w, "  merge commit %s\n", res.MergeCommit)
	default:
		fmt.Fprintf(w, "  %s\n", res.Detail)
	}
}

func renderMergePreview(w io.Writer, res mergepreview.Result) {
	fmt.Fprintf(w, "fak merge --dry-run: %s\n", res.Outcome)
	switch res.Outcome {
	case mergepreview.OutcomeEmptyNetDiff:
		fmt.Fprintln(w, "  resolves cleanly; cached diff will be empty")
	case mergepreview.OutcomeConflicts:
		fmt.Fprintf(w, "  %d conflict(s) predicted before touching the index/worktree\n", len(res.Conflicts))
		for _, p := range res.Conflicts {
			fmt.Fprintf(w, "  - %s\n", p)
		}
	case mergepreview.OutcomeCleanMerge:
		fmt.Fprintf(w, "  resolves cleanly; cached diff would change %d file(s)\n", len(res.ChangedFiles))
		for _, p := range res.ChangedFiles {
			fmt.Fprintf(w, "  - %s\n", p)
		}
	default:
		fmt.Fprintf(w, "  %s\n", res.Detail)
	}
}
