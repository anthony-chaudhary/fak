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
	dryRun := fs.Bool("dry-run", false, "preview a merge without touching the index/worktree")
	dir := fs.String("dir", "", "repo directory (default: discover from cwd)")
	target := fs.String("target", "", "ref to preview merging into HEAD (default: origin/<trunk>)")
	trunk := fs.String("trunk", "main", "trunk branch name used for the default target")
	asJSON := fs.Bool("json", false, "emit the preview as JSON")
	if err := fs.Parse(argv); err != nil {
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
	if !*dryRun {
		fmt.Fprintln(stderr, "fak merge: only --dry-run is supported; refusing to run a mutating merge")
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
	res, err := mergepreview.Preview(context.Background(), root, *target, mergepreview.RealRunner)
	if err != nil {
		fmt.Fprintf(stderr, "fak merge: %v\n", err)
		return 1
	}
	if *asJSON {
		if encErr := writeIndentedJSON(stdout, res); encErr != nil {
			fmt.Fprintf(stderr, "fak merge: %v\n", encErr)
			return 1
		}
	} else {
		renderMergePreview(stdout, res)
	}
	if res.Outcome == mergepreview.OutcomeConflicts {
		return 3
	}
	return 0
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
