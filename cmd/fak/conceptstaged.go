package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog"
)

// runConceptCLI is the `fak concept` entry point.
//
// It splits the tree-scoped (`--staged`) modes out of runConcept because they answer a
// different question. runConcept's verbs read the WORKTREE; --staged reads the same git
// tree the CONCEPT_FRESHNESS pre-commit gate scores, which under a pathspec commit is
// HEAD plus the committer's pathspec and nothing else. Those two roots must not be
// merged into one switch that treats them as interchangeable: handing the worktree to a
// tree-scoped path is exactly the peer-dirty failure conceptcatalog.CheckGitTree exists
// to prevent, and it is the failure that made the gate's printed cure unreachable
// (#5829).
func runConceptCLI(stdout, stderr io.Writer, args []string) int {
	if len(args) == 0 || !conceptStagedRequested(args[1:]) {
		return runConcept(stdout, stderr, args)
	}
	switch args[0] {
	case "generate":
		return runConceptGenerateStaged(stdout, stderr, args[1:])
	case "freshness":
		return runConceptFreshnessStaged(stdout, stderr, args[1:])
	default:
		fmt.Fprintf(stderr, "fak concept %s: --staged is supported by `generate` and `freshness` only\n", args[0])
		return 2
	}
}

// conceptStagedRequested reports whether the caller asked for tree mode. Detected before
// flag parsing so the two modes can own separate flag sets; an explicit --staged=false is
// a request for worktree mode and must fall through to runConcept.
func conceptStagedRequested(args []string) bool {
	for _, a := range args {
		if a == "--" {
			return false
		}
		switch a {
		case "-staged", "--staged", "-staged=true", "--staged=true":
			return true
		}
	}
	return false
}

// runConceptGenerateStaged is the command CONCEPT_FRESHNESS prints when it refuses. It
// regenerates inside a clean-room export of the scored tree and writes the artifacts into
// the worktree; it deliberately does NOT stage them, because on a shared trunk the
// operator names their own pathspec.
func runConceptGenerateStaged(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("concept generate --staged", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Bool("staged", false, "regenerate from the staged git tree instead of the worktree")
	tree := fs.String("tree", "", "treeish to regenerate from (default: the current index)")
	jsonOut := fs.Bool("json", false, "emit JSON")
	if fs.Parse(args) != nil {
		return 2
	}
	root, err := conceptRoot()
	if err != nil {
		fmt.Fprintln(stderr, "fak concept generate:", err)
		return 1
	}
	written, err := conceptcatalog.RegenerateFromGitTree(root, *tree, root)
	if err != nil {
		fmt.Fprintln(stderr, "fak concept generate --staged:", err)
		return 1
	}
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(map[string]any{"ok": true, "mode": "staged", "tree": *tree, "written": written})
		return 0
	}
	fmt.Fprintln(stdout, "regenerated from the staged git tree:")
	for _, p := range written {
		fmt.Fprintln(stdout, " -", p)
	}
	fmt.Fprintln(stdout, "stage these paths in the SAME commit; this is the tree CONCEPT_FRESHNESS scores")
	return 0
}

// runConceptFreshnessStaged answers the question the gate asks, so an operator can check
// a pathspec before paying for a refusal.
func runConceptFreshnessStaged(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("concept freshness --staged", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Bool("staged", false, "check the staged git tree instead of the worktree")
	tree := fs.String("tree", "", "treeish to check (default: the current index)")
	jsonOut := fs.Bool("json", false, "emit JSON")
	if fs.Parse(args) != nil {
		return 2
	}
	root, err := conceptRoot()
	if err != nil {
		fmt.Fprintln(stderr, "fak concept freshness:", err)
		return 1
	}
	res, err := conceptcatalog.CheckGitTree(root, *tree)
	if *jsonOut {
		_ = json.NewEncoder(stdout).Encode(res)
	}
	if err != nil {
		fmt.Fprintln(stderr, "fak concept freshness --staged:", err)
		return 1
	}
	if !res.Fresh {
		if !*jsonOut {
			fmt.Fprintln(stderr, "stale generated concept artifacts in the staged tree:")
			for _, p := range res.StalePaths {
				fmt.Fprintln(stderr, " -", p)
			}
			fmt.Fprintln(stderr, "regenerate:", res.Regenerate)
		}
		return 1
	}
	if !*jsonOut {
		fmt.Fprintln(stdout, "concept generated artifacts fresh in the staged tree")
	}
	return 0
}
