package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/clonescan"
)

// cmdDup is `fak dup` — the AUTHORING-TIME dedup query. Where the code-slop
// scorecard grades the whole tree a cycle after code lands (as debt), `fak dup
// query` inverts the same normalized-token clone engine into a forward question:
// given a candidate Go block, which tracked sites already hold a token-similar
// block? Run it BEFORE writing a new helper, so a clone is prevented instead of
// counted later. See docs/notes/DEDUP-EARLIER-AND-MORE-OFTEN-2026-07-03.md.
//
//	query --file F [--json] [--k N]   — sites in the tracked tree similar to F's blocks
//	query --stdin [--json] [--k N]    — same, reading the candidate block from stdin
func cmdDup(args []string) {
	if len(args) == 0 {
		dupUsage()
		os.Exit(2)
	}
	switch args[0] {
	case "query":
		cmdDupQuery(args[1:])
	case "-h", "--help", "help":
		dupUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak dup: unknown subcommand %q\n", args[0])
		dupUsage()
		os.Exit(2)
	}
}

func dupUsage() {
	fmt.Fprintln(os.Stderr, "usage: fak dup query --file <candidate.go> [--k 5] [--json]   (tracked sites similar to the candidate)")
	fmt.Fprintln(os.Stderr, "       fak dup query --stdin [--k 5] [--json]                  (read the candidate block from stdin)")
	fmt.Fprintln(os.Stderr, "")
	fmt.Fprintln(os.Stderr, "Ask \"does a block like this already exist?\" BEFORE writing it. Same normalized")
	fmt.Fprintln(os.Stderr, "token-window clone definition as the code-slop scorecard, run as a forward query.")
}

// cmdDupQuery answers the query against the git-tracked .go tree.
func cmdDupQuery(args []string) {
	fs := flag.NewFlagSet("dup query", flag.ExitOnError)
	file := fs.String("file", "", "candidate Go file to check against the tracked tree")
	stdin := fs.Bool("stdin", false, "read the candidate block from stdin instead of --file")
	k := fs.Int("k", 5, "how many matching sites to return (0 = all)")
	asJSON := fs.Bool("json", false, "emit matches as JSON")
	_ = fs.Parse(args)

	if (*file == "") == (!*stdin) {
		fmt.Fprintln(os.Stderr, "fak dup query: pass exactly one of --file F or --stdin")
		os.Exit(2)
	}

	var candidate string
	var selfPath string
	if *stdin {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak dup query: read stdin: %v\n", err)
			os.Exit(1)
		}
		candidate = string(b)
	} else {
		b, err := os.ReadFile(*file)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak dup query: %v\n", err)
			os.Exit(1)
		}
		candidate = string(b)
		// If the candidate file is itself tracked, exclude its own path so it is
		// not reported as a duplicate of itself.
		selfPath = trackedRelPath(*file)
	}

	tree, err := trackedGoTree()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak dup query: %v\n", err)
		os.Exit(1)
	}

	matches := clonescan.Query(candidate, tree, selfPath, *k)
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(matches)
		return
	}
	if len(matches) == 0 {
		fmt.Println("no token-similar block found in the tracked tree — clear to write")
		return
	}
	fmt.Printf("%d tracked site(s) already hold a token-similar block (most overlap first):\n", len(matches))
	for _, m := range matches {
		fmt.Printf("  %-3d windows  %s:%d-%d\n", m.Windows, m.File, m.StartLine, m.EndLine)
	}
	fmt.Println("\nreview these before adding the block — a shared helper may already exist.")
}

// trackedGoTree returns the git-tracked *.go files as rel-path -> source text.
func trackedGoTree() (map[string]string, error) {
	out, err := exec.Command("git", "ls-files", "*.go").Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files: %w", err)
	}
	tree := make(map[string]string)
	for _, rel := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		rel = strings.TrimSpace(rel)
		if rel == "" {
			continue
		}
		b, err := os.ReadFile(rel)
		if err != nil {
			continue // a tracked-but-deleted file; skip
		}
		tree[filepath.ToSlash(rel)] = string(b)
	}
	return tree, nil
}

// trackedRelPath returns the slash-form repo-relative path of a file if it is
// inside the repo, else "" (an untracked candidate has no self to exclude).
func trackedRelPath(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		return ""
	}
	root, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		return ""
	}
	rel, err := filepath.Rel(strings.TrimSpace(string(root)), abs)
	if err != nil || strings.HasPrefix(rel, "..") {
		return ""
	}
	return filepath.ToSlash(rel)
}
