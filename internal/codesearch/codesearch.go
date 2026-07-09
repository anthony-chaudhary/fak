// Package codesearch composes the epic #3434 code-intelligence primitives into
// one user-facing engine — the wiring that turns four otherwise-orphan libraries
// into a tool someone can actually run:
//
//   - grep/lit  -> internal/trigram   (trigram-accelerated regex + literal search)
//   - ast       -> internal/astquery  (Go AST shape query with $VAR metavariables)
//   - calls/callers -> internal/codegraph (forward/reverse call-graph BFS)
//   - feature   -> internal/selfquery (feature-card retrieval with RRF fusion)
//
// It is the backing logic for `fak dev codesearch`; Run is a thin io-injected
// entry so it can be dogfooded against the real tree in a test without building
// the whole fak binary.
package codesearch

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/astquery"
	"github.com/anthony-chaudhary/fak/internal/codegraph"
	"github.com/anthony-chaudhary/fak/internal/selfquery"
	"github.com/anthony-chaudhary/fak/internal/trigram"
)

// Usage is the one-screen help for the verb.
const Usage = `usage: fak dev codesearch <sub> [--root DIR] [--limit N] <query>
  grep <regexp>    regex search over the Go tree (trigram-accelerated)
  lit  <literal>   literal substring search over the Go tree
  ast  <pattern>   Go AST shape query with $VAR metavariables, e.g. '$_.Close()'
  calls <Name>     functions/methods reachable from Name (forward call graph)
  callers <Name>   functions/methods that (transitively) reach Name
  feature <query>  rank fak's own feature cards (RRF fusion of BM25 + simhash)`

// Run executes one codesearch sub-command, writing to stdout/stderr and returning
// a process exit code (0 ok, 2 usage error, 1 runtime error).
func Run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, Usage)
		return 2
	}
	sub := args[0]
	fs := flag.NewFlagSet("codesearch "+sub, flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "root directory to search")
	limit := fs.Int("limit", 40, "maximum results to print")
	if err := fs.Parse(args[1:]); err != nil {
		return 2
	}
	rest := fs.Args()

	switch sub {
	case "grep", "lit", "ast":
		if len(rest) == 0 {
			fmt.Fprintf(stderr, "usage: fak dev codesearch %s <pattern>\n", sub)
			return 2
		}
		query := strings.Join(rest, " ")
		files, content := collectGoFiles(*root)
		if len(files) == 0 {
			fmt.Fprintf(stderr, "no .go files under %q\n", *root)
			return 1
		}
		switch sub {
		case "grep":
			ix := indexOf(files, content)
			res, err := ix.SearchRegexp(query)
			if err != nil {
				fmt.Fprintf(stderr, "bad regexp: %v\n", err)
				return 2
			}
			return printResults(stdout, res, *limit)
		case "lit":
			ix := indexOf(files, content)
			return printResults(stdout, ix.Search(query), *limit)
		default: // ast
			return runAST(stdout, stderr, files, content, query, *limit)
		}
	case "calls", "callers":
		if len(rest) == 0 {
			fmt.Fprintf(stderr, "usage: fak dev codesearch %s <FuncOrMethodName>\n", sub)
			return 2
		}
		return runGraph(stdout, stderr, *root, sub, rest[0], *limit)
	case "feature":
		if len(rest) == 0 {
			fmt.Fprintf(stderr, "usage: fak dev codesearch feature <query>\n")
			return 2
		}
		return runFeature(stdout, stderr, *root, strings.Join(rest, " "), *limit)
	default:
		fmt.Fprintln(stderr, Usage)
		return 2
	}
}

func indexOf(files []string, content map[string]string) *trigram.Index {
	ix := &trigram.Index{}
	for _, f := range files {
		ix.Add(f, f, content[f])
	}
	return ix
}

func printResults(stdout io.Writer, res []trigram.Result, limit int) int {
	if len(res) == 0 {
		fmt.Fprintln(stdout, "(no matches)")
		return 0
	}
	sort.Slice(res, func(i, j int) bool { return res[i].Path < res[j].Path })
	for i, r := range res {
		if i >= limit {
			fmt.Fprintf(stdout, "... and %d more files\n", len(res)-limit)
			break
		}
		fmt.Fprintf(stdout, "%s  (lines %s)\n", r.Path, lineList(r.Lines))
	}
	return 0
}

func lineList(lines []int) string {
	if len(lines) > 6 {
		lines = lines[:6]
	}
	parts := make([]string, len(lines))
	for i, l := range lines {
		parts[i] = fmt.Sprintf("%d", l)
	}
	return strings.Join(parts, ",")
}

func runAST(stdout, stderr io.Writer, files []string, content map[string]string, pattern string, limit int) int {
	n := 0
	for _, f := range files {
		ms, err := astquery.Search(content[f], pattern)
		if err != nil {
			fmt.Fprintf(stderr, "bad pattern: %v\n", err)
			return 2
		}
		for _, m := range ms {
			if n >= limit {
				fmt.Fprintln(stdout, "... (limit reached)")
				return 0
			}
			fmt.Fprintf(stdout, "%s:%d  %s\n", f, m.Pos.Line, m.Text)
			n++
		}
	}
	if n == 0 {
		fmt.Fprintln(stdout, "(no matches)")
	}
	return 0
}

func runGraph(stdout, stderr io.Writer, root, sub, name string, limit int) int {
	files, content := collectGoFiles(root)
	srcs := make([]string, 0, len(files))
	for _, f := range files {
		srcs = append(srcs, content[f])
	}
	g, err := codegraph.BuildCallGraphFiles(srcs...)
	if err != nil {
		fmt.Fprintf(stderr, "parse error: %v\n", err)
		return 1
	}
	ids := g.NodesByName(name)
	if len(ids) == 0 {
		fmt.Fprintf(stdout, "no function or method named %q under %q\n", name, root)
		return 0
	}
	for _, id := range ids {
		var hits []codegraph.Hit
		rel := "reachable from"
		if sub == "callers" {
			hits, rel = g.Dependents(id), "callers of"
		} else {
			hits = g.Reaches(id)
		}
		fmt.Fprintf(stdout, "%s (%d %s):\n", id, len(hits), rel)
		for i, h := range hits {
			if i >= limit {
				fmt.Fprintf(stdout, "  ... and %d more\n", len(hits)-limit)
				break
			}
			fmt.Fprintf(stdout, "  %s  (hop %d)\n", h.ID, h.Dist)
		}
	}
	return 0
}

func runFeature(stdout, stderr io.Writer, root, query string, limit int) int {
	cat, err := selfquery.Load(root, selfquery.Options{})
	if err != nil {
		fmt.Fprintf(stderr, "load feature catalog from %q: %v\n", root, err)
		return 1
	}
	// The RRF-fusion arm (#3436) — the reachable home for HybridRRF, which the
	// production ranker does not yet call.
	ranked := selfquery.HybridRRF(cat.Cards(selfquery.PlaneAll), query)
	if len(ranked) == 0 {
		fmt.Fprintln(stdout, "(no matching feature cards)")
		return 0
	}
	for i, c := range ranked {
		if i >= limit {
			break
		}
		fmt.Fprintf(stdout, "  %-14s %s\n", c.Kind, c.Name)
	}
	return 0
}

// collectGoFiles walks root for Go source, skipping dot/underscore/vendor dirs, and
// returns the sorted file list plus a path->content map.
func collectGoFiles(root string) ([]string, map[string]string) {
	var files []string
	content := map[string]string{}
	filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			if p != root && (strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") || name == "vendor" || name == "node_modules") {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(p, ".go") {
			if b, err := os.ReadFile(p); err == nil {
				sp := filepath.ToSlash(p)
				files = append(files, sp)
				content[sp] = string(b)
			}
		}
		return nil
	})
	sort.Strings(files)
	return files, content
}
