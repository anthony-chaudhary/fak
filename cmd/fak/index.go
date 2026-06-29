package main

// fak index — the QUERYABLE SELF-INDEX (epic #1287 / C1 #1288). Instead of
// re-surveying dos.toml and INDEX.md every session, an agent ASKS:
//
//	fak index lane <path>...   which lane/leaf owns these paths (+ the commit stamp)
//	fak index leaf [<query>]   the lane taxonomy, filtered by name/tree/description
//	fak index docs <query>     the curated doc map, ranked by relevance
//
// It is a thin shell over internal/devindex, which reads the facts live from the
// files that already own them (dos.toml's [lanes.trees], the curated INDEX.md), so
// the index is a VIEW, never a competing source of truth. --json on every
// subcommand makes the same answers machine- and MCP-consumable.

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func cmdIndex(argv []string) { os.Exit(runIndex(os.Stdout, os.Stderr, argv)) }

// laneAnswer is one path's resolved lane + the ship-stamp it implies (JSON shape).
type laneAnswer struct {
	Path  string `json:"path"`
	Lane  string `json:"lane"`
	Stamp string `json:"stamp,omitempty"`
}

func runIndex(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		writeIndexUsage(stderr)
		return 2
	}
	sub := argv[0]
	fs := flag.NewFlagSet("index "+sub, flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: search upward for dos.toml)")
	asJSON := fs.Bool("json", false, "emit the answer as JSON")
	limit := fs.Int("limit", 0, "cap the number of results (0 = all)")
	// Parse flags that may appear ANYWHERE around the positional query (the natural
	// `fak index leaf cache --limit 6` order), not just before it. Go's flag package
	// stops at the first non-flag arg, so interleave Parse with positional collection.
	var args []string
	for rest := argv[1:]; ; {
		if err := fs.Parse(rest); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return 0
			}
			return 2
		}
		rest = fs.Args()
		if len(rest) == 0 {
			break
		}
		args = append(args, rest[0])
		rest = rest[1:]
	}

	rootDir := *root
	if rootDir == "" {
		rootDir = devindex.FindRoot(".")
	}
	cat, err := devindex.Load(rootDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak index: %v\n", err)
		return 1
	}

	switch sub {
	case "lane":
		return indexLane(stdout, stderr, cat, args, *asJSON)
	case "leaf", "leaves":
		return indexLeaf(stdout, stderr, cat, args, *asJSON, *limit)
	case "docs", "doc":
		return indexDocs(stdout, stderr, cat, args, *asJSON, *limit)
	default:
		fmt.Fprintf(stderr, "fak index: unknown subcommand %q\n", sub)
		writeIndexUsage(stderr)
		return 2
	}
}

func indexLane(stdout, stderr io.Writer, cat *devindex.Catalog, paths []string, asJSON bool) int {
	if len(paths) == 0 {
		fmt.Fprintln(stderr, "fak index lane: needs at least one path")
		return 2
	}
	answers := make([]laneAnswer, 0, len(paths))
	for _, p := range paths {
		lane := cat.LaneForPath(p)
		answers = append(answers, laneAnswer{Path: p, Lane: lane, Stamp: cat.SuggestStamp(p)})
	}
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, answers, "fak index lane")
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, a := range answers {
		lane := a.Lane
		if lane == "" {
			lane = "(no lane — root file?)"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", a.Path, lane, a.Stamp)
	}
	return flushTab(tw, stderr, "fak index lane")
}

func indexLeaf(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	hits := capLeaves(cat.SearchLeaves(joinArgs(args)), limit)
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, hits, "fak index leaf")
	}
	if len(hits) == 0 {
		fmt.Fprintln(stdout, "no matching leaf")
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, l := range hits {
		mark := "ok"
		if !l.Exists {
			mark = "MISSING"
		}
		desc := l.Desc
		if desc == "" {
			desc = l.Tree
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\n", l.Name, mark, desc)
	}
	return flushTab(tw, stderr, "fak index leaf")
}

func indexDocs(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "fak index docs: needs a search query")
		return 2
	}
	hits := capDocs(cat.SearchDocs(joinArgs(args)), limit)
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, hits, "fak index docs")
	}
	if len(hits) == 0 {
		fmt.Fprintln(stdout, "no matching doc")
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, d := range hits {
		fmt.Fprintf(tw, "%s\t%s\n", d.Path, d.Title)
	}
	return flushTab(tw, stderr, "fak index docs")
}

func writeIndexUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak index lane <path>...    which lane/leaf owns each path, + the (fak <leaf>) commit stamp
  fak index leaf [<query>]    the lane taxonomy, filtered by name/tree/description
  fak index docs <query>      the curated doc map (INDEX.md), ranked by relevance
  flags: --json  --limit N  --root DIR
`)
}

func joinArgs(args []string) string {
	out := ""
	for i, a := range args {
		if i > 0 {
			out += " "
		}
		out += a
	}
	return out
}

func capLeaves(ls []devindex.Leaf, limit int) []devindex.Leaf {
	if limit > 0 && len(ls) > limit {
		return ls[:limit]
	}
	return ls
}

func capDocs(ds []devindex.Doc, limit int) []devindex.Doc {
	if limit > 0 && len(ds) > limit {
		return ds[:limit]
	}
	return ds
}

func flushTab(tw *tabwriter.Writer, stderr io.Writer, label string) int {
	if err := tw.Flush(); err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", label, err)
		return 1
	}
	return 0
}
