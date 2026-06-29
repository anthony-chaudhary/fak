package main

// fak index — the QUERYABLE SELF-INDEX (epic #1287 / C1 #1288). Instead of
// re-surveying dos.toml and INDEX.md every session, an agent ASKS:
//
//	fak index lane <path>...   which lane/leaf owns these paths (+ the commit stamp)
//	fak index leaf [<query>]   the lane taxonomy, filtered by name/tree/description
//	fak index docs <query>     the curated doc map, ranked by relevance
//	fak index claims <query>   the CLAIMS.md honesty ledger: shipped/simulated/stub
//	fak index verbs [<query>]  the structured CLI-verb catalog (name/lane/synopsis)
//
// It is a thin shell over internal/devindex, which reads the facts live from the
// files that already own them (dos.toml's [lanes.trees], the curated INDEX.md, the
// CLAIMS.md ledger), so the index is a VIEW, never a competing source of truth.
// --json on every subcommand makes the same answers machine- and MCP-consumable.

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
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
	case "claims", "claim":
		return indexClaims(stdout, stderr, cat, args, *asJSON, *limit)
	case "verbs", "verb":
		return indexVerbs(stdout, stderr, cat, args, *asJSON, *limit)
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

// indexRenderHits is the shared post-search rendering for the index subcommands (leaf,
// claims, docs): emit JSON when asked, the "no matching ..." line when the result set is
// empty, else a tab-aligned table whose rows renderRow writes. cmdLabel labels the
// JSON-encode/flush errors; emptyMsg is the no-results line.
func indexRenderHits[T any](stdout, stderr io.Writer, hits []T, asJSON bool, cmdLabel, emptyMsg string, renderRow func(tw *tabwriter.Writer, row T)) int {
	if asJSON {
		return encodeJSONOrFail(stdout, stderr, hits, cmdLabel)
	}
	if len(hits) == 0 {
		fmt.Fprintln(stdout, emptyMsg)
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	for _, row := range hits {
		renderRow(tw, row)
	}
	return flushTab(tw, stderr, cmdLabel)
}

func indexLeaf(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	hits := capLeaves(cat.SearchLeaves(joinArgs(args)), limit)
	return indexRenderHits(stdout, stderr, hits, asJSON, "fak index leaf", "no matching leaf",
		func(tw *tabwriter.Writer, l devindex.Leaf) {
			mark := "ok"
			if !l.Exists {
				mark = "MISSING"
			}
			desc := l.Desc
			if desc == "" {
				desc = l.Tree
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\t%s\n", l.Name, mark, statusBadge(l.Status), desc)
		})
}

// statusBadge renders a leaf's CLAIMS.md maturity rollup as a compact, model- and
// human-readable cell — "[4 shipped · 1 stub]" — or "" when the ledger names no
// capability for the leaf (so the column stays empty rather than noisy).
func statusBadge(s devindex.Status) string {
	if s.Total() == 0 {
		return ""
	}
	var parts []string
	if s.Shipped > 0 {
		parts = append(parts, fmt.Sprintf("%d shipped", s.Shipped))
	}
	if s.Simulated > 0 {
		parts = append(parts, fmt.Sprintf("%d sim", s.Simulated))
	}
	if s.Stub > 0 {
		parts = append(parts, fmt.Sprintf("%d stub", s.Stub))
	}
	return "[" + strings.Join(parts, " · ") + "]"
}

func indexClaims(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "fak index claims: needs a search query (a lane, a token, or a capability)")
		return 2
	}
	hits := capClaims(cat.SearchClaims(joinArgs(args)), limit)
	return indexRenderHits(stdout, stderr, hits, asJSON, "fak index claims", "no matching claim",
		func(tw *tabwriter.Writer, cl devindex.Claim) {
			lanes := strings.Join(cl.Lanes, ",")
			if lanes == "" {
				lanes = "-"
			}
			fmt.Fprintf(tw, "%s\t%s\t%s\n", cl.Tag, lanes, truncRunes(cl.Text, 96))
		})
}

// indexVerbs answers `fak index verbs [<query>]` from the structured C3 verb manifest
// (#1290) — the parseable replacement for grepping usage.go's freeform prose. An empty
// query lists the whole catalog (the SearchVerbs convention), matching `fak index leaf`.
func indexVerbs(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	hits := capVerbs(cat.SearchVerbs(joinArgs(args)), limit)
	return indexRenderHits(stdout, stderr, hits, asJSON, "fak index verbs", "no matching verb",
		func(tw *tabwriter.Writer, v devindex.Verb) {
			lane := v.Lane
			if lane == "" {
				lane = "-"
			}
			fmt.Fprintf(tw, "fak %s\t%s\t%s\n", v.Name, lane, v.Synopsis)
		})
}

func indexDocs(stdout, stderr io.Writer, cat *devindex.Catalog, args []string, asJSON bool, limit int) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, "fak index docs: needs a search query")
		return 2
	}
	hits := capDocs(cat.SearchDocs(joinArgs(args)), limit)
	return indexRenderHits(stdout, stderr, hits, asJSON, "fak index docs", "no matching doc",
		func(tw *tabwriter.Writer, d devindex.Doc) {
			fmt.Fprintf(tw, "%s\t%s\n", d.Path, d.Title)
		})
}

func writeIndexUsage(w io.Writer) {
	fmt.Fprint(w, `usage:
  fak index lane <path>...    which lane/leaf owns each path, + the (fak <leaf>) commit stamp
  fak index leaf [<query>]    the lane taxonomy (+ shipped/sim/stub rollup), filtered by name/tree/desc
  fak index docs <query>      the curated doc map (INDEX.md), ranked by relevance
  fak index claims <query>    the CLAIMS.md honesty ledger, ranked by relevance (shipped/simulated/stub)
  fak index verbs [<query>]   the structured CLI-verb catalog (name/owning-lane/synopsis)
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

func capClaims(cs []devindex.Claim, limit int) []devindex.Claim {
	if limit > 0 && len(cs) > limit {
		return cs[:limit]
	}
	return cs
}

func capVerbs(vs []devindex.Verb, limit int) []devindex.Verb {
	if limit > 0 && len(vs) > limit {
		return vs[:limit]
	}
	return vs
}

// truncRunes shortens s to at most n runes (UTF-8-safe — a claim line carries em
// dashes and middots that a byte slice would split), appending an ellipsis when it
// cuts. The package-wide truncate (benchmarks.go) is byte-based and ASCII-only.
func truncRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}

func flushTab(tw *tabwriter.Writer, stderr io.Writer, label string) int {
	if err := tw.Flush(); err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", label, err)
		return 1
	}
	return 0
}
