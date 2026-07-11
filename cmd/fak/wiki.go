package main

// fak wiki — the fak-native, witness-verified repo wiki (epic #4277, mined from
// deepwiki-open @ 16f35a0 — inspire, clean-room). Two deterministic subverbs, the
// two things every LLM wiki generator in the field structurally cannot do:
//
//	fak wiki structure        emit the section→page tree projected from the
//	                          self-index (leaves/lanes/docs) — NOT LLM-inferred.
//	                          --json prints the stable wiki.json. (L1 #4278)
//	fak wiki verify <page.md> resolve every `Sources:[path:line]` code citation in
//	                          the page against the working tree; exit nonzero on any
//	                          dangling cite (missing file / out-of-range line). A CI
//	                          gate the whole pipeline runs before publish. (L3 #4280)
//
// Both are pure views over internal/wiki + internal/devindex: no LLM, no clock, no
// network. `structure` seeds the tree from ground truth; `verify` witnesses cites.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/wiki"
)

func cmdWiki(argv []string) { os.Exit(runWiki(os.Stdout, os.Stderr, argv)) }

func runWiki(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		writeWikiUsage(stderr)
		return 2
	}
	sub := argv[0]
	switch sub {
	case "structure", "struct", "tree":
		return wikiStructure(stdout, stderr, argv[1:])
	case "verify", "check":
		return wikiVerify(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		writeWikiUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak wiki: unknown subcommand %q\n", sub)
		writeWikiUsage(stderr)
		return 2
	}
}

func writeWikiUsage(w io.Writer) {
	fmt.Fprint(w, `fak wiki — witness-verified repo wiki

usage:
  fak wiki structure [--root DIR] [--json]   section→page tree from the self-index
  fak wiki verify <page.md> [--root DIR] [--json]
                                             resolve Sources:[path:line] cites vs the tree

`)
}

// wikiStructure prints the deterministic section→page tree. Text by default (a
// navigable outline), --json for the stable wiki.json payload.
func wikiStructure(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wiki structure", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wiki")
	root := fs.String("root", "", "repo root (default: search upward for dos.toml)")
	asJSON := fs.Bool("json", false, "emit the stable wiki.json")
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	rootDir := *root
	if rootDir == "" {
		rootDir = devindex.FindRoot(".")
	}
	cat, err := devindex.Load(rootDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak wiki: %v\n", err)
		return 1
	}
	tree := wiki.Structure(cat)
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, tree, "fak wiki")
	}
	return renderWikiTree(stdout, tree)
}

func renderWikiTree(stdout io.Writer, tree wiki.Tree) int {
	fmt.Fprintf(stdout, "wiki: %s (%d pages)\n", tree.Repo, tree.PageCount())
	for _, s := range tree.Sections {
		fmt.Fprintf(stdout, "\n%s (%d)\n", s.Title, len(s.Pages))
		for _, p := range s.Pages {
			files := ""
			if len(p.RelevantFiles) > 0 {
				files = "  ← " + joinCap(p.RelevantFiles, 3)
			}
			fmt.Fprintf(stdout, "  - %s%s\n", p.Title, files)
		}
	}
	return 0
}

// wikiVerify resolves the page's code citations against the tree. It exits 0 when
// every cite resolves (or the page cites nothing), 1 on any dangler — the gate.
func wikiVerify(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wiki verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wiki")
	root := fs.String("root", "", "repo root the cites resolve against (default: search upward for dos.toml)")
	asJSON := fs.Bool("json", false, "emit the danglers as JSON")
	if err := fs.Parse(argv); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(stderr, "fak wiki verify: needs a <page.md>")
		return 2
	}
	page := fs.Arg(0)
	md, err := os.ReadFile(page)
	if err != nil {
		fmt.Fprintf(stderr, "fak wiki verify: %v\n", err)
		return 1
	}
	rootDir := *root
	if rootDir == "" {
		rootDir = devindex.FindRoot(".")
	}
	dangs := wiki.VerifyCitations(rootDir, md)
	if *asJSON {
		code := encodeJSONOrFail(stdout, stderr, dangs, "fak wiki verify")
		if code == 0 && len(dangs) > 0 {
			return 1
		}
		return code
	}
	if len(dangs) == 0 {
		fmt.Fprintf(stdout, "ok: %s — every code citation resolves against %s\n", page, rootDir)
		return 0
	}
	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintf(tw, "DANGLING\tLINE\tREASON\tCITE\n")
	for _, d := range dangs {
		detail := string(d.Reason)
		if d.Reason == wiki.ReasonLineOutOfRange {
			detail = fmt.Sprintf("%s (file has %d lines)", d.Reason, d.Lines)
		}
		fmt.Fprintf(tw, "%s\tL%d\t%s\t%s\n", d.Path, d.Line, detail, d.Raw)
	}
	tw.Flush()
	fmt.Fprintf(stderr, "fak wiki verify: %d dangling citation(s) in %s\n", len(dangs), page)
	return 1
}

// joinCap joins up to n entries with ", ", appending a "+k more" tail when the
// slice is longer — keeps the text tree from spilling a huge glob list per page.
func joinCap(xs []string, n int) string {
	if len(xs) <= n {
		return joinComma(xs)
	}
	return joinComma(xs[:n]) + fmt.Sprintf(", +%d more", len(xs)-n)
}

func joinComma(xs []string) string {
	out := ""
	for i, x := range xs {
		if i > 0 {
			out += ", "
		}
		out += x
	}
	return out
}
