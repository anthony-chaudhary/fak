package main

// seoaeoscore.go wires the `fak score seo` verb over internal/seoaeoscore: the SEO/AEO
// discoverability measuring stick. It re-derives, from the git-tracked tree, whether a
// reader — or an answer engine — can DISCOVER fak and CITE it correctly, folding each
// concrete, re-derivable discoverability defect into an integer seo-debt (lower = better,
// floor 0) driven toward zero.
//
// This is the Go port of the former tools/seo_aeo_scorecard.py; the surfaces
// (default / --json / --markdown / --stamp / --compare / --scope) are preserved in behavior
// so the private snapshot and the control-pane fold are unchanged. Read-only by construction.
// (The python tool's --transfer/--rebaseline private-archive path is out of scope for the
// scoring port; regenerate the private SEO-AEO-SCORECARD.md from --markdown.)

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/seoaeoscore"
)

func cmdSEOAEOScore(argv []string) {
	os.Exit(runSEOAEOScore(os.Stdout, os.Stderr, argv))
}

// runSEOAEOScore scores discoverability, re-derived from the tree. Exit codes: 0 no
// seo-debt, 1 carries seo-debt, 2 usage/IO error.
func runSEOAEOScore(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak score seo", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	scope := fs.String("scope", "core", "page set to score: core|published")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	asMarkdown := fs.Bool("markdown", false, "emit the SEO-AEO-SCORECARD.md body")
	stamp := fs.String("stamp", "", "date stamp for the markdown header")
	comparePath := fs.String("compare", "", "print the seo-debt delta vs a prior --json baseline (proves 10x)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak score seo: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *scope != "core" && *scope != "published" {
		fmt.Fprintf(stderr, "fak score seo: --scope must be core|published, got %q\n", *scope)
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	payload := seoaeoscore.Build(root, *scope)
	if *comparePath != "" {
		base, ok := readCompareBase(stderr, "fak score seo", *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprintln(stdout, seoaeoscore.Compare(payload, base))
		if payload.OK {
			return 0
		}
		return 1
	}
	if code := emitJSONOrRender(stdout, stderr, "fak score seo", *asJSON, payload, func(w io.Writer) {
		if *asMarkdown {
			fmt.Fprintln(w, seoaeoscore.Markdown(payload, *stamp))
		} else {
			fmt.Fprintln(w, seoaeoscore.Render(payload))
		}
	}); code != 0 {
		return code
	}
	if payload.OK {
		return 0
	}
	return 1
}
