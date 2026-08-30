// Command verbsdoc renders fak's source-derived verb/refusal surface (#5934).
package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

const outputPath = "docs/generated/verb-surface.md"

const pageFrontmatter = `---
title: "fak verb surface (generated)"
description: "Generated reference for fak CLI verbs, their purpose, implementation surface, and help coverage, produced by go run ./cmd/verbsdoc from the Go source tree."
---
`

var surfaceTableMarker = []byte("\n| VERB | PURPOSE | IMPLEMENTS | DOC | PRECONDITION | REFUSES | HELP |\n")

func main() { os.Exit(run(os.Args[1:])) }

func run(args []string) int {
	fs := flag.NewFlagSet("verbsdoc", flag.ContinueOnError)
	write := fs.Bool("write", false, "write the generated page")
	printPage := fs.Bool("print", false, "print the generated page without touching disk")
	root := fs.String("root", ".", "repository root")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *write && *printPage {
		fmt.Fprintln(os.Stderr, "choose only one of -write or -print")
		return 2
	}
	surface, err := devindex.ExtractVerbSurface(*root)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	rendered, err := renderPage(surface.Markdown())
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	if *printPage {
		_, _ = os.Stdout.Write(rendered)
		return 0
	}
	path := filepath.Join(*root, filepath.FromSlash(outputPath))
	if *write {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		if err := os.WriteFile(path, rendered, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			return 1
		}
		return 0
	}
	current, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "VERB_SURFACE_SYNC: %v; run go run ./cmd/verbsdoc -write\n", err)
		return 1
	}
	if !bytes.Equal(current, rendered) {
		fmt.Fprintln(os.Stderr, "VERB_SURFACE_SYNC: generated page is stale; run go run ./cmd/verbsdoc -write")
		return 1
	}
	return 0
}

// renderPage owns the generated document shell that sits around devindex's source-derived
// table. Keeping the frontmatter and section heading here makes -write idempotent instead of
// deleting the metadata that documentation tooling requires on every regeneration.
func renderPage(surface []byte) ([]byte, error) {
	if bytes.Count(surface, surfaceTableMarker) != 1 {
		return nil, fmt.Errorf("verb surface table marker count = %d, want 1", bytes.Count(surface, surfaceTableMarker))
	}
	body := bytes.Replace(surface, surfaceTableMarker,
		append([]byte("\n## Surface table\n"), surfaceTableMarker...), 1)
	return append([]byte(pageFrontmatter), body...), nil
}
