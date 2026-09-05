package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/sessionsearch"
)

func cmdSessionSearch(argv []string) {
	os.Exit(runSessionSearch(os.Stdout, os.Stderr, argv))
}

func runSessionSearch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak sessionsearch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	query := fs.String("query", "", "query terms for recall index search")
	topK := fs.Int("top", 5, "maximum number of top hits to return")
	window := fs.Int("window", 2, "context window size around hits")
	asJSON := fs.Bool("json", false, "emit recall search results as JSON")
	journal := fs.String("journal", "", "path to tool-process journal JSONL file to index")

	if !parseFlags(fs, argv) {
		return 2
	}

	idx := sessionsearch.NewIndex()
	if *journal != "" {
		f, err := os.Open(*journal)
		if err != nil {
			fmt.Fprintf(stderr, "fak sessionsearch: open journal: %v\n", err)
			return 2
		}
		defer f.Close()
		docs, err := sessionsearch.DocsFromJournal(f)
		if err != nil {
			fmt.Fprintf(stderr, "fak sessionsearch: read journal: %v\n", err)
			return 2
		}
		for _, d := range docs {
			idx.Add(d)
		}
	}

	hits := idx.Search(*query, *topK, *window)
	if hits == nil {
		hits = []sessionsearch.Hit{}
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(hits); err != nil {
			fmt.Fprintf(stderr, "fak sessionsearch: json encode: %v\n", err)
			return 2
		}
		return 0
	}

	fmt.Fprintf(stdout, "sessionsearch: query %q matched %d hit(s)\n", *query, len(hits))
	for _, h := range hits {
		fmt.Fprintf(stdout, "  [score: %.3f] %s: %s\n", h.Score, h.Doc.ID, h.Doc.Text)
	}
	return 0
}
