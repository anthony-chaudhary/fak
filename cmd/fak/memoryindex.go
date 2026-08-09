package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/memoryindex"
)

func runMemoryIndex(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("memory index", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dir := fs.String("dir", "", "memory directory")
	write := fs.Bool("write", false, "reconcile MEMORY.md")
	asJSON := fs.Bool("json", false, "emit JSON report")
	if fs.Parse(args) != nil || *dir == "" {
		fmt.Fprintln(stderr, "usage: fak memory index --dir DIR [--write] [--json]")
		return 2
	}
	opt := memoryindex.Options{Types: []string{"project", "user", "feedback", "reference"}}
	rep, ok := memoryindex.Check(*dir, opt)
	if !ok {
		fmt.Fprintln(stderr, "memory directory has no MEMORY.md")
		return 2
	}
	if *write {
		_, after, err := memoryindex.Apply(*dir, opt)
		if err != nil {
			fmt.Fprintln(stderr, err)
			return 2
		}
		rep = after
	}
	if *asJSON {
		_ = json.NewEncoder(stdout).Encode(rep)
	} else {
		fmt.Fprintf(stdout, "missing_from_index=%d index_line_with_no_file=%d slug_filename_mismatch=%d duplicate_slug=%d type_vocabulary_violation=%d unresolved_wikilink=%d\n", rep.Counts[memoryindex.KindMissingFromIndex], rep.Counts[memoryindex.KindIndexLineNoFile], rep.Counts[memoryindex.KindSlugMismatch], rep.Counts[memoryindex.KindDuplicateSlug], rep.Counts[memoryindex.KindTypeVocabulary], rep.Counts[memoryindex.KindUnresolvedLink])
	}
	if rep.Drifted() {
		return 1
	}
	return 0
}
