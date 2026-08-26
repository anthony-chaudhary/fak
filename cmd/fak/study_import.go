package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func runStudyImport(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("study-import", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", ".", "repository root used for tracked artifact discovery")
	store := fs.String("store", "", "isolated destination directory (required unless --dry-run)")
	dryRun := fs.Bool("dry-run", false, "discover and classify without writing records")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || (!*dryRun && *store == "") {
		fmt.Fprintln(stderr, "usage: fak study-import [--repo PATH] [--dry-run | --store PATH]")
		return 2
	}
	ledger, err := studymonitor.ImportTracked(*repo, *store, *dryRun)
	if err != nil {
		fmt.Fprintf(stderr, "study-import: %v\n", err)
		return 1
	}
	encoder := json.NewEncoder(stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(ledger); err != nil {
		fmt.Fprintf(stderr, "study-import: %v\n", err)
		return 1
	}
	return 0
}
