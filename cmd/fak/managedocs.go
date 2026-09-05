package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/managedocs"
)

func cmdManageDocs(argv []string) {
	os.Exit(runManageDocs(os.Stdout, os.Stderr, argv))
}

func runManageDocs(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak managedocs", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "repository root (default: repo root)")
	docsDir := fs.String("docs-dir", "", "documentation directory relative to workspace for document sets audit")
	checkRetained := fs.Bool("check-retained", false, "run retained occurrences audit")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	}
	if *docsDir != "" {
		if err := managedocs.AuditDocumentSets(root, *docsDir); err != nil {
			fmt.Fprintf(stderr, "fak managedocs document sets audit failed: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "managedocs: document sets audit in %q passed\n", *docsDir)
		if !*checkRetained {
			return 0
		}
	}

	if err := managedocs.Audit(root); err != nil {
		fmt.Fprintf(stderr, "fak managedocs retained audit failed: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, "managedocs: retained occurrences audit passed")
	return 0
}
