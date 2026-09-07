package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachedocaudit"
)

// cmdCachedocNumbersAudit executes the doc-numbers audit CLI over operational cachevalue docs.
func cmdCachedocNumbersAudit(args []string) {
	os.Exit(runCachedocNumbersAudit(os.Stdout, os.Stderr, args))
}

// runCachedocNumbersAudit runs the cachedoc numbers audit with the given arguments.
func runCachedocNumbersAudit(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("cachedoc-numbers-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)

	root := fs.String("root", ".", "repo root")
	manifest := fs.String("manifest", "", "limit to one manifest (basename)")
	live := fs.Bool("live", false, "additionally re-derive since_floored claims from live fak")
	refresh := fs.Bool("refresh", false, "regenerate trimmed snapshots from live fak (mutates snapshot_dir)")
	asJSON := fs.Bool("json", false, "machine-readable output")

	if err := fs.Parse(args); err != nil {
		return 2
	}

	absRoot, err := filepath.Abs(*root)
	if err != nil {
		fmt.Fprintf(stderr, "doc-numbers: invalid root %q: %v\n", *root, err)
		return 2
	}

	return cachedocaudit.Run(stdout, stderr, cachedocaudit.AuditOptions{
		Root:     absRoot,
		Manifest: *manifest,
		Live:     *live,
		Refresh:  *refresh,
		AsJSON:   *asJSON,
		Today:    time.Now().UTC(),
	})
}
