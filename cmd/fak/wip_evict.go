package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/wipinventory"
)

// runWipEvict implements `fak wip evict`. It archives orphan working-tree residue
// into refs/fak/quarantine/* before removing it from the working tree.
func runWipEvict(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("wip evict", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "wip")
	repo := fs.String("C", "", "run in this git repo (default: cwd)")
	asJSON := fs.Bool("json", false, "emit the eviction result as JSON")
	dryRun := fs.Bool("dry-run", false, "report what would be evicted without snapshotting or removing")
	autoArchive := fs.Bool("auto-archive", true, "snapshot orphan residue into refs/fak/quarantine/* before removing (default: true)")
	session := fs.String("session", "", "session id (defaults to $CLAUDE_CODE_SESSION_ID, $FAK_SESSION_ID)")
	if !parseFlags(fs, argv) {
		return 2
	}
	targetDir := *repo
	if targetDir == "" {
		targetDir = "."
	}
	sessID := strings.TrimSpace(*session)
	if sessID == "" {
		sessID = firstNonEmpty(os.Getenv("CLAUDE_CODE_SESSION_ID"), os.Getenv("FAK_SESSION_ID"))
	}
	targets := fs.Args()
	if !*autoArchive {
		var removed []string
		for _, t := range targets {
			full := filepath.Join(targetDir, filepath.FromSlash(t))
			if err := os.Remove(full); err == nil {
				removed = append(removed, t)
			}
		}
		if *asJSON {
			_ = writeIndentedJSON(stdout, map[string]any{"removed": removed, "auto_archive": false})
			return 0
		}
		fmt.Fprintf(stdout, "removed %d file(s) without auto-archive\n", len(removed))
		return 0
	}
	qref, err := wipinventory.EvictOrphans(context.Background(), targetDir, wipinventory.GitRunner{}, wipinventory.EvictOptions{
		SessionID: sessID,
		Targets:   targets,
		DryRun:    *dryRun,
		Reason:    "fak wip evict",
	})
	if err != nil {
		if code, done := emitResultOrError(stdout, stderr, "fak wip evict", *asJSON, wipinventory.QuarantineRef{}, err); done {
			return code
		}
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, qref); err != nil {
			fmt.Fprintf(stderr, "fak wip evict: %v\n", err)
			return 1
		}
		return 0
	}
	if qref == nil || qref.Count == 0 {
		fmt.Fprintln(stdout, "no orphan working-tree residue to evict")
		return 0
	}
	action := "evicted"
	if *dryRun {
		action = "would evict"
	}
	fmt.Fprintf(stdout, "%s %d orphan file(s) (%d bytes) to %s (%s)\n", action, qref.Count, qref.ByteTotal, qref.Ref, qref.SHA)
	for _, f := range qref.Files {
		fmt.Fprintf(stdout, "  %s\n", f)
	}
	return 0
}
