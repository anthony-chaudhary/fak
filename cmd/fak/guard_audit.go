package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardaudit"
	"github.com/anthony-chaudhary/fak/internal/logvault"
)

func cmdGuardAudit(args []string) {
	if len(args) == 0 || args[0] != "prune" {
		fmt.Fprintln(os.Stderr, "usage: fak guard-audit prune [--apply] [--max-age 168h] [--max-files 1500] [--repo DIR] [--vault DIR] [--json]")
		os.Exit(2)
	}
	os.Exit(runGuardAuditPrune(args[1:], os.Stdout, os.Stderr, time.Now()))
}

func runGuardAuditPrune(args []string, stdout, stderr io.Writer, now time.Time) int {
	fs := flag.NewFlagSet("guard-audit prune", flag.ContinueOnError)
	fs.SetOutput(stderr)
	apply := fs.Bool("apply", false, "remove eligible mirrored journals (default: dry-run)")
	maxAge := fs.Duration("max-age", guardaudit.DefaultMaxAge, "remove mirrored journals older than this (0 disables age bound)")
	maxFiles := fs.Int("max-files", guardaudit.DefaultMaxFiles, "retain at most this many newest journals (-1 disables count bound)")
	repo := fs.String("repo", "", "repository root (default: discover from cwd)")
	vaultDir := fs.String("vault", "", "logvault directory (default: $FAK_LOG_VAULT, else sibling fak-log-vault)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *maxFiles < -1 || *maxAge < 0 {
		fmt.Fprintln(stderr, "guard-audit prune: invalid retention bound")
		return 2
	}
	if *repo == "" {
		cwd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "guard-audit prune: %v\n", err)
			return 1
		}
		*repo = findRepoRoot(cwd)
	}
	absRepo, err := filepath.Abs(*repo)
	if err != nil {
		fmt.Fprintf(stderr, "guard-audit prune: %v\n", err)
		return 1
	}
	if *vaultDir == "" {
		*vaultDir = resolveLogvaultDir(absRepo)
	}
	v := &logvault.Vault{Dir: *vaultDir}
	witnessed, err := v.WitnessedFiles("dispatch-runs")
	if err != nil {
		fmt.Fprintf(stderr, "guard-audit prune: refusing without verified logvault mirror: %v\n", err)
		return 1
	}
	rep, err := guardaudit.Plan(absRepo, *vaultDir, now, *maxAge, *maxFiles, witnessed)
	if err != nil {
		fmt.Fprintf(stderr, "guard-audit prune: %v\n", err)
		return 1
	}
	if *apply {
		if err := guardaudit.Apply(&rep); err != nil {
			fmt.Fprintf(stderr, "guard-audit prune: %v\n", err)
			return 1
		}
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(rep)
	} else {
		mode := "dry-run"
		if *apply {
			mode = "applied"
		}
		fmt.Fprintf(stdout, "guard-audit prune: %s scanned=%d mirrored=%d unmirrored=%d eligible=%d guard_audit_pruned=%d bytes=%d\n", mode, rep.Scanned, rep.Mirrored, rep.Unmirrored, len(rep.Candidates), rep.GuardAuditPruned, rep.GuardAuditPrunedBytes)
	}
	return 0
}
