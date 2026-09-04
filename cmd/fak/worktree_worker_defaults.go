package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func runWorktreeWorkerDefaults(argv []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("worktree worker defaults", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	asJSON := fs.Bool("json", false, "emit defaults report as JSON")
	if err := fs.Parse(argv); err != nil {
		return err
	}

	repoRoot := strings.TrimSpace(*root)
	if repoRoot == "" {
		repoRoot = discoverRepoRoot()
	}

	report := workerworktree.ResolveDefaults(repoRoot)

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetEscapeHTML(false)
		return enc.Encode(report)
	}

	fmt.Fprintf(stdout, "schema: %s\n", report.Schema)
	fmt.Fprintf(stdout, "repo_root: %s\n", report.RepoRoot)
	fmt.Fprintf(stdout, "worker_worktree_root: %s\n", report.WorkerWorktreeRoot)
	fmt.Fprintf(stdout, "root_source: %s\n", report.RootSource)
	fmt.Fprintf(stdout, "default_lease_identity_basis: %s\n", report.DefaultLeaseIdentityBasis)
	fmt.Fprintf(stdout, "supported_env_overrides: %s\n", strings.Join(report.SupportedEnvOverrides, ", "))
	return nil
}

func worktreeWorkerDefaults(argv []string) {
	if err := runWorktreeWorkerDefaults(argv, os.Stdout, os.Stderr); err != nil {
		os.Exit(2)
	}
}
