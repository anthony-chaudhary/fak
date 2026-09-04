package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

const managedWorktreeDocPath = "docs/managed-worker-worktrees.md"

// TestManagedWorkerWorktreesDoc_SubcommandsAndOptions verifies that the operator
// guide and runbook thoroughly documents all sub-commands, required flags, environment
// variables, and core architecture concepts.
func TestManagedWorkerWorktreesDoc_SubcommandsAndOptions(t *testing.T) {
	doc := readRepoFile(t, managedWorktreeDocPath)
	if doc == "" {
		t.Fatalf("%s is empty", managedWorktreeDocPath)
	}

	// 1. All 8 sub-commands must be documented as distinct sections/commands.
	subcommands := []string{
		"defaults",
		"prepare",
		"land",
		"reap",
		"gc",
		"list",
		"publish",
		"recover",
	}
	for _, sub := range subcommands {
		if !strings.Contains(doc, "`"+sub+"`") && !strings.Contains(doc, "### "+sub) {
			t.Errorf("%s does not document sub-command %q", managedWorktreeDocPath, sub)
		}
	}

	// 2. Core environment variables must be documented.
	envVars := []string{
		"FLEET_WORKER_WORKTREE_ROOT",
		"GOCACHE",
		"GOTMPDIR",
		"DISPATCH_WORKSPACE",
		"FLEET_WORKER_WORKTREE_DIR",
	}
	for _, env := range envVars {
		if !strings.Contains(doc, "`"+env+"`") {
			t.Errorf("%s does not document environment variable %q", managedWorktreeDocPath, env)
		}
	}

	// 3. Key options and flags across commands must be documented.
	keyFlags := []string{
		"--all-cold",
		"--apply",
		"--age-floor-min",
		"--even-if-unlanded",
		"--superseded-by",
		"--max-wait",
		"--worktree",
		"--base-sha",
		"--msg-file",
		"--paths",
		"--verify",
		"--core-lock-maintenance-witness",
		"--recovery-remote",
		"--require-remote-recovery",
		"--disambiguation-timeout-ms",
		"--unsafe-skip-symptom-witness",
		"--lane",
		"--key",
		"--lease-id",
		"--owner-pid",
		"--capacity-reason",
		"--max-age",
		"--dry-run",
		"--remote",
		"--fetch",
		"--cleanup",
		"--cleanup-remote",
		"--force",
		"--allow-peer",
	}
	for _, flag := range keyFlags {
		if !strings.Contains(doc, flag) {
			t.Errorf("%s does not document flag %q", managedWorktreeDocPath, flag)
		}
	}

	// 4. Core architecture and invariants must be explicitly explained.
	coreConcepts := []string{
		"detached",
		"land_worktree_diff",
		"OFF_TRUNK",
		"LOCALAPPDATA",
		"os.TempDir()",
		"fak.worktree.defaults.v1",
		"fak-worker-worktree-lifecycle/1",
	}
	for _, concept := range coreConcepts {
		if !strings.Contains(doc, concept) {
			t.Errorf("%s missing core concept explanation for %q", managedWorktreeDocPath, concept)
		}
	}
}

// TestManagedWorkerWorktreesDoc_Reachability verifies that the operator guide is
// reachable from all canonical repository entry points and navigation surfaces.
func TestManagedWorkerWorktreesDoc_Reachability(t *testing.T) {
	entryPoints := map[string]string{
		"docs/cli-reference.md": "managed-worker-worktrees.md",
		"AGENTS.md":             "docs/managed-worker-worktrees.md",
		"INDEX.md":              "docs/managed-worker-worktrees.md",
		"llms.txt":              "docs/managed-worker-worktrees.md",
	}

	for file, wantRef := range entryPoints {
		content := readRepoFile(t, file)
		if !strings.Contains(content, wantRef) {
			t.Errorf("%s does not link to %s (expected link containing %q)", file, managedWorktreeDocPath, wantRef)
		}
	}
}

// TestCLIReference_WorktreeWorkerSection verifies that docs/cli-reference.md
// contains the `fak worktree worker` section documenting all sub-commands and safety defaults.
func TestCLIReference_WorktreeWorkerSection(t *testing.T) {
	content := readRepoFile(t, "docs/cli-reference.md")

	if !strings.Contains(content, "## `fak worktree worker`") {
		t.Fatalf("docs/cli-reference.md missing ## `fak worktree worker` section")
	}

	subcommands := []string{
		"defaults",
		"prepare",
		"land",
		"reap",
		"gc",
		"list",
		"publish",
		"recover",
	}
	for _, sub := range subcommands {
		if !strings.Contains(content, "- `"+sub+"`:") && !strings.Contains(content, "`"+sub+"`") {
			t.Errorf("docs/cli-reference.md section does not list sub-command %q", sub)
		}
	}

	// Verify safety defaults callout.
	safetyKeywords := []string{
		"dry-run",
		"--apply",
		"read-only",
	}
	for _, kw := range safetyKeywords {
		if !strings.Contains(content, kw) {
			t.Errorf("docs/cli-reference.md section missing safety default keyword %q", kw)
		}
	}
}

// TestWorktreeWorkerUsage_HelpAndDoc verifies that the CLI usage output points
// to the operator guide and lists all sub-commands.
func TestWorktreeWorkerUsage_HelpAndDoc(t *testing.T) {
	// Capture stderr output of worktreeWorkerUsage()
	oldStderr := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stderr = w

	worktreeWorkerUsage()

	w.Close()
	os.Stderr = oldStderr

	var buf bytes.Buffer
	io.Copy(&buf, r)
	usage := buf.String()

	if !strings.Contains(usage, "docs/managed-worker-worktrees.md") {
		t.Errorf("worktreeWorkerUsage() does not mention docs/managed-worker-worktrees.md; got:\n%s", usage)
	}

	subcommands := []string{
		"defaults",
		"prepare",
		"land",
		"reap",
		"gc",
		"list",
		"publish",
		"recover",
	}
	for _, sub := range subcommands {
		if !strings.Contains(usage, sub) {
			t.Errorf("worktreeWorkerUsage() missing sub-command %q; got:\n%s", sub, usage)
		}
	}
}

// TestDevIndexVerbDoc_Worktree verifies that the devindex catalog entry for the
// worktree verb points to docs/managed-worker-worktrees.md.
func TestDevIndexVerbDoc_Worktree(t *testing.T) {
	cat := &devindex.Catalog{}
	v, ok := cat.VerbByName("worktree")
	if !ok {
		t.Fatalf("verb 'worktree' not found in devindex catalog")
	}
	if v.Doc != managedWorktreeDocPath {
		t.Errorf("devindex verb 'worktree' Doc = %q, want %q", v.Doc, managedWorktreeDocPath)
	}
}
