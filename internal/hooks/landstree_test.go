package hooks

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestLandsTreeIgnoresPeerWorktreeViolations(t *testing.T) {
	root := landsTreeFixture(t)
	if err := os.WriteFile(filepath.Join(root, "peer-untracked.md"), []byte("peer file\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	untrackedGate := func(d *StagedDiff) ([]Finding, error) {
		if d.Exists("peer-untracked.md") {
			return []Finding{{Gate: "BROKEN_LINK", File: "peer-untracked.md", Detail: "peer worktree file"}}, nil
		}
		return nil, nil
	}
	d, err := ReadStagedDiff(root)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := untrackedGate(d); len(got) != 1 {
		t.Fatalf("worktree untracked findings = %#v, want peer violation", got)
	}
	landedUntracked, err := scopedCheck("BROKEN_LINK", untrackedGate)(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(landedUntracked) != 0 {
		t.Fatalf("lands-tree untracked findings = %#v, want peer file silent", landedUntracked)
	}

	// Peer removes a TRACKED but UNSTAGED link target. The old view sees it absent and denies;
	// HEAD plus the staged README reads the committed target and stays silent.
	if err := os.Remove(filepath.Join(root, "peer.md")); err != nil {
		t.Fatal(err)
	}
	worktreeModified, err := gateBrokenLink(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(worktreeModified) != 1 {
		t.Fatalf("worktree modified findings = %#v, want unstaged missing link", worktreeModified)
	}
	landedModified, err := landstreeGateByName(t, "BROKEN_LINK").Check(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(landedModified) != 0 {
		t.Fatalf("lands-tree modified findings = %#v, want unstaged edit silent", landedModified)
	}
}
func TestLandsTreeStillDeniesStagedViolation(t *testing.T) {
	root := landsTreeFixture(t)
	readme := filepath.Join(root, "README.md")
	if err := os.WriteFile(readme, []byte("[staged](missing.md)\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	landsTreeGit(t, root, "add", "README.md")

	d, err := ReadStagedDiff(root)
	if err != nil {
		t.Fatal(err)
	}
	findings, err := landstreeGateByName(t, "BROKEN_LINK").Check(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 1 {
		t.Fatalf("findings = %#v, want staged missing link denied", findings)
	}
	if findings[0].View != ViewLandsTree {
		t.Fatalf("finding view = %q, want %q", findings[0].View, ViewLandsTree)
	}
}

func TestGateScopesClassifyEveryRegisteredGate(t *testing.T) {
	rows := map[string]GateScope{}
	for _, row := range GateScopes() {
		if _, exists := rows[row.Gate]; exists {
			t.Fatalf("duplicate scope row for %s", row.Gate)
		}
		if row.Class != ClassLandsTree && row.Why == "" {
			t.Fatalf("%s has class %s without a reason", row.Gate, row.Class)
		}
		rows[row.Gate] = row
	}
	for _, gate := range PreCommitGates() {
		if _, ok := rows[gate.Name]; !ok {
			t.Errorf("registered gate %s is unclassified", gate.Name)
		}
	}
}

func landsTreeFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	landsTreeGit(t, root, "init")
	landsTreeGit(t, root, "config", "user.name", "hooks test")
	landsTreeGit(t, root, "config", "user.email", "hooks@example.invalid")
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("baseline\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "peer.md"), []byte("committed target\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	landsTreeGit(t, root, "add", "README.md", "peer.md")
	landsTreeGit(t, root, "commit", "-m", "baseline")
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("[peer](peer.md)\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	landsTreeGit(t, root, "add", "README.md")
	return root
}

func landsTreeGit(t *testing.T, root string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func landstreeGateByName(t *testing.T, name string) Gate {
	t.Helper()
	for _, gate := range PreCommitGates() {
		if gate.Name == name {
			return gate
		}
	}
	t.Fatalf("gate %s not registered", name)
	return Gate{}
}
