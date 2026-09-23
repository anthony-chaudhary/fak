package selfupdatecmd

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfinstall"
)

// This test uses only the pre-fix API so the managed land gate can overlay it on
// the parent: the parent reports the embedded artifact revision A, while the
// repaired check reports the selected source revision B for unchanged bytes.
func TestSelfUpdateMetadataOnlyCheckUsesSelectedSource(t *testing.T) {
	const helperEnv = "FAK_TEST_METADATA_ONLY_CHECK_HELPER"
	const repoEnv = "FAK_TEST_METADATA_ONLY_REPO"
	const targetEnv = "FAK_TEST_METADATA_ONLY_TARGET"
	if os.Getenv(helperEnv) == "1" {
		Run([]string{"--check", "--root", os.Getenv(repoEnv), "--target", os.Getenv(targetEnv), "--json"})
		os.Exit(0)
	}

	dir := t.TempDir()
	repo := filepath.Join(dir, "source")
	home := filepath.Join(dir, "home")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(home, "bin"), 0o755); err != nil {
		t.Fatal(err)
	}
	targetName := "fak"
	if runtime.GOOS == "windows" {
		targetName += ".exe"
	}
	target := filepath.Join(home, "bin", targetName)
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-b", "main")
	git("config", "user.name", "Fixture")
	git("config", "user.email", "fixture@example.invalid")
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example.com/fakfixture\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "main.go"), []byte("package main\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "go.mod", "main.go")
	git("commit", "-m", "fixture A")
	sourceA := git("rev-parse", "HEAD")
	build := exec.Command("go", "build", "-buildvcs=true", "-o", target, ".")
	build.Dir = repo
	build.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=-p=1", "GOMAXPROCS=2")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build stamped fixture: %v: %s", err, out)
	}
	stamp, ok := stampOfBinary(target)
	if !ok || !stamp.HasVCS || stamp.Dirty || stamp.Revision != sourceA {
		t.Fatalf("fixture stamp = %+v, ok=%v; want clean %s", stamp, ok, sourceA)
	}
	body, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	artifactHash := sha256.Sum256(body)
	inputHash := sha256.Sum256([]byte("unchanged executable inputs"))
	candidate := selfinstall.StateUpdate{
		SelectedSourceCommit: sourceA,
		ArtifactSourceCommit: sourceA,
		BuildInputDigest:     "sha256:" + hex.EncodeToString(inputHash[:]),
		ArtifactDigest:       hex.EncodeToString(artifactHash[:]),
		ArtifactSize:         int64(len(body)),
		AppVersion:           "1.0.0",
	}
	identityPath := selfinstall.IdentityStatePath(target)
	rollback := filepath.Join(dir, "prior")
	state, err := selfinstall.AdvanceInstallIdentity(identityPath, selfinstall.InstallIdentity{}, candidate, target, rollback, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "README.md"), []byte("metadata-only source advance\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "README.md")
	git("commit", "-m", "fixture B metadata only")
	sourceB := git("rev-parse", "HEAD")
	git("update-ref", "refs/remotes/origin/main", sourceB)
	candidate.SelectedSourceCommit = sourceB
	if _, err := selfinstall.AdvanceInstallIdentity(identityPath, state, candidate, target, rollback, false); err != nil {
		t.Fatal(err)
	}

	check := func() selfUpdateReceipt {
		t.Helper()
		cmd := exec.Command(os.Args[0], "-test.run=^TestSelfUpdateMetadataOnlyCheckUsesSelectedSource$")
		cmd.Env = append(os.Environ(), helperEnv+"=1", repoEnv+"="+repo, targetEnv+"="+target, "USERPROFILE="+home, "HOME="+home, "FAK_SELF_UPDATE_INSTALLER=native")
		out, err := cmd.Output()
		if err != nil {
			t.Fatalf("self-update --check helper: %v: %s", err, out)
		}
		var receipt selfUpdateReceipt
		if err := json.Unmarshal(out, &receipt); err != nil {
			t.Fatalf("decode check receipt: %v: %q", err, out)
		}
		return receipt
	}
	receipt := check()
	if receipt.OldRevision == nil || *receipt.OldRevision != sourceB || receipt.NewRevision == nil || *receipt.NewRevision != sourceB {
		t.Fatalf("metadata-only check should attest selected source %s: %s", sourceB, formatMetadataCheckReceipt(receipt))
	}
	if err := os.WriteFile(identityPath, []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}
	receipt = check()
	if receipt.OldRevision == nil || *receipt.OldRevision != sourceA {
		t.Fatalf("malformed identity should fall back to embedded source %s: %s", sourceA, formatMetadataCheckReceipt(receipt))
	}
}

func formatMetadataCheckReceipt(receipt selfUpdateReceipt) string {
	old, next := "<nil>", "<nil>"
	if receipt.OldRevision != nil {
		old = *receipt.OldRevision
	}
	if receipt.NewRevision != nil {
		next = *receipt.NewRevision
	}
	return fmt.Sprintf("old=%s new=%s status=%s detail=%s", old, next, receipt.Status, receipt.Detail)
}
