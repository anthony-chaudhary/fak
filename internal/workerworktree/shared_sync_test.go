package workerworktree

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// deleteAddRepo is a real repository whose base commit carries two files the
// worker will delete, one it will modify, and nothing yet at the path it adds —
// the delete-plus-add shape whose post-CAS `git checkout <new> -- <paths>` used
// to abort on the deleted pathspecs and leave the root staged as the inverse of
// the landed commit (#13538).
type deleteAddRepo struct {
	root, wt, base string
	paths          []string
}

var deleteAddPaths = []string{"pkg/gone.go", "pkg/gone_test.go", "pkg/keep.go", "pkg/added.go"}

func newDeleteAddRepo(t *testing.T) deleteAddRepo {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	mustGit(t, root, "init", "-q", "-b", "main")
	mustGit(t, root, "config", "user.email", "sync@test.invalid")
	mustGit(t, root, "config", "user.name", "Sync Test")
	mustGit(t, root, "config", "commit.gpgsign", "false")
	writeRepoFile(t, root, "pkg/gone.go", "package pkg\n\nconst Gone = 1\n")
	writeRepoFile(t, root, "pkg/gone_test.go", "package pkg\n")
	writeRepoFile(t, root, "pkg/keep.go", "package pkg\n\nconst Keep = 1\n")
	mustGit(t, root, "add", "pkg")
	mustGit(t, root, "commit", "-q", "-m", "base")
	base := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))

	prep := Prepare(root, "shared-sync", "delete-add", base, t.TempDir(), nil)
	if !prep.OK {
		t.Fatalf("prepare: %+v", prep)
	}
	mustGit(t, prep.Path, "rm", "-q", "pkg/gone.go", "pkg/gone_test.go")
	writeRepoFile(t, prep.Path, "pkg/keep.go", "package pkg\n\nconst Keep = 2\n")
	writeRepoFile(t, prep.Path, "pkg/added.go", "package pkg\n\nconst Added = Keep\n")
	mustGit(t, prep.Path, "add", "pkg")
	mustGit(t, prep.Path, "commit", "-q", "-m", "fix(pkg): replace gone with added (fak workerworktree)")
	return deleteAddRepo{root: root, wt: prep.Path, base: base, paths: append([]string(nil), deleteAddPaths...)}
}

func writeRepoFile(t *testing.T, root, rel, content string) {
	t.Helper()
	abs := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// landWire is the machine-readable land result as `fak worktree worker land`
// emits it and fak-flow parses it. The assertions go through this wire shape so
// the regression also pins the contract callers actually see.
type landWire struct {
	OK             bool   `json:"ok"`
	Committed      bool   `json:"committed"`
	Code           string `json:"code"`
	CommitSHA      string `json:"commit_sha"`
	RecoveryAction string `json:"recovery_action"`
	SharedSync     *struct {
		Status    string   `json:"status"`
		Deleted   []string `json:"deleted"`
		Preserved []string `json:"preserved"`
		Unsynced  []string `json:"unsynced"`
	} `json:"shared_sync"`
}

func landWireOf(t *testing.T, res Result) landWire {
	t.Helper()
	data, err := json.Marshal(res)
	if err != nil {
		t.Fatal(err)
	}
	var w landWire
	if err := json.Unmarshal(data, &w); err != nil {
		t.Fatal(err)
	}
	return w
}

// assertRootMatchesHead is the #13538 witness: for every landed path the shared
// root's index equals HEAD and its working tree equals the index, and the files
// HEAD no longer carries are gone from disk.
func assertRootMatchesHead(t *testing.T, root string, paths []string) {
	t.Helper()
	status := mustGit(t, root, append([]string{"status", "--porcelain", "--untracked-files=all", "--"}, paths...)...)
	if strings.TrimSpace(status) != "" {
		t.Fatalf("shared root does not match HEAD for landed paths:\n%s", status)
	}
	for _, p := range paths {
		_, err := os.Stat(filepath.Join(root, filepath.FromSlash(p)))
		inHead := strings.TrimSpace(mustGit(t, root, "ls-tree", "--name-only", "HEAD", "--", p)) != ""
		if inHead && err != nil {
			t.Fatalf("%s is in HEAD but missing from the root worktree: %v", p, err)
		}
		if !inHead && !os.IsNotExist(err) {
			t.Fatalf("%s was deleted by the land but is still on disk in the root (stat err=%v)", p, err)
		}
	}
}

func TestLandIsolatedSyncsLandedDeletionsIntoSharedRoot(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "1")
	r := newDeleteAddRepo(t)
	res := landWireOf(t, Land(r.root, r.wt, r.base, "", r.paths, nil, nil))
	if !res.OK || !res.Committed || res.Code != LandResultSuccess {
		t.Fatalf("land: %+v", res)
	}
	if head := strings.TrimSpace(mustGit(t, r.root, "rev-parse", "HEAD")); head != res.CommitSHA {
		t.Fatalf("HEAD=%s, want landed %s", head, res.CommitSHA)
	}
	assertRootMatchesHead(t, r.root, r.paths)
	if res.SharedSync == nil || res.SharedSync.Status != "synced" ||
		strings.Join(res.SharedSync.Deleted, ",") != "pkg/gone.go,pkg/gone_test.go" {
		t.Fatalf("shared sync receipt: %+v", res.SharedSync)
	}
}

func TestAcceptPreparedLandSyncsLandedDeletionsIntoSharedRoot(t *testing.T) {
	r := newDeleteAddRepo(t)
	msg := filepath.Join(t.TempDir(), "message.txt")
	if err := os.WriteFile(msg, []byte("fix(pkg): replace gone with added (fak workerworktree)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	binding := ProspectiveVerificationBinding{Command: "go test ./...", Tags: []string{"unit"}}
	pass := func(string, error) Result { return Result{OK: true} }
	receipt, prep := PrepareProspectiveLand(r.root, r.wt, r.base, msg, r.paths, binding, nil, pass, nil)
	if !prep.OK || prep.Code != LandResultPrepared {
		t.Fatalf("prepare: %+v", prep)
	}
	res := landWireOf(t, AcceptPreparedLand(r.root, r.wt, PreparedLandExpectation{ReceiptID: receipt.ReceiptID, Paths: r.paths, Verification: binding}, nil))
	if !res.OK || !res.Committed || res.Code != LandResultSuccess {
		t.Fatalf("accept: %+v", res)
	}
	assertRootMatchesHead(t, r.root, r.paths)
	if res.SharedSync == nil || res.SharedSync.Status != "synced" {
		t.Fatalf("shared sync receipt: %+v", res.SharedSync)
	}
}

// A root copy of a deleted file that carries someone's local edit is peer WIP:
// the sync must drop the stale index entry (so the index matches HEAD) but keep
// the edited bytes on disk, and the result must say so loudly instead of
// reporting a clean success.
func TestLandIsolatedPreservesEditedRootCopyOfDeletedPath(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "1")
	r := newDeleteAddRepo(t)
	const peerEdit = "package pkg\n\nconst Gone = 2 // peer WIP\n"
	writeRepoFile(t, r.root, "pkg/gone.go", peerEdit)

	res := landWireOf(t, Land(r.root, r.wt, r.base, "", r.paths, nil, nil))
	if !res.OK || !res.Committed {
		t.Fatalf("the land itself must still succeed: %+v", res)
	}
	if res.Code != "landed-root-sync-incomplete" || res.SharedSync == nil || res.SharedSync.Status != "incomplete" {
		t.Fatalf("peer WIP on a deleted path must yield the typed incomplete outcome: %+v", res)
	}
	if strings.Join(res.SharedSync.Preserved, ",") != "pkg/gone.go" || !strings.Contains(res.RecoveryAction, "pkg/gone.go") {
		t.Fatalf("receipt must name the preserved path and its recovery: sync=%+v recovery=%q", res.SharedSync, res.RecoveryAction)
	}
	got, err := os.ReadFile(filepath.Join(r.root, "pkg", "gone.go"))
	if err != nil || string(got) != peerEdit {
		t.Fatalf("peer edit clobbered: %q err=%v", got, err)
	}
	if staged := strings.TrimSpace(mustGit(t, r.root, "diff", "--cached", "--name-only", "HEAD")); staged != "" {
		t.Fatalf("root index must match HEAD even while WIP is preserved; staged:\n%s", staged)
	}
	assertRootMatchesHead(t, r.root, []string{"pkg/gone_test.go", "pkg/keep.go", "pkg/added.go"})
}
