package wipref

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/committedtree"
)

func testGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	configureDispatchHelperCommand(cmd)
	cmd.Env = append(os.Environ(),
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_CONFIG_GLOBAL="+filepath.Join(dir, "absent-global-config"),
		"GIT_CONFIG_SYSTEM="+filepath.Join(dir, "absent-system-config"),
		"HOME="+dir,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return strings.TrimSpace(string(out))
}

func setupTestRepo(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	testGit(t, dir, "init", "-q")
	testGit(t, dir, "config", "user.email", "test@wip.local")
	testGit(t, dir, "config", "user.name", "wip test")
	testGit(t, dir, "config", "commit.gpgSign", "false")
	testGit(t, dir, "config", "core.autocrlf", "false")

	// Base commit
	if err := os.WriteFile(filepath.Join(dir, "base.txt"), []byte("base content\n"), 0644); err != nil {
		t.Fatalf("write base.txt: %v", err)
	}
	testGit(t, dir, "add", "base.txt")
	baseTree := testGit(t, dir, "write-tree")
	baseCommit := testGit(t, dir, "commit-tree", baseTree, "-m", "base commit")
	testGit(t, dir, "update-ref", "HEAD", baseCommit)

	return dir, baseCommit
}

func createCheckpoint(t *testing.T, dir, session, baseCommit string, files map[string]string) (string, string) {
	t.Helper()
	// Create checkpoint tree using git add / write-tree on temporary index
	tmpIdx := filepath.Join(t.TempDir(), "temp_index")
	cmdSeed := exec.Command("git", "read-tree", baseCommit)
	cmdSeed.Dir = dir
	cmdSeed.Env = append(os.Environ(), "GIT_INDEX_FILE="+tmpIdx)
	configureDispatchHelperCommand(cmdSeed)
	if out, err := cmdSeed.CombinedOutput(); err != nil {
		t.Fatalf("seed temp index: %v\n%s", err, out)
	}

	for relPath, content := range files {
		fullPath := filepath.Join(dir, relPath)
		if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
			t.Fatalf("mkdir %s: %v", relPath, err)
		}
		if err := os.WriteFile(fullPath, []byte(content), 0644); err != nil {
			t.Fatalf("write %s: %v", relPath, err)
		}
		cmdAdd := exec.Command("git", "add", relPath)
		cmdAdd.Dir = dir
		cmdAdd.Env = append(os.Environ(), "GIT_INDEX_FILE="+tmpIdx)
		configureDispatchHelperCommand(cmdAdd)
		if out, err := cmdAdd.CombinedOutput(); err != nil {
			t.Fatalf("add %s: %v\n%s", relPath, err, out)
		}
	}

	cmdWriteTree := exec.Command("git", "write-tree")
	cmdWriteTree.Dir = dir
	cmdWriteTree.Env = append(os.Environ(), "GIT_INDEX_FILE="+tmpIdx)
	configureDispatchHelperCommand(cmdWriteTree)
	treeOut, err := cmdWriteTree.CombinedOutput()
	if err != nil {
		t.Fatalf("write-tree: %v\n%s", err, treeOut)
	}
	tree := strings.TrimSpace(string(treeOut))

	stamp := Stamp{
		SessionID:      session,
		StartSHA:       baseCommit,
		Buildable:      true,
		CheckpointedAt: time.Now().Unix(),
	}
	msg, err := EncodeStamp(stamp)
	if err != nil {
		t.Fatalf("encode stamp: %v", err)
	}

	cpCommit := testGit(t, dir, "commit-tree", tree, "-p", baseCommit, "-m", msg)
	ref := SessionRef(session)
	testGit(t, dir, "update-ref", ref, cpCommit)

	return ref, cpCommit
}

func TestRunRecoveryDrill_ValidCheckpoint(t *testing.T) {
	dir, baseCommit := setupTestRepo(t)

	session := "session-valid-1"
	ref, cpCommit := createCheckpoint(t, dir, session, baseCommit, map[string]string{
		"file1.txt":     "file 1 content\n",
		"sub/file2.txt": "file 2 content in subfolder\n",
	})

	// Add an uncommitted untracked file to verify main checkout preservation
	untrackedPath := filepath.Join(dir, "uncommitted_work.txt")
	untrackedContent := "this working tree file must never be touched\n"
	if err := os.WriteFile(untrackedPath, []byte(untrackedContent), 0644); err != nil {
		t.Fatalf("write untracked: %v", err)
	}

	report, err := RunRecoveryDrill(context.Background(), dir, DrillOptions{})
	if err != nil {
		t.Fatalf("RunRecoveryDrill failed: %v", err)
	}

	if report.Schema != RecoveryDrillSchema {
		t.Errorf("schema mismatch: got %q, want %q", report.Schema, RecoveryDrillSchema)
	}
	if report.TotalDrilled != 1 {
		t.Errorf("expected 1 drilled, got %d", report.TotalDrilled)
	}
	if report.SuccessCount != 1 || report.FailureCount != 0 {
		t.Errorf("expected 1 success 0 failure, got %d success %d failure", report.SuccessCount, report.FailureCount)
	}
	if !report.MainTreePreserved {
		t.Errorf("expected MainTreePreserved to be true")
	}

	res := report.Results[0]
	if res.Ref != ref {
		t.Errorf("ref mismatch: got %q, want %q", res.Ref, ref)
	}
	if res.Session != session {
		t.Errorf("session mismatch: got %q, want %q", res.Session, session)
	}
	if res.CommitSHA != cpCommit {
		t.Errorf("commit mismatch: got %q, want %q", res.CommitSHA, cpCommit)
	}
	if res.Status != "PASS" {
		t.Errorf("expected status PASS, got %q (detail: %s)", res.Status, res.Detail)
	}
	if !res.ByteHashesMatch {
		t.Errorf("expected ByteHashesMatch true")
	}
	if !res.AttributionMatch {
		t.Errorf("expected AttributionMatch true")
	}
	if res.RestoredPathCount != 2 { // file1.txt, sub/file2.txt (checkpoint delta)
		t.Errorf("expected 2 restored paths, got %d", res.RestoredPathCount)
	}

	// Verify working tree file preserved
	readBack, err := os.ReadFile(untrackedPath)
	if err != nil {
		t.Fatalf("read untracked: %v", err)
	}
	if string(readBack) != untrackedContent {
		t.Errorf("working tree file corrupted: got %q, want %q", string(readBack), untrackedContent)
	}
}

func TestRunRecoveryDrill_RestoresExportIgnoredPaths(t *testing.T) {
	dir, baseCommit := setupTestRepo(t)

	session := "session-export-ignore"
	ref, _ := createCheckpoint(t, dir, session, baseCommit, map[string]string{
		".gitattributes": "secret.txt export-ignore\nvendor/** export-ignore\n",
		"secret.txt":     "export-ignored content\n",
		"vendor/lib.h":   "#define LIB 1\n",
		"file1.txt":      "ordinary content\n",
	})

	report, err := RunRecoveryDrill(context.Background(), dir, DrillOptions{Session: session})
	if err != nil {
		t.Fatalf("RunRecoveryDrill failed: %v", err)
	}
	res := report.Results[0]
	if res.Ref != ref {
		t.Fatalf("ref mismatch: got %q, want %q", res.Ref, ref)
	}
	if res.Status != "PASS" {
		t.Fatalf("expected PASS, got %q (detail: %s)", res.Status, res.Detail)
	}
	if res.RestoredPathCount != 4 {
		t.Errorf("expected 4 restored paths including export-ignored ones, got %d", res.RestoredPathCount)
	}
}

func TestRunRecoveryDrill_MissingObject(t *testing.T) {
	dir, baseCommit := setupTestRepo(t)

	session := "session-missing"
	ref, cpCommit := createCheckpoint(t, dir, session, baseCommit, map[string]string{
		"missing.txt": "data to delete\n",
	})

	// Delete the loose object file for cpCommit so it is missing
	sub := cpCommit[:2]
	rest := cpCommit[2:]
	objFile := filepath.Join(dir, ".git", "objects", sub, rest)
	_ = os.Chmod(objFile, 0666)
	if err := os.Remove(objFile); err != nil {
		t.Fatalf("remove object file: %v", err)
	}

	report, err := RunRecoveryDrill(context.Background(), dir, DrillOptions{Session: session})
	if err != nil {
		t.Fatalf("RunRecoveryDrill failed: %v", err)
	}

	if report.TotalDrilled != 1 {
		t.Fatalf("expected 1 drilled, got %d", report.TotalDrilled)
	}
	if report.FailureCount != 1 || report.SuccessCount != 0 {
		t.Errorf("expected failure, got %d success %d failure", report.SuccessCount, report.FailureCount)
	}
	if report.Results[0].Status != "MISSING_OBJECT" {
		t.Errorf("expected status MISSING_OBJECT, got %q (detail: %s)", report.Results[0].Status, report.Results[0].Detail)
	}
	if report.Results[0].Ref != ref {
		t.Errorf("expected ref %q, got %q", ref, report.Results[0].Ref)
	}
	if !report.MainTreePreserved {
		t.Errorf("expected MainTreePreserved true")
	}
}

func TestRunRecoveryDrill_CorruptObject(t *testing.T) {
	dir, baseCommit := setupTestRepo(t)

	session := "session-corrupt"
	_, cpCommit := createCheckpoint(t, dir, session, baseCommit, map[string]string{
		"corrupt.txt": "data to corrupt\n",
	})

	// Locate the loose object file for cpCommit
	sub := cpCommit[:2]
	rest := cpCommit[2:]
	objFile := filepath.Join(dir, ".git", "objects", sub, rest)

	// Remove read-only flag on Windows if set, and write garbage bytes
	_ = os.Chmod(objFile, 0666)
	if err := os.WriteFile(objFile, []byte{0x01, 0x02, 0x03, 0x04}, 0666); err != nil {
		t.Fatalf("corrupt object file: %v", err)
	}

	report, err := RunRecoveryDrill(context.Background(), dir, DrillOptions{Session: session})
	if err != nil {
		t.Fatalf("RunRecoveryDrill failed: %v", err)
	}

	if report.TotalDrilled != 1 {
		t.Fatalf("expected 1 drilled, got %d", report.TotalDrilled)
	}
	if report.FailureCount != 1 {
		t.Errorf("expected failure, got %d success %d failure", report.SuccessCount, report.FailureCount)
	}
	if report.Results[0].Status != "CORRUPT_OBJECT" {
		t.Errorf("expected status CORRUPT_OBJECT, got %q (detail: %s)", report.Results[0].Status, report.Results[0].Detail)
	}
	if !report.MainTreePreserved {
		t.Errorf("expected MainTreePreserved true")
	}
}

func TestRunRecoveryDrill_EmptyTree(t *testing.T) {
	dir, baseCommit := setupTestRepo(t)

	session := "session-empty"
	emptyTree := "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
	stamp := Stamp{
		SessionID:      session,
		StartSHA:       baseCommit,
		MetadataOnly:   true,
		CheckpointedAt: time.Now().Unix(),
	}
	msg, _ := EncodeStamp(stamp)
	cpCommit := testGit(t, dir, "commit-tree", emptyTree, "-p", baseCommit, "-m", msg)
	ref := SessionRef(session)
	testGit(t, dir, "update-ref", ref, cpCommit)

	report, err := RunRecoveryDrill(context.Background(), dir, DrillOptions{Session: session})
	if err != nil {
		t.Fatalf("RunRecoveryDrill failed: %v", err)
	}

	if report.TotalDrilled != 1 {
		t.Fatalf("expected 1 drilled, got %d", report.TotalDrilled)
	}
	if report.Results[0].Status != "EMPTY_TREE" {
		t.Errorf("expected status EMPTY_TREE, got %q (detail: %s)", report.Results[0].Status, report.Results[0].Detail)
	}
	if report.FailureCount != 1 {
		t.Errorf("expected failure count 1, got %d", report.FailureCount)
	}
	if !report.MainTreePreserved {
		t.Errorf("expected MainTreePreserved true")
	}
}

func TestRunRecoveryDrill_MainTreePreserved(t *testing.T) {
	dir, baseCommit := setupTestRepo(t)

	// Create 2 checkpoints
	createCheckpoint(t, dir, "session-a", baseCommit, map[string]string{
		"a.txt": "content A\n",
	})
	createCheckpoint(t, dir, "session-b", baseCommit, map[string]string{
		"b.txt": "content B\n",
	})

	// Dirty the working tree: tracked modify + untracked new file
	trackedFile := filepath.Join(dir, "base.txt")
	newTrackedContent := "modified base content\n"
	if err := os.WriteFile(trackedFile, []byte(newTrackedContent), 0644); err != nil {
		t.Fatalf("write modified base.txt: %v", err)
	}
	untrackedFile := filepath.Join(dir, "untracked.txt")
	untrackedContent := "brand new untracked file\n"
	if err := os.WriteFile(untrackedFile, []byte(untrackedContent), 0644); err != nil {
		t.Fatalf("write untracked: %v", err)
	}

	report, err := RunRecoveryDrill(context.Background(), dir, DrillOptions{})
	if err != nil {
		t.Fatalf("RunRecoveryDrill failed: %v", err)
	}

	if report.TotalDrilled != 2 {
		t.Errorf("expected 2 drilled, got %d", report.TotalDrilled)
	}
	if report.SuccessCount != 2 {
		t.Errorf("expected 2 success, got %d", report.SuccessCount)
	}
	if !report.MainTreePreserved {
		t.Errorf("expected MainTreePreserved true")
	}

	// Confirm exact contents unchanged
	baseRead, _ := os.ReadFile(trackedFile)
	if string(baseRead) != newTrackedContent {
		t.Errorf("tracked file changed")
	}
	untrackedRead, _ := os.ReadFile(untrackedFile)
	if string(untrackedRead) != untrackedContent {
		t.Errorf("untracked file changed")
	}
}

func TestSnapshotFailure_DeadlineAndBudgetAreTimeouts(t *testing.T) {
	cases := map[string]error{
		"package budget":        &committedtree.BudgetError{Budget: time.Second},
		"wrapped deadline":      fmt.Errorf("committed tree extraction canceled: %w", context.DeadlineExceeded),
		"budget inside ls-tree": fmt.Errorf("git ls-tree: %w", &committedtree.BudgetError{Budget: time.Second}),
	}
	for name, err := range cases {
		status, _ := snapshotFailure(err)
		if status != "TIMEOUT" {
			t.Errorf("%s: status %q, want TIMEOUT", name, status)
		}
	}
}

func TestSnapshotFailure_CorruptionStaysCorrupt(t *testing.T) {
	status, _ := snapshotFailure(errors.New("git cat-file: exit status 128 (fatal: inflate: data stream error)"))
	if status != "CORRUPT_OBJECT" {
		t.Errorf("status %q, want CORRUPT_OBJECT", status)
	}
}

func TestRestoreSnapshot_BudgetExpiryIsTimeoutNotCorrupt(t *testing.T) {
	t.Setenv(committedtree.ExtractionBudgetEnv, "1ns")
	dir, baseCommit := setupTestRepo(t)
	ref, _ := createCheckpoint(t, dir, "session-budget", baseCommit, map[string]string{
		"a.txt": "a\n",
	})
	tree := testGit(t, dir, "rev-parse", ref+"^{tree}")

	_, status, detail := restoreSnapshot(context.Background(), dir, tree, t.TempDir(), map[string]bool{"a.txt": true})
	if status != "TIMEOUT" {
		t.Fatalf("status %q, want TIMEOUT (detail: %s)", status, detail)
	}
}

func TestRestoreSnapshot_CallerDeadlineIsTimeoutNotCorrupt(t *testing.T) {
	dir, baseCommit := setupTestRepo(t)
	ref, _ := createCheckpoint(t, dir, "session-deadline", baseCommit, map[string]string{
		"a.txt": "a\n",
	})
	tree := testGit(t, dir, "rev-parse", ref+"^{tree}")

	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	_, status, detail := restoreSnapshot(ctx, dir, tree, t.TempDir(), nil)
	if status != "TIMEOUT" {
		t.Fatalf("status %q, want TIMEOUT (detail: %s)", status, detail)
	}
}

func TestRestoreSnapshot_ReaderFailureStandsWhenWriterIsHealthy(t *testing.T) {
	dir, baseCommit := setupTestRepo(t)
	ref, _ := createCheckpoint(t, dir, "session-reader-fail", baseCommit, map[string]string{
		"a.txt": "a\n",
		"b.txt": "b\n",
	})
	tree := testGit(t, dir, "rev-parse", ref+"^{tree}")

	// A directory where a.txt must be written makes the reader's own write fail
	// while the writer still has blobs to send.
	scratch := t.TempDir()
	if err := os.MkdirAll(filepath.Join(scratch, "a.txt"), 0755); err != nil {
		t.Fatalf("mkdir blocker: %v", err)
	}
	_, status, detail := restoreSnapshot(context.Background(), dir, tree, scratch, nil)
	if status != "HASH_MISMATCH" {
		t.Fatalf("status %q, want HASH_MISMATCH from the reader (detail: %s)", status, detail)
	}
}

func TestRestoreSnapshot_PathSelectionSkipsUnrelatedCorruptBlob(t *testing.T) {
	dir, baseCommit := setupTestRepo(t)
	ref, _ := createCheckpoint(t, dir, "session-partial", baseCommit, map[string]string{
		"a.txt": "a\n",
	})
	tree := testGit(t, dir, "rev-parse", ref+"^{tree}")

	// base.txt is in the checkpoint tree but outside the selection. Reading it
	// would fail, so a pass shows the restore never read unselected blobs.
	baseBlob := testGit(t, dir, "rev-parse", baseCommit+":base.txt")
	objFile := filepath.Join(dir, ".git", "objects", baseBlob[:2], baseBlob[2:])
	_ = os.Chmod(objFile, 0666)
	if err := os.WriteFile(objFile, []byte{0x01, 0x02, 0x03, 0x04}, 0666); err != nil {
		t.Fatalf("corrupt unrelated blob: %v", err)
	}

	scratch := t.TempDir()
	restored, status, detail := restoreSnapshot(context.Background(), dir, tree, scratch, map[string]bool{"a.txt": true})
	if status != "" {
		t.Fatalf("status %q, want success (detail: %s)", status, detail)
	}
	if restored != 1 {
		t.Errorf("restored %d, want 1", restored)
	}
	if _, err := os.Stat(filepath.Join(scratch, "base.txt")); !os.IsNotExist(err) {
		t.Errorf("unselected base.txt was restored (stat err %v)", err)
	}
}
