package goalsync

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDefaultTarget(t *testing.T) {
	wsRoot := "/path/to/workspace"
	expected := filepath.Join(wsRoot, "..", "fak-private", "goals", "fak")
	got := DefaultTarget(wsRoot)
	if got != expected {
		t.Fatalf("DefaultTarget(%q) = %q, want %q", wsRoot, got, expected)
	}
}

func setupSourceWorkspace(t *testing.T, wsRoot string) (string, string) {
	t.Helper()
	goalsDir := filepath.Join(wsRoot, "goals")
	subagentsDir := filepath.Join(goalsDir, "subagents")
	fakDir := filepath.Join(wsRoot, ".fak")
	goalParkDir := filepath.Join(fakDir, "goal-park")

	if err := os.MkdirAll(subagentsDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(goalParkDir, 0755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(filepath.Join(goalsDir, "GOAL-test.md"), []byte("# Goal Test\nSpec content\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subagentsDir, "GOAL-sub.md"), []byte("# Subagent Goal\nSub content\n"), 0644); err != nil {
		t.Fatal(err)
	}
	regPath := filepath.Join(fakDir, "goals.json")
	if err := os.WriteFile(regPath, []byte(`{"schema":"fak-goal-registry/1","goals":[]}`), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(goalParkDir, "park-1.json"), []byte(`{"schema":"fak.goal-park.v1","goal":"g1"}`), 0644); err != nil {
		t.Fatal(err)
	}

	return regPath, goalParkDir
}

func TestDiscoverSourceAndTarget(t *testing.T) {
	ws := t.TempDir()
	regPath, parkDir := setupSourceWorkspace(t, ws)

	srcs, err := DiscoverSource(ws, regPath, parkDir)
	if err != nil {
		t.Fatalf("DiscoverSource: %v", err)
	}
	if len(srcs) != 4 {
		t.Fatalf("expected 4 artifacts, got %d: %+v", len(srcs), srcs)
	}

	expectedRelPaths := map[string]ArtifactKind{
		"goals/GOAL-test.md":          KindSpec,
		"goals/subagents/GOAL-sub.md": KindSubagent,
		"registry/goals.json":         KindRegistry,
		"goal-park/park-1.json":       KindPark,
	}

	for _, a := range srcs {
		kind, ok := expectedRelPaths[a.RelPath]
		if !ok {
			t.Errorf("unexpected artifact RelPath %s", a.RelPath)
			continue
		}
		if a.Kind != kind {
			t.Errorf("artifact %s has kind %s, want %s", a.RelPath, a.Kind, kind)
		}
		if len(a.Hash) != 64 {
			t.Errorf("artifact %s has invalid hash %q", a.RelPath, a.Hash)
		}
		if a.Size <= 0 {
			t.Errorf("artifact %s has invalid size %d", a.RelPath, a.Size)
		}
		if a.ModTime.IsZero() {
			t.Errorf("artifact %s has zero mod time", a.RelPath)
		}
	}

	targetDir := t.TempDir()
	for _, a := range srcs {
		dest := filepath.Join(targetDir, filepath.FromSlash(a.RelPath))
		if err := os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(a.AbsPath)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dest, data, 0644); err != nil {
			t.Fatal(err)
		}
	}

	tgts, err := DiscoverTarget(targetDir)
	if err != nil {
		t.Fatalf("DiscoverTarget: %v", err)
	}
	if len(tgts) != 4 {
		t.Fatalf("expected 4 target artifacts, got %d: %+v", len(tgts), tgts)
	}
	for _, a := range tgts {
		kind, ok := expectedRelPaths[a.RelPath]
		if !ok {
			t.Errorf("unexpected target artifact %s", a.RelPath)
			continue
		}
		if a.Kind != kind {
			t.Errorf("target artifact %s has kind %s, want %s", a.RelPath, a.Kind, kind)
		}
	}
}

func TestStatus(t *testing.T) {
	ws := t.TempDir()
	regPath, parkDir := setupSourceWorkspace(t, ws)
	targetDir := t.TempDir()

	// 1. Target empty: all should be ActionPush ("only in source")
	st, err := Status(ws, targetDir, regPath, parkDir)
	if err != nil {
		t.Fatalf("Status empty target: %v", err)
	}
	if st.TotalCount != 4 || st.PushCount != 4 || st.InSyncCount != 0 || st.PullCount != 0 {
		t.Fatalf("unexpected counts: total=%d push=%d insync=%d pull=%d", st.TotalCount, st.PushCount, st.InSyncCount, st.PullCount)
	}
	for _, it := range st.Items {
		if it.Action != ActionPush || it.Reason != "only in source" {
			t.Errorf("item %s has action %s reason %s", it.RelPath, it.Action, it.Reason)
		}
	}

	// 2. Synchronize to target: all should be ActionNoop ("in sync")
	report, err := Push(ws, targetDir, regPath, parkDir, false, false, false)
	if err != nil {
		t.Fatalf("Push failed: %v", err)
	}
	if len(report.Transferred) != 4 {
		t.Fatalf("expected 4 transferred, got %d", len(report.Transferred))
	}

	st, err = Status(ws, targetDir, regPath, parkDir)
	if err != nil {
		t.Fatalf("Status synced: %v", err)
	}
	if st.TotalCount != 4 || st.InSyncCount != 4 || st.PushCount != 0 || st.PullCount != 0 {
		t.Fatalf("unexpected counts after push: total=%d inSync=%d push=%d pull=%d", st.TotalCount, st.InSyncCount, st.PushCount, st.PullCount)
	}

	// 3. Add file only in target -> ActionPull ("only in target")
	extraTargetFile := filepath.Join(targetDir, "goals", "GOAL-extra.md")
	if err := os.WriteFile(extraTargetFile, []byte("extra"), 0644); err != nil {
		t.Fatal(err)
	}
	st, err = Status(ws, targetDir, regPath, parkDir)
	if err != nil {
		t.Fatal(err)
	}
	if st.PullCount != 1 {
		t.Fatalf("expected 1 pull count, got %d", st.PullCount)
	}

	// 4. Update source file to be newer -> ActionPush ("source newer")
	srcGoal := filepath.Join(ws, "goals", "GOAL-test.md")
	time.Sleep(10 * time.Millisecond)
	if err := os.WriteFile(srcGoal, []byte("# Updated spec\n"), 0644); err != nil {
		t.Fatal(err)
	}
	newerTime := time.Now().Add(5 * time.Minute)
	_ = os.Chtimes(srcGoal, newerTime, newerTime)

	st, err = Status(ws, targetDir, regPath, parkDir)
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, it := range st.Items {
		if it.RelPath == "goals/GOAL-test.md" {
			found = true
			if it.Action != ActionPush || it.Reason != "source newer" {
				t.Errorf("GOAL-test.md action=%s reason=%s, want push / source newer", it.Action, it.Reason)
			}
		}
	}
	if !found {
		t.Fatal("GOAL-test.md not found in status items")
	}

	// 5. Update target file to be newer -> ActionPull ("target newer")
	tgtSub := filepath.Join(targetDir, "goals", "subagents", "GOAL-sub.md")
	if err := os.WriteFile(tgtSub, []byte("# Updated sub in target\n"), 0644); err != nil {
		t.Fatal(err)
	}
	targetNewerTime := time.Now().Add(10 * time.Minute)
	_ = os.Chtimes(tgtSub, targetNewerTime, targetNewerTime)

	st, err = Status(ws, targetDir, regPath, parkDir)
	if err != nil {
		t.Fatal(err)
	}
	found = false
	for _, it := range st.Items {
		if it.RelPath == "goals/subagents/GOAL-sub.md" {
			found = true
			if it.Action != ActionPull || it.Reason != "target newer" {
				t.Errorf("GOAL-sub.md action=%s reason=%s, want pull / target newer", it.Action, it.Reason)
			}
		}
	}
	if !found {
		t.Fatal("GOAL-sub.md not found in status items")
	}

	// 6. Same modtime, different hash -> ActionConflict ("content mismatch")
	fixedTime := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	_ = os.WriteFile(srcGoal, []byte("variant A"), 0644)
	_ = os.Chtimes(srcGoal, fixedTime, fixedTime)
	tgtGoal := filepath.Join(targetDir, "goals", "GOAL-test.md")
	_ = os.WriteFile(tgtGoal, []byte("variant B"), 0644)
	_ = os.Chtimes(tgtGoal, fixedTime, fixedTime)

	st, err = Status(ws, targetDir, regPath, parkDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, it := range st.Items {
		if it.RelPath == "goals/GOAL-test.md" {
			if it.Action != ActionConflict || it.Reason != "content mismatch" {
				t.Errorf("GOAL-test.md action=%s reason=%s, want conflict / content mismatch", it.Action, it.Reason)
			}
		}
	}
}

func TestPushAndPullRoundTrip(t *testing.T) {
	srcWs := t.TempDir()
	regPath, parkDir := setupSourceWorkspace(t, srcWs)
	targetDir := t.TempDir()

	// Push from srcWs to targetDir
	report, err := Push(srcWs, targetDir, regPath, parkDir, false, false, false)
	if err != nil {
		t.Fatalf("initial push failed: %v", err)
	}
	if report.Schema != Schema {
		t.Fatalf("schema = %q, want %q", report.Schema, Schema)
	}
	if len(report.Transferred) != 4 {
		t.Fatalf("expected 4 transferred, got %d", len(report.Transferred))
	}
	if len(report.Skipped) != 0 {
		t.Fatalf("expected 0 skipped, got %d", len(report.Skipped))
	}

	// Repeated push should skip all
	report2, err := Push(srcWs, targetDir, regPath, parkDir, false, false, false)
	if err != nil {
		t.Fatalf("second push failed: %v", err)
	}
	if len(report2.Transferred) != 0 {
		t.Fatalf("expected 0 transferred on re-push, got %d", len(report2.Transferred))
	}
	if len(report2.Skipped) != 4 {
		t.Fatalf("expected 4 skipped on re-push, got %d", len(report2.Skipped))
	}

	// Now pull into a completely clean destination workspace
	destWs := t.TempDir()
	destReg := filepath.Join(destWs, ".fak", "goals.json")
	destPark := filepath.Join(destWs, ".fak", "goal-park")

	pullReport, err := Pull(destWs, targetDir, destReg, destPark, false, false)
	if err != nil {
		t.Fatalf("pull failed: %v", err)
	}
	if len(pullReport.Transferred) != 4 {
		t.Fatalf("expected 4 pulled, got %d", len(pullReport.Transferred))
	}

	// Verify all restored files in destWs match srcWs
	for _, rel := range []string{
		"goals/GOAL-test.md",
		"goals/subagents/GOAL-sub.md",
		"registry/goals.json",
		"goal-park/park-1.json",
	} {
		var srcFile, destFile string
		switch {
		case strings.HasPrefix(rel, "goals/"):
			srcFile = filepath.Join(srcWs, filepath.FromSlash(rel))
			destFile = filepath.Join(destWs, filepath.FromSlash(rel))
		case rel == "registry/goals.json":
			srcFile = regPath
			destFile = destReg
		case strings.HasPrefix(rel, "goal-park/"):
			srcFile = filepath.Join(parkDir, strings.TrimPrefix(rel, "goal-park/"))
			destFile = filepath.Join(destPark, strings.TrimPrefix(rel, "goal-park/"))
		}

		srcData, err := os.ReadFile(srcFile)
		if err != nil {
			t.Fatalf("read src %s: %v", srcFile, err)
		}
		destData, err := os.ReadFile(destFile)
		if err != nil {
			t.Fatalf("read dest %s: %v", destFile, err)
		}
		if string(srcData) != string(destData) {
			t.Fatalf("content mismatch for %s: got %q, want %q", rel, string(destData), string(srcData))
		}
	}
}

func TestPushAtomicAndSafeCopy(t *testing.T) {
	ws := t.TempDir()
	regPath, parkDir := setupSourceWorkspace(t, ws)
	targetDir := t.TempDir()

	// Dry-run should report transferred but not write anything
	dryReport, err := Push(ws, targetDir, regPath, parkDir, false, false, true)
	if err != nil {
		t.Fatalf("dry-run push failed: %v", err)
	}
	if len(dryReport.Transferred) != 4 {
		t.Fatalf("dry-run expected 4 transferred, got %d", len(dryReport.Transferred))
	}

	// Verify targetDir is still empty
	tgts, err := DiscoverTarget(targetDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(tgts) != 0 {
		t.Fatalf("dry-run wrote files to targetDir: %d", len(tgts))
	}

	// Non-dry-run writes files atomically
	report, err := Push(ws, targetDir, regPath, parkDir, false, false, false)
	if err != nil {
		t.Fatalf("real push failed: %v", err)
	}
	if len(report.Transferred) != 4 {
		t.Fatalf("real push expected 4 transferred, got %d", len(report.Transferred))
	}

	// Check atomic file existence and permissions
	specPath := filepath.Join(targetDir, "goals", "GOAL-test.md")
	fi, err := os.Stat(specPath)
	if err != nil {
		t.Fatalf("stat target file: %v", err)
	}
	if fi.Size() == 0 {
		t.Fatal("target file is empty")
	}

	// Test concurrency protection
	testFile := filepath.Join(ws, "goals", "GOAL-test.md")
	data, info, err := readWithConcurrencyProtection(testFile)
	if err != nil {
		t.Fatalf("concurrency read failed: %v", err)
	}
	if len(data) == 0 || info == nil {
		t.Fatal("empty concurrency read result")
	}
}

func TestPullConflictProtection(t *testing.T) {
	ws := t.TempDir()
	regPath, parkDir := setupSourceWorkspace(t, ws)
	targetDir := t.TempDir()

	// Initial push
	if _, err := Push(ws, targetDir, regPath, parkDir, false, false, false); err != nil {
		t.Fatalf("initial push failed: %v", err)
	}

	// Target has older content
	targetGoal := filepath.Join(targetDir, "goals", "GOAL-test.md")
	pastTime := time.Now().Add(-10 * time.Minute)
	if err := os.WriteFile(targetGoal, []byte("target older content\n"), 0644); err != nil {
		t.Fatal(err)
	}
	_ = os.Chtimes(targetGoal, pastTime, pastTime)

	// Local workspace has newer content
	localGoal := filepath.Join(ws, "goals", "GOAL-test.md")
	futureTime := time.Now().Add(10 * time.Minute)
	localContent := []byte("local newer content\n")
	if err := os.WriteFile(localGoal, localContent, 0644); err != nil {
		t.Fatal(err)
	}
	_ = os.Chtimes(localGoal, futureTime, futureTime)

	// Pull without force must fail and refuse to overwrite newer local file
	report, err := Pull(ws, targetDir, regPath, parkDir, false, false)
	if err == nil {
		t.Fatal("expected pull to fail due to conflict with newer local file")
	}
	if !strings.Contains(err.Error(), "conflict") {
		t.Fatalf("unexpected error message: %v", err)
	}

	// Verify local file was NOT overwritten
	currentLocalData, err := os.ReadFile(localGoal)
	if err != nil {
		t.Fatal(err)
	}
	if string(currentLocalData) != string(localContent) {
		t.Fatalf("local file was overwritten despite conflict protection: got %q, want %q", string(currentLocalData), string(localContent))
	}

	// Pull WITH force must succeed and overwrite local file
	report, err = Pull(ws, targetDir, regPath, parkDir, true, false)
	if err != nil {
		t.Fatalf("forced pull failed: %v", err)
	}
	if len(report.Transferred) != 1 || report.Transferred[0] != "goals/GOAL-test.md" {
		t.Fatalf("expected goals/GOAL-test.md to be transferred on force pull: %+v", report)
	}

	// Verify local file now has target content
	overwrittenData, err := os.ReadFile(localGoal)
	if err != nil {
		t.Fatal(err)
	}
	if string(overwrittenData) != "target older content\n" {
		t.Fatalf("forced pull did not overwrite: got %q", string(overwrittenData))
	}
}

func TestPushGitCommit(t *testing.T) {
	ws := t.TempDir()
	regPath, parkDir := setupSourceWorkspace(t, ws)

	// Set up targetDir inside a real git repository
	gitRoot := t.TempDir()
	cmdInit := exec.Command("git", "init", gitRoot)
	cmdInit.Env = isolatedGitEnv()
	if out, err := cmdInit.CombinedOutput(); err != nil {
		t.Skipf("git init not available: %s (%v)", string(out), err)
	}
	cfgEmail := exec.Command("git", "-C", gitRoot, "config", "user.email", "test@example.com")
	cfgEmail.Env = isolatedGitEnv()
	_ = cfgEmail.Run()
	cfgName := exec.Command("git", "-C", gitRoot, "config", "user.name", "Test User")
	cfgName.Env = isolatedGitEnv()
	_ = cfgName.Run()

	targetDir := filepath.Join(gitRoot, "goals", "fak")
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		t.Fatal(err)
	}

	report, err := Push(ws, targetDir, regPath, parkDir, true, false, false)
	if err != nil {
		t.Fatalf("push with commit failed: %v", err)
	}
	if !report.Committed {
		t.Fatal("expected report.Committed to be true")
	}

	// Verify git log has the commit
	logCmd := exec.Command("git", "-C", gitRoot, "log", "-1", "--oneline")
	logCmd.Env = isolatedGitEnv()
	out, err := logCmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git log failed: %s: %v", string(out), err)
	}
	if !strings.Contains(string(out), "chore(goals): sync goal artifacts (fak)") {
		t.Fatalf("unexpected git commit message: %s", string(out))
	}
}
