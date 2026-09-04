package gitbroker

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func setupTestRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()

	gitRun := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=tester",
			"GIT_AUTHOR_EMAIL=tester@example.com",
			"GIT_COMMITTER_NAME=tester",
			"GIT_COMMITTER_EMAIL=tester@example.com",
		)
		windowgate.ConfigureBackgroundCommand(cmd)
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			t.Fatalf("git %s: %v (%s)", strings.Join(args, " "), err, stderr.String())
		}
		return strings.TrimSpace(stdout.String())
	}

	gitRun("init", "-q")
	gitRun("config", "user.name", "tester")
	gitRun("config", "user.email", "tester@example.com")

	baselinePath := filepath.Join(repo, "baseline.txt")
	if err := os.WriteFile(baselinePath, []byte("baseline content\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitRun("add", "baseline.txt")
	gitRun("commit", "-qm", "initial baseline")

	return repo
}

func getGitOutput(t *testing.T, repo string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("git %s: %v (%s)", strings.Join(args, " "), err, stderr.String())
	}
	return strings.TrimSpace(stdout.String())
}

func TestShadowCheckpoint(t *testing.T) {
	repo := setupTestRepo(t)

	headBefore := getGitOutput(t, repo, "rev-parse", "HEAD")
	logBefore := getGitOutput(t, repo, "log", "--oneline")

	// Modify tracked file and create an untracked file.
	baselineFile := filepath.Join(repo, "baseline.txt")
	if err := os.WriteFile(baselineFile, []byte("modified baseline\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	untrackedFile := filepath.Join(repo, "untracked.txt")
	if err := os.WriteFile(untrackedFile, []byte("untracked content\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Create shadow checkpoint.
	ref, err := CreateShadowCheckpoint(repo)
	if err != nil {
		t.Fatalf("CreateShadowCheckpoint failed: %v", err)
	}
	if ref == "" {
		t.Fatal("expected non-empty ref from CreateShadowCheckpoint")
	}

	// Verify ref has 3 parents (HEAD, index, untracked).
	p1 := getGitOutput(t, repo, "rev-parse", "--verify", ref+"^1")
	p2 := getGitOutput(t, repo, "rev-parse", "--verify", ref+"^2")
	p3 := getGitOutput(t, repo, "rev-parse", "--verify", ref+"^3")
	if p1 != headBefore {
		t.Fatalf("parent 1 (%s) != headBefore (%s)", p1, headBefore)
	}
	if p2 == "" || p3 == "" {
		t.Fatalf("expected valid parents 2 and 3: p2=%q, p3=%q", p2, p3)
	}

	// Verify refs/stash was NOT created or modified.
	cmdStash := exec.Command("git", "-C", repo, "rev-parse", "--verify", "refs/stash")
	windowgate.ConfigureBackgroundCommand(cmdStash)
	if err := cmdStash.Run(); err == nil {
		t.Fatal("refs/stash should not exist for shadow checkpoint")
	}

	// Modify tracked file further and delete untracked file.
	if err := os.WriteFile(baselineFile, []byte("further modified baseline\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(untrackedFile); err != nil {
		t.Fatal(err)
	}

	// Restore shadow checkpoint.
	if err := RestoreShadowCheckpoint(repo, ref); err != nil {
		t.Fatalf("RestoreShadowCheckpoint failed: %v", err)
	}

	// Verify both tracked and untracked files are restored to exact snapshot state.
	trackedContent, err := os.ReadFile(baselineFile)
	if err != nil {
		t.Fatalf("read baseline: %v", err)
	}
	if string(trackedContent) != "modified baseline\n" {
		t.Fatalf("unexpected tracked content: %q", string(trackedContent))
	}

	untrackedContent, err := os.ReadFile(untrackedFile)
	if err != nil {
		t.Fatalf("read untracked: %v", err)
	}
	if string(untrackedContent) != "untracked content\n" {
		t.Fatalf("unexpected untracked content: %q", string(untrackedContent))
	}

	// Verify git log / HEAD commit has NOT changed.
	headAfter := getGitOutput(t, repo, "rev-parse", "HEAD")
	if headAfter != headBefore {
		t.Fatalf("HEAD changed: before=%s, after=%s", headBefore, headAfter)
	}
	logAfter := getGitOutput(t, repo, "log", "--oneline")
	if logAfter != logBefore {
		t.Fatalf("git log changed: before=%s, after=%s", logBefore, logAfter)
	}
}

func TestShadowCheckpoint_StagedAndUnstaged(t *testing.T) {
	repo := setupTestRepo(t)

	// Add second file to commit baseline.
	file2 := filepath.Join(repo, "second.txt")
	if err := os.WriteFile(file2, []byte("second initial\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_ = getGitOutput(t, repo, "add", "second.txt")
	_ = getGitOutput(t, repo, "commit", "-qm", "add second file")
	headBefore := getGitOutput(t, repo, "rev-parse", "HEAD")

	// Baseline: staged change on baseline.txt, unstaged on second.txt, untracked file.
	baselineFile := filepath.Join(repo, "baseline.txt")
	if err := os.WriteFile(baselineFile, []byte("staged modification\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_ = getGitOutput(t, repo, "add", "baseline.txt")

	if err := os.WriteFile(file2, []byte("unstaged modification\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	untrackedFile := filepath.Join(repo, "untracked.txt")
	if err := os.WriteFile(untrackedFile, []byte("untracked text\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	checkpointer := NewGitShadowCheckpointer()
	ref, err := checkpointer.CreateShadowCheckpoint(repo)
	if err != nil {
		t.Fatalf("CreateShadowCheckpoint: %v", err)
	}

	// Corrupt everything and add new spurious files.
	if err := os.WriteFile(baselineFile, []byte("corrupted baseline\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(file2, []byte("corrupted second\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(untrackedFile, []byte("corrupted untracked\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	spuriousFile := filepath.Join(repo, "spurious.txt")
	if err := os.WriteFile(spuriousFile, []byte("spurious\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Restore using mapped ref or direct ref.
	if err := checkpointer.RestoreShadowCheckpoint(repo, ref); err != nil {
		t.Fatalf("RestoreShadowCheckpoint: %v", err)
	}

	// Verify status.
	status := getGitOutput(t, repo, "status", "--porcelain")
	lines := strings.Split(status, "\n")
	statusMap := make(map[string]string)
	for _, l := range lines {
		l = strings.TrimRight(l, "\r")
		if len(l) >= 4 {
			statusMap[strings.TrimSpace(l[3:])] = l[:2]
		}
	}

	if statusMap["baseline.txt"] != "M " {
		t.Errorf("expected baseline.txt staged (M ), got %q", statusMap["baseline.txt"])
	}
	if statusMap["second.txt"] != " M" {
		t.Errorf("expected second.txt unstaged ( M), got %q", statusMap["second.txt"])
	}
	if statusMap["untracked.txt"] != "??" {
		t.Errorf("expected untracked.txt (??), got %q", statusMap["untracked.txt"])
	}
	if _, exists := statusMap["spurious.txt"]; exists {
		t.Errorf("spurious.txt should have been cleaned up")
	}

	// Verify contents.
	bContent, _ := os.ReadFile(baselineFile)
	if string(bContent) != "staged modification\n" {
		t.Errorf("unexpected baseline content: %q", string(bContent))
	}
	sContent, _ := os.ReadFile(file2)
	if string(sContent) != "unstaged modification\n" {
		t.Errorf("unexpected second content: %q", string(sContent))
	}
	uContent, _ := os.ReadFile(untrackedFile)
	if string(uContent) != "untracked text\n" {
		t.Errorf("unexpected untracked content: %q", string(uContent))
	}

	headAfter := getGitOutput(t, repo, "rev-parse", "HEAD")
	if headAfter != headBefore {
		t.Fatalf("HEAD changed: %s -> %s", headBefore, headAfter)
	}
}

func TestShadowCheckpoint_NoUntracked(t *testing.T) {
	repo := setupTestRepo(t)
	headBefore := getGitOutput(t, repo, "rev-parse", "HEAD")

	baselineFile := filepath.Join(repo, "baseline.txt")
	if err := os.WriteFile(baselineFile, []byte("modified only\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	ref, err := CreateShadowCheckpoint(repo)
	if err != nil {
		t.Fatalf("CreateShadowCheckpoint: %v", err)
	}

	// Verify ref has 2 parents (HEAD, index), and NO parent 3.
	cmdP3 := exec.Command("git", "-C", repo, "rev-parse", "--verify", ref+"^3")
	windowgate.ConfigureBackgroundCommand(cmdP3)
	if err := cmdP3.Run(); err == nil {
		t.Fatal("expected ref^3 to not exist when no untracked files")
	}

	// Create a new untracked file and further edit baseline.
	junkFile := filepath.Join(repo, "junk.txt")
	if err := os.WriteFile(junkFile, []byte("junk\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(baselineFile, []byte("corrupted\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := RestoreShadowCheckpoint(repo, ref); err != nil {
		t.Fatalf("RestoreShadowCheckpoint: %v", err)
	}

	if _, err := os.Stat(junkFile); !os.IsNotExist(err) {
		t.Errorf("junk.txt was not cleaned up")
	}
	bContent, _ := os.ReadFile(baselineFile)
	if string(bContent) != "modified only\n" {
		t.Errorf("unexpected baseline content: %q", string(bContent))
	}

	headAfter := getGitOutput(t, repo, "rev-parse", "HEAD")
	if headAfter != headBefore {
		t.Fatalf("HEAD changed: %s -> %s", headBefore, headAfter)
	}
}

func TestShadowCheckpoint_NestedSubdirectories(t *testing.T) {
	repo := setupTestRepo(t)

	subDir := filepath.Join(repo, "nested", "sub")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatal(err)
	}
	nestedUntracked := filepath.Join(subDir, "item.txt")
	if err := os.WriteFile(nestedUntracked, []byte("nested untracked file\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	ref, err := CreateShadowCheckpoint(repo)
	if err != nil {
		t.Fatalf("CreateShadowCheckpoint: %v", err)
	}

	// Delete the entire nested directory.
	if err := os.RemoveAll(filepath.Join(repo, "nested")); err != nil {
		t.Fatal(err)
	}

	if err := RestoreShadowCheckpoint(repo, ref); err != nil {
		t.Fatalf("RestoreShadowCheckpoint: %v", err)
	}

	content, err := os.ReadFile(nestedUntracked)
	if err != nil {
		t.Fatalf("nested file not restored: %v", err)
	}
	if string(content) != "nested untracked file\n" {
		t.Fatalf("unexpected content: %q", string(content))
	}
}

func TestShadowCheckpoint_ThreadSafety(t *testing.T) {
	repo := setupTestRepo(t)
	checkpointer := NewGitShadowCheckpointer()

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			ref, err := checkpointer.CreateShadowCheckpoint(repo)
			if err != nil {
				t.Errorf("goroutine %d: %v", idx, err)
				return
			}
			if _, ok := checkpointer.GetCheckpoint(ref); !ok {
				t.Errorf("goroutine %d: ref %s not recorded", idx, ref)
			}
		}(i)
	}
	wg.Wait()
}
