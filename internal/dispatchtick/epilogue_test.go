package dispatchtick

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func initGitRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	repo := t.TempDir()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=test-user",
			"GIT_AUTHOR_EMAIL=test@localhost",
			"GIT_COMMITTER_NAME=test-user",
			"GIT_COMMITTER_EMAIL=test@localhost",
		)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
	}
	git("init", "-q")
	git("config", "user.name", "test-user")
	git("config", "user.email", "test@localhost")
	return repo
}

func gitExec(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=test-user",
		"GIT_AUTHOR_EMAIL=test@localhost",
		"GIT_COMMITTER_NAME=test-user",
		"GIT_COMMITTER_EMAIL=test@localhost",
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

func TestEpilogueSubmitAndList(t *testing.T) {
	runsDir := t.TempDir()

	t0 := time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)
	t1 := time.Date(2026, 9, 1, 10, 1, 0, 0, time.UTC)
	t2 := time.Date(2026, 9, 1, 10, 2, 0, 0, time.UTC)

	rec1 := EpilogueRecord{
		Issue:       101,
		Lane:        "kernel",
		Paths:       []string{"internal/a.go"},
		Message:     "fix: update a",
		SubmittedAt: t0,
	}
	sub1, err := SubmitEpilogue(runsDir, rec1)
	if err != nil {
		t.Fatalf("submit 1: %v", err)
	}
	if sub1.ID == "" || sub1.Status != EpilogueStatusPending || sub1.Schema != EpilogueSchema {
		t.Fatalf("unexpected record 1: %+v", sub1)
	}

	rec2 := EpilogueRecord{
		Issue:       102,
		Lane:        "docs",
		Paths:       []string{"docs/b.md"},
		Message:     "docs: update b",
		SubmittedAt: t1,
	}
	sub2, err := SubmitEpilogue(runsDir, rec2)
	if err != nil {
		t.Fatalf("submit 2: %v", err)
	}

	rec3 := EpilogueRecord{
		Issue:       103,
		Lane:        "gateway",
		Paths:       []string{"internal/c.go"},
		Message:     "feat: update c",
		Status:      EpilogueStatusLanded,
		SubmittedAt: t2,
	}
	sub3, err := SubmitEpilogue(runsDir, rec3)
	if err != nil {
		t.Fatalf("submit 3: %v", err)
	}

	// Verify files exist in <runsDir>/epilogues
	epiloguesFolder := filepath.Join(runsDir, "epilogues")
	for _, id := range []string{sub1.ID, sub2.ID, sub3.ID} {
		p := filepath.Join(epiloguesFolder, id+".json")
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("expected file %s to exist: %v", p, err)
		}
	}

	// List all FIFO
	all, err := ListEpilogues(runsDir, "")
	if err != nil {
		t.Fatalf("list all: %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("expected 3 records, got %d", len(all))
	}
	if all[0].ID != sub1.ID || all[1].ID != sub2.ID || all[2].ID != sub3.ID {
		t.Fatalf("unexpected order: %+v", all)
	}

	// List pending
	pending, err := ListEpilogues(runsDir, EpilogueStatusPending)
	if err != nil {
		t.Fatalf("list pending: %v", err)
	}
	if len(pending) != 2 {
		t.Fatalf("expected 2 pending records, got %d", len(pending))
	}

	// List landed
	landed, err := ListEpilogues(runsDir, EpilogueStatusLanded)
	if err != nil {
		t.Fatalf("list landed: %v", err)
	}
	if len(landed) != 1 || landed[0].ID != sub3.ID {
		t.Fatalf("expected 1 landed record (sub3), got %+v", landed)
	}

	// Update record 1
	updated1, err := UpdateEpilogueStatus(runsDir, sub1.ID, EpilogueStatusLanded, "sha999", "")
	if err != nil {
		t.Fatalf("update sub1: %v", err)
	}
	if updated1.Status != EpilogueStatusLanded || updated1.LandedSHA != "sha999" || updated1.LandedAt == nil {
		t.Fatalf("unexpected updated record 1: %+v", updated1)
	}
}

func TestEpilogueDrainDisjointLanes(t *testing.T) {
	repo := initGitRepo(t)

	// Create base commit
	fileA := filepath.Join(repo, "fileA.txt")
	fileB := filepath.Join(repo, "fileB.txt")
	if err := os.WriteFile(fileA, []byte("initA\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fileB, []byte("initB\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitExec(t, repo, "add", "fileA.txt", "fileB.txt")
	gitExec(t, repo, "commit", "-m", "init: base files (fak kernel)")

	// Generate patch for fileA
	if err := os.WriteFile(fileA, []byte("work1A\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	patchA := gitExec(t, repo, "diff", "fileA.txt")
	gitExec(t, repo, "checkout", "fileA.txt")

	// Generate patch for fileB
	if err := os.WriteFile(fileB, []byte("work2B\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	patchB := gitExec(t, repo, "diff", "fileB.txt")
	gitExec(t, repo, "checkout", "fileB.txt")

	runsDir := filepath.Join(repo, ".dispatch-runs")

	// Submit epilogue 1
	sub1, err := SubmitEpilogue(runsDir, EpilogueRecord{
		Issue:   201,
		Lane:    "laneA",
		Paths:   []string{"fileA.txt"},
		Patch:   patchA,
		Message: "fix(#201): update fileA (fak laneA)",
	})
	if err != nil {
		t.Fatalf("submit 1: %v", err)
	}

	// Submit epilogue 2
	sub2, err := SubmitEpilogue(runsDir, EpilogueRecord{
		Issue:   202,
		Lane:    "laneB",
		Paths:   []string{"fileB.txt"},
		Patch:   patchB,
		Message: "fix(#202): update fileB (fak laneB)",
	})
	if err != nil {
		t.Fatalf("submit 2: %v", err)
	}

	// Drain
	result, err := DrainEpilogues(repo, runsDir, EpilogueDrainOptions{})
	if err != nil {
		t.Fatalf("drain: %v", err)
	}

	if result.Total != 2 || result.Landed != 2 || result.Conflicted != 0 || result.Failed != 0 {
		t.Fatalf("unexpected drain result: %+v", result)
	}

	// Verify file contents
	dataA, _ := os.ReadFile(fileA)
	if string(dataA) != "work1A\n" {
		t.Fatalf("fileA = %q, want 'work1A\\n'", string(dataA))
	}
	dataB, _ := os.ReadFile(fileB)
	if string(dataB) != "work2B\n" {
		t.Fatalf("fileB = %q, want 'work2B\\n'", string(dataB))
	}

	// Verify records updated to landed
	landed1, _ := ListEpilogues(runsDir, EpilogueStatusLanded)
	if len(landed1) != 2 {
		t.Fatalf("expected 2 landed records, got %d", len(landed1))
	}
	if landed1[0].ID != sub1.ID || landed1[0].LandedSHA == "" {
		t.Fatalf("invalid landed record 1: %+v", landed1[0])
	}
	if landed1[1].ID != sub2.ID || landed1[1].LandedSHA == "" {
		t.Fatalf("invalid landed record 2: %+v", landed1[1])
	}

	// Verify distinct commits
	if landed1[0].LandedSHA == landed1[1].LandedSHA {
		t.Fatalf("expected distinct commits, got identical SHA: %s", landed1[0].LandedSHA)
	}
}

func TestEpilogueDrainPathConflict(t *testing.T) {
	repo := initGitRepo(t)

	fileConflict := filepath.Join(repo, "conflict.txt")
	fileDisjoint := filepath.Join(repo, "disjoint.txt")

	if err := os.WriteFile(fileConflict, []byte("line1\nline2\nline3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(fileDisjoint, []byte("disjoint_init\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitExec(t, repo, "add", "conflict.txt", "disjoint.txt")
	gitExec(t, repo, "commit", "-m", "init: conflict and disjoint files (fak kernel)")

	// Patch 1: change line1
	if err := os.WriteFile(fileConflict, []byte("MODIFIED_BY_1\nline2\nline3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	patch1 := gitExec(t, repo, "diff", "conflict.txt")
	gitExec(t, repo, "checkout", "conflict.txt")

	// Patch 2: change line1 differently (will conflict once Patch 1 lands)
	if err := os.WriteFile(fileConflict, []byte("MODIFIED_BY_2\nline2\nline3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	patch2 := gitExec(t, repo, "diff", "conflict.txt")
	gitExec(t, repo, "checkout", "conflict.txt")

	// Patch 3: disjoint file
	if err := os.WriteFile(fileDisjoint, []byte("disjoint_work3\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	patch3 := gitExec(t, repo, "diff", "disjoint.txt")
	gitExec(t, repo, "checkout", "disjoint.txt")

	runsDir := filepath.Join(repo, ".dispatch-runs")

	sub1, _ := SubmitEpilogue(runsDir, EpilogueRecord{
		Issue:   301,
		Paths:   []string{"conflict.txt"},
		Patch:   patch1,
		Message: "fix(#301): patch1 on conflict.txt (fak lane1)",
	})
	sub2, _ := SubmitEpilogue(runsDir, EpilogueRecord{
		Issue:   302,
		Paths:   []string{"conflict.txt"},
		Patch:   patch2,
		Message: "fix(#302): patch2 on conflict.txt (fak lane2)",
	})
	sub3, _ := SubmitEpilogue(runsDir, EpilogueRecord{
		Issue:   303,
		Paths:   []string{"disjoint.txt"},
		Patch:   patch3,
		Message: "fix(#303): patch3 on disjoint.txt (fak lane3)",
	})

	result, err := DrainEpilogues(repo, runsDir, EpilogueDrainOptions{})
	if err != nil {
		t.Fatalf("drain: %v", err)
	}

	if result.Total != 3 || result.Landed != 2 || result.Conflicted != 1 || result.Failed != 0 {
		t.Fatalf("unexpected result: %+v", result)
	}

	// Verify epilogue 1 landed
	recs1, _ := ListEpilogues(runsDir, "")
	statusMap := make(map[string]EpilogueStatus)
	for _, r := range recs1 {
		statusMap[r.ID] = r.Status
	}

	if statusMap[sub1.ID] != EpilogueStatusLanded {
		t.Fatalf("sub1 status = %s, want landed", statusMap[sub1.ID])
	}
	if statusMap[sub2.ID] != EpilogueStatusConflict {
		t.Fatalf("sub2 status = %s, want conflict", statusMap[sub2.ID])
	}
	if statusMap[sub3.ID] != EpilogueStatusLanded {
		t.Fatalf("sub3 status = %s, want landed", statusMap[sub3.ID])
	}

	// Verify disjoint change landed
	dataDisjoint, _ := os.ReadFile(fileDisjoint)
	if string(dataDisjoint) != "disjoint_work3\n" {
		t.Fatalf("disjoint.txt = %q, want 'disjoint_work3\\n'", string(dataDisjoint))
	}
}

func TestEpilogueDurabilityAcrossCrash(t *testing.T) {
	repo := initGitRepo(t)

	fileDurable := filepath.Join(repo, "durable.txt")
	if err := os.WriteFile(fileDurable, []byte("durable_init\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitExec(t, repo, "add", "durable.txt")
	gitExec(t, repo, "commit", "-m", "init: durable file (fak kernel)")

	if err := os.WriteFile(fileDurable, []byte("crashed_worker_work\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	patch := gitExec(t, repo, "diff", "durable.txt")
	gitExec(t, repo, "checkout", "durable.txt")

	runsDir := filepath.Join(repo, ".dispatch-runs")

	// Worker submits epilogue then crashes (process dies)
	sub, err := SubmitEpilogue(runsDir, EpilogueRecord{
		Issue:     401,
		WorkerPID: 999999, // simulated dead PID
		Paths:     []string{"durable.txt"},
		Patch:     patch,
		Message:   "fix(#401): crashed worker work (fak crash)",
	})
	if err != nil {
		t.Fatalf("submit: %v", err)
	}

	// Verify the file was durably written to disk
	recPath := filepath.Join(runsDir, "epilogues", sub.ID+".json")
	if _, err := os.Stat(recPath); err != nil {
		t.Fatalf("epilogue record not durable on disk: %v", err)
	}

	// Now controller resumes/drains
	result, err := DrainEpilogues(repo, runsDir, EpilogueDrainOptions{})
	if err != nil {
		t.Fatalf("drain: %v", err)
	}

	if result.Landed != 1 {
		t.Fatalf("expected 1 landed epilogue after crash recovery, got %+v", result)
	}

	data, _ := os.ReadFile(fileDurable)
	if string(data) != "crashed_worker_work\n" {
		t.Fatalf("durable.txt = %q, want 'crashed_worker_work\\n'", string(data))
	}
}

func TestEpilogueDrainInPlaceDirtyWorkingTree(t *testing.T) {
	// #12084: When an in-place worker queues on busy and leaves changes in root,
	// DrainEpilogues must not wipe them via git checkout or fail with git apply inversion.
	repo := initGitRepo(t)

	file := filepath.Join(repo, "inplace.txt")
	if err := os.WriteFile(file, []byte("line1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitExec(t, repo, "add", "inplace.txt")
	gitExec(t, repo, "commit", "-m", "init: inplace (fak test)")

	// Worker modifies file in-place and queues with patch
	if err := os.WriteFile(file, []byte("line1\nline2_worker_change\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	patch := gitExec(t, repo, "diff", "inplace.txt")
	// NOTICE: We do NOT checkout! File remains dirty in working tree as in real queue-on-busy!

	runsDir := filepath.Join(repo, ".dispatch-runs")
	_, err := SubmitEpilogue(runsDir, EpilogueRecord{
		Issue:       501,
		Lane:        "cmd",
		Paths:       []string{"inplace.txt"},
		WorktreeDir: repo,
		Patch:       patch,
		Message:     "feat(#501): inplace queue commit (fak cmd)",
	})
	if err != nil {
		t.Fatalf("submit: %v", err)
	}

	result, err := DrainEpilogues(repo, runsDir, EpilogueDrainOptions{})
	if err != nil {
		t.Fatalf("drain: %v", err)
	}

	if result.Landed != 1 || result.Conflicted != 0 {
		t.Fatalf("expected 1 landed, 0 conflicted, got %+v", result)
	}

	content, _ := os.ReadFile(file)
	if string(content) != "line1\nline2_worker_change\n" {
		t.Fatalf("inplace.txt content was wiped or corrupted: %q", string(content))
	}

	headMsg := gitExec(t, repo, "log", "-1", "--pretty=%B")
	if !strings.Contains(headMsg, "inplace queue commit") {
		t.Fatalf("HEAD commit message does not match: %s", headMsg)
	}
}
