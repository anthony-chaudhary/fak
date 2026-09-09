package workerworktree

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestWorkerPathspecFencingRefusesOutOfLaneWrites(t *testing.T) {
	t.Run("in_lane_changes_pass", func(t *testing.T) {
		leased := []string{"internal/workerworktree/**"}
		changed := []string{
			"internal/workerworktree/fence.go",
			"internal/workerworktree/fence_test.go",
			"internal/workerworktree/sub/nested.go",
			"internal\\workerworktree\\windows.go",
			"./internal/workerworktree/relative.go",
		}
		if err := ValidateWorkerTreeDisjointness(changed, leased); err != nil {
			t.Fatalf("expected in-lane changes to pass, got error: %v", err)
		}

		// Empty leased globs means no fence; should return nil
		if err := ValidateWorkerTreeDisjointness(changed, nil); err != nil {
			t.Fatalf("expected empty leased globs to return nil, got: %v", err)
		}
		if err := ValidateWorkerTreeDisjointness(changed, []string{}); err != nil {
			t.Fatalf("expected empty leased globs to return nil, got: %v", err)
		}
	})

	t.Run("out_of_lane_changes_fail_with_constant", func(t *testing.T) {
		leased := []string{"internal/workerworktree/**"}
		changed := []string{
			"internal/workerworktree/fence.go",
			"cmd/fak/main.go",
			"docs/README.md",
		}
		err := ValidateWorkerTreeDisjointness(changed, leased)
		if err == nil {
			t.Fatal("expected error for out-of-lane changes, got nil")
		}
		if !errors.Is(err, ErrOutOfLaneMutation) && !strings.Contains(err.Error(), ReasonOutOfLaneMutation) {
			t.Fatalf("expected error containing or wrapping %s, got: %v", ReasonOutOfLaneMutation, err)
		}
		if !strings.Contains(err.Error(), ReasonOutOfLaneMutation) {
			t.Fatalf("expected error string to contain %s, got: %v", ReasonOutOfLaneMutation, err)
		}
		for _, file := range []string{"cmd/fak/main.go", "docs/README.md"} {
			if !strings.Contains(err.Error(), file) {
				t.Fatalf("expected error to list violating file %q, got: %v", file, err)
			}
		}
		var mutErr *OutOfLaneMutationError
		if errors.As(err, &mutErr) {
			if len(mutErr.Violations) != 2 {
				t.Fatalf("expected 2 violations, got: %v", mutErr.Violations)
			}
		}
	})

	t.Run("recursive_glob_matching", func(t *testing.T) {
		// "pkg/**" matches everything under pkg
		pkgGlobs := []string{"pkg/**"}
		if err := ValidateWorkerTreeDisjointness([]string{"pkg/a.go", "pkg/b/c.go", "pkg/b/c/d/e.go"}, pkgGlobs); err != nil {
			t.Fatalf("pkg/** should match nested paths: %v", err)
		}
		if err := ValidateWorkerTreeDisjointness([]string{"other/a.go"}, pkgGlobs); err == nil {
			t.Fatal("pkg/** should not match other/a.go")
		}

		// "dir/*" matches single level under dir
		dirGlobs := []string{"dir/*"}
		if err := ValidateWorkerTreeDisjointness([]string{"dir/file.go"}, dirGlobs); err != nil {
			t.Fatalf("dir/* should match direct child: %v", err)
		}
		if err := ValidateWorkerTreeDisjointness([]string{"dir/sub/file.go"}, dirGlobs); err == nil {
			t.Fatal("dir/* should not match nested child dir/sub/file.go")
		}
		if err := ValidateWorkerTreeDisjointness([]string{"other/file.go"}, dirGlobs); err == nil {
			t.Fatal("dir/* should not match other/file.go")
		}

		// "exact/path.go" matches only that exact file
		exactGlobs := []string{"exact/path.go"}
		if err := ValidateWorkerTreeDisjointness([]string{"exact/path.go"}, exactGlobs); err != nil {
			t.Fatalf("exact/path.go should match exact path: %v", err)
		}
		if err := ValidateWorkerTreeDisjointness([]string{"exact/other.go"}, exactGlobs); err == nil {
			t.Fatal("exact/path.go should not match exact/other.go")
		}
		if err := ValidateWorkerTreeDisjointness([]string{"exact/path.go.bak"}, exactGlobs); err == nil {
			t.Fatal("exact/path.go should not match exact/path.go.bak")
		}

		// Combined globs
		combined := []string{"pkg/**", "dir/*", "exact/path.go"}
		if err := ValidateWorkerTreeDisjointness([]string{
			"pkg/sub/mod.go",
			"dir/file.go",
			"exact/path.go",
		}, combined); err != nil {
			t.Fatalf("combined globs should all match: %v", err)
		}
	})

	t.Run("pre_land_refusal_when_worker_modifies_out_of_lane_files", func(t *testing.T) {
		t.Setenv(IsolatedLandEnv, "0")
		t.Setenv(LandReadbackEnv, "0")
		patch := "diff --git a/internal/workerworktree/fence.go b/internal/workerworktree/fence.go\n" +
			"--- a/internal/workerworktree/fence.go\n+++ b/internal/workerworktree/fence.go\n@@\n-old\n+new\n" +
			"diff --git a/cmd/fak/main.go b/cmd/fak/main.go\n" +
			"--- a/cmd/fak/main.go\n+++ b/cmd/fak/main.go\n@@\n-old\n+new\n"
		nameOnly := "internal/workerworktree/fence.go\ncmd/fak/main.go\n"
		g := coreLockFake(patch, nameOnly, "feat(workerworktree): test commit")

		res := Land("/trunk", "/wt/fak-worker-wt-test", "base123", "", []string{"internal/workerworktree"}, nil, g.run,
			WithLeasedGlobs([]string{"internal/workerworktree/**"}))

		if res.OK {
			t.Fatalf("expected Land to be refused, got success: %+v", res)
		}
		if res.Code != ReasonOutOfLaneMutation {
			t.Fatalf("expected Result.Code = %q, got %q", ReasonOutOfLaneMutation, res.Code)
		}
		if !res.Preserved {
			t.Fatalf("expected Result.Preserved = true, got %v", res.Preserved)
		}
		if !strings.Contains(res.Reason, "out-of-lane write detected") {
			t.Fatalf("expected Reason to mention 'out-of-lane write detected', got: %s", res.Reason)
		}
		if !strings.Contains(res.Reason, ReasonOutOfLaneMutation) {
			t.Fatalf("expected Reason to contain %q, got: %s", ReasonOutOfLaneMutation, res.Reason)
		}
		if !strings.Contains(res.Reason, "cmd/fak/main.go") {
			t.Fatalf("expected Reason to list violating file cmd/fak/main.go, got: %s", res.Reason)
		}
		if landTouchedTrunk(g) {
			t.Fatalf("refused land must leave trunk untouched; calls: %v", g.calls)
		}

		// Now test that in-lane writes succeed through LandChecked / WithLeasedGlobs
		inLanePatch := "diff --git a/internal/workerworktree/fence.go b/internal/workerworktree/fence.go\n" +
			"--- a/internal/workerworktree/fence.go\n+++ b/internal/workerworktree/fence.go\n@@\n-old\n+new\n"
		inLaneNames := "internal/workerworktree/fence.go\n"
		gPass := coreLockFake(inLanePatch, inLaneNames, "feat(workerworktree): in-lane commit (fak workerworktree)")

		passRes := LandChecked("/trunk", "/wt/fak-worker-wt-test", "base123", "", []string{"internal/workerworktree"}, nil, gPass.run,
			WithLeasedGlobs([]string{"internal/workerworktree/**"}))

		if !passRes.OK || passRes.Code == ReasonOutOfLaneMutation {
			t.Fatalf("expected in-lane LandChecked to pass, got: %+v", passRes)
		}
	})

	t.Run("live_git_worktree_out_of_lane_refusal", func(t *testing.T) {
		if _, err := exec.LookPath("git"); err != nil {
			t.Skip("git not on PATH")
		}
		repo := t.TempDir()
		runCmd := func(args ...string) string {
			c := exec.Command("git", args...)
			c.Dir = repo
			out, err := c.CombinedOutput()
			if err != nil {
				t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
			}
			return string(out)
		}
		runCmd("init", "-q", "-b", "main")
		runCmd("config", "user.email", "fence@test")
		runCmd("config", "user.name", "fence")
		runCmd("config", "commit.gpgsign", "false")
		if err := os.WriteFile(filepath.Join(repo, "app.go"), []byte("package app\n\nfunc A() int { return 1 }\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		runCmd("add", "app.go")
		runCmd("commit", "-q", "-m", "base")

		base := TrunkHeadSHA(repo, nil)
		if base == "" {
			t.Fatal("no trunk head")
		}

		wtRoot := t.TempDir()
		prep := Prepare(repo, "app", "12337", base, wtRoot, nil)
		if !prep.OK {
			t.Fatalf("prepare: %+v", prep)
		}
		defer Reap(repo, prep.Path, nil)

		// Modify both an in-lane file and an out-of-lane file in the worktree
		if err := os.WriteFile(filepath.Join(prep.Path, "app.go"), []byte("package app\n\nfunc A() int { return 2 }\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(prep.Path, "out_of_lane.txt"), []byte("bad mutation\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		wc := func(args ...string) {
			c := exec.Command("git", args...)
			c.Dir = prep.Path
			if out, err := c.CombinedOutput(); err != nil {
				t.Fatalf("worktree git %s: %v\n%s", strings.Join(args, " "), err, out)
			}
		}
		wc("config", "user.email", "fence@test")
		wc("config", "user.name", "fence")
		wc("config", "commit.gpgsign", "false")
		wc("add", "app.go", "out_of_lane.txt")
		wc("commit", "-q", "-m", "feat(app): mutation (fak app)")

		res := Land(repo, prep.Path, base, "", []string{"app.go"}, nil, nil,
			WithLeasedGlobs([]string{"app.go"}))

		if res.OK {
			t.Fatalf("expected Land to be refused for out-of-lane mutation, got: %+v", res)
		}
		if res.Code != ReasonOutOfLaneMutation {
			t.Fatalf("expected Code %s, got %s", ReasonOutOfLaneMutation, res.Code)
		}
		if !res.Preserved {
			t.Fatalf("expected Preserved true, got false")
		}
		if !strings.Contains(res.Reason, "out-of-lane write detected") {
			t.Fatalf("expected reason to mention out-of-lane write detected, got: %s", res.Reason)
		}
		if !strings.Contains(res.Reason, "out_of_lane.txt") {
			t.Fatalf("expected reason to mention out_of_lane.txt, got: %s", res.Reason)
		}
	})
}
