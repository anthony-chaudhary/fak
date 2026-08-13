package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestCollectSweepParkedSurfacesStashAndUnmergedRef(t *testing.T) {
	root := t.TempDir()
	run := func(args ...string) string {
		t.Helper()
		c := exec.Command("git", args...)
		c.Dir = root
		c.Env = append(os.Environ(), "GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.com", "GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.com")
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("init", "-b", "main")
	if err := os.WriteFile(filepath.Join(root, "base.txt"), []byte("base\n"), 0644); err != nil {
		t.Fatal(err)
	}
	run("add", "base.txt")
	run("commit", "-m", "base")
	run("checkout", "-b", "parked-ref")
	if err := os.WriteFile(filepath.Join(root, "ref.txt"), []byte("ref\n"), 0644); err != nil {
		t.Fatal(err)
	}
	run("add", "ref.txt")
	run("commit", "-m", "parked ref")
	run("checkout", "main")
	if err := os.WriteFile(filepath.Join(root, "base.txt"), []byte("parked stash\n"), 0644); err != nil {
		t.Fatal(err)
	}
	run("stash", "push", "-m", "parked edit")
	got := collectSweepParked(root)
	if got.Count != 2 || len(got.Stashes) != 1 || len(got.Refs) != 1 {
		t.Fatalf("parked=%+v", got)
	}
	if got.Stashes[0].Name != "stash@{0}" || got.Refs[0].Name != "refs/heads/parked-ref" {
		t.Fatalf("parked=%+v", got)
	}
}

func TestRenderSweepPlanShowsParkedWorkWhenTreeClean(t *testing.T) {
	var out strings.Builder
	renderSweepPlan(&out, sweepPlan{Parked: sweepParkedSummary{Count: 1, Stashes: []sweepParkedItem{{Kind: "stash", Name: "stash@{0}", Summary: "parked edit"}}}})
	text := out.String()
	for _, want := range []string{"working tree is clean", "PARKED / HIDDEN WORK", "stash@{0}", "parked edit"} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q in %q", want, text)
		}
	}
}

func TestCollectSweepParkedIncludesWIPCheckpoints(t *testing.T) {
	repo := t.TempDir()
	run := func(args ...string) string {
		t.Helper()
		c := exec.Command("git", args...)
		c.Dir = repo
		c.Env = append(os.Environ(), "GIT_AUTHOR_NAME=test", "GIT_AUTHOR_EMAIL=test@example.invalid", "GIT_COMMITTER_NAME=test", "GIT_COMMITTER_EMAIL=test@example.invalid")
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("init", "-b", "main")
	if err := os.WriteFile(filepath.Join(repo, "base.txt"), []byte("base\n"), 0644); err != nil {
		t.Fatal(err)
	}
	run("add", "base.txt")
	run("commit", "-m", "base")
	checkpoint := run("commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", `fak-wip: {"session_id":"parked-test"}`)
	run("update-ref", "refs/fak/wip/parked-test", checkpoint)

	got := collectSweepParked(repo)
	if got.Count != 1 || len(got.Checkpoints) != 1 {
		t.Fatalf("parked = %#v, want one WIP checkpoint", got)
	}
	if got.Checkpoints[0].Kind != "checkpoint" || got.Checkpoints[0].Name != "refs/fak/wip/parked-test" {
		t.Fatalf("checkpoint = %#v", got.Checkpoints[0])
	}
}
