package witness

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime slow est=3s lane=default
// fak-test:justify why=regression when=changed:internal/witness/**
func TestExecutionMaterializerPinsBothSourcesAndOverlay(t *testing.T) {
	root := newExecutionRepo(t)
	writeRepoFile(t, root, "value.txt", "bad\n")
	writeRepoFile(t, root, "stable.txt", "invariant\n")
	gitIn(t, root, "add", ".")
	gitIn(t, root, "commit", "-qm", "parent")
	writeRepoFile(t, root, "value.txt", "good\n")
	writeRepoFile(t, root, "selected_test.go", "candidate test overlay\n")
	gitIn(t, root, "add", ".")
	gitIn(t, root, "commit", "-qm", "candidate")
	v := NewExecutionVerifier(root)
	var dirs, refs []string
	releases := 0
	ctx := WithExecutionMaterializer(context.Background(), func(ctx context.Context, repo, ref string) (string, func(), error) {
		if repo != root || len(ref) != 40 {
			t.Fatalf("provider received repo=%q ref=%q", repo, ref)
		}
		dir, cleanup, err := v.scratchWorktree(context.Background(), ref)
		if err != nil {
			return "", nil, err
		}
		dirs, refs = append(dirs, dir), append(refs, ref)
		return dir, func() { releases++; cleanup() }, nil
	})
	res := v.Verify(ctx, ExecutionSpec{Commit: "HEAD", FailToPass: []ExecutionSelector{{Command: []string{"git", "grep", "-q", "good", "--", "value.txt"}}}, PassToPass: []ExecutionSelector{{Command: []string{"git", "grep", "-q", "invariant", "--", "stable.txt"}}}})
	if res.Verdict != ExecPass || len(dirs) != 2 || dirs[0] == dirs[1] || refs[0] != res.Parent || refs[1] != res.Commit || releases != 2 {
		t.Fatalf("result=%+v dirs=%v refs=%v releases=%d", res, dirs, refs, releases)
	}
	parent, release, err := v.scratchWorktree(ctx, res.Parent)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if !overlayTestsAtRef(ctx, gitRunner, root, res.Commit, parent, []string{"selected_test.go"}) {
		t.Fatal("selected candidate test overlay failed")
	}
	for name, want := range map[string]string{"value.txt": "bad\n", "stable.txt": "invariant\n", "selected_test.go": "candidate test overlay\n"} {
		got, err := os.ReadFile(filepath.Join(parent, name))
		if err != nil || string(got) != want {
			t.Fatalf("%s=%q err=%v want=%q", name, got, err, want)
		}
	}
	assertRepoClean(t, root)
}

// fak-test:runtime slow est=1s lane=default
// fak-test:justify why=regression when=changed:internal/witness/**
func TestExecutionMaterializerContextIsolationAndFailure(t *testing.T) {
	root := newExecutionRepo(t)
	writeRepoFile(t, root, "value.txt", "parent\n")
	gitIn(t, root, "add", ".")
	gitIn(t, root, "commit", "-qm", "seed")
	v := NewExecutionVerifier(root)
	sentinel := errors.New("materializer failed")
	calls, releases := 0, 0
	ctx := WithExecutionMaterializer(context.Background(), func(context.Context, string, string) (string, func(), error) {
		calls++
		return "", func() { releases++ }, sentinel
	})
	_, _, err := v.scratchWorktree(ctx, "HEAD")
	if !errors.Is(err, sentinel) || calls != 1 || releases != 1 {
		t.Fatalf("provider failure=%v calls=%d releases=%d", err, calls, releases)
	}
	for _, plain := range []context.Context{context.Background(), WithExecutionMaterializer(context.Background(), nil)} {
		dir, release, err := v.scratchWorktree(plain, "HEAD")
		if err != nil {
			t.Fatal(err)
		}
		got, readErr := os.ReadFile(filepath.Join(dir, "value.txt"))
		release()
		if readErr != nil || strings.TrimSpace(string(got)) != "parent" || calls != 1 {
			t.Fatalf("default source=%q err=%v provider calls=%d", got, readErr, calls)
		}
	}
	canceled, cancel := context.WithCancel(ctx)
	cancel()
	_, _, err = v.scratchWorktree(canceled, "HEAD")
	if !errors.Is(err, context.Canceled) || calls != 1 {
		t.Fatalf("canceled acquire=%v provider calls=%d", err, calls)
	}
}
