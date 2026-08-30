package workerworktree

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestLoadIntentReturnsDurableUniqueIssueBinding(t *testing.T) {
	wt := filepath.Join(t.TempDir(), "fak-worker-wt-issue")
	message := "feat(test): durable binding (#9568) (fak test)\n\nSigned-off-by: Test <test@example.invalid>"
	if err := SaveIntent(wt, "abc123", message, []string{"z.txt", `dir\a.txt`, "z.txt"}); err != nil {
		t.Fatal(err)
	}
	in, err := LoadIntent(wt)
	if err != nil {
		t.Fatal(err)
	}
	if in.Schema != inventorySchema || in.IssueNumber != 9568 || in.Message != message ||
		!reflect.DeepEqual(in.Paths, []string{"dir/a.txt", "z.txt"}) {
		t.Fatalf("loaded intent = %+v", in)
	}
	if err := SaveIntent(wt, "abc123", message, []string{"z.txt", `dir\a.txt`}); err != nil {
		t.Fatalf("byte-identical canonical replay failed: %v", err)
	}
	if err := SaveIntent(wt, "abc123", message, []string{"new.txt", "z.txt", `dir\a.txt`}); err != nil {
		t.Fatalf("monotonic coordinator path expansion failed: %v", err)
	}
	expanded, err := LoadIntent(wt)
	if err != nil || !reflect.DeepEqual(expanded.Paths, []string{"dir/a.txt", "new.txt", "z.txt"}) {
		t.Fatalf("expanded intent = %+v err=%v", expanded, err)
	}
	if err := SaveIntent(wt, "abc123", message, []string{"z.txt"}); err == nil || !strings.Contains(err.Error(), "may expand but not remove") {
		t.Fatalf("path contraction error = %v", err)
	}
	if err := SaveIntent(wt, "abc123", "feat(test): substitute (#17) (fak test)", []string{"new.txt", "z.txt", `dir\a.txt`}); err == nil || !strings.Contains(err.Error(), "different coordinator metadata") {
		t.Fatalf("intent replacement error = %v", err)
	}
}

func TestIntentIssueBindingAllowsZeroAndRejectsMalformedOrMultiple(t *testing.T) {
	wt := filepath.Join(t.TempDir(), "fak-worker-wt-nonissue")
	message := "chore(test): non-issue maintenance (fak test)\n\nBody mention #999 is not the signed subject."
	if err := SaveIntent(wt, "abc123", message, []string{"a.txt"}); err != nil {
		t.Fatal(err)
	}
	in, err := LoadIntent(wt)
	if err != nil || in.IssueNumber != 0 {
		t.Fatalf("non-issue intent = %+v err=%v", in, err)
	}

	for _, tc := range []struct {
		name    string
		subject string
		want    string
	}{
		{"zero", "fix: invalid (#0)", "malformed"},
		{"letters", "fix: invalid (#oops)", "malformed"},
		{"mixed", "fix: invalid (#12x)", "malformed"},
		{"multiple", "fix: ambiguous (#12) and (#13)", "at most one"},
		{"duplicate", "fix: ambiguous (#12) and again #12", "at most one"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			candidate := filepath.Join(t.TempDir(), "worker")
			err := SaveIntent(candidate, "abc123", tc.subject, []string{"a.txt"})
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("SaveIntent(%q) error = %v, want %q", tc.subject, err, tc.want)
			}
			if _, statErr := os.Stat(intentPath(candidate)); !os.IsNotExist(statErr) {
				t.Fatalf("invalid intent persisted: stat error=%v", statErr)
			}
		})
	}
}

func TestLoadIntentFailsClosedOnMetadataOrMessageTamper(t *testing.T) {
	newIntent := func(t *testing.T) string {
		t.Helper()
		wt := filepath.Join(t.TempDir(), "worker")
		if err := SaveIntent(wt, "abc123", "fix(test): bind (#9568) (fak test)", []string{"a.txt"}); err != nil {
			t.Fatal(err)
		}
		return wt
	}

	t.Run("stored issue", func(t *testing.T) {
		wt := newIntent(t)
		raw, err := os.ReadFile(intentPath(wt))
		if err != nil {
			t.Fatal(err)
		}
		var in Intent
		if err := json.Unmarshal(raw, &in); err != nil {
			t.Fatal(err)
		}
		in.IssueNumber = 17
		raw, err = json.MarshalIndent(in, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(intentPath(wt), append(raw, '\n'), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadIntent(wt); err == nil || !strings.Contains(err.Error(), "does not match") {
			t.Fatalf("stored issue tamper error = %v", err)
		}
	})

	t.Run("message mirror", func(t *testing.T) {
		wt := newIntent(t)
		if err := os.WriteFile(messagePath(wt), []byte("fix(test): redirect (#17) (fak test)\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadIntent(wt); err == nil || !strings.Contains(err.Error(), "message mirror") {
			t.Fatalf("message mirror tamper error = %v", err)
		}
	})

	t.Run("unknown schema field", func(t *testing.T) {
		wt := newIntent(t)
		raw, err := os.ReadFile(intentPath(wt))
		if err != nil {
			t.Fatal(err)
		}
		raw = []byte(strings.Replace(string(raw), "\n}", ",\n  \"future\": true\n}", 1))
		if err := os.WriteFile(intentPath(wt), raw, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadIntent(wt); err == nil || !strings.Contains(err.Error(), "unknown field") {
			t.Fatalf("unknown field error = %v", err)
		}
	})
}

func TestLoadIntentMigratesStrictV1Fixture(t *testing.T) {
	writeV1 := func(t *testing.T, extra map[string]any) (string, string) {
		t.Helper()
		wt := filepath.Join(t.TempDir(), "legacy-worker")
		message := "fix(test): legacy binding (#9568) (fak test)"
		legacy := map[string]any{
			"schema": inventorySchemaV1, "path": wt, "base_sha": "abc123",
			"message": message, "paths": []string{"a.txt"},
		}
		for key, value := range extra {
			legacy[key] = value
		}
		if err := os.MkdirAll(intentDir(wt), 0o700); err != nil {
			t.Fatal(err)
		}
		raw, err := json.MarshalIndent(legacy, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(intentPath(wt), append(raw, '\n'), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(messagePath(wt), []byte(message+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		return wt, message
	}

	t.Run("normalize and persist v2", func(t *testing.T) {
		wt, message := writeV1(t, nil)
		loaded, err := LoadIntent(wt)
		if err != nil || loaded.Schema != inventorySchema || loaded.IssueNumber != 9568 {
			t.Fatalf("normalized legacy intent = %+v err=%v", loaded, err)
		}
		if err := SaveIntent(wt, "abc123", message, []string{"a.txt"}); err != nil {
			t.Fatalf("monotonic v1 migration failed: %v", err)
		}
		raw, err := os.ReadFile(intentPath(wt))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(raw), inventorySchema) || !strings.Contains(string(raw), `"issue_number": 9568`) {
			t.Fatalf("persisted migration is not v2: %s", raw)
		}
		if migrated, err := LoadIntent(wt); err != nil || migrated.IssueNumber != 9568 {
			t.Fatalf("migrated intent = %+v err=%v", migrated, err)
		}
	})

	t.Run("reject non-v1 field", func(t *testing.T) {
		wt, _ := writeV1(t, map[string]any{"issue_number": 17})
		if _, err := LoadIntent(wt); err == nil || !strings.Contains(err.Error(), "non-v1 issue_number") {
			t.Fatalf("legacy added-field error = %v", err)
		}
	})

	t.Run("reject mirror tamper", func(t *testing.T) {
		wt, _ := writeV1(t, nil)
		if err := os.WriteFile(messagePath(wt), []byte("fix(test): redirected (#17) (fak test)\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadIntent(wt); err == nil || !strings.Contains(err.Error(), "message mirror") {
			t.Fatalf("legacy mirror tamper error = %v", err)
		}
	})
}

func TestInventoryLandReadyIsReadOnlyAndExact(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	runGitTest(t, root, "init", repo)
	runGitTest(t, repo, "config", "user.name", "Test")
	runGitTest(t, repo, "config", "user.email", "test-at-example.invalid")
	if err := os.WriteFile(filepath.Join(repo, "owned.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer.txt"), []byte("peer-base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitTest(t, repo, "add", "owned.txt", "peer.txt")
	runGitTest(t, repo, "commit", "-m", "base")
	base := strings.TrimSpace(runGitTest(t, repo, "rev-parse", "HEAD"))
	prep := Prepare(repo, "test", "5994", base, filepath.Join(root, "workers"), nil)
	if !prep.OK {
		t.Fatalf("prepare = %+v", prep)
	}
	t.Cleanup(func() { _ = Reap(repo, prep.Path, nil) })
	if err := SaveIntent(prep.Path, base, "feat(test): land intent (#5994) (fak test)", []string{"owned.txt"}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(prep.Path, "owned.txt"), []byte("changed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer.txt"), []byte("peer-wip\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	beforeHead := strings.TrimSpace(runGitTest(t, repo, "rev-parse", "HEAD"))
	beforeIndex := runGitTest(t, repo, "write-tree")
	beforePeer, _ := os.ReadFile(filepath.Join(repo, "peer.txt"))
	rows, err := Inventory(repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 {
		t.Fatalf("rows = %+v", rows)
	}
	want := []string{"fak", "worktree", "worker", "land", "--worktree", prep.Path, "--base-sha", base, "--msg-file", messagePath(prep.Path), "--paths", "owned.txt"}
	if rows[0].State != "LAND_READY" || !reflect.DeepEqual(rows[0].DirtyPaths, []string{"owned.txt"}) || !reflect.DeepEqual(rows[0].LandArgv, want) {
		t.Fatalf("row = %+v; want argv %q", rows[0], want)
	}
	if got := strings.TrimSpace(runGitTest(t, repo, "rev-parse", "HEAD")); got != beforeHead {
		t.Fatalf("HEAD mutated: %s -> %s", beforeHead, got)
	}
	if got := runGitTest(t, repo, "write-tree"); got != beforeIndex {
		t.Fatalf("index mutated: %q -> %q", beforeIndex, got)
	}
	afterPeer, _ := os.ReadFile(filepath.Join(repo, "peer.txt"))
	if !reflect.DeepEqual(afterPeer, beforePeer) {
		t.Fatalf("peer bytes mutated")
	}
}

func TestInventoryCleanAndAmbiguousDoNotOverclaim(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	runGitTest(t, root, "init", repo)
	runGitTest(t, repo, "config", "user.name", "Test")
	runGitTest(t, repo, "config", "user.email", "test-at-example.invalid")
	_ = os.WriteFile(filepath.Join(repo, "a.txt"), []byte("a\n"), 0o644)
	_ = os.WriteFile(filepath.Join(repo, "b.txt"), []byte("b\n"), 0o644)
	runGitTest(t, repo, "add", ".")
	runGitTest(t, repo, "commit", "-m", "base")
	base := strings.TrimSpace(runGitTest(t, repo, "rev-parse", "HEAD"))
	prep := Prepare(repo, "test", "ambiguous", base, filepath.Join(root, "workers"), nil)
	if !prep.OK {
		t.Fatalf("prepare = %+v", prep)
	}
	t.Cleanup(func() { _ = Reap(repo, prep.Path, nil) })
	if err := SaveIntent(prep.Path, base, "message", []string{"a.txt"}); err != nil {
		t.Fatal(err)
	}
	rows, err := Inventory(repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].State != "CLEAN" || rows[0].NeedsOperator {
		t.Fatalf("clean row = %+v", rows)
	}
	_ = os.WriteFile(filepath.Join(prep.Path, "b.txt"), []byte("changed\n"), 0o644)
	rows, err = Inventory(repo, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].State != "NEEDS_OPERATOR" || !rows[0].NeedsOperator || len(rows[0].LandArgv) != 0 {
		t.Fatalf("ambiguous row = %+v", rows)
	}
}

func TestSamePathResolvesSymlinkedParent(t *testing.T) {
	realRoot := t.TempDir()
	realWorktree := filepath.Join(realRoot, "worker")
	if err := os.Mkdir(realWorktree, 0o755); err != nil {
		t.Fatal(err)
	}
	aliasRoot := filepath.Join(t.TempDir(), "alias")
	if err := os.Symlink(realRoot, aliasRoot); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	if !samePath(realWorktree, filepath.Join(aliasRoot, "worker")) {
		t.Fatalf("samePath did not resolve %q to %q", aliasRoot, realRoot)
	}
}
