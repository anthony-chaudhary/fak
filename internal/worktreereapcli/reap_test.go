package worktreereapcli

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/testgit"
	"github.com/anthony-chaudhary/fak/internal/wiplifecycle"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func TestMain(m *testing.M) { os.Exit(testgit.RunGitIsolated(m)) }

func gitTest(t *testing.T, dir string, args ...string) string {
	t.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	b, err := c.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v %s", args, err, b)
	}
	return strings.TrimSpace(string(b))
}
func fixture(t *testing.T) (string, string, string) {
	t.Helper()
	root := t.TempDir()
	gitTest(t, root, "init", "-q", "-b", "main")
	gitTest(t, root, "config", "user.name", "test")
	gitTest(t, root, "config", "user.email", "test@example.invalid")
	gitTest(t, root, "config", "commit.gpgsign", "false")
	if err := os.WriteFile(filepath.Join(root, "base"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitTest(t, root, "add", "base")
	gitTest(t, root, "commit", "-qm", "base")
	base := gitTest(t, root, "rev-parse", "HEAD")
	p := workerworktree.Prepare(root, "cli", "test", base, t.TempDir(), nil)
	if !p.OK {
		t.Fatalf("prepare: %+v", p)
	}
	return root, p.Path, base
}

// fak-test:runtime unit est=10ms lane=default
func TestSingleReapUsage(t *testing.T) {
	for _, args := range [][]string{nil, {"--root", "."}, {"--worktree", "x"}, {"--root", ".", "--worktree", "x", "--all-cold"}, {"--root", ".", "--worktree", "x", "--force"}, {"--root", ".", "--worktree", "x", "--apply"}, {"--root", ".", "--worktree", "x", "unexpected"}, {"--root", ".", "--worktree", "x", "--max-wait", "0s"}} {
		var out, err bytes.Buffer
		if rc := Run(&out, &err, args); rc != 2 {
			t.Fatalf("%v exit=%d out=%s err=%s", args, rc, &out, &err)
		}
	}
	var out, err bytes.Buffer
	if rc := Run(&out, &err, []string{"--help"}); rc != 0 {
		t.Fatal(rc)
	}
}

// fak-test:runtime unit est=1s lane=default
func TestSingleReapLifecycleAndRefusals(t *testing.T) {
	for _, kind := range []string{"accepted new file", "unproven new file", "unknown before", "foreign registration", "unregistered", "canonical root", "locked registration", "swapped sibling pointer"} {
		t.Run(kind, func(t *testing.T) {
			root, worker, base := fixture(t)
			target := worker
			commit := base
			if err := os.WriteFile(filepath.Join(worker, "new.txt"), []byte("unique\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "accepted new file":
				if err := os.WriteFile(filepath.Join(root, "new.txt"), []byte("unique\n"), 0o644); err != nil {
					t.Fatal(err)
				}
				gitTest(t, root, "add", "new.txt")
				gitTest(t, root, "commit", "-qm", "land")
				commit = gitTest(t, root, "rev-parse", "HEAD")
			case "unknown before":
				index := gitTest(t, worker, "rev-parse", "--path-format=absolute", "--git-path", "index")
				if err := os.WriteFile(index, []byte("invalid index"), 0o644); err != nil {
					t.Fatal(err)
				}
			case "foreign registration":
				_, target, _ = fixture(t)
			case "unregistered":
				target = filepath.Join(t.TempDir(), "fak-worker-wt-unregistered")
				if err := os.Mkdir(target, 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(target, "unique"), []byte("retain"), 0o644); err != nil {
					t.Fatal(err)
				}
			case "canonical root":
				target = root
			case "locked registration":
				gitTest(t, root, "worktree", "lock", worker)
			case "swapped sibling pointer":
				other := workerworktree.Prepare(root, "cli", "other", base, t.TempDir(), nil)
				if !other.OK {
					t.Fatalf("prepare sibling: %+v", other)
				}
				a, err := os.ReadFile(filepath.Join(worker, ".git"))
				if err != nil {
					t.Fatal(err)
				}
				b, err := os.ReadFile(filepath.Join(other.Path, ".git"))
				if err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(worker, ".git"), b, 0o644); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(other.Path, ".git"), a, 0o644); err != nil {
					t.Fatal(err)
				}
			}
			var out, stderr bytes.Buffer
			rc := RunSingle(&out, &stderr, root, target, commit, 5*time.Second)
			var result workerworktree.Result
			if err := json.Unmarshal(out.Bytes(), &result); err != nil {
				t.Fatalf("decode: %v out=%s err=%s", err, &out, &stderr)
			}
			if kind == "accepted new file" {
				if rc != 0 || !result.OK || !result.Removed || result.SupersededBy != commit {
					t.Fatalf("accept rc=%d result=%+v err=%s", rc, result, &stderr)
				}
				if _, err := os.Stat(target); !os.IsNotExist(err) {
					t.Fatalf("target remains: %v", err)
				}
			} else {
				if rc != 1 || result.OK || !result.Preserved {
					t.Fatalf("refusal rc=%d result=%+v", rc, result)
				}
				if _, err := os.Stat(target); err != nil {
					t.Fatalf("refused target lost: %v", err)
				}
				if kind == "swapped sibling pointer" && result.Code != "REAP_PATH_REFUSED" {
					t.Fatalf("swapped identity reached content proof: %+v", result)
				}
				if kind == "unknown before" && result.Code != "REAP_PRESTATE_UNKNOWN" {
					t.Fatalf("unknown inventory bypassed: %+v %s", result, &stderr)
				}
			}
			if kind == "accepted new file" || kind == "unproven new file" || kind == "unknown before" {
				receipts, err := wiplifecycle.ListWithDiagnostics(root)
				if err != nil {
					t.Fatal(err)
				}
				if len(receipts.Receipts) != 1 || receipts.Receipts[0].FinishedAt == "" {
					t.Fatalf("lifecycle not finished on return: %+v %s", receipts, &stderr)
				}
			}
		})
	}
}

// fak-test:runtime unit est=100ms lane=default
func TestSingleReapRejectsInheritedSelectors(t *testing.T) {
	root, worker, commit := fixture(t)
	unique := filepath.Join(worker, "unique-unlanded.txt")
	if err := os.WriteFile(unique, []byte("retain unique source"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_PREFIX", "GIT_NAMESPACE", "GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT"} {
		t.Run(key, func(t *testing.T) {
			value := "hostile-selector"
			switch key {
			case "GIT_DIR", "GIT_COMMON_DIR":
				value = filepath.Join(root, ".git")
			case "GIT_WORK_TREE":
				value = root
			case "GIT_INDEX_FILE":
				value = filepath.Join(root, ".git", "index")
			}
			t.Setenv(key, value)
			var out, stderr bytes.Buffer
			rc := RunSingle(&out, &stderr, root, worker, commit, time.Second)
			var result workerworktree.Result
			if err := json.Unmarshal(out.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if rc != 1 || result.OK || !result.Preserved || result.Code != "REAP_PATH_REFUSED" || result.Detail != key {
				t.Fatalf("selector was not refused: %d %+v", rc, result)
			}
			if _, err := os.Stat(worker); err != nil {
				t.Fatal(err)
			}
			if got, err := os.ReadFile(unique); err != nil || string(got) != "retain unique source" {
				t.Fatalf("unique source changed: %v", err)
			}
			if strings.Contains(out.String(), "hostile-selector") || strings.Contains(stderr.String(), "hostile-selector") {
				t.Fatal("selector value leaked")
			}
		})
	}
}
