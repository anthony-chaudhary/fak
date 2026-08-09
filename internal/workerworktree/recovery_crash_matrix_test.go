package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestRecoveryCrashMatrixWorkerLand covers #5999 rows 4-6 and 8-9 at the
// package-private commit-tree/CAS seam, using real repositories and bare remotes.
func TestRecoveryCrashMatrixWorkerLand(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	t.Run("local-candidate-before-cas", func(t *testing.T) {
		repo, wt, base := prepareCrashWorker(t, false)
		res, handled := crashLand(t, repo, wt, base, landConfig{})
		if handled || res.OK || crashGit(t, repo, "rev-parse", "HEAD") != base {
			t.Fatalf("land=%+v handled=%v trace=%s", res, handled, crashTrace)
		}
		entries, err := RecoveryEntries(repo, nil)
		if err != nil || len(entries) != 1 || entries[0].State != "RECOVERABLE" || entries[0].Durability != DurabilityLocalOnly {
			t.Fatalf("entries=%+v err=%v", entries, err)
		}
		if got := strings.TrimSpace(crashGit(t, repo, "show", entries[0].Ref+":a.txt")); got != "worker" {
			t.Fatalf("bytes=%q", got)
		}
	})
	t.Run("remote-host-loss-fresh-clone", func(t *testing.T) {
		repo, wt, base := prepareCrashWorker(t, true)
		res, handled := crashLand(t, repo, wt, base, landConfig{recoveryRemote: "origin", requireRemote: true})
		if handled || res.OK {
			t.Fatalf("land=%+v handled=%v trace=%s", res, handled, crashTrace)
		}
		remoteRows, err := RecoveryEntries(repo, nil)
		if err != nil || len(remoteRows) != 1 || remoteRows[0].Durability != DurabilityReplicated {
			t.Fatalf("source entries=%+v err=%v", remoteRows, err)
		}
		fresh := filepath.Join(t.TempDir(), "fresh")
		crashGit(t, filepath.Dir(fresh), "clone", filepath.Join(filepath.Dir(repo), "remote.git"), fresh)
		crashGit(t, fresh, "checkout", "-q", "origin/main")
		if err := FetchRecoveryMirror(fresh, "origin", nil); err != nil {
			t.Fatal(err)
		}
		entries, err := RecoveryEntries(fresh, nil)
		if err != nil || len(entries) != 1 || entries[0].Durability != DurabilityRemoteOnly {
			t.Fatalf("entries=%+v err=%v", entries, err)
		}
		if got := strings.TrimSpace(crashGit(t, fresh, "show", entries[0].MirrorRef+":a.txt")); got != "worker" {
			t.Fatalf("bytes=%q", got)
		}
	})
	t.Run("post-cas-landed-no-duplicate", func(t *testing.T) {
		repo, wt, base := prepareCrashWorker(t, false)
		diff := crashGit(t, wt, "diff", base+"..HEAD")
		res, handled := landIsolated(repo, wt, diff, crashMessage(t), []string{"a.txt"}, nil, nil)
		if !handled || !res.OK {
			t.Fatalf("land=%+v handled=%v", res, handled)
		}
		head := crashGit(t, repo, "rev-parse", "HEAD")
		entries, err := RecoveryEntries(repo, nil)
		if err != nil || len(entries) != 1 || entries[0].State != "LANDED" || entries[0].Durability != DurabilityLanded {
			t.Fatalf("entries=%+v err=%v", entries, err)
		}
		again := Land(repo, wt, base, "", []string{"a.txt"}, nil, nil)
		if again.Committed || crashGit(t, repo, "rev-parse", "HEAD") != head {
			t.Fatalf("duplicate=%+v", again)
		}
	})
	t.Run("cleanup-refuses-and-report-is-immutable", func(t *testing.T) {
		repo, wt, base := prepareCrashWorker(t, true)
		_, _ = crashLand(t, repo, wt, base, landConfig{recoveryRemote: "origin", requireRemote: true})
		rows, err := RecoveryEntries(repo, nil)
		if err != nil || len(rows) != 1 {
			t.Fatalf("rows=%+v err=%v", rows, err)
		}
		ref := rows[0].Ref
		localBefore := crashGit(t, repo, "for-each-ref", "--format=%(refname) %(objectname)", "refs/fak")
		remoteBefore := crashGit(t, repo, "ls-remote", "--refs", "origin", ref)
		if err := DeleteRecoveryRef(repo, ref, false, nil); err == nil {
			t.Fatal("unlanded cleanup allowed")
		}
		report := CleanupRemoteRecoveryRef(repo, "origin", ref, filepath.Base(wt), false, false, nil)
		if report.Applied || crashGit(t, repo, "for-each-ref", "--format=%(refname) %(objectname)", "refs/fak") != localBefore || crashGit(t, repo, "ls-remote", "--refs", "origin", ref) != remoteBefore {
			t.Fatalf("report mutated refs: %+v", report)
		}
		if err := DeleteRecoveryRef(repo, RecoveryRefPrefix+"unknown/deadbeef", false, nil); err == nil {
			t.Fatal("unknown cleanup allowed")
		}
	})
	t.Run("remote-outage-never-replicated", func(t *testing.T) {
		repo, wt, base := prepareCrashWorker(t, false)
		_, _ = crashLand(t, repo, wt, base, landConfig{recoveryRemote: filepath.Join(t.TempDir(), "offline.git")})
		entries, err := RecoveryEntries(repo, nil)
		if err != nil || len(entries) != 1 || entries[0].Durability != DurabilityLocalOnly {
			t.Fatalf("entries=%+v err=%v", entries, err)
		}
	})
}
func crashLand(t *testing.T, repo, wt, base string, cfg landConfig) (Result, bool) {
	t.Helper()
	t.Setenv(IsolatedLandRetryEnv, "1")
	diff := crashGit(t, wt, "diff", base+"..HEAD")
	return landIsolated(repo, wt, diff, crashMessage(t), []string{"a.txt"}, crashCASGit, crashEnvGit, cfg)
}
func prepareCrashWorker(t *testing.T, remote bool) (repo, wt, base string) {
	t.Helper()
	root := t.TempDir()
	repo = filepath.Join(root, "repo")
	os.Mkdir(repo, 0o755)
	crashGit(t, repo, "init", "-q", "-b", "main")
	crashGit(t, repo, "config", "user.name", "Test")
	crashGit(t, repo, "config", "user.email", "test-at-example.invalid")
	crashGit(t, repo, "config", "commit.gpgsign", "false")
	os.WriteFile(filepath.Join(repo, "a.txt"), []byte("one\n"), 0o644)
	crashGit(t, repo, "add", "a.txt")
	crashGit(t, repo, "commit", "-q", "-m", "base")
	base = crashGit(t, repo, "rev-parse", "HEAD")
	if remote {
		bare := filepath.Join(root, "remote.git")
		crashGit(t, root, "init", "--bare", "-q", bare)
		crashGit(t, repo, "remote", "add", "origin", bare)
		crashGit(t, repo, "push", "-q", "-u", "origin", "main")
	}
	p := Prepare(repo, "matrix", "5999", base, filepath.Join(root, "workers"), nil)
	if !p.OK {
		t.Fatalf("prepare=%+v", p)
	}
	wt = p.Path
	os.WriteFile(filepath.Join(wt, "a.txt"), []byte("worker\n"), 0o644)
	crashGit(t, wt, "config", "user.name", "Worker")
	crashGit(t, wt, "config", "user.email", "worker-at-example.invalid")
	crashGit(t, wt, "add", "a.txt")
	crashGit(t, wt, "commit", "-q", "-m", "test(recovery): matrix worker (#5999) (fak workerworktree)")
	return
}
func crashMessage(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "message.txt")
	os.WriteFile(path, []byte("test(recovery): matrix worker (#5999) (fak workerworktree)\n"), 0o644)
	return path
}
func crashCASGit(root string, args []string) (int, string) { return crashRun(root, nil, args) }
func crashEnvGit(root string, env map[string]string, args []string) (int, string) {
	return crashRun(root, env, args)
}

var crashTrace string

func crashRun(root string, env map[string]string, args []string) (int, string) {
	crashTrace += strings.Join(args, " ") + "\n"
	if len(args) >= 2 && args[0] == "update-ref" && args[1] == "refs/heads/main" {
		return 1, "injected crash before trunk CAS"
	}
	c := exec.Command("git", args...)
	c.Dir = root
	if len(env) > 0 {
		c.Env = os.Environ()
		for k, v := range env {
			c.Env = append(c.Env, k+"="+v)
		}
	}
	out, err := c.CombinedOutput()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), string(out)
		}
		return 127, err.Error()
	}
	return 0, string(out)
}
func crashGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	out, err := c.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}
