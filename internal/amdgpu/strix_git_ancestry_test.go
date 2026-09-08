package amdgpu

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestStrixGitSnapshotAncestryUsesOnlyValidatedPrivateObjects(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("the fixed /usr/bin/git contract is exercised under WSL/Linux")
	}
	if _, err := os.Stat(strixGitAncestryGitPath); err != nil {
		t.Skipf("fixed Git unavailable: %v", err)
	}
	if strixGitSemanticEpoch != "c3d5ac66e4bfdfa7eee783bd7b06abb8dbaec584" {
		t.Fatalf("semantic epoch = %q", strixGitSemanticEpoch)
	}

	runGit := func(t *testing.T, repo string, stdin []byte, args ...string) string {
		t.Helper()
		cmd := exec.Command(strixGitAncestryGitPath, args...)
		cmd.Dir = repo
		cmd.Env = []string{
			"HOME=/nonexistent",
			"PATH=/usr/bin:/bin",
			"LANG=C",
			"LC_ALL=C",
			"GIT_CONFIG_NOSYSTEM=1",
			"GIT_CONFIG_GLOBAL=/dev/null",
			"GIT_AUTHOR_NAME=fak-test",
			"GIT_AUTHOR_EMAIL=fak-test@example.invalid",
			"GIT_COMMITTER_NAME=fak-test",
			"GIT_COMMITTER_EMAIL=fak-test@example.invalid",
		}
		cmd.Stdin = bytes.NewReader(stdin)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}

	live := filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(live, 0o700); err != nil {
		t.Fatal(err)
	}
	runGit(t, live, nil, "init", "-q")
	writeRevision := func(marker byte) {
		body := bytes.Repeat([]byte{'a'}, 64<<10)
		for i := 0; i < len(body); i += 4096 {
			body[i] = marker
		}
		if err := os.WriteFile(filepath.Join(live, "payload"), body, 0o600); err != nil {
			t.Fatal(err)
		}
		runGit(t, live, nil, "add", "payload")
		runGit(t, live, nil, "commit", "-q", "-m", "revision")
	}
	writeRevision('0')
	preEpoch := runGit(t, live, nil, "rev-parse", "HEAD")
	writeRevision('1')
	epoch := runGit(t, live, nil, "rev-parse", "HEAD")
	writeRevision('2')
	descendant := runGit(t, live, nil, "rev-parse", "HEAD")
	tree := runGit(t, live, nil, "rev-parse", "HEAD^{tree}")
	unrelated := runGit(t, live, []byte("unrelated\n"), "commit-tree", tree)
	runGit(t, live, nil, "update-ref", "refs/heads/unrelated", unrelated)

	assertRefusal := func(t *testing.T, got strixGitAncestry, err error) {
		t.Helper()
		if got != strixGitAncestryUnattested || err == nil {
			t.Fatalf("classification = %v, err = %v; want unattested refusal", got, err)
		}
		var refusal *strixGitAncestryRefusal
		if !errors.As(err, &refusal) || refusal.token != strixGitSnapshotUnattestedToken {
			t.Fatalf("error = %T %v, want redacted ancestry refusal", err, err)
		}
		for _, secret := range []string{live, "payload", observedSecretEnvironmentValue} {
			if strings.Contains(err.Error(), secret) {
				t.Fatalf("refusal leaked %q: %v", secret, err)
			}
		}
	}
	capture := func(t *testing.T) (*strixGitObjectSnapshot, string) {
		t.Helper()
		snapshot, err := newStrixGitObjectSnapshot(context.Background(), live)
		if err != nil {
			t.Fatal(err)
		}
		return snapshot, snapshot.gitDir()
	}
	classify := func(t *testing.T, observed string, want strixGitAncestry) {
		t.Helper()
		snapshot, snapshotPath := capture(t)
		got, err := classifyStrixGitSnapshotAncestryWithOptions(context.Background(), snapshot, observed, strixGitAncestryOptions{epoch: epoch})
		if err != nil || got != want {
			t.Fatalf("classification(%s) = %v, %v; want %v", observed, got, err, want)
		}
		if snapshot.gitDir() != "" {
			t.Fatal("successful classification retained snapshot ownership")
		}
		if _, err := os.Stat(snapshotPath); !os.IsNotExist(err) {
			t.Fatalf("classification left snapshot behind: %v", err)
		}
	}

	t.Run("loose equality descendant pre-epoch and unrelated", func(t *testing.T) {
		classify(t, epoch, strixGitAncestryEpochEqual)
		classify(t, descendant, strixGitAncestryDescendant)
		classify(t, preEpoch, strixGitAncestryPreEpoch)
		classify(t, unrelated, strixGitAncestryUnrelated)
	})

	t.Run("repacked delta history", func(t *testing.T) {
		runGit(t, live, nil, "repack", "-adf", "--window=50", "--depth=50")
		indices, err := filepath.Glob(filepath.Join(live, ".git", "objects", "pack", "*.idx"))
		if err != nil || len(indices) == 0 {
			t.Fatalf("repacked indices: %v %v", indices, err)
		}
		verify := runGit(t, live, nil, "verify-pack", "-v", indices[0])
		hasDelta := false
		for _, line := range strings.Split(verify, "\n") {
			if fields := strings.Fields(line); len(fields) >= 7 && len(fields[0]) == 40 {
				hasDelta = true
				break
			}
		}
		if !hasDelta {
			t.Fatal("real repack did not produce a delta object")
		}
		classify(t, descendant, strixGitAncestryDescendant)
	})

	t.Run("live root disappearance and hostile controls cannot influence result", func(t *testing.T) {
		snapshot, _ := capture(t)
		for name, body := range map[string][]byte{
			filepath.Join(live, ".git", "config"):                              []byte("[include]\npath=/hostile\n"),
			filepath.Join(live, ".git", "shallow"):                             []byte(epoch + "\n"),
			filepath.Join(live, ".git", "info", "grafts"):                      []byte(descendant + " " + unrelated + "\n"),
			filepath.Join(live, ".git", "objects", "info", "alternates"):       []byte("/hostile\n"),
			filepath.Join(live, ".git", "objects", "info", "commit-graph"):     []byte("hostile"),
			filepath.Join(live, ".git", "objects", "pack", "multi-pack-index"): []byte("hostile"),
			filepath.Join(live, ".git", "refs", "replace", epoch):              []byte(unrelated + "\n"),
		} {
			if err := os.MkdirAll(filepath.Dir(name), 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(name, body, 0o600); err != nil {
				t.Fatal(err)
			}
		}
		gone := live + "-gone"
		if err := os.Rename(live, gone); err != nil {
			t.Fatal(err)
		}
		if err := os.RemoveAll(gone); err != nil {
			t.Fatal(err)
		}
		got, err := classifyStrixGitSnapshotAncestryWithOptions(context.Background(), snapshot, descendant, strixGitAncestryOptions{epoch: epoch})
		if err != nil || got != strixGitAncestryDescendant {
			t.Fatalf("classification after live deletion = %v, %v", got, err)
		}
	})

	// Re-create the live repository for refusal and command-contract cases.
	live = filepath.Join(t.TempDir(), "repo")
	if err := os.MkdirAll(live, 0o700); err != nil {
		t.Fatal(err)
	}
	runGit(t, live, nil, "init", "-q")
	writeRevision('0')
	preEpoch = runGit(t, live, nil, "rev-parse", "HEAD")
	writeRevision('1')
	epoch = runGit(t, live, nil, "rev-parse", "HEAD")
	writeRevision('2')
	descendant = runGit(t, live, nil, "rev-parse", "HEAD")

	t.Run("exact fixed commands environment and cwd", func(t *testing.T) {
		t.Setenv("GIT_OBJECT_DIRECTORY", observedSecretEnvironmentValue)
		t.Setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", observedSecretEnvironmentValue)
		t.Setenv("GIT_CONFIG_COUNT", "1")
		t.Setenv("GIT_CONFIG_KEY_0", "core.sshCommand")
		t.Setenv("GIT_CONFIG_VALUE_0", observedSecretEnvironmentValue)
		t.Setenv("GIT_DIR", observedSecretEnvironmentValue)
		t.Setenv("GIT_COMMON_DIR", observedSecretEnvironmentValue)
		t.Setenv("GIT_WORK_TREE", observedSecretEnvironmentValue)
		t.Setenv("GIT_INDEX_FILE", observedSecretEnvironmentValue)
		t.Setenv("GIT_SHALLOW_FILE", observedSecretEnvironmentValue)
		t.Setenv("GIT_REPLACE_REF_BASE", observedSecretEnvironmentValue)
		t.Setenv("GIT_NAMESPACE", observedSecretEnvironmentValue)
		snapshot, snapshotPath := capture(t)
		var commands []strixGitCommand
		runner := func(ctx context.Context, command strixGitCommand) strixGitCommandResult {
			commands = append(commands, command)
			return runStrixGitCommand(ctx, command)
		}
		got, err := classifyStrixGitSnapshotAncestryWithOptions(context.Background(), snapshot, descendant, strixGitAncestryOptions{epoch: epoch, run: runner})
		if err != nil || got != strixGitAncestryDescendant {
			t.Fatalf("classification = %v, %v", got, err)
		}
		wantArgv := [][]string{
			{"--no-replace-objects", "--git-dir=" + snapshotPath, "fsck", "--full", "--strict", "--no-reflogs", "--no-dangling", descendant, epoch},
			{"--no-replace-objects", "--git-dir=" + snapshotPath, "cat-file", "-t", descendant},
			{"--no-replace-objects", "--git-dir=" + snapshotPath, "cat-file", "-t", epoch},
			{"--no-replace-objects", "--git-dir=" + snapshotPath, "merge-base", "--is-ancestor", epoch, descendant},
		}
		if len(commands) != len(wantArgv) {
			t.Fatalf("commands = %d, want %d: %#v", len(commands), len(wantArgv), commands)
		}
		wantEnv := []string{
			"HOME=/nonexistent",
			"XDG_CONFIG_HOME=/nonexistent",
			"PATH=/usr/bin:/bin",
			"LANG=C",
			"LC_ALL=C",
			"GIT_CONFIG_NOSYSTEM=1",
			"GIT_CONFIG_GLOBAL=/dev/null",
			"GIT_CONFIG_SYSTEM=/dev/null",
			"GIT_TERMINAL_PROMPT=0",
			"GIT_ASKPASS=/bin/false",
			"SSH_ASKPASS=/bin/false",
			"GIT_OPTIONAL_LOCKS=0",
			"GIT_NO_LAZY_FETCH=1",
			"GIT_PROTOCOL_FROM_USER=0",
			"GIT_ALLOW_PROTOCOL=",
			"GIT_CEILING_DIRECTORIES=/",
			"NO_COLOR=1",
		}
		for i := range commands {
			if commands[i].path != strixGitAncestryGitPath || commands[i].dir != "/" || !slices.Equal(commands[i].args, wantArgv[i]) || !slices.Equal(commands[i].env, wantEnv) {
				t.Fatalf("command[%d] = %#v; want path=%q argv=%q env=%q cwd=/", i, commands[i], strixGitAncestryGitPath, wantArgv[i], wantEnv)
			}
			for _, value := range commands[i].env {
				if strings.Contains(value, observedSecretEnvironmentValue) {
					t.Fatalf("hostile environment leaked into command: %q", value)
				}
			}
		}
	})

	t.Run("invalid revision refuses before Git and still destroys snapshot", func(t *testing.T) {
		snapshot, snapshotPath := capture(t)
		calls := 0
		got, err := classifyStrixGitSnapshotAncestryWithOptions(context.Background(), snapshot, strings.ToUpper(descendant), strixGitAncestryOptions{
			epoch: epoch,
			run: func(context.Context, strixGitCommand) strixGitCommandResult {
				calls++
				return strixGitCommandResult{}
			},
		})
		assertRefusal(t, got, err)
		if calls != 0 || snapshot.gitDir() != "" {
			t.Fatalf("invalid revision calls=%d root=%q", calls, snapshot.gitDir())
		}
		if _, statErr := os.Stat(snapshotPath); !os.IsNotExist(statErr) {
			t.Fatalf("invalid revision left snapshot: %v", statErr)
		}
	})

	t.Run("snapshot allowlist and format are revalidated before Git", func(t *testing.T) {
		for _, mutate := range []func(*testing.T, string){
			func(t *testing.T, root string) {
				if err := os.WriteFile(filepath.Join(root, "HEAD"), []byte("ref: refs/heads/main\n"), 0o600); err != nil {
					t.Fatal(err)
				}
			},
			func(t *testing.T, root string) {
				body := strixGitSnapshotConfig + "[extensions]\n\tobjectFormat = sha256\n"
				if err := os.WriteFile(filepath.Join(root, "config"), []byte(body), 0o600); err != nil {
					t.Fatal(err)
				}
			},
		} {
			snapshot, _ := capture(t)
			mutate(t, snapshot.gitDir())
			calls := 0
			got, err := classifyStrixGitSnapshotAncestryWithOptions(context.Background(), snapshot, descendant, strixGitAncestryOptions{
				epoch: epoch,
				run: func(context.Context, strixGitCommand) strixGitCommandResult {
					calls++
					return strixGitCommandResult{}
				},
			})
			assertRefusal(t, got, err)
			if calls != 0 {
				t.Fatalf("invalid snapshot invoked Git %d times", calls)
			}
		}
	})

	t.Run("missing loose object refuses strict connectivity", func(t *testing.T) {
		snapshot, _ := capture(t)
		missing := filepath.Join(snapshot.gitDir(), "objects", preEpoch[:2], preEpoch[2:])
		if err := os.Remove(missing); err != nil {
			t.Fatal(err)
		}
		got, err := classifyStrixGitSnapshotAncestryWithOptions(context.Background(), snapshot, descendant, strixGitAncestryOptions{epoch: epoch})
		assertRefusal(t, got, err)
	})

	t.Run("missing and corrupt pack data refuses", func(t *testing.T) {
		runGit(t, live, nil, "repack", "-adf", "--window=50", "--depth=50")
		for _, mode := range []string{"missing-index", "corrupt-index", "corrupt-pack"} {
			snapshot, _ := capture(t)
			packs, globErr := filepath.Glob(filepath.Join(snapshot.gitDir(), "objects", "pack", "*.pack"))
			if globErr != nil || len(packs) == 0 {
				t.Fatalf("snapshot packs: %v %v", packs, globErr)
			}
			target := strings.TrimSuffix(packs[0], ".pack") + ".idx"
			if mode == "corrupt-pack" {
				target = packs[0]
			}
			if strings.HasPrefix(mode, "corrupt-") {
				body, readErr := os.ReadFile(target)
				if readErr != nil {
					t.Fatal(readErr)
				}
				body[len(body)/2] ^= 0xff
				if writeErr := os.WriteFile(target, body, 0o600); writeErr != nil {
					t.Fatal(writeErr)
				}
			} else if removeErr := os.Remove(target); removeErr != nil {
				t.Fatal(removeErr)
			}
			got, err := classifyStrixGitSnapshotAncestryWithOptions(context.Background(), snapshot, descendant, strixGitAncestryOptions{epoch: epoch})
			assertRefusal(t, got, err)
		}
	})

	t.Run("timeout refuses and destroys snapshot", func(t *testing.T) {
		snapshot, _ := capture(t)
		got, err := classifyStrixGitSnapshotAncestryWithOptions(context.Background(), snapshot, descendant, strixGitAncestryOptions{
			epoch:   epoch,
			timeout: 5 * time.Millisecond,
			run: func(ctx context.Context, _ strixGitCommand) strixGitCommandResult {
				<-ctx.Done()
				return strixGitCommandResult{err: ctx.Err()}
			},
		})
		assertRefusal(t, got, err)
		if snapshot.gitDir() != "" {
			t.Fatal("timeout retained snapshot")
		}
	})

	t.Run("cleanup failure suppresses classification and preserves retry ownership", func(t *testing.T) {
		snapshot, owned := capture(t)
		calls := 0
		snapshot.removeAll = func(name string) error {
			calls++
			if name != owned {
				return errors.New("cleanup escaped ownership")
			}
			if calls == 1 {
				return errors.New("injected cleanup failure")
			}
			return os.RemoveAll(name)
		}
		got, err := classifyStrixGitSnapshotAncestryWithOptions(context.Background(), snapshot, descendant, strixGitAncestryOptions{epoch: epoch})
		assertRefusal(t, got, err)
		var cleanupErr *strixGitSnapshotCleanupError
		if !errors.As(err, &cleanupErr) || snapshot.gitDir() != owned {
			t.Fatalf("cleanup error/ownership = %T %v, %q; want retained %q", err, err, snapshot.gitDir(), owned)
		}
		if err := snapshot.close(); err != nil {
			t.Fatalf("cleanup retry: %v", err)
		}
	})

	t.Run("nil context and nil snapshot refuse", func(t *testing.T) {
		for _, tc := range []struct {
			ctx      context.Context
			snapshot *strixGitObjectSnapshot
		}{
			{ctx: nil, snapshot: nil},
			{ctx: context.Background(), snapshot: nil},
		} {
			got, err := classifyStrixGitSnapshotAncestry(tc.ctx, tc.snapshot, descendant)
			assertRefusal(t, got, err)
		}
		snapshot, snapshotPath := capture(t)
		got, err := classifyStrixGitSnapshotAncestry(nil, snapshot, descendant)
		assertRefusal(t, got, err)
		if snapshot.gitDir() != "" {
			t.Fatal("nil context retained snapshot ownership")
		}
		if _, statErr := os.Stat(snapshotPath); !os.IsNotExist(statErr) {
			t.Fatalf("nil context left snapshot: %v", statErr)
		}
	})
}

const observedSecretEnvironmentValue = "/live/hostile/objects"
