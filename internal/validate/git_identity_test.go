package validate

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime integration est=20s lane=default
func TestValidateGitIdentity(t *testing.T) {
	t.Run("private-tip-index-with-hostile-caller", func(t *testing.T) {
		repo, tip := identityTestRepo(t)
		wantIndex := identityTestGit(t, repo, "ls-files", "--stage", "-z")
		identityTestWrite(t, repo, "staged.json", "staged peer")
		identityTestGit(t, repo, "add", "staged.json")
		identityTestWrite(t, repo, "tracked.json", "unstaged peer")
		identityTestWrite(t, repo, "untracked.json", "untracked peer")
		// Fetch must not execute the source repository's configured pack hook.
		identityTestGit(t, repo, "config", "uploadpack.packObjectsHook", "exit 97")
		identityTestWrite(t, repo, ".git/hooks/reference-transaction", "must never execute")
		before := identityTestSnapshot(t, repo)
		dir := t.TempDir()
		identityTestWrite(t, dir, "tracked.json", "owned overlay")
		identityTestWrite(t, dir, "new-owned.json", "new overlay remains untracked")
		for _, key := range []string{"GIT_DIR", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY"} {
			t.Setenv(key, filepath.Join(repo, ".git"))
		}
		t.Setenv("GIT_WORK_TREE", repo)
		t.Setenv("GIT_INDEX_FILE", filepath.Join(repo, ".git", "index"))
		t.Setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", filepath.Join(repo, ".git", "objects"))
		t.Setenv("GIT_PREFIX", "peer/")
		t.Setenv("GIT_NAMESPACE", "peer")
		t.Setenv("GIT_CONFIG_PARAMETERS", "malformed inherited configuration")
		t.Setenv("GIT_CONFIG_COUNT", "1")
		t.Setenv("GIT_CONFIG_KEY_0", "core.worktree")
		t.Setenv("GIT_CONFIG_VALUE_0", repo)
		t.Setenv("GIT_TEMPLATE_DIR", filepath.Join(repo, ".git"))
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := prepareValidateGitIdentityWithin(ctx, repo, dir, tip); err != nil {
			t.Fatal(err)
		}
		if got := identityTestGit(t, dir, "rev-parse", "HEAD"); strings.TrimSpace(got) != tip {
			t.Fatalf("HEAD=%q, want %s", got, tip)
		}
		if head, err := os.ReadFile(filepath.Join(dir, ".git", "HEAD")); err != nil || strings.TrimSpace(string(head)) != tip {
			t.Fatalf("HEAD is not detached at the requested tip: %q, %v", head, err)
		}
		if got := identityTestGit(t, dir, "ls-files", "--stage", "-z"); got != wantIndex {
			t.Fatalf("candidate index=%q, want requested-tip entries=%q", got, wantIndex)
		}
		top := strings.TrimSpace(identityTestGit(t, dir, "rev-parse", "--show-toplevel"))
		wantTop, err := filepath.EvalSymlinks(dir)
		if err != nil || filepath.Clean(top) != filepath.Clean(wantTop) {
			t.Fatalf("candidate top-level=%q, want %q: %v", top, wantTop, err)
		}
		if _, err := os.Stat(filepath.Join(dir, ".git", "objects", "info", "alternates")); !os.IsNotExist(err) {
			t.Fatalf("candidate unexpectedly depends on alternates: %v", err)
		}
		if entries, err := os.ReadDir(filepath.Join(dir, ".git", "fak-empty")); err != nil || len(entries) != 0 {
			t.Fatalf("template/hooks directory is not empty: %v, %v", entries, err)
		}
		if body, err := os.ReadFile(filepath.Join(dir, "tracked.json")); err != nil || string(body) != "owned overlay" {
			t.Fatalf("identity setup changed overlay bytes: %q, %v", body, err)
		}
		if after := identityTestSnapshot(t, repo); !reflect.DeepEqual(before, after) {
			t.Fatal("source files, index, config, HEAD, refs, hooks or objects changed")
		}
		// Prove the fetched identity still works after the source is unavailable.
		moved := repo + "-unavailable"
		if err := os.Rename(repo, moved); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := os.Rename(moved, repo); err != nil {
				t.Error(err)
			}
		})
		if got := identityTestGit(t, dir, "show", "HEAD:tracked.json"); got != "{}\n" {
			t.Fatalf("candidate cannot read its own objects: %q", got)
		}
	})

	t.Run("refuse-existing-metadata-and-clean-failed-fetch", func(t *testing.T) {
		repo, tip := identityTestRepo(t)
		before := identityTestSnapshot(t, repo)
		for _, shape := range []string{"file", "directory", "symlink"} {
			t.Run(shape, func(t *testing.T) {
				dir := t.TempDir()
				gitDir := filepath.Join(dir, ".git")
				switch shape {
				case "file":
					identityTestWrite(t, dir, ".git", "gitdir: "+filepath.Join(repo, ".git")+"\n")
				case "directory":
					identityTestWrite(t, dir, ".git/keep", "owned by someone else")
				case "symlink":
					if err := os.Symlink(filepath.Join(repo, ".git"), gitDir); err != nil {
						t.Skipf("symlink unavailable: %v", err)
					}
				}
				preserve := identityTestSnapshot(t, dir)
				if err := prepareValidateGitIdentityWithin(context.Background(), repo, dir, tip); err == nil {
					t.Fatal("existing metadata accepted")
				}
				if after := identityTestSnapshot(t, dir); !reflect.DeepEqual(preserve, after) {
					t.Fatal("pre-existing metadata changed")
				}
			})
		}
		dir := t.TempDir()
		identityTestWrite(t, dir, "tracked.json", "preserve overlay")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := prepareValidateGitIdentityWithin(ctx, repo, dir, strings.Repeat("0", len(tip))); err == nil || !strings.Contains(err.Error(), "git fetch:") {
			t.Fatalf("missing object did not fail at fetch: %v", err)
		}
		if _, err := os.Lstat(filepath.Join(dir, ".git")); !os.IsNotExist(err) {
			t.Fatalf("failed identity left metadata: %v", err)
		}
		if body, err := os.ReadFile(filepath.Join(dir, "tracked.json")); err != nil || string(body) != "preserve overlay" {
			t.Fatalf("failed identity changed overlay: %q, %v", body, err)
		}
		cancelled, stop := context.WithCancel(context.Background())
		stop()
		if err := prepareValidateGitIdentityWithin(cancelled, repo, dir, tip); err != context.Canceled {
			t.Fatalf("cancelled preparation=%v", err)
		}
		if _, err := os.Lstat(filepath.Join(dir, ".git")); !os.IsNotExist(err) {
			t.Fatalf("cancelled identity left metadata: %v", err)
		}
		if after := identityTestSnapshot(t, repo); !reflect.DeepEqual(before, after) {
			t.Fatal("failed preparation changed source")
		}
	})

	for _, mode := range []string{"native", "native-events", "wsl"} {
		t.Run(mode, func(t *testing.T) {
			if testing.Short() {
				t.Skip("nested Go integration")
			}
			if mode == "wsl" {
				overlayWSLTestRequire(t)
			} else if runtime.GOOS == "windows" {
				t.Skip("native Go integration on non-Windows hosts")
			}
			eventFixtureEnvironment(t)
			repo, tip := identityTestRepo(t)
			identityTestWrite(t, repo, "app/app.go", "package app\nfunc Value() string { return \"owned\" }\n")
			identityTestWrite(t, repo, "peer.json", "peer untracked")
			before := identityTestSnapshot(t, repo)
			var stdout, stderr bytes.Buffer
			args := []string{"--root", repo, "--ref", tip, "--mine", "app/app.go", "--mine", "app/app_test.go", "--mine", "requested-tip.txt", "--test-only", "--json", "--progress=false", "--timeout", "1m", "--wsl-tests=false"}
			if mode == "native-events" {
				args = append(args, "--test-events")
			}
			if mode == "wsl" {
				args = append(args, "--wsl-tests=true")
			}
			code := Run(&stdout, &stderr, args)
			var got validateResult
			if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
				t.Fatalf("decode: %v; code=%d stdout=%s stderr=%s", err, code, stdout.String(), stderr.String())
			}
			if code != 0 || !got.OK {
				t.Fatalf("code=%d receipt=%s stderr=%s", code, stdout.String(), stderr.String())
			}
			if !reflect.DeepEqual(got.Tested, []string{"fixture.test/identity/app"}) {
				t.Fatalf("selected tests=%v, want the fixture app", got.Tested)
			}
			testPassed := false
			identityPhases := 0
			for _, phase := range got.Phases {
				if phase.Name == "test" && phase.Status == "ok" {
					testPassed = true
				}
				if phase.Name == "git_identity" && phase.Status == "ok" {
					identityPhases++
				}
			}
			wantIdentityPhases := 1
			if mode == "wsl" {
				wantIdentityPhases = 0 // WSL materializes its identity before returning.
			}
			if !testPassed || identityPhases != wantIdentityPhases {
				t.Fatalf("identity phases=%d, want %d", identityPhases, wantIdentityPhases)
			}
			if mode == "native-events" {
				if got.TestEvents == nil || !got.TestEvents.Complete || len(got.TestEvents.Tests) != 1 || got.TestEvents.Tests[0].Test != "TestCandidateIdentity" || got.TestEvents.Tests[0].Action != "pass" || !bytes.Contains(got.TestEvents.Stdout, []byte("IDENTITY_ASSERTIONS_COMPLETED")) {
					t.Fatalf("missing nonvacuous event witness: %+v", got.TestEvents)
				}
				if _, err := os.Stat(got.TestEvents.Candidate.Root); !os.IsNotExist(err) {
					t.Fatalf("candidate leaked after completion: %v", err)
				}
			}
			if after := identityTestSnapshot(t, repo); !reflect.DeepEqual(before, after) {
				t.Fatal("validation changed source")
			}
		})
	}

	t.Run("metadata-overlay-fails-closed", func(t *testing.T) {
		repo, tip := identityTestRepo(t)
		before := identityTestSnapshot(t, repo)
		var stdout, stderr bytes.Buffer
		code := Run(&stdout, &stderr, []string{"--root", repo, "--ref", tip, "--mine", ".git/config", "--test-only", "--wsl-tests=false", "--json", "--progress=false"})
		var got validateResult
		if err := json.Unmarshal(stdout.Bytes(), &got); err != nil || code != 1 || got.OK || len(got.Failures) != 1 || got.Failures[0].Step != "git_identity" {
			t.Fatalf("metadata overlay: code=%d decode=%v stdout=%s stderr=%s", code, err, stdout.String(), stderr.String())
		}
		for _, phase := range got.Phases {
			if phase.Name == "test" || phase.Name == "list_graph" {
				t.Fatalf("metadata overlay reached %s", phase.Name)
			}
		}
		if after := identityTestSnapshot(t, repo); !reflect.DeepEqual(before, after) {
			t.Fatal("metadata overlay changed source")
		}
	})

	t.Run("wsl-materializer-retains-identity-and-failure-cleanup", func(t *testing.T) {
		for _, fail := range []bool{false, true} {
			t.Run(fmt.Sprint(fail), func(t *testing.T) {
				repo := t.TempDir()
				fake := installWSLRetryTestFake(t, func(int) ([]byte, error) {
					if fail {
						return []byte("fatal: identity read-tree failed"), errors.New("exit status 128")
					}
					return nil, nil
				})
				dir, err := extractCommittedTipWSLWithin(context.Background(), repo, wslRetryTestTip)
				scripts := wslRetryTestScripts(t, fake, repo)
				if len(scripts) != 1 {
					t.Fatalf("materialization calls=%d", len(scripts))
				}
				script := scripts[0]
				init := strings.Index(script, " init --quiet;")
				index := strings.Index(script, " read-tree "+posixQuote(wslRetryTestTip))
				head := strings.Index(script, " update-ref --no-deref HEAD "+posixQuote(wslRetryTestTip))
				if init < 0 || index <= init || head <= index || !strings.Contains(script, "/.git/objects/info/alternates") {
					t.Fatalf("existing WSL Git identity contract changed: %s", script)
				}
				_, cleanups := fake.counts()
				if fail {
					if dir != "" || err == nil || !strings.Contains(err.Error(), "identity read-tree failed") || cleanups != 1 {
						t.Fatalf("WSL identity failure dir=%q err=%v cleanups=%d", dir, err, cleanups)
					}
					_ = wslRetryTestCleanupTarget(t, fake)
				} else if dir == "" || err != nil || cleanups != 0 {
					t.Fatalf("WSL identity success dir=%q err=%v cleanups=%d", dir, err, cleanups)
				}
			})
		}
	})

	for _, failure := range []string{"missing-git", "timeout"} {
		t.Run(failure, func(t *testing.T) {
			if testing.Short() {
				t.Skip("validation timeout integration")
			}
			repo, tip := identityTestRepo(t)
			temp := t.TempDir()
			for _, key := range []string{"TMPDIR", "TMP", "TEMP"} {
				t.Setenv(key, temp)
			}
			previous := validatePhaseHook
			t.Cleanup(func() { validatePhaseHook = previous })
			reached, ranTests := false, false
			validatePhaseHook = func(ctx context.Context, name string) {
				if name == "test" {
					ranTests = true
				}
				if name != "git_identity" {
					return
				}
				reached = true
				if failure == "timeout" {
					<-ctx.Done()
				} else {
					t.Setenv("PATH", t.TempDir())
				}
			}
			var stdout, stderr bytes.Buffer
			code := Run(&stdout, &stderr, []string{"--root", repo, "--ref", tip, "--mine", "app/app_test.go", "--test-only", "--wsl-tests=false", "--json", "--progress=false", "--timeout", "5s"})
			var got validateResult
			if err := json.Unmarshal(stdout.Bytes(), &got); err != nil || !reached || ranTests || code != 1 || got.OK {
				t.Fatalf("reached=%v ranTests=%v code=%d decode=%v stdout=%s stderr=%s", reached, ranTests, code, err, stdout.String(), stderr.String())
			}
			if got.TimedOut != (failure == "timeout") {
				t.Fatalf("timeout classification=%v for %s", got.TimedOut, failure)
			}
			if len(got.Failures) == 0 || got.Failures[0].Step != "git_identity" {
				t.Fatalf("identity failure not retained: %+v", got.Failures)
			}
			matches, err := filepath.Glob(filepath.Join(temp, "fak-committed-tree-*"))
			if err != nil || len(matches) != 0 {
				t.Fatalf("failed validation leaked owned candidates: %v, %v", matches, err)
			}
		})
	}
}

func identityTestRepo(t *testing.T) (string, string) {
	t.Helper()
	repo := t.TempDir()
	identityTestWrite(t, repo, "go.mod", "module fixture.test/identity\n\ngo 1.26\n")
	identityTestWrite(t, repo, "tracked.json", "{}\n")
	identityTestWrite(t, repo, "app/app.go", "package app\nfunc Value() string { return \"base\" }\n")
	identityTestWrite(t, repo, "app/app_test.go", `package app
import("bytes";"os";"os/exec";"path/filepath";"strings";"testing")
func TestMain(m *testing.M) {
 for _, key := range []string{"GIT_DIR","GIT_WORK_TREE","GIT_INDEX_FILE","GIT_COMMON_DIR","GIT_OBJECT_DIRECTORY","GIT_ALTERNATE_OBJECT_DIRECTORIES","GIT_PREFIX","GIT_NAMESPACE"} { os.Unsetenv(key) }
 os.Exit(m.Run())
}
func TestCandidateIdentity(t *testing.T) {
 if Value() != "owned" { t.Fatal("owned overlay missing") }
 root,err:=filepath.Abs("..");if err!=nil{t.Fatal(err)}
 run:=func(args ...string) []byte { c:=exec.Command("git",args...);c.Dir=root;b,e:=c.CombinedOutput();if e!=nil{t.Fatalf("git %v: %v: %s",args,e,b)};return b }
 if info,err:=os.Lstat(filepath.Join(root,".git"));err!=nil||!info.IsDir(){t.Fatalf("private .git missing: %v",err)}
 head:=strings.TrimSpace(string(run("rev-parse","HEAD")))
 expected,err:=os.ReadFile(filepath.Join(root,"requested-tip.txt"));if err!=nil{t.Fatal(err)}
 if want:=strings.TrimSpace(string(expected));want==""||head!=want{t.Fatalf("HEAD=%q want=%q",head,want)}
 entries:=bytes.Split(bytes.TrimRight(run("ls-files","-z","--","*.json"),"\x00"),[]byte{0})
 if len(entries)!=1||string(entries[0])!="tracked.json"{t.Fatalf("tracked JSON corpus=%q",entries)}
 body,err:=os.ReadFile(filepath.Join(root,string(entries[0])));if err!=nil||!bytes.Equal(body,[]byte("{}\n")){t.Fatalf("tracked JSON bytes=%q: %v",body,err)}
 if _,err:=os.Stat(filepath.Join(root,"peer.json"));!os.IsNotExist(err){t.Fatalf("peer leaked: %v",err)}
 t.Log("IDENTITY_ASSERTIONS_COMPLETED")
}
`)
	identityTestGit(t, repo, "init", "--quiet", "--template="+t.TempDir())
	identityTestGit(t, repo, "add", ".")
	identityTestGit(t, repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "fixture")
	tip := strings.TrimSpace(identityTestGit(t, repo, "rev-parse", "HEAD"))
	identityTestWrite(t, repo, "requested-tip.txt", tip+"\n")
	return repo, tip
}

func identityTestWrite(t *testing.T, root, name, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func identityTestGit(t *testing.T, root string, args ...string) string {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = root
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if !strings.HasPrefix(strings.ToUpper(key), "GIT_") {
			cmd.Env = append(cmd.Env, entry)
		}
	}
	cmd.Env = append(cmd.Env, "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_SYSTEM="+os.DevNull, "GIT_CONFIG_GLOBAL="+os.DevNull, "GIT_TERMINAL_PROMPT=0", "GIT_OPTIONAL_LOCKS=0")
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, stderr.String())
	}
	return string(out)
}

func identityTestSnapshot(t *testing.T, root string) map[string]string {
	t.Helper()
	snapshot := make(map[string]string)
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		name, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if entry.Type()&os.ModeSymlink != 0 {
			target, err := os.Readlink(path)
			snapshot[name] = "symlink:" + target
			return err
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		snapshot[name] = fmt.Sprintf("%o:%x", info.Mode(), sha256.Sum256(body))
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}
