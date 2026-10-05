package workerworktree

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func preservingTestRunner(t *testing.T, calls *[]string) GitRunner {
	t.Helper()
	git, err := exec.LookPath("git")
	if err != nil {
		t.Fatal(err)
	}
	git, err = filepath.Abs(git)
	if err != nil {
		t.Fatal(err)
	}
	env := preservingLibraryFixtureEnv(os.Environ(), git, preservingFixtureIdentity(t))
	return func(dir string, args []string) (int, string) {
		joined := strings.Join(args, " ")
		*calls = append(*calls, joined)
		for _, forbidden := range []string{"worktree remove", "worktree prune", "worktree unlock", "reset ", "clean ", "update-ref", "--force"} {
			if strings.Contains(joined, forbidden) {
				t.Fatalf("preserving prepare issued mutation %q", joined)
			}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, git, args...)
		cmd.Dir = dir
		cmd.Env = env
		output, err := cmd.CombinedOutput()
		if err != nil {
			if ctx.Err() != nil {
				return ReapTimeoutExitCode, string(output)
			}
			if exit, ok := err.(*exec.ExitError); ok {
				return exit.ExitCode(), string(output)
			}
			return 127, string(output)
		}
		return 0, string(output)
	}
}

func TestPreparePreservingCreatesExactNewWorkerWithoutCleanup(t *testing.T) {
	f := newPreservingFixture(t)
	root := preservingTempDir(t)
	var calls []string
	checks := 0
	res := preparePreserving(context.Background(), f.repo, "cmd", "new-only", f.base, root,
		OwnerStamp{PID: os.Getpid(), LeaseID: "test-admitted-lease"}, preservingTestRunner(t, &calls), func(context.Context) error { checks++; return nil })
	if !res.OK || res.Reused || checks < 3 {
		t.Fatalf("prepare=%+v admission checks=%d", res, checks)
	}
	assertPreservingTreeLive(t, f.repo, res.Path)
	if got := preservingFixtureGit(t, res.Path, "rev-parse", "HEAD"); strings.TrimSpace(got) != f.base {
		t.Fatalf("HEAD=%s", got)
	}
	stamp, err := readOwnerStamp(res.Path)
	if err != nil || stamp.PID != os.Getpid() || stamp.LeaseID != "test-admitted-lease" {
		t.Fatalf("owner=%+v err=%v", stamp, err)
	}
	if _, err := os.Stat(poolMemberPath(res.Path)); !os.IsNotExist(err) {
		t.Fatalf("unexpected pool state: %v", err)
	}
}

func TestPreparePreservingKeepsDirtyUnknownAndIdleWorkers(t *testing.T) {
	f := newPreservingFixture(t)
	root := preservingTempDir(t)
	neighbors := []string{Path("cmd", "dirty", root), Path("cmd", "unknown", root), Path("cmd", "idle", root)}
	before := map[string]string{}
	for i, p := range neighbors {
		preservingFixtureGit(t, f.repo, "worktree", "add", "--detach", p, f.base)
		name := filepath.Join(p, "preserve-proof.txt")
		text := fmt.Sprintf("source and evidence %d\n", i)
		if err := os.WriteFile(name, []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
		before[name] = text
	}
	old := time.Now().Add(-time.Hour)
	if err := writeOwnerStamp(neighbors[0], OwnerStamp{PID: deadOwnerPID, LeaseID: "old", CreatedAt: old}); err != nil {
		t.Fatal(err)
	}
	idle := neighbors[2]
	if err := recordPoolLease(idle, "cmd", OwnerStamp{PID: deadOwnerPID, LeaseID: "old", CreatedAt: old}); err != nil {
		t.Fatal(err)
	}
	// Mark the neighbor idle without invoking pool cleanup or GC.
	meta, err := readPoolMember(idle)
	if err != nil {
		t.Fatal(err)
	}
	meta.State = poolStateIdle
	if err := atomicWriteJSON(poolMemberPath(idle), meta, 0o600); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{OwnerStampPath(neighbors[0]), poolMemberPath(idle)} {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		before[p] = string(b)
	}
	var calls []string
	res := preparePreserving(context.Background(), f.repo, "cmd", "new-neighbor", f.base, root,
		OwnerStamp{PID: os.Getpid(), LeaseID: "test-admitted-lease"}, preservingTestRunner(t, &calls), func(context.Context) error { return nil })
	if !res.OK {
		t.Fatalf("prepare=%+v", res)
	}
	for p, expected := range before {
		b, err := os.ReadFile(p)
		if err != nil || string(b) != expected {
			t.Fatalf("neighbor changed: %s (%v)", p, err)
		}
	}
	for _, p := range neighbors {
		assertPreservingTreeLive(t, f.repo, p)
	}
}

func TestPreparePreservingRefusesExistingTargetAndSidecars(t *testing.T) {
	for _, kind := range []string{"directory", "symlink", "owner", "pool", "idle", "intent", "message"} {
		t.Run(kind, func(t *testing.T) {
			f := newPreservingFixture(t)
			root := preservingTempDir(t)
			target := Path("cmd", "collision", root)
			var path string
			switch kind {
			case "directory":
				if err := os.Mkdir(target, 0o755); err != nil {
					t.Fatal(err)
				}
			case "symlink":
				if err := os.Symlink("missing-preserved-target", target); err != nil {
					t.Fatal(err)
				}
			case "owner":
				path = OwnerStampPath(target)
			case "pool":
				path = poolMemberPath(target)
			case "idle":
				path = poolLegacyMarker(filepath.Dir(target), filepath.Base(target))
			case "message":
				path = messagePath(target)
			case "intent":
				path = intentPath(target)
			}
			if path != "" {
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte("unknown preserve metadata"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			var calls []string
			res := preparePreserving(context.Background(), f.repo, "cmd", "collision", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, preservingTestRunner(t, &calls), func(context.Context) error { return nil })
			if res.OK || !res.Preserved {
				t.Fatalf("collision=%+v", res)
			}
			for _, c := range calls {
				if strings.Contains(c, "worktree add") {
					t.Fatalf("add on conflict: %s", c)
				}
			}
			if path != "" {
				b, _ := os.ReadFile(path)
				if string(b) != "unknown preserve metadata" {
					t.Fatal("metadata changed")
				}
			} else if _, err := os.Lstat(target); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestPreparePreservingRefusesRegistrationAndUnreadableInventory(t *testing.T) {
	for _, kind := range []string{"registration", "unreadable"} {
		t.Run(kind, func(t *testing.T) {
			f := newPreservingFixture(t)
			root := preservingTempDir(t)
			target := Path("cmd", "new", root)
			var calls []string
			baseGit := preservingTestRunner(t, &calls)
			git := func(dir string, args []string) (int, string) {
				if strings.Join(args, " ") == "worktree list --porcelain" {
					switch kind {
					case "registration":
						return 0, "worktree " + target + "\nprunable\n"
					case "unreadable":
						return 1, "inventory unavailable"
					}
				}
				return baseGit(dir, args)
			}
			res := preparePreserving(context.Background(), f.repo, "cmd", "new", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, git, func(context.Context) error { return nil })
			if res.OK || !res.Preserved {
				t.Fatalf("refusal=%+v", res)
			}
			for _, c := range calls {
				if strings.Contains(c, "worktree add") {
					t.Fatalf("created after refusal: %s", c)
				}
			}
		})
	}
}

func TestPreparePreservingRetainsPartialAddAndReadinessFailure(t *testing.T) {
	for _, kind := range []string{"index-failure", "timeout", "wrong-head"} {
		t.Run(kind, func(t *testing.T) {
			f := newPreservingFixture(t)
			root := preservingTempDir(t)
			target := Path("cmd", "partial", root)
			var calls []string
			baseGit := preservingTestRunner(t, &calls)
			git := func(dir string, args []string) (int, string) {
				joined := strings.Join(args, " ")
				if strings.Contains(joined, "worktree add") && kind != "wrong-head" {
					if err := os.Mkdir(target, 0o755); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(filepath.Join(target, "partial-proof"), []byte("retain"), 0o644); err != nil {
						t.Fatal(err)
					}
					if kind == "timeout" {
						return ReapTimeoutExitCode, "deadline"
					}
					return 1, "index failure"
				}
				if dir == target && joined == "rev-parse HEAD" && kind == "wrong-head" {
					return 0, "wrong-head"
				}
				return baseGit(dir, args)
			}
			res := preparePreserving(context.Background(), f.repo, "cmd", "partial", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, git, func(context.Context) error { return nil })
			if res.OK || !res.Preserved {
				t.Fatalf("failure=%+v", res)
			}
			if _, err := os.Stat(target); err != nil {
				t.Fatalf("partial removed: %v", err)
			}
			if _, err := os.Stat(OwnerStampPath(target)); !os.IsNotExist(err) {
				t.Fatalf("ready metadata after failure: %v", err)
			}
		})
	}
}

func TestPreparePreservingRetainsAdmissionAndTargetLockRefusals(t *testing.T) {
	f := newPreservingFixture(t)
	root := preservingTempDir(t)
	var calls []string
	git := preservingTestRunner(t, &calls)
	owner := OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}
	refused := preparePreserving(context.Background(), f.repo, "cmd", "refused", f.base, root, owner, git, func(context.Context) error { return fmt.Errorf("stale lease or low disk") })
	if refused.OK || refused.Code != "PRESERVATION_ADMISSION_REFUSED" || len(calls) != 0 {
		t.Fatalf("admission=%+v calls=%v", refused, calls)
	}
	t.Setenv(PrepareLockWaitEnv, "1ms")
	target := Path("cmd", "busy", root)
	if err := withPrepareTargetLock(target, time.Millisecond, func() {
		res := preparePreserving(context.Background(), f.repo, "cmd", "busy", f.base, root, owner, git, func(context.Context) error { return nil })
		if res.OK || res.Code != PrepareCodeBusy || !res.Preserved {
			t.Fatalf("lock refusal=%+v", res)
		}
	}); err != nil {
		t.Fatal(err)
	}
}

func TestPreparePreservingKeepsOldCleanRegistration(t *testing.T) {
	f := newPreservingFixture(t)
	root := preservingTempDir(t)
	neighbor := Path("cmd", "old-clean", root)
	preservingFixtureGit(t, f.repo, "worktree", "add", "--detach", neighbor, f.base)
	old := time.Now().Add(-2 * time.Hour)
	if err := writeOwnerStamp(neighbor, OwnerStamp{PID: deadOwnerPID, LeaseID: "unreleased", CreatedAt: old}); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(OwnerStampPath(neighbor))
	if err != nil {
		t.Fatal(err)
	}
	var calls []string
	res := preparePreserving(context.Background(), f.repo, "cmd", "new-beside-clean", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, preservingTestRunner(t, &calls), func(context.Context) error { return nil })
	if !res.OK {
		t.Fatalf("prepare=%+v", res)
	}
	assertPreservingTreeLive(t, f.repo, neighbor)
	after, err := os.ReadFile(OwnerStampPath(neighbor))
	if err != nil || string(after) != string(before) {
		t.Fatal("clean neighbor metadata changed")
	}
}

func TestPreparePreservingRetainsWorkerOnMetadataFailure(t *testing.T) {
	f := newPreservingFixture(t)
	root := preservingTempDir(t)
	target := Path("cmd", "metadata-failure", root)
	var calls []string
	baseGit := preservingTestRunner(t, &calls)
	git := func(dir string, args []string) (int, string) {
		rc, out := baseGit(dir, args)
		if strings.Contains(strings.Join(args, " "), "worktree add") && rc == 0 {
			// A competing external writer makes owner-sidecar publication fail.
			if err := os.MkdirAll(OwnerStampPath(target), 0o755); err != nil {
				t.Fatal(err)
			}
		}
		return rc, out
	}
	res := preparePreserving(context.Background(), f.repo, "cmd", "metadata-failure", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, git, func(context.Context) error { return nil })
	if res.OK || !res.Preserved || res.Code != "PRESERVATION_TARGET_CONFLICT" {
		t.Fatalf("metadata refusal=%+v", res)
	}
	assertPreservingTreeLive(t, f.repo, target)
	if _, err := os.Stat(filepath.Join(target, WorkerLeaseFileName)); !os.IsNotExist(err) {
		t.Fatalf("worker lease after failed owner publication: %v", err)
	}
}

func TestPreparePreservingRefusesUnpinnedBaseAndMissingAdmission(t *testing.T) {
	f := newPreservingFixture(t)
	root := preservingTempDir(t)
	owner := OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}
	var calls []string
	git := preservingTestRunner(t, &calls)
	res := preparePreserving(context.Background(), f.repo, "cmd", "symbolic", "HEAD", root, owner, git, func(context.Context) error { return nil })
	if res.OK || res.Code != "PREPARE_NOT_READY" {
		t.Fatalf("symbolic base=%+v", res)
	}
	res = preparePreserving(context.Background(), f.repo, "cmd", "missing-gate", f.base, root, owner, git, nil)
	if res.OK || res.Code != "PRESERVATION_ADMISSION_REFUSED" {
		t.Fatalf("missing gate=%+v", res)
	}
}

func TestPreparePreservingDoesNotTreatWorkerCountAsAdmissionLimit(t *testing.T) {
	f := newPreservingFixture(t)
	root := preservingTempDir(t)
	var calls []string
	baseGit := preservingTestRunner(t, &calls)
	git := func(dir string, args []string) (int, string) {
		if strings.Join(args, " ") == "worktree list --porcelain" {
			rc, listing := baseGit(dir, args)
			for i := 0; i <= AdvisoryCapacitySetpoint; i++ {
				listing += fmt.Sprintf("\nworktree /kept/fak-worker-wt-cmd-%d\n", i)
			}
			return rc, listing
		}
		return baseGit(dir, args)
	}
	res := preparePreserving(context.Background(), f.repo, "cmd", "resource-admitted", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, git, func(context.Context) error { return nil })
	if !res.OK {
		t.Fatalf("advisory count became hard cap: %+v", res)
	}
}

func TestPreparePreservingPathsAndIntentCannotExpandAuthority(t *testing.T) {
	for _, p := range []string{"", " ", ".", "./", "..", "../cmd/fak", "/cmd/fak", "cmd/*", "cmd/../internal", "cmd\n/fak"} {
		if err := ValidatePreservingPaths([]string{p}); err == nil {
			t.Fatalf("accepted nonconcrete path %q", p)
		}
	}
	if err := ValidatePreservingPaths([]string{"cmd/fak/main.go"}); err != nil {
		t.Fatal(err)
	}
	in := PreservingIntent{Message: "test preservation intent", Paths: []string{"internal/other.go"}, AdmittedPaths: []string{"cmd/fak/main.go"}}
	if err := validatePreservingIntent([]PreservingIntent{in}); err == nil {
		t.Fatal("intent expanded authority")
	}
}

func TestPreparePreservingLeaseLossBeforeAndAfterMetadata(t *testing.T) {
	for _, lossAt := range []int{2, 3, 4} {
		t.Run(fmt.Sprint(lossAt), func(t *testing.T) {
			f := newPreservingFixture(t)
			root := preservingTempDir(t)
			var calls []string
			count := 0
			res := preparePreserving(context.Background(), f.repo, "cmd", "lease-loss", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, preservingTestRunner(t, &calls), func(context.Context) error {
				count++
				if count == lossAt {
					return fmt.Errorf("STALE_LEASE")
				}
				return nil
			})
			if res.OK || res.Code != "PRESERVATION_ADMISSION_REFUSED" {
				t.Fatalf("lease loss=%+v checks=%d", res, count)
			}
			if lossAt >= 3 {
				assertPreservingTreeLive(t, f.repo, res.Path)
			}
		})
	}
}

func TestPreparePreservingMetadataExclusiveAndTempSymlinks(t *testing.T) {
	for _, kind := range []string{"existing", "symlink", "temp-symlink"} {
		t.Run(kind, func(t *testing.T) {
			root := preservingTempDir(t)
			path := filepath.Join(root, "metadata.json")
			evidence := filepath.Join(root, "evidence")
			if err := os.WriteFile(evidence, []byte("untouched"), 0o600); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "existing":
				if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
					t.Fatal(err)
				}
			case "symlink":
				if err := os.Symlink(evidence, path); err != nil {
					t.Fatal(err)
				}
			case "temp-symlink":
				if err := os.Symlink(evidence, path+".tmp"); err != nil {
					t.Fatal(err)
				}
			}
			if err := writePreservingJSON(path, map[string]string{"new": "forbidden"}); err == nil {
				t.Fatal("metadata overwrite admitted")
			}
			data, _ := os.ReadFile(evidence)
			if string(data) != "untouched" {
				t.Fatal("symlink referent changed")
			}
			if kind == "existing" {
				data, _ := os.ReadFile(path)
				if string(data) != "original" {
					t.Fatal("existing metadata replaced")
				}
			}
		})
	}
	if err := checkPreservingAbsent("unreadable", func(string) (os.FileInfo, error) { return nil, os.ErrPermission }); err == nil {
		t.Fatal("access error became absence")
	}
}

type preservingFailWriter struct {
	file  *os.File
	phase string
}

func (w preservingFailWriter) Write(data []byte) (int, error) {
	if w.phase == "write" {
		n, _ := w.file.Write(data[:1])
		return n, fmt.Errorf("write failure")
	}
	return w.file.Write(data)
}
func (w preservingFailWriter) Sync() error {
	if w.phase == "sync" {
		return fmt.Errorf("sync failure")
	}
	return w.file.Sync()
}
func (w preservingFailWriter) Close() error {
	err := w.file.Close()
	if w.phase == "close" {
		return fmt.Errorf("close failure")
	}
	return err
}

func TestPreparePreservingWriteFailuresKeepEvidence(t *testing.T) {
	for _, phase := range []string{"write", "sync", "close"} {
		t.Run(phase, func(t *testing.T) {
			path := filepath.Join(preservingTempDir(t), "owner.json")
			err := writePreservingFileWith(path, []byte("attempt evidence"), func(path string) (io.WriteCloser, error) {
				f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
				if err != nil {
					return nil, err
				}
				return preservingFailWriter{file: f, phase: phase}, nil
			})
			if err == nil {
				t.Fatal("write failure became success")
			}
			if data, err := os.ReadFile(path); err != nil || len(data) == 0 {
				t.Fatalf("evidence removed: %v", err)
			}
		})
	}
}

func TestPreparePreservingIntentPublicationStaysInsideTargetLock(t *testing.T) {
	f := newPreservingFixture(t)
	root := preservingTempDir(t)
	target := Path("cmd", "intent", root)
	var calls []string
	baseGit := preservingTestRunner(t, &calls)
	blocked := false
	t.Setenv(PrepareLockWaitEnv, "1ms")
	git := func(dir string, args []string) (int, string) {
		if strings.Join(args, " ") == "worktree list --porcelain" {
			if _, err := os.Lstat(intentPath(target)); err == nil {
				if err := withPrepareTargetLock(target, time.Millisecond, func() { t.Fatal("intent escaped target lock") }); err == nil {
					t.Fatal("lock contention lost")
				} else {
					blocked = true
				}
			}
		}
		return baseGit(dir, args)
	}
	in := PreservingIntent{Message: "test immutable intent", Paths: []string{"target.txt"}, AdmittedPaths: []string{"target.txt"}}
	res := preparePreserving(context.Background(), f.repo, "cmd", "intent", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, git, func(context.Context) error { return nil }, in)
	if !res.OK || !blocked {
		t.Fatalf("intent=%+v blocked=%v", res, blocked)
	}
	before, err := os.ReadFile(intentPath(target))
	if err != nil {
		t.Fatal(err)
	}
	if err := writePreservingIntent(target, f.base, []PreservingIntent{in}); err == nil {
		t.Fatal("existing intent re-published")
	}
	after, _ := os.ReadFile(intentPath(target))
	if string(after) != string(before) {
		t.Fatal("intent replaced")
	}
}

func TestPreparePreservingRefusesHooksAndUnknownCheckoutConfig(t *testing.T) {
	f := newPreservingFixture(t)
	hook := filepath.Join(preservingTempDir(t), "post-checkout")
	if err := os.Symlink("missing-hook", hook); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"config", "config-access", "hook", "hook-access"} {
		t.Run(kind, func(t *testing.T) {
			var calls []string
			baseGit := preservingTestRunner(t, &calls)
			git := func(dir string, args []string) (int, string) {
				if args[0] == "config" {
					if kind == "config" {
						return 0, "filter.test.smudge unsupported"
					}
					if kind == "config-access" {
						return 2, "permission"
					}
				}
				if args[0] == "rev-parse" && len(args) == 3 && args[2] == "hooks/post-checkout" {
					if kind == "hook-access" {
						return 1, "unreadable"
					}
					return 0, hook + "\n"
				}
				return baseGit(dir, args)
			}
			if _, err := preservingCheckoutPolicy(f.repo, f.base, git); err == nil {
				t.Fatal("unqualified checkout admitted")
			}
		})
	}
}

func TestPreparePreservingRejectsTrackedLeaseMetadata(t *testing.T) {
	f := newPreservingFixture(t)
	root := preservingTempDir(t)
	var calls []string
	baseGit := preservingTestRunner(t, &calls)
	git := func(dir string, args []string) (int, string) {
		if args[0] == "ls-tree" {
			return 0, "lease.json\x00"
		}
		return baseGit(dir, args)
	}
	res := preparePreserving(context.Background(), f.repo, "cmd", "tracked-lease", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, git, func(context.Context) error { return nil })
	if res.OK || res.Code != "PRESERVATION_TARGET_CONFLICT" {
		t.Fatalf("tracked lease=%+v", res)
	}
	for _, c := range calls {
		if strings.Contains(c, "worktree add") {
			t.Fatal("checkout after reserved tracked metadata")
		}
	}
}

// Isolated repository setup deliberately does not call ordinary Prepare or
// touch sweep/backend globals. All inherited Git configuration is excluded.
func newPreservingFixture(t *testing.T) reapProofFixture {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("Git unavailable")
	}
	home := preservingTempDir(t)
	t.Setenv("FAK_PRESERVING_LIBRARY_FIXTURE_HOME", home)
	for _, key := range []string{"HOME", "USERPROFILE", "XDG_CONFIG_HOME"} {
		t.Setenv(key, home)
	}
	config := filepath.Join(preservingTempDir(t), "empty.gitconfig")
	if err := os.WriteFile(config, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GIT_CONFIG_COUNT", "0")
	t.Setenv("GIT_CONFIG_PARAMETERS", "")
	t.Setenv("GIT_CONFIG_GLOBAL", config)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
	t.Setenv("GIT_OPTIONAL_LOCKS", "0")
	repo := preservingTempDir(t)
	preservingFixtureGit(t, repo, "init", "-q", "-b", "main")
	preservingFixtureGit(t, repo, "config", "user.name", "preserving fixture")
	preservingFixtureGit(t, repo, "config", "user.email", "preserving@test")
	preservingFixtureGit(t, repo, "config", "commit.gpgsign", "false")
	for _, name := range []string{"target.txt", "peer.txt"} {
		if err := os.WriteFile(filepath.Join(repo, name), []byte("base\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	preservingFixtureGit(t, repo, "add", "target.txt", "peer.txt")
	preservingFixtureGit(t, repo, "commit", "-q", "-m", "fixture")
	return reapProofFixture{repo: repo, base: strings.TrimSpace(preservingFixtureGit(t, repo, "rev-parse", "HEAD"))}
}

// The fixture constructor installs this private identity with restored t.Setenv.
// Parent policy reads and child Git attribute lookup use exactly the same roots.
// The sanitizer still drops ambient HOME/XDG values rather than inheriting them.
func preservingFixtureIdentity(t *testing.T) string {
	t.Helper()
	home := os.Getenv("FAK_PRESERVING_LIBRARY_FIXTURE_HOME")
	if !filepath.IsAbs(home) {
		t.Fatal("explicit private fixture identity required")
	}
	for _, key := range []string{"HOME", "USERPROFILE", "XDG_CONFIG_HOME"} {
		if os.Getenv(key) != home {
			t.Fatalf("parent/child fixture identity mismatch for %s", key)
		}
	}
	return home
}

func preservingFixtureGit(t *testing.T, root string, args ...string) string {
	t.Helper()
	git, err := exec.LookPath("git")
	if err != nil {
		t.Fatal(err)
	}
	git, err = filepath.Abs(git)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, git, args...)
	cmd.Env = preservingLibraryFixtureEnv(os.Environ(), git, preservingFixtureIdentity(t))
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("fixture Git %v: %v %s", args, err, out)
	}
	return string(out)
}

func TestPreparePreservingCanonicalRelativeAndAliasRoots(t *testing.T) {
	f := newPreservingFixture(t)
	alias := filepath.Join(preservingTempDir(t), "repo-alias")
	if err := os.Symlink(f.repo, alias); err != nil {
		t.Fatal(err)
	}
	root, workers, err := CanonicalPreservingRoots(alias, "workers")
	if err != nil {
		t.Fatal(err)
	}
	physical, _ := filepath.EvalSymlinks(f.repo)
	if root != physical || workers != filepath.Join(physical, "workers") {
		t.Fatalf("identity=%s,%s", root, workers)
	}
	var calls []string
	res := preparePreserving(context.Background(), alias, "cmd", "relative", f.base, "workers", OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, preservingTestRunner(t, &calls), func(context.Context) error { return nil })
	if !res.OK || res.Path != Path("cmd", "relative", workers) {
		t.Fatalf("relative prepare=%+v", res)
	}
	assertPreservingTreeLive(t, f.repo, res.Path)
	var again []string
	retry := preparePreserving(context.Background(), f.repo, "cmd", "relative", f.base, workers, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, preservingTestRunner(t, &again), func(context.Context) error { return nil })
	if retry.OK || retry.Code != PrepareCodeOrphanTargetRefused {
		t.Fatalf("alias bypass=%+v", retry)
	}
}

func TestPreparePreservingRevocationDuringFinalRegistrationReadback(t *testing.T) {
	f := newPreservingFixture(t)
	root := preservingTempDir(t)
	target := Path("cmd", "final-revocation", root)
	var calls []string
	revoked := false
	baseGit := preservingTestRunner(t, &calls)
	git := func(dir string, args []string) (int, string) {
		rc, out := baseGit(dir, args)
		if strings.Join(args, " ") == "worktree list --porcelain" {
			if _, err := os.Lstat(OwnerStampPath(target)); err == nil {
				revoked = true
			}
		}
		return rc, out
	}
	res := preparePreserving(context.Background(), f.repo, "cmd", "final-revocation", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "admitted"}, git, func(context.Context) error {
		if revoked {
			return fmt.Errorf("STALE_LEASE")
		}
		return nil
	})
	if !revoked || res.OK || res.Code != "PRESERVATION_ADMISSION_REFUSED" {
		t.Fatalf("publication=%+v revoked=%v", res, revoked)
	}
	assertPreservingTreeLive(t, f.repo, target)
}

func preservingTempDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestPreparePreservingRejectsFenceNormalizedWhitespace(t *testing.T) {
	for _, p := range []string{" target.txt", "target.txt ", "\ttarget.txt", "target.txt\t", "\u00a0target.txt", "target.txt\u00a0"} {
		if normalizeFencePath(p) == p {
			t.Fatalf("fixture did not change under fence: %q", p)
		}
		if err := ValidatePreservingPaths([]string{p}); err == nil {
			t.Fatalf("accepted fenced/stored pathname mismatch %q", p)
		}
		in := PreservingIntent{Message: "immutable fixture", Paths: []string{p}, AdmittedPaths: []string{"target.txt"}}
		if err := validatePreservingIntent([]PreservingIntent{in}); err == nil {
			t.Fatalf("intent accepted normalized path %q", p)
		}
	}
	if err := ValidatePreservingPaths([]string{"directory/inner space.txt"}); err != nil {
		t.Fatalf("unchanged concrete filename refused: %v", err)
	}
}

func preservingLibraryFixtureEnv(base []string, git, template string) []string {
	var env []string
	for _, entry := range base {
		key, _, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		switch strings.ToUpper(key) {
		case "SYSTEMROOT", "WINDIR", "TMPDIR", "TMP", "TEMP":
			env = append(env, entry)
		}
	}
	path := filepath.Dir(git) + string(os.PathListSeparator) + "/usr/bin" + string(os.PathListSeparator) + "/bin"
	env = append(env, "PATH="+path, "HOME="+template, "USERPROFILE="+template, "XDG_CONFIG_HOME="+template, "GOMAXPROCS=2", "GIT_CONFIG_GLOBAL="+os.DevNull, "GIT_CONFIG_SYSTEM="+os.DevNull, "GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_COUNT=0", "GIT_TEMPLATE_DIR="+template, "GIT_TERMINAL_PROMPT=0", "GIT_OPTIONAL_LOCKS=0", "GIT_ALLOW_PROTOCOL=file")
	return env
}
func assertPreservingTreeLive(t *testing.T, root, target string) {
	t.Helper()
	if _, err := os.Lstat(filepath.Join(target, ".git")); err != nil {
		t.Fatal(err)
	}
	listing := preservingFixtureGit(t, root, "worktree", "list", "--porcelain")
	for _, entry := range parseWorktreeListing(listing) {
		if samePath(entry.Path, target) && !entry.Prunable {
			return
		}
	}
	t.Fatalf("fixture target not registered: %s", target)
}

func TestPreparePreservingFixtureEnvironmentConfinement(t *testing.T) {
	git := filepath.Join(t.TempDir(), "git")
	template := t.TempDir()
	hostile := []string{"GIT_CONFIG", "GIT_TEMPLATE_DIR", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_EXEC_PATH", "GIT_NAMESPACE", "GIT_CEILING_DIRECTORIES", "GIT_DISCOVERY_ACROSS_FILESYSTEM", "GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS", "GIT_TRACE", "GIT_TRACE2_EVENT", "GIT_TRACE_PACK_ACCESS", "GIT_SSH", "GIT_SSH_COMMAND", "GIT_SSH_VARIANT", "GIT_ASKPASS", "SSH_ASKPASS", "SSH_AUTH_SOCK", "SSH_AGENT_PID", "FAK_WORKSPACE_ROOT", "FAK_LEASE_ID", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "HOME", "USERPROFILE", "XDG_CONFIG_HOME", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "BASH_ENV", "ENV", "FAK_PRESERVING_NATIVE_HELPER_CONFIG", "FAK_PRESERVING_CLI_FIXTURE", "PATH"}
	var base []string
	for _, key := range hostile {
		base = append(base, key+"=ambient-poison")
	}
	base = append(base, "TMPDIR="+template)
	got := map[string]string{}
	for _, entry := range preservingLibraryFixtureEnv(base, git, template) {
		key, value, ok := strings.Cut(entry, "=")
		if !ok {
			t.Fatal("malformed fixture environment")
		}
		if _, exists := got[key]; exists {
			t.Fatalf("duplicate environment key %s", key)
		}
		got[key] = value
	}
	for _, key := range hostile {
		if got[key] == "ambient-poison" {
			t.Fatalf("inherited injection %s", key)
		}
	}
	for key, value := range map[string]string{"GIT_CONFIG_GLOBAL": os.DevNull, "GIT_CONFIG_SYSTEM": os.DevNull, "GIT_CONFIG_COUNT": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_TEMPLATE_DIR": template, "GIT_ALLOW_PROTOCOL": "file", "HOME": template, "TMPDIR": template} {
		if got[key] != value {
			t.Fatalf("%s=%q, want %q", key, got[key], value)
		}
	}
	if !strings.HasPrefix(got["PATH"], filepath.Dir(git)+string(os.PathListSeparator)) {
		t.Fatal("unbound Git path")
	}
}

func preservingLibraryHostileEnvironment(t *testing.T) (string, string) {
	t.Helper()
	root := t.TempDir()
	template := filepath.Join(root, "template")
	hooks := filepath.Join(template, "hooks")
	if err := os.MkdirAll(hooks, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(hooks, "post-checkout"), []byte("nonexecutable hostile-template sentinel"), 0600); err != nil {
		t.Fatal(err)
	}
	config := filepath.Join(root, "redirected-config")
	before := "[core]\n hooksPath = " + hooks + "\n"
	if err := os.WriteFile(config, []byte(before), 0600); err != nil {
		t.Fatal(err)
	}
	for key, value := range map[string]string{"GIT_CONFIG": config, "GIT_TEMPLATE_DIR": template, "GIT_CONFIG_PARAMETERS": "invalid hostile config parameters", "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.hooksPath", "GIT_CONFIG_VALUE_0": hooks, "GIT_DIR": filepath.Join(root, "absent-git"), "GIT_COMMON_DIR": filepath.Join(root, "absent-common"), "GIT_WORK_TREE": root, "GIT_INDEX_FILE": filepath.Join(root, "redirected-index"), "GIT_OBJECT_DIRECTORY": filepath.Join(root, "redirected-objects"), "GIT_ALTERNATE_OBJECT_DIRECTORIES": filepath.Join(root, "absent-alternates"), "GIT_EXEC_PATH": root, "GIT_SSH_COMMAND": "nonexistent-fixture-ssh", "GIT_SSH": filepath.Join(root, "absent-ssh"), "GIT_ASKPASS": filepath.Join(root, "absent-askpass"), "SSH_AUTH_SOCK": filepath.Join(root, "absent-agent"), "GIT_TRACE": filepath.Join(root, "redirected-trace"), "FAK_WORKSPACE_ROOT": root} {
		t.Setenv(key, value)
	}
	t.Cleanup(func() {
		for _, name := range []string{"redirected-index", "redirected-objects", "redirected-trace"} {
			if _, err := os.Lstat(filepath.Join(root, name)); !os.IsNotExist(err) {
				t.Errorf("ambient redirect written: %s (%v)", name, err)
			}
		}
	})
	return config, before
}

func TestPreparePreservingFixtureHostileGitEnvironment(t *testing.T) {
	config, before := preservingLibraryHostileEnvironment(t)
	f := newPreservingFixture(t)
	var calls []string
	res := preparePreserving(context.Background(), f.repo, "cmd", "hostile-environment", f.base, preservingTempDir(t), OwnerStamp{PID: os.Getpid(), LeaseID: "fixture-admitted"}, preservingTestRunner(t, &calls), func(context.Context) error { return nil })
	if !res.OK {
		t.Fatalf("confined prepare=%+v", res)
	}
	assertPreservingTreeLive(t, f.repo, res.Path)
	if data, err := os.ReadFile(config); err != nil || string(data) != before {
		t.Fatalf("ambient config changed: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(f.repo, ".git", "hooks", "post-checkout")); !os.IsNotExist(err) {
		t.Fatal("ambient template imported")
	}
}
