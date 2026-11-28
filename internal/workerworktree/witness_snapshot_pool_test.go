package workerworktree

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func witnessPoolFixture(t *testing.T) (string, string, string) {
	t.Helper()
	root, parent := newSiblingTopologyRepo(t)
	if err := os.Mkdir(filepath.Join(parent, "sibling"), 0755); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{".gitignore": "ignored\n", "value": "parent\n", "go.mod": "module fixture\n\ngo 1.22\n"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}
	mustGit(t, root, "add", ".")
	mustGit(t, root, "commit", "-qm", "parent")
	before := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))
	if err := os.WriteFile(filepath.Join(root, "value"), []byte("candidate\n"), 0644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, root, "add", ".")
	mustGit(t, root, "commit", "-qm", "candidate")
	return root, before, strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))
}

func assertWitnessSnapshot(t *testing.T, dir, ref, content string) {
	t.Helper()
	if got := strings.TrimSpace(mustGit(t, dir, "rev-parse", "HEAD")); got != ref {
		t.Fatalf("HEAD=%s want=%s", got, ref)
	}
	if got := mustGit(t, dir, "status", "--porcelain", "--ignored", "--untracked-files=all"); strings.TrimSpace(got) != "" {
		t.Fatalf("dirty snapshot: %s", got)
	}
	if rc, _ := defaultGit(dir, []string{"symbolic-ref", "-q", "HEAD"}); rc != 1 {
		t.Fatalf("snapshot HEAD is attached: exit=%d", rc)
	}
	got, err := os.ReadFile(filepath.Join(dir, "value"))
	if err != nil || string(got) != content {
		t.Fatalf("source=%q err=%v want=%q", got, err, content)
	}
}

// fak-test:runtime slow est=5s lane=default
// fak-test:justify why=regression when=changed:internal/workerworktree/**
func TestWitnessSnapshotPoolReuseCleansAndPreservesTopology(t *testing.T) {
	root, parent, candidate := witnessPoolFixture(t)
	var events []WitnessSnapshotEvent
	ctx := WithWitnessSnapshotObserver(context.Background(), func(e WitnessSnapshotEvent) { events = append(events, e) })
	dir, release, err := AcquireWitnessSnapshot(ctx, root, candidate)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(release)
	if filepath.Dir(dir) != filepath.Dir(root) {
		t.Fatalf("sibling topology lost: root=%s snapshot=%s", root, dir)
	}
	if _, err := os.Stat(filepath.Join(dir, "..", "sibling")); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"value", "untracked", "ignored"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("residue"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	release()
	release()
	warm, releaseWarm, err := AcquireWitnessSnapshot(ctx, root, parent)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(releaseWarm)
	if warm != dir {
		t.Fatalf("warm acquisition missed retained checkout: %s != %s", warm, dir)
	}
	assertWitnessSnapshot(t, warm, parent, "parent\n")
	releaseWarm()
	if len(events) != 4 || events[0].Reused || !events[2].Reused || events[0].Phase != "prepare" || events[1].Phase != "release" || events[2].Ref != parent || events[3].Phase != "release" {
		t.Fatalf("events=%+v", events)
	}
	for _, e := range events {
		raw, err := json.Marshal(e)
		if err != nil || e.Error != "" || e.Duration < 0 || !strings.Contains(string(raw), `"duration_ns":`) {
			t.Fatalf("event=%+v json=%s err=%v", e, raw, err)
		}
	}
}

// fak-test:runtime slow est=6s lane=default
// fak-test:justify why=regression when=changed:internal/workerworktree/**
func TestWitnessSnapshotPoolActiveSourcesDisjointAndThirdTransient(t *testing.T) {
	root, parent, candidate := witnessPoolFixture(t)
	type acquired struct {
		dir     string
		release func()
		err     error
	}
	ready := make(chan acquired, 2)
	first, releaseFirst, err := AcquireWitnessSnapshot(context.Background(), root, parent)
	if err != nil {
		t.Fatal(err)
	}
	ready <- acquired{first, releaseFirst, nil}
	go func() {
		dir, release, err := AcquireWitnessSnapshot(context.Background(), root, candidate)
		ready <- acquired{dir, release, err}
	}()
	var dirs []string
	for range 2 {
		got := <-ready
		if got.err != nil {
			t.Fatal(got.err)
		}
		t.Cleanup(got.release)
		dirs = append(dirs, got.dir)
	}
	if dirs[0] == dirs[1] {
		t.Fatal("active leases share a directory")
	}
	var thirdEvent WitnessSnapshotEvent
	ctx := WithWitnessSnapshotObserver(context.Background(), func(e WitnessSnapshotEvent) {
		if e.Phase == "prepare" {
			thirdEvent = e
		}
	})
	third, release, err := AcquireWitnessSnapshot(ctx, root, candidate)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(release)
	if third == dirs[0] || third == dirs[1] || thirdEvent.Reused || !strings.HasPrefix(filepath.Base(third), ".fak-witness-once-") {
		t.Fatalf("third=%s active=%v event=%+v", third, dirs, thirdEvent)
	}
	assertWitnessSnapshot(t, third, candidate, "candidate\n")
	seen := map[string]bool{}
	for _, dir := range dirs {
		ref := strings.TrimSpace(mustGit(t, dir, "rev-parse", "HEAD"))
		seen[ref] = true
		content := "parent\n"
		if ref == candidate {
			content = "candidate\n"
		}
		assertWitnessSnapshot(t, dir, ref, content)
	}
	if !seen[parent] || !seen[candidate] {
		t.Fatalf("active source pins=%v", seen)
	}
	release()
	if _, err := os.Stat(third); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("transient survived release: %v", err)
	}
}

// fak-test:runtime slow est=4s lane=default
// fak-test:justify why=regression when=changed:internal/workerworktree/**
func TestWitnessSnapshotPoolCancellationAndFailedResetQuarantine(t *testing.T) {
	root, parent, candidate := witnessPoolFixture(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	dir, release, err := AcquireWitnessSnapshot(ctx, root, candidate)
	if !errors.Is(err, context.Canceled) || dir != "" || release != nil {
		t.Fatalf("canceled lease dir=%q releaseNil=%v err=%v", dir, release == nil, err)
	}
	dir, release, err = AcquireWitnessSnapshot(context.Background(), root, candidate)
	if err != nil {
		t.Fatal(err)
	}
	release()
	failGit := func(root string, args []string) (int, string) {
		if len(args) > 0 && (args[0] == "checkout" || args[0] == "reset") {
			if _, err := os.Stat(witnessSnapshotUnavailablePath(dir)); err != nil {
				t.Fatalf("source mutation preceded durable exclusion: %v", err)
			}
			return 1, "injected reset failure"
		}
		return defaultGit(root, args)
	}
	s := newTopologyCandidateSession(root, failGit)
	s.poolPrefix = witnessSnapshotPoolPrefix
	if ok, _ := s.create(parent); ok || s.failed == "" {
		t.Fatal("failed reset accepted or silently replaced")
	}
	if err := s.discardWitnessSnapshot(); err != nil {
		if _, markerErr := os.Stat(witnessSnapshotUnavailablePath(dir)); markerErr != nil {
			t.Fatalf("cleanup failed without durable exclusion: cleanup=%v marker=%v", err, markerErr)
		}
	}
	if _, err := os.Stat(witnessSnapshotUnavailablePath(dir)); err != nil {
		t.Fatalf("failed preparation lost durable exclusion: %v", err)
	}
	if _, err := os.Stat(dir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed snapshot remains reusable: %s err=%v", dir, err)
	}
	fresh, closeFresh, err := AcquireWitnessSnapshot(context.Background(), root, parent)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(closeFresh)
	if fresh == dir {
		t.Fatal("quarantined path reused")
	}
	assertWitnessSnapshot(t, fresh, parent, "parent\n")
}

// fak-test:runtime slow est=3s lane=default
// fak-test:justify why=regression when=changed:internal/workerworktree/**
func TestWitnessSnapshotPoolPriorExclusionNeverMutatesCheckout(t *testing.T) {
	root, parent, candidate := witnessPoolFixture(t)
	dir, release, err := AcquireWitnessSnapshot(context.Background(), root, candidate)
	if err != nil {
		t.Fatal(err)
	}
	release()
	if err := beginWitnessSnapshotMutation(dir); err != nil {
		t.Fatal(err)
	}
	if err := beginWitnessSnapshotMutation(dir); !errors.Is(err, os.ErrExist) {
		t.Fatalf("existing exclusion overwritten: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "value"), []byte("quarantined evidence\n"), 0644); err != nil {
		t.Fatal(err)
	}
	fresh, closeFresh, err := AcquireWitnessSnapshot(context.Background(), root, parent)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(closeFresh)
	if fresh == dir {
		t.Fatal("prior exclusion reused")
	}
	assertWitnessSnapshot(t, fresh, parent, "parent\n")
	got, err := os.ReadFile(filepath.Join(dir, "value"))
	if err != nil || string(got) != "quarantined evidence\n" {
		t.Fatalf("excluded checkout mutated: %q err=%v", got, err)
	}
}

// fak-test:runtime slow est=12s lane=default
// fak-test:justify why=regression when=changed:internal/workerworktree/**
func TestWitnessSnapshotPoolManagedWorkspaceUsesCommittedCanonicalTopology(t *testing.T) {
	const committedWorkspace = "go 1.22\n\nuse (\n .\n ../fak\n)\n"
	for _, tc := range []struct {
		name, working                       string
		nested, absent, canonical, relative bool
	}{
		{"generated-relative-workspace", "go 1.22\n\nuse (\n .\n ../../../fak\n)\n", false, false, false, false},
		{"nested-local-only-workspace", "go 1.22\n\nuse .\n", true, false, false, false},
		{"absent-working-workspace", "", false, true, false, false},
		{"relative-canonical-root", committedWorkspace, false, false, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := t.TempDir()
			root := filepath.Join(base, "fak-private")
			public := filepath.Join(base, "fak")
			for _, dir := range []string{root, public} {
				if err := os.Mkdir(dir, 0755); err != nil {
					t.Fatal(err)
				}
			}
			write := func(dir, name, contents string) {
				t.Helper()
				if err := os.WriteFile(filepath.Join(dir, name), []byte(contents), 0644); err != nil {
					t.Fatal(err)
				}
			}
			write(public, "go.mod", "module fixture.public\n\ngo 1.22\n")
			write(root, "go.mod", "module fixture.private\n\ngo 1.22\n")
			write(root, "go.work", committedWorkspace)
			write(root, ".gitignore", ".worktrees/\n")
			mustGit(t, root, "init", "-q")
			mustGit(t, root, "config", "user.email", "fixture@example.invalid")
			mustGit(t, root, "config", "user.name", "fixture")
			mustGit(t, root, "add", ".")
			mustGit(t, root, "commit", "-qm", "committed sibling workspace")
			ref := strings.TrimSpace(mustGit(t, root, "rev-parse", "HEAD"))
			managed := root
			if !tc.canonical {
				managed = filepath.Join(root, ".worktrees", "ticket-parent")
				mustGit(t, root, "worktree", "add", "--detach", managed, ref)
			}
			if tc.nested {
				child := filepath.Join(managed, ".worktrees", "ticket-child")
				mustGit(t, managed, "worktree", "add", "--detach", child, ref)
				managed = child
			}
			if tc.absent {
				if err := os.Remove(filepath.Join(managed, "go.work")); err != nil {
					t.Fatal(err)
				}
			} else {
				write(managed, "go.work", tc.working)
			}
			callerStatus := mustGit(t, managed, "status", "--porcelain")
			acquireRoot := managed
			if tc.relative {
				cwd, err := os.Getwd()
				if err != nil {
					t.Fatal(err)
				}
				acquireRoot, err = filepath.Rel(cwd, managed)
				if err != nil {
					t.Fatal(err)
				}
			}
			dir, release, err := AcquireWitnessSnapshot(context.Background(), acquireRoot, ref)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				release()
				cmd := exec.Command("git", "worktree", "remove", "--force", dir)
				cmd.Dir = root
				if out, err := cmd.CombinedOutput(); err != nil {
					t.Errorf("remove fixture snapshot: %v: %s", err, out)
				}
			})
			if filepath.Clean(filepath.Dir(dir)) != filepath.Clean(base) {
				t.Errorf("snapshot placed beside managed checkout instead of canonical repository: got=%s want-parent=%s", dir, base)
			}
			workspace, err := os.ReadFile(filepath.Join(dir, "go.work"))
			if err != nil || string(workspace) != committedWorkspace {
				t.Fatalf("snapshot must retain exact committed workspace: %q err=%v", workspace, err)
			}
			if got := strings.TrimSpace(mustGit(t, dir, "rev-parse", "HEAD")); got != ref {
				t.Fatalf("snapshot HEAD=%s want=%s", got, ref)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, "go", "list", "-m")
			cmd.Dir = dir
			cmd.Env = append(os.Environ(), "GOWORK="+filepath.Join(dir, "go.work"), "GOTOOLCHAIN=local")
			out, listErr := cmd.CombinedOutput()
			if listErr != nil || !strings.Contains(string(out), "fixture.private") || !strings.Contains(string(out), "fixture.public") {
				t.Errorf("committed snapshot workspace must resolve canonical sibling modules: %v: %s", listErr, out)
			}
			if got := mustGit(t, managed, "status", "--porcelain"); got != callerStatus {
				t.Errorf("caller changed: got=%q want=%q", got, callerStatus)
			}
			callerWorkspace, readErr := os.ReadFile(filepath.Join(managed, "go.work"))
			if tc.absent {
				if !errors.Is(readErr, os.ErrNotExist) {
					t.Errorf("absent caller workspace was replaced: %v", readErr)
				}
			} else if readErr != nil || string(callerWorkspace) != tc.working {
				t.Errorf("generated caller workspace changed: %q err=%v", callerWorkspace, readErr)
			}
		})
	}
}

// fak-test:runtime slow est=3s lane=default
// fak-test:justify why=regression when=changed:internal/workerworktree/**
func TestWitnessSnapshotSweepReapsDeadOwnerOnceCandidateAndLockedRegistration(t *testing.T) {
	root, parent, _ := witnessPoolFixture(t)
	parentDir := filepath.Dir(root)
	leaked := filepath.Join(parentDir, ".fak-witness-once-999999-mwszod1x-123456")
	mustGit(t, root, "worktree", "add", "--detach", leaked, parent)
	admin := candidateAdminDir(leaked)
	if admin == "" {
		t.Fatalf("expected git admin dir for %s", leaked)
	}
	if err := os.WriteFile(filepath.Join(admin, "locked"), []byte("initializing\n"), 0644); err != nil {
		t.Fatal(err)
	}
	report := SweepTopologyCandidates(root, parentDir, defaultGit, CandidateSweepOptions{
		Now:          time.Now(),
		Apply:        true,
		ProcessAlive: func(int) bool { return false },
	})
	if report.Reaped < 1 {
		t.Fatalf("expected dead-owner .fak-witness-once-* to be reaped, got report=%+v", report)
	}
	if _, err := os.Stat(leaked); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("leaked .fak-witness-once-* dir survived sweep: %v", err)
	}
	if _, err := os.Stat(admin); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("locked admin registration survived sweep/prune: %s err=%v", admin, err)
	}
	if list := mustGit(t, root, "worktree", "list", "--porcelain"); strings.Contains(list, filepath.Base(leaked)) {
		t.Fatalf("leaked worktree still registered after sweep: %s", list)
	}
}
