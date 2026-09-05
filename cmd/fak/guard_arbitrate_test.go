package main

import (
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

func guardArbitrateTestRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	cmd := exec.Command("git", "init", "--quiet")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(`[lanes]
exclusive = ["exclusive"]
[lanes.trees]
cmd = ["cmd/**"]
docs = ["docs/**"]
exclusive = ["internal/**"]
`), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

func guardArbitrateSeedLease(t *testing.T, root, id string, tree []string) {
	t.Helper()
	store := leaseref.NewInDir(root)
	_, err := store.Acquire(context.Background(), leaseref.Record{
		ID: id, Holder: "peer", TreeGlobs: tree, AcquiredAt: time.Now().Unix(), TTLSeconds: 60,
	})
	if err != nil {
		t.Fatalf("seed lease: %v", err)
	}
}

func TestGuardArbitrateEnforceRefusesOverlap(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-cmd", []string{"cmd/**"})
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce, Root: root, Tree: []string{"cmd/fak/**"},
	})
	if lease != nil {
		lease.Close()
		t.Fatal("overlap unexpectedly acquired a lease")
	}
	if err == nil || !strings.Contains(err.Error(), "COLLISION_RISK") || !strings.Contains(err.Error(), "peer-cmd") {
		t.Fatalf("overlap error = %v, want COLLISION_RISK naming peer-cmd", err)
	}
}

func TestGuardArbitrateDisjointPublishesAndReleases(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-docs", []string{"docs/**"})
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce, Root: root, Tree: []string{"cmd/**"},
	})
	if err != nil {
		t.Fatalf("disjoint acquire: %v", err)
	}
	if lease == nil {
		t.Fatal("disjoint acquire returned no lease")
	}
	id := lease.record.ID
	if _, ok, err := lease.store.Get(context.Background(), id); err != nil || !ok {
		t.Fatalf("published lease read-back: ok=%v err=%v", ok, err)
	}
	lease.Close()
	if _, ok, err := lease.store.Get(context.Background(), id); err != nil || ok {
		t.Fatalf("released lease read-back: ok=%v err=%v", ok, err)
	}
}

func TestGuardArbitrateShadowLogsWithoutPublishing(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-cmd", []string{"cmd/**"})
	var stderr strings.Builder
	lease, err := guardArbitrateAcquire(context.Background(), &stderr, guardArbitrateConfig{
		Mode: guardArbitrateModeShadow, Root: root, Tree: []string{"cmd/**"}, ShowShadowNotice: true,
	})
	if err != nil || lease != nil {
		t.Fatalf("shadow = lease %v err %v", lease, err)
	}
	if got := stderr.String(); !strings.Contains(got, "shadow would refuse") || !strings.Contains(got, "peer-cmd") {
		t.Fatalf("shadow log = %q", got)
	}
}

func TestGuardArbitrateCompactStartupSuppressesShadowCollision(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-cmd", []string{"cmd/**"})
	var stderr strings.Builder
	lease, err := guardArbitrateAcquire(context.Background(), &stderr, guardArbitrateConfig{
		Mode: guardArbitrateModeShadow, Root: root, Tree: []string{"cmd/**"},
	})
	if err != nil || lease != nil {
		t.Fatalf("shadow = lease %v err %v", lease, err)
	}
	if got := stderr.String(); got != "" {
		t.Fatalf("shadow collision polluted compact startup: %q", got)
	}
}

func TestGuardArbitrateExpiredReadIsBoundedByMode(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	originalLive := guardArbitrateLive
	originalLimit := guardArbitrateShadowLimit
	guardArbitrateShadowLimit = 20 * time.Millisecond
	guardArbitrateLive = func(_ *leaseref.Store, ctx context.Context, _ time.Time) ([]leaseref.Record, []string, error) {
		<-ctx.Done() // Model the serial cat-file reader staying blocked until its budget expires.
		return nil, nil, ctx.Err()
	}
	t.Cleanup(func() {
		guardArbitrateLive = originalLive
		guardArbitrateShadowLimit = originalLimit
	})

	t.Run("shadow continues", func(t *testing.T) {
		var stderr strings.Builder
		started := time.Now()
		lease, err := guardArbitrateAcquire(context.Background(), &stderr, guardArbitrateConfig{
			Mode: guardArbitrateModeShadow, Root: root, Tree: []string{"cmd/**"},
		})
		if elapsed := time.Since(started); elapsed < guardArbitrateShadowLimit || elapsed > 250*time.Millisecond {
			t.Fatalf("blocked shadow admission took %s, want deadline-driven continuation near %s", elapsed, guardArbitrateShadowLimit)
		}
		if err != nil || lease != nil {
			t.Fatalf("shadow = lease %v err %v", lease, err)
		}
		if got := stderr.String(); !strings.Contains(got, "shadow mode continuing without a lease") {
			t.Fatalf("shadow timeout diagnostic = %q", got)
		}
	})

	t.Run("enforce refuses", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		lease, err := guardArbitrateAcquire(ctx, io.Discard, guardArbitrateConfig{
			Mode: guardArbitrateModeEnforce, Root: root, Tree: []string{"cmd/**"},
		})
		if lease != nil {
			lease.Close()
			t.Fatal("enforce returned a lease after its ledger read was cancelled")
		}
		if err == nil || !strings.Contains(err.Error(), "COLLISION_RISK") || !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("enforce error = %v, want fail-closed COLLISION_RISK wrapping context deadline", err)
		}
	})
}

func TestGuardArbitrateForceBypassesNonExclusiveOverlap(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-cmd", []string{"cmd/**"})
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce, Root: root, Tree: []string{"cmd/fak/**"}, Force: true,
	})
	if err != nil {
		t.Fatalf("force non-exclusive acquire: %v", err)
	}
	if lease == nil {
		t.Fatal("force non-exclusive acquire returned no lease")
	}
	lease.Close()
}

func TestGuardArbitrateForceStillHonorsExclusive(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	guardArbitrateSeedLease(t, root, "peer-exclusive", []string{"internal/**"})
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce, Root: root, Tree: []string{"docs/**"}, Force: true,
	})
	if lease != nil {
		lease.Close()
		t.Fatal("force bypassed live exclusive lease")
	}
	if err == nil || !strings.Contains(err.Error(), "peer-exclusive") {
		t.Fatalf("exclusive force error = %v", err)
	}
}

func TestGuardArbitrateUnreadableWorkspaceFailsOpen(t *testing.T) {
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce, Root: filepath.Join(t.TempDir(), "missing"), Tree: []string{"cmd/**"},
	})
	if err != nil || lease != nil {
		t.Fatalf("fail-open = lease %v err %v", lease, err)
	}
}

func TestGuardArbitrateFlagValueParsesLeaseProfile(t *testing.T) {
	cfg := guardArbitrateConfig{Mode: guardArbitrateModeShadow}
	value := guardArbitrateFlagValue{cfg: &cfg}
	if err := value.Set("mode=enforce,lane=gateway,tree=internal/gateway/**,tree=cmd/fak/**,force=true"); err != nil {
		t.Fatal(err)
	}
	if cfg.Mode != guardArbitrateModeEnforce || cfg.Lane != "gateway" || !cfg.Force {
		t.Fatalf("config = %+v", cfg)
	}
	if got, want := cfg.Tree, []string{"internal/gateway/**", "cmd/fak/**"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("tree = %v, want %v", got, want)
	}
}

func TestGuardArbitrateFlagValueRejectsUnknownField(t *testing.T) {
	cfg := guardArbitrateConfig{Mode: guardArbitrateModeShadow}
	if err := (guardArbitrateFlagValue{cfg: &cfg}).Set("branch=main"); err == nil || !strings.Contains(err.Error(), "unknown lease field") {
		t.Fatalf("error = %v", err)
	}
}

func TestGuardArbitrateDetachedWorktreeEnforce(t *testing.T) {
	root := guardArbitrateTestRepo(t)

	runGit := func(dir string, args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}

	runGit(root, "add", "dos.toml")
	runGit(root, "-c", "user.name=Tester", "-c", "user.email=tester@example.com", "commit", "-m", "initial")

	wtPath := filepath.Join(t.TempDir(), "wt")
	runGit(root, "worktree", "add", "--detach", wtPath, "HEAD")

	gitMarker := filepath.Join(wtPath, ".git")
	fi, err := os.Stat(gitMarker)
	if err != nil {
		t.Fatalf("stat worktree .git: %v", err)
	}
	if fi.IsDir() {
		t.Fatalf("worktree .git is a directory, want file")
	}

	// 1. Call guardArbitrateAcquire in guardArbitrateModeEnforce from wtPath.
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce,
		Root: wtPath,
		Tree: []string{"cmd/**"},
	})
	if err != nil {
		t.Fatalf("worktree acquire: %v", err)
	}
	if lease == nil {
		t.Fatal("worktree acquire returned nil lease")
	}

	// Verify lease was published and retrievable.
	leaseID := lease.record.ID
	if _, ok, err := lease.store.Get(context.Background(), leaseID); err != nil || !ok {
		t.Fatalf("worktree lease read-back: ok=%v err=%v", ok, err)
	}

	// 2. Verify concurrent overlapping acquire from primary repo is refused with COLLISION_RISK.
	rootOverlap, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce,
		Root: root,
		Tree: []string{"cmd/fak/**"},
	})
	if rootOverlap != nil {
		rootOverlap.Close()
		t.Fatal("overlapping acquire from root unexpectedly succeeded")
	}
	if err == nil || !strings.Contains(err.Error(), "COLLISION_RISK") {
		t.Fatalf("overlapping acquire from root error = %v, want COLLISION_RISK", err)
	}

	// 3. Verify concurrent overlapping acquire from worktree is refused with COLLISION_RISK.
	wtOverlap, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce,
		Root: wtPath,
		Tree: []string{"cmd/**"},
	})
	if wtOverlap != nil {
		wtOverlap.Close()
		t.Fatal("overlapping acquire from worktree unexpectedly succeeded")
	}
	if err == nil || !strings.Contains(err.Error(), "COLLISION_RISK") {
		t.Fatalf("overlapping acquire from worktree error = %v, want COLLISION_RISK", err)
	}

	// 4. Close the lease and verify it releases.
	lease.Close()
	if _, ok, err := lease.store.Get(context.Background(), leaseID); err != nil || ok {
		t.Fatalf("released worktree lease read-back: ok=%v err=%v", ok, err)
	}

	// Verify acquiring after release succeeds.
	afterRelease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce,
		Root: wtPath,
		Tree: []string{"cmd/**"},
	})
	if err != nil {
		t.Fatalf("acquire after release failed: %v", err)
	}
	if afterRelease == nil {
		t.Fatal("acquire after release returned nil lease")
	}
	afterRelease.Close()

	// 5. Also verify non-busy lock failure in enforce mode fails closed with COLLISION_RISK.
	commonDir, err := resolveGitCommonDir(wtPath)
	if err != nil {
		t.Fatalf("resolveGitCommonDir: %v", err)
	}
	lockPath := filepath.Join(commonDir, "fak-guard-arbitrate.lock")
	_ = os.RemoveAll(lockPath)
	if err := os.Mkdir(lockPath, 0o755); err != nil {
		t.Fatalf("mkdir lockPath: %v", err)
	}
	t.Cleanup(func() {
		_ = os.RemoveAll(lockPath)
	})

	failLease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce,
		Root: wtPath,
		Tree: []string{"cmd/**"},
	})
	if failLease != nil {
		failLease.Close()
		t.Fatal("acquire unexpectedly succeeded when lock path was a directory")
	}
	if err == nil || !strings.Contains(err.Error(), "COLLISION_RISK: guard admission serialization is unavailable") {
		t.Fatalf("non-busy lock failure error = %v, want COLLISION_RISK: guard admission serialization is unavailable", err)
	}
}

func TestGuardArbitrateResolveGitCommonDir(t *testing.T) {
	root := guardArbitrateTestRepo(t)

	// Primary repo directory.
	dir, err := resolveGitCommonDir(root)
	if err != nil {
		t.Fatalf("resolveGitCommonDir(root): %v", err)
	}
	if want := filepath.Clean(filepath.Join(root, ".git")); dir != want {
		t.Fatalf("dir = %q, want %q", dir, want)
	}

	// Missing commondir falls back to gitDir.
	tempDir := t.TempDir()
	fakeGitDir := filepath.Join(tempDir, "fake-git")
	if err := os.Mkdir(fakeGitDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, ".git"), []byte("gitdir: "+fakeGitDir+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, err := resolveGitCommonDir(tempDir)
	if err != nil {
		t.Fatalf("fallback to gitDir: %v", err)
	}
	if want := filepath.Clean(fakeGitDir); got != want {
		t.Fatalf("got = %q, want %q", got, want)
	}

	// Relative commondir resolved against gitDir.
	subGit := filepath.Join(fakeGitDir, "sub")
	if err := os.Mkdir(subGit, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(subGit, "commondir"), []byte("..\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, ".git"), []byte("gitdir: fake-git/sub\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gotSub, err := resolveGitCommonDir(tempDir)
	if err != nil {
		t.Fatalf("relative commondir: %v", err)
	}
	if want := filepath.Clean(fakeGitDir); gotSub != want {
		t.Fatalf("gotSub = %q, want %q", gotSub, want)
	}

	// Invalid .git file returns error.
	if err := os.WriteFile(filepath.Join(tempDir, ".git"), []byte("not-a-gitdir-line\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := resolveGitCommonDir(tempDir); err == nil {
		t.Fatal("expected error for invalid .git pointer")
	}
}

func TestGuardArbitrateGitDirFailureModes(t *testing.T) {
	invalidRoot := t.TempDir()
	if err := os.WriteFile(filepath.Join(invalidRoot, "dos.toml"), []byte("[lanes]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(invalidRoot, ".git"), []byte("not-a-gitdir\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Enforce mode fails closed.
	lease, err := guardArbitrateAcquire(context.Background(), io.Discard, guardArbitrateConfig{
		Mode: guardArbitrateModeEnforce,
		Root: invalidRoot,
		Tree: []string{"cmd/**"},
	})
	if lease != nil {
		lease.Close()
		t.Fatal("acquire unexpectedly returned a lease")
	}
	if err == nil || !strings.Contains(err.Error(), "COLLISION_RISK: guard admission git directory unavailable") {
		t.Fatalf("err = %v, want COLLISION_RISK: guard admission git directory unavailable", err)
	}

	// Shadow mode fails open and logs.
	var stderr strings.Builder
	lease, err = guardArbitrateAcquire(context.Background(), &stderr, guardArbitrateConfig{
		Mode: guardArbitrateModeShadow,
		Root: invalidRoot,
		Tree: []string{"cmd/**"},
	})
	if err != nil || lease != nil {
		t.Fatalf("shadow lease=%v err=%v, want nil, nil", lease, err)
	}
	if got := stderr.String(); !strings.Contains(got, "fak guard: arbitrate fail-open; git directory unavailable") {
		t.Fatalf("shadow log = %q, want 'git directory unavailable'", got)
	}
}

func TestGuardArbitrateNonBusyLockFailureShadowFailsOpen(t *testing.T) {
	root := guardArbitrateTestRepo(t)
	lockPath := filepath.Join(root, ".git", "fak-guard-arbitrate.lock")
	_ = os.RemoveAll(lockPath)
	if err := os.Mkdir(lockPath, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.RemoveAll(lockPath)
	})

	var stderr strings.Builder
	lease, err := guardArbitrateAcquire(context.Background(), &stderr, guardArbitrateConfig{
		Mode: guardArbitrateModeShadow,
		Root: root,
		Tree: []string{"cmd/**"},
	})
	if err != nil || lease != nil {
		t.Fatalf("shadow lease=%v err=%v, want nil, nil", lease, err)
	}
	if got := stderr.String(); !strings.Contains(got, "fak guard: arbitrate fail-open; admission lock unavailable") {
		t.Fatalf("shadow log = %q, want 'admission lock unavailable'", got)
	}
}
