package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/committedbuildwitness"
)

func TestDispatchTreeBuildCachesSuccessByHead(t *testing.T) {
	oldBuild, oldHead := dispatchTreeBuildCommand, dispatchTreeBuildHead
	builds := 0
	head := "a"
	dispatchTreeBuildCommand = func(string) (string, error) { builds++; return "", nil }
	dispatchTreeBuildHead = func(string) string { return head }
	dispatchTreeBuildSuccesses.Lock()
	dispatchTreeBuildSuccesses.byRoot = nil
	dispatchTreeBuildSuccesses.Unlock()
	t.Cleanup(func() {
		dispatchTreeBuildCommand, dispatchTreeBuildHead = oldBuild, oldHead
		dispatchTreeBuildSuccesses.Lock()
		dispatchTreeBuildSuccesses.byRoot = nil
		dispatchTreeBuildSuccesses.Unlock()
	})
	root := t.TempDir()
	dispatchProbeTreeBuild(root)
	dispatchProbeTreeBuild(root)
	if builds != 1 {
		t.Fatalf("unchanged HEAD builds=%d, want one", builds)
	}
	head = "b"
	dispatchProbeTreeBuild(root)
	if builds != 2 {
		t.Fatalf("changed HEAD builds=%d, want rebuild", builds)
	}
}

func TestDispatchTreeBuildDoesNotCacheFailure(t *testing.T) {
	oldBuild, oldHead := dispatchTreeBuildCommand, dispatchTreeBuildHead
	builds := 0
	dispatchTreeBuildCommand = func(string) (string, error) {
		builds++
		if builds == 1 {
			return "pkg/broken", errors.New("exit status 1")
		}
		return "", nil
	}
	dispatchTreeBuildHead = func(string) string { return "a" }
	dispatchTreeBuildSuccesses.Lock()
	dispatchTreeBuildSuccesses.byRoot = nil
	dispatchTreeBuildSuccesses.Unlock()
	t.Cleanup(func() {
		dispatchTreeBuildCommand, dispatchTreeBuildHead = oldBuild, oldHead
		dispatchTreeBuildSuccesses.Lock()
		dispatchTreeBuildSuccesses.byRoot = nil
		dispatchTreeBuildSuccesses.Unlock()
	})
	root := t.TempDir()
	if got := dispatchProbeTreeBuild(root); !got.Poisoned {
		t.Fatalf("first probe=%+v, want poison", got)
	}
	if got := dispatchProbeTreeBuild(root); got.Poisoned || builds != 2 {
		t.Fatalf("failure was cached: probe=%+v builds=%d", got, builds)
	}
}

func TestDispatchProbeTreeBuildNamesCompilerFailure(t *testing.T) {
	oldBuild, oldHead := dispatchTreeBuildCommand, dispatchTreeBuildHead
	dispatchTreeBuildHead = func(string) string { return "red-head" }
	dispatchTreeBuildCommand = func(string) (string, error) {
		return "# example/broken\nbroken.go:3: undefined: nope", errors.New("exit status 1")
	}
	t.Cleanup(func() { dispatchTreeBuildCommand, dispatchTreeBuildHead = oldBuild, oldHead })
	got := dispatchProbeTreeBuild(t.TempDir())
	if !got.Poisoned || got.Package != "# example/broken" {
		t.Fatalf("got=%+v", got)
	}
}
func TestDispatchProbeTreeBuildGreen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) { return "", nil }
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("got=%+v", got)
	}
}
func TestDispatchProbeTreeBuildMissingGoFailsOpen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) { return "", errors.New("executable file not found") }
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("got=%+v", got)
	}
}

// A probe root without a Go module (a bare temp dir) makes `go build` answer
// "go.mod file not found ..." on stderr with a bare "exit status 1" error. That
// is infrastructure-missing, not a red tree, so it must fail open — otherwise the
// #3583 poison gate freezes every dispatch test that probes a temp workspace.
func TestDispatchProbeTreeBuildNoModuleFailsOpen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) {
		return "go: go.mod file not found in current directory or any parent directory; see 'go help modules'", errors.New("exit status 1")
	}
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("got=%+v", got)
	}
}

// A build that exceeds the 90s probe cap is killed by the deadline, not by a
// compiler diagnostic. dispatchTreeBuildCommand wraps that as
// context.DeadlineExceeded; the probe must fail open (a loaded host is
// infrastructure, not a poisoned tree) rather than freeze the fleet.
func TestDispatchProbeTreeBuildTimeoutFailsOpen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) {
		return "", fmt.Errorf("tree build probe timed out after 90s: %w", context.DeadlineExceeded)
	}
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("timed-out build must fail open, got=%+v", got)
	}
}

// A SIGKILL reap (an OOM kill under fleet load) surfaces as "signal: killed"
// with no compiler diagnostic — infrastructure, not poison, so it fails open.
func TestDispatchProbeTreeBuildKilledFailsOpen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) {
		return "", errors.New("signal: killed")
	}
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("killed build must fail open, got=%+v", got)
	}
}

// Once the probe root is git-init'd (as the dispatch tick test harness leaves it),
// `go build` names the missing module differently: "cannot find main module, but
// found .git/config ...". That is still infrastructure-missing, not a red tree, so
// it must fail open — otherwise every git-init'd tick test refuses TREE_POISONED.
func TestDispatchProbeTreeBuildUnbornRepositoryFailsOpen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) {
		return "fatal: not a valid object name: HEAD", errors.New("git archive: fatal: not a valid object name: HEAD")
	}
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("unborn repository is missing probe infrastructure, got=%+v", got)
	}
}

func TestDispatchProbeTreeBuildNonRepositoryFailsOpen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) {
		return "", errors.New("git archive: fatal: not a git repository (or any of the parent directories): .git")
	}
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("non-repository probe root is missing infrastructure, got=%+v", got)
	}
}
func TestDispatchProbeTreeBuildGitDirNoModuleFailsOpen(t *testing.T) {
	old := dispatchTreeBuildCommand
	dispatchTreeBuildCommand = func(string) (string, error) {
		return "go: cannot find main module, but found .git/config in /tmp/x\n\tto create a module there, run:\n\tgo mod init", errors.New("exit status 1")
	}
	t.Cleanup(func() { dispatchTreeBuildCommand = old })
	if got := dispatchProbeTreeBuild(t.TempDir()); got.Poisoned {
		t.Fatalf("got=%+v", got)
	}
}

func TestDispatchTreeBuildIgnoresBrokenUntrackedPeerFile(t *testing.T) {
	root := t.TempDir()
	runGit := func(args ...string) {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.com/peerdirty\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	mainPath := filepath.Join(root, "cmd", "fak", "main.go")
	if err := os.WriteFile(mainPath, []byte("package main\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit("init")
	runGit("config", "user.email", "test@example.com")
	runGit("config", "user.name", "test")
	runGit("add", "go.mod", "cmd/fak/main.go")
	runGit("commit", "-m", "fixture")
	if err := os.WriteFile(filepath.Join(root, "cmd", "fak", "peer_wip.go"), []byte("package main\nvar _ = undefinedPeerSymbol\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	literal := exec.Command("go", "build", "./cmd/fak")
	literal.Dir = root
	if err := literal.Run(); err == nil {
		t.Fatal("literal peer-dirty tree unexpectedly built")
	}
	if got := dispatchProbeTreeBuild(root); got.Poisoned || got.Error != "" {
		t.Fatalf("committed tree probe = %+v, want green despite unrelated untracked WIP", got)
	}
}

func TestDispatchProbeTreeBuildReusesCrossProcessCommittedWitness(t *testing.T) {
	root := t.TempDir()
	cmd := exec.Command("git", "init")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v\n%s", err, out)
	}
	const head = "0123456789abcdef0123456789abcdef01234567"
	committedbuildwitness.Record(root, head, "ci-preflight", time.Now())

	oldHead, oldBuild := dispatchTreeBuildHead, dispatchTreeBuildCommand
	defer func() {
		dispatchTreeBuildHead, dispatchTreeBuildCommand = oldHead, oldBuild
	}()
	dispatchTreeBuildHead = func(string) string { return head }
	builds := 0
	dispatchTreeBuildCommand = func(string) (string, error) {
		builds++
		return "", errors.New("must not compile a witnessed HEAD")
	}

	if got := dispatchProbeTreeBuild(root); got.Poisoned || got.Error != "" {
		t.Fatalf("cache hit returned %+v", got)
	}
	if builds != 0 {
		t.Fatalf("builds=%d want 0 after independent witness", builds)
	}
}

func TestDispatchProbeTreeBuildReusesProvenanceMatchedBinary(t *testing.T) {
	const head = "0123456789abcdef0123456789abcdef01234567"
	oldHead, oldStamp, oldBuild := dispatchTreeBuildHeadContext, dispatchTreeBuildStamp, dispatchTreeBuildCommandContext
	dispatchTreeBuildHeadContext = func(context.Context, string) string { return head }
	dispatchTreeBuildStamp = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: head, HasVCS: true}
	}
	builds := 0
	dispatchTreeBuildCommandContext = func(context.Context, string) (string, error) {
		builds++
		return "", errors.New("matched binary must not rebuild")
	}
	dispatchTreeBuildSuccesses.Lock()
	dispatchTreeBuildSuccesses.byRoot = nil
	dispatchTreeBuildSuccesses.Unlock()
	t.Cleanup(func() {
		dispatchTreeBuildHeadContext, dispatchTreeBuildStamp, dispatchTreeBuildCommandContext = oldHead, oldStamp, oldBuild
		dispatchTreeBuildSuccesses.Lock()
		dispatchTreeBuildSuccesses.byRoot = nil
		dispatchTreeBuildSuccesses.Unlock()
	})

	check, evidence := dispatchProbeTreeBuildContext(context.Background(), t.TempDir())
	if check.Poisoned || check.Error != "" {
		t.Fatalf("matched binary check = %+v", check)
	}
	if builds != 0 {
		t.Fatalf("builds = %d, want 0", builds)
	}
	if !evidence.Reused || evidence.Source != "running_binary" || evidence.RequestedCommit != head || evidence.BinaryRevision != head {
		t.Fatalf("evidence = %+v, want provenance-valid running-binary reuse", evidence)
	}
}

func TestDispatchPreflightTreeBuildCancelsWithinCallerBudget(t *testing.T) {
	const head = "fedcba9876543210fedcba9876543210fedcba98"
	oldHead, oldStamp, oldBuild := dispatchTreeBuildHeadContext, dispatchTreeBuildStamp, dispatchTreeBuildCommandContext
	dispatchTreeBuildHeadContext = func(context.Context, string) string { return head }
	dispatchTreeBuildStamp = func() binstamp.Stamp {
		return binstamp.Stamp{Revision: "different-revision", HasVCS: true}
	}
	buildCanceled := make(chan error, 1)
	dispatchTreeBuildCommandContext = func(ctx context.Context, _ string) (string, error) {
		<-ctx.Done()
		buildCanceled <- ctx.Err()
		return "", ctx.Err()
	}
	dispatchTreeBuildSuccesses.Lock()
	dispatchTreeBuildSuccesses.byRoot = nil
	dispatchTreeBuildSuccesses.Unlock()
	t.Cleanup(func() {
		dispatchTreeBuildHeadContext, dispatchTreeBuildStamp, dispatchTreeBuildCommandContext = oldHead, oldStamp, oldBuild
		dispatchTreeBuildSuccesses.Lock()
		dispatchTreeBuildSuccesses.byRoot = nil
		dispatchTreeBuildSuccesses.Unlock()
	})

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	started := time.Now()
	check := dispatchPreflightTree(ctx, t.TempDir(), nil)
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("blocked tree build returned after %s, want caller budget", elapsed)
	}
	if check.Poisoned || check.Error != "" {
		t.Fatalf("canceled tree build must fail open, got %+v", check)
	}
	select {
	case err := <-buildCanceled:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("tree build cancellation = %v, want deadline exceeded", err)
		}
	default:
		t.Fatal("tree build did not observe the caller deadline")
	}
}

func TestDispatchPreflightTreePreservesCompilerPoison(t *testing.T) {
	const head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	oldHead, oldStamp, oldBuild := dispatchTreeBuildHeadContext, dispatchTreeBuildStamp, dispatchTreeBuildCommandContext
	dispatchTreeBuildHeadContext = func(context.Context, string) string { return head }
	dispatchTreeBuildStamp = func() binstamp.Stamp { return binstamp.Stamp{Revision: "different-revision", HasVCS: true} }
	dispatchTreeBuildCommandContext = func(context.Context, string) (string, error) {
		return "# example/broken\nbroken.go:3: undefined: nope", errors.New("exit status 1")
	}
	dispatchTreeBuildSuccesses.Lock()
	dispatchTreeBuildSuccesses.byRoot = nil
	dispatchTreeBuildSuccesses.Unlock()
	t.Cleanup(func() {
		dispatchTreeBuildHeadContext, dispatchTreeBuildStamp, dispatchTreeBuildCommandContext = oldHead, oldStamp, oldBuild
		dispatchTreeBuildSuccesses.Lock()
		dispatchTreeBuildSuccesses.byRoot = nil
		dispatchTreeBuildSuccesses.Unlock()
	})

	check := dispatchPreflightTree(context.Background(), t.TempDir(), nil)
	if !check.Poisoned || check.Package != "# example/broken" || check.Error == "" {
		t.Fatalf("compiler failure = %+v, want preserved poison diagnostics", check)
	}
}

func TestDispatchPreflightTreePreservesNonCancellationInfrastructureError(t *testing.T) {
	const head = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	oldHead, oldStamp, oldBuild := dispatchTreeBuildHeadContext, dispatchTreeBuildStamp, dispatchTreeBuildCommandContext
	dispatchTreeBuildHeadContext = func(context.Context, string) string { return head }
	dispatchTreeBuildStamp = func() binstamp.Stamp { return binstamp.Stamp{Revision: "different-revision", HasVCS: true} }
	dispatchTreeBuildCommandContext = func(context.Context, string) (string, error) {
		return "", errors.New("executable file not found")
	}
	dispatchTreeBuildSuccesses.Lock()
	dispatchTreeBuildSuccesses.byRoot = nil
	dispatchTreeBuildSuccesses.Unlock()
	t.Cleanup(func() {
		dispatchTreeBuildHeadContext, dispatchTreeBuildStamp, dispatchTreeBuildCommandContext = oldHead, oldStamp, oldBuild
		dispatchTreeBuildSuccesses.Lock()
		dispatchTreeBuildSuccesses.byRoot = nil
		dispatchTreeBuildSuccesses.Unlock()
	})

	check := dispatchPreflightTree(context.Background(), t.TempDir(), nil)
	if check.Poisoned || check.Error != "executable file not found" {
		t.Fatalf("infrastructure failure = %+v, want unchanged fail-open diagnostic", check)
	}
}
