package selfinstall

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRealRunnerKeepsGoWorkOutsideRepoScratch(t *testing.T) {
	repoTmp := filepath.Join(t.TempDir(), "_scratch", "go-tmp")
	if err := os.MkdirAll(repoTmp, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GOTMPDIR", repoTmp)

	out, ok := RealRunner(context.Background(), "", "go", "env", "GOTMPDIR")
	if !ok {
		t.Fatalf("go env GOTMPDIR failed: %s", out)
	}
	if got, want := filepath.Clean(strings.TrimSpace(out)), filepath.Clean(os.TempDir()); !strings.EqualFold(got, want) {
		t.Fatalf("child GOTMPDIR = %q, want OS temp %q outside sweepable repo scratch", got, want)
	}
}

func TestGoRunnerEnvLeavesNonGoChildrenUnchanged(t *testing.T) {
	if got := goRunnerEnv("git", []string{"GOTMPDIR=repo-scratch", "GOCACHE=ambient"}, "os-temp", "owned-cache"); got != nil {
		t.Fatalf("non-Go child env = %v, want nil (inherit unchanged)", got)
	}
}

func TestSelfUpdateGoCachePathsAreStableAndAttemptBounded(t *testing.T) {
	cacheBase := filepath.Join(t.TempDir(), "user-cache")
	tempDir := filepath.Join(t.TempDir(), "temp")
	primaryA, recoveryA := selfUpdateGoCachePaths(cacheBase, tempDir)
	primaryB, recoveryB := selfUpdateGoCachePaths(cacheBase, tempDir)
	if primaryA != primaryB || recoveryA != recoveryB {
		t.Fatalf("cache paths changed per attempt: first=%q/%q second=%q/%q", primaryA, recoveryA, primaryB, recoveryB)
	}
	if primaryA != filepath.Join(cacheBase, "fak", "self-update", "go-build-v1") {
		t.Fatalf("primary=%q, want one stable fak-owned warm cache", primaryA)
	}
	if recoveryA != filepath.Join(tempDir, "fak-self-update-go-build-recovery-v1") {
		t.Fatalf("recovery=%q, want one fixed cleanup target rather than per-attempt residue", recoveryA)
	}
}

func TestGoCacheRunnerOwnsGoEnvironmentAndLeavesNonGoUnchanged(t *testing.T) {
	root := t.TempDir()
	primary := filepath.Join(root, "primary")
	recovery := filepath.Join(root, "recovery")
	tempDir := filepath.Join(root, "tmp")
	t.Setenv("GOCACHE", filepath.Join(root, "ambient"))
	t.Setenv("GOTMPDIR", filepath.Join(root, "repo-scratch"))

	var goEnv []string
	var gitEnv []string
	raw := func(_ context.Context, _ string, name string, _ []string, env []string) (string, bool) {
		if name == "go" {
			goEnv = append([]string(nil), env...)
		} else if name == "git" {
			gitEnv = append([]string(nil), env...)
		}
		return "", true
	}
	run, cleanup, err := newGoCacheRunner(primary, recovery, tempDir, raw)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	if _, ok := run(context.Background(), root, "go", "env", "GOCACHE"); !ok {
		t.Fatal("Go child failed")
	}
	if _, ok := run(context.Background(), root, "git", "status"); !ok {
		t.Fatal("Git child failed")
	}
	if got := envValue(goEnv, "GOCACHE"); got != primary {
		t.Fatalf("Go child GOCACHE=%q, want update-owned %q", got, primary)
	}
	if got := envValue(goEnv, "GOTMPDIR"); got != tempDir {
		t.Fatalf("Go child GOTMPDIR=%q, want %q", got, tempDir)
	}
	if gitEnv != nil {
		t.Fatalf("non-Go child env=%v, want nil (inherit unchanged)", gitEnv)
	}
}

func TestGoCacheRunnerRetriesOnlyCacheLifecycleFailureOnceAndCleansRecovery(t *testing.T) {
	root := t.TempDir()
	primary := filepath.Join(root, "primary")
	recovery := filepath.Join(root, "recovery")
	if err := os.MkdirAll(recovery, 0o755); err != nil {
		t.Fatal(err)
	}
	stale := filepath.Join(recovery, "left-by-hard-killed-attempt")
	if err := os.WriteFile(stale, []byte("stale"), 0o600); err != nil {
		t.Fatal(err)
	}
	var calls []string
	raw := func(_ context.Context, _ string, _ string, _ []string, env []string) (string, bool) {
		cache := envValue(env, "GOCACHE")
		calls = append(calls, cache)
		switch len(calls) {
		case 1:
			return fmt.Sprintf("could not import strings (open %s/deadbeef-d: no such file or directory)", cache), false
		case 2:
			return "", true
		default:
			return fmt.Sprintf("open %s/deadbeef-d: no such file or directory", cache), false
		}
	}
	run, cleanup, err := newGoCacheRunner(primary, recovery, filepath.Join(root, "tmp"), raw)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("stale recovery residue was not reaped at next start: %v", err)
	}
	if out, ok := run(context.Background(), root, "go", "vet", "./..."); !ok {
		t.Fatalf("bounded recovery failed: %s", out)
	}
	if out, ok := run(context.Background(), root, "go", "vet", "./..."); ok || !strings.Contains(out, "after the one bounded recovery") {
		t.Fatalf("second cache loss out=%q ok=%v, want precise exhaustion without another retry", out, ok)
	}
	if len(calls) != 3 || calls[0] != primary || calls[1] != recovery || calls[2] != recovery {
		t.Fatalf("cache calls=%v, want primary then exactly two recovery-cache commands", calls)
	}
	if _, err := os.Stat(recovery); err != nil {
		t.Fatalf("recovery cache should exist before cleanup: %v", err)
	}
	cleanup()
	if _, err := os.Stat(recovery); !os.IsNotExist(err) {
		t.Fatalf("recovery cache remains after cleanup: %v", err)
	}
	if _, err := os.Stat(primary); err != nil {
		t.Fatalf("warm primary cache should be retained: %v", err)
	}
}

func TestGoCacheRunnerOutcomeReadout(t *testing.T) {
	root := t.TempDir()
	primary := filepath.Join(root, "primary")
	recovery := filepath.Join(root, "recovery")
	var calls int
	raw := func(_ context.Context, _ string, _ string, _ []string, env []string) (string, bool) {
		calls++
		switch calls {
		case 1:
			return "", true
		case 2:
			return "compile failed", false
		case 3:
			return fmt.Sprintf("open %s/deadbeef-d: no such file or directory", envValue(env, "GOCACHE")), false
		default:
			return "retry failed", false
		}
	}
	var readout strings.Builder
	report := func(counts GoCacheOutcomeCounts) {
		fmt.Fprintln(&readout, FormatGoCacheOutcomeCounts(counts))
	}
	run, cleanup, err := newGoCacheRunner(primary, recovery, root, raw, report)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	if out, ok := run(context.Background(), root, "go", "build", "./cmd/fak"); !ok {
		t.Fatalf("successful command failed: %s", out)
	}
	if out, ok := run(context.Background(), root, "go", "vet", "./..."); ok {
		t.Fatalf("ordinary error unexpectedly succeeded: %s", out)
	}
	if out, ok := run(context.Background(), root, "go", "test", "./..."); ok {
		t.Fatalf("cache lifecycle refusal unexpectedly succeeded: %s", out)
	}

	want := "self-update: go-cache outcomes success=1 refusal=0 error=0\n" +
		"self-update: go-cache outcomes success=1 refusal=0 error=1\n" +
		"self-update: go-cache outcomes success=1 refusal=1 error=1\n"
	if got := readout.String(); got != want {
		t.Fatalf("Go-cache outcome readout = %q, want %q", got, want)
	}
	t.Log(strings.TrimSpace(readout.String()))
}

func TestGoCacheRunnerCleansRecoveryAfterSuccessfulRun(t *testing.T) {
	root := t.TempDir()
	primary := filepath.Join(root, "primary")
	recovery := filepath.Join(root, "recovery")
	calls := 0
	raw := func(_ context.Context, _ string, _ string, _ []string, env []string) (string, bool) {
		calls++
		if calls == 1 {
			return fmt.Sprintf("open %s/deadbeef-d: no such file or directory", envValue(env, "GOCACHE")), false
		}
		return "", true
	}
	run, cleanup, err := newGoCacheRunner(primary, recovery, root, raw)
	if err != nil {
		t.Fatal(err)
	}
	if out, ok := run(context.Background(), root, "go", "build", "./cmd/fak"); !ok {
		t.Fatalf("recovered build failed: %s", out)
	}
	cleanup()
	if _, err := os.Stat(recovery); !os.IsNotExist(err) {
		t.Fatalf("successful run left recovery cache behind: %v", err)
	}
	if _, err := os.Stat(primary); err != nil {
		t.Fatalf("successful run removed warm primary cache: %v", err)
	}
}

func TestGoCacheRunnerDoesNotRetryOrdinaryVetFailure(t *testing.T) {
	root := t.TempDir()
	calls := 0
	raw := func(_ context.Context, _ string, _ string, _ []string, _ []string) (string, bool) {
		calls++
		return "internal/example/example.go:12:2: undefined: definitelyBroken", false
	}
	run, cleanup, err := newGoCacheRunner(filepath.Join(root, "primary"), filepath.Join(root, "recovery"), root, raw)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	out, ok := run(context.Background(), root, "go", "vet", "./...")
	if ok || calls != 1 || !strings.Contains(out, "undefined: definitelyBroken") {
		t.Fatalf("out=%q ok=%v calls=%d, want one fail-closed non-cache attempt", out, ok, calls)
	}
}

func TestGoCacheLifecycleFailureMatchesOnlyOwnedCrossPlatformPath(t *testing.T) {
	if !goCacheLifecycleFailure(`could not import strings (open C:\Users\dev\Caches\fak\self-update\go-build-v1\ab\dead-d: The system cannot find the path specified.)`, `c:\users\DEV\caches\fak\self-update\go-build-v1`) {
		t.Fatal("Windows cache disappearance was not classified case-insensitively")
	}
	if !goCacheLifecycleFailure("open /tmp/fak/cache/ab/dead-d: no such file or directory", "/tmp/fak/cache") {
		t.Fatal("Unix cache disappearance was not classified")
	}
	if goCacheLifecycleFailure("open /repo/internal/missing.go: no such file or directory", "/tmp/fak/cache") {
		t.Fatal("unrelated missing source file was relabeled as a cache lifecycle failure")
	}
}

// This is the real build-to-vet component witness for #10071. The first vet attempt models
// the captured Go failure at the filesystem seam by deleting the cache that the preceding real
// build populated and returning Go's observed missing-entry diagnostic. The bounded retry then
// runs real `go vet` in a fresh cache. This stays deterministic while proving the actual Go
// build and vet commands share one owned lifecycle rather than the ambient developer cache.
func TestGoCacheRunnerRealBuildThenCacheDeletionThenVetRecovery(t *testing.T) {
	if testing.Short() {
		t.Skip("real Go build/vet witness")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain is required")
	}
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.com/cachewitness\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "main.go"), []byte("package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"ok\") }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cacheRoot := t.TempDir()
	primary := filepath.Join(cacheRoot, "owned-primary")
	recovery := filepath.Join(cacheRoot, "owned-recovery")
	ambient := filepath.Join(cacheRoot, "ambient")
	t.Setenv("GOCACHE", ambient)

	vetCalls := 0
	raw := func(ctx context.Context, dir, name string, args []string, env []string) (string, bool) {
		if name == "go" && len(args) > 0 && args[0] == "vet" {
			vetCalls++
			if vetCalls == 1 {
				cache := envValue(env, "GOCACHE")
				if err := os.RemoveAll(cache); err != nil {
					return err.Error(), false
				}
				return fmt.Sprintf("could not import strings (open %s/deadbeef-d: no such file or directory)", cache), false
			}
		}
		return runCommandWithEnv(ctx, dir, name, args, env)
	}
	run, cleanup, err := newGoCacheRunner(primary, recovery, t.TempDir(), raw)
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	if out, ok := run(ctx, root, "go", "build", "./..."); !ok {
		t.Fatalf("real build failed: %s", out)
	}
	if out, ok := run(ctx, root, "go", "vet", "./..."); !ok {
		t.Fatalf("real vet did not recover after cache deletion: %s", out)
	}
	if vetCalls != 2 {
		t.Fatalf("vet calls=%d, want failed cache seam plus one real recovery", vetCalls)
	}
	if _, err := os.Stat(ambient); !os.IsNotExist(err) {
		t.Fatalf("ambient developer cache was used or created: %v", err)
	}
	if !strings.EqualFold(filepath.Clean(envValue(goRunnerEnv("go.exe", os.Environ(), root, primary), "GOCACHE")), filepath.Clean(primary)) {
		t.Fatal("Windows go.exe path did not receive the owned cache")
	}
}

func envValue(env []string, want string) string {
	for i := len(env) - 1; i >= 0; i-- {
		key, value, ok := strings.Cut(env[i], "=")
		if ok && strings.EqualFold(key, want) {
			return value
		}
	}
	return ""
}
