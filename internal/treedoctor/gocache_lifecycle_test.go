package treedoctor

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestSweepGoCacheUsesSizeAgeHysteresis(t *testing.T) {
	root := filepath.Join(t.TempDir(), "go-build")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)
	old1 := writeGoCacheFixture(t, root, "00", 8, now.Add(-10*24*time.Hour))
	old2 := writeGoCacheFixture(t, root, "01", 8, now.Add(-9*24*time.Hour))
	recent := writeGoCacheFixture(t, root, "02", 8, now.Add(-time.Hour))
	rep := SweepGoCache(GoCacheOptions{Root: root, Now: now, HighBytes: 20, LowBytes: 12, MinAge: 7 * 24 * time.Hour, ActiveBuild: func() (bool, error) { return false, nil }}, true)
	if len(rep.Reaped) != 2 || rep.BytesAfter != 8 {
		t.Fatalf("report = %+v", rep)
	}
	for _, p := range []string{old1, old2} {
		if _, err := os.Stat(filepath.Join(p, "entry")); !os.IsNotExist(err) {
			t.Fatalf("stale file remains: %s (%v)", p, err)
		}
		if info, err := os.Stat(p); err != nil || !info.IsDir() {
			t.Fatalf("cache shard was removed: %s (%v)", p, err)
		}
	}
	if _, err := os.Stat(filepath.Join(recent, "entry")); err != nil {
		t.Fatalf("recent path removed: %v", err)
	}
}

func TestSweepGoCachePreservesFreshSiblingInSameShard(t *testing.T) {
	root := newGoCacheRoot(t)
	now := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)
	shard := filepath.Join(root, "00")
	if err := os.Mkdir(shard, 0o755); err != nil {
		t.Fatal(err)
	}
	stale := filepath.Join(shard, "stale-a")
	fresh := filepath.Join(shard, "fresh-a")
	for path, mod := range map[string]time.Time{stale: now.Add(-10 * 24 * time.Hour), fresh: now.Add(-time.Hour)} {
		if err := os.WriteFile(path, make([]byte, 8), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, mod, mod); err != nil {
			t.Fatal(err)
		}
	}
	rep := SweepGoCache(GoCacheOptions{Root: root, Now: now, HighBytes: 10, LowBytes: 8, MinAge: 7 * 24 * time.Hour, FreeBytesKnown: true, FreeBytes: 1 << 40, ActiveBuild: func() (bool, error) { return false, nil }}, true)
	if len(rep.Reaped) != 1 || filepath.Base(rep.Reaped[0]) != filepath.Base(stale) {
		t.Fatalf("report=%+v", rep)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("stale sibling remains: %v", err)
	}
	if _, err := os.Stat(fresh); err != nil {
		t.Fatalf("fresh sibling removed: %v", err)
	}
	if info, err := os.Stat(shard); err != nil || !info.IsDir() {
		t.Fatalf("shard removed: %v", err)
	}
}

func TestGoCacheRootFromEnvHandlesNilLookup(t *testing.T) {
	want := filepath.Join(t.TempDir(), "go-build")
	got := GoCacheRootFromEnv(nil, func() (string, error) { return filepath.Dir(want), nil })
	if got != want {
		t.Fatalf("root=%q want=%q", got, want)
	}
	if got := GoCacheRootFromEnv(nil, nil); got != "" {
		t.Fatalf("nil providers returned %q", got)
	}
}

func TestValidateGoCacheRootRejectsProtectedModelStoresAndCustomBasename(t *testing.T) {
	for _, marker := range []string{"fak-models", "qwen3.8", "llama.cpp", "ollama", "models"} {
		t.Run(marker, func(t *testing.T) {
			root := filepath.Join(t.TempDir(), marker, "go-build")
			if err := os.MkdirAll(root, 0o755); err != nil {
				t.Fatal(err)
			}
			if _, _, err := validateGoCacheRoot(root); err == nil || !strings.Contains(err.Error(), "protected") {
				t.Fatalf("root %q err=%v", root, err)
			}
		})
	}
	custom := filepath.Join(t.TempDir(), "custom-go-cache")
	if err := os.Mkdir(custom, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, _, err := validateGoCacheRoot(custom); err == nil || !strings.Contains(err.Error(), "non-go-build") {
		t.Fatalf("custom root err=%v", err)
	}
}

func TestAcquireGoCacheLockConservativelyRecoversDeadOwner(t *testing.T) {
	root := newGoCacheRoot(t)
	lock := filepath.Join(root, ".fak-gocache-lifecycle.lock")
	if err := os.WriteFile(lock, []byte(`{"pid":4242,"token":"dead-owner"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	unlock, err := acquireGoCacheLockWith(root, func(pid int) (bool, error) {
		if pid != 4242 {
			t.Fatalf("pid=%d", pid)
		}
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if b, err := os.ReadFile(lock); err != nil || strings.Contains(string(b), "dead-owner") {
		t.Fatalf("lock not replaced: %q err=%v", b, err)
	}
	unlock()
	if _, err := os.Stat(lock); !os.IsNotExist(err) {
		t.Fatalf("lock remains after unlock: %v", err)
	}
}

func TestAcquireGoCacheLockFailsClosedForLiveUnknownAndMalformedOwners(t *testing.T) {
	for _, tc := range []struct {
		name, contents string
		alive          func(int) (bool, error)
	}{
		{name: "live", contents: `{"pid":4242,"token":"live"}`, alive: func(int) (bool, error) { return true, nil }},
		{name: "unknown", contents: `{"pid":4242,"token":"unknown"}`, alive: func(int) (bool, error) { return false, errors.New("unavailable") }},
		{name: "malformed", contents: `not-json`, alive: func(int) (bool, error) { t.Fatal("malformed lock queried"); return false, nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := newGoCacheRoot(t)
			lock := filepath.Join(root, ".fak-gocache-lifecycle.lock")
			if err := os.WriteFile(lock, []byte(tc.contents), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := acquireGoCacheLockWith(root, tc.alive); err == nil {
				t.Fatal("lock unexpectedly acquired")
			}
			if b, err := os.ReadFile(lock); err != nil || string(b) != tc.contents {
				t.Fatalf("existing lock changed: %q err=%v", b, err)
			}
		})
	}
}

func TestSweepGoCachePressureTriggerAndActiveBuildSkip(t *testing.T) {
	root := filepath.Join(t.TempDir(), "go-build")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	path := writeGoCacheFixture(t, root, "aa", 8, now.Add(-10*24*time.Hour))
	rep := SweepGoCache(GoCacheOptions{Root: root, Now: now, HighBytes: 100, LowBytes: 1, MinAge: time.Hour, MinFreeBytes: 50, FreeBytes: 10, FreeBytesKnown: true, ActiveBuild: func() (bool, error) { return true, nil }}, true)
	if rep.Skipped != "active build" || len(rep.Reaped) != 0 {
		t.Fatalf("report = %+v", rep)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("active cache removed: %v", err)
	}
}

func TestSweepGoCacheApplyRequiresBuildWitness(t *testing.T) {
	root := filepath.Join(t.TempDir(), "go-build")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	path := writeGoCacheFixture(t, root, "bb", 8, now.Add(-10*24*time.Hour))
	rep := SweepGoCache(GoCacheOptions{Root: root, Now: now, HighBytes: 2, LowBytes: 1, MinAge: time.Hour}, true)
	if rep.Skipped != "active-build state unavailable" {
		t.Fatalf("report = %+v", rep)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("cache removed without witness: %v", err)
	}
}

func TestSweepGoCacheDryRunNeverMutates(t *testing.T) {
	root := filepath.Join(t.TempDir(), "go-build")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	path := writeGoCacheFixture(t, root, "cc", 8, now.Add(-10*24*time.Hour))
	rep := SweepGoCache(GoCacheOptions{Root: root, Now: now, HighBytes: 2, LowBytes: 1, MinAge: time.Hour}, false)
	if len(rep.Candidates) != 1 || len(rep.Reaped) != 0 {
		t.Fatalf("report = %+v", rep)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("dry run removed cache: %v", err)
	}
}

func TestGoCachePathsAreCrossPlatformAndRootBounded(t *testing.T) {
	cases := []struct{ os, root string }{
		{"darwin", "/Users/test/Library/Caches/go-build"},
		{"windows", `C:\\Users\\test\\AppData\\Local\\go-build`},
	}
	for _, tc := range cases {
		t.Run(tc.os, func(t *testing.T) {
			if tc.os == "windows" && runtime.GOOS != "windows" && filepath.Clean(tc.root) == "" {
				t.Fatal("windows path lost")
			}
			if containsModelStore(tc.root) {
				t.Fatalf("Go cache root overlaps protected model store: %s", tc.root)
			}
		})
	}
	for _, p := range []string{"/Users/test/.cache/fak-models", "/Users/test/Library/Caches/llama.cpp", `C:\\Users\\test\\.ollama`, "/Users/test/models"} {
		if !containsModelStore(p) {
			t.Fatalf("model store not recognized: %s", p)
		}
	}
}

func containsModelStore(path string) bool {
	s := strings.ReplaceAll(strings.ToLower(path), `\`, "/")
	for _, marker := range []string{"fak-models", "llama.cpp", "/.ollama", "/models"} {
		if strings.Contains(s, marker) {
			return true
		}
	}
	return false
}

func writeGoCacheFixture(t *testing.T, root, name string, size int, mod time.Time) string {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	file := filepath.Join(dir, "entry")
	if err := os.WriteFile(file, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(file, mod, mod); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(dir, mod, mod); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestSweepGoCacheLockContentionPrecedesCensus(t *testing.T) {
	root := newGoCacheRoot(t)
	if err := os.WriteFile(filepath.Join(root, ".fak-gocache-lifecycle.lock"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	calls := 0
	rep := SweepGoCache(GoCacheOptions{Root: root, ActiveBuild: func() (bool, error) { calls++; return false, nil }, Progress: func(GoCacheProgress) error { t.Fatal("census ran under contention"); return nil }}, true)
	if rep.Skipped != "lifecycle lock busy" || calls != 0 || rep.ScanEntries != 0 {
		t.Fatalf("report=%+v calls=%d", rep, calls)
	}
}

func TestSweepGoCacheRechecksBuildAndCallbackErrorsFailClosed(t *testing.T) {
	root := newGoCacheRoot(t)
	now := time.Now()
	path := writeGoCacheFixture(t, root, "aa", 8, now.Add(-48*time.Hour))
	calls := 0
	rep := SweepGoCache(GoCacheOptions{Root: root, Now: now, HighBytes: 1, LowBytes: 0, MinAge: time.Hour, ActiveBuild: func() (bool, error) {
		calls++
		if calls == 2 {
			return false, errors.New("lost witness")
		}
		return false, nil
	}}, true)
	if calls != 2 || rep.Skipped != "build state unknown" || rep.Err == "" {
		t.Fatalf("report=%+v calls=%d", rep, calls)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("mutated after witness error: %v", err)
	}
}

func TestSweepGoCacheProgressAndIncompleteBudget(t *testing.T) {
	root := newGoCacheRoot(t)
	now := time.Now()
	writeGoCacheFixture(t, root, "aa", 8, now.Add(-48*time.Hour))
	writeGoCacheFixture(t, root, "bb", 8, now.Add(-48*time.Hour))
	progress := 0
	rep := SweepGoCache(GoCacheOptions{Root: root, Now: now, HighBytes: 1, LowBytes: 1, MinAge: time.Hour, MaxWalkEntries: 1, Progress: func(p GoCacheProgress) error { progress++; return nil }}, false)
	if rep.ScanComplete || rep.IncompleteReason != "entry budget exhausted" || rep.Skipped != "incomplete census" || progress == 0 {
		t.Fatalf("report=%+v progress=%d", rep, progress)
	}
}

func TestSweepGoCacheApplyDoesNotMutateAfterIncompleteCensus(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts func(root string, now time.Time) GoCacheOptions
	}{
		{name: "entry-budget", opts: func(root string, now time.Time) GoCacheOptions {
			return GoCacheOptions{Root: root, Now: now, HighBytes: 1, LowBytes: 1, MinAge: time.Hour, MaxWalkEntries: 1, FreeBytesKnown: true, FreeBytes: 1 << 40, ActiveBuild: func() (bool, error) { return false, nil }}
		}},
		{name: "canceled", opts: func(root string, now time.Time) GoCacheOptions {
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			return GoCacheOptions{Root: root, Now: now, HighBytes: 1, LowBytes: 1, MinAge: time.Hour, Context: ctx, FreeBytesKnown: true, FreeBytes: 1 << 40, ActiveBuild: func() (bool, error) { return false, nil }}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := newGoCacheRoot(t)
			now := time.Now()
			first := writeGoCacheFixture(t, root, "aa", 8, now.Add(-48*time.Hour))
			second := writeGoCacheFixture(t, root, "bb", 8, now.Add(-48*time.Hour))
			rep := SweepGoCache(tc.opts(root, now), true)
			if rep.ScanComplete || rep.Skipped != "incomplete census" || len(rep.Reaped) != 0 {
				t.Fatalf("report=%+v", rep)
			}
			for _, dir := range []string{first, second} {
				if _, err := os.Stat(filepath.Join(dir, "entry")); err != nil {
					t.Fatalf("cache mutated after incomplete census: %v", err)
				}
			}
		})
	}
}

func TestSweepGoCacheSurfacesExactManagedCleanupCommands(t *testing.T) {
	rep := SweepGoCache(GoCacheOptions{}, false)
	joined := strings.Join(rep.CleanupHints, "\n")
	for _, command := range []string{
		"fak worktree worker reap --all-cold",
		"fak worktree worker reap --all-cold --apply",
		"fak git-daily --gotmp-dir <GOTMPDIR>",
	} {
		if !strings.Contains(joined, command) {
			t.Fatalf("missing cleanup command %q in %#v", command, rep.CleanupHints)
		}
	}
}

func TestSweepGoCacheProgressCallbackError(t *testing.T) {
	root := newGoCacheRoot(t)
	writeGoCacheFixture(t, root, "aa", 8, time.Now().Add(-48*time.Hour))
	want := errors.New("stop progress")
	rep := SweepGoCache(GoCacheOptions{Root: root, Progress: func(GoCacheProgress) error { return want }}, false)
	if !strings.Contains(rep.Err, want.Error()) {
		t.Fatalf("report=%+v", rep)
	}
}

func TestSweepGoCacheProtectedRootAndSymlinkContainment(t *testing.T) {
	protected := filepath.Join(t.TempDir(), "models", "go-build")
	if err := os.MkdirAll(protected, 0o755); err != nil {
		t.Fatal(err)
	}
	if rep := SweepGoCache(GoCacheOptions{Root: protected}, false); !strings.Contains(rep.Err, "protected") {
		t.Fatalf("protected report=%+v", rep)
	}
	root := newGoCacheRoot(t)
	outside := t.TempDir()
	target := filepath.Join(outside, "victim")
	if err := os.WriteFile(target, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(root, "aa")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	rep := SweepGoCache(GoCacheOptions{Root: root, HighBytes: 1, LowBytes: 1, MinAge: time.Hour}, false)
	if len(rep.Candidates) != 0 {
		t.Fatalf("symlink candidate admitted: %+v", rep)
	}
	if _, err := os.Stat(target); err != nil {
		t.Fatal(err)
	}
}

func TestSweepGoCacheStaleTargetShortfallAndByteSemantics(t *testing.T) {
	root := newGoCacheRoot(t)
	now := time.Now()
	writeGoCacheFixture(t, root, "old", 4, now.Add(-48*time.Hour))
	writeGoCacheFixture(t, root, "new", 12, now)
	dry := SweepGoCache(GoCacheOptions{Root: root, Now: now, HighBytes: 1, LowBytes: 1, MinAge: 24 * time.Hour}, false)
	if dry.BytesAfterSemantics != "projected" || dry.TargetShortfallBytes == 0 || dry.CandidateBytes != 4 || dry.CandidateBytesKnown != 1 || dry.CandidateBytesUnknown != 0 {
		t.Fatalf("dry=%+v", dry)
	}
	apply := SweepGoCache(GoCacheOptions{Root: root, Now: now, HighBytes: 1, LowBytes: 1, MinAge: 24 * time.Hour, ActiveBuild: func() (bool, error) { return false, nil }}, true)
	if apply.BytesAfterSemantics != "actual" || apply.BytesAfter != apply.BytesBefore-apply.ReclaimedBytes {
		t.Fatalf("apply=%+v", apply)
	}
}

func newGoCacheRoot(t *testing.T) string {
	t.Helper()
	root := filepath.Join(t.TempDir(), "go-build")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestSweepGoCacheCancellationIsIncomplete(t *testing.T) {
	root := newGoCacheRoot(t)
	writeGoCacheFixture(t, root, "aa", 8, time.Now().Add(-48*time.Hour))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	rep := SweepGoCache(GoCacheOptions{Root: root, Context: ctx}, false)
	if rep.ScanComplete || rep.Skipped != "incomplete census" || !strings.Contains(rep.IncompleteReason, "canceled") {
		t.Fatalf("report=%+v", rep)
	}
}

func TestSweepGoCacheRevalidatesStalenessBeforeRemoval(t *testing.T) {
	root := newGoCacheRoot(t)
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)
	dir := writeGoCacheFixture(t, root, "aa", 8, now.Add(-48*time.Hour))
	path := filepath.Join(dir, "entry")
	buildChecks := 0
	removeCalls := 0
	rep := SweepGoCache(GoCacheOptions{
		Root:      root,
		Now:       now,
		HighBytes: 1,
		LowBytes:  0,
		MinAge:    time.Hour,
		ActiveBuild: func() (bool, error) {
			buildChecks++
			if buildChecks == 2 {
				if err := os.Chtimes(path, now, now); err != nil {
					return false, err
				}
			}
			return false, nil
		},
		Remove: func(string) error {
			removeCalls++
			return nil
		},
	}, true)
	if rep.Err != "" || removeCalls != 0 || len(rep.Reaped) != 0 || rep.ReclaimedBytes != 0 || rep.BytesAfter != rep.BytesBefore {
		t.Fatalf("report=%+v remove_calls=%d", rep, removeCalls)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("refreshed cache entry removed: %v", err)
	}
}
