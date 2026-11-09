package main

// devbuildcache_test.go is the closed contract for the fak build bootstrap
// cache (devbuildcache.go): the digest equals a direct walk of the same files,
// mtime+size changes invalidate it, a hit is reused without rebuilding, a
// corrupted or TTL-expired entry is rebuilt, the kill-switch bypasses the
// cache, and every failure falls open to the legacy handoff.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

// writeDevBuildFixture lays out a minimal checkout-shaped tree: go.mod
// (required by the digest), the three keyed source dirs, and go.sum.
func writeDevBuildFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, f := range []struct{ rel, body string }{
		{"go.mod", "module example.com/fixture\n\ngo 1.26\n"},
		{"go.sum", "fake-sum\n"},
		{filepath.Join("cmd", "fak-dev", "main.go"), "package main\nfunc main() {}\n"},
		{filepath.Join("cmd", "fak-dev", "util.go"), "package main\nfunc util() {}\n"},
		{filepath.Join("internal", "devcmd", "run.go"), "package devcmd\nfunc Run() {}\n"},
		{filepath.Join("internal", "appversion", "version.go"), "package appversion\nconst V = \"1\"\n"},
		// A non-Go file and a subdirectory must be ignored by the digest walk.
		{filepath.Join("cmd", "fak-dev", "notes.txt"), "not go source\n"},
	} {
		path := filepath.Join(root, f.rel)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(f.body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(filepath.Join(root, "cmd", "fak-dev", "testdata"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "cmd", "fak-dev", "testdata", "nested.go"), []byte("package testdata\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

// directDevBuildDigest recomputes the digest straight from os.Stat on the same
// file set — the independent walk the memoized digest must agree with.
func directDevBuildDigest(t *testing.T, root string, relDirs, relFiles []string) string {
	t.Helper()
	var rows []string
	for _, dir := range relDirs {
		entries, err := os.ReadDir(filepath.Join(root, dir))
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
				continue
			}
			info, err := os.Stat(filepath.Join(root, dir, entry.Name()))
			if err != nil {
				t.Fatal(err)
			}
			rows = append(rows, dir+"/"+entry.Name()+"|"+strconv.FormatInt(info.Size(), 10)+"|"+strconv.FormatInt(info.ModTime().UnixNano(), 10))
		}
	}
	for _, file := range relFiles {
		info, err := os.Stat(filepath.Join(root, file))
		if err != nil {
			t.Fatal(err)
		}
		rows = append(rows, file+"|"+strconv.FormatInt(info.Size(), 10)+"|"+strconv.FormatInt(info.ModTime().UnixNano(), 10))
	}
	sum := sha256.Sum256([]byte(strings.Join(rows, "\n")))
	return hex.EncodeToString(sum[:])
}

func TestDevBuildDigestRowsMatchesDirectWalk(t *testing.T) {
	resetDevBuildDigestMemo()
	t.Cleanup(resetDevBuildDigestMemo)
	root := writeDevBuildFixture(t)

	got, err := devBuildDigestRows(root, devBuildKeyDirs, devBuildKeyFiles)
	if err != nil {
		t.Fatalf("digest: %v", err)
	}
	want := directDevBuildDigest(t, root, devBuildKeyDirs, devBuildKeyFiles)
	if got != want {
		t.Fatalf("digest = %s, want direct-walk digest %s", got, want)
	}
	again, err := devBuildDigestRows(root, devBuildKeyDirs, devBuildKeyFiles)
	if err != nil || again != got {
		t.Fatalf("digest not deterministic: %s vs %s (err %v)", again, got, err)
	}
}

func TestDevBuildDigestRowsInvalidatesOnMtimeAndSize(t *testing.T) {
	resetDevBuildDigestMemo()
	t.Cleanup(resetDevBuildDigestMemo)
	root := writeDevBuildFixture(t)
	before, err := devBuildDigestRows(root, devBuildKeyDirs, devBuildKeyFiles)
	if err != nil {
		t.Fatal(err)
	}

	// Size change: appending a byte must move the digest.
	path := filepath.Join(root, "cmd", "fak-dev", "util.go")
	if err := os.WriteFile(path, []byte("package main\nfunc util() {}\n// touched\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	afterSize, err := devBuildDigestRows(root, devBuildKeyDirs, devBuildKeyFiles)
	if err != nil {
		t.Fatal(err)
	}
	if afterSize == before {
		t.Fatal("size change did not invalidate the digest")
	}

	// mtime-only change: same bytes, bumped mtime, must also move it.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	future := info.ModTime().Add(10 * time.Second)
	if err := os.Chtimes(path, future, future); err != nil {
		t.Fatal(err)
	}
	afterMtime, err := devBuildDigestRows(root, devBuildKeyDirs, devBuildKeyFiles)
	if err != nil {
		t.Fatal(err)
	}
	if afterMtime == afterSize {
		t.Fatal("mtime-only change did not invalidate the digest")
	}

	// The memoized entry-point agrees with the fresh digest after a reset.
	resetDevBuildDigestMemo()
	memoized, err := devBuildSourceDigest(root)
	if err != nil || memoized != afterMtime {
		t.Fatalf("memoized digest = %s (err %v), want %s", memoized, err, afterMtime)
	}
	if twice, err := devBuildSourceDigest(root); err != nil || twice != memoized {
		t.Fatalf("memo did not serve the cached digest: %s vs %s (err %v)", twice, memoized, err)
	}
}

func TestDevBuildSourceDigestRequiresCheckout(t *testing.T) {
	resetDevBuildDigestMemo()
	t.Cleanup(resetDevBuildDigestMemo)
	if _, err := devBuildSourceDigest(t.TempDir()); err == nil {
		t.Fatal("digest succeeded on a non-checkout dir, want an error (cache not applicable)")
	}
}

// fakeDevBuild counts invocations and writes a platform-magic "binary".
func fakeDevBuild(calls *int) func(goTool, root, outPath string) error {
	return func(_, _, outPath string) error {
		*calls++
		if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
			return err
		}
		// Match the host format accepted by the production corruption guard.
		// This fixture is never executed; keep the guard itself unchanged.
		header := "MZ"
		switch runtime.GOOS {
		case "linux":
			header = "\x7fELF"
		case "darwin":
			header = "\xcf\xfa\xed\xfe" // little-endian 64-bit Mach-O
		}
		body := header + "fake-fak-dev-binary-payload"
		if err := os.WriteFile(outPath+".staging", []byte(body), 0o755); err != nil {
			return err
		}
		return os.Rename(outPath+".staging", outPath)
	}
}

func TestDevBuildCachedFakDevRoundTripCorruptionTTLAndSwitch(t *testing.T) {
	resetDevBuildDigestMemo()
	t.Cleanup(func() {
		resetDevBuildDigestMemo()
		devBuildCacheDirOverride = ""
		devBuildCacheBuildFn = devBuildGoBuildFakDev
		devBuildCacheNow = time.Now
		devBuildCacheTTL = 24 * time.Hour
	})
	root := writeDevBuildFixture(t)
	devBuildCacheDirOverride = t.TempDir()
	var calls int
	devBuildCacheBuildFn = fakeDevBuild(&calls)

	// Miss -> one build -> hit.
	exe, ok := devBuildCachedFakDev(root, "go")
	if !ok || calls != 1 || !strings.Contains(exe, "fak-dev-") {
		t.Fatalf("first call: ok=%v calls=%d exe=%s, want one build and a key-named entry", ok, calls, exe)
	}
	// Warm handoff: the entry is re-exec'd, not rebuilt.
	if exe2, ok := devBuildCachedFakDev(root, "go"); !ok || exe2 != exe || calls != 1 {
		t.Fatalf("second call: ok=%v calls=%d exe=%s, want a cache hit with no rebuild", ok, calls, exe2)
	}
	// Corruption guard: a truncated entry is rebuilt, never exec'd.
	if err := os.WriteFile(exe, []byte("M"), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, ok := devBuildCachedFakDev(root, "go"); !ok || calls != 2 {
		t.Fatalf("corrupt entry: ok=%v calls=%d, want a rebuild", ok, calls)
	}
	// TTL guard: an entry older than the TTL is rebuilt even though the key matches.
	built, err := os.Stat(exe)
	if err != nil {
		t.Fatal(err)
	}
	devBuildCacheNow = func() time.Time { return built.ModTime().Add(devBuildCacheTTL + time.Minute) }
	devBuildCachedFakDev(root, "go") // the expired miss triggers the rebuild path (ok is clock-skewed false)
	if calls != 3 {
		t.Fatalf("expired entry: calls=%d, want a TTL rebuild", calls)
	}
	// With the clock sane again the rebuilt entry is served without another build.
	devBuildCacheNow = time.Now
	if exe3, ok := devBuildCachedFakDev(root, "go"); !ok || exe3 == "" || calls != 3 {
		t.Fatalf("post-TTL entry: ok=%v calls=%d exe=%s, want a hit on the rebuilt entry", ok, calls, exe3)
	}
	// Kill-switch: the cache is bypassed entirely (no hit, no build).
	t.Setenv("FAK_DEV_BUILDCACHE", "off")
	if exe, ok := devBuildCachedFakDev(root, "go"); ok {
		t.Fatalf("kill-switch still served %s", exe)
	}
	if calls != 3 {
		t.Fatalf("kill-switch rebuilt (calls=%d), want the cache fully bypassed", calls)
	}
}

func TestDevBuildCachedFakDevFailsOpen(t *testing.T) {
	resetDevBuildDigestMemo()
	t.Cleanup(func() {
		resetDevBuildDigestMemo()
		devBuildCacheDirOverride = ""
		devBuildCacheBuildFn = devBuildGoBuildFakDev
	})
	root := writeDevBuildFixture(t)
	devBuildCacheDirOverride = t.TempDir()
	devBuildCacheBuildFn = func(_, _, _ string) error { return fmt.Errorf("boom") }
	if exe, ok := devBuildCachedFakDev(root, "go"); ok {
		t.Fatalf("failed build served %s, want a fail-open miss", exe)
	}
}
