package main

// devbuildcache.go makes the `fak build` source-checkout fallback (dev.go) pay
// the fak-dev build ONCE instead of on every invocation. The fallback exists so
// a source checkout can bootstrap the separately buildable fak-dev artifact
// through `go run ./cmd/fak-dev`; on a Windows dev host each `go run` re-pays
// module-graph resolution, build-cache validation, and the link on every call
// (witnessed 3.1-11.5s warm for `go run ./cmd/fak version`, dominated by
// Defender first-open scans of GOCACHE files —
// PLAYBOOK-WINDOWS-DEFENDER-GO-EXCLUSIONS-2026-11-05). Caching the linked
// binary and re-execing it while the entry sources are unchanged removes that
// per-call tax from the launch path itself.
//
// Contract (closed, tested in devbuildcache_test.go):
//
//   - KEY: mtime+size rows over cmd/fak-dev/*.go, internal/devcmd/*.go,
//     internal/appversion/*.go, go.mod, go.sum, sha256-digested. The walk is
//     memoized in-process (devBuildDigestMemo) so repeated handoffs in one
//     process walk once; a process is one `fak build` invocation, so the memo
//     never outlives the decision it feeds.
//   - BOUNDED STALENESS: the key covers the direct entry packages only;
//     transitive internal-package edits are caught by the 24h TTL. Operators
//     force refresh with FAK_DEV_BUILDCACHE=off or by deleting the cache dir;
//     `go install ./cmd/fak-dev` removes the fallback path entirely.
//   - CORRUPTION GUARD: an entry is written to a temp file and atomically
//     renamed under its key-derived name (a hit therefore cannot be a binary
//     built from a different key), and a hit must start with the platform
//     executable magic. A truncated/partial write can never be exec'd.
//   - FAIL-OPEN: a disabled switch, an unreadable key, a failed build, or a
//     failed lookup falls back to the legacy `go run` handoff — the cache can
//     only skip work, never break or change the handoff's observable behavior
//     (same argv, same streams, same child cwd). A failed cache build retries
//     through the legacy path, so compile errors surface exactly as before, at
//     the cost of one extra build attempt on that failure path.

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// devBuildKeyDirs and devBuildKeyFiles are the cache key's source set: the
// fak-dev entry package, the devcmd implementation package it dispatches
// through, the appversion package its usage row reads, and the module files.
// Top-level *.go only (subdirectories hold testdata, not build inputs).
var (
	devBuildKeyDirs  = []string{"cmd/fak-dev", "internal/devcmd", "internal/appversion"}
	devBuildKeyFiles = []string{"go.mod", "go.sum"}
)

var (
	// devBuildCacheTTL bounds how long an entry may be re-exec'd without a
	// rebuild even when its key still matches (guards missed invalidation:
	// clock skew, restored-from-backup files carrying old mtimes).
	devBuildCacheTTL = 24 * time.Hour
	// devBuildCacheNow is the wall clock used for TTL decisions; injectable
	// so the TTL guard is testable without sleeping.
	devBuildCacheNow = time.Now
	// devBuildCacheDirOverride replaces the cache base directory (tests point
	// it at a fixture tree; empty = os.UserCacheDir).
	devBuildCacheDirOverride string
	// devBuildCacheBuildFn produces the binary at outPath. Production wires
	// devBuildGoBuildFakDev (one `go build -o`); tests inject a fake builder.
	devBuildCacheBuildFn = devBuildGoBuildFakDev
)

// devBuildDigestMemo is the in-process mtime+size-keyed memo: repoRoot ->
// source digest. Errors are not memoized (a transient failure retries on the
// next handoff).
var (
	devBuildDigestMu   sync.Mutex
	devBuildDigestMemo = map[string]string{}
)

func resetDevBuildDigestMemo() {
	devBuildDigestMu.Lock()
	defer devBuildDigestMu.Unlock()
	devBuildDigestMemo = map[string]string{}
}

// devBuildCacheDisabled reports whether the operator turned the cache off.
// Default is on; the switch exists so a suspicious binary can be ruled out
// with one env var instead of a cache-dir hunt.
func devBuildCacheDisabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_DEV_BUILDCACHE"))) {
	case "off", "0", "false", "no":
		return true
	}
	return false
}

// devBuildSourceDigest returns the memoized mtime+size digest of the fak-dev
// entry sources under root. An error means "cache not applicable here" (not a
// checkout, unreadable tree) and always falls open to the legacy handoff.
func devBuildSourceDigest(root string) (string, error) {
	devBuildDigestMu.Lock()
	defer devBuildDigestMu.Unlock()
	if d, ok := devBuildDigestMemo[root]; ok {
		return d, nil
	}
	d, err := devBuildDigestRows(root, devBuildKeyDirs, devBuildKeyFiles)
	if err != nil {
		return "", err
	}
	devBuildDigestMemo[root] = d
	return d, nil
}

// devBuildDigestRows hashes one deterministic row per input: directories
// contribute their top-level *.go files, files contribute themselves. Rows are
// "relpath|size|mtimeUnixNano", so any change the build would see (size,
// mtime) moves the digest — the same mtime+size key the build cache itself
// uses. go.mod is required (it is what makes root a checkout); go.sum is
// optional.
func devBuildDigestRows(root string, relDirs, relFiles []string) (string, error) {
	if strings.TrimSpace(root) == "" {
		return "", errors.New("devbuildcache: empty repo root")
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		return "", fmt.Errorf("devbuildcache: %s is not a checkout (go.mod: %v)", root, err)
	}
	var rows []string
	for _, dir := range relDirs {
		entries, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(dir)))
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue // a partial tree still digests; the build fallback decides
			}
			return "", fmt.Errorf("devbuildcache: read %s: %w", dir, err)
		}
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
				continue
			}
			row, err := devBuildStatRow(filepath.Join(root, filepath.FromSlash(dir), entry.Name()), dir+"/"+entry.Name())
			if err != nil {
				return "", err
			}
			rows = append(rows, row)
		}
	}
	for _, file := range relFiles {
		row, err := devBuildStatRow(filepath.Join(root, filepath.FromSlash(file)), file)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) && file == "go.sum" {
				continue
			}
			return "", err
		}
		rows = append(rows, row)
	}
	sum := sha256.Sum256([]byte(strings.Join(rows, "\n")))
	return hex.EncodeToString(sum[:]), nil
}

func devBuildStatRow(path, label string) (string, error) {
	info, err := os.Stat(path)
	if err != nil {
		return "", fmt.Errorf("devbuildcache: stat %s: %w", label, err)
	}
	return label + "|" + strconv.FormatInt(info.Size(), 10) + "|" + strconv.FormatInt(info.ModTime().UnixNano(), 10), nil
}

// devBuildCachePath is the key-named entry path: <cache>/fak/dev-buildcache/
// fak-dev-<digest16>[.exe]. Embedding the digest in the name is the first
// corruption guard — a hit cannot be a binary built from a different key.
func devBuildCachePath(digest string) (string, error) {
	base := devBuildCacheDirOverride
	if base == "" {
		user, err := os.UserCacheDir()
		if err != nil || strings.TrimSpace(user) == "" {
			return "", fmt.Errorf("devbuildcache: user cache dir unavailable: %v", err)
		}
		base = user
	}
	name := "fak-dev-" + digest[:16]
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(base, "fak", "dev-buildcache", name), nil
}

// devBuildCacheLookup returns the entry when it exists, passes the executable
// magic check, and is inside the TTL.
func devBuildCacheLookup(path string) (string, bool) {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return "", false
	}
	if devBuildCacheNow().Sub(info.ModTime()) > devBuildCacheTTL {
		return "", false
	}
	if !devBuildExecutableMagic(path) {
		return "", false
	}
	return path, true
}

// devBuildExecutableMagic reads the platform executable magic — the second
// corruption guard, so a truncated or text-file entry is rebuilt, not exec'd.
func devBuildExecutableMagic(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	var buf [4]byte
	n, _ := io.ReadFull(f, buf[:])
	switch runtime.GOOS {
	case "windows":
		return n >= 2 && buf[0] == 'M' && buf[1] == 'Z'
	case "linux":
		return n == 4 && buf[0] == 0x7f && buf[1] == 'E' && buf[2] == 'L' && buf[3] == 'F'
	case "darwin":
		if n < 4 {
			return false
		}
		macho := [][4]byte{
			{0xfe, 0xed, 0xfa, 0xce}, {0xce, 0xfa, 0xed, 0xfe}, // 32-bit, BE/LE
			{0xfe, 0xed, 0xfa, 0xcf}, {0xcf, 0xfa, 0xed, 0xfe}, // 64-bit, BE/LE
		}
		for _, m := range macho {
			if buf == m {
				return true
			}
		}
		return false
	default:
		return n > 0
	}
}

// devBuildGoBuildFakDev is the production builder: one `go build -o` into a
// temp file, then an atomic rename onto the key-named entry. Build errors go
// to the caller's stderr exactly as the legacy `go run` path would print them.
func devBuildGoBuildFakDev(goTool, root, outPath string) error {
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return fmt.Errorf("devbuildcache: mkdir: %w", err)
	}
	tmpPath := outPath + ".tmp-" + strconv.Itoa(os.Getpid())
	_ = os.Remove(tmpPath)
	cmd := exec.Command(goTool, "build", "-o", tmpPath, "./cmd/fak-dev")
	cmd.Dir = root
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("devbuildcache: build fak-dev: %w", err)
	}
	if err := os.Rename(tmpPath, outPath); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("devbuildcache: publish entry: %w", err)
	}
	return nil
}

// devBuildCachedFakDev resolves the cached fak-dev binary for root, building
// it once on a miss. Every failure mode returns false so the caller falls open
// to the legacy `go run` handoff.
func devBuildCachedFakDev(root, goTool string) (string, bool) {
	if devBuildCacheDisabled() {
		return "", false
	}
	digest, err := devBuildSourceDigest(root)
	if err != nil {
		return "", false
	}
	path, err := devBuildCachePath(digest)
	if err != nil {
		return "", false
	}
	if exe, ok := devBuildCacheLookup(path); ok {
		return exe, true
	}
	if err := devBuildCacheBuildFn(goTool, root, path); err != nil {
		return "", false
	}
	return devBuildCacheLookup(path)
}
