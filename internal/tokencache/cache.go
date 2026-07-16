// Package tokencache is the persisted, content-addressed backing store for
// clonescan's per-file tokenization (#4330).
//
// clonescan.BuildTreeIndex re-lexes every tracked .go file on every commit gate and
// every `dup guard`, even though almost none changed between invocations — the
// expensive step, qualifyingWindows(goTokens(src,false)) -> (keys, spans), is a pure
// deterministic function of a file's exact bytes. This package memoizes that output
// under the bytes' content hash, so an unchanged file is a file read instead of a
// re-lex, which is what lets the push rung and the CI dup job actually scale.
//
// MODELLED ON internal/witness/cache.go (the verdict cache), borrowing its mechanics
// — a git-common-dir-anchored dir, sha256-named entries, and a Windows-safe atomic
// temp-file + rename write — not the package.
//
// FLEET-SHARED BY PLACEMENT: entries live under <git-common-dir>/fak/token-cache,
// resolved via `git rev-parse --git-common-dir`, so they sit INSIDE .git — never
// committed, never caught by FILE_ADMISSION/DEAD_CODE/god-file, and shared across
// every concurrent session on the shared clone (one session's tokenization is every
// peer's hit).
//
// ACCELERATE-NEVER-GATE: every read/write failure degrades to the exact uncached
// path. A nil *Cache is safe to pass to BuildTreeIndex (every op is a miss/no-op).
//
// ESCAPE HATCH: FAK_TOKEN_CACHE=off (or 0/false/no) disables the cache entirely.
// SIZE BOUND: FAK_TOKEN_CACHE_MAX_BYTES caps the on-disk footprint (FIFO by mod-time,
// enforced once per Open so the per-invocation cost is one directory scan).
package tokencache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/clonescan"
)

// FlagEnv disables the token cache when set to off/0/false/no.
const FlagEnv = "FAK_TOKEN_CACHE"

// MaxBytesEnv overrides the on-disk byte budget (default defaultMaxBytes).
const MaxBytesEnv = "FAK_TOKEN_CACHE_MAX_BYTES"

const entrySchema = "fak-token-window-cache/v1"

// defaultMaxBytes bounds the cache dir when FAK_TOKEN_CACHE_MAX_BYTES is unset. Large
// enough that a full ~5.7k-file tree fits many times over; the budget exists to cap
// unbounded growth across invocations, not to evict a working set.
const defaultMaxBytes = 256 << 20 // 256 MiB

// Cache implements clonescan.WindowCache against a git-common-dir-anchored directory.
var _ clonescan.WindowCache = (*Cache)(nil)

// Enabled reports whether the token cache is active. Off when FAK_TOKEN_CACHE is
// off/0/false/no; on by default (every entry is content-addressed under the tokenizer
// version, so a stale hit is structurally impossible — the switch is for forensics).
func Enabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(FlagEnv))) {
	case "off", "0", "false", "no":
		return false
	default:
		return true
	}
}

// TokenCacheDir maps a git common dir to the fleet-shared token-cache directory. It
// lives inside .git, so it is never committed and never scanned by the tree gates.
func TokenCacheDir(gitCommonDir string) string {
	return filepath.Join(gitCommonDir, "fak", "token-cache")
}

// Cache is a file-backed WindowCache. version tags every entry (and its content
// address), so a tokenizer change produces a different key and misses stale windows.
type Cache struct {
	dir     string
	version string
}

// New constructs a cache rooted at dir, tagging entries with version. A "" dir or ""
// version yields a cache whose every op is a miss/no-op, so callers never branch.
func New(dir, version string) *Cache {
	return &Cache{dir: dir, version: version}
}

// Open resolves <git-common-dir>/fak/token-cache for the repo rooted at root and
// returns a ready cache tagged with clonescan.TokenizerVersion, or nil when the cache
// is disabled or the common dir cannot be resolved. The return type is the
// clonescan.WindowCache seam and a nil result is a true nil interface, so a caller can
// pass it straight to BuildTreeIndex and get the exact uncached path. It also enforces
// the byte budget once (a single directory scan) before returning.
func Open(root string) clonescan.WindowCache {
	if !Enabled() {
		return nil
	}
	dir, ok := commonDir(root)
	if !ok {
		return nil
	}
	c := New(TokenCacheDir(dir), clonescan.TokenizerVersion())
	c.prune(maxBytes())
	return c
}

// commonDir resolves the git common dir for root (absolute), or ok=false. The common
// dir is shared across worktrees of the same clone, so the cache is fleet-shared.
func commonDir(root string) (string, bool) {
	cmd := exec.Command("git", "rev-parse", "--git-common-dir")
	if strings.TrimSpace(root) != "" {
		cmd.Dir = root
	}
	out, err := cmd.Output()
	if err != nil {
		return "", false
	}
	dir := strings.TrimSpace(string(out))
	if dir == "" {
		return "", false
	}
	if !filepath.IsAbs(dir) && strings.TrimSpace(root) != "" {
		dir = filepath.Join(root, dir)
	}
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		return "", false
	}
	return dir, true
}

// entry is one memoized tokenization on disk. Digest re-content-addresses the entry on
// read: a mismatch (corruption, or a filename collision that cannot happen under
// sha256) is ignored, never trusted.
type entry struct {
	Schema  string   `json:"schema"`
	Version string   `json:"version"`
	Digest  string   `json:"digest"`
	Keys    []string `json:"keys"`
	Spans   [][2]int `json:"spans"`
}

// digest content-addresses src UNDER the tokenizer version: a version bump or a
// one-byte source change yields a different digest, so both are misses by
// construction — invalidation is content-addressing, not bookkeeping.
func (c *Cache) digest(src string) string {
	h := sha256.New()
	h.Write([]byte(c.version))
	h.Write([]byte{0})
	h.Write([]byte(src))
	return hex.EncodeToString(h.Sum(nil))
}

func (c *Cache) path(dig string) string {
	return filepath.Join(c.dir, dig+".json")
}

// Get returns the memoized (keys, spans) for the exact bytes src. Unreadable,
// unparseable, stale-schema, wrong-version, or digest-mismatched entries are misses.
func (c *Cache) Get(src string) (keys []string, spans [][2]int, ok bool) {
	if c == nil || strings.TrimSpace(c.dir) == "" || strings.TrimSpace(c.version) == "" {
		return nil, nil, false
	}
	dig := c.digest(src)
	b, err := os.ReadFile(c.path(dig))
	if err != nil {
		return nil, nil, false
	}
	var e entry
	if json.Unmarshal(b, &e) != nil {
		return nil, nil, false
	}
	if e.Schema != entrySchema || e.Version != c.version || e.Digest != dig || len(e.Keys) != len(e.Spans) {
		return nil, nil, false
	}
	return e.Keys, e.Spans, true
}

// Put memoizes the (keys, spans) for src, best-effort. Entries land via temp-file +
// rename so a concurrent peer reads a whole JSON object or none. A write failure is
// swallowed — the cache accelerates, it never gates.
func (c *Cache) Put(src string, keys []string, spans [][2]int) {
	if c == nil || strings.TrimSpace(c.dir) == "" || strings.TrimSpace(c.version) == "" || len(keys) != len(spans) {
		return
	}
	dig := c.digest(src)
	b, err := json.Marshal(entry{Schema: entrySchema, Version: c.version, Digest: dig, Keys: keys, Spans: spans})
	if err != nil {
		return
	}
	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return
	}
	tmp, err := os.CreateTemp(c.dir, ".entry-*.tmp")
	if err != nil {
		return
	}
	name := tmp.Name()
	_, werr := tmp.Write(b)
	cerr := tmp.Close()
	if werr != nil || cerr != nil {
		os.Remove(name)
		return
	}
	p := c.path(dig)
	if err := os.Rename(name, p); err == nil {
		return
	}
	// Windows refuses Rename over an existing destination. Replace the stale entry; if
	// a peer already wrote a valid one, the remove may briefly turn a hit into a miss,
	// which only degrades to recompute.
	_ = os.Remove(p)
	if err := os.Rename(name, p); err != nil {
		os.Remove(name)
	}
}

// maxBytes reads the byte budget, falling back to defaultMaxBytes on unset/invalid.
func maxBytes() int64 {
	v := strings.TrimSpace(os.Getenv(MaxBytesEnv))
	if v == "" {
		return defaultMaxBytes
	}
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil || n <= 0 {
		return defaultMaxBytes
	}
	return n
}

// prune enforces the byte budget FIFO by oldest modification time. Best-effort: any
// error leaves the cache untouched. Run once per Open so the per-invocation cost is a
// single directory scan, not one scan per cached file.
func (c *Cache) prune(budget int64) {
	if c == nil || strings.TrimSpace(c.dir) == "" || budget <= 0 {
		return
	}
	ents, err := os.ReadDir(c.dir)
	if err != nil {
		return
	}
	type fileInfo struct {
		path    string
		size    int64
		modTime int64
	}
	var files []fileInfo
	var total int64
	for _, de := range ents {
		if de.IsDir() || !strings.HasSuffix(de.Name(), ".json") {
			continue
		}
		info, err := de.Info()
		if err != nil {
			continue
		}
		total += info.Size()
		files = append(files, fileInfo{filepath.Join(c.dir, de.Name()), info.Size(), info.ModTime().UnixNano()})
	}
	if total <= budget {
		return
	}
	sort.Slice(files, func(i, j int) bool { return files[i].modTime < files[j].modTime })
	for _, f := range files {
		if total <= budget {
			break
		}
		if os.Remove(f.path) == nil {
			total -= f.size
		}
	}
}
