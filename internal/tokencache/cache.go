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
// RETENTION: FAK_TOKEN_CACHE_MAX_BYTES and FAK_TOKEN_CACHE_MAX_ENTRIES bound immutable
// entries; FAK_TOKEN_CACHE_TEMP_GRACE protects active atomic-write temporaries. Open
// performs startup recovery and BuildTreeIndex coalesces one maintenance pass after a
// batch, so sustained writers converge without scanning the directory for every Put.
package tokencache

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/clonescan"
	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// FlagEnv disables the token cache when set to off/0/false/no.
const FlagEnv = "FAK_TOKEN_CACHE"

// MaxBytesEnv overrides the on-disk byte budget (default defaultMaxBytes).
const MaxBytesEnv = "FAK_TOKEN_CACHE_MAX_BYTES"

// MaxEntriesEnv overrides the immutable-entry count ceiling.
const MaxEntriesEnv = "FAK_TOKEN_CACHE_MAX_ENTRIES"

// TempGraceEnv overrides the minimum age for abandoned atomic-write temporaries.
const TempGraceEnv = "FAK_TOKEN_CACHE_TEMP_GRACE"

const entrySchema = "fak-token-window-cache/v1"

const maintenanceReceiptSchema = "fak-token-cache-maintenance/v1"

// defaultMaxBytes bounds the cache dir when FAK_TOKEN_CACHE_MAX_BYTES is unset. Large
// enough that a full ~5.7k-file tree fits many times over; the budget exists to cap
// unbounded growth across invocations, not to evict a working set.
const defaultMaxBytes = 256 << 20 // 256 MiB

const defaultMaxEntries = 10_000

const defaultTempGrace = 24 * time.Hour

const (
	VerdictWithinLimits = "within_limits"
	VerdictPruned       = "pruned"
	VerdictPartial      = "partial"
	VerdictLockBusy     = "skipped_lock_busy"
	VerdictDisabled     = "disabled"
	VerdictUnavailable  = "unavailable"
	VerdictUnsafePath   = "unsafe_path"
	VerdictError        = "error"
)

// MaintenanceOptions is the deterministic retention envelope. Zero-valued fields
// select the documented environment/default value; explicit values must be positive.
type MaintenanceOptions struct {
	MaxBytes   int64         `json:"max_bytes"`
	MaxEntries int           `json:"max_entries"`
	TempGrace  time.Duration `json:"-"`
}

// MaintenanceReceipt is the exact observation made while holding the shared
// maintenance lock. JSON keeps the grace in seconds so receipts are stable and easy to
// consume outside Go.
type MaintenanceReceipt struct {
	Schema                string `json:"schema"`
	MaxBytes              int64  `json:"max_bytes"`
	MaxEntries            int    `json:"max_entries"`
	TempGraceSeconds      int64  `json:"temp_grace_seconds"`
	BeforeBytes           int64  `json:"before_bytes"`
	BeforeEntries         int    `json:"before_entries"`
	AfterBytes            int64  `json:"after_bytes"`
	AfterEntries          int    `json:"after_entries"`
	RemovedBytes          int64  `json:"removed_bytes"`
	RemovedEntries        int    `json:"removed_entries"`
	StaleTempsBefore      int    `json:"stale_temps_before"`
	StaleTempsRemoved     int    `json:"stale_temps_removed"`
	StaleTempBytesRemoved int64  `json:"stale_temp_bytes_removed"`
	StaleTempsAfter       int    `json:"stale_temps_after"`
	SkippedLockedFiles    int    `json:"skipped_locked_files"`
	Complete              bool   `json:"complete"`
	Verdict               string `json:"verdict"`
	Detail                string `json:"detail,omitempty"`
}

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
	dir       string
	version   string
	commonDir string
	dirty     atomic.Bool
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
	c.commonDir = dir
	_ = c.maintain(MaintenanceDefaults(), time.Now(), os.Remove)
	return c
}

// commonDirCmd builds the `git rev-parse --git-common-dir` probe for root with its
// console window suppressed. Split out from commonDir so a regression test can assert
// the suppression (#5129) without spawning git: an unsuppressed spawn here flashes a
// window under background automation and trips the pre-push DESKTOP_POPUP_REGRESSION
// guard, stalling every worker's push.
func commonDirCmd(root string) *exec.Cmd {
	cmd := exec.Command("git", "rev-parse", "--git-common-dir")
	if strings.TrimSpace(root) != "" {
		cmd.Dir = root
	}
	// Suppress the console window: this resolve runs inside the commit hook and under
	// background automation, where a windowless parent would otherwise flash a child.
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd
}

// commonDir resolves the git common dir for root (absolute), or ok=false. The common
// dir is shared across worktrees of the same clone, so the cache is fleet-shared.
func commonDir(root string) (string, bool) {
	cmd := commonDirCmd(root)
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
		c.dirty.Store(true)
		return
	}
	// Windows refuses Rename over an existing destination. Replace the stale entry; if
	// a peer already wrote a valid one, the remove may briefly turn a hit into a miss,
	// which only degrades to recompute.
	_ = os.Remove(p)
	if err := os.Rename(name, p); err != nil {
		os.Remove(name)
	} else {
		c.dirty.Store(true)
	}
}

// MaintenanceDefaults resolves the documented retention settings. Invalid or
// non-positive overrides fail safe to the default instead of disabling a ceiling.
func MaintenanceDefaults() MaintenanceOptions {
	return MaintenanceOptions{
		MaxBytes:   positiveInt64Env(MaxBytesEnv, defaultMaxBytes),
		MaxEntries: positiveIntEnv(MaxEntriesEnv, defaultMaxEntries),
		TempGrace:  positiveDurationEnv(TempGraceEnv, defaultTempGrace),
	}
}

// Maintain resolves the fleet-shared cache for root and runs one best-effort,
// nonblocking retention pass. It refuses a symlink or containment escape before any
// deletion. Disabled/unavailable caches return observable no-op receipts.
func Maintain(root string, opts MaintenanceOptions) MaintenanceReceipt {
	opts = normalizeOptions(opts)
	receipt := newReceipt(opts)
	if !Enabled() {
		receipt.Verdict = VerdictDisabled
		return receipt
	}
	common, ok := commonDir(root)
	if !ok {
		receipt.Verdict = VerdictUnavailable
		return receipt
	}
	c := New(TokenCacheDir(common), clonescan.TokenizerVersion())
	c.commonDir = common
	return c.maintain(opts, time.Now(), os.Remove)
}

// Maintain is the optional BuildTreeIndex lifecycle hook. It deliberately returns no
// error or receipt: cache maintenance accelerates and bounds disk use but never gates
// tokenization. Operators use the package Maintain function for a receipt.
func (c *Cache) Maintain() {
	if c == nil || !c.dirty.Swap(false) {
		return
	}
	receipt := c.maintain(MaintenanceDefaults(), time.Now(), os.Remove)
	if receipt.Verdict == VerdictLockBusy || receipt.Verdict == VerdictPartial || receipt.Verdict == VerdictError {
		c.dirty.Store(true)
	}
}

// prune enforces the byte budget FIFO by oldest modification time. Best-effort: any
// error leaves the cache usable. Kept for the focused byte-bound regression; production
// Open and batch-finalization use the shared maintenance lock and full envelope.
func (c *Cache) prune(budget int64) {
	_ = c.maintain(MaintenanceOptions{MaxBytes: budget, MaxEntries: defaultMaxEntries, TempGrace: defaultTempGrace}, time.Now(), os.Remove)
}

type maintenanceFile struct {
	path    string
	name    string
	size    int64
	modTime time.Time
}

const (
	maintenanceQuietWindow = 5 * time.Millisecond
	maintenanceMaxPasses   = 32
)

func (c *Cache) maintain(opts MaintenanceOptions, now time.Time, remove func(string) error) MaintenanceReceipt {
	opts = normalizeOptions(opts)
	receipt := newReceipt(opts)
	if c == nil || strings.TrimSpace(c.dir) == "" {
		receipt.Verdict = VerdictUnavailable
		return receipt
	}
	if c.commonDir != "" && !maintenancePathSafe(c.commonDir, c.dir) {
		receipt.Verdict = VerdictUnsafePath
		return receipt
	}
	if err := os.MkdirAll(filepath.Dir(c.dir), 0o755); err != nil {
		receipt.Verdict = VerdictError
		receipt.Detail = "open maintenance parent"
		return receipt
	}
	lockPath := c.maintenanceLockPath()
	writeMaintenancePending(lockPath)
	lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		receipt.Verdict = VerdictError
		receipt.Detail = "open maintenance lock"
		return receipt
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		if errors.Is(err, flock.ErrLockBusy) {
			receipt.Verdict = VerdictLockBusy
			return receipt
		}
		receipt.Verdict = VerdictError
		receipt.Detail = "acquire maintenance lock"
		return receipt
	}
	defer flock.Unlock(lock)

	combined := newReceipt(opts)
	for pass := 0; pass < maintenanceMaxPasses; pass++ {
		clearMaintenancePending(lockPath)
		current := c.maintainPass(opts, now, remove)
		mergeMaintenanceReceipt(&combined, current, pass == 0)
		if current.Verdict == VerdictError {
			return combined
		}
		time.Sleep(maintenanceQuietWindow)
		if !maintenancePending(lockPath) {
			finalizeMaintenanceReceipt(&combined, opts)
			return combined
		}
	}
	combined.Complete = false
	combined.Verdict = VerdictPartial
	combined.Detail = "maintenance write burst remained active"
	return combined
}

func (c *Cache) maintainPass(opts MaintenanceOptions, now time.Time, remove func(string) error) MaintenanceReceipt {
	receipt := newReceipt(opts)
	files, staleTemps, err := scanMaintenanceFiles(c.dir, now, opts.TempGrace)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			receipt.Complete = true
			receipt.Verdict = VerdictWithinLimits
			return receipt
		}
		receipt.Verdict = VerdictError
		receipt.Detail = "scan cache"
		return receipt
	}
	for _, f := range files {
		receipt.BeforeBytes += f.size
	}
	receipt.BeforeEntries = len(files)
	receipt.StaleTempsBefore = len(staleTemps)

	for _, f := range staleTemps {
		if err := remove(f.path); err != nil {
			receipt.SkippedLockedFiles++
			continue
		}
		receipt.StaleTempsRemoved++
		receipt.StaleTempBytesRemoved += f.size
	}

	sort.Slice(files, func(i, j int) bool {
		if files[i].modTime.Equal(files[j].modTime) {
			return files[i].name < files[j].name
		}
		return files[i].modTime.Before(files[j].modTime)
	})
	bytesLeft := receipt.BeforeBytes
	entriesLeft := receipt.BeforeEntries
	for _, f := range files {
		if bytesLeft <= opts.MaxBytes && entriesLeft <= opts.MaxEntries {
			break
		}
		if err := remove(f.path); err != nil {
			receipt.SkippedLockedFiles++
			continue
		}
		bytesLeft -= f.size
		entriesLeft--
		receipt.RemovedBytes += f.size
		receipt.RemovedEntries++
	}

	after, staleAfter, err := scanMaintenanceFiles(c.dir, now, opts.TempGrace)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		receipt.Verdict = VerdictError
		receipt.Detail = "rescan cache"
		return receipt
	}
	for _, f := range after {
		receipt.AfterBytes += f.size
	}
	receipt.AfterEntries = len(after)
	receipt.StaleTempsAfter = len(staleAfter)
	receipt.Complete = true
	finalizeMaintenanceReceipt(&receipt, opts)
	return receipt
}

func finalizeMaintenanceReceipt(receipt *MaintenanceReceipt, opts MaintenanceOptions) {
	switch {
	case receipt.SkippedLockedFiles > 0 || receipt.AfterBytes > opts.MaxBytes || receipt.AfterEntries > opts.MaxEntries || receipt.StaleTempsAfter > 0:
		receipt.Verdict = VerdictPartial
	case receipt.RemovedEntries > 0 || receipt.StaleTempsRemoved > 0:
		receipt.Verdict = VerdictPruned
	default:
		receipt.Verdict = VerdictWithinLimits
	}
}

func mergeMaintenanceReceipt(dst *MaintenanceReceipt, src MaintenanceReceipt, first bool) {
	if first {
		dst.BeforeBytes = src.BeforeBytes
		dst.BeforeEntries = src.BeforeEntries
		dst.StaleTempsBefore = src.StaleTempsBefore
	}
	dst.AfterBytes = src.AfterBytes
	dst.AfterEntries = src.AfterEntries
	dst.StaleTempsAfter = src.StaleTempsAfter
	dst.RemovedBytes += src.RemovedBytes
	dst.RemovedEntries += src.RemovedEntries
	dst.StaleTempsRemoved += src.StaleTempsRemoved
	dst.StaleTempBytesRemoved += src.StaleTempBytesRemoved
	dst.SkippedLockedFiles += src.SkippedLockedFiles
	dst.Complete = src.Complete
	dst.Verdict = src.Verdict
	dst.Detail = src.Detail
}

func (c *Cache) maintenanceLockPath() string {
	if c.commonDir != "" {
		return filepath.Join(filepath.Dir(c.dir), "token-cache-maintenance.lock")
	}
	return filepath.Join(c.dir, ".maintenance.lock")
}

func writeMaintenancePending(lockPath string) {
	f, err := os.OpenFile(lockPath+".pending", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err == nil {
		_ = f.Close()
	}
}

func clearMaintenancePending(lockPath string) {
	_ = os.Remove(lockPath + ".pending")
}

func maintenancePending(lockPath string) bool {
	_, err := os.Stat(lockPath + ".pending")
	return err == nil
}

func scanMaintenanceFiles(dir string, now time.Time, grace time.Duration) (entries, staleTemps []maintenanceFile, err error) {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, err
	}
	cutoff := now.Add(-grace)
	for _, de := range ents {
		if de.IsDir() {
			continue
		}
		isEntry := strings.HasSuffix(de.Name(), ".json")
		isTemp := strings.HasPrefix(de.Name(), ".entry-") && strings.HasSuffix(de.Name(), ".tmp")
		if !isEntry && !isTemp {
			continue
		}
		info, err := de.Info()
		if err != nil {
			continue
		}
		f := maintenanceFile{path: filepath.Join(dir, de.Name()), name: de.Name(), size: info.Size(), modTime: info.ModTime()}
		if isEntry {
			entries = append(entries, f)
		} else if info.ModTime().Before(cutoff) {
			staleTemps = append(staleTemps, f)
		}
	}
	return entries, staleTemps, nil
}

func normalizeOptions(opts MaintenanceOptions) MaintenanceOptions {
	defaults := MaintenanceDefaults()
	if opts.MaxBytes <= 0 {
		opts.MaxBytes = defaults.MaxBytes
	}
	if opts.MaxEntries <= 0 {
		opts.MaxEntries = defaults.MaxEntries
	}
	if opts.TempGrace <= 0 {
		opts.TempGrace = defaults.TempGrace
	}
	return opts
}

func newReceipt(opts MaintenanceOptions) MaintenanceReceipt {
	return MaintenanceReceipt{
		Schema:           maintenanceReceiptSchema,
		MaxBytes:         opts.MaxBytes,
		MaxEntries:       opts.MaxEntries,
		TempGraceSeconds: int64(opts.TempGrace / time.Second),
	}
}

func positiveInt64Env(name string, fallback int64) int64 {
	n, err := strconv.ParseInt(strings.TrimSpace(os.Getenv(name)), 10, 64)
	if err != nil || n <= 0 {
		return fallback
	}
	return n
}

func positiveIntEnv(name string, fallback int) int {
	n, err := strconv.Atoi(strings.TrimSpace(os.Getenv(name)))
	if err != nil || n <= 0 {
		return fallback
	}
	return n
}

func positiveDurationEnv(name string, fallback time.Duration) time.Duration {
	d, err := time.ParseDuration(strings.TrimSpace(os.Getenv(name)))
	if err != nil || d <= 0 {
		return fallback
	}
	return d
}

func maintenancePathSafe(common, cache string) bool {
	commonAbs, err := filepath.Abs(common)
	if err != nil {
		return false
	}
	cacheAbs, err := filepath.Abs(cache)
	if err != nil || filepath.Clean(cacheAbs) != filepath.Clean(TokenCacheDir(commonAbs)) || !pathWithin(commonAbs, cacheAbs) {
		return false
	}
	commonReal, err := filepath.EvalSymlinks(commonAbs)
	if err != nil {
		return false
	}
	probe := cacheAbs
	for {
		if _, err := os.Lstat(probe); err == nil {
			break
		} else if !errors.Is(err, fs.ErrNotExist) {
			return false
		}
		parent := filepath.Dir(probe)
		if parent == probe {
			return false
		}
		probe = parent
	}
	probeReal, err := filepath.EvalSymlinks(probe)
	if err != nil {
		return false
	}
	return pathWithin(commonReal, probeReal)
}

func pathWithin(root, target string) bool {
	rel, err := filepath.Rel(root, target)
	if err != nil {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}
