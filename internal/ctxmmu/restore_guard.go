package ctxmmu

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	quarantineLedgerEnv           = "FAK_QUARANTINE_LEDGER_PATH"
	quarantineLedgerStateDirEnv   = "FAK_STATE_DIR"
	maxQuarantineLedgerEntries    = 32768
	quarantineLedgerSoftCap       = 28672 // proactively compact before hitting hard cap
	quarantineLedgerCompactTarget = 24576 // retain freshest 75% after compaction
	quarantineLedgerDefaultTTL    = 7 * 24 * time.Hour
	maxQuarantineLedgerWALBytes   = 8 << 20
	maxQuarantineLedgerKeyLength  = 128
	quarantineLockTimeout         = 10 * time.Second
	quarantineStaleLockTTL        = 30 * time.Second
)

var ErrQuarantineLedgerCapacity = errors.New("ctxmmu: quarantine authority capacity exhausted")

type quarantineRecord struct {
	Op     string `json:"op"`
	Digest string `json:"digest"`
	Time   int64  `json:"time"`
}

// QuarantineLedger tracks quarantined digests in memory and optionally persists
// refusal entries to disk/store so refusal survives process restarts (#12055, #12057).
type QuarantineLedger struct {
	mu            sync.RWMutex
	path          string
	entries       map[string]int64 // digest -> UnixNano timestamp
	walSize       int64
	fileModTime   time.Time
	fileSize      int64
	fileExists    bool
	loadErr       error // an unreadable authority refuses generic restore; it never means "no quarantines"
	maxEntries    int
	softCap       int
	compactTarget int
	ttl           time.Duration
	nowFn         func() time.Time
}

func isGoTestBinary() bool {
	if len(os.Args) == 0 {
		return false
	}
	base := strings.ToLower(filepath.Base(os.Args[0]))
	return strings.HasSuffix(base, ".test") || strings.HasSuffix(base, ".test.exe") || strings.Contains(base, ".test.")
}

// defaultQuarantineLedgerPath returns an explicit state-directory path (never
// CWD-relative) so importing or running in arbitrary working directories cannot
// leak .fak/ctxmmu/quarantine.jsonl into the working tree (#14521).
func defaultQuarantineLedgerPath() string {
	if isGoTestBinary() {
		return defaultQuarantineLedgerPathForTest
	}
	if stateDir := strings.TrimSpace(os.Getenv(quarantineLedgerStateDirEnv)); stateDir != "" {
		return filepath.Join(stateDir, "ctxmmu", "quarantine.jsonl")
	}
	if xdgState := strings.TrimSpace(os.Getenv("XDG_STATE_HOME")); xdgState != "" {
		return filepath.Join(xdgState, "fak", "ctxmmu", "quarantine.jsonl")
	}
	if cacheDir, err := os.UserCacheDir(); err == nil && strings.TrimSpace(cacheDir) != "" {
		return filepath.Join(cacheDir, "fak", "ctxmmu", "quarantine.jsonl")
	}
	return filepath.Join(os.TempDir(), "fak", "ctxmmu", "quarantine.jsonl")
}

// defaultQuarantineLedgerPathForTest stays empty so test binaries persist
// nothing unless a test opts in to exercise the durable default.
var defaultQuarantineLedgerPathForTest string

// SetDefaultQuarantineLedgerPathForTest sets the default ledger path a test
// binary resolves when no explicit path is configured.
func SetDefaultQuarantineLedgerPathForTest(path string) {
	defaultQuarantineLedgerPathForTest = path
}

// NewQuarantineLedger constructs a new QuarantineLedger. If path is provided,
// existing entries are loaded and subsequent mutations are appended.
func NewQuarantineLedger(path ...string) *QuarantineLedger {
	p := ""
	if len(path) > 0 && path[0] != "" {
		p = path[0]
	} else if envPath, ok := os.LookupEnv(quarantineLedgerEnv); ok {
		switch strings.ToLower(strings.TrimSpace(envPath)) {
		case "off", "0", "none":
			// Generic restore has multiple durable sources (gateway CAS, page-out
			// codecs, active resolvers, and caller-named images). No single process
			// env switch proves all are absent, so the authority remains mandatory.
			p = defaultQuarantineLedgerPath()
		case "":
			// Empty is the default, not an accidental disable.
			p = defaultQuarantineLedgerPath()
		default:
			p = envPath
		}
	} else {
		p = defaultQuarantineLedgerPath()
	}
	l := &QuarantineLedger{
		path:          p,
		entries:       make(map[string]int64),
		maxEntries:    maxQuarantineLedgerEntries,
		softCap:       quarantineLedgerSoftCap,
		compactTarget: quarantineLedgerCompactTarget,
		ttl:           quarantineLedgerDefaultTTL,
		nowFn:         time.Now,
	}
	if p != "" {
		l.loadErr = l.loadInitial()
	}
	return l
}

// Path returns the configured WAL file path (empty for in-memory ledgers).
func (l *QuarantineLedger) Path() string {
	if l == nil {
		return ""
	}
	return l.path
}

func (l *QuarantineLedger) now() time.Time {
	if l.nowFn != nil {
		return l.nowFn()
	}
	return time.Now()
}

func (l *QuarantineLedger) loadInitial() error {
	if l.path == "" {
		return nil
	}
	// Do not create directories or lockfiles when the ledger file does not yet exist.
	if _, err := os.Stat(l.path); err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	unlock, err := l.acquireFileLock()
	if err != nil {
		return err
	}
	defer unlock()
	return l.loadLocked()
}

func (l *QuarantineLedger) loadLocked() error {
	if l.path == "" {
		return nil
	}
	f, err := os.Open(l.path)
	if err != nil {
		if os.IsNotExist(err) {
			clear(l.entries)
			l.walSize = 0
			l.fileSize = 0
			l.fileModTime = time.Time{}
			l.fileExists = false
			return nil
		}
		return err
	}
	defer f.Close()
	info, statErr := f.Stat()
	if statErr != nil {
		return statErr
	}

	next := make(map[string]int64)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var rec quarantineRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return fmt.Errorf("quarantine ledger line %d: %w", lineNo, err)
		}
		key, err := quarantineLedgerKey(rec.Digest)
		if err != nil {
			return fmt.Errorf("quarantine ledger line %d: %w", lineNo, err)
		}
		switch rec.Op {
		case "quarantine":
			next[key] = rec.Time
		case "clear":
			delete(next, key)
		default:
			return fmt.Errorf("quarantine ledger line %d: unknown operation %q", lineNo, rec.Op)
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	l.entries = next
	l.walSize = info.Size()
	l.fileSize = info.Size()
	l.fileModTime = info.ModTime()
	l.fileExists = true
	if len(l.entries) >= l.maxEntries {
		l.enforceRetentionLocked(l.now().UnixNano(), l.effectiveCompactTarget())
	}
	return nil
}

func (l *QuarantineLedger) needsReloadLocked() bool {
	if l.path == "" {
		return false
	}
	info, err := os.Stat(l.path)
	if err != nil {
		if os.IsNotExist(err) {
			return l.fileExists
		}
		return true
	}
	return !l.fileExists || info.Size() != l.fileSize || !info.ModTime().Equal(l.fileModTime)
}

func (l *QuarantineLedger) syncFileMetaLocked() {
	if l.path == "" {
		return
	}
	if info, err := os.Stat(l.path); err == nil {
		l.fileSize = info.Size()
		l.fileModTime = info.ModTime()
		l.fileExists = true
		l.walSize = info.Size()
	}
}

func (l *QuarantineLedger) acquireFileLock() (func(), error) {
	dir := filepath.Dir(l.path)
	if err := ensureQuarantineLedgerDir(dir); err != nil {
		return nil, err
	}
	lockPath := l.path + ".lock"
	deadline := time.Now().Add(quarantineLockTimeout)
	backoff := time.Millisecond
	for {
		f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
		if err == nil {
			_, _ = fmt.Fprintf(f, "%d %d\n", os.Getpid(), time.Now().UnixNano())
			_ = f.Close()
			return func() {
				_ = os.Remove(lockPath)
			}, nil
		}
		if !os.IsExist(err) && !os.IsPermission(err) {
			return nil, fmt.Errorf("acquire quarantine ledger lock %s: %w", lockPath, err)
		}
		if os.IsExist(err) {
			if info, statErr := os.Stat(lockPath); statErr == nil && time.Since(info.ModTime()) > quarantineStaleLockTTL {
				_ = os.Remove(lockPath)
				continue
			}
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("timeout acquiring quarantine ledger lock %s: %w", lockPath, err)
		}
		time.Sleep(backoff)
		if backoff < 25*time.Millisecond {
			backoff *= 2
			if backoff > 25*time.Millisecond {
				backoff = 25 * time.Millisecond
			}
		}
	}
}

func (l *QuarantineLedger) effectiveCompactTarget() int {
	target := l.compactTarget
	if target <= 0 || target >= l.maxEntries {
		target = l.maxEntries * 3 / 4
	}
	if target >= l.maxEntries {
		target = l.maxEntries - 1
	}
	if target < 0 {
		target = 0
	}
	return target
}

func (l *QuarantineLedger) effectiveSoftCap() int {
	soft := l.softCap
	if soft <= 0 || soft > l.maxEntries {
		soft = l.maxEntries
	}
	return soft
}

func (l *QuarantineLedger) pruneExpiredLocked(now int64) bool {
	if l.ttl <= 0 {
		return false
	}
	cutoff := now - int64(l.ttl)
	pruned := false
	for k, ts := range l.entries {
		if ts > 0 && ts < cutoff {
			delete(l.entries, k)
			pruned = true
		}
	}
	return pruned
}

func (l *QuarantineLedger) enforceRetentionLocked(now int64, target int) bool {
	changed := l.pruneExpiredLocked(now)
	if len(l.entries) <= target {
		return changed
	}
	type entryAge struct {
		key string
		ts  int64
	}
	items := make([]entryAge, 0, len(l.entries))
	for k, ts := range l.entries {
		items = append(items, entryAge{key: k, ts: ts})
	}
	sort.Slice(items, func(i, j int) bool {
		if items[i].ts != items[j].ts {
			return items[i].ts < items[j].ts
		}
		return items[i].key < items[j].key
	})
	drop := len(l.entries) - target
	for i := 0; i < drop; i++ {
		delete(l.entries, items[i].key)
	}
	return true
}

// RecordQuarantine registers a digest as quarantined.
func (l *QuarantineLedger) RecordQuarantine(digest string) error {
	if l == nil {
		return errors.New("ctxmmu: nil quarantine authority")
	}
	key, err := quarantineLedgerKey(digest)
	if err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.path != "" {
		unlock, err := l.acquireFileLock()
		if err != nil {
			l.loadErr = err
			return err
		}
		defer unlock()
		if l.needsReloadLocked() {
			if err := l.loadLocked(); err != nil {
				l.loadErr = err
				return err
			}
			l.loadErr = nil
		}
	}
	if l.loadErr != nil {
		return l.loadErr
	}
	if _, exists := l.entries[key]; exists {
		return nil
	}
	now := l.now().UnixNano()
	needCompact := len(l.entries)+1 >= l.effectiveSoftCap() || len(l.entries)+1 >= l.maxEntries
	if needCompact {
		target := l.effectiveCompactTarget()
		if target >= l.maxEntries {
			target = l.maxEntries - 1
		}
		l.enforceRetentionLocked(now, target)
	}
	if l.path != "" {
		rec, err := marshalQuarantineRecordAt("quarantine", key, now)
		if err != nil {
			return err
		}
		if needCompact || l.walSize+int64(len(rec)) > maxQuarantineLedgerWALBytes {
			l.pruneExpiredLocked(now)
			l.entries[key] = now
			err = l.rewriteSnapshotLocked()
			if err != nil {
				delete(l.entries, key)
				l.loadErr = err
				return err
			}
			l.syncFileMetaLocked()
			return nil
		}
		if err = l.appendBytesLocked(rec); err != nil {
			l.loadErr = err
			return err
		}
		l.syncFileMetaLocked()
	}
	l.entries[key] = now
	return nil
}

// ClearQuarantine clears a quarantined digest.
func (l *QuarantineLedger) ClearQuarantine(digest string) error {
	if l == nil {
		return errors.New("ctxmmu: nil quarantine authority")
	}
	key, err := quarantineLedgerKey(digest)
	if err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.path != "" {
		unlock, err := l.acquireFileLock()
		if err != nil {
			l.loadErr = err
			return err
		}
		defer unlock()
		if l.needsReloadLocked() {
			if err := l.loadLocked(); err != nil {
				l.loadErr = err
				return err
			}
			l.loadErr = nil
		}
	}
	if l.loadErr != nil {
		return l.loadErr
	}
	prevTS, exists := l.entries[key]
	if !exists {
		return nil
	}
	now := l.now().UnixNano()
	if l.path != "" {
		rec, err := marshalQuarantineRecordAt("clear", key, now)
		if err != nil {
			return err
		}
		if l.walSize+int64(len(rec)) > maxQuarantineLedgerWALBytes {
			delete(l.entries, key)
			err = l.rewriteSnapshotLocked()
			if err != nil {
				l.entries[key] = prevTS
				l.loadErr = err
				return err
			}
			l.syncFileMetaLocked()
			return nil
		}
		if err = l.appendBytesLocked(rec); err != nil {
			l.loadErr = err
			return err
		}
		l.syncFileMetaLocked()
	}
	delete(l.entries, key)
	return nil
}

func (l *QuarantineLedger) refreshForRead() {
	if l.path == "" {
		return
	}
	l.mu.RLock()
	need := l.needsReloadLocked()
	l.mu.RUnlock()
	if !need {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if !l.needsReloadLocked() {
		return
	}
	if _, err := os.Stat(l.path); err == nil {
		unlock, lockErr := l.acquireFileLock()
		if lockErr != nil {
			l.loadErr = lockErr
			return
		}
		defer unlock()
	}
	if err := l.loadLocked(); err != nil {
		l.loadErr = err
	} else {
		l.loadErr = nil
	}
}

// IsQuarantined checks whether a digest is currently quarantined.
func (l *QuarantineLedger) IsQuarantined(digest string) bool {
	if l == nil {
		return false
	}
	key, err := quarantineLedgerKey(digest)
	if err != nil {
		return false
	}
	l.refreshForRead()

	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.loadErr != nil {
		return true
	}

	_, ok := l.entries[key]
	return ok
}

// Len reports the count of currently quarantined entries.
func (l *QuarantineLedger) Len() int {
	if l == nil {
		return 0
	}
	l.refreshForRead()
	l.mu.RLock()
	defer l.mu.RUnlock()
	return len(l.entries)
}

func quarantineLedgerKey(digest string) (string, error) {
	key := cleanDigest(digest)
	if key == "" {
		return "", errors.New("ctxmmu: empty quarantine digest")
	}
	if len(key) > maxQuarantineLedgerKeyLength {
		return "", errors.New("ctxmmu: quarantine digest too long")
	}
	return key, nil
}

func marshalQuarantineRecordAt(op, digest string, ts int64) ([]byte, error) {
	rec := quarantineRecord{
		Op:     op,
		Digest: digest,
		Time:   ts,
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

func (l *QuarantineLedger) appendBytesLocked(record []byte) error {
	dir := filepath.Dir(l.path)
	if err := ensureQuarantineLedgerDir(dir); err != nil {
		return err
	}
	_, statErr := os.Stat(l.path)
	created := os.IsNotExist(statErr)
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(record); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if created {
		if err := syncQuarantineLedgerCreation(dir); err != nil {
			return err
		}
	}
	l.walSize += int64(len(record))
	return nil
}

func (l *QuarantineLedger) rewriteSnapshotLocked() error {
	if err := ensureQuarantineLedgerDir(filepath.Dir(l.path)); err != nil {
		return err
	}
	keys := make([]string, 0, len(l.entries))
	for key := range l.entries {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	tmp, err := os.CreateTemp(filepath.Dir(l.path), ".quarantine-snapshot-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	ok := false
	defer func() {
		_ = tmp.Close()
		if !ok {
			_ = os.Remove(tmpName)
		}
	}()
	var size int64
	for _, key := range keys {
		ts := l.entries[key]
		if ts == 0 {
			ts = l.now().UnixNano()
		}
		rec, err := marshalQuarantineRecordAt("quarantine", key, ts)
		if err != nil {
			return err
		}
		if _, err := tmp.Write(rec); err != nil {
			return err
		}
		size += int64(len(rec))
	}
	if err := tmp.Sync(); err != nil {
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := replaceQuarantineLedgerFile(tmpName, l.path); err != nil {
		return err
	}
	ok = true
	l.walSize = size
	return nil
}

// ensureQuarantineLedgerDir makes every missing directory component durable
// before any quarantine bytes can be published. Syncing only the leaf would
// still allow a crash to forget a newly-created ancestor such as .fak.
func ensureQuarantineLedgerDir(dir string) error {
	dir = filepath.Clean(dir)
	missing := make([]string, 0, 2)
	for current := dir; ; current = filepath.Dir(current) {
		if _, err := os.Stat(current); err == nil {
			break
		} else if !os.IsNotExist(err) {
			return err
		}
		missing = append(missing, current)
		parent := filepath.Dir(current)
		if parent == current {
			return fmt.Errorf("ctxmmu: no existing parent for quarantine authority %s", dir)
		}
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	for i := len(missing) - 1; i >= 0; i-- {
		if err := syncQuarantineLedgerCreation(filepath.Dir(missing[i])); err != nil {
			return err
		}
	}
	return nil
}

var (
	defaultQuarantineLedgerMu sync.RWMutex
	defaultQuarantineLedger   *QuarantineLedger // lazily initialized on first use (#14521)
)

func newDefaultQuarantineLedger() *QuarantineLedger {
	return NewQuarantineLedger()
}

// DefaultQuarantineLedger returns the process-default QuarantineLedger,
// lazily initializing it on first use to avoid init()-time filesystem I/O.
func DefaultQuarantineLedger() *QuarantineLedger {
	defaultQuarantineLedgerMu.RLock()
	l := defaultQuarantineLedger
	defaultQuarantineLedgerMu.RUnlock()
	if l != nil {
		return l
	}
	defaultQuarantineLedgerMu.Lock()
	defer defaultQuarantineLedgerMu.Unlock()
	if defaultQuarantineLedger == nil {
		defaultQuarantineLedger = newDefaultQuarantineLedger()
	}
	return defaultQuarantineLedger
}

// SetDefaultQuarantineLedger overrides the process-default QuarantineLedger.
func SetDefaultQuarantineLedger(l *QuarantineLedger) {
	defaultQuarantineLedgerMu.Lock()
	defer defaultQuarantineLedgerMu.Unlock()
	defaultQuarantineLedger = l
}

// ResetQuarantineLedgerForTest resets the process-default QuarantineLedger so
// the next access re-initializes it lazily from the current environment.
func ResetQuarantineLedgerForTest() {
	defaultQuarantineLedgerMu.Lock()
	defer defaultQuarantineLedgerMu.Unlock()
	defaultQuarantineLedger = nil
}

// RecordQuarantine records a quarantined digest in the default ledger.
func RecordQuarantine(digest string) error {
	return DefaultQuarantineLedger().RecordQuarantine(digest)
}

// ClearQuarantine clears a quarantined digest from the default ledger.
func ClearQuarantine(digest string) error {
	return DefaultQuarantineLedger().ClearQuarantine(digest)
}

// IsQuarantined reports whether a digest is quarantined in the default ledger.
func IsQuarantined(digest string) bool {
	return DefaultQuarantineLedger().IsQuarantined(digest)
}
