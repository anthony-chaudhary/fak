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
	quarantineLedgerEnv          = "FAK_QUARANTINE_LEDGER_PATH"
	quarantineLedgerDefaultPath  = ".fak/ctxmmu/quarantine.jsonl"
	maxQuarantineLedgerEntries   = 32768
	maxQuarantineLedgerWALBytes  = 8 << 20
	maxQuarantineLedgerKeyLength = 128
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
	mu      sync.RWMutex
	path    string
	entries map[string]bool
	walSize int64
	loadErr error // an unreadable authority refuses generic restore; it never means "no quarantines"
}

// NewQuarantineLedger constructs a new QuarantineLedger. If path is provided,
// existing entries are loaded and subsequent mutations are appended.
func NewQuarantineLedger(path ...string) *QuarantineLedger {
	p := quarantineLedgerDefaultPath
	if len(path) > 0 && path[0] != "" {
		p = path[0]
	} else if envPath, ok := os.LookupEnv(quarantineLedgerEnv); ok {
		switch strings.ToLower(strings.TrimSpace(envPath)) {
		case "off", "0", "none":
			// Generic restore has multiple durable sources (gateway CAS, page-out
			// codecs, active resolvers, and caller-named images). No single process
			// env switch proves all are absent, so the authority remains mandatory.
		case "":
			// Empty is the default, not an accidental disable.
		default:
			p = envPath
		}
	}
	l := &QuarantineLedger{
		path:    p,
		entries: make(map[string]bool),
	}
	if p != "" {
		l.loadErr = l.load()
	}
	return l
}

func (l *QuarantineLedger) load() error {
	if l.path == "" {
		return nil
	}
	f, err := os.Open(l.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()
	if info, err := f.Stat(); err == nil {
		l.walSize = info.Size()
	}

	scanner := bufio.NewScanner(f)
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
			if !l.entries[key] && len(l.entries) >= maxQuarantineLedgerEntries {
				clear(l.entries)
				return ErrQuarantineLedgerCapacity
			}
			l.entries[key] = true
		case "clear":
			delete(l.entries, key)
		default:
			return fmt.Errorf("quarantine ledger line %d: unknown operation %q", lineNo, rec.Op)
		}
	}
	return scanner.Err()
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
	if l.loadErr != nil {
		return l.loadErr
	}
	if l.entries[key] {
		return nil
	}
	if len(l.entries) >= maxQuarantineLedgerEntries {
		return ErrQuarantineLedgerCapacity
	}
	if l.path != "" {
		rec, err := marshalQuarantineRecord("quarantine", key)
		if err != nil {
			return err
		}
		if l.walSize+int64(len(rec)) > maxQuarantineLedgerWALBytes {
			l.entries[key] = true
			err = l.rewriteSnapshotLocked()
			if err != nil {
				delete(l.entries, key)
				l.loadErr = err
				return err
			}
		} else if err = l.appendBytesLocked(rec); err != nil {
			l.loadErr = err
			return err
		}
	}
	l.entries[key] = true
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
	if l.loadErr != nil {
		return l.loadErr
	}
	if !l.entries[key] {
		return nil
	}
	if l.path != "" {
		rec, err := marshalQuarantineRecord("clear", key)
		if err != nil {
			return err
		}
		if l.walSize+int64(len(rec)) > maxQuarantineLedgerWALBytes {
			delete(l.entries, key)
			err = l.rewriteSnapshotLocked()
			if err != nil {
				l.entries[key] = true
				l.loadErr = err
				return err
			}
		} else if err = l.appendBytesLocked(rec); err != nil {
			l.loadErr = err
			return err
		}
	}
	delete(l.entries, key)
	return nil
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

	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.loadErr != nil {
		return true
	}

	return l.entries[key]
}

// Len reports the count of currently quarantined entries.
func (l *QuarantineLedger) Len() int {
	if l == nil {
		return 0
	}
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

func marshalQuarantineRecord(op, digest string) ([]byte, error) {
	rec := quarantineRecord{
		Op:     op,
		Digest: digest,
		Time:   time.Now().UnixNano(),
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
		rec, err := marshalQuarantineRecord("quarantine", key)
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
	defaultQuarantineLedger   = newDefaultQuarantineLedger()
)

func newDefaultQuarantineLedger() *QuarantineLedger {
	return NewQuarantineLedger()
}

// DefaultQuarantineLedger returns the process-default QuarantineLedger.
func DefaultQuarantineLedger() *QuarantineLedger {
	defaultQuarantineLedgerMu.RLock()
	defer defaultQuarantineLedgerMu.RUnlock()
	return defaultQuarantineLedger
}

// SetDefaultQuarantineLedger overrides the process-default QuarantineLedger.
func SetDefaultQuarantineLedger(l *QuarantineLedger) {
	defaultQuarantineLedgerMu.Lock()
	defer defaultQuarantineLedgerMu.Unlock()
	defaultQuarantineLedger = l
}

// ResetQuarantineLedgerForTest resets the process-default QuarantineLedger.
func ResetQuarantineLedgerForTest() {
	defaultQuarantineLedgerMu.Lock()
	defer defaultQuarantineLedgerMu.Unlock()
	defaultQuarantineLedger = newDefaultQuarantineLedger()
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
