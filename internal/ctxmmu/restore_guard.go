package ctxmmu

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"
)

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
	loadErr error // an unreadable authority refuses generic restore; it never means "no quarantines"
}

// NewQuarantineLedger constructs a new QuarantineLedger. If path is provided,
// existing entries are loaded and subsequent mutations are appended.
func NewQuarantineLedger(path ...string) *QuarantineLedger {
	p := ""
	if len(path) > 0 && path[0] != "" {
		p = path[0]
	} else if envPath := os.Getenv("FAK_QUARANTINE_LEDGER_PATH"); envPath != "" {
		p = envPath
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
		clean := cleanDigest(rec.Digest)
		if strings.TrimSpace(rec.Digest) == "" || clean == "" {
			return fmt.Errorf("quarantine ledger line %d: empty digest", lineNo)
		}
		switch rec.Op {
		case "quarantine":
			l.entries[rec.Digest] = true
			if clean != "" {
				l.entries[clean] = true
				l.entries["sha256:"+clean] = true
			}
		case "clear":
			delete(l.entries, rec.Digest)
			if clean != "" {
				delete(l.entries, clean)
				delete(l.entries, "sha256:"+clean)
			}
		default:
			return fmt.Errorf("quarantine ledger line %d: unknown operation %q", lineNo, rec.Op)
		}
	}
	return scanner.Err()
}

// RecordQuarantine registers a digest as quarantined.
func (l *QuarantineLedger) RecordQuarantine(digest string) {
	if l == nil {
		return
	}
	digest = strings.TrimSpace(digest)
	if digest == "" {
		return
	}
	clean := cleanDigest(digest)
	l.mu.Lock()
	defer l.mu.Unlock()

	l.entries[digest] = true
	if clean != "" {
		l.entries[clean] = true
		l.entries["sha256:"+clean] = true
	}
	if l.path != "" {
		if err := l.appendRecordLocked("quarantine", digest); err != nil {
			l.loadErr = err
		}
	}
}

// ClearQuarantine clears a quarantined digest.
func (l *QuarantineLedger) ClearQuarantine(digest string) {
	if l == nil {
		return
	}
	digest = strings.TrimSpace(digest)
	if digest == "" {
		return
	}
	clean := cleanDigest(digest)
	l.mu.Lock()
	defer l.mu.Unlock()

	delete(l.entries, digest)
	if clean != "" {
		delete(l.entries, clean)
		delete(l.entries, "sha256:"+clean)
	}
	if l.path != "" {
		if err := l.appendRecordLocked("clear", digest); err != nil {
			l.loadErr = err
		}
	}
}

// IsQuarantined checks whether a digest is currently quarantined.
func (l *QuarantineLedger) IsQuarantined(digest string) bool {
	if l == nil {
		return false
	}
	digest = strings.TrimSpace(digest)
	if digest == "" {
		return false
	}
	clean := cleanDigest(digest)

	l.mu.RLock()
	defer l.mu.RUnlock()
	if l.loadErr != nil {
		return true
	}

	if l.entries[digest] {
		return true
	}
	if clean != "" {
		if l.entries[clean] || l.entries["sha256:"+clean] {
			return true
		}
	}
	return false
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

func (l *QuarantineLedger) appendRecordLocked(op, digest string) error {
	rec := quarantineRecord{
		Op:     op,
		Digest: digest,
		Time:   time.Now().UnixNano(),
	}
	b, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	if _, err := f.Write(append(b, '\n')); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
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
func RecordQuarantine(digest string) {
	DefaultQuarantineLedger().RecordQuarantine(digest)
}

// ClearQuarantine clears a quarantined digest from the default ledger.
func ClearQuarantine(digest string) {
	DefaultQuarantineLedger().ClearQuarantine(digest)
}

// IsQuarantined reports whether a digest is quarantined in the default ledger.
func IsQuarantined(digest string) bool {
	return DefaultQuarantineLedger().IsQuarantined(digest)
}
