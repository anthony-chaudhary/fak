package trajectory

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

const (
	AuditSnapshotUsageSchema     = "fak-trajectory-audit-snapshot-usage/1"
	AuditSnapshotUsageFoldSchema = "fak-trajectory-audit-snapshot-usage-fold/1"
	auditSnapshotUsageLockWait   = 2 * time.Second
)

// AuditSnapshotUsageRow is one deliberately unlinkable invocation record. Its
// closed fields cannot carry paths, hostnames, transcript identity, or hashes.
type AuditSnapshotUsageRow struct {
	Schema     string    `json:"schema"`
	ObservedAt time.Time `json:"observed_at"`
	Operation  string    `json:"operation"`
	Outcome    string    `json:"outcome"`
	Reason     string    `json:"reason,omitempty"`
}

// AuditSnapshotUsageWeek is the deterministic ISO-week adoption fold.
type AuditSnapshotUsageWeek struct {
	Week       string         `json:"week"`
	Total      int            `json:"total"`
	Operations map[string]int `json:"operations"`
	Outcomes   map[string]int `json:"outcomes"`
}

type AuditSnapshotUsageFold struct {
	Schema string                   `json:"schema"`
	Weeks  []AuditSnapshotUsageWeek `json:"weeks"`
}

var auditSnapshotUsageAppendMu sync.Mutex

// AppendAuditSnapshotUsage appends exactly one 0600 JSONL row under an advisory
// lock on the explicit ledger file. It never creates a parent or sidecar path.
func AppendAuditSnapshotUsage(path string, row AuditSnapshotUsageRow) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("trajectory audit snapshot usage: ledger path is required")
	}
	if row.Schema == "" {
		row.Schema = AuditSnapshotUsageSchema
	}
	if err := validateAuditSnapshotUsageRow(row); err != nil {
		return err
	}
	payload, err := json.Marshal(row)
	if err != nil {
		return fmt.Errorf("trajectory audit snapshot usage: encode row: %w", err)
	}
	payload = append(payload, '\n')

	auditSnapshotUsageAppendMu.Lock()
	defer auditSnapshotUsageAppendMu.Unlock()
	if info, err := os.Lstat(path); err == nil {
		if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("trajectory audit snapshot usage: ledger must be a regular file")
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("trajectory audit snapshot usage: inspect ledger: %w", err)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("trajectory audit snapshot usage: open ledger: %w", err)
	}
	defer file.Close()
	if err := file.Chmod(0o600); err != nil {
		return fmt.Errorf("trajectory audit snapshot usage: restrict ledger: %w", err)
	}
	deadline := time.Now().Add(auditSnapshotUsageLockWait)
	for {
		err = flock.TryLock(file)
		if err == nil {
			break
		}
		if !errors.Is(err, flock.ErrLockBusy) {
			return fmt.Errorf("trajectory audit snapshot usage: lock ledger: %w", err)
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("trajectory audit snapshot usage: ledger lock busy")
		}
		time.Sleep(10 * time.Millisecond)
	}
	defer flock.Unlock(file) // best effort; close also releases the advisory lock
	if _, err := file.Write(payload); err != nil {
		return fmt.Errorf("trajectory audit snapshot usage: append ledger: %w", err)
	}
	if err := file.Sync(); err != nil {
		return fmt.Errorf("trajectory audit snapshot usage: sync ledger: %w", err)
	}
	return nil
}

// FoldAuditSnapshotUsage reads strict JSONL and returns ISO weeks in ascending order.
func FoldAuditSnapshotUsage(r io.Reader) ([]AuditSnapshotUsageWeek, error) {
	weeks := make(map[string]*AuditSnapshotUsageWeek)
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	line := 0
	for scanner.Scan() {
		line++
		var row AuditSnapshotUsageRow
		decoder := json.NewDecoder(strings.NewReader(scanner.Text()))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&row); err != nil {
			return nil, fmt.Errorf("trajectory audit snapshot usage: decode line %d: %w", line, err)
		}
		if err := validateAuditSnapshotUsageRow(row); err != nil {
			return nil, fmt.Errorf("trajectory audit snapshot usage: decode line %d: %w", line, err)
		}
		year, week := row.ObservedAt.UTC().ISOWeek()
		key := fmt.Sprintf("%04d-W%02d", year, week)
		fold := weeks[key]
		if fold == nil {
			fold = &AuditSnapshotUsageWeek{Week: key, Operations: make(map[string]int), Outcomes: make(map[string]int)}
			weeks[key] = fold
		}
		fold.Total++
		fold.Operations[row.Operation]++
		fold.Outcomes[row.Outcome]++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("trajectory audit snapshot usage: read ledger: %w", err)
	}
	keys := make([]string, 0, len(weeks))
	for key := range weeks {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]AuditSnapshotUsageWeek, 0, len(keys))
	for _, key := range keys {
		result = append(result, *weeks[key])
	}
	return result, nil
}

func ReadAuditSnapshotUsage(path string) ([]AuditSnapshotUsageWeek, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("trajectory audit snapshot usage: open ledger: %w", err)
	}
	defer file.Close()
	return FoldAuditSnapshotUsage(file)
}

func WriteAuditSnapshotUsageFold(w io.Writer, weeks []AuditSnapshotUsageWeek) error {
	encoder := json.NewEncoder(w)
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(AuditSnapshotUsageFold{Schema: AuditSnapshotUsageFoldSchema, Weeks: weeks}); err != nil {
		return fmt.Errorf("trajectory audit snapshot usage: encode fold: %w", err)
	}
	return nil
}

func validateAuditSnapshotUsageRow(row AuditSnapshotUsageRow) error {
	if row.Schema != AuditSnapshotUsageSchema || row.ObservedAt.IsZero() || !row.ObservedAt.Equal(row.ObservedAt.UTC()) {
		return fmt.Errorf("invalid row identity")
	}
	if row.Operation != "capture" && row.Operation != "replay" {
		return fmt.Errorf("invalid operation")
	}
	if row.Outcome != "success" && row.Outcome != "refused" && row.Outcome != "error" {
		return fmt.Errorf("invalid outcome")
	}
	if len(row.Reason) > 64 {
		return fmt.Errorf("invalid reason")
	}
	for _, char := range row.Reason {
		if (char < 'A' || char > 'Z') && (char < '0' || char > '9') && char != '_' {
			return fmt.Errorf("invalid reason")
		}
	}
	return nil
}
