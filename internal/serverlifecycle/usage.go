package serverlifecycle

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

const (
	UsageLedgerSchema   = "fak.server-lifecycle-usage/v1"
	UsageLedgerFilename = "server-lifecycle-usage.jsonl"
)

// UsageRow is the privacy-safe durable record of one lifecycle invocation.
// It intentionally excludes instance paths, process IDs, hostnames, and model names.
type UsageRow struct {
	Schema     string    `json:"schema"`
	ObservedAt time.Time `json:"observed_at"`
	Operation  string    `json:"operation"`
	Outcome    string    `json:"outcome"`
	State      State     `json:"state,omitempty"`
	Reason     string    `json:"reason,omitempty"`
}

// WeeklyUsage is one deterministic ISO-week fold of lifecycle usage.
type WeeklyUsage struct {
	Week       string         `json:"week"`
	Total      int            `json:"total"`
	Operations map[string]int `json:"operations"`
	Outcomes   map[string]int `json:"outcomes"`
}

var usageAppendMu sync.Mutex

func recordInvocation(dir, operation string, result Result, invocationErr error) error {
	if dir == "" {
		return nil
	}
	outcome := "ok"
	if invocationErr != nil {
		outcome = "error"
		var refusal *RefusalError
		if errors.As(invocationErr, &refusal) {
			outcome = "refused"
		}
	}
	row := UsageRow{
		Schema: UsageLedgerSchema, ObservedAt: time.Now().UTC(), Operation: operation,
		Outcome: outcome, State: result.State, Reason: result.Reason,
	}
	payload, err := json.Marshal(row)
	if err != nil {
		return fmt.Errorf("encode lifecycle usage: %w", err)
	}
	payload = append(payload, '\n')

	usageAppendMu.Lock()
	defer usageAppendMu.Unlock()
	file, err := os.OpenFile(usageLedgerPath(dir), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("open lifecycle usage ledger: %w", err)
	}
	if _, err = file.Write(payload); err == nil {
		err = file.Sync()
	}
	closeErr := file.Close()
	if err != nil {
		return fmt.Errorf("append lifecycle usage ledger: %w", err)
	}
	if closeErr != nil {
		return fmt.Errorf("close lifecycle usage ledger: %w", closeErr)
	}
	return nil
}

func usageLedgerPath(dir string) string { return filepathJoin(dir, UsageLedgerFilename) }

// FoldUsage reads JSONL usage rows and returns counts ordered by ISO week.
func FoldUsage(r io.Reader) ([]WeeklyUsage, error) {
	weeks := make(map[string]*WeeklyUsage)
	scanner := bufio.NewScanner(r)
	line := 0
	for scanner.Scan() {
		line++
		var row UsageRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("decode lifecycle usage line %d: %w", line, err)
		}
		if row.Schema != UsageLedgerSchema || row.ObservedAt.IsZero() || row.Operation == "" || row.Outcome == "" {
			return nil, fmt.Errorf("decode lifecycle usage line %d: invalid row", line)
		}
		year, week := row.ObservedAt.ISOWeek()
		key := fmt.Sprintf("%04d-W%02d", year, week)
		fold := weeks[key]
		if fold == nil {
			fold = &WeeklyUsage{Week: key, Operations: make(map[string]int), Outcomes: make(map[string]int)}
			weeks[key] = fold
		}
		fold.Total++
		fold.Operations[row.Operation]++
		fold.Outcomes[row.Outcome]++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read lifecycle usage: %w", err)
	}
	keys := make([]string, 0, len(weeks))
	for key := range weeks {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	folds := make([]WeeklyUsage, 0, len(keys))
	for _, key := range keys {
		folds = append(folds, *weeks[key])
	}
	return folds, nil
}

// filepathJoin is a seam kept local so usage rows never need to retain the path.
func filepathJoin(dir, name string) string { return filepath.Join(dir, name) }
