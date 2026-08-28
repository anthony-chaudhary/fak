package modelperfobs

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

const (
	QwenSwapUsageSchema  = "fak-qwen-swap-usage/1"
	QwenSwapCodecVersion = 1

	QwenSwapDirectionOut    = "swap-out"
	QwenSwapDirectionIn     = "restore-in"
	QwenSwapOutcomeSuccess  = "success"
	QwenSwapOutcomeError    = "error"
	QwenSwapResultCommitted = "committed"
	QwenSwapResultRefused   = "refused"
)

// QwenSwapUsageRow is the privacy-safe durable record of one production codec
// invocation. It deliberately carries no path, host, user, model, prompt, or payload.
type QwenSwapUsageRow struct {
	Schema     string    `json:"schema"`
	ObservedAt time.Time `json:"observed_at"`
	Version    int       `json:"version"`
	Outcome    string    `json:"outcome"`
	Bytes      int64     `json:"bytes"`
	Direction  string    `json:"direction"`
	Result     string    `json:"result"`
}

// QwenSwapWeeklyUsage is the deterministic Monday-UTC fold of codec invocations.
// Succeeded counts only rows whose codec outcome and scheduler result both committed.
type QwenSwapWeeklyUsage struct {
	WeekStart   string `json:"week_start"`
	Invocations int    `json:"invocations"`
	Bytes       int64  `json:"bytes"`
	SwapOut     int    `json:"swap_out"`
	RestoreIn   int    `json:"restore_in"`
	Succeeded   int    `json:"succeeded"`
	Refused     int    `json:"refused"`
	Errors      int    `json:"errors"`
}

var qwenSwapUsageAppendMu sync.Mutex

// AppendQwenSwapUsage appends one validated row to the caller-declared target.
// An empty target is the explicit disabled state and performs no filesystem mutation.
func AppendQwenSwapUsage(path string, row QwenSwapUsageRow) error {
	if path == "" {
		return nil
	}
	if err := validateQwenSwapUsage(row); err != nil {
		return err
	}
	qwenSwapUsageAppendMu.Lock()
	defer qwenSwapUsageAppendMu.Unlock()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create Qwen swap usage ledger directory: %w", err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return fmt.Errorf("open Qwen swap usage ledger: %w", err)
	}
	writeErr := jsonlledger.AppendValidated(f, row, validateQwenSwapUsage)
	if writeErr == nil {
		writeErr = f.Sync()
	}
	closeErr := f.Close()
	if err := errors.Join(writeErr, closeErr); err != nil {
		return fmt.Errorf("append Qwen swap usage ledger: %w", err)
	}
	return nil
}

// FoldQwenSwapUsage reads a strict JSONL ledger and returns Monday-UTC rows in order.
func FoldQwenSwapUsage(path string) ([]QwenSwapWeeklyUsage, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	counts := make(map[string]*QwenSwapWeeklyUsage)
	scanner := bufio.NewScanner(f)
	line := 0
	for scanner.Scan() {
		line++
		var row QwenSwapUsageRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("decode Qwen swap usage row %d: %w", line, err)
		}
		if err := validateQwenSwapUsage(row); err != nil {
			return nil, fmt.Errorf("decode Qwen swap usage row %d: %w", line, err)
		}
		week := mondayUTC(row.ObservedAt).Format("2006-01-02")
		count := counts[week]
		if count == nil {
			count = &QwenSwapWeeklyUsage{WeekStart: week}
			counts[week] = count
		}
		count.Invocations++
		count.Bytes += row.Bytes
		switch row.Direction {
		case QwenSwapDirectionOut:
			count.SwapOut++
		case QwenSwapDirectionIn:
			count.RestoreIn++
		}
		if row.Outcome == QwenSwapOutcomeSuccess && row.Result == QwenSwapResultCommitted {
			count.Succeeded++
		}
		if row.Result == QwenSwapResultRefused {
			count.Refused++
		}
		if row.Outcome == QwenSwapOutcomeError {
			count.Errors++
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read Qwen swap usage ledger: %w", err)
	}
	weeks := make([]string, 0, len(counts))
	for week := range counts {
		weeks = append(weeks, week)
	}
	sort.Strings(weeks)
	result := make([]QwenSwapWeeklyUsage, 0, len(weeks))
	for _, week := range weeks {
		result = append(result, *counts[week])
	}
	return result, nil
}

func validateQwenSwapUsage(row QwenSwapUsageRow) error {
	if row.Schema != QwenSwapUsageSchema || row.Version != QwenSwapCodecVersion || row.ObservedAt.IsZero() {
		return errors.New("invalid Qwen swap usage identity")
	}
	if row.Bytes < 0 {
		return errors.New("invalid Qwen swap usage bytes")
	}
	if row.Direction != QwenSwapDirectionOut && row.Direction != QwenSwapDirectionIn {
		return errors.New("invalid Qwen swap usage direction")
	}
	if row.Outcome != QwenSwapOutcomeSuccess && row.Outcome != QwenSwapOutcomeError {
		return errors.New("invalid Qwen swap usage outcome")
	}
	if row.Result != QwenSwapResultCommitted && row.Result != QwenSwapResultRefused {
		return errors.New("invalid Qwen swap usage result")
	}
	if row.Outcome == QwenSwapOutcomeError && row.Result == QwenSwapResultCommitted {
		return errors.New("Qwen swap error cannot claim committed success")
	}
	return nil
}
