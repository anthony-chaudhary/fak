package ultracodebench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"
)

const AccountingUsageSchema = "fak.ultracode.accounting-usage.v1"

// AccountingUsageRow is the public-safe adoption record for one accounting invocation.
// It deliberately carries no arguments, paths, hostnames, or provider identifiers.
type AccountingUsageRow struct {
	Schema      string                  `json:"schema"`
	ObservedAt  time.Time               `json:"observed_at"`
	Invocations int                     `json:"invocations"`
	Outcomes    AccountingOutcomeCounts `json:"outcomes"`
}

// AccountingUsageWeek folds authoritative-accounting adoption by ISO week.
type AccountingUsageWeek struct {
	Week        string                  `json:"week"`
	Invocations int                     `json:"invocations"`
	Outcomes    AccountingOutcomeCounts `json:"outcomes"`
}

// AppendAccountingUsage durably records one invocation and its receipt outcomes.
func AppendAccountingUsage(path string, observedAt time.Time, receipts ...AccountingReceipt) (AccountingUsageRow, error) {
	if path == "" {
		return AccountingUsageRow{}, fmt.Errorf("accounting usage ledger path is required")
	}
	if observedAt.IsZero() {
		return AccountingUsageRow{}, fmt.Errorf("accounting usage observed_at is required")
	}
	for i, receipt := range receipts {
		if err := receipt.Validate(); err != nil {
			return AccountingUsageRow{}, fmt.Errorf("receipt %d: %w", i, err)
		}
	}
	row := AccountingUsageRow{
		Schema: AccountingUsageSchema, ObservedAt: observedAt.UTC(), Invocations: 1,
		Outcomes: accountingOutcomeCounts(receipts...),
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		return AccountingUsageRow{}, fmt.Errorf("encode accounting usage: %w", err)
	}
	encoded = append(encoded, '\n')
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return AccountingUsageRow{}, fmt.Errorf("open accounting usage ledger: %w", err)
	}
	if _, err := f.Write(encoded); err != nil {
		_ = f.Close()
		return AccountingUsageRow{}, fmt.Errorf("append accounting usage ledger: %w", err)
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return AccountingUsageRow{}, fmt.Errorf("sync accounting usage ledger: %w", err)
	}
	if err := f.Close(); err != nil {
		return AccountingUsageRow{}, fmt.Errorf("close accounting usage ledger: %w", err)
	}
	return row, nil
}

// FoldAccountingUsage reads a ledger and returns deterministic per-ISO-week totals.
func FoldAccountingUsage(path string) ([]AccountingUsageWeek, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open accounting usage ledger: %w", err)
	}
	defer f.Close()

	weeks := make(map[string]AccountingUsageWeek)
	scanner := bufio.NewScanner(f)
	line := 0
	for scanner.Scan() {
		line++
		var row AccountingUsageRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("decode accounting usage line %d: %w", line, err)
		}
		if row.Schema != AccountingUsageSchema || row.ObservedAt.IsZero() || row.Invocations != 1 {
			return nil, fmt.Errorf("invalid accounting usage line %d", line)
		}
		year, week := row.ObservedAt.UTC().ISOWeek()
		key := fmt.Sprintf("%04d-W%02d", year, week)
		fold := weeks[key]
		fold.Week = key
		fold.Invocations += row.Invocations
		fold.Outcomes.Success += row.Outcomes.Success
		fold.Outcomes.Refusal += row.Outcomes.Refusal
		fold.Outcomes.Error += row.Outcomes.Error
		weeks[key] = fold
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan accounting usage ledger: %w", err)
	}
	keys := make([]string, 0, len(weeks))
	for key := range weeks {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	folds := make([]AccountingUsageWeek, 0, len(keys))
	for _, key := range keys {
		folds = append(folds, weeks[key])
	}
	return folds, nil
}
