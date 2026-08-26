package modelperfobs

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"time"
)

const KVCapacityUsageSchema = "fak-kv-capacity-usage/1"

type KVCapacityUsageRow struct {
	Schema           string    `json:"schema"`
	Timestamp        time.Time `json:"timestamp"`
	Dialect          KVDialect `json:"dialect"`
	Outcome          string    `json:"outcome"`
	ValidationIssues int       `json:"validation_issues"`
}

type KVCapacityWeeklyCount struct {
	WeekStart   string `json:"week_start"`
	Invocations int    `json:"invocations"`
	Valid       int    `json:"valid"`
	Invalid     int    `json:"invalid"`
}

// NormalizeKVCapacity is the durable invocation seam. It normalizes one
// sample and appends the outcome to ledgerPath before returning success.
func NormalizeKVCapacity(ledgerPath string, at time.Time, current KVMetricSample, previous *KVMetricSample) (KVCapacitySnapshot, error) {
	snapshot := normalizeKVCapacity(current, previous)
	outcome := "valid"
	if !snapshot.Validation.Valid {
		outcome = "invalid"
	}
	row := KVCapacityUsageRow{Schema: KVCapacityUsageSchema, Timestamp: at.UTC(), Dialect: current.Dialect, Outcome: outcome, ValidationIssues: len(snapshot.Validation.Issues)}
	if err := appendKVCapacityUsage(ledgerPath, row); err != nil {
		return snapshot, fmt.Errorf("record KV capacity normalization: %w", err)
	}
	return snapshot, nil
}

func appendKVCapacityUsage(path string, row KVCapacityUsageRow) error {
	if path == "" {
		return errors.New("ledger path is required")
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	encErr := json.NewEncoder(f).Encode(row)
	closeErr := f.Close()
	return errors.Join(encErr, closeErr)
}

func FoldKVCapacityUsage(path string) ([]KVCapacityWeeklyCount, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	counts := make(map[string]*KVCapacityWeeklyCount)
	scanner := bufio.NewScanner(f)
	line := 0
	for scanner.Scan() {
		line++
		var row KVCapacityUsageRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("decode ledger row %d: %w", line, err)
		}
		if row.Schema != KVCapacityUsageSchema {
			return nil, fmt.Errorf("ledger row %d: unsupported schema %q", line, row.Schema)
		}
		week := mondayUTC(row.Timestamp).Format("2006-01-02")
		count := counts[week]
		if count == nil {
			count = &KVCapacityWeeklyCount{WeekStart: week}
			counts[week] = count
		}
		count.Invocations++
		switch row.Outcome {
		case "valid":
			count.Valid++
		case "invalid":
			count.Invalid++
		default:
			return nil, fmt.Errorf("ledger row %d: unsupported outcome %q", line, row.Outcome)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	weeks := make([]string, 0, len(counts))
	for week := range counts {
		weeks = append(weeks, week)
	}
	sort.Strings(weeks)
	result := make([]KVCapacityWeeklyCount, 0, len(weeks))
	for _, week := range weeks {
		result = append(result, *counts[week])
	}
	return result, nil
}

func mondayUTC(at time.Time) time.Time {
	at = at.UTC()
	days := (int(at.Weekday()) + 6) % 7
	date := at.AddDate(0, 0, -days)
	return time.Date(date.Year(), date.Month(), date.Day(), 0, 0, 0, 0, time.UTC)
}
