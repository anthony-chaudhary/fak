package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

const (
	guardDisableUsageSchema        = "fak-guard-disable-usage/1"
	guardDisableUsageSummarySchema = "fak-guard-disable-usage-summary/1"
	guardDisableUsageSuccess       = "success"
	guardDisableUsageChildNonzero  = "child_nonzero"
	guardDisableUsageLaunchError   = "launch_error"
)

type guardDisableUsageRow struct {
	Schema  string `json:"schema"`
	At      string `json:"at"`
	Outcome string `json:"outcome"`
}

type guardDisableUsageWeek struct {
	Week         string `json:"week"`
	Invocations  int    `json:"invocations"`
	Success      int    `json:"success"`
	ChildNonzero int    `json:"child_nonzero"`
	LaunchError  int    `json:"launch_error"`
}

type guardDisableUsageSummary struct {
	Schema string                  `json:"schema"`
	Weeks  []guardDisableUsageWeek `json:"weeks"`
}

func guardDisableUsageDefaultPath() (string, error) {
	if !usagelog.Enabled() {
		return "", nil
	}
	base := strings.TrimSpace(usageLogPath())
	if base == "" {
		return "", errors.New("usage log path is empty")
	}
	return filepath.Join(filepath.Dir(base), "guard-disable-usage.jsonl"), nil
}

func appendGuardDisableUsage(path string, row guardDisableUsageRow) error {
	if path == "" {
		return nil
	}
	if row.Schema == "" {
		row.Schema = guardDisableUsageSchema
	}
	if row.Schema != guardDisableUsageSchema {
		return fmt.Errorf("guard disable usage schema %q is not supported", row.Schema)
	}
	if _, err := time.Parse(time.RFC3339, row.At); err != nil {
		return fmt.Errorf("guard disable usage timestamp: %w", err)
	}
	if !guardDisableUsageOutcomeValid(row.Outcome) {
		return fmt.Errorf("guard disable usage outcome %q is not supported", row.Outcome)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create guard disable usage directory: %w", err)
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return fmt.Errorf("open guard disable usage lock: %w", err)
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		return fmt.Errorf("lock guard disable usage ledger: %w", err)
	}
	defer flock.Unlock(lock)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open guard disable usage ledger: %w", err)
	}
	defer f.Close()
	if err := json.NewEncoder(f).Encode(row); err != nil {
		return fmt.Errorf("append guard disable usage ledger: %w", err)
	}
	return f.Sync()
}

func foldGuardDisableUsage(path string) ([]guardDisableUsageWeek, error) {
	if path == "" {
		return nil, nil
	}
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("open guard disable usage ledger: %w", err)
	}
	defer f.Close()
	weeks := map[string]*guardDisableUsageWeek{}
	scanner := bufio.NewScanner(f)
	for line := 1; scanner.Scan(); line++ {
		if strings.TrimSpace(scanner.Text()) == "" {
			continue
		}
		var row guardDisableUsageRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("read guard disable usage line %d: %w", line, err)
		}
		if row.Schema != guardDisableUsageSchema || !guardDisableUsageOutcomeValid(row.Outcome) {
			return nil, fmt.Errorf("read guard disable usage line %d: unsupported schema or outcome", line)
		}
		at, err := time.Parse(time.RFC3339, row.At)
		if err != nil {
			return nil, fmt.Errorf("read guard disable usage line %d timestamp: %w", line, err)
		}
		year, week := at.UTC().ISOWeek()
		key := fmt.Sprintf("%04d-W%02d", year, week)
		if weeks[key] == nil {
			weeks[key] = &guardDisableUsageWeek{Week: key}
		}
		w := weeks[key]
		w.Invocations++
		switch row.Outcome {
		case guardDisableUsageSuccess:
			w.Success++
		case guardDisableUsageChildNonzero:
			w.ChildNonzero++
		case guardDisableUsageLaunchError:
			w.LaunchError++
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan guard disable usage ledger: %w", err)
	}
	keys := make([]string, 0, len(weeks))
	for key := range weeks {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]guardDisableUsageWeek, 0, len(keys))
	for _, key := range keys {
		out = append(out, *weeks[key])
	}
	return out, nil
}

func guardDisableUsageOutcomeValid(outcome string) bool {
	return outcome == guardDisableUsageSuccess || outcome == guardDisableUsageChildNonzero || outcome == guardDisableUsageLaunchError
}

func recordGuardDisableUsage(stderr io.Writer, path string, pathErr error, outcome string) {
	if pathErr != nil {
		fmt.Fprintf(stderr, "fak guard disable: usage ledger warning: %v\n", pathErr)
		return
	}
	err := appendGuardDisableUsage(path, guardDisableUsageRow{
		Schema:  guardDisableUsageSchema,
		At:      time.Now().UTC().Format(time.RFC3339),
		Outcome: outcome,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak guard disable: usage ledger warning: %v\n", err)
	}
}

func runGuardDisableUsage(stdout, stderr io.Writer, path string, pathErr error, jsonOut bool) int {
	if pathErr != nil {
		fmt.Fprintf(stderr, "fak guard disable: usage ledger: %v\n", pathErr)
		return 1
	}
	weeks, err := foldGuardDisableUsage(path)
	if err != nil {
		fmt.Fprintf(stderr, "fak guard disable: usage ledger: %v\n", err)
		return 1
	}
	if jsonOut {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(guardDisableUsageSummary{Schema: guardDisableUsageSummarySchema, Weeks: weeks}); err != nil {
			fmt.Fprintf(stderr, "fak guard disable: usage ledger: %v\n", err)
			return 1
		}
		return 0
	}
	if len(weeks) == 0 {
		fmt.Fprintln(stdout, "guard disable usage: no recorded invocations")
		return 0
	}
	for _, week := range weeks {
		fmt.Fprintf(stdout, "%s invocations=%d success=%d child_nonzero=%d launch_error=%d\n",
			week.Week, week.Invocations, week.Success, week.ChildNonzero, week.LaunchError)
	}
	return 0
}
