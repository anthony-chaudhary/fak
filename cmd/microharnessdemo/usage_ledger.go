package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const usageLedgerSchema = "fak.microharness.usage.v1"

type usageRow struct {
	Schema           string    `json:"schema"`
	At               time.Time `json:"at"`
	Mode             string    `json:"mode"`
	Model            string    `json:"model,omitempty"`
	Outcome          string    `json:"outcome"`
	Completions      int       `json:"completions"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
}

type weeklyUsage struct {
	Week             string `json:"week"`
	Invocations      int    `json:"invocations"`
	Completed        int    `json:"completed"`
	Failed           int    `json:"failed"`
	Live             int    `json:"live"`
	Fixture          int    `json:"fixture"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
}

func defaultUsageLedgerPath() (string, error) {
	root, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "fak", "microharness-usage.jsonl"), nil
}

func appendUsage(path string, row usageRow) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("usage ledger path is required")
	}
	if row.Schema == "" {
		row.Schema = usageLedgerSchema
	}
	if row.At.IsZero() {
		row.At = time.Now().UTC()
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("create usage ledger directory: %w", err)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open usage ledger: %w", err)
	}
	defer f.Close()
	if err := json.NewEncoder(f).Encode(row); err != nil {
		return fmt.Errorf("append usage ledger: %w", err)
	}
	return nil
}

func foldWeeklyUsage(path string) ([]weeklyUsage, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return []weeklyUsage{}, nil
		}
		return nil, err
	}
	defer f.Close()
	byWeek := map[string]*weeklyUsage{}
	scan := bufio.NewScanner(f)
	line := 0
	for scan.Scan() {
		line++
		var row usageRow
		if err := json.Unmarshal(scan.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("usage ledger line %d: %w", line, err)
		}
		if row.Schema != usageLedgerSchema || row.At.IsZero() {
			return nil, fmt.Errorf("usage ledger line %d: unsupported or incomplete row", line)
		}
		year, week := row.At.ISOWeek()
		key := fmt.Sprintf("%04d-W%02d", year, week)
		out := byWeek[key]
		if out == nil {
			out = &weeklyUsage{Week: key}
			byWeek[key] = out
		}
		out.Invocations++
		if row.Outcome == "completed" {
			out.Completed++
		} else {
			out.Failed++
		}
		if row.Mode == "live" {
			out.Live++
		} else if row.Mode == "fixture" {
			out.Fixture++
		}
		out.PromptTokens += row.PromptTokens
		out.CompletionTokens += row.CompletionTokens
	}
	if err := scan.Err(); err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(byWeek))
	for key := range byWeek {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]weeklyUsage, 0, len(keys))
	for _, key := range keys {
		out = append(out, *byWeek[key])
	}
	return out, nil
}
