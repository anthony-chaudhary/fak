package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const codexSubmitUsageSchema = "fak.sessions.codex_submit_usage/1"

type codexSubmitUsageRow struct {
	Schema  string    `json:"schema"`
	At      time.Time `json:"at"`
	Week    string    `json:"week"`
	Mode    string    `json:"mode"`
	Outcome string    `json:"outcome"`
}

type codexSubmitWeekCounts struct {
	Week     string         `json:"week"`
	Total    int            `json:"total"`
	Modes    map[string]int `json:"modes"`
	Outcomes map[string]int `json:"outcomes"`
}

type codexSubmitUsageSummary struct {
	Schema string                  `json:"schema"`
	Weeks  []codexSubmitWeekCounts `json:"weeks"`
}

var codexSubmitUsageNow = time.Now

func codexSubmitUsagePath(codexHome string) (string, error) {
	home, err := resolvedCodexLoopHome(codexHome)
	if err != nil {
		return "", err
	}
	return filepath.Join(home, "fak-ledgers", "codex-userpromptsubmit.jsonl"), nil
}

func codexSubmitMode(argv []string) string {
	if codexLoopHookOverrideEnabled(os.Getenv(guardActiveEnv)) {
		return "guarded"
	}
	if codexLoopHookOverrideEnabled(os.Getenv(codexLoopHookHardenedEnv)) || codexLoopHookBoolFlagEnabled(argv, "hardened") {
		return "hardened"
	}
	return "permissive"
}

func codexSubmitHomeArg(argv []string) string {
	for i, arg := range argv {
		if arg == "--codex-home" && i+1 < len(argv) {
			return argv[i+1]
		}
		if strings.HasPrefix(arg, "--codex-home=") {
			return strings.TrimPrefix(arg, "--codex-home=")
		}
	}
	return ""
}

func codexSubmitIsUsageSummary(argv []string) bool {
	return codexLoopHookBoolFlagEnabled(argv, "usage-summary")
}

func appendCodexSubmitUsage(argv []string, outcome string) error {
	if codexSubmitIsUsageSummary(argv) {
		return nil
	}
	path, err := codexSubmitUsagePath(codexSubmitHomeArg(argv))
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	now := codexSubmitUsageNow().UTC()
	year, week := now.ISOWeek()
	row := codexSubmitUsageRow{Schema: codexSubmitUsageSchema, At: now, Week: fmt.Sprintf("%04d-W%02d", year, week), Mode: codexSubmitMode(argv), Outcome: outcome}
	body, err := json.Marshal(row)
	if err != nil {
		return err
	}
	body = append(body, '\n')
	fh, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	if _, err = fh.Write(body); err == nil {
		err = fh.Sync()
	}
	closeErr := fh.Close()
	if err != nil {
		return err
	}
	return closeErr
}

func foldCodexSubmitUsage(rd io.Reader) (codexSubmitUsageSummary, error) {
	byWeek := map[string]*codexSubmitWeekCounts{}
	scanner := bufio.NewScanner(rd)
	scanner.Buffer(make([]byte, 0, 4096), 1<<20)
	for scanner.Scan() {
		var row codexSubmitUsageRow
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return codexSubmitUsageSummary{}, fmt.Errorf("decode usage row: %w", err)
		}
		if row.Schema != codexSubmitUsageSchema || row.Week == "" || row.Mode == "" || row.Outcome == "" {
			return codexSubmitUsageSummary{}, fmt.Errorf("invalid usage row")
		}
		counts := byWeek[row.Week]
		if counts == nil {
			counts = &codexSubmitWeekCounts{Week: row.Week, Modes: map[string]int{}, Outcomes: map[string]int{}}
			byWeek[row.Week] = counts
		}
		counts.Total++
		counts.Modes[row.Mode]++
		counts.Outcomes[row.Outcome]++
	}
	if err := scanner.Err(); err != nil {
		return codexSubmitUsageSummary{}, err
	}
	weeks := make([]codexSubmitWeekCounts, 0, len(byWeek))
	for _, counts := range byWeek {
		weeks = append(weeks, *counts)
	}
	sort.Slice(weeks, func(i, j int) bool { return weeks[i].Week < weeks[j].Week })
	return codexSubmitUsageSummary{Schema: "fak.sessions.codex_submit_usage_summary/1", Weeks: weeks}, nil
}

func emitCodexSubmitUsageSummary(stdout, stderr io.Writer, codexHome string) int {
	path, err := codexSubmitUsagePath(codexHome)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: usage ledger path: %v\n", err)
		return 1
	}
	fh, err := os.Open(path)
	if os.IsNotExist(err) {
		return encodeCodexSubmitJSON(stdout, codexSubmitUsageSummary{Schema: "fak.sessions.codex_submit_usage_summary/1", Weeks: []codexSubmitWeekCounts{}})
	}
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: open usage ledger: %v\n", err)
		return 1
	}
	defer fh.Close()
	summary, err := foldCodexSubmitUsage(fh)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop-hook: fold usage ledger: %v\n", err)
		return 1
	}
	return encodeCodexSubmitJSON(stdout, summary)
}

func encodeCodexSubmitJSON(w io.Writer, value any) int {
	if err := json.NewEncoder(w).Encode(value); err != nil {
		return 1
	}
	return 0
}
