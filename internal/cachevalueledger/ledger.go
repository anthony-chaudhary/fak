package cachevalueledger

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

const (
	Schema          = "fak-cache-value-ledger/1"
	DefaultLedgerRel = "docs/nightrun/cache-value.jsonl"
)

type Row struct {
	Schema      string           `json:"schema"`
	Date        string           `json:"date"`
	SessionType string           `json:"session_type"`
	Context     string           `json:"context"`
	PID         int              `json:"pid"`
	UnixMillis  int64            `json:"unix_millis"`
	Turns       uint64           `json:"turns"`
	PromptTokens uint64          `json:"prompt_tokens"`
	ReusedTokens uint64          `json:"reused_tokens"`
	FrozenTurns  uint64          `json:"frozen_turns"`
	PartialTurns uint64          `json:"partial_turns"`
	ColdTurns    uint64          `json:"cold_turns"`
	ReuseRatio   float64         `json:"reuse_ratio"`
	Stats       cacheobs.Stats  `json:"stats"`
	GeneratedAt string           `json:"generated_at"`
}

func ParseLedger(content string) []Row {
	var rows []Row
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row Row
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.Date == "" || row.SessionType == "" {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

func AppendLedgerLine(row Row) (string, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func NewRow(sessionType, context string, stats cacheobs.Stats, now time.Time) Row {
	return Row{
		Schema:       Schema,
		Date:         now.UTC().Format("2006-01-02"),
		SessionType:  sessionType,
		Context:      context,
		PID:          os.Getpid(),
		UnixMillis:   now.UnixMilli(),
		Turns:        stats.Turns,
		PromptTokens: stats.PromptTokens,
		ReusedTokens: stats.ReusedTokens,
		FrozenTurns:  stats.FrozenTurns,
		PartialTurns: stats.PartialTurns,
		ColdTurns:    stats.ColdTurns,
		ReuseRatio:   stats.ReuseRatio,
		Stats:        stats,
		GeneratedAt:  now.UTC().Format(time.RFC3339),
	}
}

func Append(sessionType, context, ledgerPath string, stats cacheobs.Stats) error {
	now := time.Now()
	row := NewRow(sessionType, context, stats, now)
	line, err := AppendLedgerLine(row)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(ledgerPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.WriteString(line + "\n"); err != nil {
		return err
	}
	return nil
}

func ReadLedgerFile(path string) []Row {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return ParseLedger(string(b))
}

func ToVCacheTurns(rows []Row) []vcachegov.TelemetryRow {
	var turns []vcachegov.TelemetryRow
	for _, r := range rows {
		if r.Turns == 0 {
			continue
		}
		t := vcachegov.TelemetryRow{
			InputTokens:              float64(r.PromptTokens),
			CacheCreationInputTokens: float64(r.PromptTokens - r.ReusedTokens),
			CacheReadInputTokens:     float64(r.ReusedTokens),
			Ephemeral1hInputTokens:   0,
			Ephemeral5mInputTokens:   0,
		}
		turns = append(turns, t)
	}
	return turns
}

// ScoreLedgerResult summarizes the cache-value ledger for regression gate checks.
type ScoreLedgerResult struct {
	TotalSessions        int     `json:"total_sessions"`
	TotalTurns           uint64  `json:"total_turns"`
	TotalCacheHitTokens  uint64  `json:"total_cache_hit_tokens"`
	TotalGeneratedTokens uint64  `json:"total_generated_tokens"`
	OverallMultiplier    float64 `json:"overall_multiplier"`
	LowMultiplierSessions int    `json:"low_multiplier_sessions"`
}

// ScoreLedger reads the ledger and computes aggregate metrics for the regression gate.
func ScoreLedger(path string) (ScoreLedgerResult, error) {
	var result ScoreLedgerResult
	rows := ReadLedgerFile(path)
	for _, r := range rows {
		if r.Turns == 0 {
			continue
		}
		result.TotalSessions++
		result.TotalTurns += r.Turns
		result.TotalCacheHitTokens += r.ReusedTokens
		result.TotalGeneratedTokens += r.PromptTokens
		if r.ReuseRatio < 1.0 {
			result.LowMultiplierSessions++
		}
	}
	if result.TotalGeneratedTokens > 0 {
		result.OverallMultiplier = float64(result.TotalCacheHitTokens) / float64(result.TotalGeneratedTokens-result.TotalCacheHitTokens)
	}
	return result, nil
}