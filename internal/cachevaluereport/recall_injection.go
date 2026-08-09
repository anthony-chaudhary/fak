package cachevaluereport

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const DefaultRecallInjectionLedger = ".fak/recall-injections.jsonl"
const RecallInputUSDPerMillionTokens = 3.0

type RecallInjectionRow struct {
	Schema          string  `json:"schema"`
	At              string  `json:"at"`
	Records         int     `json:"records"`
	EstimatedTokens int     `json:"estimated_tokens"`
	EstimatedUSD    float64 `json:"estimated_usd"`
}

type RecallInjectionDebit struct {
	Injections      int     `json:"injections"`
	Records         int     `json:"records"`
	EstimatedTokens int     `json:"estimated_tokens"`
	EstimatedUSD    float64 `json:"estimated_usd"`
}

func AppendRecallInjection(path string, records, tokens int, now time.Time) error {
	if records <= 0 || tokens <= 0 {
		return nil
	}
	if path == "" {
		path = DefaultRecallInjectionLedger
	}
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	row := RecallInjectionRow{Schema: "fak-recall-injection/1", At: now.UTC().Format(time.RFC3339), Records: records, EstimatedTokens: tokens, EstimatedUSD: float64(tokens) * RecallInputUSDPerMillionTokens / 1_000_000}
	return json.NewEncoder(f).Encode(row)
}

func ReadRecallInjectionDebit(path string) (RecallInjectionDebit, error) {
	var out RecallInjectionDebit
	if path == "" {
		path = DefaultRecallInjectionLedger
	}
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return out, nil
	}
	if err != nil {
		return out, err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		var row RecallInjectionRow
		if err := json.Unmarshal(s.Bytes(), &row); err != nil {
			return out, fmt.Errorf("recall injection ledger: %w", err)
		}
		if row.Schema != "fak-recall-injection/1" || row.Records < 0 || row.EstimatedTokens < 0 || row.EstimatedUSD < 0 {
			return out, fmt.Errorf("recall injection ledger: invalid row")
		}
		out.Injections++
		out.Records += row.Records
		out.EstimatedTokens += row.EstimatedTokens
		out.EstimatedUSD += row.EstimatedUSD
	}
	return out, s.Err()
}
