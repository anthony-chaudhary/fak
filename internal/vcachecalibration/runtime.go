package vcachecalibration

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"time"
)

// RuntimeConstants is the fresh, measured subset a live request path may trust.
// Each value remains independently optional: one probe can measure minimum prefix
// without pretending it also measured TTL or read pricing.
type RuntimeConstants struct {
	Provider          string  `json:"provider"`
	Model             string  `json:"model,omitempty"`
	Source            string  `json:"source"`
	TS                string  `json:"ts"`
	TTLMillis         int64   `json:"ttl_millis,omitempty"`
	TTLMeasured       bool    `json:"ttl_measured,omitempty"`
	MinPrefixTokens   int64   `json:"min_prefix_tokens,omitempty"`
	MinPrefixMeasured bool    `json:"min_prefix_measured,omitempty"`
	ReadMult          float64 `json:"read_mult,omitempty"`
	ReadMultMeasured  bool    `json:"read_mult_measured,omitempty"`
	Write5mMult       float64 `json:"write_5m_mult,omitempty"`
	Write5mMeasured   bool    `json:"write_5m_measured,omitempty"`
	Write1hMult       float64 `json:"write_1h_mult,omitempty"`
	Write1hMeasured   bool    `json:"write_1h_measured,omitempty"`
}

// FreshRuntimeConstants selects the newest fresh row matching provider and model.
// A model-specific request never consumes constants fitted for another model; a
// provider-only row is the safe fallback. Missing, stale, or observation-only rows
// return ok=false, preserving the caller's static defaults.
func FreshRuntimeConstants(path, provider, model string, now time.Time, ttl time.Duration) (RuntimeConstants, bool, string) {
	provider = strings.ToLower(strings.TrimSpace(provider))
	model = strings.ToLower(strings.TrimSpace(model))
	if provider == "" {
		return RuntimeConstants{}, false, "provider is empty"
	}
	rows, err := readCalibrationRows(path)
	if err != nil {
		return RuntimeConstants{}, false, "calibration ledger unreadable"
	}
	if ttl <= 0 {
		ttl = DefaultCalibrationTTL
	}
	var best *ProviderCalibration
	bestRank := -1
	for i := range rows {
		row := &rows[i]
		if strings.ToLower(strings.TrimSpace(row.Provider)) != provider {
			continue
		}
		rowModel := strings.ToLower(strings.TrimSpace(row.Model))
		rank := 0
		if model != "" {
			if rowModel == model {
				rank = 2
			} else if rowModel != "" {
				continue
			}
		} else if rowModel != "" {
			// The request model is forwarded dynamically. Carry the newest provider
			// calibration to the gateway, which re-checks the actual wire model.
			rank = 1
		}
		if best == nil || rank > bestRank || (rank == bestRank && row.TS > best.TS) {
			best, bestRank = row, rank
		}
	}
	if best == nil {
		return RuntimeConstants{}, false, "no matching provider/model calibration"
	}
	ts, err := time.Parse(time.RFC3339Nano, best.TS)
	if err != nil || now.Sub(ts) > ttl {
		return RuntimeConstants{}, false, "matching calibration is stale"
	}
	out := RuntimeConstants{
		Provider: best.Provider, Model: best.Model, Source: best.Source, TS: best.TS,
		TTLMillis: best.TTLMillis, TTLMeasured: best.TTLMeasured,
		MinPrefixTokens: best.MinPrefixTokens, MinPrefixMeasured: best.MinPrefixMeasured,
		ReadMult: best.ReadMult, ReadMultMeasured: best.ReadMultMeasured,
		Write5mMult: best.Write5mMult, Write5mMeasured: best.Write5mMeasured,
		Write1hMult: best.Write1hMult, Write1hMeasured: best.Write1hMeasured,
	}
	if !out.TTLMeasured && !out.MinPrefixMeasured && !out.ReadMultMeasured && !out.Write5mMeasured && !out.Write1hMeasured {
		return RuntimeConstants{}, false, "fresh row has no measured runtime constants"
	}
	return out, true, "fresh measured calibration"
}

func readCalibrationRows(path string) ([]ProviderCalibration, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []ProviderCalibration
	s := bufio.NewScanner(f)
	s.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for s.Scan() {
		var row ProviderCalibration
		if json.Unmarshal(s.Bytes(), &row) == nil && ValidateCalibration(row) == nil {
			rows = append(rows, row)
		}
	}
	return rows, s.Err()
}
