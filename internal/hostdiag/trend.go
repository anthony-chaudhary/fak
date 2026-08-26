package hostdiag

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	TrendSchema        = "fak.hostdiag-trend.v1"
	trendIdentityLimit = 10
)

// Trend is an observational summary of privacy-safe correlation identities.
// Its windows are half-open: baseline precedes recent, and recent ends at AsOfUTC.
type Trend struct {
	Schema     string          `json:"schema"`
	AsOfUTC    string          `json:"as_of_utc"`
	Recent     TrendWindow     `json:"recent"`
	Baseline   TrendWindow     `json:"baseline"`
	Comparison TrendComparison `json:"comparison"`
}

type TrendWindow struct {
	StartUTC      string          `json:"start_utc"`
	EndUTC        string          `json:"end_utc"`
	Total         int             `json:"total"`
	Crash         int             `json:"crash"`
	Hang          int             `json:"hang"`
	ActiveUTCDays int             `json:"active_utc_days"`
	PerDay        []TrendDay      `json:"per_day"`
	TopApps       []TrendIdentity `json:"top_apps"`
	TopEvents     []TrendIdentity `json:"top_events"`
}

type TrendDay struct {
	Date  string `json:"date"`
	Total int    `json:"total"`
	Crash int    `json:"crash"`
	Hang  int    `json:"hang"`
}

type TrendIdentity struct {
	Identity string `json:"identity"`
	Total    int    `json:"total"`
	Crash    int    `json:"crash"`
	Hang     int    `json:"hang"`
}

type TrendComparison struct {
	Observational bool    `json:"observational"`
	Causal        bool    `json:"causal"`
	TotalDelta    int     `json:"total_delta"`
	CrashDelta    int     `json:"crash_delta"`
	HangDelta     int     `json:"hang_delta"`
	TotalRatio    float64 `json:"total_ratio"`
	CrashRatio    float64 `json:"crash_ratio"`
	HangRatio     float64 `json:"hang_ratio"`
	BaselineZero  bool    `json:"baseline_zero"`
}

type trendRow struct {
	Schema        string `json:"schema"`
	CorrelationID string `json:"correlation_id"`
	TimeMS        int64  `json:"time_ms"`
	EventName     string `json:"event_name"`
	App           string `json:"app"`
}

// SummarizeTrend reads a mixed-schema JSONL stream. Valid foreign schemas are
// ignored, while rows claiming the correlation schema are validated strictly.
func SummarizeTrend(r io.Reader, now time.Time, recent, baseline time.Duration) (Trend, error) {
	if recent <= 0 || baseline <= 0 {
		return Trend{}, fmt.Errorf("recent and baseline must be positive")
	}
	now = now.UTC()
	recentStart := now.Add(-recent)
	baselineStart := recentStart.Add(-baseline)
	result := Trend{
		Schema: TrendSchema, AsOfUTC: now.Format(time.RFC3339Nano),
		Recent:   emptyTrendWindow(recentStart, now),
		Baseline: emptyTrendWindow(baselineStart, recentStart),
	}
	seen := make(map[string]struct{})
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 4<<20)
	line := 0
	for scanner.Scan() {
		line++
		raw := bytes.TrimSpace(scanner.Bytes())
		if len(raw) == 0 {
			continue
		}
		var envelope struct {
			Schema string `json:"schema"`
		}
		if err := json.Unmarshal(raw, &envelope); err != nil {
			return Trend{}, fmt.Errorf("line %d: malformed JSON: %w", line, err)
		}
		if envelope.Schema != CorrelationSchema {
			continue
		}
		var row trendRow
		if err := json.Unmarshal(raw, &row); err != nil {
			return Trend{}, fmt.Errorf("line %d: malformed correlation: %w", line, err)
		}
		if err := validateTrendRow(row); err != nil {
			return Trend{}, fmt.Errorf("line %d: malformed correlation: %w", line, err)
		}
		if _, duplicate := seen[row.CorrelationID]; duplicate {
			continue
		}
		seen[row.CorrelationID] = struct{}{}
		at := time.UnixMilli(row.TimeMS).UTC()
		switch {
		case !at.Before(recentStart) && at.Before(now):
			addTrendRow(&result.Recent, row, at)
		case !at.Before(baselineStart) && at.Before(recentStart):
			addTrendRow(&result.Baseline, row, at)
		}
	}
	if err := scanner.Err(); err != nil {
		return Trend{}, fmt.Errorf("read ledger: %w", err)
	}
	finishTrendWindow(&result.Recent)
	finishTrendWindow(&result.Baseline)
	result.Comparison = TrendComparison{
		Observational: true, Causal: false,
		TotalDelta:   result.Recent.Total - result.Baseline.Total,
		CrashDelta:   result.Recent.Crash - result.Baseline.Crash,
		HangDelta:    result.Recent.Hang - result.Baseline.Hang,
		TotalRatio:   finiteRatio(result.Recent.Total, result.Baseline.Total),
		CrashRatio:   finiteRatio(result.Recent.Crash, result.Baseline.Crash),
		HangRatio:    finiteRatio(result.Recent.Hang, result.Baseline.Hang),
		BaselineZero: result.Baseline.Total == 0,
	}
	return result, nil
}

func emptyTrendWindow(start, end time.Time) TrendWindow {
	return TrendWindow{StartUTC: start.Format(time.RFC3339Nano), EndUTC: end.Format(time.RFC3339Nano), PerDay: []TrendDay{}, TopApps: []TrendIdentity{}, TopEvents: []TrendIdentity{}}
}

func validateTrendRow(row trendRow) error {
	if strings.TrimSpace(row.CorrelationID) == "" {
		return fmt.Errorf("missing correlation_id")
	}
	if row.TimeMS <= 0 {
		return fmt.Errorf("invalid time_ms")
	}
	if !safeEventIdentity(row.EventName) {
		return fmt.Errorf("invalid event_name")
	}
	return nil
}

func safeEventIdentity(s string) bool {
	if s == "" || len(s) > 96 {
		return false
	}
	for _, r := range s {
		if (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' {
			continue
		}
		return false
	}
	return true
}

func normalizeAppIdentity(app string) string {
	app = strings.TrimSpace(strings.ReplaceAll(app, `\`, "/"))
	app = strings.ToLower(filepath.Base(app))
	if app == "." || app == "" || len(app) > 96 {
		return ""
	}
	for _, r := range app {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '.' || r == '_' || r == '-' {
			continue
		}
		return ""
	}
	return app
}

func rowKind(event string) (crash, hang bool) {
	u := strings.ToUpper(event)
	return strings.Contains(u, "CRASH") || strings.Contains(u, "FAULT"), strings.Contains(u, "HANG") || strings.Contains(u, "RADAR")
}

func addTrendRow(window *TrendWindow, row trendRow, at time.Time) {
	crash, hang := rowKind(row.EventName)
	window.Total++
	if crash {
		window.Crash++
	}
	if hang {
		window.Hang++
	}
	date := at.Format("2006-01-02")
	window.PerDay = appendCountDay(window.PerDay, date, crash, hang)
	if app := normalizeAppIdentity(row.App); app != "" {
		window.TopApps = appendCountIdentity(window.TopApps, app, crash, hang)
	}
	window.TopEvents = appendCountIdentity(window.TopEvents, row.EventName, crash, hang)
}

func appendCountDay(rows []TrendDay, date string, crash, hang bool) []TrendDay {
	for i := range rows {
		if rows[i].Date == date {
			rows[i].Total++
			if crash {
				rows[i].Crash++
			}
			if hang {
				rows[i].Hang++
			}
			return rows
		}
	}
	row := TrendDay{Date: date, Total: 1}
	if crash {
		row.Crash = 1
	}
	if hang {
		row.Hang = 1
	}
	return append(rows, row)
}

func appendCountIdentity(rows []TrendIdentity, identity string, crash, hang bool) []TrendIdentity {
	for i := range rows {
		if rows[i].Identity == identity {
			rows[i].Total++
			if crash {
				rows[i].Crash++
			}
			if hang {
				rows[i].Hang++
			}
			return rows
		}
	}
	row := TrendIdentity{Identity: identity, Total: 1}
	if crash {
		row.Crash = 1
	}
	if hang {
		row.Hang = 1
	}
	return append(rows, row)
}

func finishTrendWindow(window *TrendWindow) {
	sort.Slice(window.PerDay, func(i, j int) bool { return window.PerDay[i].Date < window.PerDay[j].Date })
	window.ActiveUTCDays = len(window.PerDay)
	order := func(rows []TrendIdentity) {
		sort.Slice(rows, func(i, j int) bool {
			if rows[i].Total != rows[j].Total {
				return rows[i].Total > rows[j].Total
			}
			return rows[i].Identity < rows[j].Identity
		})
	}
	order(window.TopApps)
	order(window.TopEvents)
	if len(window.TopApps) > trendIdentityLimit {
		window.TopApps = window.TopApps[:trendIdentityLimit]
	}
	if len(window.TopEvents) > trendIdentityLimit {
		window.TopEvents = window.TopEvents[:trendIdentityLimit]
	}
}

func finiteRatio(value, baseline int) float64 {
	if baseline == 0 {
		return 0
	}
	ratio := float64(value) / float64(baseline)
	if math.IsInf(ratio, 0) || math.IsNaN(ratio) {
		return 0
	}
	return ratio
}
