package fleettrend

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	Schema        = "fleet-trend/1"
	DefaultLedger = "docs/nightrun/fleet-status-history.jsonl"
	DefaultCap    = 500
)

type MetricDef struct {
	Key   string
	Label string
}

var Metrics = []MetricDef{
	{Key: "usable", Label: "usable"},
	{Key: "live", Label: "live"},
	{Key: "sessions", Label: "sessions"},
	{Key: "escalate", Label: "escalate"},
}

var blocks = []rune("▁▂▃▄▅▆▇█")

func MetricsOf(snap map[string]any) map[string]float64 {
	sessions := asMap(snap["sessions"])
	byCategory := asMap(sessions["by_category"])
	accounts := asMap(snap["accounts"])
	system := asMap(snap["system"])
	return map[string]float64{
		"usable":   number(accounts["usable"]),
		"live":     number(byCategory["LIVE"]),
		"sessions": number(sessions["total"]),
		"escalate": number(system["escalate"]),
	}
}

func Append(path string, metrics map[string]float64, now string, capRows int) (map[string]any, error) {
	row := map[string]any{"ts": now}
	for _, metric := range Metrics {
		if v, ok := metrics[metric.Key]; ok {
			row[metric.Key] = compactNumber(v)
		}
	}
	rows := readRows(path)
	rows = append(rows, row)
	if capRows > 0 && len(rows) > capRows {
		rows = rows[len(rows)-capRows:]
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return nil, err
		}
	}
	return row, nil
}

func Tail(path string, n int) []map[string]any {
	rows := readRows(path)
	if n <= 0 || n >= len(rows) {
		return rows
	}
	return rows[len(rows)-n:]
}

func Spark(values []float64) string {
	if len(values) == 0 {
		return ""
	}
	lo, hi := values[0], values[0]
	for _, v := range values[1:] {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if hi <= lo {
		return strings.Repeat(string(blocks[0]), len(values))
	}
	span := hi - lo
	var b strings.Builder
	for _, v := range values {
		idx := int((v-lo)/span*float64(len(blocks)-1) + 0.5)
		if idx < 0 {
			idx = 0
		}
		if idx >= len(blocks) {
			idx = len(blocks) - 1
		}
		b.WriteRune(blocks[idx])
	}
	return b.String()
}

type Trend struct {
	Key   string  `json:"key"`
	First float64 `json:"first"`
	Last  float64 `json:"last"`
	Delta float64 `json:"delta"`
	Spark string  `json:"spark"`
	N     int     `json:"n"`
}

func MetricTrend(rows []map[string]any, key string) (Trend, bool) {
	series := make([]float64, 0, len(rows))
	for _, row := range rows {
		switch v := row[key].(type) {
		case int:
			series = append(series, float64(v))
		case int64:
			series = append(series, float64(v))
		case float64:
			series = append(series, v)
		case json.Number:
			if f, err := v.Float64(); err == nil {
				series = append(series, f)
			}
		}
	}
	if len(series) == 0 {
		return Trend{}, false
	}
	delta := math.Round((series[len(series)-1]-series[0])*1000) / 1000
	return Trend{
		Key:   key,
		First: series[0],
		Last:  series[len(series)-1],
		Delta: delta,
		Spark: Spark(series),
		N:     len(series),
	}, true
}

func RenderLine(rows []map[string]any) string {
	if len(rows) == 0 {
		return ""
	}
	var parts []string
	for _, metric := range Metrics {
		trend, ok := MetricTrend(rows, metric.Key)
		if !ok {
			continue
		}
		arrow := formatNumber(trend.Last)
		if trend.N > 1 {
			arrow = formatNumber(trend.First) + "→" + formatNumber(trend.Last)
		}
		delta := ""
		if trend.N > 1 && trend.Delta != 0 {
			sign := ""
			if trend.Delta > 0 {
				sign = "+"
			}
			delta = fmt.Sprintf(" (%s%s over %d)", sign, formatNumber(trend.Delta), trend.N)
		}
		parts = append(parts, strings.TrimSpace(fmt.Sprintf("%s %s %s%s", metric.Label, arrow, trend.Spark, delta)))
	}
	if len(parts) == 0 {
		return ""
	}
	return "trend: " + strings.Join(parts, " · ")
}

func ISONow() string {
	return time.Now().UTC().Truncate(time.Second).Format(time.RFC3339)
}

func readRows(path string) []map[string]any {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	var rows []map[string]any
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		dec := json.NewDecoder(strings.NewReader(line))
		dec.UseNumber()
		var row map[string]any
		if err := dec.Decode(&row); err != nil {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

func asMap(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return map[string]any{}
}

func number(v any) float64 {
	switch x := v.(type) {
	case int:
		return float64(x)
	case int64:
		return float64(x)
	case float64:
		return x
	case json.Number:
		f, _ := x.Float64()
		return f
	case string:
		f, _ := strconv.ParseFloat(x, 64)
		return f
	default:
		return 0
	}
}

func compactNumber(v float64) any {
	if math.IsNaN(v) || math.IsInf(v, 0) {
		return 0
	}
	if math.Trunc(v) == v {
		return int(v)
	}
	return v
}

func formatNumber(v float64) string {
	if math.Trunc(v) == v {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}
