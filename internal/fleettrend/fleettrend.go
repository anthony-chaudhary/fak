package fleettrend

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
)

const (
	Schema        = "fleet-trend/1"
	DefaultLedger = ".fak/nightrun/fleet-status-history.jsonl"
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

// Counters are monotonic cumulative totals persisted alongside the gauges so the
// throughput SLO (WindowRates) can diff them across a rolling window. They are
// NOT rendered by RenderLine — only the rate derivation reads them. A snapshot
// that omits the throughput object simply never sets them, so a window without
// counters honestly reports its rates as absent rather than a fabricated zero.
var Counters = []MetricDef{
	{Key: "lands", Label: "lands"},     // git-WITNESSED lands (see landsWitnessedKey)
	{Key: "resumes", Label: "resumes"}, // auto-resume launches
	{Key: "deaths", Label: "deaths"},   // worker crash/exhaust exits
	{Key: landsWitnessedKey, Label: "lands_witnessed"},
}

// landsWitnessedKey persists whether the lands counter is git-witnessed (1) or
// self-reported (0/absent). Goodput leans on lands, so the render surfaces this
// provenance rather than letting a self-reported number pass as witnessed. Ties
// to the OBSERVED-vs-WITNESSED provenance tag the upstream snapshot must fill.
const landsWitnessedKey = "lands_witnessed"

var blocks = []rune("▁▂▃▄▅▆▇█")

func MetricsOf(snap map[string]any) map[string]float64 {
	sessions := asMap(snap["sessions"])
	byCategory := asMap(sessions["by_category"])
	accounts := asMap(snap["accounts"])
	system := asMap(snap["system"])
	m := map[string]float64{
		"usable":   number(accounts["usable"]),
		"live":     number(byCategory["LIVE"]),
		"sessions": number(sessions["total"]),
		"escalate": number(system["escalate"]),
	}
	// Cumulative throughput counters live under an optional "throughput" object so
	// a health snapshot that carries none leaves the rate columns unset (honest
	// absence). lands_witness is a provenance string the producer stamps "git"
	// when the lands total was counted from origin/main-ancestor commits.
	if tp := asMap(snap["throughput"]); len(tp) > 0 {
		for _, c := range []string{"lands", "resumes", "deaths"} {
			if v, ok := tp[c]; ok {
				m[c] = number(v)
			}
		}
		if w, _ := tp["lands_witness"].(string); w == "git" || w == "witnessed" {
			m[landsWitnessedKey] = 1
		}
	}
	return m
}

func Append(path string, metrics map[string]float64, now string, capRows int) (map[string]any, error) {
	row := map[string]any{"ts": now}
	for _, metric := range Metrics {
		if v, ok := metrics[metric.Key]; ok {
			row[metric.Key] = compactNumber(v)
		}
	}
	for _, counter := range Counters {
		if v, ok := metrics[counter.Key]; ok {
			row[counter.Key] = compactNumber(v)
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
	rows := foldedRows(path)
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

// Rate is one per-hour figure derived over the rolling window. Present is false
// when the window lacks the two counter-bearing, timestamped endpoints needed to
// derive it, so callers render "n/a" instead of a fabricated 0/hr.
type Rate struct {
	PerHour float64 `json:"per_hour"`
	Delta   float64 `json:"delta"`
	Present bool    `json:"present"`
}

// Rates is the fleet-throughput SLO over a rolling window: how fast work lands
// against how fast workers resume and die. Goodput = landsΔ / (landsΔ + deathsΔ)
// answers whether adding workers shipped work or just churned. LandsWitnessed
// carries whether the lands total is git-witnessed (honest) or self-reported.
type Rates struct {
	WindowHours    float64 `json:"window_hours"`
	Ticks          int     `json:"ticks"`
	Lands          Rate    `json:"lands"`
	Resumes        Rate    `json:"resumes"`
	Deaths         Rate    `json:"deaths"`
	Goodput        float64 `json:"goodput"`
	GoodputPresent bool    `json:"goodput_present"`
	LandsWitnessed bool    `json:"lands_witnessed"`
}

// WindowRates derives resumes/hr, deaths/hr, lands/hr, and a goodput ratio from
// the cumulative counters in the given rows (already trimmed to the window by
// Tail). Each rate diffs the first and last row that carries its counter and
// divides by the wall-clock hours between their timestamps — so the arithmetic
// is a pure function of the ledger, no clock read. The second return is false
// when no counter could be derived (a window with no throughput data at all).
func WindowRates(rows []map[string]any) (Rates, bool) {
	out := Rates{Ticks: len(rows)}
	if len(rows) > 0 {
		if first, ok1 := parseRowTime(rows[0]["ts"]); ok1 {
			if last, ok2 := parseRowTime(rows[len(rows)-1]["ts"]); ok2 {
				if h := last.Sub(first).Hours(); h > 0 {
					out.WindowHours = h
				}
			}
		}
	}
	out.Lands = counterRate(rows, "lands")
	out.Resumes = counterRate(rows, "resumes")
	out.Deaths = counterRate(rows, "deaths")
	if out.Lands.Present && out.Deaths.Present {
		if denom := out.Lands.Delta + out.Deaths.Delta; denom > 0 {
			out.Goodput = out.Lands.Delta / denom
			out.GoodputPresent = true
		}
	}
	out.LandsWitnessed = lastCounterFlag(rows, landsWitnessedKey)
	return out, out.Lands.Present || out.Resumes.Present || out.Deaths.Present
}

// counterRate reads the first and last timestamped rows that carry key and
// returns their delta over the hours between them. It needs two such rows: a
// single data point yields no rate. A negative delta (a counter reset when the
// producing process restarted) clamps to 0 rather than emitting a spurious rate.
func counterRate(rows []map[string]any, key string) Rate {
	var first, last float64
	var firstTime, lastTime time.Time
	var n int
	for _, row := range rows {
		v, ok := counterValue(row[key])
		if !ok {
			continue
		}
		ts, tsok := parseRowTime(row["ts"])
		if !tsok {
			continue
		}
		if n == 0 {
			first, firstTime = v, ts
		}
		last, lastTime = v, ts
		n++
	}
	if n < 2 {
		return Rate{}
	}
	hours := lastTime.Sub(firstTime).Hours()
	if hours <= 0 {
		return Rate{}
	}
	delta := last - first
	if delta < 0 {
		delta = 0
	}
	return Rate{PerHour: delta / hours, Delta: delta, Present: true}
}

// lastCounterFlag reports whether the most recent row carrying key sets it to a
// truthy (>= 1) value — used to read the lands-provenance flag off the freshest
// snapshot in the window.
func lastCounterFlag(rows []map[string]any, key string) bool {
	for i := len(rows) - 1; i >= 0; i-- {
		if v, ok := counterValue(rows[i][key]); ok {
			return v >= 1
		}
	}
	return false
}

// RenderThroughput is the one fleet-status row: the four SLO figures plus the
// window label. Absent counters render "n/a" (not 0/hr), and the lands figure
// carries a provenance tag so a self-reported goodput never reads as witnessed.
func RenderThroughput(rows []map[string]any) string {
	r, ok := WindowRates(rows)
	if !ok {
		return ""
	}
	goodput := "n/a"
	if r.GoodputPresent {
		goodput = fmt.Sprintf("%.0f%%", r.Goodput*100)
	}
	line := fmt.Sprintf("throughput: lands %s · resumes %s · deaths %s · goodput %s (%s)",
		fmtRate(r.Lands), fmtRate(r.Resumes), fmtRate(r.Deaths), goodput, windowLabel(r))
	if r.Lands.Present {
		if r.LandsWitnessed {
			line += " [lands: git-witnessed]"
		} else {
			line += " [lands: self-reported]"
		}
	}
	return line
}

func fmtRate(r Rate) string {
	if !r.Present {
		return "n/a"
	}
	return fmt.Sprintf("%.1f/hr", r.PerHour)
}

func windowLabel(r Rates) string {
	ticks := fmt.Sprintf("%d ticks", r.Ticks)
	if r.Ticks == 1 {
		ticks = "1 tick"
	}
	if r.WindowHours <= 0 {
		return "over " + ticks
	}
	return fmt.Sprintf("over %.1fh · %s", r.WindowHours, ticks)
}

// parseRowTime parses a row's RFC3339 ts stamp (the format ISONow writes). A
// non-timestamp value (older torn rows, test fixtures) yields ok=false so the
// rate derivation skips it rather than crashing.
func parseRowTime(v any) (time.Time, bool) {
	s, ok := v.(string)
	if !ok || s == "" {
		return time.Time{}, false
	}
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Time{}, false
	}
	return t, true
}

// counterValue reads a numeric counter, reporting presence separately from value
// so a missing key is distinguishable from a real 0 (number() folds both to 0).
func counterValue(v any) (float64, bool) {
	switch x := v.(type) {
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	case float64:
		return x, true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	}
	return 0, false
}

func ISONow() string {
	return time.Now().UTC().Truncate(time.Second).Format(time.RFC3339)
}

// readRows reads the whole ledger from scratch. The write path (Append) uses it
// so the byte-for-byte rewrite it performs never depends on cached fold state.
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

// foldState accumulates the parsed rows in file order — the same slice readRows
// returns, so Tail/RenderLine see identical data.
type foldState struct {
	rows []map[string]any
}

var (
	foldMu    sync.Mutex
	foldCkpts = map[string]jsonlledger.Checkpoint[foldState]{}
)

// foldRow decodes one JSONL line exactly as readRows did — UseNumber so integer
// counts round-trip as json.Number, malformed lines dropped — and folds it in.
func foldRow(s foldState, raw json.RawMessage) foldState {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var row map[string]any
	if err := dec.Decode(&row); err != nil {
		return s
	}
	s.rows = append(s.rows, row)
	return s
}

// foldedRows is the read/render path readRows used to serve: it returns every
// ledger row, but folds only the bytes appended since the last read of this
// path via jsonlledger.TailFold, keyed by an in-process checkpoint. The dozens
// of panes that poll the trend re-read only the delta instead of the whole
// capped ledger each tick. A rewrite (the ledger dropping its oldest row at cap)
// changes the bytes ending at the prior offset, so TailFold re-folds in full —
// the output stays byte-identical to a from-scratch read. On any I/O error it
// falls back to that from-scratch read, matching readRows' best-effort contract.
func foldedRows(path string) []map[string]any {
	key := path
	if abs, err := filepath.Abs(path); err == nil {
		key = abs
	}
	foldMu.Lock()
	defer foldMu.Unlock()
	ck, err := jsonlledger.TailFold(path, foldCkpts[key], foldState{}, foldRow)
	if err != nil {
		delete(foldCkpts, key)
		return readRows(path)
	}
	foldCkpts[key] = ck
	if len(ck.State.rows) == 0 {
		return nil
	}
	// Hand back a copy: callers (and Append) mutate the slice they receive, and
	// the cached fold state must not move under a later delta read.
	out := make([]map[string]any, len(ck.State.rows))
	copy(out, ck.State.rows)
	return out
}

// resetFoldCache clears the in-process checkpoint cache. Tests that reuse a
// ledger path across scenarios call it to fold from a clean slate.
func resetFoldCache() {
	foldMu.Lock()
	defer foldMu.Unlock()
	foldCkpts = map[string]jsonlledger.Checkpoint[foldState]{}
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
