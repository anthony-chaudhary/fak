package usagelog

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// usage_metrics.go is the READ-side projection of a usage journal into a
// BOUNDED, deterministic set of Prometheus gauges/counters for Grafana — the
// answer to "which fak verbs run, how often, how slow, and which fail?" on a
// dashboard instead of in `fak usage`'s text fold.
//
// It is deliberately dependency-free (no Prometheus client library), matching
// the hand-rolled exposition style already used by cmd/fak's fleet/cachevalue
// exporters, and it is pure given the rows + an injected clock, so it is
// trivially unit-testable and cannot drift from `fak usage`.
//
// HONESTY FENCE — unchanged from the package doc: every number here is an
// OBSERVED self-report of the `fak` PROCESS (verb, exit code, timing). It is
// never a witness of any downstream effect, and the invocation set is only as
// complete as usagelog_record.go's documented os.Exit coverage gap allows.

// DefaultMetricMaxSeries bounds the number of {verb,outcome} series a projection
// keeps. The verb vocabulary is closed and small (a few hundred strings), but a
// malformed or adversarial journal could carry an unbounded set of verb values,
// so the bound is enforced with the drop REPORTED (MetricProjection.Truncated),
// never silent — a truncated panel that reads as complete is worse than none.
const DefaultMetricMaxSeries = 200

// MetricOptions parameterizes a projection. The zero value folds every row with
// the default series bound and no time cutoff.
type MetricOptions struct {
	// Since drops rows older than this wall-clock cutoff (zero = fold all).
	Since time.Time
	// MaxSeries bounds the kept {verb,outcome} series. <=0 selects
	// DefaultMetricMaxSeries; a negative value disables the bound (tests only).
	MaxSeries int
}

// VerbOutcomeStat is one aggregated {verb,outcome} series: an OBSERVED roll-up of
// the process's own rows. Duration is in milliseconds (Row.DurationMS).
type VerbOutcomeStat struct {
	Verb           string          `json:"verb"`
	Outcome        TerminalOutcome `json:"outcome"`
	Count          int             `json:"count"`
	SumMS          int64           `json:"sum_ms"`
	MaxMS          int64           `json:"max_ms"`
	LastMS         int64           `json:"last_ms"`
	LastTSUnixNano int64           `json:"last_ts_unix_nano"`
}

// MetricProjection is the folded, render-ready view of a usage journal.
type MetricProjection struct {
	Rows             int               `json:"rows"`                // rows folded after the Since cutoff
	Errors           int               `json:"errors"`              // folded rows with exit_code != 0
	Series           []VerbOutcomeStat `json:"series"`              // bounded, deterministic order
	Truncated        int               `json:"truncated"`           // series kept OUT by MaxSeries
	OldestTSUnixNano int64             `json:"oldest_ts_unix_nano"` // oldest folded row timestamp
	NewestTSUnixNano int64             `json:"newest_ts_unix_nano"` // newest folded row timestamp
}

// ProjectRows folds usage rows into a bounded MetricProjection. It is pure and
// deterministic: rows whose schema is not SchemaV1 are skipped (a foreign JSONL
// line never pollutes the projection), and the series order is stable
// (count descending, then verb, then outcome).
func ProjectRows(rows []Row, opt MetricOptions) MetricProjection {
	maxSeries := opt.MaxSeries
	if maxSeries == 0 {
		maxSeries = DefaultMetricMaxSeries
	}
	var sinceUnixNano int64
	if !opt.Since.IsZero() {
		sinceUnixNano = opt.Since.UnixNano()
	}

	type key struct {
		verb    string
		outcome TerminalOutcome
	}
	acc := map[key]*VerbOutcomeStat{}
	var out MetricProjection

	for _, r := range rows {
		if r.Schema != SchemaV1 {
			continue
		}
		if sinceUnixNano > 0 && r.TSUnixNano < sinceUnixNano {
			continue
		}
		out.Rows++
		if r.ExitCode != 0 {
			out.Errors++
		}
		if out.OldestTSUnixNano == 0 || r.TSUnixNano < out.OldestTSUnixNano {
			out.OldestTSUnixNano = r.TSUnixNano
		}
		if r.TSUnixNano > out.NewestTSUnixNano {
			out.NewestTSUnixNano = r.TSUnixNano
		}
		k := key{verb: strings.TrimSpace(r.Verb), outcome: ClassifyTerminalOutcome(r.ExitCode)}
		s := acc[k]
		if s == nil {
			s = &VerbOutcomeStat{Verb: k.verb, Outcome: k.outcome}
			acc[k] = s
		}
		s.Count++
		s.SumMS += r.DurationMS
		if r.DurationMS > s.MaxMS {
			s.MaxMS = r.DurationMS
		}
		// The journal is append-ordered (chronological), so the last row seen for
		// a series is its most recent invocation. A caller folding an out-of-order
		// slice gets the last row in slice order, not the newest timestamp; rows
		// read via ReadRows are always in file (chronological) order.
		s.LastMS = r.DurationMS
		s.LastTSUnixNano = r.TSUnixNano
	}

	out.Series = make([]VerbOutcomeStat, 0, len(acc))
	for _, s := range acc {
		out.Series = append(out.Series, *s)
	}
	sort.Slice(out.Series, func(i, j int) bool {
		a, b := out.Series[i], out.Series[j]
		if a.Count != b.Count {
			return a.Count > b.Count
		}
		if a.Verb != b.Verb {
			return a.Verb < b.Verb
		}
		return terminalOutcomeRank(a.Outcome) < terminalOutcomeRank(b.Outcome)
	})
	if maxSeries >= 0 && len(out.Series) > maxSeries {
		out.Truncated = len(out.Series) - maxSeries
		out.Series = out.Series[:maxSeries]
	}
	return out
}

// RenderPrometheus renders the projection as Prometheus 0.0.4 text exposition.
// The `fak_cli_` namespace makes these the LOCAL CLI-invocation families, distinct
// from `fak_fleet_*` (session/usage ledger) and `fak_harness_*` (harness tool
// adjudication). now stamps the freshness gauge so a stalled exporter is visible.
func (p MetricProjection) RenderPrometheus(now time.Time) string {
	var b strings.Builder
	declare := map[string]bool{}
	header := func(name, help, typ string) {
		if declare[name] {
			return
		}
		declare[name] = true
		fmt.Fprintf(&b, "# HELP %s %s\n", name, escapeHelpString(help))
		fmt.Fprintf(&b, "# TYPE %s %s\n", name, typ)
	}
	sample := func(name string, val float64, labelPairs ...string) {
		b.WriteString(name)
		writeMetricLabels(&b, labelPairs)
		b.WriteByte(' ')
		b.WriteString(formatMetricFloat(val))
		b.WriteByte('\n')
	}

	header("fak_cli_usage_rows_in_window", "CLI-invocation usage rows folded after the freshness window cutoff.", "gauge")
	sample("fak_cli_usage_rows_in_window", float64(p.Rows))

	header("fak_cli_usage_errors_in_window", "Folded CLI-invocation rows that exited non-zero.", "gauge")
	sample("fak_cli_usage_errors_in_window", float64(p.Errors))

	header("fak_cli_usage_series_truncated", "Higher-frequency {verb,outcome} series kept OUT by the cardinality bound; non-zero means a per-verb panel is incomplete.", "gauge")
	sample("fak_cli_usage_series_truncated", float64(p.Truncated))

	header("fak_cli_usage_oldest_timestamp_seconds", "Wall-clock time of the oldest folded usage row (Unix epoch seconds).", "gauge")
	if p.OldestTSUnixNano > 0 {
		sample("fak_cli_usage_oldest_timestamp_seconds", float64(p.OldestTSUnixNano)/1e9)
	}

	header("fak_cli_usage_newest_timestamp_seconds", "Wall-clock time of the newest folded usage row (Unix epoch seconds).", "gauge")
	if p.NewestTSUnixNano > 0 {
		sample("fak_cli_usage_newest_timestamp_seconds", float64(p.NewestTSUnixNano)/1e9)
	}

	header("fak_cli_invocations_total", "Observed top-level `fak` invocations per verb and terminal outcome (process self-report, not a downstream witness).", "counter")
	header("fak_cli_invocation_duration_ms_total", "Cumulative observed wall-clock milliseconds per verb and terminal outcome.", "counter")
	header("fak_cli_invocation_duration_ms_max", "Largest observed single-invocation wall-clock per verb and terminal outcome.", "gauge")
	header("fak_cli_invocation_last_duration_ms", "Wall-clock of the most recent observed invocation per verb and terminal outcome.", "gauge")
	header("fak_cli_invocation_last_timestamp_seconds", "Wall-clock time of the most recent observed invocation per verb and terminal outcome (Unix epoch seconds).", "gauge")

	for _, s := range p.Series {
		lbl := []string{"verb", s.Verb, "outcome", string(s.Outcome)}
		sample("fak_cli_invocations_total", float64(s.Count), lbl...)
		sample("fak_cli_invocation_duration_ms_total", float64(s.SumMS), lbl...)
		sample("fak_cli_invocation_duration_ms_max", float64(s.MaxMS), lbl...)
		sample("fak_cli_invocation_last_duration_ms", float64(s.LastMS), lbl...)
		if s.LastTSUnixNano > 0 {
			sample("fak_cli_invocation_last_timestamp_seconds", float64(s.LastTSUnixNano)/1e9, lbl...)
		}
	}

	header("fak_cli_usage_render_timestamp_seconds", "Wall-clock time this projection was rendered (Unix epoch seconds); a stalled exporter shows here.", "gauge")
	sample("fak_cli_usage_render_timestamp_seconds", float64(now.Unix()))
	return b.String()
}

// DefaultMaxReadBytes bounds how many trailing bytes of a usage journal the
// metric fold reads. A usage.jsonl grows without bound (one row per fak
// invocation, forever), and `fak fleet metrics --serve` folds it on EVERY
// scrape; parsing a multi-hundred-MB journal per scrape would both stall the
// scrape and waste memory. The exporter only needs RECENT rows (its Since
// window), and rows are append-ordered, so reading the tail is sufficient and
// honest: a fold that hits the bound still returns the newest rows, and the
// oldest-timestamp gauge names how far back it actually reached.
const DefaultMaxReadBytes = 8 << 20 // 8 MiB

// ReadTail reads the trailing at most maxBytes of a usage journal and returns
// its rows in order. A MISSING file is the empty journal (nil, nil). A partial
// leading line (the tail began mid-row) is discarded, and a torn final line is
// tolerated by stopping at the last well-formed row — the same robustness
// ReadRows holds. maxBytes <= 0 selects DefaultMaxReadBytes.
func ReadTail(path string, maxBytes int64) ([]Row, error) {
	if maxBytes <= 0 {
		maxBytes = DefaultMaxReadBytes
	}
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("usagelog: read %s: %w", path, err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("usagelog: stat %s: %w", path, err)
	}
	start := int64(0)
	midLine := false
	if info.Size() > maxBytes {
		start = info.Size() - maxBytes
		midLine = true // the byte at start is inside a row; drop its partial prefix
	}
	if _, err := f.Seek(start, io.SeekStart); err != nil {
		return nil, fmt.Errorf("usagelog: seek %s: %w", path, err)
	}
	var out []Row
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	first := true
	for sc.Scan() {
		line := sc.Bytes()
		if first && midLine {
			first = false
			continue // partial prefix of a row the window cut in half
		}
		first = false
		if len(line) == 0 {
			continue
		}
		var r Row
		if err := json.Unmarshal(line, &r); err != nil {
			break
		}
		out = append(out, r)
	}
	if err := sc.Err(); err != nil {
		return out, fmt.Errorf("usagelog: scan %s: %w", path, err)
	}
	return out, nil
}

// FoldPath reads a usage journal at path and projects it. A missing journal is an
// honest empty projection (Rows 0), never an error — the same contract ReadRows
// holds. The read is bounded to the journal TAIL (DefaultMaxReadBytes) so a
// long-lived, multi-hundred-MB journal does not stall a per-scrape fold; see
// ReadTail. A read error is returned so the exporter can flip a readability gauge.
func FoldPath(path string, opt MetricOptions) (MetricProjection, error) {
	rows, err := ReadTail(path, DefaultMaxReadBytes)
	if err != nil {
		return MetricProjection{}, err
	}
	return ProjectRows(rows, opt), nil
}

// escapeHelpString and writeMetricLabels mirror the exporter-local helpers used by
// cmd/fak's hand-rolled Prometheus writers, kept here so internal/usagelog can
// render without importing a CLI package.
func escapeHelpString(s string) string {
	if !strings.ContainsAny(s, "\\\n") {
		return s
	}
	return strings.NewReplacer(`\`, `\\`, "\n", `\n`).Replace(s)
}

func writeMetricLabels(b *strings.Builder, pairs []string) {
	if len(pairs) < 2 {
		return
	}
	b.WriteByte('{')
	for i := 0; i+1 < len(pairs); i += 2 {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(pairs[i])
		b.WriteString(`="`)
		b.WriteString(escapeLabelValueString(pairs[i+1]))
		b.WriteByte('"')
	}
	b.WriteByte('}')
}

// escapeLabelValueString escapes a Prometheus label VALUE: backslash, double
// quote, and newline must all be escaped inside a "..." label (unlike HELP text,
// where a quote is literal), or the exposition is malformed.
func escapeLabelValueString(s string) string {
	if !strings.ContainsAny(s, "\\\"\n") {
		return s
	}
	return strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`).Replace(s)
}

func formatMetricFloat(v float64) string {
	if v == float64(int64(v)) {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}
