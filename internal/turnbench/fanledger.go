package turnbench

// fanledger.go — the CADENCE seam for the fan-out warm-vs-cold sweep (#3630).
//
// fanout.go MEASURES shared-prefix KV reuse across N siblings once. This file makes that
// figure TRENDABLE: a scheduled sweep extracts the headline reuse figure at the widest
// fan-out cell, stamps it with the run's date, and appends it as one JSONL row to a
// durable ledger; a fold over the ledger yields the dated first→last reuse trend the
// cadence exists to surface. The clock lives at the CALLER (the CLI/workflow) — this
// package never reads time.Now, so the fold and its tests stay deterministic and
// `go test ./internal/turnbench` is reproducible.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
)

// FanLedgerSchema versions one appended fan-out reuse row (the cadence ledger's line
// schema). Bump it if a field's meaning changes so a fold can tell a stale mix apart.
const FanLedgerSchema = "fak.fanout-reuse-ledger.v1"

// Fan-out reuse trend direction verdicts — the first→last movement of the trended
// cross-agent reuse figure over the cadence window.
const (
	FanTrendSingle     = "single"     // one data point — a baseline, no movement yet
	FanTrendRising     = "rising"     // cross-agent reuse improved over the window
	FanTrendFlat       = "flat"       // cross-agent reuse unchanged over the window
	FanTrendRegressing = "regressing" // cross-agent reuse dropped over the window
)

// FanLedgerRow is ONE dated fan-out warm-vs-cold reuse figure — the per-run datum the
// scheduled sweep appends so shared-prefix reuse is trended over time instead of measured
// once and forgotten. It records the MEASURED headline at a fixed headline cell (the
// widest fan-out point, where cross-agent reuse is most saturated): the sibling-only
// cross-agent dedup uplift and the exact (N−1)·prefix reuse geometry, next to the modeled
// prefix-cache tax reclaimed. Date is stamped by the CALLER (the CLI/workflow clock),
// never in this package.
type FanLedgerRow struct {
	Schema            string  `json:"schema"`
	Date              string  `json:"date"` // RFC3339 UTC instant the sweep ran (caller-stamped)
	AppVersion        string  `json:"app_version"`
	Profile           string  `json:"profile"`
	Agents            int     `json:"agents"` // N at the headline cell
	SubTurns          int     `json:"sub_turns"`
	PrefixTokens      int     `json:"prefix_tokens"`
	Trials            int     `json:"trials"`
	Seed              int64   `json:"seed"`
	CrossUpliftP50    int     `json:"cross_uplift_p50"`    // MEASURED sibling-only dedup turns at N (warm vs isolated)
	PrefixTokensSaved int     `json:"prefix_tokens_saved"` // MEASURED (N−1)·prefix reuse geometry at N
	TaxClawedBack     float64 `json:"tax_clawed_back"`     // MODELED prefix-cache tax fraction reclaimed at N
}

// JSONL renders the row as a single compact JSON line (no trailing newline) — the exact
// bytes appended to the ledger, one row per line.
func (r FanLedgerRow) JSONL() []byte {
	b, _ := json.Marshal(r)
	return b
}

// HeadlineFanCell returns the sweep's HEADLINE cell — the widest fan-out point, where
// cross-agent reuse is most saturated — breaking ties toward the larger prefix then the
// longer sub-agent session. It reports false for an empty sweep. This is the single cell
// the cadence ledger trends, so the per-run figure is comparable across runs.
func HeadlineFanCell(sw FanoutSweep) (FanoutCell, bool) {
	if len(sw.Cells) == 0 {
		return FanoutCell{}, false
	}
	best := sw.Cells[0]
	for _, c := range sw.Cells[1:] {
		if fanCellWider(c, best) {
			best = c
		}
	}
	return best, true
}

// fanCellWider reports whether c is a wider headline cell than best: more agents, then a
// larger prefix, then a longer sub-agent session.
func fanCellWider(c, best FanoutCell) bool {
	if c.Agents != best.Agents {
		return c.Agents > best.Agents
	}
	if c.PrefixTokens != best.PrefixTokens {
		return c.PrefixTokens > best.PrefixTokens
	}
	return c.SubTurns > best.SubTurns
}

// FanLedgerRowFromSweep extracts the headline reuse figure from a completed sweep and
// stamps it with the given RFC3339 date (the caller's clock), returning the ledger row to
// append. It reports false for an empty sweep. The date is the ONLY non-deterministic
// input, kept at the boundary so the fold and tests stay reproducible.
func FanLedgerRowFromSweep(sw FanoutSweep, date string) (FanLedgerRow, bool) {
	cell, ok := HeadlineFanCell(sw)
	if !ok {
		return FanLedgerRow{}, false
	}
	return FanLedgerRow{
		Schema:            FanLedgerSchema,
		Date:              date,
		AppVersion:        sw.AppVersion,
		Profile:           sw.Profile.Name,
		Agents:            cell.Agents,
		SubTurns:          cell.SubTurns,
		PrefixTokens:      cell.PrefixTokens,
		Trials:            sw.Trials,
		Seed:              sw.Seed,
		CrossUpliftP50:    cell.CrossUplift.P50,
		PrefixTokensSaved: cell.PrefixTokensSaved,
		TaxClawedBack:     cell.TaxClawedBack,
	}, true
}

// FanTrend is the folded trend over a fan-out reuse ledger: every row in date order plus
// the first→last delta on the headline reuse axes, so "is shared-prefix reuse holding,
// rising, or regressing over the cadence?" is answerable at a glance — the visible trend
// fold the cadence exists to produce.
type FanTrend struct {
	Schema                 string         `json:"schema"`
	Count                  int            `json:"count"`
	Rows                   []FanLedgerRow `json:"rows"`
	First                  FanLedgerRow   `json:"first"`
	Last                   FanLedgerRow   `json:"last"`
	CrossUpliftDelta       int            `json:"cross_uplift_delta"`
	PrefixTokensSavedDelta int            `json:"prefix_tokens_saved_delta"`
	Direction              string         `json:"direction"`
}

// FoldFanLedger reads a JSONL fan-out reuse ledger (one FanLedgerRow per non-blank line),
// orders the rows by date, and folds them into a dated trend carrying the first→last
// reuse delta and a direction verdict. Blank lines are skipped; a malformed line is an
// error (a silently-dropped row would understate the trend). An empty ledger folds to a
// zero-count trend with no error, so a first-ever run renders cleanly.
func FoldFanLedger(r io.Reader) (FanTrend, error) {
	var rows []FanLedgerRow
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row FanLedgerRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return FanTrend{}, fmt.Errorf("turnbench: fan ledger line %d: %w", len(rows)+1, err)
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return FanTrend{}, fmt.Errorf("turnbench: read fan ledger: %w", err)
	}
	return foldFanRows(rows), nil
}

// foldFanRows folds already-parsed rows into a trend. RFC3339 UTC ('Z') dates sort
// lexicographically == chronologically, so a plain string sort orders the window.
func foldFanRows(rows []FanLedgerRow) FanTrend {
	t := FanTrend{Schema: FanLedgerSchema, Count: len(rows), Rows: rows}
	if len(rows) == 0 {
		t.Direction = FanTrendSingle
		return t
	}
	ordered := append([]FanLedgerRow(nil), rows...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Date < ordered[j].Date })
	t.Rows = ordered
	t.First = ordered[0]
	t.Last = ordered[len(ordered)-1]
	t.CrossUpliftDelta = t.Last.CrossUpliftP50 - t.First.CrossUpliftP50
	t.PrefixTokensSavedDelta = t.Last.PrefixTokensSaved - t.First.PrefixTokensSaved
	switch {
	case len(ordered) == 1:
		t.Direction = FanTrendSingle
	case t.CrossUpliftDelta > 0:
		t.Direction = FanTrendRising
	case t.CrossUpliftDelta < 0:
		t.Direction = FanTrendRegressing
	default:
		t.Direction = FanTrendFlat
	}
	return t
}

// JSON renders the folded trend as a stable-indented artifact (the same encoding every
// other report in this package uses).
func (t FanTrend) JSON() []byte { return marshalArtifact(&t) }

// Render renders the folded trend as a compact human card for a CI step summary or
// stderr — the dated per-run reuse figures, then the first→last delta and direction.
// Deterministic given the trend.
func (t FanTrend) Render() string {
	var b strings.Builder
	fmt.Fprintf(&b, "fan-out reuse trend — %d run(s), direction=%s\n", t.Count, t.Direction)
	if t.Count == 0 {
		b.WriteString("  (ledger empty — the next scheduled run seeds the baseline)\n")
		return b.String()
	}
	for _, r := range t.Rows {
		fmt.Fprintf(&b, "  %s  profile=%s N=%d sub_turns=%d prefix=%d  cross_uplift_p50=%d  prefix_tokens_saved=%d  tax_back=%.1f%%\n",
			r.Date, r.Profile, r.Agents, r.SubTurns, r.PrefixTokens,
			r.CrossUpliftP50, r.PrefixTokensSaved, r.TaxClawedBack*100)
	}
	if t.Count >= 2 {
		fmt.Fprintf(&b, "  Δ first→last: cross_uplift_p50 %+d · prefix_tokens_saved %+d (%s)\n",
			t.CrossUpliftDelta, t.PrefixTokensSavedDelta, t.Direction)
	}
	return b.String()
}
