package nightrun

import (
	"bufio"
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"time"
)

// CollectSchema tags each durable collection row so a reader can validate the
// line and a future format can be versioned without ambiguity.
const CollectSchema = "fak-nightrun-collect/1"

// DefaultLedgerRel is the committed, append-only collection ledger — one JSONL
// row per collected (or attempted) datum. It lives under docs/ so it is durable
// trunk evidence of what the fleet has gathered, not a regenerable build
// artifact, mirroring docs/cadence/history.jsonl.
const DefaultLedgerRel = "docs/nightrun/collected.jsonl"

// Outcome is the OBSERVED result of attempting a collection — never an asserted
// success. "collected" means the command ran to a zero exit and produced an
// artifact; the row records what happened, the run loop does not embellish it.
type Outcome string

const (
	OutcomeCollected Outcome = "collected" // ran clean, artifact captured
	OutcomeFailed    Outcome = "failed"    // ran, non-zero exit / no artifact
	OutcomeTimeout   Outcome = "timeout"   // exceeded the per-task wall-clock budget; killed (partial artifact kept)
	OutcomeDryRun    Outcome = "dry-run"   // printed only; nothing executed (a summary state — never written to the ledger)
	OutcomeSkipped   Outcome = "skipped"   // a manual/placeholder Run (operator-setup witness); surfaced but never executed — also never written to the ledger
)

// CollectRow is one durable, append-only collection record. It is a flat
// projection so the ledger is a self-describing time series: which box collected
// which task, when, with what command, and what was OBSERVED.
type CollectRow struct {
	Schema      string  `json:"schema"`
	Date        string  `json:"date"`                   // YYYY-MM-DD (UTC)
	Box         string  `json:"box"`                    // the machine that collected it
	TaskID      string  `json:"task_id"`                // the Task.ID join key
	Value       string  `json:"value"`                  // the Task's importance class, for trend reads
	Command     string  `json:"command"`                // the exact command run (or that would run)
	Outcome     string  `json:"outcome"`                // collected | failed | dry-run | skipped
	Artifact    string  `json:"artifact,omitempty"`     // captured output path, when any
	Number      string  `json:"number,omitempty"`       // first parsed unit-bearing token, best-effort (else empty)
	DurationSec float64 `json:"duration_sec,omitempty"` // wall time of the run
	GeneratedAt string  `json:"generated_at"`           // RFC3339 stamp
}

// ParseLedger reads an append-only JSONL ledger, tolerating blank lines and
// skipping any line that is not a valid row (so a hand-edit cannot crash the
// reader). Rows are returned in file order. Mirrors cadencereport.ParseLedger.
func ParseLedger(content string) []CollectRow {
	var rows []CollectRow
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row CollectRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if row.TaskID == "" || row.Date == "" {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// AppendLedgerLine renders the JSONL line for a row (no trailing newline). The
// caller appends it with a newline; keeping the rendering pure makes the writer
// testable without touching disk. Mirrors cadencereport.AppendLedgerLine.
func AppendLedgerLine(row CollectRow) (string, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// NewCollectRow builds a row from a completed (or dry-run) attempt, stamping the
// schema and the date derived from now. The caller supplies the OBSERVED fields.
func NewCollectRow(t Task, box string, outcome Outcome, artifact, number string, dur time.Duration, now time.Time) CollectRow {
	return CollectRow{
		Schema:      CollectSchema,
		Date:        now.UTC().Format("2006-01-02"),
		Box:         box,
		TaskID:      t.ID,
		Value:       string(t.Value),
		Command:     t.Run,
		Outcome:     string(outcome),
		Artifact:    artifact,
		Number:      number,
		DurationSec: round1(dur.Seconds()),
		GeneratedAt: now.UTC().Format(time.RFC3339),
	}
}

// lastCollected returns the most recent SUCCESSFUL collection row for taskID on
// box, and whether one exists. Only OutcomeCollected counts — a failed or
// dry-run attempt does not make a datum fresh. Comparison is by (date, then
// generated_at) so a same-day re-run is ordered after the earlier one.
func lastCollected(rows []CollectRow, taskID, box string) (CollectRow, bool) {
	var best CollectRow
	found := false
	for _, r := range rows {
		if r.TaskID != taskID || r.Outcome != string(OutcomeCollected) {
			continue
		}
		if box != "" && r.Box != box { // empty/other-box row is "not collected here"
			continue
		}
		if !found || laterThan(r, best) {
			best, found = r, true
		}
	}
	return best, found
}

func laterThan(a, b CollectRow) bool {
	if a.Date != b.Date {
		return a.Date > b.Date
	}
	return a.GeneratedAt > b.GeneratedAt
}

func round1(f float64) float64 {
	return float64(int64(f*10+0.5)) / 10
}

// BenchmarkAuthorityPath is the path to the published benchmark authority document.
const BenchmarkAuthorityPath = "BENCHMARK-AUTHORITY.md"

// BenchmarkAuthorityDate returns the last updated date of BENCHMARK-AUTHORITY.md,
// or the zero time if the file cannot be read or the date cannot be parsed.
func BenchmarkAuthorityDate(root string) time.Time {
	path := strings.Join([]string{root, BenchmarkAuthorityPath}, string(os.PathSeparator))
	b, err := os.ReadFile(path)
	if err != nil {
		return time.Time{}
	}
	re := regexp.MustCompile(`\*\*Last updated:\*\*\s*(\d{4}-\d{2}-\d{2})`)
	m := re.FindSubmatch(b)
	if m == nil {
		return time.Time{}
	}
	t, err := time.Parse("2006-01-02", string(m[1]))
	if err != nil {
		return time.Time{}
	}
	return t
}

// LedgerGapReport summarizes collected rows that are newer than or missing from
// the published benchmark surface.
type LedgerGapReport struct {
	AuthorityDate string            `json:"authority_date"` // YYYY-MM-DD from BENCHMARK-AUTHORITY.md
	NewerThan     []CollectRow      `json:"newer_than"`     // rows with date > authority_date
	Collected     []CollectRow      `json:"collected"`      // all rows with outcome=collected
	TotalRows     int               `json:"total_rows"`
	TotalCollected int              `json:"total_collected"`
}

// CompareWithAuthority compares the ledger against the published benchmark surface
// (BENCHMARK-AUTHORITY.md) and returns a report of rows that are newer than the
// authority date.
func CompareWithAuthority(rows []CollectRow, root string) LedgerGapReport {
	authorityDate := BenchmarkAuthorityDate(root)
	var newerThan []CollectRow
	var collected []CollectRow
	for _, r := range rows {
		if r.Outcome == string(OutcomeCollected) {
			collected = append(collected, r)
			if !authorityDate.IsZero() {
				rowDate, err := time.Parse("2006-01-02", r.Date)
				if err == nil && rowDate.After(authorityDate) {
					newerThan = append(newerThan, r)
				}
			}
		}
	}
	var authDateStr string
	if !authorityDate.IsZero() {
		authDateStr = authorityDate.Format("2006-01-02")
	} else {
		authDateStr = "(unknown)"
	}
	return LedgerGapReport{
		AuthorityDate:  authDateStr,
		NewerThan:      newerThan,
		Collected:      collected,
		TotalRows:      len(rows),
		TotalCollected: len(collected),
	}
}
