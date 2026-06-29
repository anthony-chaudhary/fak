package nightrun

import (
	"bufio"
	"encoding/json"
	"os"
	"regexp"
	"strconv"
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
	OutcomeDryRun    Outcome = "dry-run"   // printed only; nothing executed (a summary state — never written to the ledger by the run loop, but a valid recorded outcome)
	OutcomeSkipped   Outcome = "skipped"   // a manual/placeholder Run (operator-setup witness); surfaced but never auto-executed by the run loop — but a valid recorded outcome (a bridge fold of a hand-run datum)
	// OutcomePassed / OutcomeDegraded are the BRIDGE-FOLD outcomes. A heavy HW-gated
	// witness (a GLM-5.2 decode on a DGX, a CUDA-graph parity run) is reached over the
	// Slack bridge, NOT by running `fak nightrun run --apply` on-box, so its result is
	// folded into the ledger by an agent. "passed" = the witness ran and met its bar (a
	// real collected datum, possibly with a measured number); "degraded" = it ran and
	// emitted output but did NOT meet its bar (slow/incoherent — recorded honestly, not
	// dropped). These are admitted by the enum so a bridge fold goes THROUGH the typed
	// builder (NewCollectRow / RecordRow) and the ledger validator, instead of being a
	// hand-written off-schema row the tool structurally could not stamp (#1140).
	OutcomePassed   Outcome = "passed"   // bridge-folded: the off-box witness ran and met its bar (a collected datum)
	OutcomeDegraded Outcome = "degraded" // bridge-folded: the off-box witness ran but did NOT meet its bar (recorded honestly)
)

// validOutcomes is the closed set of outcome tokens a ledger row may carry. A row
// whose outcome is outside this set is off-schema — the validator (ValidateLedger)
// rejects it, and the typed builders never produce one. dry-run is included because a
// caller may record a planned datum; the run loop itself still never appends one.
var validOutcomes = map[Outcome]bool{
	OutcomeCollected: true,
	OutcomeFailed:    true,
	OutcomeTimeout:   true,
	OutcomeDryRun:    true,
	OutcomeSkipped:   true,
	OutcomePassed:    true,
	OutcomeDegraded:  true,
}

// IsValidOutcome reports whether o is a member of the closed outcome vocabulary.
func IsValidOutcome(o Outcome) bool { return validOutcomes[o] }

// CollectedOutcome reports whether an outcome means a datum was actually GATHERED on
// this box — the freshness signal lastCollected/next() read. "collected" is the run
// loop's clean-capture; "passed" is the bridge fold of a witness that met its bar.
// "degraded" is deliberately NOT collected: it ran but missed its bar (slow/incoherent),
// so it must not mark the datum fresh and suppress a re-measure.
func CollectedOutcome(o Outcome) bool {
	return o == OutcomeCollected || o == OutcomePassed
}

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
	Note        string  `json:"note,omitempty"`         // free-text context for a bridge-folded witness (what was observed off-box); empty for a run-loop row
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

// RecordInput is the bridge-fold payload: an off-box witness an agent reached over the
// Slack bridge (a DGX GLM-5.2 decode, a CUDA-graph parity run) and is folding into the
// ledger by hand. It carries the OBSERVED fields an attempt would have produced —
// distinct from NewCollectRow, which derives Command/Value from a Task it actually ran.
type RecordInput struct {
	TaskID   string        // the backlog Task.ID this datum belongs to (validated against the registry)
	Box      string        // the box the witness ran on (FAK_BOX_ID or a stable alias)
	Value    string        // the importance class, for trend reads (defaults to the registered Task's Value when empty)
	Command  string        // the exact command the bridge ran
	Outcome  Outcome       // an OBSERVED, in-vocabulary outcome (passed/degraded/collected/failed/timeout)
	Number   string        // a measured headline number, or "" (never fabricated)
	Note     string        // free-text context (what was observed off-box)
	Artifact string        // an artifact path, or ""
	Duration time.Duration // wall time, or 0
}

// RecordRow builds a schema-valid CollectRow from a bridge-fold payload, so a witness
// reached over the bridge appends THROUGH the typed builder (and the validator) instead
// of a hand-written off-schema line. It is the engine behind a `fak nightrun record`
// subcommand: the cmd layer collects the flags, resolves the registered Task for Value/
// defaults, and calls this. It returns an error when the outcome is off-vocabulary, so a
// typo can never reach the ledger. The registered Task supplies the Value default; pass
// the zero Task only when recording a row whose Value is given explicitly.
func RecordRow(in RecordInput, registered Task, now time.Time) (CollectRow, error) {
	if strings.TrimSpace(in.TaskID) == "" {
		return CollectRow{}, errMissingTaskID
	}
	if !IsValidOutcome(in.Outcome) {
		return CollectRow{}, &OffSchemaError{TaskID: in.TaskID, Outcome: string(in.Outcome)}
	}
	value := strings.TrimSpace(in.Value)
	if value == "" {
		value = string(registered.Value)
	}
	return CollectRow{
		Schema:      CollectSchema,
		Date:        now.UTC().Format("2006-01-02"),
		Box:         in.Box,
		TaskID:      in.TaskID,
		Value:       value,
		Command:     in.Command,
		Outcome:     string(in.Outcome),
		Artifact:    in.Artifact,
		Number:      in.Number,
		Note:        in.Note,
		DurationSec: round1(in.Duration.Seconds()),
		GeneratedAt: now.UTC().Format(time.RFC3339),
	}, nil
}

// errMissingTaskID is returned by RecordRow when no task id is supplied.
var errMissingTaskID = &LedgerRowError{Reason: "a recorded row needs a task_id"}

// LedgerRowError is a generic, structured ledger-row defect (a missing field, an
// unregistered id). It is a typed error so a caller can branch on the class.
type LedgerRowError struct {
	TaskID string
	Reason string
}

func (e *LedgerRowError) Error() string {
	if e.TaskID == "" {
		return "nightrun ledger: " + e.Reason
	}
	return "nightrun ledger: task " + e.TaskID + ": " + e.Reason
}

// OffSchemaError reports a row whose outcome is outside the closed vocabulary — the
// exact defect #1140 found in the hand-folded rows (`passed`/`degraded` before the enum
// admitted them, and any future typo).
type OffSchemaError struct {
	TaskID  string
	Outcome string
}

func (e *OffSchemaError) Error() string {
	return "nightrun ledger: task " + e.TaskID + ": off-schema outcome " + strconv.Quote(e.Outcome) +
		" (not in collected|failed|timeout|dry-run|skipped|passed|degraded)"
}

// LedgerDefect is one reason a ledger row failed validation, keyed to the offending
// row's line (1-based) and task id, so a CI report points at the exact line to fix.
type LedgerDefect struct {
	Line    int    `json:"line"`    // 1-based line number in the ledger file
	TaskID  string `json:"task_id"` // the row's task id (may be empty/unregistered)
	Outcome string `json:"outcome"` // the row's outcome token
	Reason  string `json:"reason"`  // why it failed (off-schema outcome / unregistered task id)
}

// TaskIDSet returns the set of Task.IDs in tasks — the `registeredIDs` argument
// ValidateLedger expects. The cmd layer assembles the Backlog (built-ins + overlay) and
// passes Backlog through this so the validator knows every legitimate task id, including
// overlay rows, without importing the overlay-file path itself.
func TaskIDSet(tasks []Task) map[string]bool {
	ids := make(map[string]bool, len(tasks))
	for _, t := range tasks {
		ids[t.ID] = true
	}
	return ids
}

// ValidateLedger checks every row of a ledger against the schema and the registered
// backlog: a row is a defect when its outcome is off the closed vocabulary OR its
// task_id is not a registered backlog/overlay Task. registeredIDs is the set of known
// Task.IDs (the caller passes the assembled Backlog's ids, including any overlay) — so
// the validator stays decoupled from the overlay-file path. It is the engine behind a
// `fak nightrun ledger --check` CI gate. A row that fails ParseLedger (bad JSON / no
// id+date) is already dropped by the reader; this validates the rows that DID parse, so
// pass the rows from ParseLedger. Returns the defects in file order (empty == clean).
func ValidateLedger(rows []CollectRow, registeredIDs map[string]bool) []LedgerDefect {
	var defects []LedgerDefect
	for i, r := range rows {
		line := i + 1
		if !IsValidOutcome(Outcome(r.Outcome)) {
			defects = append(defects, LedgerDefect{
				Line: line, TaskID: r.TaskID, Outcome: r.Outcome,
				Reason: "off-schema outcome (not in collected|failed|timeout|dry-run|skipped|passed|degraded)",
			})
		}
		if registeredIDs != nil && !registeredIDs[r.TaskID] {
			defects = append(defects, LedgerDefect{
				Line: line, TaskID: r.TaskID, Outcome: r.Outcome,
				Reason: "task_id not a registered backlog/overlay Task",
			})
		}
	}
	return defects
}

// lastCollected returns the most recent SUCCESSFUL collection row for taskID on
// box, and whether one exists. Only a collected-class outcome counts (the run
// loop's "collected" or a bridge-folded "passed") — a failed, timed-out, skipped,
// dry-run, or "degraded" (ran but missed its bar) attempt does not make a datum
// fresh. Comparison is by (date, then generated_at) so a same-day re-run is ordered
// after the earlier one.
func lastCollected(rows []CollectRow, taskID, box string) (CollectRow, bool) {
	var best CollectRow
	found := false
	for _, r := range rows {
		if r.TaskID != taskID || !CollectedOutcome(Outcome(r.Outcome)) {
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
	AuthorityDate  string       `json:"authority_date"` // YYYY-MM-DD from BENCHMARK-AUTHORITY.md
	NewerThan      []CollectRow `json:"newer_than"`     // rows with date > authority_date
	Collected      []CollectRow `json:"collected"`      // all rows with outcome=collected
	TotalRows      int          `json:"total_rows"`
	TotalCollected int          `json:"total_collected"`
}

// CompareWithAuthority compares the ledger against the published benchmark surface
// (BENCHMARK-AUTHORITY.md) and returns a report of rows that are newer than the
// authority date.
func CompareWithAuthority(rows []CollectRow, root string) LedgerGapReport {
	authorityDate := BenchmarkAuthorityDate(root)
	var newerThan []CollectRow
	var collected []CollectRow
	for _, r := range rows {
		if CollectedOutcome(Outcome(r.Outcome)) {
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
