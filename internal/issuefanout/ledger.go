package issuefanout

// ledger.go gives the issue-fanout planner its dogfood invocation ledger (#2515):
// every invocation appends ONE durable JSONL row (timestamp + lane + outcome), so
// adoption or neglect of the verb shows up in a per-week fold instead of an
// anecdote. It is the durable, time-stamped counterpart of OutcomeCounts (the pure
// in-memory accumulator): OutcomeCounts answers "how many of each outcome this
// process saw"; the ledger answers "was the verb actually used, week over week".
//
// The row is deliberately privacy-clean: it carries only the schema, an RFC3339
// UTC timestamp, the lane token, and the outcome bucket — never a path, hostname,
// title, or spine_ref — so a committed ledger cannot leak a private boundary
// (PUBLIC_LEAK). The package stays pure: the caller owns the clock (passes the
// invocation time) and the file (passes the io.Writer/Reader); this leaf only
// encodes a row and folds a stream, exactly as it only decides WHAT to fan out
// and leaves filing to the caller.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
)

// This leaf's failure contract (see failure_paths_test.go) is that the package
// constructs contract refusals ONLY via refusef, so ClassifyOutcome can bucket
// every rejection. The ledger's I/O and decode paths are not contract refusals —
// a write that fails or a corrupt row is a genuine, caller-facing error — so they
// return the UNDERLYING error unwrapped (never a fresh errors.New/fmt.Errorf),
// keeping the "errors only via refusef" invariant intact while still surfacing the
// real cause (a bad timestamp, a malformed line) to the caller.

// LedgerSchema identifies one durable invocation-ledger row.
const LedgerSchema = "fak.issue-fanout-ledger.v1"

// LedgerRow is one durable record of a single planner invocation. Its fields are
// the minimum a per-week adoption fold needs and nothing a committed file could
// leak: when it ran (UTC), which lane, and how it came out.
type LedgerRow struct {
	Schema  string  `json:"schema"`
	At      string  `json:"at"`      // RFC3339 UTC, e.g. "2026-07-07T13:04:05Z"
	Leaf    string  `json:"leaf"`    // lane token, e.g. "issuefanout" — never a path/host
	Outcome Outcome `json:"outcome"` // success | refused | error
}

// NewLedgerRow builds the durable row for one invocation: the caller supplies the
// invocation time (the package reads no clock) and the invocation's error, which
// ClassifyOutcome buckets exactly as the outcome fold does — a nil error is a
// success, a *Refusal a deliberate refusal, anything else an error. The timestamp
// is normalized to UTC so a committed ledger carries no local-zone (host) hint.
func NewLedgerRow(leaf string, at time.Time, err error) LedgerRow {
	return LedgerRow{
		Schema:  LedgerSchema,
		At:      at.UTC().Format(time.RFC3339),
		Leaf:    strings.TrimSpace(leaf),
		Outcome: ClassifyOutcome(err),
	}
}

// AppendRow encodes row as one JSONL line (trailing newline) to w — the durable
// append the done-condition asks for. The caller owns the file handle (open in
// append mode) and the clock; the package never touches disk, so the append seam
// is drop-in on any io.Writer (a file, a buffer, a test sink).
func AppendRow(w io.Writer, row LedgerRow) error {
	b, err := json.Marshal(row)
	if err != nil {
		return err
	}
	_, err = w.Write(append(b, '\n'))
	return err
}

// BuildLogged runs Build and appends its outcome as one durable ledger row — the
// one-call seam a verb or wave uses so every planner invocation leaves a durable
// record, not just narration. The caller passes the invocation time (pure: no
// clock read) and the ledger writer; a nil writer is a no-op append, so the seam
// is drop-in on any existing call site. It mirrors BuildInto, which folds the
// outcome into in-memory counts; the two compose (a caller can do both).
func BuildLogged(in Input, at time.Time, w io.Writer) (Plan, error) {
	plan, err := Build(in)
	if w != nil {
		// A row records that the invocation happened and how it came out; an
		// append failure must not mask the planner's own result, so it is
		// returned only when Build itself succeeded.
		if appErr := AppendRow(w, NewLedgerRow(in.Leaf, at, err)); appErr != nil && err == nil {
			return plan, appErr
		}
	}
	return plan, err
}

// WeekCount is one ISO-week bucket of invocation outcomes — the fold the done
// condition asks for ("counts per week").
type WeekCount struct {
	Week    string `json:"week"` // ISO year-week, e.g. "2026-W28"
	Success int    `json:"success"`
	Refused int    `json:"refused"`
	Error   int    `json:"error"`
	Total   int    `json:"total"`
}

// FoldWeekly reads a JSONL ledger stream and folds it into per-ISO-week outcome
// counts, oldest week first. It is the surface the ledger exists for: a durable
// answer to "was the verb used, and how did it go, week over week". A blank line
// is skipped; a malformed row or timestamp is a hard error naming the offending
// line, so a corrupt ledger surfaces instead of silently under-counting.
func FoldWeekly(r io.Reader) ([]WeekCount, error) {
	buckets := map[string]*WeekCount{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row LedgerRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, err
		}
		t, err := time.Parse(time.RFC3339, row.At)
		if err != nil {
			return nil, err
		}
		y, wk := t.UTC().ISOWeek()
		key := fmt.Sprintf("%04d-W%02d", y, wk)
		b := buckets[key]
		if b == nil {
			b = &WeekCount{Week: key}
			buckets[key] = b
		}
		switch row.Outcome {
		case OutcomeSuccess:
			b.Success++
		case OutcomeRefused:
			b.Refused++
		default:
			b.Error++
		}
		b.Total++
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	weeks := make([]WeekCount, 0, len(buckets))
	for _, b := range buckets {
		weeks = append(weeks, *b)
	}
	sort.Slice(weeks, func(i, j int) bool { return weeks[i].Week < weeks[j].Week })
	return weeks, nil
}

// RenderWeekly prints the per-week fold for a human: a headline total plus one line
// per week, mirroring Render/RenderOutcomes/RenderAdoption so a `fak issue fanout`
// adoption readout reads identically to the rest of the leaf.
func RenderWeekly(weeks []WeekCount) string {
	var b strings.Builder
	var s, r, e, tot int
	for _, w := range weeks {
		s += w.Success
		r += w.Refused
		e += w.Error
		tot += w.Total
	}
	fmt.Fprintf(&b, "issuefanout ledger: %d invocation(s) across %d week(s) (%d success, %d refused, %d error)\n",
		tot, len(weeks), s, r, e)
	for _, w := range weeks {
		fmt.Fprintf(&b, "  [%s] %d invocation(s): %d success, %d refused, %d error\n",
			w.Week, w.Total, w.Success, w.Refused, w.Error)
	}
	return b.String()
}
