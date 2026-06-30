package trendreport

// trendreport.go is the generic, consumer-agnostic substrate three trend-reports
// (internal/cadencereport, internal/milestonereport, and the dojo board) currently
// re-declare verbatim: the durable-JSONL ledger plumbing (parse / latest-prior /
// append-line), the per-tick direction word, the embeddable control-pane envelope,
// and the advisory gate whose only failing finding is the caller's *_unmeasured
// token. It imports nothing internal (stdlib + generics only) so it can sit at the
// foundation tier and a fourth report is authored without copy-paste.
//
// Migrating the existing consumers onto this substrate is a documented FOLLOW-ON
// (#1437): this lane only CREATES the shared spine + its tests. The consumers stay
// byte-identical until they are switched over in a later, behavior-preserving wave.

import (
	"bufio"
	"encoding/json"
	"sort"
	"strings"
)

// Row is the one thing a durable ledger line must expose for the generic ledger
// plumbing to order it: the (date, generated_at) key the parser keeps a line on
// and latestBefore sorts by. cadencereport.LedgerRow and milestonereport.LedgerRow
// both already carry these fields; a consumer satisfies Row with a one-line method.
//
// Date is the coarse ordering key (a YYYY-MM-DD tick); GeneratedAt breaks a
// same-day tie so a same-day re-run trends against the earlier same-day tick. A
// row whose Date is empty is treated as not-a-row by ParseLedger (a hand-edit
// can't crash the reader).
type Row interface {
	Key() (date, generatedAt string)
}

// ParseLedger parses an append-only JSONL ledger into []T, tolerating blank lines
// and skipping any line that is not a valid row OR carries no Date (so a hand-edit
// can't crash the reader). Rows are returned in file order. T must be a concrete
// row type (not a pointer) that satisfies Row; the parser unmarshals each line into
// a fresh T and keeps it only when its Date key is non-empty.
//
// This is the generic form of the identical ParseLedger both cadencereport and
// milestonereport hand-roll: same scanner buffer, same blank-line + bad-line +
// empty-Date tolerance, lifted to be row-type-agnostic.
func ParseLedger[T Row](content string) []T {
	var rows []T
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row T
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if date, _ := row.Key(); date == "" {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// AppendLedgerLine renders the JSONL line for a row (no trailing newline). The
// caller appends it to the ledger file with a newline; keeping the rendering pure
// makes the writer testable without touching disk. Generic over any row type so
// every report shares one marshaller.
func AppendLedgerLine[T any](row T) (string, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// LatestBefore returns the most recent prior row relative to `row`, comparing by
// (date, then generated_at) so a same-day re-run trends against the earlier
// same-day tick. A prior row with the exact same generated_at as `row` is excluded
// (idempotent re-append). With no eligible prior the second result is false — the
// caller renders that as the "new" / first-tick trend.
//
// This is the generic form of the byte-identical latestBefore both consumers
// hand-roll; it operates on the Row key, never on a concrete field name.
func LatestBefore[T Row](row T, prior []T) (T, bool) {
	_, rowGen := row.Key()
	cands := make([]T, 0, len(prior))
	for _, p := range prior {
		_, pGen := p.Key()
		if pGen != "" && pGen == rowGen {
			continue
		}
		cands = append(cands, p)
	}
	if len(cands) == 0 {
		var zero T
		return zero, false
	}
	sort.SliceStable(cands, func(i, j int) bool {
		di, gi := cands[i].Key()
		dj, gj := cands[j].Key()
		if di != dj {
			return di < dj
		}
		return gi < gj
	})
	return cands[len(cands)-1], true
}

// DirectionWord renders the sign of a per-tick integer delta as a trend word
// (up | down | flat). Shared by the per-dimension delta lines across the reports.
func DirectionWord(delta int) string {
	switch {
	case delta > 0:
		return "up"
	case delta < 0:
		return "down"
	default:
		return "flat"
	}
}

// DirectionWordF is the float form of DirectionWord, for dimensions whose delta is
// a percentage (the milestone climb/roadmap deltas). A zero delta — including a
// tiny one that rounds to zero — is "flat".
func DirectionWordF(delta float64) string {
	switch {
	case delta > 0:
		return "up"
	case delta < 0:
		return "down"
	default:
		return "flat"
	}
}

// Envelope is the embeddable common head every trend-report's Report struct
// re-declares verbatim: the schema/ok/verdict/finding/reason/next-action triple,
// the ambient (workspace, commit, generated_at, date) stamp, and the two gate
// fields set only for the --check --json envelope. A consumer embeds it and adds
// its own dimension fields:
//
//	type Report struct {
//	    trendreport.Envelope
//	    Scores Scores `json:"scores"`
//	    ...
//	}
//
// The json tags match the fields the existing two reports already emit, so a
// consumer that embeds Envelope produces the same envelope JSON it does today.
type Envelope struct {
	Schema      string `json:"schema"`
	OK          bool   `json:"ok"`
	Verdict     string `json:"verdict"`
	Finding     string `json:"finding"`
	Reason      string `json:"reason"`
	NextAction  string `json:"next_action"`
	Workspace   string `json:"workspace"`
	Commit      string `json:"commit"`
	GeneratedAt string `json:"generated_at"`
	Date        string `json:"date"`
	// gate fields, set only for the --check --json envelope.
	GateExit    *int   `json:"gate_exit,omitempty"`
	GateMessage string `json:"gate_message,omitempty"`
}

// Opts carries the ambient context the fold stamps onto an Envelope. It is the
// generic form of each report's FoldOpts; Stamp applies it.
type Opts struct {
	Workspace   string
	Commit      string
	GeneratedAt string
	Date        string
}

// Stamp returns an Envelope seeded with the schema and the ambient context. A
// consumer's Fold calls Stamp once, then sets OK/Verdict/Finding/Reason/NextAction
// from its own dimension logic.
func Stamp(schema string, opts Opts) Envelope {
	return Envelope{
		Schema:      schema,
		Workspace:   opts.Workspace,
		Commit:      opts.Commit,
		GeneratedAt: opts.GeneratedAt,
		Date:        opts.Date,
	}
}

// Verdict constants are the closed report-envelope verdict vocabulary the gate
// reconciles to. They are advisory verdicts (mirror, not a second quality gate):
// OK records the tick; ACTION marks an INCOMPLETE report (a dimension could not be
// measured), never a quality regression.
const (
	VerdictOK     = "OK"
	VerdictAction = "ACTION"
)

// GateVerdict is one advisory-gate decision: the process exit code plus the human
// message. It is the generic return of each report's CheckGate.
type GateVerdict struct {
	Exit    int
	Message string
}

// AdvisoryGate is the shared advisory CI gate over a folded report. It fails ONLY
// when the report's Finding is the caller's `unmeasuredFinding` token — a dimension
// could not be measured, so the report itself is incomplete. Every other finding
// (a recorded tick, a score/climb-regression advisory) passes: a trend report is a
// MIRROR, not a second quality gate — the scorecard ratchet owns debt regressions.
//
// `label` is the report's short upper-case name ("CADENCE", "MILESTONE", ...). The
// returned message is `<LABEL> INCOMPLETE: <reason>` on exit 1 and
// `<LABEL> OK: <reason>` on exit 0, matching the two existing reports' wording.
func AdvisoryGate(label, finding, reason, unmeasuredFinding string) GateVerdict {
	if finding == unmeasuredFinding {
		return GateVerdict{Exit: 1, Message: label + " INCOMPLETE: " + reason}
	}
	return GateVerdict{Exit: 0, Message: label + " OK: " + reason}
}

// WithGate returns a copy of the envelope reconciled to a gate decision, for the
// --check --json envelope: OK + Verdict follow the exit code, and the two gate
// fields are populated. It is the generic form of each report's
// (Report).WithGate, lifted to the embedded Envelope.
func (e Envelope) WithGate(v GateVerdict) Envelope {
	q := e
	q.OK = v.Exit == 0
	if v.Exit == 0 {
		q.Verdict = VerdictOK
	} else {
		q.Verdict = VerdictAction
	}
	c := v.Exit
	q.GateExit = &c
	q.GateMessage = v.Message
	return q
}
