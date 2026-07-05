package trendreport

// trendreport.go is the generic, consumer-agnostic ENVELOPE substrate the fak
// trend-reports (internal/cadencereport, internal/milestonereport, the dojo
// board) used to re-declare verbatim: the embeddable control-pane envelope, the
// advisory gate whose only failing finding is the caller's *_unmeasured token,
// the per-tick direction word, and the JSONL append-line marshaller. The ledger
// READ plumbing (parse / latest-prior) lives in internal/jsonlledger — the
// substrate the consumers already delegate to — so it is deliberately NOT
// re-declared here. This package imports nothing internal (stdlib + generics
// only) so it sits at the foundation tier and a fourth report is authored
// without copy-paste.

import "encoding/json"

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
