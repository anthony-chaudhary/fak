package trendreport

import "encoding/json"

// AppendLedgerLine serializes a ledger row into a JSONL string without trailing newline.
func AppendLedgerLine[T any](row T) (string, error) {
	b, err := json.Marshal(row)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// DirectionWord converts a signed integer difference into a trend word (up, down, flat).
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

// Envelope defines the embeddable common header for report metadata and gate outcomes.
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
	GateExit    *int   `json:"gate_exit,omitempty"`
	GateMessage string `json:"gate_message,omitempty"`
}

// Opts encapsulates workspace and timestamp metadata stamped into a report envelope.
type Opts struct {
	Workspace   string
	Commit      string
	GeneratedAt string
	Date        string
}

// Stamp initializes a new Envelope populated with the given schema and ambient options.
func Stamp(schema string, opts Opts) Envelope {
	return Envelope{
		Schema:      schema,
		Workspace:   opts.Workspace,
		Commit:      opts.Commit,
		GeneratedAt: opts.GeneratedAt,
		Date:        opts.Date,
	}
}

// Verdict constants specify closed outcomes for report gates.
const (
	VerdictOK     = "OK"
	VerdictAction = "ACTION"
)

// GateVerdict represents an advisory gate evaluation with exit code and message.
type GateVerdict struct {
	Exit    int
	Message string
}

// AdvisoryGate evaluates report findings and returns exit 1 only for unmeasured states.
func AdvisoryGate(label, finding, reason, unmeasuredFinding string) GateVerdict {
	if finding == unmeasuredFinding {
		return GateVerdict{Exit: 1, Message: label + " INCOMPLETE: " + reason}
	}
	return GateVerdict{Exit: 0, Message: label + " OK: " + reason}
}

// WithGate creates a copy of the envelope updated with advisory gate evaluation results.
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
