// summary.go implements the interrupt-safe, always-written sweep ledger: one
// deterministic record of how every outcome of a sweep was adjudicated, with
// per-status counts, written in a single idempotent call so a deferred write
// (defer/finally) still yields a complete ledger when the sweep is interrupted
// mid-run.
package benchsweep

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// SummarySchema is the schema tag every ledger carries. A ledger whose tag is
// absent or unknown is refused rather than best-effort rendered: the tag tells
// a reader which fields were pinned.
const SummarySchema = "fak.benchsweep.summary/1"

// ReasonLedgerInconsistent is the closed token for a ledger that violates its
// own accounting contract.
const ReasonLedgerInconsistent = "LEDGER_INCONSISTENT"

// LedgerError is a typed ledger refusal carrying a token from a closed
// vocabulary, so a caller can branch on the reason instead of matching prose.
type LedgerError struct {
	Reason string
	Detail string
}

func (e *LedgerError) Error() string { return e.Reason + ": " + e.Detail }

// Paired binds one outcome to its chain verdict, in input order.
type Paired struct {
	Outcome Outcome
	Verdict Verdict
}

// Sweep runs every outcome through ch and returns the paired results in input
// order. Rescues are logged by the chain as they happen.
func Sweep(ch *Chain, outcomes []Outcome) []Paired {
	pairs := make([]Paired, len(outcomes))
	for i, o := range outcomes {
		pairs[i] = Paired{Outcome: o, Verdict: ch.Evaluate(o)}
	}
	return pairs
}

// Rejection is one refused outcome's ledger row: which combo, the status it
// carried, the closed reason token, and the decisive check.
type Rejection struct {
	ComboID string `json:"combo_id"`
	Status  string `json:"status"`
	Reason  string `json:"reason"`
	Check   string `json:"check"`
}

// Rescue is one rescued outcome's ledger row: carried flagged, never silent.
type Rescue struct {
	ComboID  string        `json:"combo_id"`
	Status   string        `json:"status"`
	Duration time.Duration `json:"duration"`
	Check    string        `json:"check"`
}

// Ledger is the complete sweep summary: every outcome accounted for exactly
// once, as accepted, rescued, or rejected, with per-status counts over the
// whole sweep.
type Ledger struct {
	Schema   string `json:"schema"`
	Total    int    `json:"total"`
	Accepted int    `json:"accepted"`
	Rescued  int    `json:"rescued"`
	Rejected int    `json:"rejected"`
	// ByStatus counts every outcome by its reported status, whatever its
	// verdict: the operator sees the whole sweep's status mix at a glance.
	ByStatus map[string]int `json:"by_status"`
	// Rescues lists every rescued outcome, flagged.
	Rescues []Rescue `json:"rescues,omitempty"`
	// Rejections lists every refused outcome with its reason token.
	Rejections []Rejection `json:"rejections,omitempty"`
}

// BuildLedger folds paired outcomes into the complete ledger. It is pure and
// deterministic: the same pairs always build the same ledger. Zero pairs yield
// a valid empty ledger (schema set, zero counts, empty by_status), not an
// error — a sweep that expanded to nothing still reports. A pair whose verdict
// kind falls outside the closed vocabulary is refused, because such a row
// could not be accounted for honestly.
func BuildLedger(pairs []Paired) (Ledger, error) {
	l := Ledger{Schema: SummarySchema, ByStatus: map[string]int{}}
	for _, p := range pairs {
		l.Total++
		l.ByStatus[p.Outcome.Status]++
		switch {
		case p.Verdict.IsAccepted():
			l.Accepted++
		case p.Verdict.IsRescued():
			l.Rescued++
			l.Rescues = append(l.Rescues, Rescue{
				ComboID:  p.Outcome.Combo.ID,
				Status:   p.Outcome.Status,
				Duration: p.Outcome.Duration,
				Check:    p.Verdict.Check,
			})
		case p.Verdict.IsRejected():
			l.Rejected++
			l.Rejections = append(l.Rejections, Rejection{
				ComboID: p.Outcome.Combo.ID,
				Status:  p.Outcome.Status,
				Reason:  p.Verdict.Reason,
				Check:   p.Verdict.Check,
			})
		default:
			return Ledger{}, &LedgerError{
				Reason: ReasonLedgerInconsistent,
				Detail: fmt.Sprintf("outcome %q has verdict kind %q outside the closed vocabulary", p.Outcome.Combo.ID, p.Verdict.Kind),
			}
		}
	}
	return l, nil
}

// Validate checks the ledger's closed accounting contract: the schema tag,
// non-negative counts, the two partition invariants (verdict counts and
// per-status counts each sum to Total), and that every detail row names a
// status present in ByStatus with a non-empty check and, for rejections, a
// non-empty reason token. A ledger that violated any of these would silently
// misreport a sweep, so it is refused, not best-effort rendered.
func (l Ledger) Validate() error {
	if l.Schema != SummarySchema {
		return &LedgerError{Reason: ReasonLedgerInconsistent, Detail: fmt.Sprintf("schema %q is not %q", l.Schema, SummarySchema)}
	}
	if l.Total < 0 || l.Accepted < 0 || l.Rescued < 0 || l.Rejected < 0 {
		return &LedgerError{Reason: ReasonLedgerInconsistent, Detail: "counts must be non-negative"}
	}
	if l.Accepted+l.Rescued+l.Rejected != l.Total {
		return &LedgerError{Reason: ReasonLedgerInconsistent, Detail: fmt.Sprintf(
			"verdict counts %d+%d+%d do not sum to total %d", l.Accepted, l.Rescued, l.Rejected, l.Total)}
	}
	sum := 0
	for _, n := range l.ByStatus {
		if n <= 0 {
			return &LedgerError{Reason: ReasonLedgerInconsistent, Detail: "per-status counts must be positive"}
		}
		sum += n
	}
	if sum != l.Total {
		return &LedgerError{Reason: ReasonLedgerInconsistent, Detail: fmt.Sprintf("per-status counts sum to %d, want total %d", sum, l.Total)}
	}
	for _, r := range l.Rescues {
		if r.Check == "" || l.ByStatus[r.Status] <= 0 {
			return &LedgerError{Reason: ReasonLedgerInconsistent, Detail: fmt.Sprintf("rescue row %q is not accounted for", r.ComboID)}
		}
	}
	for _, j := range l.Rejections {
		if j.Check == "" || j.Reason == "" || l.ByStatus[j.Status] <= 0 {
			return &LedgerError{Reason: ReasonLedgerInconsistent, Detail: fmt.Sprintf("rejection row %q is not accounted for", j.ComboID)}
		}
	}
	return nil
}

// Writer is bound to one ledger path. It exists so a sweep loop can arm the
// always-written guarantee in one line:
//
//	w := benchsweep.NewWriter(ledgerPath)
//	defer func() {
//		if l, err := benchsweep.BuildLedger(benchsweep.Sweep(chain, outcomes)); err == nil {
//			_ = w.Write(l)
//		}
//	}()
//
// Even when the sweep is interrupted mid-run, the deferred write emits the
// complete ledger over every outcome recorded so far.
type Writer struct {
	path string
}

// NewWriter binds a Writer to path.
func NewWriter(path string) *Writer { return &Writer{path: path} }

// Path returns the bound ledger path.
func (w *Writer) Path() string { return w.path }

// Write renders the ledger deterministically and installs it at the bound
// path in a single call. It is idempotent: repeated calls with the same
// ledger leave byte-identical files, and a call with a newer ledger replaces
// the old one wholesale (temp file plus rename, with the portable fallback of
// removing the destination first where rename cannot replace). Write refuses a
// ledger that fails Validate; filesystem failures come back as plain errors.
func (w *Writer) Write(l Ledger) error {
	if err := l.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(&l, "", "  ")
	if err != nil {
		return &LedgerError{Reason: ReasonLedgerInconsistent, Detail: fmt.Sprintf("render: %v", err)}
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(w.path), 0o755); err != nil {
		return fmt.Errorf("benchsweep: create ledger dir: %w", err)
	}
	if err := installFile(w.path, data); err != nil {
		return fmt.Errorf("benchsweep: write ledger %s: %w", w.path, err)
	}
	return nil
}

// installFile writes data to path as a single replacement: temp file in the
// same directory, sync, rename. Where rename cannot replace an existing file
// (Windows), the destination is removed first and the rename retried once.
func installFile(path string, data []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".benchsweep-ledger-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	cleanup := func() { _ = os.Remove(tmpName) }
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		cleanup()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		cleanup()
		return err
	}
	if err := tmp.Close(); err != nil {
		cleanup()
		return err
	}
	if err := os.Rename(tmpName, path); err == nil {
		return nil
	}
	// Portable fallback: rename over an existing destination fails on
	// Windows; remove it and retry once.
	if rmErr := os.Remove(path); rmErr != nil && !os.IsNotExist(rmErr) {
		cleanup()
		return err
	}
	if err2 := os.Rename(tmpName, path); err2 != nil {
		cleanup()
		return err2
	}
	return nil
}
