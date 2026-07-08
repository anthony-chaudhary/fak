// Rung D3 (issue #1879): the ledger-verified progress read. Track D re-verifies a baton on
// reload — D1 (reload.go) re-checks the cursor's git anchor (start_sha); this rung reads the
// OTHER half of the cursor, its ledger_ref (baton.go's ProgressCursor.LedgerRef), and exposes
// the successor's progress as the intent/run ledger ACTUALLY RECORDS it — never a number the
// closing leg asserted. The whole point of the no-`claimed`-field invariant (baton.go,
// TestBatonHasNoClaimedField) is that progress is a re-verifiable cursor; this file makes the
// read that turns that cursor into progress carry the SAME shape, so a self-report has
// nowhere to enter even at read time. It mirrors the fail-closed A2A digest shape of
// dos_status (CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md): a pointer to VERIFIED progress,
// structurally never a self-report.
//
// Reuse, not a new format. Progress comes from the EXISTING run/intent ledger — a JSONL row
// stream (issue "Out of scope: no new ledger format"). ParseLedgerProgress projects it with
// the shared jsonlledger.Parse reader every report package already uses, dropping unknown
// fields exactly as internal/resume/outcome.go's Attempt does; this file adds no on-disk
// format and no schema beyond the two fields a progress row must carry to be re-readable.
//
// Pure + fail-closed, like its siblings. ReadVerifiedProgress reads through an INJECTED
// LedgerReader (the resolve.go discipline: the store is a probe, so the read is unit-testable
// without a live ledger). An empty ledger_ref (the cursor names no ledger) and a ledger the
// reader cannot reach both yield ProgressUnknown with NO steps — the successor is handed
// nothing it cannot re-verify, never a trusted absence. Wiring a file-backed LedgerReader
// into the live reload/driver path is a later rung; here the seam and its contract test are
// the artifact.
package relay

import "github.com/anthony-chaudhary/fak/internal/jsonlledger"

// ProgressVerdict is the closed outcome of reading verified progress from the intent ledger
// via a cursor. Like dos_status it states only what the ledger says — never a self-report.
type ProgressVerdict string

const (
	// ProgressVerified means the ledger was reachable through the cursor's ledger_ref and the
	// progress steps were read FROM it. It is the only verdict that carries steps.
	ProgressVerified ProgressVerdict = "verified"
	// ProgressUnknown means no verified progress could be read: the cursor names no ledger
	// (empty ledger_ref) or the reader could not reach the ledger. Fail-closed — an unreadable
	// ledger is never reported as "zero progress verified", only as unknown, so a successor
	// re-derives from durable state rather than trusting an absence it cannot check.
	ProgressUnknown ProgressVerdict = "unknown"
)

// ProgressStep is one verified progress event AS THE LEDGER RECORDS IT: a durable pointer the
// successor can re-read and a short display note. It is never a claim the baton asserted — it
// is a row that already exists in the run/intent ledger. There is no percentage and no "done"
// boolean a closing leg could set; a consumer counts steps or re-resolves their Refs, it does
// not read a self-reported number.
type ProgressStep struct {
	// Ref is the durable store-native pointer the ledger row records (a commit SHA, an issue
	// "#1234", a memory slug, a path) — the thing a successor re-reads to re-verify this step
	// actually happened.
	Ref string `json:"ref"`
	// Note is a short, display-only ledger note. Like a tombstone note it can explain the step
	// but is never consumed as progress; the Ref is the verifiable half.
	Note string `json:"note,omitempty"`
}

// VerifiedProgress is the typed result of reading progress through the cursor: the verdict,
// the ledger_ref that drove it, the steps read FROM the ledger, and a display reason. It
// mirrors the dos_status fail-closed shape and shares the baton's load-bearing invariant —
// there is NO `claimed` field anywhere in this type tree (TestVerifiedProgressHasNoClaimedField
// asserts it reflectively). Progress is ONLY ever the ledger's own rows; there is no place a
// closing leg could write a number.
type VerifiedProgress struct {
	// Verdict is verified only when the ledger was read through the cursor; unknown otherwise.
	Verdict ProgressVerdict `json:"verdict"`
	// LedgerRef echoes the cursor's ledger_ref that was read (empty when the cursor named none).
	LedgerRef string `json:"ledger_ref,omitempty"`
	// Steps are the verified progress events read from the ledger, in ledger order. Nil on
	// ProgressUnknown — no verified progress means no steps, never a fabricated zero row.
	Steps []ProgressStep `json:"steps,omitempty"`
	// Reason is a short, display-only explanation of the verdict — never consumed as progress.
	Reason string `json:"reason"`
}

// LedgerReader re-reads the intent/run ledger row(s) a cursor's ledger_ref names and returns
// the verified progress steps recorded there. It is INJECTED per-store, like the C3 Resolver
// (resolve.go): a hermetic reader drives the contract test; a file-backed reader that reuses
// ParseLedgerProgress over the durable ledger is the production wiring. A non-nil error means
// the store was unreachable (-> ProgressUnknown, fail-closed) and must not be used to signal
// "no progress" — an empty step slice with a nil error is the honest "ledger read, nothing
// recorded yet" answer.
type LedgerReader interface {
	ReadProgress(ledgerRef string) ([]ProgressStep, error)
}

// ReadVerifiedProgress reads the successor's progress from the intent ledger through the
// cursor's ledger_ref and the injected reader, and exposes it in the no-`claimed` shape.
// Fail-closed on both edges: an empty ledger_ref (the cursor pins no ledger anchor) and a
// reader error (the ledger is unreachable) each return ProgressUnknown with no steps — the
// successor gets only progress it can re-verify, never a trusted absence. On a successful read
// the returned Steps are EXACTLY the ledger's rows, in order, which is the "matches the
// ledger" property the done condition pins. Pure over the injected reader: no clock, and I/O
// only through the reader.
func ReadVerifiedProgress(cur ProgressCursor, lr LedgerReader) VerifiedProgress {
	if cur.LedgerRef == "" {
		return VerifiedProgress{
			Verdict: ProgressUnknown,
			Reason:  "progress_cursor.ledger_ref is empty; no intent-ledger anchor to read verified progress from",
		}
	}
	steps, err := lr.ReadProgress(cur.LedgerRef)
	if err != nil {
		return VerifiedProgress{
			Verdict:   ProgressUnknown,
			LedgerRef: cur.LedgerRef,
			Reason:    "intent ledger could not be read; failing closed rather than trusting an absence: " + err.Error(),
		}
	}
	return VerifiedProgress{
		Verdict:   ProgressVerified,
		LedgerRef: cur.LedgerRef,
		Steps:     steps,
		Reason:    "verified progress read from intent ledger " + cur.LedgerRef,
	}
}

// ledgerProgressRow is the relay's minimal projection of one run/intent-ledger JSONL row: the
// two fields a progress read consumes. Every other column the ledger carries is dropped
// (jsonlledger.Parse unmarshals into this type, so unknown fields are ignored) — the same
// "keep your own row type, drop the rest" discipline internal/resume/outcome.go's Attempt
// uses. A row with no ref points at nothing re-readable and is filtered out, so a malformed or
// header line cannot inject a phantom step.
type ledgerProgressRow struct {
	Ref  string `json:"ref"`
	Note string `json:"note"`
}

// ParseLedgerProgress projects a run/intent ledger's JSONL content into ordered progress
// steps, reusing the shared jsonlledger.Parse reader (no new format, no I/O — the caller
// supplies the already-read bytes). Blank and malformed lines are skipped by the parser; a row
// with an empty ref is dropped here (nothing to re-read). This is the reuse half of the
// issue's "reuse the existing run/intent ledger": a file-backed LedgerReader reads the store
// and hands its bytes here.
func ParseLedgerProgress(content string) []ProgressStep {
	rows := jsonlledger.Parse(content, func(r ledgerProgressRow) bool {
		return r.Ref != ""
	})
	steps := make([]ProgressStep, 0, len(rows))
	for _, r := range rows {
		steps = append(steps, ProgressStep{Ref: r.Ref, Note: r.Note})
	}
	return steps
}
