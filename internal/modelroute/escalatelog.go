package modelroute

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// THE DURABLE SPEND LEDGER: the record `AfterAttempt`'s escalation budget is counted from
// (epic #5416, track D).
//
// AfterAttempt is bounded by `priorEscalations` — how many rungs this work item has already
// bought — and it takes that number from the caller, because only the caller knows what a
// work item is. Nothing produced it. An actuator that recomputed it from memory would reset
// the budget on every process restart, which is not a bounded ladder at all: it is an
// unbounded one with a short memory, and the failure is invisible because each individual
// tick looks correctly bounded.
//
// This is the same shape as the evidence journal in evidencelog.go — append-only, one JSON
// line per event, torn lines tolerated — and its READING RULES ARE THE MIRROR IMAGE. That
// inversion is the whole reason this is a separate file rather than another row type over
// there, and it is worth stating plainly because the code looks almost identical:
//
//	evidence journal            spend ledger
//	------------------------    ------------------------------------
//	a row is a CREDIT           a row is a DEBIT
//	losing one weakens a        losing one HIDES money already spent
//	  grade (safe)                (unsafe)
//	counting one twice          counting one twice ends the budget
//	  manufactures a grade        early (safe)
//	  (unsafe)
//
// So the evidence journal drops what it cannot use and reports the count, and this one
// CHARGES what it cannot use. A row that exists and does not parse is proof that something
// was recorded; the only safe reading of a debit whose owner is unknown is that it belongs
// to the item currently asking. See EscalationTally.Spent.
//
// The alternative — refusing to count at all while any row is unreadable — was rejected on
// purpose. A torn line in an append-only file is permanent (the next append lands after it,
// it is never repaired), so "refuse while torn" is "never escalate again", and a fleet whose
// escalation ladder silently switched off for good would look exactly like one whose work
// simply stopped needing bigger models.
//
// Pure and stdlib-only, like the rest of this leaf: writing the rows is the caller's job.

// EscalationRecord is ONE rung bought: which work item, which rung it moved between, and the
// closed reason `AfterAttempt` gave for authorising it.
//
// Item is an opaque caller key — an issue number, a task id, a run id. This package does not
// know what a work item is and deliberately does not learn: the same rule that makes
// AfterAttempt take priorEscalations as a parameter.
type EscalationRecord struct {
	// ID is the dedupe key — anything stable per escalation event. A row without one cannot
	// be deduplicated and is therefore counted on its own, which is the conservative
	// direction for a debit.
	ID string `json:"id,omitempty"`
	// Item is the work item that spent the rung.
	Item string `json:"item"`
	// From and To are the rungs it moved between, for a legible ledger.
	From PlacementZone `json:"from,omitempty"`
	To   PlacementZone `json:"to,omitempty"`
	// Reason is AfterAttempt's closed reason token. Free text is never written here.
	Reason string `json:"reason,omitempty"`
	// At is when the rung was bought. Optional; this ledger has no freshness window,
	// because a budget that expired quietly would be no budget at all.
	At time.Time `json:"at,omitempty"`
}

// AppendEscalation writes one escalation as a single JSON line.
//
// The caller's contract is the strict one: append BEFORE launching the escalated attempt,
// and do not launch if the append fails. A rung bought without a recorded debit is a rung
// outside the budget, and the whole point of the budget is that nothing is outside it.
func AppendEscalation(w io.Writer, e EscalationRecord) error {
	b, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("marshal escalation: %w", err)
	}
	if _, err := w.Write(append(b, '\n')); err != nil {
		return fmt.Errorf("append escalation: %w", err)
	}
	return nil
}

// ReadEscalations parses a spend ledger, reporting rather than hiding what it could not use.
//
// Same contract as ReadTurnOutcomes: the error is reserved for a failure of the READER, and
// content problems come back in JournalStats. What the caller must do with Malformed is the
// opposite here, and TallyEscalations already does it — read the tally, not the raw stats.
func ReadEscalations(r io.Reader) ([]EscalationRecord, JournalStats, error) {
	var out []EscalationRecord
	stats, err := scanJSONLines(r, func(line []byte) bool {
		var e EscalationRecord
		if json.Unmarshal(line, &e) != nil {
			return false
		}
		out = append(out, e)
		return true
	})
	if err != nil {
		return out, stats, fmt.Errorf("read escalation ledger: %w", err)
	}
	return out, stats, nil
}

// EscalationTally is what the ledger says each work item has already spent.
type EscalationTally struct {
	// ByItem is the deduplicated count of rungs bought, per work item.
	ByItem map[string]int `json:"by_item,omitempty"`
	// Unattributable is the number of debits the ledger holds but cannot assign: rows that
	// did not parse, and rows that parsed with no Item. It is NOT a diagnostic — Spent adds
	// it to every item's count, because a debit whose owner is unknown must be assumed to be
	// this one's or the budget is not a bound.
	Unattributable int `json:"unattributable,omitempty"`
	// Duplicates is how many rows were dropped as repeats of an id already seen. Reported so
	// an operator can tell a double-writing producer from a genuinely busy ladder.
	Duplicates int `json:"duplicates,omitempty"`
}

// Spent is the number to hand AfterAttempt as priorEscalations.
//
// It charges the unattributable rows to every item. That over-counts when a fleet has torn
// rows and several active items, and over-counting a budget can only stop an escalation that
// might have been allowed — an operator sees an item that would not escalate and a non-zero
// Unattributable telling them exactly why. Under-counting would instead spend money nobody
// authorised, silently, and would look correct from every angle.
func (t EscalationTally) Spent(item string) int {
	return t.ByItem[strings.TrimSpace(item)] + t.Unattributable
}

// TallyEscalations folds a ledger read into per-item spend.
//
// stats comes from ReadEscalations, so the rows that never became records are charged too:
// this function is the only place that sees both halves, and splitting the judgement across
// two call sites is how a caller ends up counting the parsed rows and forgetting the torn
// ones.
func TallyEscalations(records []EscalationRecord, stats JournalStats) EscalationTally {
	t := EscalationTally{ByItem: map[string]int{}, Unattributable: stats.Malformed}
	seen := map[string]bool{}
	for _, e := range records {
		if id := strings.TrimSpace(e.ID); id != "" {
			if seen[id] {
				t.Duplicates++
				continue
			}
			seen[id] = true
		}
		item := strings.TrimSpace(e.Item)
		if item == "" {
			// It happened, and the ledger does not say to whom. Charged to everyone, for the
			// same reason a torn row is.
			t.Unattributable++
			continue
		}
		t.ByItem[item]++
	}
	return t
}
