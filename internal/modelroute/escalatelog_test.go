package modelroute

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func mustAppendEscalations(t *testing.T, recs ...EscalationRecord) string {
	t.Helper()
	var buf bytes.Buffer
	for _, r := range recs {
		if err := AppendEscalation(&buf, r); err != nil {
			t.Fatal(err)
		}
	}
	return buf.String()
}

func tallyOf(t *testing.T, raw string) EscalationTally {
	t.Helper()
	recs, stats, err := ReadEscalations(strings.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	return TallyEscalations(recs, stats)
}

// A rung bought must still be a rung bought after a restart. The whole point of
// AfterAttempt's budget is that it bounds a work item across attempts, and an actuator that
// recounted from memory would hand it 0 forever — an unbounded ladder that looks bounded
// from inside any single tick.
func TestASpentRungSurvivesTheProcessThatSpentIt(t *testing.T) {
	raw := mustAppendEscalations(t,
		EscalationRecord{ID: "a", Item: "5416", From: ZoneDevice, To: ZoneFleet, Reason: ReasonEarnedByUnderpower, At: time.Now()},
		EscalationRecord{ID: "b", Item: "5416", From: ZoneFleet, To: ZoneVendor, Reason: ReasonEarnedByUnderpower},
		EscalationRecord{ID: "c", Item: "9999", From: ZoneDevice, To: ZoneFleet},
	)
	tally := tallyOf(t, raw)
	if got := tally.Spent("5416"); got != 2 {
		t.Errorf("spent(5416) = %d, want 2", got)
	}
	if got := tally.Spent("9999"); got != 1 {
		t.Errorf("spent(9999) = %d, want 1", got)
	}
	if got := tally.Spent("never-escalated"); got != 0 {
		t.Errorf("spent(never-escalated) = %d, want 0", got)
	}
	// The key is trimmed on both sides of the ledger. A caller that passes a padded item id
	// must not be handed a budget of zero for an item that has already spent its rungs.
	if got := tally.Spent("  5416\t"); got != 2 {
		t.Errorf("spent(padded) = %d, want 2", got)
	}
}

// The inversion this file exists for. The evidence journal DROPS a row it cannot parse,
// because a lost credit can only weaken a grade. A lost DEBIT hides money already spent, so
// the ledger charges what it cannot read — to every item, since it cannot tell whose it was.
func TestADebitTheLedgerCannotReadIsStillCharged(t *testing.T) {
	raw := mustAppendEscalations(t, EscalationRecord{ID: "a", Item: "5416"}) +
		"{\"item\":\"5416\",\"fro\n" + // torn
		mustAppendEscalations(t, EscalationRecord{ID: "c", Item: "9999"})

	tally := tallyOf(t, raw)
	if tally.Unattributable != 1 {
		t.Fatalf("unattributable = %d, want 1 — the torn row was dropped like a credit", tally.Unattributable)
	}
	// Charged to BOTH items, including the one whose own rows are all readable: the ledger
	// does not know who spent it, and guessing "not this one" is how a budget leaks.
	if got := tally.Spent("5416"); got != 2 {
		t.Errorf("spent(5416) = %d, want 2", got)
	}
	if got := tally.Spent("9999"); got != 2 {
		t.Errorf("spent(9999) = %d, want 2", got)
	}
	// Including an item with no rows of its own — an unreadable debit could have been its.
	if got := tally.Spent("fresh-item"); got != 1 {
		t.Errorf("spent(fresh-item) = %d, want 1", got)
	}
}

// A row that parses but names no item is the same problem in a different costume, and gets
// the same answer. Reporting it as a diagnostic and counting it as zero would be a silent
// discount on exactly the rows a broken producer emits.
func TestAnUnownedDebitIsChargedLikeAnUnreadableOne(t *testing.T) {
	raw := mustAppendEscalations(t,
		EscalationRecord{ID: "a", Item: "5416"},
		EscalationRecord{ID: "b", Item: "   "},
		EscalationRecord{ID: "c"},
	)
	tally := tallyOf(t, raw)
	if tally.Unattributable != 2 {
		t.Errorf("unattributable = %d, want 2", tally.Unattributable)
	}
	if got := tally.Spent("5416"); got != 3 {
		t.Errorf("spent(5416) = %d, want 3", got)
	}
}

// Deduplication runs in the SAFE direction here and is still worth doing: a double-writing
// producer would otherwise exhaust every budget and look like a fleet doing hard work. The
// count is reported so the two are distinguishable.
func TestARepeatedDebitIsCountedOnceAndReported(t *testing.T) {
	raw := mustAppendEscalations(t,
		EscalationRecord{ID: "a", Item: "5416"},
		EscalationRecord{ID: "a", Item: "5416"},
		EscalationRecord{ID: "a", Item: "5416"},
	)
	tally := tallyOf(t, raw)
	if got := tally.Spent("5416"); got != 1 {
		t.Errorf("spent = %d, want 1", got)
	}
	if tally.Duplicates != 2 {
		t.Errorf("duplicates = %d, want 2", tally.Duplicates)
	}
}

// A row with no id cannot be deduplicated, so it counts on its own. That is the conservative
// direction for a debit, and the opposite of what the evidence journal does with an
// un-deduplicable credit (which it reports, because there it inflates a claim).
func TestAnUnidentifiedDebitCountsOnItsOwn(t *testing.T) {
	raw := mustAppendEscalations(t,
		EscalationRecord{Item: "5416"},
		EscalationRecord{Item: "5416"},
	)
	tally := tallyOf(t, raw)
	if got := tally.Spent("5416"); got != 2 {
		t.Errorf("spent = %d, want 2", got)
	}
	if tally.Duplicates != 0 {
		t.Errorf("duplicates = %d, want 0 — nothing identified them as repeats", tally.Duplicates)
	}
}

// The tally is the number AfterAttempt is bounded by, so the two have to agree in practice
// and not just on paper: a ledger holding the item's whole budget must stop it.
func TestTheLedgerActuallyStopsTheLadderItBounds(t *testing.T) {
	bounds := EscalationBounds{Ceiling: ZoneVendor, MaxAttempts: 2}
	underpowered := AttemptResult{Fail: FailUnderpowered}
	at := Placement{Zone: ZoneDevice}

	empty := tallyOf(t, "")
	if v := AfterAttempt(at, underpowered, bounds, empty.Spent("5416")); !v.Escalates() {
		t.Fatalf("a fresh item did not escalate: %+v", v)
	}

	spent := tallyOf(t, mustAppendEscalations(t,
		EscalationRecord{ID: "a", Item: "5416"},
		EscalationRecord{ID: "b", Item: "5416"},
	))
	v := AfterAttempt(at, underpowered, bounds, spent.Spent("5416"))
	if v.Escalates() {
		t.Errorf("an item at its budget escalated anyway: %+v", v)
	}
	if v.Reason != ReasonBudgetSpent {
		t.Errorf("reason = %q, want %q", v.Reason, ReasonBudgetSpent)
	}

	// And one torn row is enough to stop an item that had one rung left — the safe
	// direction, and visible in Unattributable rather than silent.
	nearly := tallyOf(t, mustAppendEscalations(t, EscalationRecord{ID: "a", Item: "5416"})+"{torn\n")
	if v := AfterAttempt(at, underpowered, bounds, nearly.Spent("5416")); v.Escalates() {
		t.Errorf("an unreadable debit bought a rung: %+v", v)
	}
}

// The ledger round-trips through the format it claims to write, so a reader change that
// silently stops parsing the writer's output cannot pass.
func TestTheLedgerReadsWhatItWrites(t *testing.T) {
	want := EscalationRecord{
		ID: "e1", Item: "5416", From: ZoneDevice, To: ZoneFleet,
		Reason: ReasonEarnedByUnderpower, At: time.Now().UTC().Truncate(time.Second),
	}
	recs, stats, err := ReadEscalations(strings.NewReader(mustAppendEscalations(t, want)))
	if err != nil {
		t.Fatal(err)
	}
	if stats.Lines != 1 || stats.Malformed != 0 {
		t.Errorf("stats = %+v, want one clean line", stats)
	}
	if len(recs) != 1 {
		t.Fatalf("round trip returned %d records, want 1", len(recs))
	}
	got := recs[0]
	if !got.At.Equal(want.At) {
		t.Errorf("at = %v, want %v", got.At, want.At)
	}
	got.At, want.At = time.Time{}, time.Time{}
	if got != want {
		t.Errorf("round trip = %+v, want %+v", got, want)
	}
}

// Blank lines are not debits. A ledger a fleet has been appending to for months collects
// them, and charging one would slowly disable the ladder for reasons nobody could find.
func TestBlankLinesAreNotDebits(t *testing.T) {
	raw := "\n  \n" + mustAppendEscalations(t, EscalationRecord{ID: "a", Item: "5416"}) + "\n\n"
	tally := tallyOf(t, raw)
	if got := tally.Spent("5416"); got != 1 {
		t.Errorf("spent = %d, want 1", got)
	}
	if tally.Unattributable != 0 {
		t.Errorf("unattributable = %d, want 0", tally.Unattributable)
	}
}

// An oversized line is a content problem, not a reader failure — same rule as the evidence
// journal, since they share one scanner. It is charged, because here it is a debit.
func TestAnOversizedRowIsChargedRatherThanFatal(t *testing.T) {
	raw := "{\"item\":\"5416\",\"reason\":\"" + strings.Repeat("x", maxOutcomeLine+16) + "\"}\n"
	recs, stats, err := ReadEscalations(strings.NewReader(raw))
	if err != nil {
		t.Fatalf("an oversized row was fatal: %v", err)
	}
	if len(recs) != 0 || stats.Malformed != 1 {
		t.Errorf("recs=%d stats=%+v, want zero records and one malformed", len(recs), stats)
	}
	if got := TallyEscalations(recs, stats).Spent("5416"); got != 1 {
		t.Errorf("spent = %d, want 1 — an oversized debit went uncharged", got)
	}
}
