package metrics

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

const crashTTL int64 = 900

// crashStorm builds n CHILD_CRASH events sharing one cause, spaced a second apart so
// the newest observation instant is deterministic.
func crashStorm(n int, class string, globs []string, hash string, startUnix int64) []knownbad.CrashEvent {
	out := make([]knownbad.CrashEvent, 0, n)
	for i := 0; i < n; i++ {
		out = append(out, knownbad.CrashEvent{
			ReasonClass: class,
			TreeGlobs:   globs,
			FailureHash: hash,
			AtUnix:      startUnix + int64(i),
			ObservedBy:  "worker-1",
		})
	}
	return out
}

// ACCEPTANCE (#3586), the ticketing half: 300 CHILD_CRASH events sharing a signature
// cost ONE issue carrying the count — not 300 issues — and the readout says so in the
// words the operator needs: "300 crashes, 1 cause".
func TestFoldCrashSpamFilesOneTicketForOneCause(t *testing.T) {
	const now int64 = 1_000_000
	// A real coalescer pass, not a hand-built row: the two halves must agree.
	ledger, stats := knownbad.CoalesceCrashes(nil, crashStorm(300, "SIGNAL_CRASH", []string{"internal/foo/**"}, "sha256:deadbeef", now), now, crashTTL)
	if stats.Events != 300 || len(ledger) != 1 {
		t.Fatalf("coalescer produced %d rows from %d events, want 1 row", len(ledger), stats.Events)
	}

	got := FoldCrashSpam(ledger, nil, now)

	if len(got.Actions) != 1 {
		t.Fatalf("300 same-cause crashes planned %d tickets, want exactly 1: %+v", len(got.Actions), got.Actions)
	}
	act := got.Actions[0]
	if act.Action != CrashTicketOpen {
		t.Errorf("action = %q, want %q (no ticket covers this window yet)", act.Action, CrashTicketOpen)
	}
	if act.Occurrences != 300 || act.NewSince != 300 {
		t.Errorf("occurrences/new_since = %d/%d, want 300/300", act.Occurrences, act.NewSince)
	}
	if act.Number != 0 {
		t.Errorf("an open carries issue number %d, want 0 (nothing filed yet)", act.Number)
	}
	if act.Signature != ledger[0].Signature {
		t.Errorf("ticket signature %q does not name the coalesced row %q", act.Signature, ledger[0].Signature)
	}
	if act.Key != CrashTicketKey(ledger[0].Signature) || act.Key == "" {
		t.Errorf("dedup key = %q, want the signature-bound key", act.Key)
	}
	// The cost of the storm is one gh call, and the count rides the title.
	if got.Tickets != 1 || got.Opened != 1 || got.Refreshed != 0 || got.Suppressed != 0 {
		t.Errorf("readout = %+v, want exactly one opened ticket", got)
	}
	if want := "crash storm: SIGNAL_CRASH over internal/foo (300 crashes, 1 cause)"; act.Title != want {
		t.Errorf("title = %q, want %q", act.Title, want)
	}
	// The line the issue asks to read at a glance.
	if want := "300 crashes, 1 cause"; got.Line() != want {
		t.Errorf("Line() = %q, want %q", got.Line(), want)
	}
	if got.Amplification() != 300 {
		t.Errorf("amplification = %v, want 300 crashes carried per filing", got.Amplification())
	}
	if got.Schema != CrashSpamSchema {
		t.Errorf("schema = %q, want %q", got.Schema, CrashSpamSchema)
	}
}

// ACCEPTANCE (#3586): two DISTINCT signatures cost two tickets — the fold dedups a
// cause, it never merges causes. The loudest cause leads so the readout head is the
// one worth looking at.
func TestFoldCrashSpamKeepsDistinctCausesApart(t *testing.T) {
	const now int64 = 2_000_000
	events := append(
		crashStorm(4, "SIGNAL_CRASH", []string{"internal/foo"}, "sha256:aaa", now),
		crashStorm(9, "OOM", []string{"internal/bar"}, "sha256:bbb", now)...,
	)
	ledger, _ := knownbad.CoalesceCrashes(nil, events, now, crashTTL)

	got := FoldCrashSpam(ledger, nil, now)

	if len(got.Actions) != 2 || got.Causes != 2 || got.Tickets != 2 {
		t.Fatalf("two causes planned %+v, want two separate tickets", got)
	}
	if got.Actions[0].Key == got.Actions[1].Key {
		t.Fatalf("distinct causes collapsed onto one dedup key %q", got.Actions[0].Key)
	}
	// Loudest first: the 9-crash OOM outranks the 4-crash signal crash.
	if got.Actions[0].Occurrences != 9 || got.Actions[1].Occurrences != 4 {
		t.Errorf("counts = %d/%d, want the loudest cause (9) first", got.Actions[0].Occurrences, got.Actions[1].Occurrences)
	}
	if got.Crashes != 13 {
		t.Errorf("total crashes = %d, want 13 across both causes", got.Crashes)
	}
	if want := "13 crashes, 2 causes"; got.Line() != want {
		t.Errorf("Line() = %q, want %q", got.Line(), want)
	}
}

// ACCEPTANCE (#3586): re-emission WITHIN the window increments the count on the
// existing ticket and does not duplicate — and a pass where nothing moved files
// nothing at all, which is where the savings actually live.
func TestFoldCrashSpamRefreshesInWindowThenSuppresses(t *testing.T) {
	const now int64 = 3_000_000
	globs := []string{"internal/foo"}

	first, _ := knownbad.CoalesceCrashes(nil, crashStorm(40, "OOM", globs, "", now), now, crashTTL)
	ledger := append([]knownbad.Record{}, first...)

	opened := FoldCrashSpam(ledger, nil, now)
	if len(opened.Actions) != 1 || opened.Actions[0].Action != CrashTicketOpen {
		t.Fatalf("first pass = %+v, want one opened ticket", opened)
	}

	// The filer opens issue #77 stating 40, and records what it filed.
	filed := []FiledCrashTicket{{
		Signature:    opened.Actions[0].Signature,
		Number:       77,
		Occurrences:  opened.Actions[0].Occurrences,
		WindowAtUnix: opened.Actions[0].FirstSeenAtUnix,
	}}

	// A second storm 60s later, still inside the same 900s window.
	later := now + 60
	second, _ := knownbad.CoalesceCrashes(ledger, crashStorm(60, "OOM", globs, "", later), later, crashTTL)
	ledger = append(ledger, second...)

	refreshed := FoldCrashSpam(ledger, filed, later)
	if len(refreshed.Actions) != 1 {
		t.Fatalf("re-emission planned %d tickets, want 1 (an edit, not a second thread)", len(refreshed.Actions))
	}
	act := refreshed.Actions[0]
	if act.Action != CrashTicketRefresh {
		t.Errorf("action = %q, want %q", act.Action, CrashTicketRefresh)
	}
	if act.Number != 77 {
		t.Errorf("refresh targets issue %d, want the existing #77", act.Number)
	}
	if act.Occurrences != 100 || act.NewSince != 60 {
		t.Errorf("occurrences/new_since = %d/%d, want 100/60 (40 already filed)", act.Occurrences, act.NewSince)
	}
	if act.Key != filed[0].Signature && act.Signature != filed[0].Signature {
		t.Errorf("refresh changed the cause: %q -> %q", filed[0].Signature, act.Signature)
	}
	if refreshed.Opened != 0 || refreshed.Refreshed != 1 || refreshed.Tickets != 1 {
		t.Errorf("readout = %+v, want a single refresh and no new thread", refreshed)
	}

	// The filer writes 100 onto #77; the very next pass over the same ledger must cost
	// NOTHING — a known storm that has not moved is pure spam to re-file.
	filed[0].Occurrences = 100
	quiet := FoldCrashSpam(ledger, filed, later)
	if len(quiet.Actions) != 1 || quiet.Actions[0].Action != CrashTicketSuppress {
		t.Fatalf("steady-state pass = %+v, want the signature suppressed", quiet)
	}
	if quiet.Tickets != 0 || quiet.Suppressed != 1 {
		t.Errorf("readout = %+v, want zero filings on an unchanged storm", quiet)
	}
	if quiet.Actions[0].NewSince != 0 {
		t.Errorf("new_since = %d on a suppress, want 0", quiet.Actions[0].NewSince)
	}
	// The cause is still legible even when nothing is filed.
	if want := "100 crashes, 1 cause"; quiet.Line() != want {
		t.Errorf("Line() = %q, want %q", quiet.Line(), want)
	}
}

// Once the window LAPSES, the same cause opens a NEW ticket rather than editing a
// thread that covered a window which has since closed — "one ticket per signature per
// WINDOW". The stale ticket number is deliberately not carried over.
func TestFoldCrashSpamOpensNewTicketAfterWindowLapse(t *testing.T) {
	const now int64 = 4_000_000
	globs := []string{"internal/foo"}

	first, _ := knownbad.CoalesceCrashes(nil, crashStorm(10, "OOM", globs, "", now), now, crashTTL)
	filed := []FiledCrashTicket{{
		Signature:    first[0].Signature,
		Number:       88,
		Occurrences:  10,
		WindowAtUnix: first[0].DiscoveredAtUnix,
	}}

	// Well past the TTL: the coalescer opens a fresh window with a restarted count.
	after := now + crashTTL + 1
	reopened, _ := knownbad.CoalesceCrashes(first, crashStorm(7, "OOM", globs, "", after), after, crashTTL)
	ledger := append(append([]knownbad.Record{}, first...), reopened...)

	got := FoldCrashSpam(ledger, filed, after)
	if len(got.Actions) != 1 {
		t.Fatalf("post-lapse pass planned %d tickets, want 1", len(got.Actions))
	}
	act := got.Actions[0]
	if act.Action != CrashTicketOpen || got.Opened != 1 {
		t.Errorf("action = %q, want a NEW ticket for the new window", act.Action)
	}
	if act.Number != 0 {
		t.Errorf("a new window carried the lapsed issue %d, want 0", act.Number)
	}
	if act.Occurrences != 7 || act.NewSince != 7 {
		t.Errorf("occurrences/new_since = %d/%d, want the restarted 7/7", act.Occurrences, act.NewSince)
	}
	if act.Signature != first[0].Signature {
		t.Errorf("a new window must keep the same signature (same cause)")
	}
}

// A signature that is resolved, revoked, or expired stops costing filings the moment
// it stops being live, and a hand-recorded known-bad (no coalesced count) is not
// auto-ticketed here at all — this fold owns the CRASH surface, not the whole ledger.
func TestFoldCrashSpamIgnoresNonLiveAndNonCrashRows(t *testing.T) {
	const now int64 = 5_000_000
	globs := []string{"internal/foo"}

	crash, _ := knownbad.CoalesceCrashes(nil, crashStorm(12, "OOM", globs, "", now), now, crashTTL)
	handRecorded := knownbad.NewRecord("build", []string{"internal/bar"}, "flaky build", "operator", "", now, crashTTL)
	expired, _ := knownbad.CoalesceCrashes(nil, crashStorm(5, "OOM", []string{"internal/baz"}, "", now-crashTTL-10), now-crashTTL-10, crashTTL)

	ledger := append(append(append([]knownbad.Record{}, crash...), handRecorded), expired...)

	got := FoldCrashSpam(ledger, nil, now)
	if len(got.Actions) != 1 || got.Causes != 1 {
		t.Fatalf("fold planned %+v, want only the one live crash cause", got.Actions)
	}
	if got.Actions[0].Signature != crash[0].Signature {
		t.Errorf("ticketed %q, want the live crash signature %q", got.Actions[0].Signature, crash[0].Signature)
	}

	// A resolved cause drops out entirely: a fixed storm files nothing.
	resolvedLedger := append(append([]knownbad.Record{}, crash...), crash[0].WithResolve("fixer", now+5, "tests"))
	quiet := FoldCrashSpam(resolvedLedger, nil, now+5)
	if len(quiet.Actions) != 0 || quiet.Tickets != 0 {
		t.Errorf("a resolved cause planned %+v, want no filings", quiet)
	}
	if want := "0 crashes, 0 causes"; quiet.Line() != want {
		t.Errorf("Line() = %q, want %q", quiet.Line(), want)
	}
	if quiet.Amplification() != 0 {
		t.Errorf("amplification with no filings = %v, want 0", quiet.Amplification())
	}
}

// The dedup key is signature-bound and stable: the same cause always yields the same
// key (so the marker dedups across passes and processes), distinct causes never
// collide, and an unusable signature yields no key at all rather than an undedupable
// issue — which the fold counts instead of filing.
func TestCrashTicketKeyIsStableAndSignatureBound(t *testing.T) {
	a := knownbad.Signature("OOM", []string{"internal/foo"}, "")
	b := knownbad.Signature("OOM", []string{"internal/bar"}, "")

	if CrashTicketKey(a) != CrashTicketKey(a) || CrashTicketKey(a) == "" {
		t.Fatalf("key is not stable for one signature: %q", CrashTicketKey(a))
	}
	if CrashTicketKey(a) == CrashTicketKey(b) {
		t.Errorf("distinct signatures share key %q", CrashTicketKey(a))
	}
	if got := CrashTicketKey("   "); got != "" {
		t.Errorf("blank signature yielded key %q, want \"\"", got)
	}

	// A live row whose signature cannot be keyed is dropped and COUNTED, never filed.
	unkeyable := knownbad.NewRecord("OOM", []string{"internal/foo"}, "", "", "", 100, crashTTL).WithOccurrences(9, 100)
	unkeyable.Signature = "sha256:"
	got := FoldCrashSpam([]knownbad.Record{unkeyable}, nil, 100)
	if got.DroppedUnkeyed != 1 || len(got.Actions) != 0 {
		t.Errorf("readout = %+v, want the unkeyable row dropped and counted", got)
	}
}
