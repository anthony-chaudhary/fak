package steerpr

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// `fak steer ack` (#5028): the operator's half of the RESIDUAL band. An ack
// records that a HUMAN reviewed a residual unit. Three properties are pinned
// here, matching the ticket's done condition:
//
//  1. The ack binds to the unit's exact member SHA SET at ack time, not to
//     the unit's name — a commit that later joins (or leaves) the unit
//     invalidates the ack, because the human never looked at the new set.
//  2. The ack NEVER moves the machine band. "RESIDUAL (acked by X)" is the
//     most an ack can ever render; CLEARED stays reachable only through the
//     witness bit (the #5036 fence in antigaming_test.go and
//     overlay_ack_nonforge_test.go).
//  3. The ledger is append-only and attributable: every row carries who and
//     when, and a new row never rewrites an old one.

// ackFoldUnit folds one single-leaf unit from the given member SHAs, all
// graded with the same verdict.
func ackFoldUnit(t *testing.T, verdict Verdict, shas ...string) Unit {
	t.Helper()
	commits := make([]Commit, 0, len(shas))
	for _, sha := range shas {
		commits = append(commits, Commit{
			SHA:     sha,
			Subject: "feat(gateway): a claim the diff did not prove (fak gateway)",
			Leaf:    "gateway",
			Type:    "feat",
			Verdict: verdict,
		})
	}
	units, unstamped := FoldUnits(commits)
	if len(units) != 1 || len(unstamped) != 0 {
		t.Fatalf("FoldUnits() = %d units (%d unstamped), want exactly 1 unit", len(units), len(unstamped))
	}
	return units[0]
}

func ackLedger(t *testing.T) string {
	t.Helper()
	return AckLedgerPath(t.TempDir())
}

// An ack is a record of a specific person having reviewed a specific set of
// commits; a row missing either leg is refused, not defaulted.
func TestAckRequiresAttributionAndAMemberSet(t *testing.T) {
	now := time.Now()
	if _, err := NewAck("gateway", "", []string{"aaa"}, "", now); err == nil {
		t.Error("NewAck with no `by` should refuse: an unattributable ack records nobody having looked")
	}
	if _, err := NewAck("gateway", "op", nil, "", now); err == nil {
		t.Error("NewAck with no SHAs should refuse: an ack binds to what was reviewed, and an empty set reviewed nothing")
	}
	if _, err := NewAck("", "op", []string{"aaa"}, "", now); err == nil {
		t.Error("NewAck with no leaf should refuse: an ack must name the unit it lands on")
	}
	if err := AppendAck(ackLedger(t), Ack{Schema: AckSchema, Leaf: "gateway"}); err == nil {
		t.Error("AppendAck should refuse an incomplete row: the ledger stays attributable")
	}
}

// The ledger is append-only: a second append leaves the first row
// byte-identical, and rows read back in order with who/when/what intact.
func TestAckLedgerIsAppendOnlyAndAttributable(t *testing.T) {
	path := ackLedger(t)
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	first, err := NewAck("gateway", "op-one", []string{"bbb", "aaa", "aaa", " bbb "}, "looked fine", at)
	if err != nil {
		t.Fatalf("NewAck: %v", err)
	}
	if err := AppendAck(path, first); err != nil {
		t.Fatalf("AppendAck: %v", err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	second, err := NewAck("kvcache", "op-two", []string{"ccc"}, "", at.Add(time.Hour))
	if err != nil {
		t.Fatalf("NewAck: %v", err)
	}
	if err := AppendAck(path, second); err != nil {
		t.Fatalf("AppendAck: %v", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	if !strings.HasPrefix(string(after), string(before)) {
		t.Errorf("append rewrote an existing row — the ledger must be append-only:\nbefore=%q\nafter=%q", before, after)
	}

	rows := LoadAcks(path)
	if len(rows) != 2 {
		t.Fatalf("LoadAcks() = %d rows, want 2", len(rows))
	}
	got := rows[0]
	if got.Schema != AckSchema || got.Leaf != "gateway" || got.By != "op-one" || got.Note != "looked fine" {
		t.Errorf("row 0 = %#v, want schema/leaf/by/note preserved", got)
	}
	if got.At == "" {
		t.Errorf("row 0 has no timestamp — an ack must say when the human looked")
	}
	if want := []string{"aaa", "bbb"}; len(got.SHAs) != 2 || got.SHAs[0] != want[0] || got.SHAs[1] != want[1] {
		t.Errorf("row 0 SHAs = %v, want deduped+sorted %v", got.SHAs, want)
	}
	if rows[1].By != "op-two" {
		t.Errorf("row 1 by = %q, want op-two", rows[1].By)
	}
}

// The ack binds to the SHA set, not the unit name: exact-set match is order-
// and duplicate-insensitive, and ANY drift — added, removed, or substituted
// member — uncovers it.
func TestAckBindsToSHASetNotUnitName(t *testing.T) {
	a, err := NewAck("gateway", "op", []string{"aaa", "bbb"}, "", time.Now())
	if err != nil {
		t.Fatalf("NewAck: %v", err)
	}
	if !a.Covers([]string{"bbb", "aaa"}) {
		t.Error("Covers should be order-insensitive over the same set")
	}
	if !a.Covers([]string{"aaa", "bbb", "bbb"}) {
		t.Error("Covers should be duplicate-insensitive over the same set")
	}
	if a.Covers([]string{"aaa", "bbb", "ccc"}) {
		t.Error("a NEW member must invalidate the ack: the human never reviewed ccc")
	}
	if a.Covers([]string{"aaa"}) {
		t.Error("a REMOVED member must invalidate the ack: it was a review of a different set")
	}
	if a.Covers([]string{"aaa", "ddd"}) {
		t.Error("a SUBSTITUTED member must invalidate the ack")
	}
	if a.Covers(nil) {
		t.Error("an empty set is covered by nothing")
	}
	if _, ok := AckFor([]Ack{a}, "other-leaf", []string{"aaa", "bbb"}); ok {
		t.Error("AckFor must not match an ack recorded against a different unit")
	}
}

// Done condition (b): a new member commit invalidates a prior ack and the unit
// reads RESIDUAL/unacked again.
func TestNewMemberInvalidatesAckAndUnitReadsUnacked(t *testing.T) {
	path := ackLedger(t)
	reviewed := ackFoldUnit(t, VerdictUnwitnessed, "aaa", "bbb")
	row, err := NewAck(reviewed.Leaf, "op-jane", UnitSHAs(reviewed), "", time.Now())
	if err != nil {
		t.Fatalf("NewAck: %v", err)
	}
	if err := AppendAck(path, row); err != nil {
		t.Fatalf("AppendAck: %v", err)
	}
	acks := LoadAcks(path)

	if a, ok := AckFor(acks, reviewed.Leaf, UnitSHAs(reviewed)); !ok || a.By != "op-jane" {
		t.Fatalf("AckFor(reviewed set) = %#v,%v — the ack should cover the exact set that was reviewed", a, ok)
	}
	if got := BandLabel(reviewed.Band, row, true); got != "RESIDUAL (acked by op-jane)" {
		t.Errorf("BandLabel(acked) = %q, want %q", got, "RESIDUAL (acked by op-jane)")
	}

	// A commit joins the unit: the set changed, so the ack no longer covers.
	grown := ackFoldUnit(t, VerdictUnwitnessed, "aaa", "bbb", "ccc")
	if _, ok := AckFor(acks, grown.Leaf, UnitSHAs(grown)); ok {
		t.Error("a new member joined the unit; the prior ack must be invalidated, not silently bless code the human never saw")
	}
	if got := BandLabel(grown.Band, Ack{}, false); got != string(BandResidual) {
		t.Errorf("BandLabel(unacked) = %q, want bare %q — the unit reads RESIDUAL/unacked again", got, BandResidual)
	}
}

// Done condition (a) + the #5036 fence: acking changes NOTHING about the
// machine band or the posted residual count, and the render is
// "RESIDUAL (acked by X)" — never CLEARED.
func TestAckDoesNotChangeMachineBand(t *testing.T) {
	path := ackLedger(t)
	unit := ackFoldUnit(t, VerdictUnwitnessed, "aaa", "bbb")
	if unit.Band != BandResidual {
		t.Fatalf("unit band = %q, want %q before the ack", unit.Band, BandResidual)
	}

	row, err := NewAck(unit.Leaf, "op-jane", UnitSHAs(unit), "seems fine", time.Now())
	if err != nil {
		t.Fatalf("NewAck: %v", err)
	}
	if err := AppendAck(path, row); err != nil {
		t.Fatalf("AppendAck: %v", err)
	}

	// Re-fold the same commits the way any honest re-tick does: the ack is not
	// an input to the fold — structurally, there is nowhere to pass it — so the
	// band and the residual count cannot have moved.
	refolded := ackFoldUnit(t, VerdictUnwitnessed, "aaa", "bbb")
	if refolded.Band != BandResidual {
		t.Errorf("band = %q after acking, want %q: an ack must never move the machine band", refolded.Band, BandResidual)
	}
	if got := Residual([]Unit{refolded}); got != 1 {
		t.Errorf("Residual() = %d after acking, want 1: an acked residual still owes attention", got)
	}

	a, ok := AckFor(LoadAcks(path), refolded.Leaf, UnitSHAs(refolded))
	if !ok {
		t.Fatal("the ack should cover the unchanged member set")
	}
	label := BandLabel(refolded.Band, a, ok)
	if !strings.HasPrefix(label, string(BandResidual)) {
		t.Errorf("BandLabel = %q, want it to LEAD with the honest band %q", label, BandResidual)
	}
	if !strings.Contains(label, "acked by op-jane") {
		t.Errorf("BandLabel = %q, want the acked state rendered beside the band", label)
	}
	if strings.Contains(label, string(BandCleared)) {
		t.Errorf("BandLabel = %q renders CLEARED for an acked residual — that is an ack laundered into a witness", label)
	}
}

// The ledger is shared and append-only, so the LATEST covering row wins, and a
// torn or foreign line is skipped rather than poisoning its neighbours.
func TestAckLatestRowWinsAndLoadIsBestEffort(t *testing.T) {
	path := ackLedger(t)
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	for i, by := range []string{"op-one", "op-two"} {
		row, err := NewAck("gateway", by, []string{"aaa", "bbb"}, "", at.Add(time.Duration(i)*time.Hour))
		if err != nil {
			t.Fatalf("NewAck: %v", err)
		}
		if err := AppendAck(path, row); err != nil {
			t.Fatalf("AppendAck: %v", err)
		}
	}
	// A torn line in the middle of the ledger must not eat the rows around it.
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("open ledger: %v", err)
	}
	if _, err := f.WriteString("{torn row\n"); err != nil {
		t.Fatalf("write torn row: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close ledger: %v", err)
	}

	acks := LoadAcks(path)
	if len(acks) != 2 {
		t.Fatalf("LoadAcks() = %d rows, want 2 (torn line skipped, neighbours kept)", len(acks))
	}
	if a, ok := AckFor(acks, "gateway", []string{"bbb", "aaa"}); !ok || a.By != "op-two" {
		t.Errorf("AckFor = %#v,%v, want the latest covering row (op-two)", a, ok)
	}

	if got := LoadAcks(filepath.Join(t.TempDir(), "no-such-ledger.jsonl")); got != nil {
		t.Errorf("LoadAcks(missing) = %#v, want nil: a missing ledger is an empty ledger", got)
	}
}
