package modelroute

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
	"time"
)

// Witnesses for the durable evidence journal (epic #5416, tracks D and F).
//
// Every rule here is about ONE failure: a grade that is stronger than the turns behind it.
// A journal is the easiest place in the system to manufacture capability — replay a file,
// forget to stamp a row, let one model's numbers absorb another's provenance — so the fold
// is written to lose evidence rather than to invent it, and these tests are the record of
// which losses are deliberate.

func at(day int) time.Time {
	return time.Date(2026, 7, day, 12, 0, 0, 0, time.UTC)
}

func TestAJournalRoundTripsThroughItsOwnWriter(t *testing.T) {
	want := []TurnOutcome{
		{ID: "t1", Model: "tiny", Class: ClassRoutine, Zone: ZoneDevice, Success: true, Verify: VerifyWitness, At: at(20)},
		{ID: "t2", Model: "corp-mid", Class: ClassNormalImpl, Zone: ZoneFleet, Success: false, Verify: VerifyJudge, At: at(21)},
	}
	var buf bytes.Buffer
	for _, o := range want {
		if err := AppendTurnOutcome(&buf, o); err != nil {
			t.Fatal(err)
		}
	}
	if n := strings.Count(buf.String(), "\n"); n != 2 {
		t.Fatalf("wrote %d lines for 2 outcomes — a journal must stay one record per line", n)
	}
	got, stats, err := ReadTurnOutcomes(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("round trip lost a field:\n got %+v\nwant %+v", got, want)
	}
	if stats.Lines != 2 || stats.Malformed != 0 {
		t.Errorf("stats = %+v, want 2 clean lines", stats)
	}
}

func TestATornLineDoesNotDiscardTheGoodCorpus(t *testing.T) {
	// The realistic crash: a fleet appending to this file dies mid-write. Refusing the
	// whole journal would throw away thousands of verified turns over one partial row,
	// and "I refuse to read" is not a safety property here — reading SILENTLY would be.
	body := `{"id":"a","model":"tiny","class":"routine","success":true,"verify":"witness"}

{"id":"b","model":"tiny","class":"routine","success":true,"verify":"witn`
	got, stats, err := ReadTurnOutcomes(strings.NewReader(body))
	if err != nil {
		t.Fatalf("a torn line was reported as a reader failure: %v", err)
	}
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("good records were lost: %+v", got)
	}
	if stats.Lines != 2 || stats.Malformed != 1 {
		t.Errorf("stats = %+v, want 2 lines / 1 malformed (the blank line is not a record)", stats)
	}
}

func TestAnAbsurdlyLongLineIsMalformedNotAnUnboundedBuffer(t *testing.T) {
	body := `{"id":"a","model":"tiny","class":"routine","success":true,"verify":"witness"}` + "\n" +
		`{"id":"b","model":"` + strings.Repeat("x", maxOutcomeLine+16) + `"}` + "\n"
	got, stats, err := ReadTurnOutcomes(strings.NewReader(body))
	if err != nil {
		t.Fatalf("an over-long line was reported as a reader failure: %v", err)
	}
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("the records before the bad line were lost: %+v", got)
	}
	if stats.Malformed != 1 {
		t.Errorf("stats = %+v, want the over-long line counted as malformed", stats)
	}
}

func TestARepeatedOutcomeIsCountedOnce(t *testing.T) {
	// Replaying a journal, or a producer that ships a retried turn twice, is the cheapest
	// possible way to manufacture a grade: the same 20 successes appended 3 times reads as
	// 60. The id is what makes that refusable.
	outcomes := []TurnOutcome{
		{ID: "t1", Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
		{ID: "t1", Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
		{ID: "t2", Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
	}
	ev, stats := FoldTurnOutcomes(outcomes, FoldOptions{})
	if len(ev["tiny"]) != 1 || ev["tiny"][0].Attempts != 2 {
		t.Fatalf("evidence = %+v, want 2 attempts from 3 rows", ev["tiny"])
	}
	if stats.Counted != 2 || stats.Duplicates != 1 {
		t.Errorf("stats = %+v, want 2 counted / 1 duplicate", stats)
	}
}

func TestOutcomesWithNoIDAreKeptButReportedAsInflatable(t *testing.T) {
	// A missing id is not proof of a duplicate, so the rows are kept. But a corpus that
	// cannot be deduplicated is one a broken producer can inflate at will, and the number
	// that says so has to reach the operator.
	outcomes := []TurnOutcome{
		{Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
		{Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
	}
	ev, stats := FoldTurnOutcomes(outcomes, FoldOptions{})
	if ev["tiny"][0].Attempts != 2 {
		t.Fatalf("un-id'd rows were dropped as duplicates of each other: %+v", ev["tiny"])
	}
	if stats.Undeduplicable != 2 || stats.Duplicates != 0 {
		t.Errorf("stats = %+v, want 2 undeduplicable / 0 duplicates", stats)
	}
}

func TestUndatedEvidenceCannotSatisfyAFreshnessWindow(t *testing.T) {
	// Capability is a property of a model AS DEPLOYED — a requantised checkpoint is a
	// different model wearing the same id — so "the last 30 days" is a real question. A
	// row with no timestamp cannot be shown to be inside the window, and admitting it
	// anyway would let stale evidence keep a grade alive forever.
	outcomes := []TurnOutcome{
		{ID: "fresh", Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness, At: at(25)},
		{ID: "stale", Model: "tiny", Class: ClassRoutine, Success: false, Verify: VerifyWitness, At: at(1)},
		{ID: "undated", Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
	}
	ev, stats := FoldTurnOutcomes(outcomes, FoldOptions{Since: at(20)})
	// The success flags differ so this pins WHICH row survived, not just how many: a
	// window with its comparison inverted keeps exactly one row too, and it is the wrong one.
	if len(ev["tiny"]) != 1 || ev["tiny"][0].Attempts != 1 || ev["tiny"][0].Successes != 1 {
		t.Fatalf("evidence = %+v, want only the in-window (successful) row", ev["tiny"])
	}
	if stats.OutsideWindow != 1 || stats.Undated != 1 {
		t.Errorf("stats = %+v, want the two exclusions counted APART — one needs a wider "+
			"window, the other needs a producer that stamps its rows", stats)
	}
	// With no window asked for, a missing date costs nothing.
	if _, s := FoldTurnOutcomes(outcomes, FoldOptions{}); s.Counted != 3 || s.Undated != 0 {
		t.Errorf("stats = %+v, want all 3 counted when no window was requested", s)
	}
}

func TestProvenanceIsNotMergedByTheFold(t *testing.T) {
	// 100 self-reported turns and 20 witnessed ones about the same model and class. If the
	// fold merged them into one row it would have to pick a provenance, and any pick is a
	// lie: the honest shape is two rows, and GradeCapability's own merge — which keeps the
	// WEAKEST provenance of what it merged — decides what that buys.
	var outcomes []TurnOutcome
	for i := 0; i < 100; i++ {
		outcomes = append(outcomes, TurnOutcome{ID: "self" + string(rune('a'+i%26)) + string(rune('a'+i/26)),
			Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyNone})
	}
	for i := 0; i < 20; i++ {
		outcomes = append(outcomes, TurnOutcome{ID: "wit" + string(rune('a'+i)),
			Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness})
	}
	ev, _ := FoldTurnOutcomes(outcomes, FoldOptions{})
	if len(ev["tiny"]) != 2 {
		t.Fatalf("evidence = %+v, want one row per provenance", ev["tiny"])
	}
	// And the grade that follows is the witnessed 20, not the 120.
	g := GradeCapability("tiny", ev["tiny"], DefaultGradeFloor())
	if !g.Measured || g.Attempts != 20 || g.Verify != VerifyWitness {
		t.Errorf("the self-reported turns diluted their way into the grade: %+v", g)
	}
	if g.Dropped != 100 {
		t.Errorf("dropped = %d, want the 100 self-reported attempts reported as refused", g.Dropped)
	}
}

func TestAnUnattributedOutcomeIsDroppedNotFiledUnderTheEmptyModel(t *testing.T) {
	ev, stats := FoldTurnOutcomes([]TurnOutcome{
		{ID: "a", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
		{ID: "b", Model: "tiny", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
	}, FoldOptions{})
	if _, ok := ev[""]; ok {
		t.Errorf("an outcome with no model created an empty-named model: %+v", ev)
	}
	if stats.Unattributed != 1 || stats.Counted != 1 {
		t.Errorf("stats = %+v, want 1 unattributed / 1 counted", stats)
	}
}

func TestTheFoldIsDeterministicAndFailuresAreCountedAsAttempts(t *testing.T) {
	outcomes := []TurnOutcome{
		{ID: "1", Model: "m", Class: ClassRoutine, Success: true, Verify: VerifyWitness},
		{ID: "2", Model: "m", Class: ClassNormalImpl, Success: false, Verify: VerifyJudge},
		{ID: "3", Model: "m", Class: ClassNormalImpl, Success: true, Verify: VerifyWitness},
		{ID: "4", Model: "m", Class: ClassRoutine, Success: false, Verify: VerifyWitness},
	}
	first, _ := FoldTurnOutcomes(outcomes, FoldOptions{})
	shuffled := []TurnOutcome{outcomes[2], outcomes[0], outcomes[3], outcomes[1]}
	second, _ := FoldTurnOutcomes(shuffled, FoldOptions{})
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("the fold is order-dependent:\n %+v\n %+v", first, second)
	}
	rows := first["m"]
	if len(rows) != 3 {
		t.Fatalf("rows = %+v, want one per (class, provenance)", rows)
	}
	if rows[0].Class != ClassNormalImpl || rows[len(rows)-1].Class != ClassRoutine {
		t.Errorf("rows are not sorted by class then provenance: %+v", rows)
	}
	// A failed turn is an ATTEMPT. Dropping it would grade every model at 100%.
	total := int64(0)
	for _, r := range rows {
		total += r.Attempts
	}
	if total != 4 {
		t.Errorf("attempts = %d, want 4 — a failure is evidence too", total)
	}
	for _, r := range rows {
		if r.Successes > r.Attempts {
			t.Errorf("row claims more successes than attempts: %+v", r)
		}
	}
}

func TestAJournalGradesAndPlacesEndToEnd(t *testing.T) {
	// The whole track in one test: append real turns, read them back, fold, grade, place.
	// Nobody asserts a capability anywhere in this path.
	var buf bytes.Buffer
	for i := 0; i < 30; i++ {
		o := TurnOutcome{ID: "d" + string(rune('a'+i)), Model: "tiny", Class: ClassRoutine,
			Zone: ZoneDevice, Success: i%10 != 0, Verify: VerifyWitness, At: at(20)}
		if err := AppendTurnOutcome(&buf, o); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < 30; i++ {
		o := TurnOutcome{ID: "f" + string(rune('a'+i)), Model: "corp-mid", Class: ClassNormalImpl,
			Zone: ZoneFleet, Success: i%10 != 0, Verify: VerifyJudge, At: at(21)}
		if err := AppendTurnOutcome(&buf, o); err != nil {
			t.Fatal(err)
		}
	}
	outcomes, stats, err := ReadTurnOutcomes(&buf)
	if err != nil || stats.Malformed != 0 || len(outcomes) != 60 {
		t.Fatalf("read: %v stats=%+v n=%d", err, stats, len(outcomes))
	}
	ev, fold := FoldTurnOutcomes(outcomes, FoldOptions{Since: at(15)})
	if fold.Counted != 60 || fold.Undated != 0 {
		t.Fatalf("fold = %+v, want every stamped row inside the window", fold)
	}
	grades := GradeCandidates([]string{"frontier", "corp-mid", "tiny"}, ev, DefaultGradeFloor())
	var candidates []Candidate
	for _, g := range grades {
		candidates = append(candidates, g.Candidate())
	}
	r := threeZoneRoster()
	routine, err := r.Place(ClassRoutine, candidates)
	if err != nil {
		t.Fatal(err)
	}
	if routine.Zone != ZoneDevice || !routine.Measured {
		t.Errorf("routine work did not reach the device rung from journal evidence alone: %+v", routine)
	}
	impl, err := r.Place(ClassNormalImpl, candidates)
	if err != nil {
		t.Fatal(err)
	}
	if impl.Zone != ZoneFleet || impl.Model != "corp-mid" {
		t.Errorf("ordinary implementation did not reach the fleet rung: zone=%s model=%s", impl.Zone, impl.Model)
	}
	// A 90% record is above the 80% floor and below a stricter one: the same journal,
	// read by a fleet that demands more, grades nobody and everything stays on the vendor.
	strict := DefaultGradeFloor()
	strict.MinSuccessRate = 0.95
	var strictCandidates []Candidate
	for _, g := range GradeCandidates([]string{"frontier", "corp-mid", "tiny"}, ev, strict) {
		strictCandidates = append(strictCandidates, g.Candidate())
	}
	if p, err := r.Place(ClassRoutine, strictCandidates); err != nil || p.Zone != ZoneVendor {
		t.Errorf("a stricter floor did not pull the work back to the vendor: %+v (%v)", p, err)
	}
}
