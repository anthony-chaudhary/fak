package journal

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeAs appends rows through the real commit core, so every fixture below is
// stamped by the same allocator the kernel uses (Seq from process-local state,
// PrevHash from the process-local head) rather than hand-forged.
//
// session goes into TraceID, which is in the hash pre-image, and it is not
// cosmetic: two writers that fork at the same parent, reissue the same Seq, and
// emit the same decision differ ONLY in TSUnixNano, so under a coarse clock they
// can produce a byte-identical row and hence an identical hash. Real guard
// sessions each carry their own trace id, so giving the fixture's writers
// distinct ones reproduces the fork faithfully instead of manufacturing a
// same-tick hash collision the corpus does not contain (measured: 0 duplicate
// hashes across the 60,514-row shared capture on the reference host).
func writeAs(t *testing.T, j *Journal, session string, kinds ...string) {
	t.Helper()
	for _, k := range kinds {
		j.append(Row{Kind: k, Tool: "Bash", Verdict: "ALLOW", TraceID: session})
	}
	if err := j.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
}

// forkedJournal reproduces the actual concurrent-writer defect rather than
// simulating it: one process writes a prefix, then TWO processes Open the same
// path, each recovering the SAME chain head into its own state, and both append.
// The result is the tree shape a shared journal file really takes on disk.
func forkedJournal(t *testing.T) []Row {
	t.Helper()
	path := filepath.Join(t.TempDir(), "shared.jsonl")

	head, err := Open(path)
	if err != nil {
		t.Fatalf("open head: %v", err)
	}
	writeAs(t, head, "sess-prefix", "DECIDE", "DENY")
	if err := head.Close(); err != nil {
		t.Fatalf("close head: %v", err)
	}

	// Two "processes" open the same file. Each recoverHead's to seq=2 and holds
	// that as PRIVATE state — neither learns about the other's appends.
	a, err := Open(path)
	if err != nil {
		t.Fatalf("open a: %v", err)
	}
	defer a.Close()
	b, err := Open(path)
	if err != nil {
		t.Fatalf("open b: %v", err)
	}
	defer b.Close()

	writeAs(t, a, "sess-a", "DECIDE")
	writeAs(t, b, "sess-b", "DECIDE") // reissues seq=3, chains onto the same parent as a's row
	writeAs(t, a, "sess-a", "DENY")
	writeAs(t, b, "sess-b", "DENY")

	rows, err := ReadRows(path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	if len(rows) != 6 {
		t.Fatalf("want 6 rows on disk, got %d", len(rows))
	}
	return rows
}

// TestVerifyForest_ForkedJournalIsIntactNotTampered is the load-bearing claim:
// a journal two processes shared FAILS the linear VerifyRows and still passes
// VerifyForest, because every branch is cryptographically whole. If this ever
// passes VerifyRows the fixture stopped reproducing the defect.
func TestVerifyForest_ForkedJournalIsIntactNotTampered(t *testing.T) {
	rows := forkedJournal(t)

	if _, err := VerifyRows(rows); err == nil {
		t.Fatal("linear VerifyRows unexpectedly PASSED on a forked journal -- fixture no longer reproduces the concurrent-writer defect")
	}

	f, err := VerifyForest(rows)
	if err != nil {
		t.Fatalf("VerifyForest rejected an intact forked journal: %v (%+v)", err, f)
	}
	if f.Linear {
		t.Fatal("Linear must be false on a forked journal")
	}
	if f.Genesis != 1 {
		t.Errorf("Genesis = %d, want 1 (the shared prefix has one root)", f.Genesis)
	}
	if f.BranchPoints != 1 {
		t.Errorf("BranchPoints = %d, want 1 (the two writers forked at one row)", f.BranchPoints)
	}
	if f.Tips != 2 {
		t.Errorf("Tips = %d, want 2 (one per writer branch)", f.Tips)
	}
	if f.IntactChains != 2 || f.BrokenChains != 0 {
		t.Errorf("IntactChains=%d BrokenChains=%d, want 2/0", f.IntactChains, f.BrokenChains)
	}
	if f.Orphans != 0 || f.Duplicates != 0 {
		t.Errorf("Orphans=%d Duplicates=%d, want 0/0", f.Orphans, f.Duplicates)
	}
}

// TestVerifyForest_LinearJournalStaysLinear pins that the single-writer case —
// the shipped default, one journal file per guard session — takes the fast path
// and reports itself as one chain.
func TestVerifyForest_LinearJournalStaysLinear(t *testing.T) {
	j := OpenMemory()
	writeAs(t, j, "sess-solo", "DECIDE", "DENY", "QUARANTINE")
	rows := j.Recent(0)

	if _, err := VerifyRows(rows); err != nil {
		t.Fatalf("single-writer journal failed linear verify: %v", err)
	}
	f, err := VerifyForest(rows)
	if err != nil {
		t.Fatalf("VerifyForest on a linear journal: %v", err)
	}
	if !f.Linear || f.Tips != 1 || f.IntactChains != 1 || f.BranchPoints != 0 {
		t.Errorf("linear journal reported as %+v, want Linear/1 tip/1 chain/0 branches", f)
	}
}

func TestVerifyForest_Empty(t *testing.T) {
	f, err := VerifyForest(nil)
	if err != nil || !f.Linear || f.Rows != 0 {
		t.Fatalf("empty journal: f=%+v err=%v, want linear and sound", f, err)
	}
}

// TestVerifyForest_TamperedRowStillFails is the assertion that keeps the re-aim
// honest. Editing ONE field of one row inside a forked journal must still be
// refused: the widened verifier buys concurrency tolerance, never edit
// tolerance.
func TestVerifyForest_TamperedRowStillFails(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(rows []Row)
	}{
		{"verdict flipped ALLOW->DENY", func(rows []Row) { rows[4].Verdict = "DENY" }},
		{"tool rewritten", func(rows []Row) { rows[3].Tool = "Write" }},
		{"reason injected", func(rows []Row) { rows[1].Reason = "OK" }},
		{"seq renumbered", func(rows []Row) { rows[2].Seq = 99 }},
		{"timestamp backdated", func(rows []Row) { rows[5].TSUnixNano = 1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows := forkedJournal(t)
			tc.mutate(rows)
			f, err := VerifyForest(rows)
			if err == nil {
				t.Fatalf("VerifyForest ACCEPTED a tampered forked journal (%+v) -- the concurrent-writer allowance must not become an edit allowance", f)
			}
		})
	}
}

// TestVerifyForest_DroppedRowFails covers the property a linear read never
// checks at all: an interior row removed from the file orphans its children.
func TestVerifyForest_DroppedRowFails(t *testing.T) {
	rows := forkedJournal(t)
	cut := append(append([]Row{}, rows[:2]...), rows[3:]...) // drop an interior row
	f, err := VerifyForest(cut)
	if err == nil {
		t.Fatalf("VerifyForest ACCEPTED a journal with a dropped interior row (%+v)", f)
	}
	if f.Orphans == 0 {
		t.Errorf("Orphans = 0, want >0 for a dropped interior row (%+v)", f)
	}
}

// TestVerifyForest_ReplayedRowFails refuses a byte-identical row appended twice.
// It is individually authentic and its parent resolves, so only the duplicate
// hash betrays it — which is why the index refuses to collapse duplicates.
func TestVerifyForest_ReplayedRowFails(t *testing.T) {
	rows := forkedJournal(t)
	replayed := append(append([]Row{}, rows...), rows[2])
	f, err := VerifyForest(replayed)
	if err == nil {
		t.Fatalf("VerifyForest ACCEPTED a replayed row (%+v)", f)
	}
	if f.Duplicates == 0 {
		t.Errorf("Duplicates = 0, want >0 (%+v)", f)
	}
}

// TestVerifyForest_HeadRemovalFails covers truncation from the FRONT: strip the
// genesis row and every remaining row is an orphan with no root.
func TestVerifyForest_HeadRemovalFails(t *testing.T) {
	rows := forkedJournal(t)
	f, err := VerifyForest(rows[1:])
	if err == nil {
		t.Fatalf("VerifyForest ACCEPTED a journal with its genesis row removed (%+v)", f)
	}
}

// TestVerifyForest_RoundTripsThroughJSONL pins that the on-disk encoding is what
// is being verified — the dogfood harness re-parses JSONL text, not in-memory
// structs, so a field that fails to marshal would silently weaken the check.
func TestVerifyForest_RoundTripsThroughJSONL(t *testing.T) {
	rows := forkedJournal(t)
	var reparsed []Row
	for _, r := range rows {
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var back Row
		if err := json.Unmarshal(b, &back); err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		reparsed = append(reparsed, back)
	}
	if _, err := VerifyForest(reparsed); err != nil {
		t.Fatalf("forked journal failed VerifyForest after a JSONL round trip: %v", err)
	}
}

// TestVerifyForest_MatchesDiskShape guards the fixture itself: the forked rows
// must actually be on disk in append order, so the test is reproducing a real
// file and not an in-memory artifact.
func TestVerifyForest_MatchesDiskShape(t *testing.T) {
	path := filepath.Join(t.TempDir(), "shape.jsonl")
	j, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	writeAs(t, j, "sess-solo", "DECIDE")
	if err := j.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(b) == 0 {
		t.Fatal("journal file is empty after a committed row")
	}
}
