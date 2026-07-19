package trajctl

import (
	"math"
	"testing"
)

const dayMillis = int64(24 * 60 * 60 * 1000)

// scoreRowsFor counts the raw KindScore rows for an objective in a ledger.
func scoreRowsFor(rows []Row, objectiveID string) int {
	n := 0
	for _, r := range rows {
		if r.Kind == KindScore && r.Score != nil && r.Score.ObjectiveID == objectiveID {
			n++
		}
	}
	return n
}

// summaryRowsFor counts the KindSummary rows for an objective in a ledger.
func summaryRowsFor(rows []Row, objectiveID string) int {
	n := 0
	for _, r := range rows {
		if r.Kind == KindSummary && r.Summary != nil && r.Summary.ObjectiveID == objectiveID {
			n++
		}
	}
	return n
}

// assertDigestsWithinTolerance is the compaction round-trip check: the fold over
// compacted history must match the fold over full history within tolerance.
func assertDigestsWithinTolerance(t *testing.T, want, got []MethodDigest, tol float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("digest length = %d, want %d\nwant=%+v\ngot=%+v", len(got), len(want), want, got)
	}
	for i := range want {
		w, g := want[i], got[i]
		if w.ObjectiveID != g.ObjectiveID || w.Method != g.Method {
			t.Fatalf("digest[%d] key = (%s,%s), want (%s,%s)", i, g.ObjectiveID, g.Method, w.ObjectiveID, w.Method)
		}
		if w.Count != g.Count {
			t.Fatalf("digest[%d] %s/%s count = %d, want %d", i, w.ObjectiveID, w.Method, g.Count, w.Count)
		}
		if math.Abs(w.Sum-g.Sum) > tol {
			t.Fatalf("digest[%d] %s/%s sum = %v, want %v", i, w.ObjectiveID, w.Method, g.Sum, w.Sum)
		}
		if math.Abs(w.Mean()-g.Mean()) > tol {
			t.Fatalf("digest[%d] %s/%s mean = %v, want %v", i, w.ObjectiveID, w.Method, g.Mean(), w.Mean())
		}
		if math.Abs(w.Last-g.Last) > tol || math.Abs(w.First-g.First) > tol {
			t.Fatalf("digest[%d] %s/%s endpoints = (%v,%v), want (%v,%v)", i, w.ObjectiveID, w.Method, g.First, g.Last, w.First, w.Last)
		}
		if w.FirstUnixMillis != g.FirstUnixMillis || w.LastUnixMillis != g.LastUnixMillis {
			t.Fatalf("digest[%d] %s/%s timestamps = (%d,%d), want (%d,%d)", i, w.ObjectiveID, w.Method, g.FirstUnixMillis, g.LastUnixMillis, w.FirstUnixMillis, w.LastUnixMillis)
		}
	}
}

func latestCommitProgress(t *testing.T, s State, objectiveID string) float64 {
	t.Helper()
	c, ok := s.CurveFor(objectiveID)
	if !ok {
		t.Fatalf("CurveFor(%q) missing", objectiveID)
	}
	return c.Latest
}

// TestCompactionRoundTripsClosedObjectiveHistory is the issue #2570 done condition:
// a closed objective older than the retention window compacts to summary rows, the
// fold over the compacted history matches the fold over the full history within
// tolerance, the curve still folds a Latest for the compacted objective, and open
// or recently-closed objectives are left untouched. Compaction is idempotent.
func TestCompactionRoundTripsClosedObjectiveHistory(t *testing.T) {
	now := int64(1_000_000) * dayMillis
	oldBase := now - 60*dayMillis // well past a 14-day window

	rows := []Row{
		ObjectiveRecord(Objective{ID: "closed-old", Statement: "ship X", Status: StatusMet}),
		ObjectiveRecord(Objective{ID: "open-now", Statement: "ship Y", Status: StatusActive}),
		ObjectiveRecord(Objective{ID: "closed-recent", Statement: "ship Z", Status: StatusAbandoned}),
	}
	// closed-old: two methods of per-turn history, all older than the window.
	for i, v := range []float64{0.1, 0.2, 0.35, 0.5, 0.75, 1.0} {
		rows = append(rows, ScoreRecord(ScoreRow{ObjectiveID: "closed-old", Value: v, Method: CommitScorerMethod, Version: "v1", Witness: W3, UnixMillis: oldBase + int64(i)*dayMillis}))
	}
	for i, v := range []float64{0.2, 0.4, 0.6} {
		rows = append(rows, ScoreRecord(ScoreRow{ObjectiveID: "closed-old", Value: v, Method: ActivityDivergenceScorerMethod, Version: "v1", Witness: W2, UnixMillis: oldBase + int64(i)*dayMillis}))
	}
	// open-now: recent activity — must NOT compact.
	for i, v := range []float64{0.3, 0.6} {
		rows = append(rows, ScoreRecord(ScoreRow{ObjectiveID: "open-now", Value: v, Method: CommitScorerMethod, Version: "v1", Witness: W3, UnixMillis: now - int64(i+1)*dayMillis}))
	}
	// closed-recent: closed but inside the window — must NOT compact.
	rows = append(rows, ScoreRecord(ScoreRow{ObjectiveID: "closed-recent", Value: 0.9, Method: CommitScorerMethod, Version: "v1", Witness: W3, UnixMillis: now - 2*dayMillis}))

	before := DigestLedger(rows)
	compacted, stats := Compact(rows, CompactOptions{NowUnixMillis: now, RetentionDays: 14})
	after := DigestLedger(compacted)

	// The round-trip: fold over compacted history == fold over full history.
	assertDigestsWithinTolerance(t, before, after, 1e-9)

	if stats.ObjectivesCompacted != 1 {
		t.Fatalf("objectives compacted = %d, want 1 (only closed-old)", stats.ObjectivesCompacted)
	}
	if stats.ScoreRowsCompacted != 9 {
		t.Fatalf("score rows compacted = %d, want 9", stats.ScoreRowsCompacted)
	}
	if len(compacted) >= len(rows) {
		t.Fatalf("compaction did not shrink the ledger: %d -> %d", len(rows), len(compacted))
	}
	if n := scoreRowsFor(compacted, "closed-old"); n != 0 {
		t.Fatalf("closed-old still has %d raw score rows, want 0", n)
	}
	if n := summaryRowsFor(compacted, "closed-old"); n != 2 {
		t.Fatalf("closed-old summary rows = %d, want 2 (one per method)", n)
	}
	if n := scoreRowsFor(compacted, "open-now"); n != 2 {
		t.Fatalf("open-now was compacted (%d score rows left)", n)
	}
	if n := scoreRowsFor(compacted, "closed-recent"); n != 1 {
		t.Fatalf("closed-recent was compacted (%d score rows left)", n)
	}

	// The curve still folds over compacted history: Latest is preserved.
	fullLatest := latestCommitProgress(t, Fold(rows), "closed-old")
	compLatest := latestCommitProgress(t, Fold(compacted), "closed-old")
	if math.Abs(fullLatest-compLatest) > 1e-9 {
		t.Fatalf("curve Latest drifted across compaction: %v -> %v", fullLatest, compLatest)
	}
	if math.Abs(fullLatest-1.0) > 1e-9 {
		t.Fatalf("closed-old Latest = %v, want 1.0", fullLatest)
	}

	// Idempotent: compacting again is a fold-preserving no-op.
	twice, stats2 := Compact(compacted, CompactOptions{NowUnixMillis: now, RetentionDays: 14})
	assertDigestsWithinTolerance(t, after, DigestLedger(twice), 1e-9)
	if stats2.ScoreRowsCompacted != 0 {
		t.Fatalf("second compaction touched %d score rows, want 0 (idempotent)", stats2.ScoreRowsCompacted)
	}
	if summaryRowsFor(twice, "closed-old") != 2 {
		t.Fatalf("second compaction changed summary-row count")
	}
}

// TestCompactionSurvivesLedgerRoundTrip proves a summary row is a durable, valid,
// re-parseable ledger row: compacted rows marshal, parse back, and fold identically.
func TestCompactionSurvivesLedgerRoundTrip(t *testing.T) {
	now := int64(500_000) * dayMillis
	oldBase := now - 90*dayMillis
	rows := []Row{ObjectiveRecord(Objective{ID: "obj", Statement: "ship", Status: StatusMet})}
	for i, v := range []float64{0.25, 0.5, 1.0} {
		rows = append(rows, ScoreRecord(ScoreRow{ObjectiveID: "obj", Value: v, Method: CommitScorerMethod, Version: "v1", Witness: W3, UnixMillis: oldBase + int64(i)*dayMillis}))
	}
	compacted, _ := Compact(rows, CompactOptions{NowUnixMillis: now})

	var sb string
	for _, r := range compacted {
		sb += mustJSON(t, r) + "\n"
	}
	parsed := ParseLedger(sb)
	assertDigestsWithinTolerance(t, DigestLedger(compacted), DigestLedger(parsed), 1e-9)

	// A summary row must pass Validate to survive ParseLedger's filter.
	for _, r := range compacted {
		if err := Validate(r); err != nil {
			t.Fatalf("compacted row fails Validate: %v (%+v)", err, r)
		}
	}
}

// TestIsCurrentSchemaPinsVersion documents the schema pin: only the pinned schema id
// is current, and SchemaVersion is its integer major.
func TestIsCurrentSchemaPinsVersion(t *testing.T) {
	if !IsCurrentSchema(Schema) {
		t.Fatalf("pinned schema %q not current", Schema)
	}
	if IsCurrentSchema("fak-trajctl/2") || IsCurrentSchema("other") {
		t.Fatalf("a foreign/newer schema must not read as current")
	}
	if SchemaVersion != 1 {
		t.Fatalf("SchemaVersion = %d, want 1", SchemaVersion)
	}
}
