package wipref

import "testing"

// TestClassifyVocabulary walks every census class the classifier can emit, including
// the precedence edges (liveness beats a landing; a landing beats the closed split;
// an unresolved dead ref is UNKNOWN, never a clean estimate).
func TestClassifyVocabulary(t *testing.T) {
	cases := []struct {
		name  string
		facts CensusFacts
		want  CensusClass
	}{
		{"live short-circuits even if landed", CensusFacts{Live: true, Landed: true}, CensusLive},
		{"live short-circuits even if subsumed", CensusFacts{Live: true, Resolved: true, Subsumed: true}, CensusLive},
		{"landed beats the closed split", CensusFacts{Landed: true, Resolved: true}, CensusLanded},
		{"dead + unresolved is unknown", CensusFacts{Resolved: false}, CensusUnknown},
		{"dead + byte-identical payload is a clean estimate", CensusFacts{Resolved: true, PayloadRead: true, PayloadFiles: 1}, CensusClosedCleanEstimate},
		{"dead + diverged payload is first-class", CensusFacts{Resolved: true, PayloadRead: true, PayloadFiles: 1, PayloadDiverged: 1}, CensusDiverged},
		{"dead + absent payload is recoverable", CensusFacts{Resolved: true, PayloadRead: true, PayloadFiles: 1, PayloadAbsent: 1}, CensusClosedDirtyRecoverable},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Classify(c.facts)
			if got != c.want {
				t.Errorf("Classify(%+v) = %s, want %s", c.facts, got, c.want)
			}
			// Every class must carry a non-empty, auditable reason.
			if CensusReason(got) == "" {
				t.Errorf("class %s has an empty reason", got)
			}
		})
	}
}

// TestCensusSafetyInvariant is the headline #5340 safety guarantee, stated as a pure
// fact: a dead session (not live, not landed) whose delta is non-empty and NOT
// subsumed by HEAD classifies CLOSED_DIRTY_RECOVERABLE — kept — and NEVER
// CLOSED_CLEAN_ESTIMATE. A future reap that acted on the census must not be able to
// collect this ref; if this test ever flips to a clean estimate, recoverable WIP
// would be at risk of deletion.
func TestCensusSafetyInvariant(t *testing.T) {
	dirty := CensusFacts{Live: false, Landed: false, Resolved: true, PayloadRead: true, PayloadFiles: 1, PayloadAbsent: 1}
	got := Classify(dirty)
	if got == CensusClosedCleanEstimate {
		t.Fatalf("SAFETY VIOLATION: a non-empty unlanded dead-session delta classified %s (collectible)", got)
	}
	if got != CensusClosedDirtyRecoverable {
		t.Fatalf("Classify(dirty) = %s, want CLOSED_DIRTY_RECOVERABLE (kept)", got)
	}
}

// TestDeltaSubsumed exercises the line-level subsumption oracle over the cases that
// decide clean-estimate vs recoverable.
func TestDeltaSubsumed(t *testing.T) {
	head := map[string]map[string]bool{
		"f.go": {"kept one": true, "kept two": true},
	}
	cases := []struct {
		name    string
		added   map[string][]string
		removed map[string][]string
		want    bool
	}{
		{
			name:  "every added line already in HEAD -> subsumed",
			added: map[string][]string{"f.go": {"kept one", "kept two"}},
			want:  true,
		},
		{
			name:  "an added line HEAD lacks -> not subsumed",
			added: map[string][]string{"f.go": {"kept one", "brand new work"}},
			want:  false,
		},
		{
			name:  "added line in a file HEAD does not have -> not subsumed",
			added: map[string][]string{"newfile.go": {"anything"}},
			want:  false,
		},
		{
			// The safety teeth: a checkpoint that removed a line still living in HEAD
			// (its deletion not yet landed) is recoverable, NOT subsumed — even though it
			// added nothing (the added-line test alone would vacuously pass).
			name:    "removed a line still in HEAD -> not subsumed",
			removed: map[string][]string{"f.go": {"kept one"}},
			want:    false,
		},
		{
			name:    "removed a line HEAD no longer has -> that removal is reflected",
			removed: map[string][]string{"f.go": {"already gone"}},
			want:    true,
		},
		{
			name: "no added or removed content lines -> not subsumed (emptiness is separate)",
			want: false,
		},
		{
			name:    "added present AND removed reflected -> subsumed",
			added:   map[string][]string{"f.go": {"kept two"}},
			removed: map[string][]string{"f.go": {"already gone"}},
			want:    true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := DeltaSubsumed(c.added, c.removed, head); got != c.want {
				t.Errorf("DeltaSubsumed = %v, want %v", got, c.want)
			}
		})
	}
}

// TestBuildCensusFoldAndSort proves the aggregate fold: correct per-class tallies, a
// correct total, and a deterministic session-sorted (object-tie-broken) verdict order.
func TestBuildCensusFold(t *testing.T) {
	verdicts := []CensusVerdict{
		{Session: "z", Object: "o-z", Class: CensusClosedDirtyRecoverable},
		{Session: "a", Object: "o-a2", Class: CensusLanded},
		{Session: "a", Object: "o-a1", Class: CensusLive},
		{Session: "m", Object: "o-m", Class: CensusClosedCleanEstimate},
		{Session: "q", Object: "o-q", Class: CensusClosedCleanEstimate},
		{Session: "u", Object: "o-u", Class: CensusUnknown},
	}
	rep := BuildCensus(verdicts)

	wantCounts := CensusCounts{
		Total: 6, Landed: 1, Live: 1,
		ClosedCleanEstimate: 2, ClosedDirtyRecoverable: 1, Unknown: 1,
	}
	if rep.Counts != wantCounts {
		t.Errorf("counts = %+v, want %+v", rep.Counts, wantCounts)
	}
	// Session-sorted; ties within a session broken by object id ("o-a1" < "o-a2").
	wantOrder := []string{"o-a1", "o-a2", "o-m", "o-q", "o-u", "o-z"}
	if len(rep.Verdicts) != len(wantOrder) {
		t.Fatalf("verdict count = %d, want %d", len(rep.Verdicts), len(wantOrder))
	}
	for i, want := range wantOrder {
		if rep.Verdicts[i].Object != want {
			t.Errorf("verdict[%d].Object = %q, want %q", i, rep.Verdicts[i].Object, want)
		}
	}
	// BuildCensus must not mutate the caller's slice order.
	if verdicts[0].Object != "o-z" {
		t.Errorf("BuildCensus reordered the input slice in place")
	}
}
