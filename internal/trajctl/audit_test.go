package trajctl

import "testing"

// fixtureResolver marks any evidence ref whose Ref is in dead as dangling and
// every other ref as verified, so the audit is deterministic and host-free.
func fixtureResolver(dead ...string) EvidenceResolver {
	set := map[string]bool{}
	for _, d := range dead {
		set[d] = true
	}
	return func(ev EvidenceRef) EvidenceStatus {
		if set[ev.Ref] {
			return EvidenceDangling
		}
		return EvidenceVerified
	}
}

func scoreRow(obj string, value float64, w WitnessRung, evRef string) Row {
	return ScoreRecord(ScoreRow{
		ObjectiveID: obj,
		Value:       value,
		Method:      "commit-progress",
		Version:     "v1",
		Witness:     w,
		Evidence:    []EvidenceRef{{Kind: "commit", Ref: evRef}},
	})
}

func TestAuditReportsDanglingEvidenceWorstFirst(t *testing.T) {
	rows := []Row{
		scoreRow("obj-1", 0.4, W2, "live-a"),  // verified, not stale
		scoreRow("obj-1", 0.9, W3, "dead-hi"), // dangling, high value -> worst
		scoreRow("obj-1", 0.3, W3, "dead-lo"), // dangling, lower value
	}
	rep := Audit(rows, fixtureResolver("dead-hi", "dead-lo"))

	if rep.Scores != 3 {
		t.Fatalf("Scores = %d, want 3", rep.Scores)
	}
	if rep.StaleRows != 2 || rep.Dangling != 2 {
		t.Fatalf("StaleRows=%d Dangling=%d, want 2 and 2", rep.StaleRows, rep.Dangling)
	}
	if len(rep.Stale) != 2 {
		t.Fatalf("len(Stale) = %d, want 2", len(rep.Stale))
	}
	if got := rep.Stale[0].Evidence[0].Ref.Ref; got != "dead-hi" {
		t.Fatalf("worst-first ordering: Stale[0] ref = %q, want dead-hi", got)
	}
	if got := rep.Stale[1].Evidence[0].Ref.Ref; got != "dead-lo" {
		t.Fatalf("Stale[1] ref = %q, want dead-lo", got)
	}
	if rep.Stale[0].Evidence[0].Status != EvidenceDangling {
		t.Fatalf("Stale[0] status = %q, want dangling", rep.Stale[0].Evidence[0].Status)
	}
}

func TestFoldVerifiedDemotesStaleRowToW0(t *testing.T) {
	rows := []Row{
		scoreRow("obj-1", 0.9, W3, "dead"), // dangling -> must be demoted
		scoreRow("obj-1", 0.5, W3, "live"), // verified -> keeps its rung
	}
	st := FoldVerified(rows, fixtureResolver("dead"))

	if len(st.Scores) != 2 {
		t.Fatalf("len(Scores) = %d, want 2", len(st.Scores))
	}
	if st.Scores[0].Witness != W0 {
		t.Fatalf("stale row witness = %q, want W0 (demoted)", st.Scores[0].Witness)
	}
	if st.Scores[1].Witness != W3 {
		t.Fatalf("verified row witness = %q, want W3 (unchanged)", st.Scores[1].Witness)
	}
	// The plain fold must NOT demote — demotion is a read-time verification act.
	if plain := Fold(rows); plain.Scores[0].Witness != W3 {
		t.Fatalf("Fold demoted a row it should not have: %q", plain.Scores[0].Witness)
	}
}

func TestAuditCleanLedgerIsClean(t *testing.T) {
	rows := []Row{
		scoreRow("obj-1", 0.4, W2, "live-a"),
		scoreRow("obj-1", 0.7, W3, "live-b"),
	}
	rep := Audit(rows, fixtureResolver()) // nothing dangling

	if rep.StaleRows != 0 || rep.Dangling != 0 || len(rep.Stale) != 0 {
		t.Fatalf("clean ledger audited dirty: StaleRows=%d Dangling=%d Stale=%d", rep.StaleRows, rep.Dangling, len(rep.Stale))
	}
	// A clean ledger folds identically whether or not it is re-verified.
	if FoldVerified(rows, fixtureResolver()).Scores[1].Witness != W3 {
		t.Fatal("clean ledger demoted a verified row")
	}
}

func TestAuditNilResolverNeverDemotes(t *testing.T) {
	rows := []Row{scoreRow("obj-1", 0.9, W3, "whatever")}
	if rep := Audit(rows, nil); rep.StaleRows != 0 {
		t.Fatalf("nil resolver marked %d rows stale, want 0 (unknown is conservative)", rep.StaleRows)
	}
	if st := FoldVerified(rows, nil); st.Scores[0].Witness != W3 {
		t.Fatalf("nil resolver demoted a row to %q, want W3 untouched", st.Scores[0].Witness)
	}
}
