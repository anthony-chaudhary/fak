package catchupscore

import "testing"

func lvl(behind, total int) *Level { return &Level{Behind: behind, Total: total} }

// No evidence at any level folds to the honest INSUFFICIENT card: caught up 1.0,
// nothing outstanding, ok, and an empty worklist -- never a spurious 0/F.
func TestCatchUpNoEvidenceIsInsufficient(t *testing.T) {
	number, backlog, present, worklist := CatchUp(Facts{})
	if number != 1 {
		t.Errorf("empty facts: number = %v, want 1.0", number)
	}
	if backlog != 0 {
		t.Errorf("empty facts: backlog = %d, want 0", backlog)
	}
	if present != 0 {
		t.Errorf("empty facts: present = %d, want 0", present)
	}
	if len(worklist) != 0 {
		t.Errorf("empty facts: worklist = %v, want empty", worklist)
	}

	p := Compose(Facts{})
	if !p.OK {
		t.Errorf("empty facts: payload not OK, want OK")
	}
	if got := p.Corpus[DebtKey]; got != 0 {
		t.Errorf("empty facts: %s = %v, want 0", DebtKey, got)
	}
	if got := p.Corpus[BacklogKey]; got != 0 {
		t.Errorf("empty facts: %s = %v, want 0", BacklogKey, got)
	}
	if got := p.Corpus["levels_total"]; got != len(Levels) {
		t.Errorf("levels_total = %v, want %d", got, len(Levels))
	}
}

// A fully caught-up level (behind 0) clears the pass line: no debt, ok.
func TestCatchUpCaughtUpLevelIsClean(t *testing.T) {
	number, backlog, present, worklist := CatchUp(Facts{Index: lvl(0, 40)})
	if number != 1 {
		t.Errorf("caught up: number = %v, want 1.0", number)
	}
	if backlog != 0 {
		t.Errorf("caught up: backlog = %d, want 0", backlog)
	}
	if present != 1 {
		t.Errorf("caught up: present = %d, want 1", present)
	}
	if len(worklist) != 1 || worklist[0].InDebt {
		t.Errorf("caught up: worklist = %+v, want one row not in debt", worklist)
	}
	if p := Compose(Facts{Index: lvl(0, 40)}); !p.OK || p.Corpus[DebtKey] != 0 {
		t.Errorf("caught up: ok=%v debt=%v, want ok/0", p.OK, p.Corpus[DebtKey])
	}
}

// A behind level books exactly one debt, flips ok, and contributes its raw behind
// count to the unbounded backlog.
func TestCatchUpBehindLevelBooksDebt(t *testing.T) {
	// 10 untriaged of 50 open -> 40/50 = 0.8 caught up == pass line (NOT behind).
	if p := Compose(Facts{Intake: lvl(10, 50)}); !p.OK {
		t.Errorf("fraction exactly on pass line must not be debt: ok=%v debt=%v", p.OK, p.Corpus[DebtKey])
	}
	// 20 untriaged of 50 open -> 30/50 = 0.6 caught up < 0.8 -> behind.
	number, backlog, present, worklist := CatchUp(Facts{Intake: lvl(20, 50)})
	if present != 1 {
		t.Fatalf("present = %d, want 1", present)
	}
	if backlog != 20 {
		t.Errorf("backlog = %d, want 20 (raw behind count)", backlog)
	}
	if worklist[0].CaughtUp != 0.6 || !worklist[0].InDebt {
		t.Errorf("row = %+v, want caught_up 0.6 in debt", worklist[0])
	}
	if number != 0.6 {
		t.Errorf("number = %v, want 0.6", number)
	}
	p := Compose(Facts{Intake: lvl(20, 50)})
	if p.OK || p.Corpus[DebtKey] != 1 {
		t.Errorf("behind: ok=%v debt=%v, want not-ok/1", p.OK, p.Corpus[DebtKey])
	}
}

// The backlog headline is UNBOUNDED: a level three times as far behind reads three
// times as large, never saturating a bounded 0..1 bar. Both levels sit at fraction
// 0.0 (fully behind) yet contribute their distinct raw counts.
func TestCatchUpBacklogIsUnbounded(t *testing.T) {
	_, backlog1, _, _ := CatchUp(Facts{Intake: lvl(100, 100)})
	_, backlog3, _, _ := CatchUp(Facts{Intake: lvl(300, 300)})
	if backlog1 != 100 || backlog3 != 300 {
		t.Fatalf("backlog1=%d backlog3=%d, want 100 and 300", backlog1, backlog3)
	}
	if backlog3 != 3*backlog1 {
		t.Errorf("backlog must scale linearly with behind: %d vs 3*%d", backlog3, backlog1)
	}
	// Sum across levels: 300 untriaged + 12 red checks + 5 overdue loops = 317.
	_, backlog, present, _ := CatchUp(Facts{
		Intake: lvl(300, 300),
		Trunk:  lvl(12, 12),
		Loops:  lvl(5, 8),
	})
	if present != 3 || backlog != 317 {
		t.Errorf("multi-level backlog: present=%d backlog=%d, want 3 and 317", present, backlog)
	}
}

// The worklist is worst-first: lowest caught-up fraction first, then most-behind,
// then canonical level order breaking ties.
func TestCatchUpWorklistWorstFirst(t *testing.T) {
	f := Facts{
		Intake:      lvl(0, 100), // 1.00 caught up
		Measurement: lvl(3, 10),  // 0.70 caught up
		Index:       lvl(9, 10),  // 0.10 caught up  <- worst
		Loops:       lvl(3, 10),  // 0.70 caught up (ties Measurement; canonical order wins)
	}
	_, _, present, worklist := CatchUp(f)
	if present != 4 {
		t.Fatalf("present = %d, want 4", present)
	}
	got := []string{worklist[0].Level, worklist[1].Level, worklist[2].Level, worklist[3].Level}
	want := []string{LevelIndex, LevelMeasurement, LevelLoops, LevelIntake}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("worklist order = %v, want %v", got, want)
		}
	}
}

// A total below behind is clamped to behind (fully behind reads 0.0, never a
// nonsense negative fraction), and the row reports the clamped total.
func TestCatchUpTotalBelowBehindClamps(t *testing.T) {
	_, _, _, worklist := CatchUp(Facts{Trunk: &Level{Behind: 5, Total: 2}})
	if worklist[0].CaughtUp != 0 {
		t.Errorf("caught_up = %v, want 0.0 (fully behind)", worklist[0].CaughtUp)
	}
	if worklist[0].Total != 5 {
		t.Errorf("total = %d, want 5 (clamped up to behind)", worklist[0].Total)
	}
}

// Every measured level with a defect names its retire action; the composite grade
// is driven by the mean caught-up fraction (GradeStd over 100*number).
func TestComposeDefectNamesRetireAction(t *testing.T) {
	p := Compose(Facts{Index: lvl(9, 10)})
	var found bool
	for _, k := range p.KPIs {
		if k.Key == LevelIndex {
			if len(k.Defects) != 1 {
				t.Fatalf("index KPI defects = %v, want exactly one", k.Defects)
			}
			found = true
		}
	}
	if !found {
		t.Fatalf("no index KPI in payload")
	}
}
