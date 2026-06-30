package heavinessscore

import (
	"strings"
	"testing"
)

// These exercise each KPI on a DEFECT fixture (the friction it must catch) and a CLEAN fixture
// (the healthy surface it must pass), plus the parsers that turn source text into a Surface. The
// KPIs are pure functions of a Surface, so the fixtures are Surfaces built by hand -- no tree.

func TestParseVerbs_DistinctSortedFromDispatch(t *testing.T) {
	src := `
		switch os.Args[1] {
		case "run":
			cmdRun(rest)
		case "commit":
			cmdCommit(rest)
		case "commit": // a duplicated case (alias) must collapse to one verb
			cmdCommit(rest)
		case "operator":
			cmdOperator(rest)
		}
		// a non-verb case on some other switch must not be miscounted:
		switch mode { case "Upper": }
	`
	got := ParseVerbs(src)
	want := []string{"commit", "operator", "run"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("ParseVerbs = %v, want %v", got, want)
	}
}

func TestIsMetaVerb(t *testing.T) {
	for _, v := range []string{"conflation-scorecard", "dojo-rsi", "loop-score", "scorecard"} {
		if !isMetaVerb(v) {
			t.Errorf("%q should be a meta verb", v)
		}
	}
	for _, v := range []string{"commit", "run", "operator", "guard"} {
		if isMetaVerb(v) {
			t.Errorf("%q should NOT be a meta verb", v)
		}
	}
}

func TestParseSurface_CountsFromSource(t *testing.T) {
	// ParseSurface reads files; here we drive the regexes directly through a small synthetic
	// guard/dos source to prove the flag + reason counters.
	guard := `fs.String("a","",""); fs.Bool("b",false,""); fs.Int("c",0,""); fs.Duration("d",0,"")`
	if got := len(reGuardFlag.FindAllString(guard, -1)); got != 4 {
		t.Errorf("front-door flag count = %d, want 4", got)
	}
	dos := "[reasons.OFF_TRUNK]\nx=1\n[reasons.PUBLIC_LEAK]\n[other.THING]\n[reasons.A_B_C]\n"
	if got := len(reReasonsBlock.FindAllString(dos, -1)); got != 3 {
		t.Errorf("refusal-reason count = %d, want 3", got)
	}
}

func TestDocMapCoversSteering_UnindexedIsDebt(t *testing.T) {
	// A doc map that mentions "human-steerable" but not the actual surface must still be a defect:
	// the incidental Charter word must NOT satisfy the specific tokens.
	s := Surface{DocMap: strings.ToLower("the ten principles: agentic, human-steerable, win-win-win")}
	k := kpiDocMapCoversSteering(s)
	if len(k.Defects) != 2 {
		t.Fatalf("an unindexed doc map should yield 2 defects, got %d: %v", len(k.Defects), k.Defects)
	}
	if k.Score != 0 {
		t.Errorf("score=%v want 0 when nothing is covered", k.Score)
	}
}

func TestDocMapCoversSteering_IndexedIsClean(t *testing.T) {
	s := Surface{DocMap: strings.ToLower(
		"[Steerability scorecard](docs/STEERABILITY-SCORECARD.md): run `fak steering`. " +
			"[Operator-heaviness](docs/OPERATOR-HEAVINESS.md): heaviness_pressure.")}
	k := kpiDocMapCoversSteering(s)
	if len(k.Defects) != 0 {
		t.Fatalf("an indexed doc map is clean, got defects %v", k.Defects)
	}
	if k.Score != 100 {
		t.Errorf("clean score=%v want 100", k.Score)
	}
}

func TestAppealChannelWired(t *testing.T) {
	if d := kpiAppealChannelWired(Surface{AppealWired: false}).Defects; len(d) != 1 {
		t.Errorf("an unwired appeal channel must be 1 defect, got %v", d)
	}
	if d := kpiAppealChannelWired(Surface{AppealWired: true}).Defects; len(d) != 0 {
		t.Errorf("a wired appeal channel is clean, got %v", d)
	}
}

func TestCLIVerbCount_SoftBelowCeilingHardAbove(t *testing.T) {
	below := kpiCLIVerbCount(Surface{Verbs: make([]string, 136)})
	if len(below.Defects) != 0 {
		t.Errorf("136 verbs (under the hard ceiling) must NOT be hard debt, got %v", below.Defects)
	}
	if len(below.Soft) != 1 {
		t.Errorf("136 verbs (over the soft line) must be a soft signal, got %v", below.Soft)
	}
	above := kpiCLIVerbCount(Surface{Verbs: make([]string, verbHardCeiling+1)})
	if len(above.Defects) != 1 {
		t.Errorf("a verb count past the hard ceiling must be hard debt, got %v", above.Defects)
	}
	clean := kpiCLIVerbCount(Surface{Verbs: make([]string, verbSoftLine-1)})
	if len(clean.Soft) != 0 || clean.Score != 100 {
		t.Errorf("under the soft line must be clean score 100, got soft=%v score=%v", clean.Soft, clean.Score)
	}
}

func TestFrontDoorFlagBurden_HardPastCeiling(t *testing.T) {
	if d := kpiFrontDoorFlagBurden(Surface{FrontDoorFlags: flagHardCeiling + 1}).Defects; len(d) != 1 {
		t.Errorf("a flag count past the ceiling must be hard debt, got %v", d)
	}
	if k := kpiFrontDoorFlagBurden(Surface{FrontDoorFlags: 49}); len(k.Defects) != 0 || len(k.Soft) != 1 {
		t.Errorf("49 flags must be soft, not hard, got defects=%v soft=%v", k.Defects, k.Soft)
	}
}

func TestRefusalVocabSize_SoftOnly(t *testing.T) {
	k := kpiRefusalVocabSize(Surface{RefusalReasons: 20})
	if len(k.Defects) != 0 {
		t.Errorf("refusal vocab is SOFT and must never emit hard debt, got %v", k.Defects)
	}
	if len(k.Soft) != 1 {
		t.Errorf("20 reasons (over soft line) must be a soft signal, got %v", k.Soft)
	}
}

func TestPressure_SumsOverages(t *testing.T) {
	// 136 verbs (+46), 49 flags (+29), 20 reasons (+8), 15/136 meta = 11% (+3) = 86.
	s := Surface{
		Verbs:          make([]string, 136),
		MetaVerbs:      make([]string, 15),
		FrontDoorFlags: 49,
		RefusalReasons: 20,
	}
	if got := Pressure(s); got != 86 {
		t.Errorf("Pressure = %d, want 86", got)
	}
	// A surface at/under every soft line has zero pressure.
	light := Surface{Verbs: make([]string, verbSoftLine), FrontDoorFlags: flagSoftLine, RefusalReasons: reasonSoftLine}
	if got := Pressure(light); got != 0 {
		t.Errorf("a surface at the soft lines must have zero pressure, got %d", got)
	}
}

func TestMagnitudeScore_Bounds(t *testing.T) {
	if magnitudeScore(10, 20, 80) != 100 {
		t.Error("at/below soft line must be 100")
	}
	if magnitudeScore(80, 20, 80) != 0 {
		t.Error("at/above ceiling must be 0")
	}
	if got := magnitudeScore(50, 20, 80); got <= 0 || got >= 100 {
		t.Errorf("between lines must be strictly 0<score<100, got %v", got)
	}
}

func TestBuild_DebtCountsHardDefectsOnly(t *testing.T) {
	// Build over a throwaway empty dir: no source files -> 0 verbs/flags/reasons, unwired appeal,
	// undiscoverable doc map. Debt must be exactly the HARD defects (2 docmap + 1 appeal = 3), and
	// the SOFT magnitude KPIs (all zero/empty) must add no debt.
	p := Build(t.TempDir())
	debt, ok := p.Corpus[DebtKey].(int)
	if !ok {
		t.Fatalf("corpus[%s] is not an int: %T", DebtKey, p.Corpus[DebtKey])
	}
	if debt != 3 {
		t.Errorf("empty-tree debt = %d, want 3 (2 docmap + 1 appeal)", debt)
	}
	if p.Schema != Schema {
		t.Errorf("schema = %q, want %q", p.Schema, Schema)
	}
	if _, ok := p.Corpus["heaviness_pressure"]; !ok {
		t.Error("corpus must carry heaviness_pressure")
	}
}
