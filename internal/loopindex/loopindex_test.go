package loopindex

import (
	"bytes"
	"testing"
)

// stage builds a Stage with one keystone probe (pass=keystonePass) plus n extra
// non-keystone probes of which extraPass pass, so a test can dial both Wired and
// Health precisely.
func stage(name string, floor float64, keystonePass bool, extra, extraPass int) Stage {
	probes := []Probe{{Name: name + "_keystone", Detail: "keystone", Keystone: true, Pass: keystonePass}}
	for i := 0; i < extra; i++ {
		probes = append(probes, Probe{Name: name + "_p", Detail: "probe", Pass: i < extraPass})
	}
	return Stage{Name: name, Signal: name + "-signal", Floor: floor, Probes: probes}
}

// fullLoop builds the six canonical stages with the given per-stage (keystonePass,
// extra, extraPass), each floor 0.5.
func fullLoop(spec map[string][3]int) Loop {
	order := []string{StageOrient, StagePlan, StageAct, StageVerify, StageShip, StageLearn}
	var stages []Stage
	for _, nm := range order {
		s := spec[nm]
		stages = append(stages, stage(nm, 0.5, s[0] == 1, s[1], s[2]))
	}
	return Loop{Stages: stages}
}

func TestScoreDeterministic(t *testing.T) {
	loop := fullLoop(map[string][3]int{
		StageOrient: {0, 2, 0}, // unwired
		StagePlan:   {0, 2, 1},
		StageAct:    {1, 2, 2}, // wired, healthy
		StageVerify: {1, 2, 0}, // wired but below floor
		StageShip:   {0, 1, 0},
		StageLearn:  {1, 3, 3}, // wired, healthy
	})
	a := Score(loop)
	b := Score(loop)
	if a.Corpus != b.Corpus {
		t.Fatalf("Score not deterministic: %+v vs %+v", a.Corpus, b.Corpus)
	}
}

func TestDebtCountsUnwiredAndBelowFloor(t *testing.T) {
	loop := fullLoop(map[string][3]int{
		StageOrient: {0, 2, 0}, // unwired -> debt
		StagePlan:   {0, 2, 1}, // unwired -> debt
		StageAct:    {1, 2, 2}, // wired health 1.0 -> ok
		StageVerify: {1, 2, 0}, // wired health 1/3 < 0.5 -> debt
		StageShip:   {0, 1, 0}, // unwired -> debt
		StageLearn:  {1, 3, 3}, // wired health 1.0 -> ok
	})
	rep := Score(loop)
	if rep.Corpus.LoopIndexDebt != 4 {
		t.Fatalf("loopindex_debt = %d, want 4", rep.Corpus.LoopIndexDebt)
	}
	if rep.Corpus.WiredStages != 3 {
		t.Fatalf("wired stages = %d, want 3", rep.Corpus.WiredStages)
	}
	if rep.OK {
		t.Fatal("OK should be false with debt > 0")
	}
}

func TestAllWitnessedIsClean(t *testing.T) {
	loop := fullLoop(map[string][3]int{
		StageOrient: {1, 1, 1},
		StagePlan:   {1, 1, 1},
		StageAct:    {1, 1, 1},
		StageVerify: {1, 1, 1},
		StageShip:   {1, 1, 1},
		StageLearn:  {1, 1, 1},
	})
	rep := Score(loop)
	if rep.Corpus.LoopIndexDebt != 0 {
		t.Fatalf("debt = %d, want 0", rep.Corpus.LoopIndexDebt)
	}
	if rep.Corpus.LoopIndex != 100 {
		t.Fatalf("loop-index = %d, want 100", rep.Corpus.LoopIndex)
	}
	if rep.Corpus.Value != 1 || rep.Corpus.LoopValue != 1 || rep.Corpus.WitnessedValue != 1 || rep.Corpus.LegacyScore != 100 || rep.Corpus.LegacyScoreScale != 100 {
		t.Fatalf("continuous value fields not stamped: %+v", rep.Corpus)
	}
	if rep.Corpus.Grade != "A" {
		t.Fatalf("grade = %q, want A", rep.Corpus.Grade)
	}
	if !rep.OK || rep.Verdict != "OK" {
		t.Fatalf("want OK verdict, got %s", rep.Verdict)
	}
}

func TestAllVibeIsZeroIndexMaxDebt(t *testing.T) {
	loop := fullLoop(map[string][3]int{
		StageOrient: {0, 1, 0},
		StagePlan:   {0, 1, 0},
		StageAct:    {0, 1, 0},
		StageVerify: {0, 1, 0},
		StageShip:   {0, 1, 0},
		StageLearn:  {0, 1, 0},
	})
	rep := Score(loop)
	if rep.Corpus.LoopIndex != 0 {
		t.Fatalf("loop-index = %d, want 0 when nothing is wired", rep.Corpus.LoopIndex)
	}
	if rep.Corpus.LoopIndexDebt != 6 {
		t.Fatalf("debt = %d, want 6", rep.Corpus.LoopIndexDebt)
	}
	if rep.Corpus.WitnessedIndex != 0 {
		t.Fatalf("witnessed-index = %d, want 0 with no wired stages", rep.Corpus.WitnessedIndex)
	}
}

func TestUnwiredContributesZeroButWitnessedIgnoresIt(t *testing.T) {
	// Three wired+healthy stages, three unwired. LoopIndex averages over all six
	// (unwired = 0) -> 50; WitnessedIndex averages only the wired -> 100.
	loop := fullLoop(map[string][3]int{
		StageOrient: {1, 1, 1},
		StagePlan:   {1, 1, 1},
		StageAct:    {1, 1, 1},
		StageVerify: {0, 1, 0},
		StageShip:   {0, 1, 0},
		StageLearn:  {0, 1, 0},
	})
	rep := Score(loop)
	if rep.Corpus.LoopIndex != 50 {
		t.Fatalf("loop-index = %d, want 50", rep.Corpus.LoopIndex)
	}
	if rep.Corpus.Value != 0.5 || rep.KPIs[0].Value != 1 || rep.KPIs[3].Value != 0 {
		t.Fatalf("continuous values = corpus %+v kpis %+v", rep.Corpus, rep.KPIs)
	}
	if rep.Corpus.WitnessedIndex != 100 {
		t.Fatalf("witnessed-index = %d, want 100", rep.Corpus.WitnessedIndex)
	}
}

func TestWorstFirstIsLoopOrder(t *testing.T) {
	// orient ok, plan failing, act failing -> worst-first should pick plan (earlier
	// in loop order), not act.
	loop := fullLoop(map[string][3]int{
		StageOrient: {1, 1, 1},
		StagePlan:   {0, 1, 0},
		StageAct:    {0, 1, 0},
		StageVerify: {1, 1, 1},
		StageShip:   {1, 1, 1},
		StageLearn:  {1, 1, 1},
	})
	rep := Score(loop)
	worst := worstStage(rep.KPIs)
	if worst.Name != StagePlan {
		t.Fatalf("worst-first stage = %q, want plan", worst.Name)
	}
}

func TestMalformedShapeIsMaxDebtNotPanic(t *testing.T) {
	// Wrong stage count.
	rep := Score(Loop{Stages: []Stage{stage(StageOrient, 0.5, true, 1, 1)}})
	if rep.OK {
		t.Fatal("malformed loop must not be OK")
	}
	if rep.Corpus.LoopIndexDebt != 6 {
		t.Fatalf("malformed debt = %d, want 6 (one per canonical stage)", rep.Corpus.LoopIndexDebt)
	}
	// Wrong order.
	bad := fullLoop(map[string][3]int{
		StageOrient: {1, 1, 1}, StagePlan: {1, 1, 1}, StageAct: {1, 1, 1},
		StageVerify: {1, 1, 1}, StageShip: {1, 1, 1}, StageLearn: {1, 1, 1},
	})
	bad.Stages[0], bad.Stages[1] = bad.Stages[1], bad.Stages[0]
	if Score(bad).OK {
		t.Fatal("out-of-order loop must not be OK")
	}
}

func TestRenderMentionsHeadline(t *testing.T) {
	loop := fullLoop(map[string][3]int{
		StageOrient: {1, 1, 1}, StagePlan: {0, 1, 0}, StageAct: {1, 1, 1},
		StageVerify: {1, 1, 1}, StageShip: {0, 1, 0}, StageLearn: {1, 1, 1},
	})
	var b bytes.Buffer
	Render(&b, Score(loop))
	out := b.String()
	for _, want := range []string{"loopindex_debt", "loop-index value=", "legacy score", "plan", "DEBT"} {
		if !bytes.Contains([]byte(out), []byte(want)) {
			t.Errorf("Render output missing %q:\n%s", want, out)
		}
	}
	if bytes.Contains([]byte(out), []byte("/100")) {
		t.Fatalf("Render output must use continuous values, got:\n%s", out)
	}
}
