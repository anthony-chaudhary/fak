package sessionobs

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"
)

// cleanCorpus is a minimal corpus that satisfies every HARD rung: linked, both
// value and waste classes present, structured, with behavior signals.
func cleanCorpus() []Record {
	return []Record{
		{SessionID: "a", AssistantTurns: 12, ToolCalls: 30, OutputTokens: 4000,
			Outcome: OutcomeShipped, Signals: Signals{Commits: 1, GoalEvents: 1}},
		{SessionID: "b", AssistantTurns: 8, ToolCalls: 14, OutputTokens: 1200,
			Outcome: OutcomeStopped, Signals: Signals{StopEvents: 1, GuardRefusals: 2}},
		{SessionID: "c", AssistantTurns: 5, ToolCalls: 9, OutputTokens: 600,
			Outcome: OutcomeNoOp},
	}
}

func fullPipeline() Pipeline {
	return Pipeline{CorpusCommitted: true, LoopConsumes: true, Registered: true}
}

func TestScoreCleanCorpusZeroDebt(t *testing.T) {
	rep := Score(cleanCorpus(), fullPipeline())
	if rep.Corpus.SessionObsDebt != 0 {
		t.Fatalf("clean corpus should be zero debt, got %d (kpis: %+v)", rep.Corpus.SessionObsDebt, rep.KPIs)
	}
	if !rep.OK || rep.Verdict != "OK" {
		t.Fatalf("clean corpus should verdict OK, got ok=%v verdict=%q", rep.OK, rep.Verdict)
	}
	if rep.Corpus.Grade != "A" {
		t.Fatalf("clean corpus should grade A, got %q (score %d)", rep.Corpus.Grade, rep.Corpus.Score)
	}
	if rep.Corpus.LinkedFrac != 1.0 {
		t.Fatalf("clean corpus should be fully linked, got %.3f", rep.Corpus.LinkedFrac)
	}
}

func TestScoreEmptyCorpusFailsCaptureAndLadder(t *testing.T) {
	rep := Score(nil, Pipeline{})
	// capture, structure, link, separable, corpus_committed, loop_consumes = 6 HARD rungs.
	if rep.Corpus.SessionObsDebt != 6 {
		t.Fatalf("empty corpus + empty pipeline should be 6 HARD debt, got %d", rep.Corpus.SessionObsDebt)
	}
	if got := kpi(rep, "corpus_nonempty"); got.Debt != 1 {
		t.Errorf("corpus_nonempty should be DEBT on an empty corpus, got %+v", got)
	}
	if rep.OK {
		t.Errorf("empty corpus must not be OK")
	}
	if rep.Verdict != "ACTION" {
		t.Errorf("empty corpus verdict should be ACTION, got %q", rep.Verdict)
	}
	// worst-first: the lowest unbuilt rung is capture.
	if got := worstHard(rep.KPIs); got.Name != "corpus_nonempty" {
		t.Errorf("worst-first rung should be corpus_nonempty, got %q", got.Name)
	}
}

func TestOutcomeLinkRateIsolated(t *testing.T) {
	// Structured records but all Unknown outcome: only the two link-rung KPIs fail.
	corpus := []Record{
		{SessionID: "a", AssistantTurns: 4, Outcome: OutcomeUnknown},
		{SessionID: "b", AssistantTurns: 3, Outcome: OutcomeUnknown},
	}
	rep := Score(corpus, fullPipeline())
	if rep.Corpus.SessionObsDebt != 2 {
		t.Fatalf("all-unknown corpus should fail exactly link + separable (2 debt), got %d (kpis %+v)",
			rep.Corpus.SessionObsDebt, failing(rep))
	}
	if got := kpi(rep, "outcome_link_rate"); got.Debt != 1 {
		t.Errorf("outcome_link_rate should be DEBT when nothing is linked, got %+v", got)
	}
	if got := kpi(rep, "value_waste_separable"); got.Debt != 1 {
		t.Errorf("value_waste_separable should be DEBT with no value/waste, got %+v", got)
	}
}

func TestValueWasteSeparableNeedsBothClasses(t *testing.T) {
	// All shipped: linked + structured pass, but there is no waste contrast.
	corpus := []Record{
		{SessionID: "a", AssistantTurns: 4, Outcome: OutcomeShipped, Signals: Signals{Commits: 1}},
		{SessionID: "b", AssistantTurns: 6, Outcome: OutcomeShipped, Signals: Signals{Commits: 1}},
	}
	rep := Score(corpus, fullPipeline())
	if got := kpi(rep, "value_waste_separable"); got.Debt != 1 {
		t.Fatalf("all-value corpus should fail value_waste_separable, got %+v", got)
	}
	if got := kpi(rep, "outcome_link_rate"); got.Debt != 0 {
		t.Errorf("all-shipped corpus is fully linked; outcome_link_rate should pass, got %+v", got)
	}
}

func TestPipelineLearnRungs(t *testing.T) {
	// A perfectly linked corpus still owes the LEARN rungs until the corpus is
	// committed and a loop consumes it. This is the gap increment 1 honestly reports.
	rep := Score(cleanCorpus(), Pipeline{})
	if got := kpi(rep, "corpus_committed"); got.Debt != 1 {
		t.Errorf("corpus_committed should be DEBT with an empty pipeline, got %+v", got)
	}
	if got := kpi(rep, "loop_consumes"); got.Debt != 1 {
		t.Errorf("loop_consumes should be DEBT with an empty pipeline, got %+v", got)
	}
	if got := kpi(rep, "registered_in_control_pane"); got.Hard {
		t.Errorf("registered_in_control_pane must be SOFT, never HARD")
	}
	if got := kpi(rep, "registered_in_control_pane"); got.Debt != 0 {
		t.Errorf("a SOFT rung must never contribute debt, got %+v", got)
	}
	if rep.Corpus.SessionObsDebt != 2 {
		t.Fatalf("clean corpus + empty pipeline should owe exactly the 2 learn rungs, got %d", rep.Corpus.SessionObsDebt)
	}
}

func TestClassifyOutcome(t *testing.T) {
	cases := []struct {
		committed, witnessed, stopped, mutated bool
		want                                   Outcome
	}{
		{true, true, false, true, OutcomeShipped},
		{true, false, false, true, OutcomeClaimed},
		{true, true, true, true, OutcomeShipped}, // a commit + witness wins over a stray stop
		{false, false, true, true, OutcomeStopped},
		{false, false, false, false, OutcomeNoOp},
		{false, false, false, true, OutcomeUnknown}, // mutated but no commit/stop -> unlinked
	}
	for i, c := range cases {
		if got := ClassifyOutcome(c.committed, c.witnessed, c.stopped, c.mutated); got != c.want {
			t.Errorf("case %d ClassifyOutcome(%v,%v,%v,%v)=%v, want %v",
				i, c.committed, c.witnessed, c.stopped, c.mutated, got, c.want)
		}
	}
}

func TestScoreIsDeterministic(t *testing.T) {
	corpus, pipe := cleanCorpus(), fullPipeline()
	a := Score(corpus, pipe)
	b := Score(corpus, pipe)
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("Score must be deterministic: two runs differ\n a=%+v\n b=%+v", a, b)
	}
	// And it must not depend on a clock or any hidden state: marshal stability.
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if !bytes.Equal(ja, jb) {
		t.Fatalf("Score JSON must be byte-identical across runs")
	}
}

func TestRenderSmoke(t *testing.T) {
	var buf bytes.Buffer
	Render(&buf, Score(cleanCorpus(), fullPipeline()))
	if buf.Len() == 0 {
		t.Fatal("Render produced no output")
	}
	if !bytes.Contains(buf.Bytes(), []byte("sessionobs_debt=0")) {
		t.Errorf("Render should report the debt headline, got:\n%s", buf.String())
	}
}

func TestOutcomeStringStable(t *testing.T) {
	for o, want := range map[Outcome]string{
		OutcomeUnknown: "unknown", OutcomeShipped: "shipped", OutcomeClaimed: "claimed",
		OutcomeStopped: "stopped", OutcomeNoOp: "noop", Outcome(99): "unknown",
	} {
		if got := o.String(); got != want {
			t.Errorf("Outcome(%d).String()=%q, want %q", o, got, want)
		}
	}
}

// --- helpers -------------------------------------------------------------

func kpi(rep Report, name string) KPI {
	for _, k := range rep.KPIs {
		if k.Name == name {
			return k
		}
	}
	return KPI{Name: "MISSING:" + name}
}

func failing(rep Report) []string {
	var out []string
	for _, k := range rep.KPIs {
		if k.Debt > 0 {
			out = append(out, k.Name)
		}
	}
	return out
}
