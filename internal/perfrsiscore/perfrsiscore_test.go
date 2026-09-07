package perfrsiscore

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestScoreLoopTurnScoresConfiguredInput(t *testing.T) {
	receipt := ScoreLoopTurn("testdata/complete.json")
	if receipt.Schema != LoopTurnSchema || receipt.Status != LoopTurnScored || receipt.Reason != "SCORE_COMPLETE" {
		t.Fatalf("receipt header=%+v", receipt)
	}
	if receipt.Snapshot == "" || receipt.LoopHealth == nil || receipt.PerformanceRSIDebt == nil || receipt.DominantBottleneck == "" {
		t.Fatalf("scored receipt is missing bounded score evidence: %+v", receipt)
	}
	assertSingleInvocationOutcome(t, receipt.InvocationOutcomes, OutcomeSuccess)
	rendered := FormatLoopTurnReceipt(receipt)
	for _, want := range []string{`"schema":"fak-performance-rsi-loop-turn/1"`, `"status":"scored"`, `"reason":"SCORE_COMPLETE"`, `"invocation_outcomes":{"success":1,"refusal":0,"error":0}`} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("rendered receipt missing %q: %s", want, rendered)
		}
	}
}

func TestScoreLoopTurnMakesUnavailableInputExplicitAndNonfatal(t *testing.T) {
	tests := []struct {
		input   string
		outcome InvocationOutcome
	}{
		{input: "", outcome: OutcomeRefusal},
		{input: filepath.Join(t.TempDir(), "missing.json"), outcome: OutcomeError},
	}
	for _, tc := range tests {
		input := tc.input
		receipt := ScoreLoopTurn(input)
		if receipt.Schema != LoopTurnSchema || receipt.Status != LoopTurnUnavailable || receipt.Reason != "SCORE_INPUT_UNAVAILABLE" {
			t.Fatalf("input %q receipt header=%+v", input, receipt)
		}
		if strings.TrimSpace(receipt.UnavailableDiagnostic) == "" {
			t.Fatalf("input %q silently lost the unavailable diagnostic: %+v", input, receipt)
		}
		if receipt.LoopHealth != nil || receipt.PerformanceRSIDebt != nil {
			t.Fatalf("input %q invented score data: %+v", input, receipt)
		}
		assertSingleInvocationOutcome(t, receipt.InvocationOutcomes, tc.outcome)
		assertFormattedInvocationOutcome(t, receipt, tc.outcome)
	}
}

func assertSingleInvocationOutcome(t *testing.T, counts OutcomeCounts, want InvocationOutcome) {
	t.Helper()
	if counts.Total() != 1 {
		t.Fatalf("invocation outcome total = %d, want 1: %+v", counts.Total(), counts)
	}
	wantCounts := OutcomeCounts{}
	wantCounts.observe(want)
	if counts != wantCounts {
		t.Fatalf("invocation outcomes = %+v, want %+v", counts, wantCounts)
	}
}

func assertFormattedInvocationOutcome(t *testing.T, receipt LoopTurnReceipt, want InvocationOutcome) {
	t.Helper()
	var formatted LoopTurnReceipt
	if err := json.Unmarshal([]byte(FormatLoopTurnReceipt(receipt)), &formatted); err != nil {
		t.Fatalf("formatted receipt is not queryable JSON: %v", err)
	}
	assertSingleInvocationOutcome(t, formatted.InvocationOutcomes, want)
}

func fixture(t *testing.T) Evidence {
	t.Helper()
	e, err := Load("testdata/complete.json")
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestAll16ExactlyOnceAndDeterministic(t *testing.T) {
	e := fixture(t)
	if len(e.Dimensions) != 16 { //boundarylint:ignore CHANGE_DETECTOR_TEST the scorecard schema exposes exactly 16 named dimensions and this test guards that schema
		t.Fatalf("got %d", len(e.Dimensions))
	}
	a := Score(e)
	b := Score(e)
	aj, _ := json.Marshal(a)
	bj, _ := json.Marshal(b)
	if !bytes.Equal(aj, bj) {
		t.Fatal("nondeterministic output")
	}
}

func TestIssue9768DogfoodGetsPoorNamedLoopHealthDebt(t *testing.T) {
	e, err := Load("../../docs/_witnesses/issue-9768-performance-rsi-dogfood/input.json")
	if err != nil {
		t.Fatal(err)
	}
	r := Score(e)
	if r.LoopHealth == nil || r.DebtSummary == nil {
		t.Fatalf("missing loop-health summaries: %+v", r)
	}
	if r.LoopHealth.Score != 62.6 || r.LoopHealth.Grade != "D" || r.LoopHealth.Clean {
		t.Fatalf("loop health=%+v, want D 62.6 and non-clean", r.LoopHealth)
	}
	if r.DebtSummary.DimensionsMeasured != 15 || r.DebtSummary.DimensionsTotal != 16 ||
		r.DebtSummary.Behind != 8 || r.DebtSummary.Unknown != 1 ||
		r.DebtSummary.Total != 9 || r.DebtSummary.PerformanceRSIDebt != 9 {
		t.Fatalf("debt summary=%+v, want 15/16 measured, 8 BEHIND, 1 UNKNOWN, debt 9", r.DebtSummary)
	}
	wantEvidence := []string{
		"cycle_time", "improvement_yield", "evaluation_latency", "experiment_throughput",
		"discovery_freshness", "production_transfer", "hardware_utilization",
		"automation_coverage", "compounding_rate",
	}
	if len(r.DebtSummary.Evidence) != len(wantEvidence) {
		t.Fatalf("named evidence=%d, want %d: %+v", len(r.DebtSummary.Evidence), len(wantEvidence), r.DebtSummary.Evidence)
	}
	for i, want := range wantEvidence {
		got := r.DebtSummary.Evidence[i]
		if got.Dimension != want || got.Source == "" || got.NextAction == "" {
			t.Errorf("evidence[%d]=%+v, want named dimension %q with source and action", i, got, want)
		}
	}
	for _, index := range []int{0, 3} {
		got := r.DebtSummary.Evidence[index]
		if got.NormalizedRatio == nil || *got.NormalizedRatio != 0.01 {
			t.Errorf("%s ratio=%v, want 0.01", got.Dimension, got.NormalizedRatio)
		}
	}
	unknown := r.DebtSummary.Evidence[6]
	if unknown.Dimension != "hardware_utilization" || unknown.Status != "UNKNOWN" ||
		unknown.NormalizedRatio != nil || unknown.Source != "fixture/hardware" {
		t.Errorf("UNKNOWN evidence=%+v", unknown)
	}
}

func TestLoopHealthCapsOverTargetCreditAndSeparatesGradeFromClean(t *testing.T) {
	e := fixture(t)
	for i := range e.Dimensions {
		target := *e.Dimensions[i].Target
		e.Dimensions[i].Current = &target
	}
	r := Score(e)
	if r.LoopHealth == nil || r.DebtSummary == nil ||
		r.LoopHealth.Score != 100 || r.LoopHealth.Grade != "A" || !r.LoopHealth.Clean ||
		r.DebtSummary.Total != 0 || len(r.DebtSummary.Evidence) != 0 {
		t.Fatalf("clean health=%+v debt=%+v", r.LoopHealth, r.DebtSummary)
	}

	overTarget := *e.Dimensions[1].Target * 1000
	e.Dimensions[1].Current = &overTarget
	e.Dimensions[0].Current = nil
	r = Score(e)
	if r.LoopHealth.Score != 93.8 || r.LoopHealth.Grade != "A" || r.LoopHealth.Clean {
		t.Fatalf("capped/unknown health=%+v, want A 93.8 but non-clean", r.LoopHealth)
	}
	if r.DebtSummary.Total != 1 || r.DebtSummary.Unknown != 1 ||
		len(r.DebtSummary.Evidence) != 1 || r.DebtSummary.Evidence[0].Dimension != e.Dimensions[0].ID {
		t.Fatalf("capped/unknown debt=%+v", r.DebtSummary)
	}
}

func TestLegacyReportDecodesAndComparesWithoutHealthFields(t *testing.T) {
	f, err := os.Open("../../docs/_witnesses/issue-9752-performance-rsi-snapshot.json")
	if err != nil {
		t.Fatal(err)
	}
	prior, err := DecodeReport(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	if prior.LoopHealth != nil || prior.DebtSummary != nil {
		t.Fatalf("legacy report synthesized serialized health fields: %+v %+v", prior.LoopHealth, prior.DebtSummary)
	}
	current := Score(fixture(t))
	if err := Compare(&current, prior); err != nil {
		t.Fatalf("legacy comparison compatibility: %v", err)
	}
	if current.Comparison == nil || current.Comparison.PriorSnapshot != prior.Snapshot {
		t.Fatalf("comparison=%+v", current.Comparison)
	}
}

func committedCompositionInputs(t *testing.T) []ComposeInput {
	t.Helper()
	names := []string{
		"issue-9780-performance-rsi-cycle.json",
		"issue-9781-performance-rsi-improvement.json",
		"issue-9782-performance-rsi-provenance.json",
		"issue-9783-performance-rsi-learning.json",
	}
	inputs := make([]ComposeInput, 0, len(names))
	for _, name := range names {
		path := filepath.Join("..", "..", "docs", "_witnesses", name)
		e, err := Load(path)
		if err != nil {
			t.Fatal(err)
		}
		inputs = append(inputs, ComposeInput{Source: name, Evidence: e})
	}
	return inputs
}

func TestComposeV1CommittedReceiptsIsScorecardReadyAndDeterministic(t *testing.T) {
	inputs := committedCompositionInputs(t)
	got, err := ComposeV1("issue-9823-composed", inputs)
	if err != nil {
		t.Fatal(err)
	}
	reversed := append([]ComposeInput(nil), inputs...)
	for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
		reversed[i], reversed[j] = reversed[j], reversed[i]
	}
	again, err := ComposeV1("issue-9823-composed", reversed)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := json.Marshal(got)
	b, _ := json.Marshal(again)
	if !bytes.Equal(a, b) {
		t.Fatal("composition changed with receipt argument order")
	}

	wantContracts := map[string]struct {
		unit   string
		target float64
	}{
		"cycle_time":          {unit: "hours", target: 0.02},
		"improvement_yield":   {unit: "percent", target: 100},
		"production_transfer": {unit: "hours", target: 100},
		"compounding_rate":    {unit: "percent", target: 100},
	}
	for _, d := range got.Dimensions {
		want, ok := wantContracts[d.ID]
		if !ok {
			continue
		}
		if d.Unit != want.unit || d.Target == nil || *d.Target != want.target {
			t.Errorf("%s contract=%+v, want unit=%q target=%g", d.ID, d, want.unit, want.target)
		}
		delete(wantContracts, d.ID)
	}
	if len(wantContracts) != 0 {
		t.Fatalf("missing owner-selected contracts: %v", wantContracts)
	}

	report := Score(got)
	if report.UnknownDebt != 1 {
		t.Fatalf("UNKNOWN debt=%d, want 1", report.UnknownDebt)
	}
	var unknown []string
	for _, d := range report.Dimensions {
		if d.Status == "UNKNOWN" {
			unknown = append(unknown, d.ID)
		}
	}
	if strings.Join(unknown, ",") != "hardware_utilization" {
		t.Fatalf("UNKNOWN dimensions=%v, want only hardware_utilization", unknown)
	}

	encoded, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(bytes.NewReader(encoded)); err != nil {
		t.Fatalf("composed evidence was not accepted by scorecard decoder: %v", err)
	}
}

func TestComposeV1RejectsDuplicateSectionOwnersDeterministically(t *testing.T) {
	inputs := committedCompositionInputs(t)
	cycle := inputs[0].Evidence
	_, err := ComposeV1("duplicate-cycle", []ComposeInput{
		{Source: "z-cycle.json", Evidence: cycle},
		{Source: "a-cycle.json", Evidence: cycle},
	})
	want := `fak-performance-rsi-composition/1: section "cycle" has multiple owners "a-cycle.json" and "z-cycle.json"; fix: provide exactly one receipt for that section`
	if err == nil || err.Error() != want {
		t.Fatalf("error=%v, want %q", err, want)
	}
}

func TestComposeV1RejectsIncompatibleUnownedDimensionContract(t *testing.T) {
	inputs := committedCompositionInputs(t)
	for i := range inputs[1].Evidence.Dimensions {
		if inputs[1].Evidence.Dimensions[i].ID == "hardware_utilization" {
			target := 90.0
			inputs[1].Evidence.Dimensions[i].Target = &target
		}
	}
	_, err := ComposeV1("hardware-conflict", inputs)
	if err == nil {
		t.Fatal("accepted incompatible hardware contract without a hardware receipt")
	}
	for _, want := range []string{
		`fak-performance-rsi-composition/1`,
		`dimension "hardware_utilization"`,
		`without owning section "hardware"`,
		`issue-9780-performance-rsi-cycle.json`,
		`issue-9781-performance-rsi-improvement.json`,
		`add one "hardware" receipt or align the contracts`,
	} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error=%q, want substring %q", err, want)
		}
	}
}

func learningEvidence(t *testing.T) Evidence {
	t.Helper()
	e := fixture(t)
	e.Learning = &Learning{Schema: LearningSchema, Rows: []LearningRow{
		{CycleID: "c1", HypothesisID: "h1", RecurrenceKey: "parser", PredictedImprovementPercent: 20, ConfidencePercent: 50, ObservedImprovementPercent: 10, LearningID: "l1", LearningRecorded: true, CycleTimeHours: 10, Engine: "fak-native", Artifact: "artifact-c1"},
		{CycleID: "c2", HypothesisID: "h2", RecurrenceKey: "parser", PredictedImprovementPercent: 12, ConfidencePercent: 30, ObservedImprovementPercent: 0, LearningReused: true, PriorLearningID: "l1", CycleTimeHours: 8, Engine: "fak-native", Artifact: "artifact-c2"},
		{CycleID: "c3", HypothesisID: "h3", RecurrenceKey: "parser", PredictedImprovementPercent: 8, ConfidencePercent: 20, ObservedImprovementPercent: 0, LearningReused: true, PriorLearningID: "l1", RepeatedFailure: true, CycleTimeHours: 6, Engine: "fak-native", Artifact: "artifact-c3"},
	}}
	for i := range e.Dimensions {
		switch e.Dimensions[i].ID {
		case "hypothesis_calibration", "learning_retention", "compounding_rate":
			e.Dimensions[i].Direction = Higher
			e.Dimensions[i].Unit = "percent"
		}
	}
	return e
}

func TestPerformanceRSILearningAcceptance(t *testing.T) {
	e := learningEvidence(t)
	before := make(map[string]*float64)
	for _, d := range e.Dimensions {
		before[d.ID] = d.Current
	}
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]float64{"hypothesis_calibration": 89.8, "learning_retention": 100, "compounding_rate": 40}
	for _, d := range got.Dimensions {
		value, derived := want[d.ID]
		if derived {
			if d.Current == nil || math.Abs(*d.Current-value) > 1e-9 ||
				d.Source != "learning:"+LearningSchema ||
				d.EvidenceKind != "performance_rsi_learning_receipt" || d.Engine != "fak-native" {
				t.Errorf("%s=%+v want %.1f and strict receipt provenance", d.ID, d, value)
			}
		} else if d.Current != before[d.ID] && (d.Current == nil || before[d.ID] == nil || *d.Current != *before[d.ID]) {
			t.Errorf("unrelated dimension %s mutated", d.ID)
		}
	}
}

func TestPerformanceRSILearningCompoundingUsesChronologicalKeyAndLatestReuse(t *testing.T) {
	e := learningEvidence(t)
	e.Learning.Rows = []LearningRow{
		{CycleID: "first-1", HypothesisID: "first-h1", RecurrenceKey: "first", PredictedImprovementPercent: 100, ConfidencePercent: 20, ObservedImprovementPercent: 100, LearningID: "first-learning", LearningRecorded: true, CycleTimeHours: 20, Engine: "fak-native", Artifact: "first-original"},
		{CycleID: "second-1", HypothesisID: "second-h1", RecurrenceKey: "second", PredictedImprovementPercent: 100, ConfidencePercent: 20, ObservedImprovementPercent: 100, LearningID: "second-learning", LearningRecorded: true, CycleTimeHours: 10, Engine: "fak-native", Artifact: "second-original"},
		{CycleID: "first-2", HypothesisID: "first-h2", RecurrenceKey: "first", PredictedImprovementPercent: 100, ConfidencePercent: 20, ObservedImprovementPercent: 100, LearningReused: true, PriorLearningID: "first-learning", CycleTimeHours: 15, Engine: "fak-native", Artifact: "first-reuse-1"},
		{CycleID: "second-2", HypothesisID: "second-h2", RecurrenceKey: "second", PredictedImprovementPercent: 100, ConfidencePercent: 20, ObservedImprovementPercent: 100, LearningReused: true, PriorLearningID: "second-learning", CycleTimeHours: 5, Engine: "fak-native", Artifact: "second-reuse"},
		{CycleID: "first-3", HypothesisID: "first-h3", RecurrenceKey: "first", PredictedImprovementPercent: 100, ConfidencePercent: 20, ObservedImprovementPercent: 100, LearningReused: true, PriorLearningID: "first-learning", CycleTimeHours: 12, Engine: "fak-native", Artifact: "first-reuse-2"},
	}

	got, err := decodeCycleEvidence(t, e)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range got.Dimensions {
		if d.ID == "compounding_rate" && (d.Current == nil || *d.Current != 40) {
			t.Fatalf("compounding_rate=%v, want 40 from first original (20h) and its latest reuse (12h)", d.Current)
		}
	}
}

func TestPerformanceRSILearningRefusals(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Evidence)
	}{
		{"insufficient history", func(e *Evidence) { e.Learning.Rows = e.Learning.Rows[:1] }},
		{"no recurrence", func(e *Evidence) {
			e.Learning.Rows[1].RecurrenceKey, e.Learning.Rows[2].RecurrenceKey = "other", "third"
		}},
		{"no valid reuse", func(e *Evidence) {
			e.Learning.Rows[1].LearningReused = false
			e.Learning.Rows[1].PriorLearningID = ""
			e.Learning.Rows[2].LearningReused = false
			e.Learning.Rows[2].PriorLearningID = ""
		}},
		{"false compounding", func(e *Evidence) { e.Learning.Rows[2].CycleTimeHours = 10 }},
		{"negative compounding", func(e *Evidence) { e.Learning.Rows[2].CycleTimeHours = 11 }},
		{"duplicate cycle", func(e *Evidence) { e.Learning.Rows[1].CycleID = "c1" }},
		{"range", func(e *Evidence) { e.Learning.Rows[1].ConfidencePercent = 101 }},
		{"engine", func(e *Evidence) { e.Learning.Rows[1].Engine = "fak-native/qwen" }},
		{"forward reference", func(e *Evidence) {
			e.Learning.Rows[1].PriorLearningID = "later"
			e.Learning.Rows[2].LearningID = "later"
			e.Learning.Rows[2].LearningRecorded = true
		}},
		{"false repeated failure", func(e *Evidence) { e.Learning.Rows[2].RepeatedFailure = false }},
		{"canonical dimension", func(e *Evidence) {
			for i := range e.Dimensions {
				if e.Dimensions[i].ID == "learning_retention" {
					e.Dimensions[i].Unit = "ratio"
				}
			}
		}},
		{"later key cannot rescue false compounding for chronological key", func(e *Evidence) {
			e.Learning.Rows = append([]LearningRow{
				{CycleID: "earliest", HypothesisID: "earliest-h", RecurrenceKey: "earliest", PredictedImprovementPercent: 10, ConfidencePercent: 10, ObservedImprovementPercent: 10, LearningID: "earliest-learning", LearningRecorded: true, CycleTimeHours: 5, Engine: "fak-native", Artifact: "earliest-original"},
				{CycleID: "later", HypothesisID: "later-h", RecurrenceKey: "later", PredictedImprovementPercent: 10, ConfidencePercent: 10, ObservedImprovementPercent: 10, LearningID: "later-learning", LearningRecorded: true, CycleTimeHours: 10, Engine: "fak-native", Artifact: "later-original"},
				{CycleID: "earliest-reuse", HypothesisID: "earliest-reuse-h", RecurrenceKey: "earliest", PredictedImprovementPercent: 10, ConfidencePercent: 10, ObservedImprovementPercent: 10, LearningReused: true, PriorLearningID: "earliest-learning", CycleTimeHours: 5, Engine: "fak-native", Artifact: "earliest-reuse"},
				{CycleID: "later-reuse", HypothesisID: "later-reuse-h", RecurrenceKey: "later", PredictedImprovementPercent: 10, ConfidencePercent: 10, ObservedImprovementPercent: 10, LearningReused: true, PriorLearningID: "later-learning", CycleTimeHours: 5, Engine: "fak-native", Artifact: "later-reuse"},
			}, e.Learning.Rows...)
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := learningEvidence(t)
			tc.edit(&e)
			before, _ := json.Marshal(e)
			if err := validate(&e); err == nil {
				t.Fatal("accepted invalid learning receipt")
			}
			after, _ := json.Marshal(e)
			if !bytes.Equal(before, after) {
				t.Fatal("validation mutated evidence before refusing")
			}
		})
	}
}
func TestUnknownDebtAndDominantBottleneck(t *testing.T) {
	e := fixture(t)
	e.Dimensions[0].Current = nil
	e.Dimensions[1].Current = nil
	r := Score(e)
	if r.UnknownDebt != 2 || r.DominantBottleneck != "evaluation_latency" {
		t.Fatalf("debt=%d bottleneck=%s", r.UnknownDebt, r.DominantBottleneck)
	}
}
func TestBottleneckSelectionUsesLowestRatio(t *testing.T) {
	r := Score(fixture(t))
	if r.DominantBottleneck != "evaluation_latency" {
		t.Fatal(r.DominantBottleneck)
	}
}
func TestUnsaturatedRatioAnd100x(t *testing.T) {
	e := fixture(t)
	v := 250.0
	e.Dimensions[1].Current = &v
	r := Score(e)
	if *r.Dimensions[1].NormalizedRatio != 2.5 || r.TargetMultiplier != 100 {
		t.Fatalf("%+v", r)
	}
}
func TestInvalidValues(t *testing.T) {
	for _, v := range []float64{-1, math.NaN(), math.Inf(1)} {
		e := fixture(t)
		e.Dimensions[0].Current = &v
		b, _ := json.Marshal(e)
		if _, err := Decode(bytes.NewReader(b)); err == nil {
			t.Fatalf("accepted %v", v)
		}
	}
}
func TestRenderersAndComparison(t *testing.T) {
	e := fixture(t)
	r := Score(e)
	prior := r
	prior.Snapshot = "prior"
	old := .1
	prior.Dimensions[0].NormalizedRatio = &old
	if err := Compare(&r, prior); err != nil {
		t.Fatal(err)
	}
	human := RenderHuman(r)
	markdown := RenderMarkdown(r)
	for name, rendered := range map[string]string{"human": human, "Markdown": markdown} {
		for _, want := range []string{"invocation outcomes", "success=1", "refusal=0", "error=0"} {
			if !strings.Contains(rendered, want) {
				t.Fatalf("%s renderer missing %q: %s", name, want, rendered)
			}
		}
	}
	b, err := MarshalJSON(r)
	if err != nil || !bytes.Contains(b, []byte(`"comparison"`)) || !bytes.Contains(b, []byte(`"invocation_outcomes"`)) {
		t.Fatalf("json: %v", err)
	}
	assertSingleInvocationOutcome(t, r.InvocationOutcomes, OutcomeSuccess)
}
func TestNativeProvenanceAndNoLlamaFallback(t *testing.T) {
	e := fixture(t)
	e.Dimensions[2].EvidenceKind = "native_benchmark"
	e.Dimensions[2].Engine = ""
	b, _ := json.Marshal(e)
	if _, err := Decode(bytes.NewReader(b)); err == nil {
		t.Fatal("accepted unnamed native engine")
	}
	e.Dimensions[2].Engine = "llama.cpp"
	b, _ = json.Marshal(e)
	if _, err := Decode(bytes.NewReader(b)); err == nil {
		t.Fatal("accepted llama fallback")
	}
	e.Dimensions[2].Engine = "fak-native/qwen3.8"
	b, _ = json.Marshal(e)
	if _, err := Decode(bytes.NewReader(b)); err != nil {
		t.Fatal(err)
	}
}
func TestFixtureIsVersioned(t *testing.T) {
	b, err := os.ReadFile("testdata/complete.json")
	if err != nil || !bytes.Contains(b, []byte(EvidenceSchema)) {
		t.Fatal(err)
	}
}

func cycleEvidence(t *testing.T) Evidence {
	t.Helper()
	e := fixture(t)
	e.Cycle = &Cycle{
		Schema:                CycleSchema,
		Engine:                "fak-native/qwen3.8",
		IdeaAt:                "2026-08-28T10:00:00Z",
		QueueAt:               "2026-08-28T10:10:00Z",
		ExecutionAt:           "2026-08-28T10:30:00Z",
		EvaluationAt:          "2026-08-28T11:00:00Z",
		LandingAt:             "2026-08-28T11:30:00Z",
		LearningAt:            "2026-08-28T12:00:00Z",
		OperatorActiveSeconds: 1800,
	}
	for i := range e.Dimensions {
		switch e.Dimensions[i].ID {
		case "cycle_time", "evaluation_latency":
			e.Dimensions[i].Unit = "hours"
			e.Dimensions[i].Current = nil
		case "experiment_throughput":
			e.Dimensions[i].Unit = "experiments/day"
			e.Dimensions[i].Current = nil
		case "automation_coverage":
			e.Dimensions[i].Unit = "percent"
			e.Dimensions[i].Current = nil
		}
	}
	return e
}

func decodeCycleEvidence(t *testing.T, e Evidence) (Evidence, error) {
	t.Helper()
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	return Decode(bytes.NewReader(b))
}

func TestCycleDerivesFourDimensions(t *testing.T) {
	e, err := decodeCycleEvidence(t, cycleEvidence(t))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]float64{
		"cycle_time":            2,
		"evaluation_latency":    .5,
		"experiment_throughput": 12,
		"automation_coverage":   75,
	}
	r := Score(e)
	for _, d := range r.Dimensions {
		expected, ok := want[d.ID]
		if !ok {
			continue
		}
		if d.Status == "UNKNOWN" || d.Current == nil || *d.Current != expected {
			t.Errorf("%s: status=%s current=%v, want %v", d.ID, d.Status, d.Current, expected)
		}
		if d.Source != "cycle:"+CycleSchema {
			t.Errorf("%s source=%q", d.ID, d.Source)
		}
	}
}

func TestCycleRejectsInvalidRows(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Cycle)
	}{
		{"unsupported schema", func(c *Cycle) { c.Schema = "fak-performance-rsi-cycle/2" }},
		{"missing stage", func(c *Cycle) { c.QueueAt = "" }},
		{"malformed timestamp", func(c *Cycle) { c.ExecutionAt = "yesterday" }},
		{"reordered stages", func(c *Cycle) { c.LandingAt = "2026-08-28T10:45:00Z" }},
		{"negative operator duration", func(c *Cycle) { c.OperatorActiveSeconds = -1 }},
		{"non-native engine", func(c *Cycle) { c.Engine = "llama.cpp" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := cycleEvidence(t)
			tc.mutate(e.Cycle)
			if _, err := decodeCycleEvidence(t, e); err == nil {
				t.Fatal("accepted invalid cycle")
			}
		})
	}
}

func TestCycleRejectsMissingOperatorDuration(t *testing.T) {
	e := cycleEvidence(t)
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	b = bytes.Replace(b, []byte(`,"operator_active_seconds":1800`), nil, 1)
	if _, err := Decode(bytes.NewReader(b)); err == nil {
		t.Fatal("accepted missing operator_active_seconds")
	}
}

func improvementEvidence(t *testing.T) Evidence {
	t.Helper()
	e, err := Load("../../docs/_witnesses/issue-9781-performance-rsi-improvement.json")
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestImprovementDerivesReceiptDimensionsOnly(t *testing.T) {
	e := improvementEvidence(t)
	baseline, err := Load("testdata/complete.json")
	if err != nil {
		t.Fatal(err)
	}
	prior := make(map[string]Dimension, len(baseline.Dimensions))
	for _, d := range baseline.Dimensions {
		prior[d.ID] = d
	}
	want := map[string]float64{
		"improvement_yield": 20, "receipt_coverage": 100,
		"quality_gate_coverage": 100, "attribution_quality": 100,
	}
	for _, d := range e.Dimensions {
		value, derived := want[d.ID]
		if derived {
			if d.Current == nil || *d.Current != value ||
				d.Source != "improvement:"+ImprovementSchema ||
				d.EvidenceKind != "improvement_receipt" || d.Engine != "fak-native/qwen3.8" {
				t.Errorf("%s not deterministically derived: %+v", d.ID, d)
			}
			continue
		}
		p := prior[d.ID]
		if (d.Current == nil) != (p.Current == nil) || (d.Current != nil && *d.Current != *p.Current) ||
			d.Source != p.Source || d.EvidenceKind != p.EvidenceKind || d.Engine != p.Engine {
			t.Errorf("unrelated dimension %s changed: got %+v want %+v", d.ID, d, p)
		}
	}
}

func TestImprovementRejectsInvalidReceipts(t *testing.T) {
	boolp := func(v bool) *bool { return &v }
	tests := []struct {
		name string
		edit func(*Improvement)
		want string
	}{
		{"schema", func(r *Improvement) { r.Schema = "fak-performance-rsi-improvement/2" }, "schema"},
		{"units", func(r *Improvement) { r.Baseline.Unit = "seconds" }, "milliseconds"},
		{"quality parity", func(r *Improvement) { r.Quality.Parity = boolp(false) }, "quality"},
		{"missing quality parity", func(r *Improvement) { r.Quality.Parity = nil }, "quality"},
		{"envelope", func(r *Improvement) { r.CandidateEnvelope.BatchSize = 2 }, "matched"},
		{"overhead excluded", func(r *Improvement) { r.NetTrueGain.IncludesOverhead = boolp(false) }, "include"},
		{"causal binding", func(r *Improvement) { r.Causal.IsolatesChange = nil }, "causal"},
		{"engine", func(r *Improvement) { r.Engine = "llama.cpp" }, "fak-native"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := improvementEvidence(t)
			tc.edit(e.Improvement)
			b, err := json.Marshal(e)
			if err != nil {
				t.Fatal(err)
			}
			_, err = Decode(bytes.NewReader(b))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestImprovementReceiptStrictJSON(t *testing.T) {
	e := improvementEvidence(t)
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatal(err)
	}
	raw["improvement"].(map[string]any)["unexpected"] = true
	b, _ = json.Marshal(raw)
	if _, err := Decode(bytes.NewReader(b)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected strict JSON refusal, got %v", err)
	}
}

func provenanceEvidence(t *testing.T) Evidence {
	t.Helper()
	e, err := Load("../../docs/_witnesses/issue-9782-performance-rsi-provenance.json")
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestProvenanceAcceptanceAndFormula(t *testing.T) {
	e := provenanceEvidence(t)
	want := map[string]float64{
		"discovery_freshness": 16.051666666666666,
		"adaptation_speed":    0.8580555555555556,
		"reuse_ratio":         100,
		"production_transfer": 82.59222222222222,
	}
	for _, d := range e.Dimensions {
		value, ok := want[d.ID]
		if !ok {
			continue
		}
		if d.Current == nil || math.Abs(*d.Current-value) > 1e-9 {
			t.Errorf("%s current=%v want %.12g", d.ID, d.Current, value)
		}
		if d.Source != "provenance:"+ProvenanceSchema || d.EvidenceKind != "research_transfer_receipt" || d.Engine != "fak-native" {
			t.Errorf("%s provenance metadata=%+v", d.ID, d)
		}
	}
}

func TestProvenancePreservesUnrelatedDimensions(t *testing.T) {
	baseline, err := Load("testdata/complete.json")
	if err != nil {
		t.Fatal(err)
	}
	got := provenanceEvidence(t)
	owned := map[string]bool{"discovery_freshness": true, "adaptation_speed": true, "reuse_ratio": true, "production_transfer": true}
	prior := make(map[string]Dimension, len(baseline.Dimensions))
	for _, d := range baseline.Dimensions {
		prior[d.ID] = d
	}
	for _, d := range got.Dimensions {
		if owned[d.ID] {
			continue
		}
		p := prior[d.ID]
		if (d.Current == nil) != (p.Current == nil) || (d.Current != nil && *d.Current != *p.Current) ||
			d.Source != p.Source || d.EvidenceKind != p.EvidenceKind || d.Engine != p.Engine || d.Unit != p.Unit {
			t.Errorf("unrelated dimension %s changed: got %+v want %+v", d.ID, d, p)
		}
	}
}

func TestProvenanceRefusesInvalidReceiptsWithoutMutation(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Provenance)
	}{
		{"schema", func(r *Provenance) { r.Schema = "fak-performance-rsi-provenance/2" }},
		{"repository", func(r *Provenance) { r.Source.Repository = "" }},
		{"revision", func(r *Provenance) { r.Source.Revision = "main" }},
		{"timeline", func(r *Provenance) { r.DiscoveryAt = "2026-08-24T00:00:00Z" }},
		{"explicit start", func(r *Provenance) { r.AdaptationStartExplicit = false }},
		{"experiment", func(r *Provenance) { r.Experiment.Linked = false }},
		{"classification", func(r *Provenance) { r.Reuse.Classification = "invented_here" }},
		{"negative reused", func(r *Provenance) { r.Reuse.ReusedMechanisms = -1 }},
		{"empty total", func(r *Provenance) { r.Reuse.ReusedMechanisms = 0 }},
		{"commit", func(r *Provenance) { r.Production.CommitSHA = "5e0db65c5" }},
		{"module prefix", func(r *Provenance) { r.Production.ModuleAtRev = "internal/qwen38quantrun@r17+gdeadbee" }},
		{"engine", func(r *Provenance) { r.Production.Engine = "llama.cpp" }},
		{"unit", func(r *Provenance) { r.Unit = "days" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := provenanceEvidence(t)
			for i := range e.Dimensions {
				if e.Dimensions[i].ID == "discovery_freshness" {
					v := 777.0
					e.Dimensions[i].Current = &v
					e.Dimensions[i].Source = "sentinel"
				}
			}
			tc.edit(e.Provenance)
			if err := applyProvenance(&e); err == nil {
				t.Fatal("expected refusal")
			}
			for _, d := range e.Dimensions {
				if d.ID == "discovery_freshness" && (d.Current == nil || *d.Current != 777 || d.Source != "sentinel") {
					t.Fatalf("invalid receipt partially mutated dimension: %+v", d)
				}
			}
		})
	}
}

func TestProvenanceStrictJSON(t *testing.T) {
	b, err := os.ReadFile("../../docs/_witnesses/issue-9782-performance-rsi-provenance.json")
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatal(err)
	}
	raw["provenance"].(map[string]any)["unexpected"] = true
	b, _ = json.Marshal(raw)
	if _, err := Decode(bytes.NewReader(b)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected strict JSON refusal, got %v", err)
	}
}

func hardwareEvidence(t *testing.T) Evidence {
	t.Helper()
	e := fixture(t)
	e.Hardware = &Hardware{
		Schema: HardwareSchema,
		Runs: []HardwareRun{
			{
				EnqueuedAt:           "2026-08-28T10:00:00Z",
				StartedAt:            "2026-08-28T10:10:00Z",
				EndedAt:              "2026-08-28T11:10:00Z",
				RequestedDeviceClass: "cuda-l4",
				ActiveUtilization:    50,
				UtilizationUnit:      "percent",
				WorkloadID:           "workload-a",
				Engine:               "fak-native",
			},
			{
				EnqueuedAt:           "2026-08-28T12:00:00Z",
				StartedAt:            "2026-08-28T12:20:00Z",
				EndedAt:              "2026-08-28T15:20:00Z",
				RequestedDeviceClass: "cuda-h100",
				ActiveUtilization:    90,
				UtilizationUnit:      "percent",
				WorkloadID:           "workload-b",
				Engine:               "fak-native",
			},
		},
	}
	for i := range e.Dimensions {
		if e.Dimensions[i].ID == "hardware_utilization" {
			e.Dimensions[i].Direction = Higher
			e.Dimensions[i].Unit = "percent"
		}
	}
	return e
}

func TestHardwareAcceptanceUsesOnlyDurationWeightedMeasuredUtilization(t *testing.T) {
	e := hardwareEvidence(t)
	before := make(map[string]Dimension, len(e.Dimensions))
	for _, d := range e.Dimensions {
		before[d.ID] = d
	}
	got, err := decodeCycleEvidence(t, e)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range got.Dimensions {
		if d.ID == "hardware_utilization" {
			if d.Current == nil || *d.Current != 80 {
				t.Fatalf("hardware_utilization=%v, want duration-weighted 80", d.Current)
			}
			if d.Source != "hardware:"+HardwareSchema+";queue_delay_seconds_total=1800;queue_delay_seconds_mean=900" {
				t.Errorf("source=%q", d.Source)
			}
			if d.EvidenceKind != "hardware_utilization_receipt" || d.Engine != "fak-native" {
				t.Errorf("hardware evidence metadata=%+v", d)
			}
			continue
		}
		want := before[d.ID]
		if (d.Current == nil) != (want.Current == nil) ||
			(d.Current != nil && *d.Current != *want.Current) ||
			d.Source != want.Source || d.EvidenceKind != want.EvidenceKind ||
			d.Engine != want.Engine || d.Direction != want.Direction ||
			d.Unit != want.Unit || d.NextAction != want.NextAction ||
			(d.Target == nil) != (want.Target == nil) ||
			(d.Target != nil && *d.Target != *want.Target) {
			t.Errorf("unrelated dimension %s changed: got %+v want %+v", d.ID, d, want)
		}
	}
}

func TestHardwareRejectsInvalidRunsBeforeMutation(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Evidence)
	}{
		{"schema", func(e *Evidence) { e.Hardware.Schema = "fak-performance-rsi-hardware/2" }},
		{"empty runs", func(e *Evidence) { e.Hardware.Runs = nil }},
		{"missing enqueued", func(e *Evidence) { e.Hardware.Runs[0].EnqueuedAt = "" }},
		{"missing started", func(e *Evidence) { e.Hardware.Runs[0].StartedAt = "" }},
		{"missing ended", func(e *Evidence) { e.Hardware.Runs[0].EndedAt = "" }},
		{"invalid timestamp", func(e *Evidence) { e.Hardware.Runs[0].StartedAt = "now" }},
		{"non UTC", func(e *Evidence) { e.Hardware.Runs[0].StartedAt = "2026-08-28T10:10:00-07:00" }},
		{"enqueued after started", func(e *Evidence) { e.Hardware.Runs[0].EnqueuedAt = "2026-08-28T10:11:00Z" }},
		{"ended equals started", func(e *Evidence) { e.Hardware.Runs[0].EndedAt = e.Hardware.Runs[0].StartedAt }},
		{"unsupported unit", func(e *Evidence) { e.Hardware.Runs[0].UtilizationUnit = "%" }},
		{"empty device", func(e *Evidence) { e.Hardware.Runs[0].RequestedDeviceClass = " " }},
		{"empty workload", func(e *Evidence) { e.Hardware.Runs[0].WorkloadID = " " }},
		{"wrong engine", func(e *Evidence) { e.Hardware.Runs[0].Engine = "llama.cpp" }},
		{"nonfinite utilization", func(e *Evidence) { e.Hardware.Runs[0].ActiveUtilization = math.NaN() }},
		{"negative utilization", func(e *Evidence) { e.Hardware.Runs[0].ActiveUtilization = -1 }},
		{"overrange utilization", func(e *Evidence) { e.Hardware.Runs[0].ActiveUtilization = 100.01 }},
		{"typed local no GPU", func(e *Evidence) {
			e.Hardware.Runs[0].TerminalEvidence = &HardwareTerminalEvidence{Type: "local-no-gpu"}
		}},
		{"unsupported terminal evidence", func(e *Evidence) {
			e.Hardware.Runs[0].TerminalEvidence = &HardwareTerminalEvidence{Type: "gpu-driver-missing"}
		}},
		{"canonical dimension", func(e *Evidence) {
			for i := range e.Dimensions {
				if e.Dimensions[i].ID == "hardware_utilization" {
					e.Dimensions[i].Unit = "ratio"
				}
			}
		}},
		{"invalid second run", func(e *Evidence) { e.Hardware.Runs[1].ActiveUtilization = 101 }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := hardwareEvidence(t)
			for i := range e.Dimensions {
				if e.Dimensions[i].ID == "hardware_utilization" {
					sentinel := 17.0
					e.Dimensions[i].Current = &sentinel
					e.Dimensions[i].Source = "sentinel"
					e.Dimensions[i].EvidenceKind = "sentinel"
					e.Dimensions[i].Engine = "sentinel"
				}
			}
			tc.edit(&e)
			if err := applyHardware(&e); err == nil {
				t.Fatal("accepted invalid hardware receipt")
			}
			for _, d := range e.Dimensions {
				if d.ID == "hardware_utilization" &&
					(d.Current == nil || *d.Current != 17 || d.Source != "sentinel" ||
						d.EvidenceKind != "sentinel" || d.Engine != "sentinel") {
					t.Fatalf("invalid receipt partially mutated hardware dimension: %+v", d)
				}
			}
		})
	}
}

func TestHardwareAllowsBenignDeviceClassContainingLocalNoGPUText(t *testing.T) {
	e := hardwareEvidence(t)
	e.Hardware.Runs[0].RequestedDeviceClass = "simulator-local-no-gpu-compatible"
	got, err := decodeCycleEvidence(t, e)
	if err != nil {
		t.Fatal(err)
	}
	if got.Hardware.Runs[0].RequestedDeviceClass != "simulator-local-no-gpu-compatible" {
		t.Fatalf("requested_device_class=%q", got.Hardware.Runs[0].RequestedDeviceClass)
	}
}

func TestHardwareRejectsTypedTerminalEvidencePrecisely(t *testing.T) {
	tests := []struct {
		evidenceType string
		want         string
	}{
		{
			evidenceType: "local-no-gpu",
			want:         `hardware run 0 terminal_evidence type "local-no-gpu": local-no-GPU is a terminal blocker, not a hardware utilization measurement`,
		},
		{
			evidenceType: "gpu-driver-missing",
			want:         `hardware run 0 terminal_evidence type "gpu-driver-missing" is unsupported; measurement receipts require measured runs, not terminal blockers`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.evidenceType, func(t *testing.T) {
			e := hardwareEvidence(t)
			e.Hardware.Runs[0].TerminalEvidence = &HardwareTerminalEvidence{Type: tc.evidenceType}
			if err := applyHardware(&e); err == nil || err.Error() != tc.want {
				t.Fatalf("error=%v, want %q", err, tc.want)
			}
		})
	}
}

func TestHardwareStrictJSONAndRequiredMeasuredUtilization(t *testing.T) {
	e := hardwareEvidence(t)
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var base map[string]any
	if err := json.Unmarshal(b, &base); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		edit func(map[string]any)
	}{
		{"unknown receipt field", func(doc map[string]any) {
			doc["hardware"].(map[string]any)["private_node"] = "secret-host"
		}},
		{"unknown run host", func(doc map[string]any) {
			doc["hardware"].(map[string]any)["runs"].([]any)[0].(map[string]any)["host"] = "secret-host"
		}},
		{"unknown run path", func(doc map[string]any) {
			doc["hardware"].(map[string]any)["runs"].([]any)[0].(map[string]any)["private_path"] = "/private/lab"
		}},
		{"typed terminal evidence", func(doc map[string]any) {
			doc["hardware"].(map[string]any)["runs"].([]any)[0].(map[string]any)["terminal_evidence"] = map[string]any{"type": "local-no-gpu"}
		}},
		{"unsupported terminal evidence", func(doc map[string]any) {
			doc["hardware"].(map[string]any)["runs"].([]any)[0].(map[string]any)["terminal_evidence"] = map[string]any{"type": "gpu-driver-missing"}
		}},
		{"unknown terminal evidence field", func(doc map[string]any) {
			doc["hardware"].(map[string]any)["runs"].([]any)[0].(map[string]any)["terminal_evidence"] = map[string]any{
				"type": "local-no-gpu", "host": "secret-host",
			}
		}},
		{"missing terminal evidence type", func(doc map[string]any) {
			doc["hardware"].(map[string]any)["runs"].([]any)[0].(map[string]any)["terminal_evidence"] = map[string]any{}
		}},
		{"missing measured utilization", func(doc map[string]any) {
			delete(doc["hardware"].(map[string]any)["runs"].([]any)[0].(map[string]any), "active_utilization")
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			copyJSON, _ := json.Marshal(base)
			var doc map[string]any
			_ = json.Unmarshal(copyJSON, &doc)
			tc.edit(doc)
			invalid, _ := json.Marshal(doc)
			if _, err := Decode(bytes.NewReader(invalid)); err == nil {
				t.Fatal("accepted non-strict or incomplete hardware receipt")
			}
		})
	}
}

func TestEdgeDecodeRejectsEmptyMalformedAndTrailingInputs(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{name: "empty", input: "", want: "decode evidence:"},
		{name: "whitespace", input: " \n\t", want: "decode evidence:"},
		{name: "truncated", input: `{"schema":"fak-performance-rsi-evidence/1"`, want: "decode evidence:"},
		{name: "wrong top-level type", input: `[]`, want: "decode evidence:"},
		{name: "trailing value", input: `{} {}`, want: "decode evidence: trailing JSON value"},
		{name: "unknown hostile field", input: `{"private_token":"do-not-accept"}`, want: "decode evidence:"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Decode(strings.NewReader(tc.input))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Decode() error = %v, want containing %q", err, tc.want)
			}
		})
	}
}

func TestAdversarialDecodeRejectsOversizedDimensionSet(t *testing.T) {
	e := fixture(t)
	const oversizedCount = 10_000
	e.Dimensions = make([]Dimension, oversizedCount)
	for i := range e.Dimensions {
		e.Dimensions[i] = Dimension{
			ID:         "cycle_time",
			Direction:  Lower,
			Unit:       "hours",
			Source:     "adversarial generated input",
			NextAction: "reject before scoring",
		}
	}
	encoded, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(bytes.NewReader(encoded)); err == nil || !strings.Contains(err.Error(), "want exactly 16") {
		t.Fatalf("Decode() error = %v, want oversized dimension rejection", err)
	}
}

func TestEdgeScoreLoopTurnPreservesExistingLoadErrorPaths(t *testing.T) {
	malformed := filepath.Join(t.TempDir(), "malformed.json")
	if err := os.WriteFile(malformed, []byte(`{"schema":`), 0o600); err != nil {
		t.Fatal(err)
	}
	strictInvalid := filepath.Join(t.TempDir(), "strict-invalid.json")
	if err := os.WriteFile(strictInvalid, []byte(`{"schema":"not-performance-rsi-v1"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		input   string
		outcome InvocationOutcome
	}{
		{input: malformed, outcome: OutcomeRefusal},
		{input: strictInvalid, outcome: OutcomeRefusal},
		{input: t.TempDir(), outcome: OutcomeError},
	} {
		input := tc.input
		receipt := ScoreLoopTurn(input)
		if receipt.Status != LoopTurnUnavailable || receipt.Reason != "SCORE_INPUT_UNAVAILABLE" {
			t.Fatalf("input %q receipt header = %+v", input, receipt)
		}
		if strings.TrimSpace(receipt.UnavailableDiagnostic) == "" {
			t.Fatalf("input %q omitted load diagnostic: %+v", input, receipt)
		}
		if receipt.LoopHealth != nil || receipt.PerformanceRSIDebt != nil || receipt.DominantBottleneck != "" {
			t.Fatalf("input %q invented score data: %+v", input, receipt)
		}
		assertSingleInvocationOutcome(t, receipt.InvocationOutcomes, tc.outcome)
		assertFormattedInvocationOutcome(t, receipt, tc.outcome)
	}
}

func assertRefusalMessage(t *testing.T, err error, wants ...string) {
	t.Helper()
	if err == nil {
		t.Fatal("invalid input was accepted")
	}
	for _, want := range wants {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error %q does not name required fix %q", err, want)
		}
	}
}

func decodeForRefusal(e Evidence) error {
	b, err := json.Marshal(e)
	if err != nil {
		return err
	}
	_, err = Decode(bytes.NewReader(b))
	return err
}

func refusalDimension(e *Evidence, id string) *Dimension {
	for i := range e.Dimensions {
		if e.Dimensions[i].ID == id {
			return &e.Dimensions[i]
		}
	}
	panic("fixture missing dimension " + id)
}

func TestRefusalContractsNameRequiredEvidenceFix(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Evidence)
		want []string
	}{
		{"schema", func(e *Evidence) { e.Schema = "old" }, []string{"schema", EvidenceSchema, "fix:"}},
		{"snapshot", func(e *Evidence) { e.Snapshot = " " }, []string{"snapshot", "required", "fix:"}},
		{"target multiplier", func(e *Evidence) { e.TargetMultiplier = 99 }, []string{"target_multiplier", "100x", "fix:"}},
		{"dimension count", func(e *Evidence) { e.Dimensions = e.Dimensions[:15] }, []string{"dimensions", "exactly 16", "fix:"}},
		{"duplicate dimension", func(e *Evidence) { e.Dimensions[1].ID = e.Dimensions[0].ID }, []string{"dimension", "more than once", "fix:"}},
		{"dimension metadata", func(e *Evidence) { e.Dimensions[0].Source = "" }, []string{"dimension", "source", "unit", "next_action", "fix:"}},
		{"dimension direction", func(e *Evidence) { e.Dimensions[0].Direction = "sideways" }, []string{"dimension", "direction", "fix:"}},
		{"dimension current", func(e *Evidence) { v := -1.0; e.Dimensions[0].Current = &v }, []string{"dimension", "current", "fix:"}},
		{"dimension target", func(e *Evidence) { v := 0.0; e.Dimensions[0].Target = &v }, []string{"dimension", "target", "fix:"}},
		{"llama fallback", func(e *Evidence) { e.Dimensions[0].Engine = "llama.cpp" }, []string{"dimension", "llama.cpp", "native", "fix:"}},
		{"native benchmark engine", func(e *Evidence) { e.Dimensions[0].EvidenceKind = "native_benchmark"; e.Dimensions[0].Engine = "other" }, []string{"dimension", "fak-native engine", "fix:"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := fixture(t)
			tc.edit(&e)
			assertRefusalMessage(t, decodeForRefusal(e), tc.want...)
		})
	}
}

func TestRefusalContractsNameRequiredSectionFix(t *testing.T) {
	tests := []struct {
		name string
		base func(*testing.T) Evidence
		edit func(*Evidence)
		want []string
	}{
		{"cycle schema", cycleEvidence, func(e *Evidence) { e.Cycle.Schema = "old" }, []string{"cycle schema", CycleSchema, "fix:"}},
		{"cycle engine", cycleEvidence, func(e *Evidence) { e.Cycle.Engine = "llama.cpp" }, []string{"cycle engine", "fak-native", "without llama.cpp", "fix:"}},
		{"cycle active seconds", cycleEvidence, func(e *Evidence) { e.Cycle.OperatorActiveSeconds = -1 }, []string{"operator_active_seconds", "nonnegative", "finite", "fix:"}},
		{"cycle required timestamp", cycleEvidence, func(e *Evidence) { e.Cycle.QueueAt = "" }, []string{"cycle queue_at", "required", "fix:"}},
		{"cycle ordering", cycleEvidence, func(e *Evidence) { e.Cycle.QueueAt = e.Cycle.IdeaAt }, []string{"strictly ordered", "queue_at", "idea_at", "fix:"}},
		{"cycle unit", cycleEvidence, func(e *Evidence) { refusalDimension(e, "cycle_time").Unit = "fortnights" }, []string{"cycle derivation", "cycle_time", "unsupported duration unit", "fix:"}},
		{"improvement schema", improvementEvidence, func(e *Evidence) { e.Improvement.Schema = "old" }, []string{"improvement schema", ImprovementSchema, "fix:"}},
		{"improvement engine", improvementEvidence, func(e *Evidence) { e.Improvement.Engine = "llama.cpp" }, []string{"improvement engine", "fak-native", "without llama.cpp", "fix:"}},
		{"improvement hypothesis", improvementEvidence, func(e *Evidence) { e.Improvement.Hypothesis = "" }, []string{"hypothesis", "required", "fix:"}},
		{"improvement module revision", improvementEvidence, func(e *Evidence) { e.Improvement.ChangedModule = "module" }, []string{"changed_module", "module@rev", "fix:"}},
		{"improvement measures", improvementEvidence, func(e *Evidence) { e.Improvement.Baseline.Value = 0 }, []string{"baseline", "candidate", "positive finite milliseconds", "fix:"}},
		{"improvement quality", improvementEvidence, func(e *Evidence) { e.Improvement.Quality.Parity = nil }, []string{"quality gate", "passing baseline/candidate parity", "fix:"}},
		{"improvement envelope", improvementEvidence, func(e *Evidence) { e.Improvement.CandidateEnvelope.Model = "other" }, []string{"operating envelopes", "complete", "matched", "fix:"}},
		{"improvement gain units", improvementEvidence, func(e *Evidence) { e.Improvement.NetTrueGain.Unit = "ratio" }, []string{"net_true_gain", "percent", "overhead", "fix:"}},
		{"improvement gain arithmetic", improvementEvidence, func(e *Evidence) { e.Improvement.NetTrueGain.Value++ }, []string{"net_true_gain", "baseline minus candidate and overhead", "fix:"}},
		{"improvement causal binding", improvementEvidence, func(e *Evidence) { e.Improvement.Causal.IsolatesChange = nil }, []string{"causal binding", "isolating", "milliseconds", "fix:"}},
		{"provenance schema", provenanceEvidence, func(e *Evidence) { e.Provenance.Schema = "old" }, []string{"provenance schema", ProvenanceSchema, "fix:"}},
		{"provenance source revision", provenanceEvidence, func(e *Evidence) { e.Provenance.Source.Revision = "main" }, []string{"source revision", "40-character lowercase hex", "fix:"}},
		{"provenance adaptation", provenanceEvidence, func(e *Evidence) { e.Provenance.AdaptationStartExplicit = false }, []string{"adaptation start", "explicit", "fix:"}},
		{"provenance experiment", provenanceEvidence, func(e *Evidence) { e.Provenance.Experiment.Linked = false }, []string{"experiment", "linked", "artifact", "fix:"}},
		{"provenance reuse", provenanceEvidence, func(e *Evidence) { e.Provenance.Reuse.Classification = "new" }, []string{"reuse classification", "adapted_known_art", "fix:"}},
		{"provenance commit", provenanceEvidence, func(e *Evidence) { e.Provenance.Production.CommitSHA = "main" }, []string{"production commit_sha", "40-character lowercase hex", "fix:"}},
		{"provenance timeline", provenanceEvidence, func(e *Evidence) { e.Provenance.DiscoveryAt = "1999-01-01T00:00:00Z" }, []string{"timeline", "source <= discovery <= adaptation <= landing", "fix:"}},
		{"learning schema", learningEvidence, func(e *Evidence) { e.Learning.Schema = "old" }, []string{"learning schema", LearningSchema, "fix:"}},
		{"learning history", learningEvidence, func(e *Evidence) { e.Learning.Rows = e.Learning.Rows[:1] }, []string{"at least two rows", "history", "fix:"}},
		{"learning recurrence key", learningEvidence, func(e *Evidence) { e.Learning.Rows[0].RecurrenceKey = "" }, []string{"row 0", "recurrence_key", "required", "fix:"}},
		{"learning confidence", learningEvidence, func(e *Evidence) { e.Learning.Rows[0].ConfidencePercent = -1 }, []string{"row 0", "confidence", "finite in [0,100]", "fix:"}},
		{"learning engine artifact", learningEvidence, func(e *Evidence) { e.Learning.Rows[0].Artifact = "" }, []string{"row 0", "engine fak-native", "artifact", "fix:"}},
		{"learning reuse reference", learningEvidence, func(e *Evidence) { e.Learning.Rows[1].PriorLearningID = "missing" }, []string{"row 1", "prior_learning_id", "earlier learning", "same recurrence_key", "fix:"}},
		{"hardware schema", hardwareEvidence, func(e *Evidence) { e.Hardware.Schema = "old" }, []string{"hardware schema", HardwareSchema, "fix:"}},
		{"hardware runs", hardwareEvidence, func(e *Evidence) { e.Hardware.Runs = nil }, []string{"hardware", "at least one measured run", "fix:"}},
		{"hardware device", hardwareEvidence, func(e *Evidence) { e.Hardware.Runs[0].RequestedDeviceClass = "" }, []string{"run 0", "requested_device_class", "required", "fix:"}},
		{"hardware blocker", hardwareEvidence, func(e *Evidence) {
			e.Hardware.Runs[0].TerminalEvidence = &HardwareTerminalEvidence{Type: "local-no-gpu"}
		}, []string{"run 0", "terminal blocker", "not a hardware utilization measurement"}},
		{"hardware engine", hardwareEvidence, func(e *Evidence) { e.Hardware.Runs[0].Engine = "other" }, []string{"run 0", "engine", "exactly fak-native", "fix:"}},
		{"hardware utilization", hardwareEvidence, func(e *Evidence) { e.Hardware.Runs[0].ActiveUtilization = 101 }, []string{"run 0", "active_utilization", "directly measured", "[0,100]", "fix:"}},
		{"hardware timestamp", hardwareEvidence, func(e *Evidence) { e.Hardware.Runs[0].StartedAt = "yesterday" }, []string{"run 0", "started_at", "malformed RFC3339 timestamp", "fix:"}},
		{"hardware timeline", hardwareEvidence, func(e *Evidence) { e.Hardware.Runs[0].EndedAt = e.Hardware.Runs[0].StartedAt }, []string{"run 0", "timeline", "enqueued_at <= started_at < ended_at", "fix:"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := tc.base(t)
			tc.edit(&e)
			assertRefusalMessage(t, decodeForRefusal(e), tc.want...)
		})
	}
}

func TestErrorContractsNameRequiredCallerFix(t *testing.T) {
	tests := []struct {
		name   string
		reject func(*testing.T) error
		want   []string
	}{
		{"decode unknown field", func(t *testing.T) error { _, err := Decode(strings.NewReader(`{"unknown":true}`)); return err }, []string{"decode evidence", "unknown field", "unknown"}},
		{"decode trailing value", func(t *testing.T) error { _, err := Decode(strings.NewReader(`{} {}`)); return err }, []string{"decode evidence", "trailing JSON value", "fix:"}},
		{"compose snapshot", func(t *testing.T) error {
			_, err := ComposeV1(" ", []ComposeInput{{Source: "x", Evidence: fixture(t)}})
			return err
		}, []string{CompositionSchema, "snapshot", "required", "fix:"}},
		{"compose receipts", func(t *testing.T) error { _, err := ComposeV1("snapshot", nil); return err }, []string{CompositionSchema, "at least one receipt", "required", "fix:"}},
		{"compose source", func(t *testing.T) error {
			_, err := ComposeV1("snapshot", []ComposeInput{{Evidence: fixture(t)}})
			return err
		}, []string{CompositionSchema, "receipt 0 source", "required", "fix:"}},
		{"compose duplicate source", func(t *testing.T) error {
			e := fixture(t)
			_, err := ComposeV1("snapshot", []ComposeInput{{Source: "same", Evidence: e}, {Source: "same", Evidence: e}})
			return err
		}, []string{CompositionSchema, "receipt source", "appears more than once", "fix:"}},
		{"compose no section", func(t *testing.T) error {
			_, err := ComposeV1("snapshot", []ComposeInput{{Source: "base.json", Evidence: fixture(t)}})
			return err
		}, []string{CompositionSchema, "base.json", "no composable section", "fix:"}},
		{"compare schema", func(t *testing.T) error {
			current := Score(fixture(t))
			prior := current
			prior.Schema = "old"
			return Compare(&current, prior)
		}, []string{"prior schema", ReportSchema, "fix:"}},
		{"compare missing dimension", func(t *testing.T) error {
			current := Score(fixture(t))
			prior := current
			prior.Dimensions = prior.Dimensions[1:]
			return Compare(&current, prior)
		}, []string{"prior snapshot missing dimension", "cycle_time", "fix:"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) { assertRefusalMessage(t, tc.reject(t), tc.want...) })
	}
}

func TestRequiredLoopTurnRefusalsNameRecovery(t *testing.T) {
	fileNotFound := "no such file"
	if runtime.GOOS == "windows" {
		fileNotFound = "cannot find the file"
	}
	tests := []struct {
		name, input string
		want        []string
	}{
		{"unset input", "", []string{LoopTurnInputEnv, "not set", "fix:"}},
		{"missing file", filepath.Join(t.TempDir(), "missing.json"), []string{"missing.json", fileNotFound}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := ScoreLoopTurn(tc.input)
			if r.Status != LoopTurnUnavailable || r.Reason != "SCORE_INPUT_UNAVAILABLE" {
				t.Fatalf("refusal header = %+v", r)
			}
			for _, want := range tc.want {
				if !strings.Contains(r.UnavailableDiagnostic, want) {
					t.Fatalf("diagnostic %q does not name recovery %q", r.UnavailableDiagnostic, want)
				}
			}
		})
	}
}

func TestValidationErrorsIncludeActionableFixDirective(t *testing.T) {
	e := fixture(t)
	e.Dimensions = append([]Dimension(nil), e.Dimensions...)
	v := 0.0
	e.Dimensions[0].Target = &v
	err := validate(&e)
	if err == nil || !strings.Contains(err.Error(), "; fix: set target to a finite, positive threshold") {
		t.Fatalf("expected target fix error, got %v", err)
	}

	current := Score(fixture(t))
	prior := current
	prior.Dimensions = prior.Dimensions[1:]
	err = Compare(&current, prior)
	if err == nil || !strings.Contains(err.Error(), "; fix: ensure the prior scorecard contains all canonical dimensions") {
		t.Fatalf("expected compare fix error, got %v", err)
	}

	_, err = DecodeReport(strings.NewReader(`{malformed`))
	if err == nil || !strings.Contains(err.Error(), "; fix: provide valid JSON conforming to") {
		t.Fatalf("expected decode report fix error, got %v", err)
	}
}
