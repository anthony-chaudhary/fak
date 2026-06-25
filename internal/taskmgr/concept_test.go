package taskmgr

import (
	"reflect"
	"testing"
)

func TestNormalizeConcept(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"observe", "observe"},
		{"Verify", "verify"},
		{"  TOOL  ", "tool"},
		{"Model", "model"},
		{"", "other"},
		{"   ", "other"},
		{"custom-bench", "custom-bench"},
		{"  custom-bench  ", "custom-bench"},
		{"MyExperiment", "MyExperiment"},
	}
	for _, tc := range cases {
		if got := NormalizeConcept(tc.in); got != tc.want {
			t.Errorf("NormalizeConcept(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestDefaultConceptsStableAndCopy(t *testing.T) {
	want := []string{"observe", "adjudicate", "tool", "model", "cache", "io", "verify", "wait", "other"}
	got := DefaultConcepts()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("DefaultConcepts() = %v, want %v", got, want)
	}
	// Mutating the returned slice must not affect package state.
	got[0] = "mutated"
	if again := DefaultConcepts(); again[0] != "observe" {
		t.Fatalf("DefaultConcepts() aliases package state: %v", again)
	}
}

func TestIsDefaultConcept(t *testing.T) {
	for _, c := range DefaultConcepts() {
		if !IsDefaultConcept(c) {
			t.Errorf("IsDefaultConcept(%q) = false, want true", c)
		}
	}
	for _, c := range []string{"", "custom", "Observe", " verify "} {
		if IsDefaultConcept(c) {
			t.Errorf("IsDefaultConcept(%q) = true, want false (exact match only)", c)
		}
	}
}

func TestStartConceptStepNormalizes(t *testing.T) {
	m := NewManager()
	task, err := m.StartTask(TaskSpec{TaskID: "task_c"})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	if _, err := task.StartConceptStep(StepSpec{StepID: "s1", Concept: "  Verify "}); err != nil {
		t.Fatalf("start concept step: %v", err)
	}
	if _, err := task.StartConceptStep(StepSpec{StepID: "s2", Concept: ""}); err != nil {
		t.Fatalf("start empty-concept step: %v", err)
	}
	if _, err := task.StartConceptStep(StepSpec{StepID: "s3", Concept: "lab-only"}); err != nil {
		t.Fatalf("start custom-concept step: %v", err)
	}

	snap := m.Snapshot()
	steps := snap.Tasks[0].Steps
	gotByID := map[string]string{}
	for _, s := range steps {
		gotByID[s.StepID] = s.Concept
	}
	wantByID := map[string]string{"s1": "verify", "s2": "other", "s3": "lab-only"}
	if !reflect.DeepEqual(gotByID, wantByID) {
		t.Fatalf("concepts = %v, want %v", gotByID, wantByID)
	}
}

func TestNamedConceptHelpers(t *testing.T) {
	m := NewManager()
	task, err := m.StartTask(TaskSpec{TaskID: "task_named"})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	starts := []struct {
		start func() (*Step, error)
		want  string
	}{
		{func() (*Step, error) { return task.StartObserveStep("o", "observe") }, ConceptObserve},
		{func() (*Step, error) { return task.StartModelStep("m", "model") }, ConceptModel},
		{func() (*Step, error) { return task.StartToolStep("t", "tool") }, ConceptTool},
		{func() (*Step, error) { return task.StartVerifyStep("v", "verify") }, ConceptVerify},
	}
	for i, s := range starts {
		if _, err := s.start(); err != nil {
			t.Fatalf("named helper %d: %v", i, err)
		}
	}
	got := map[string]string{}
	for _, st := range m.Snapshot().Tasks[0].Steps {
		got[st.StepID] = st.Concept
	}
	want := map[string]string{"o": ConceptObserve, "m": ConceptModel, "t": ConceptTool, "v": ConceptVerify}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("named-helper concepts = %v, want %v", got, want)
	}
}

// TestConceptAggregationOrderStable proves the snapshot aggregates concept runtime
// in a stable, name-sorted order regardless of the order steps were started, and
// that repeated concepts fold into a single bucket.
func TestConceptAggregationOrderStable(t *testing.T) {
	m := NewManager()
	task, err := m.StartTask(TaskSpec{TaskID: "task_agg"})
	if err != nil {
		t.Fatalf("start task: %v", err)
	}
	steps := []struct{ id, concept string }{
		{"s_model", ConceptModel},
		{"s_cache1", ConceptCache},
		{"s_observe", ConceptObserve},
		{"s_cache2", ConceptCache},
	}
	for _, s := range steps {
		if _, err := task.StartConceptStep(StepSpec{StepID: s.id, Concept: s.concept}); err != nil {
			t.Fatalf("start %s: %v", s.id, err)
		}
	}
	snap := m.Snapshot()
	var order []string
	cacheSteps := 0
	for _, cu := range snap.Concepts {
		order = append(order, cu.Concept)
		if cu.Concept == ConceptCache {
			cacheSteps = cu.Steps
		}
	}
	want := []string{"cache", "model", "observe"}
	if !reflect.DeepEqual(order, want) {
		t.Fatalf("concept order = %v, want sorted %v", order, want)
	}
	if cacheSteps != 2 {
		t.Fatalf("cache steps folded = %d, want 2", cacheSteps)
	}
}
