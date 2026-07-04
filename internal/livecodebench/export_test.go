package livecodebench

import (
	"bytes"
	"encoding/json"
	"testing"
)

// TestCustomEvaluatorItemsFixtureRoundTrip pins the ordering + question_id
// mapping against the committed fixture: the emitted custom_evaluator input
// must line up one-for-one, in order, with the fixture's items, and survive a
// JSON round-trip unchanged.
func TestCustomEvaluatorItemsFixtureRoundTrip(t *testing.T) {
	f, err := LoadFile("testdata/fixture.json")
	if err != nil {
		t.Fatalf("load fixture: %v", err)
	}

	items, err := CustomEvaluatorItems(f)
	if err != nil {
		t.Fatalf("CustomEvaluatorItems: %v", err)
	}
	if len(items) != len(f.Items) {
		t.Fatalf("item count = %d, want %d", len(items), len(f.Items))
	}
	for i, it := range items {
		src := f.Items[i]
		if it.QuestionID != src.QuestionID {
			t.Errorf("item %d question_id = %q, want %q (order/mapping drift)", i, it.QuestionID, src.QuestionID)
		}
		if len(it.CodeList) != len(src.CodeList) {
			t.Fatalf("item %d code_list len = %d, want %d", i, len(it.CodeList), len(src.CodeList))
		}
		for j := range it.CodeList {
			if it.CodeList[j] != src.CodeList[j] {
				t.Errorf("item %d code_list[%d] = %q, want %q", i, j, it.CodeList[j], src.CodeList[j])
			}
		}
	}

	var buf bytes.Buffer
	if err := WriteCustomEvaluatorInput(&buf, f); err != nil {
		t.Fatalf("WriteCustomEvaluatorInput: %v", err)
	}

	// code_list must always be present (no omitempty) so the checker never sees
	// a question with a missing key.
	if !bytes.Contains(buf.Bytes(), []byte(`"code_list"`)) {
		t.Errorf("emitted JSON is missing the code_list key: %s", buf.String())
	}

	var back []CustomEvalItem
	if err := json.Unmarshal(buf.Bytes(), &back); err != nil {
		t.Fatalf("re-decode emitted JSON: %v", err)
	}
	if len(back) != len(items) {
		t.Fatalf("round-trip count = %d, want %d", len(back), len(items))
	}
	for i := range back {
		if back[i].QuestionID != items[i].QuestionID {
			t.Errorf("round-trip item %d question_id = %q, want %q", i, back[i].QuestionID, items[i].QuestionID)
		}
	}
}

// TestCustomEvaluatorItemsRejectsMissingCodeList proves an ungradeable item
// (no generations) is refused rather than emitted as an empty entry.
func TestCustomEvaluatorItemsRejectsMissingCodeList(t *testing.T) {
	f := Fixture{
		Schema:         FixtureSchema,
		ReleaseVersion: "fixture_release",
		StartDate:      "2026-01-01",
		EndDate:        "2026-01-02",
		Items: []FixtureItem{
			{QuestionID: "q1", Scenario: "codegeneration", Prompt: "p", CodeList: []string{"code"}},
			{QuestionID: "q2", Scenario: "codegeneration", Prompt: "p"}, // no code_list
		},
	}
	if _, err := CustomEvaluatorItems(f); err == nil {
		t.Fatal("expected error for item with no code_list, got nil")
	}
}

// TestCustomEvaluatorItemsRejectsEmpty guards the empty-fixture edge.
func TestCustomEvaluatorItemsRejectsEmpty(t *testing.T) {
	if _, err := CustomEvaluatorItems(Fixture{}); err == nil {
		t.Fatal("expected error for empty fixture, got nil")
	}
}
