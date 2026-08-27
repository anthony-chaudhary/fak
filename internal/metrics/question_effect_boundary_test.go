package metrics

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestQuestionEffectBoundaryReplay(t *testing.T) {
	shown := QuestionState{
		Question:          "Approve deployment?",
		RelevantEvidence:  []byte("artifact=abc"),
		GoverningRevision: "policy-r7",
		AuthorityTenure:   "oncall-42",
	}
	receipt, err := NewQuestionReceipt(shown)
	if err != nil {
		t.Fatal(err)
	}
	valid, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}

	changed := func(update func(*QuestionState)) QuestionState {
		state := shown
		state.RelevantEvidence = append([]byte(nil), shown.RelevantEvidence...)
		update(&state)
		return state
	}
	cases := []struct {
		name    string
		receipt []byte
		current QuestionState
		want    ReceiptCheck
	}{
		{"missing", nil, shown, ReceiptCheck{Refusal: ReceiptMissing, Cause: string(QuestionReceiptOmitted)}},
		{"malformed", []byte(`{"version":1}`), shown, ReceiptCheck{Refusal: ReceiptMalformed, Cause: string(QuestionReceiptShapeInvalid)}},
		{"question changed", valid, changed(func(s *QuestionState) { s.Question = "Approve another deployment?" }), ReceiptCheck{Refusal: ReceiptStale, Cause: string(QuestionReceiptQuestionChanged)}},
		{"evidence changed", valid, changed(func(s *QuestionState) { s.RelevantEvidence = []byte("artifact=def") }), ReceiptCheck{Refusal: ReceiptStale, Cause: string(QuestionReceiptEvidenceChanged)}},
		{"governing input changed", valid, changed(func(s *QuestionState) { s.GoverningRevision = "policy-r8" }), ReceiptCheck{Refusal: ReceiptStale, Cause: string(QuestionReceiptGoverningInputChanged)}},
		{"authority changed", valid, changed(func(s *QuestionState) { s.AuthorityTenure = "oncall-43" }), ReceiptCheck{Refusal: ReceiptStale, Cause: string(QuestionReceiptAuthorityChanged)}},
		{"unchanged", valid, shown, ReceiptCheck{Accepted: true}},
		// Queue depth, clock, and other external state are intentionally not
		// inputs to QuestionState, so their changes cannot invalidate a receipt.
		{"unrelated external state changed", valid, shown, ReceiptCheck{Accepted: true}},
	}

	for _, consumer := range []string{"interactive", "headless"} {
		t.Run(consumer, func(t *testing.T) {
			var boundary QuestionEffectBoundary
			var effects []string
			wantCounts := make(map[QuestionReceiptRefusal]uint64)
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					before := len(effects)
					got := boundary.Apply(tc.receipt, tc.current, func() {
						effects = append(effects, consumer+":"+tc.name)
					})
					if got != tc.want {
						t.Fatalf("check = %+v, want %+v", got, tc.want)
					}
					wantEffects := before
					if tc.want.Accepted {
						wantEffects++
					} else {
						wantCounts[QuestionReceiptRefusal{Refusal: tc.want.Refusal, Cause: QuestionReceiptCause(tc.want.Cause)}]++
					}
					if len(effects) != wantEffects {
						t.Fatalf("captured %d effects, want %d", len(effects), wantEffects)
					}
				})
			}
			if !reflect.DeepEqual(boundary.RefusalCounts(), wantCounts) {
				t.Fatalf("counts = %#v, want %#v", boundary.RefusalCounts(), wantCounts)
			}
			if len(effects) != 2 {
				t.Fatalf("captured effects = %q, want two accepted effects", effects)
			}
		})
	}
}

func TestQuestionEffectBoundaryInvalidCurrentStateRefusesWithoutEffect(t *testing.T) {
	receipt, err := NewQuestionReceipt(QuestionState{
		Question: "question", GoverningRevision: "policy", AuthorityTenure: "authority",
	})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}

	var boundary QuestionEffectBoundary
	effects := 0
	got := boundary.Apply(raw, QuestionState{}, func() { effects++ })
	want := ReceiptCheck{Refusal: ReceiptMalformed, Cause: string(QuestionReceiptOrStateInvalid)}
	if got != want || effects != 0 {
		t.Fatalf("check = %+v, effects = %d; want %+v and zero", got, effects, want)
	}
}
