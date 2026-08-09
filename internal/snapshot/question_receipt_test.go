package snapshot

import (
	"encoding/json"
	"testing"
)

func TestQuestionReceiptStateMachineReplay(t *testing.T) {
	shown := QuestionState{Question: "Approve deployment?", RelevantEvidence: []byte("artifact=abc"), GoverningRevision: "policy-r7", AuthorityTenure: "oncall-42"}
	receipt, err := NewQuestionReceipt(shown)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(receipt)
	parsed, check := ParseQuestionReceipt(raw)
	if !check.Accepted {
		t.Fatalf("unchanged receipt parse refused: %+v", check)
	}
	if got := VerifyQuestionReceipt(parsed, shown); !got.Accepted {
		t.Fatalf("unchanged receipt refused: %+v", got)
	}

	cases := []struct {
		name   string
		mutate func(QuestionState) QuestionState
		cause  string
	}{
		{"question", func(s QuestionState) QuestionState { s.Question = "Approve a different deployment?"; return s }, "question_changed"},
		{"evidence", func(s QuestionState) QuestionState { s.RelevantEvidence = []byte("artifact=def"); return s }, "evidence_changed"},
		{"policy", func(s QuestionState) QuestionState { s.GoverningRevision = "policy-r8"; return s }, "governing_input_changed"},
		{"authority", func(s QuestionState) QuestionState { s.AuthorityTenure = "oncall-43"; return s }, "authority_changed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := VerifyQuestionReceipt(parsed, tc.mutate(shown))
			if got.Refusal != ReceiptStale || got.Cause != tc.cause {
				t.Fatalf("got %+v", got)
			}
		})
	}
}

func TestQuestionReceiptMissingAndMalformedFailClosed(t *testing.T) {
	if _, got := ParseQuestionReceipt(nil); got.Refusal != ReceiptMissing || got.Accepted {
		t.Fatalf("missing: %+v", got)
	}
	for _, raw := range [][]byte{[]byte("{"), []byte(`{"version":1}`), []byte(`{"version":1,"question_id":"bad","evidence_digest":"bad","policy_revision":"p","authority_tenure":"a","extra":true}`)} {
		if _, got := ParseQuestionReceipt(raw); got.Refusal != ReceiptMalformed || got.Accepted {
			t.Fatalf("malformed %q: %+v", raw, got)
		}
	}
}

func TestQuestionReceiptNormalizationAndUnrelatedState(t *testing.T) {
	a, err := NewQuestionReceipt(QuestionState{Question: " Approve   deployment? ", RelevantEvidence: []byte("e"), GoverningRevision: "p", AuthorityTenure: "a"})
	if err != nil {
		t.Fatal(err)
	}
	b, err := NewQuestionReceipt(QuestionState{Question: "Approve deployment?", RelevantEvidence: []byte("e"), GoverningRevision: "p", AuthorityTenure: "a"})
	if err != nil {
		t.Fatal(err)
	}
	if a != b {
		t.Fatalf("normalized identity unstable: %#v %#v", a, b)
	}
	// Clock, queue depth, and answer text are intentionally absent from QuestionState.
	if got := VerifyQuestionReceipt(a, QuestionState{Question: "Approve deployment?", RelevantEvidence: []byte("e"), GoverningRevision: "p", AuthorityTenure: "a"}); !got.Accepted {
		t.Fatalf("unrelated state invalidated receipt: %+v", got)
	}
}
