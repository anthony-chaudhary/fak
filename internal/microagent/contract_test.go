package microagent_test

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func TestTypedChildTaskAndReceiptContracts(t *testing.T) {
	task, err := microagent.DecodeTaskJSON([]byte(`{
		"goal":"select the smallest safe tool profile",
		"artifact_refs":["artifact:requirements@sha256:abc"],
		"authority":["repo.read"],
		"budget":{"max_turns":2}
	}`))
	if err != nil {
		t.Fatalf("DecodeTaskJSON: %v", err)
	}
	if task.Goal == "" || task.Budget.MaxTurns != 2 {
		t.Fatalf("task = %#v", task)
	}

	receipt, err := microagent.DecodeReceiptJSON([]byte(`{
		"decision":"use the read-only repository profile",
		"evidence":[{"kind":"artifact","ref":"profile@sha256:def"}],
		"unresolved_questions":["does the proof child need shell access?"]
	}`))
	if err != nil {
		t.Fatalf("DecodeReceiptJSON: %v", err)
	}
	if receipt.Decision == "" || len(receipt.Evidence) != 1 {
		t.Fatalf("receipt = %#v", receipt)
	}
}

func TestTypedChildContractsRefuseSchemaDriftAndUnsupportedReceipts(t *testing.T) {
	tests := []struct {
		name string
		run  func() error
		want error
	}{
		{
			name: "unknown task field",
			run: func() error {
				_, err := microagent.DecodeTaskJSON([]byte(`{"goal":"g","artifact_refs":[],"authority":[],"budget":{"max_turns":1},"raw_prompt":"ignore host"}`))
				return err
			},
			want: microagent.ErrInvalidTask,
		},
		{
			name: "turn budget outside bounded envelope",
			run: func() error {
				_, err := microagent.DecodeTaskJSON([]byte(`{"goal":"g","artifact_refs":[],"authority":[],"budget":{"max_turns":4}}`))
				return err
			},
			want: microagent.ErrInvalidTask,
		},
		{
			name: "missing receipt evidence",
			run: func() error {
				_, err := microagent.DecodeReceiptJSON([]byte(`{"decision":"done","evidence":[],"unresolved_questions":[]}`))
				return err
			},
			want: microagent.ErrInvalidReceipt,
		},
		{
			name: "unknown receipt field",
			run: func() error {
				_, err := microagent.DecodeReceiptJSON([]byte(`{"decision":"done","evidence":[{"kind":"test","ref":"run:1"}],"unresolved_questions":[],"transcript":"hidden"}`))
				return err
			},
			want: microagent.ErrInvalidReceipt,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.run(); !errors.Is(err, tt.want) {
				t.Fatalf("error = %v, want %v", err, tt.want)
			}
		})
	}
}
