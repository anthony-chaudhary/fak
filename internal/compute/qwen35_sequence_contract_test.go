package compute

import (
	"errors"
	"reflect"
	"testing"
)

func TestQwen35SequenceEmbeddingContractSeparatesTableAndRowPanelModes(t *testing.T) {
	table := Qwen35SequencePrefillRequest{
		TokenIDs:       []int{1, 2, 1},
		Hidden:         8,
		TokenEmbedding: Tensor{Shape: []int{11, 8}},
	}
	vocab, shape, err := qwen35SequenceEmbeddingContract(table)
	if err != nil || vocab != 11 || !reflect.DeepEqual(shape, []int{11, 8}) {
		t.Fatalf("table contract vocab=%d shape=%v err=%v", vocab, shape, err)
	}

	rows := table
	rows.TokenEmbeddingRows = true
	rows.TokenEmbeddingVocab = 11
	rows.TokenEmbedding.Shape = []int{3, 8}
	vocab, shape, err = qwen35SequenceEmbeddingContract(rows)
	if err != nil || vocab != 11 || !reflect.DeepEqual(shape, []int{3, 8}) {
		t.Fatalf("row-panel contract vocab=%d shape=%v err=%v", vocab, shape, err)
	}
}

func TestQwen35SequenceEmbeddingContractRejectsAmbiguousModes(t *testing.T) {
	for name, req := range map[string]Qwen35SequencePrefillRequest{
		"row panel without vocabulary": {
			TokenIDs: []int{1, 2}, Hidden: 8, TokenEmbeddingRows: true,
			TokenEmbedding: Tensor{Shape: []int{2, 8}},
		},
		"table with vocabulary override": {
			TokenIDs: []int{1, 2}, Hidden: 8, TokenEmbeddingVocab: 11,
			TokenEmbedding: Tensor{Shape: []int{11, 8}},
		},
		"table without vocabulary shape": {
			TokenIDs: []int{1, 2}, Hidden: 8,
			TokenEmbedding: Tensor{Shape: []int{2}},
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, _, err := qwen35SequenceEmbeddingContract(req)
			var contractErr *Qwen35SequenceError
			if err == nil || !errors.As(err, &contractErr) || contractErr.Stage != "tensor-preflight" {
				t.Fatalf("error=%v, want typed tensor-preflight refusal", err)
			}
		})
	}
}
